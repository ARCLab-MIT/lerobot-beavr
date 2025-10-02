import argparse
from typing import Tuple

import torch

from lerobot.policies.dact.mamba_policy import Mamba2


def build_minimal_mamba_for_demo() -> Mamba2:
    """Create a minimal Mamba2 instance only to reuse its depthwise conv path.

    Configuration:
    - d_conv (kernel length) = 3
    - conv channels = d_ssm + 2 * ngroups * d_state = 1 + 2 * 1 * 1 = 3
      Channel order matches the implementation: [x, B, C].
    - headdim = 1, d_ssm = 1, d_state = 1, ngroups = 1

    Note: We do not use projections or SSM in this example; we only use the
    conv1d module and its activation to match the implementation path.
    """
    model = Mamba2(
        d_model=1,         # arbitrary, unused in this demo
        d_state=1,         # yields 2*d_state for B and C channels
        d_conv=3,          # K = 3
        expand=1,          # keep d_inner = d_model for simplicity (unused)
        headdim=1,
        d_ssm=1,           # 1 x-channel
        ngroups=1,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=torch.float32,
    )
    return model


def set_depthwise_kernels(
    model: Mamba2,
    w_x_newest_to_oldest: Tuple[float, float, float],
    w_b_newest_to_oldest: Tuple[float, float, float],
    w_c_newest_to_oldest: Tuple[float, float, float],
    b_x: float = 0.0,
    b_b: float = 0.0,
    b_c: float = 0.0,
) -> None: 
    """Program the depthwise kernels into model.conv1d.

    The implementation in Mamba2's fallback conv path stores the newest input at
    index -1 of the conv_state width dimension. The weight is multiplied
    elementwise without reversing. Therefore, to apply weights in the semantic
    order [newest, prev, oldest], we must store them in the tensor indices as
    [oldest, prev, newest].
    """
    # Map newest->oldest to tensor order [oldest, prev, newest]
    def reorder(newest_prev_oldest):
        newest, prev, oldest = newest_prev_oldest
        return torch.tensor([oldest, prev, newest], dtype=model.conv1d.weight.dtype)

    w_x = reorder(w_x_newest_to_oldest)
    w_b = reorder(w_b_newest_to_oldest)
    w_c = reorder(w_c_newest_to_oldest)

    # Depthwise: weight shape = (channels, 1, K)
    weight = torch.zeros_like(model.conv1d.weight)
    # Channel order: [x, B, C]
    weight[0, 0, :] = w_x
    weight[1, 0, :] = w_b
    weight[2, 0, :] = w_c
    model.conv1d.weight.data.copy_(weight)

    if model.conv1d.bias is not None:
        model.conv1d.bias.data.copy_(
            torch.tensor([b_x, b_b, b_c], dtype=model.conv1d.bias.dtype)
        )


def run_depthwise_conv_step_t1(verbose: bool = True) -> None:
    """Run a single t=1 step through the depthwise conv path and print internals.

    Given parameters (as requested):
    - K = 3, Channels = 3 (x, B, C)
    - Raw inputs at t=1: x=2.00, B=1.00, C=0.50
    - Kernels (newest → oldest):
        * w_x = [0.5000, 0.3000, 0.2000], b_x = 0.00
        * w_B = [0.4000, 0.4000, 0.0000], b_B = 0.00
        * w_C = [0.6000, 0.0000, 0.0000], b_C = 0.00

    We reproduce the exact update and multiply used in the Mamba2 fallback path:
      conv_state <- roll left by 1 along width
      conv_state[:, :, -1] <- xBC(t)
      xbc_pre_act = sum(conv_state * weight, dim=width)
      xbc_post_act = SiLU(xbc_pre_act)
    """
    model = build_minimal_mamba_for_demo()

    # Program the provided kernels (newest → oldest ordering as per request)
    set_depthwise_kernels(
        model,
        w_x_newest_to_oldest=(0.5000, 0.3000, 0.2000),
        w_b_newest_to_oldest=(0.4000, 0.4000, 0.0000),
        w_c_newest_to_oldest=(0.6000, 0.0000, 0.0000),
        b_x=0.0,
        b_b=0.0,
        b_c=0.0,
    )

    # Allocate conv_state exactly as Mamba2 does
    conv_state, _ = model.allocate_inference_cache(batch_size=1, max_seqlen=1, dtype=torch.float32)

    # Inputs at t=1 (no history yet, so older entries are zeros)
    x_t = torch.tensor([2.0], dtype=torch.float32)
    b_t = torch.tensor([1.0], dtype=torch.float32)
    c_t = torch.tensor([0.5], dtype=torch.float32)
    xbc_t = torch.stack([x_t, b_t, c_t], dim=-1)  # shape (1, 3)

    if verbose:
        print("Config:")
        print("  K = 3, channels = [x, B, C]")
        print("  Inputs t=1: x=2.00, B=1.00, C=0.50")
        print("  Kernels (newest→oldest):")
        print("    w_x = [0.5, 0.3, 0.2], b_x = 0.0")
        print("    w_B = [0.4, 0.4, 0.0], b_B = 0.0")
        print("    w_C = [0.6, 0.0, 0.0], b_C = 0.0")
        print()

    # Show initial conv_state (all zeros)
    if verbose:
        print("conv_state BEFORE update (shape: B=1, C=3, W=3):")
        print(conv_state)

    # Mamba2 fallback conv update
    conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
    conv_state[:, :, -1] = xbc_t

    if verbose:
        print("\nconv_state AFTER update (index W=-1 is newest):")
        print(conv_state)

    # Depthwise multiply-accumulate (no reversal, matches implementation)
    weight_dw = torch.permute(model.conv1d.weight, (0, 2, 1)).squeeze(-1)  # (C, W)
    # xbc_pre_act: (B=1, C=3)
    xbc_pre_act = torch.sum(conv_state * weight_dw, dim=-1)
    if model.conv1d.bias is not None:
        xbc_pre_act = xbc_pre_act + model.conv1d.bias

    # Activation is SiLU in the code path
    xbc_post_act = model.act(xbc_pre_act)

    if verbose:
        # For clarity, print per-channel details
        labels = ["x", "B", "C"]
        print("\nWeights used internally (tensor order [oldest, prev, newest]):")
        for i, name in enumerate(labels):
            print(f"  {name}: {weight_dw[i].tolist()}")

        print("\nPre-activation conv outputs [x, B, C]:")
        print(xbc_pre_act)

        print("\nPost-activation (SiLU) conv outputs [x, B, C]:")
        print(xbc_post_act)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a minimal depthwise causal conv example through Mamba2's conv path "
            "(t=1 only), printing intermediate computations."
        )
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose prints (useful to disable logging afterwards).",
    )
    args = parser.parse_args()
    run_depthwise_conv_step_t1(verbose=not args.quiet)


if __name__ == "__main__":
    main()

