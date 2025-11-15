import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from einops import rearrange, repeat
from torch import Tensor

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from huggingface_hub import PyTorchModelHubMixin
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter
from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined

from lerobot.policies.dact.configuration_mact import MACTConfig


@dataclass
class Mamba2Config:
    # Mamba2 config fields
    dim_model: int = 512
    n_heads: int = 8
    dim_state: int = 512
    dim_conv: int = 4
    conv_init: int | None = None
    expand: int = 2
    dim_headdim: int = 128
    dim_ssm: int | None = None  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
    n_groups: int = 1
    A_init_range: tuple[int, int] = (1, 16)
    D_has_hdim: bool = False
    rmsnorm: bool = True
    norm_before_gate: bool = False
    dt_min: float = 0.001
    dt_max: float = 0.02
    dt_init_floor: float = 1e-4
    dt_limit: tuple[float, float] = (0.0, float("inf"))
    bias: bool = False
    conv_bias: bool = True
    process_group: torch.distributed.ProcessGroup | None = None
    # Fused kernel and sharding options
    mamba2_chunk_size: int = 256
    use_mem_eff_path: bool = True
    layer_idx: int | None = None  # Absorb kwarg for general module
    sequence_parallel: bool = True
    mamba2_activation: str = "silu"
    # Device and dtype for tensor creation
    device: str | None = "cuda"
    dtype: torch.dtype | None = torch.float32


class Mamba2(nn.Module, PyTorchModelHubMixin):

    def __init__(self, config: Mamba2Config):
        """Initialize Mamba2 layer with configuration.

        Args:
            config: Mamba2Config containing model hyperparameters
        """
        super().__init__()

        # Handle case where config.process_group might not be a valid ProcessGroup
        self.process_group = config.process_group if hasattr(config.process_group, 'size') else None
        self.world_size = 1 if self.process_group is None else self.process_group.size()
        self.local_rank = 0 if self.process_group is None else self.process_group.rank()
        self.dim_inner = (config.expand * config.dim_model) // self.world_size
        self.dt_limit = config.dt_limit
        self.activation = config.mamba2_activation
        self.chunk_size = config.mamba2_chunk_size
        self.use_mem_eff_path = config.use_mem_eff_path
        self.layer_idx = config.layer_idx
        self.dim_ssm = self.dim_inner if config.dim_ssm is None else config.dim_ssm // self.world_size

        # Compute derived dimensions for Mamba layers
        assert config.n_groups % self.world_size == 0
        self.ngroups = config.n_groups // self.world_size
        assert self.dim_ssm % config.dim_headdim == 0
        self.nheads = self.dim_ssm // config.dim_headdim

        # Store configuration flags
        self.D_has_hdim = config.D_has_hdim
        self.rmsnorm = config.rmsnorm
        self.norm_before_gate = config.norm_before_gate

        # Store base dimensions
        self.dim_state = config.dim_state
        self.headdim = config.dim_headdim
        self.dim_conv = config.dim_conv

        # Input projection dimension: [z, x, B, C, dt]
        dim_in_proj = 2 * self.dim_inner + 2 * self.ngroups * self.dim_state + self.nheads

        if self.process_group is None:
            self.in_proj = nn.Linear(
                in_features=config.dim_model,
                out_features=dim_in_proj,
                bias=config.bias,
            )

        else:
            self.in_proj = ColumnParallelLinear(
                in_features=config.dim_model,
                out_features=dim_in_proj * self.world_size,
                bias=config.bias,
                process_group=self.process_group,
                sequence_parallel=config.sequence_parallel,
            )

        conv_dim = self.dim_ssm + 2 * self.ngroups * self.dim_state

        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=config.conv_bias,
            kernel_size=self.dim_conv,
            groups=conv_dim,
            padding=self.dim_conv - 1,
        )

        if config.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -config.conv_init, config.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(config.n_heads) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        )

        dt = torch.clamp(dt, min=config.dt_init_floor)

        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert config.A_init_range[0] > 0 and config.A_init_range[1] >= config.A_init_range[0]

        # A "transition matrix" parameter
        A = torch.empty(config.n_heads, dtype=torch.float32, device=config.device).uniform_(*config.A_init_range)  # noqa: N806
        A_log = torch.log(A).to(dtype=config.dtype)  # noqa: N806
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip connection" parameter
        self.D = nn.Parameter(torch.ones(self.dim_ssm if config.D_has_hdim else config.n_heads, device=config.device))
        self.D._no_weight_decay = True

        if config.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.dim_ssm, eps=1e-5, norm_before_gate=config.norm_before_gate,
                                     group_size=self.dim_ssm // config.n_groups)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.dim_inner, config.dim_model, bias=config.bias)

        else:
            self.out_proj = RowParallelLinear(
                dim_out=self.dim_inner * self.world_size,
                dim_in=config.dim_model,
                bias=config.bias,
                process_group=self.process_group,
                sequence_parallel=config.sequence_parallel,
            )

    def forward(
        self,
        u: Tensor,
        seqlen: int = None,
        seq_idx: Tensor = None,
        cu_seqlens: Tensor = None,
        inference_params: Any = None
    ):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        seqlen_og = seqlen

        if seqlen is None:
            batch, seqlen, _ = u.shape

        else:
            batch_seqlen, _ = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None

        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)

            if inference_params.seqlen_offset > 0:
                # States are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u)  # (B, L, dim_in_proj) or (B * L, dim_in_proj)

        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)

        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads,) or (dim_inner, dim_state)  # noqa: N806

        dt_limit_kwargs = {} if self.dt_limit == [0.0, float("inf")] else {"dt_limit": self.dt_limit}

        if self.use_mem_eff_path and inference_params is None:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt=zxbcdt,
                conv1d_weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                conv1d_bias=self.conv1d.bias,
                dt_bias=self.dt_bias,
                A=A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )

            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")

            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
        else:
            dim_mlp = (zxbcdt.shape[-1] - 2 * self.dim_ssm - 2 * self.ngroups * self.dim_state - self.nheads) // 2
            z0, x0, z, xBC, dt = torch.split(  # noqa: N806
                zxbcdt,
                [dim_mlp, dim_mlp, self.dim_ssm, self.dim_ssm + 2 * self.ngroups * self.dim_state, self.nheads],
                dim=-1
            )

            if conv_state is not None:
                if cu_seqlens is None:
                    # If we just take xBC[:, :, -self.dim_conv :], it will error if seqlen < self.dim_conv
                    # Instead F.pad will pad with zeros if seqlen < self.dim_conv, and truncate otherwise.
                    xBC_t = rearrange(xBC, "b l d -> b d l")  # noqa: N806
                    conv_state.copy_(F.pad(xBC_t, (self.dim_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)

                else:
                    assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
                    assert batch == 1, "varlen inference only supports batch dimension 1"

                    conv_varlen_states = causal_conv1d_varlen_states(
                        xBC.squeeze(0),
                        cu_seqlens,
                        state_len=conv_state.shape[-1]
                    )

                    conv_state.copy_(conv_varlen_states)

            assert self.activation in ["silu", "swish"]

            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"

                xBC = self.act(  # noqa: N806
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.dim_conv - 1)]  # noqa: N806
                )  # (B, L, dim_ssm + 2 * ngroups * d_state)

            else:
                xBC = causal_conv1d_fn(  # noqa: N806
                    xBC.transpose(1, 2),  # noqa: N806
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)

            x, B, C = torch.split(xBC, [self.dim_ssm, self.ngroups * self.dim_state, self.ngroups * self.dim_state], dim=-1)  # noqa: N806

            y = mamba_chunk_scan_combined(
                x=rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt=dt,
                A=A,
                B=rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                C=rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
                return_varlen_states=cu_seqlens is not None and inference_params is not None,
            )

            if ssm_state is not None:
                y, last_state, *rest = y

                if cu_seqlens is None:
                    ssm_state.copy_(last_state)

                else:
                    varlen_states = rest[0]
                    ssm_state.copy_(varlen_states)

            y = rearrange(y, "b l h p -> b l (h p)")

            if self.rmsnorm:
                y = self.norm(y, z)

            if dim_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)

            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")

            out = self.out_proj(y)

        return out

    def step(self, hidden_states: Tensor, conv_state: Tensor, ssm_state: Tensor):
        """Single-step inference for autoregressive decoding.

        Args:
            hidden_states: Input token (batch, 1, hidden_dim)
            conv_state: Convolutional state from previous step
            ssm_state: SSM state from previous step

        Returns:
            tuple: (output, conv_state, ssm_state)
        """
        dtype = hidden_states.dtype

        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"

        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)

        dim_mlp = (zxbcdt.shape[-1] - 2 * self.dim_ssm - 2 * self.ngroups * self.dim_state - self.nheads) // 2

        z0, x0, z, xBC, dt = torch.split(  # noqa: N806
            zxbcdt,
            [dim_mlp, dim_mlp, self.dim_ssm, self.dim_ssm + 2 * self.ngroups * self.dim_state, self.nheads],
            dim=-1
        )

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)  # noqa: N806
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias  # noqa: N806
            xBC = self.act(xBC).to(dtype=dtype)  # noqa: N806
        else:
            xBC = causal_conv1d_update(  # noqa: N806
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.dim_ssm, self.ngroups * self.dim_state, self.ngroups * self.dim_state], dim=-1)  # noqa: N806
        A = -torch.exp(self.A_log.float())  # (nheads,)  # noqa: N806

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)  # noqa: N806
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)  # noqa: N806
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)

        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.dim_state).to(dtype=torch.float32)  # noqa: N806
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)  # noqa: N806
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)  # noqa: N806
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)  # noqa: N806
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )

            y = rearrange(y, "b h p -> b (h p)")

        if self.rmsnorm:
            y = self.norm(y, z)

        if dim_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)

        out = self.out_proj(y)

        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size: int, dtype: torch.dtype = None):
        """Allocate inference cache for efficient autoregressive generation.

        Args:
            batch_size: Number of sequences to cache
            dtype: Data type for cache tensors (defaults to model dtype)

        Returns:
            tuple: (conv_state, ssm_state) cache tensors
        """
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype

        conv_state = torch.zeros(
            batch_size, self.dim_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)

        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.dim_state, device=device, dtype=ssm_dtype
        )

        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params: Any, batch_size: int, initialize_states: bool = False):
        """Retrieve or initialize inference states from cache.

        Args:
            inference_params: Cache object containing layer states
            batch_size: Batch size for state initialization
            initialize_states: Whether to reset states to zero

        Returns:
            tuple: (conv_state, ssm_state) from cache
        """
        assert self.layer_idx is not None

        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size,
                self.dim_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)

            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.dim_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )

            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)

        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]

            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()

        return conv_state, ssm_state


class CrossCameraAttention(nn.Module):
    def __init__(self, config: MACTConfig):
        super().__init__()
        dim_model = config.dim_model
        n_heads = config.n_heads

        self.multihead_attn = nn.MultiheadAttention(dim_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        attn_output, _ = self.multihead_attn(query, key, value)
        return self.norm(query + attn_output)


class CrossModalAttention(nn.Module):
    def __init__(self, config: MACTConfig):
        super().__init__()
        dim_model = config.dim_model
        n_heads = config.n_heads
        lowdim_dim = config.robot_state_feature.shape[0]

        self.proj_lowdim = nn.Sequential(
            nn.Linear(lowdim_dim, 128),
            nn.GELU(),
            nn.Linear(128, 512),
            nn.Dropout(0.2),
            nn.Linear(512, dim_model)
        )

        self.multihead_attn = nn.MultiheadAttention(dim_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        # Note: key and value are already projected outside this module
        attn_output, _ = self.multihead_attn(query, key, value)
        return self.norm(query + attn_output)
