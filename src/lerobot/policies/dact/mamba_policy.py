import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_update
except ImportError:
    causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None


# 0) MambaConfig
#########################################
class MambaConfig:
    def __init__(self):
        self.d_model = 2048
        self.d_state = 512
        self.d_conv = 4
        self.expand = 2
        self.headdim = 128
        self.A_init_range = (1, 16)
        self.dt_min = 0.001
        self.dt_max = 0.02
        self.dt_init_floor = 1e-4
        self.dt_limit = (0.0, float("inf"))
        self.chunk_size = 256

#########################################
# 1)  Mamba2
#########################################

class Mamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=256,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=128,
        d_ssm=None,
        ngroups=1,
        A_init_range=(1, 16),  # noqa: N803
        D_has_hdim=False,  # noqa: N803
        rmsnorm=False,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.02,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        chunk_size=256,
        use_mem_eff_path=False,
        layer_idx=None,
        process_group=None,
        sequence_parallel=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm
        self.ngroups = ngroups
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)  # noqa: N806
        A_log = torch.log(A).to(dtype=dtype)  # noqa: N806
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        # No RMSNorm dependency by default

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """Simple sequential forward built from step().

        Args:
            u: (B, L, D)
        Returns:
            (B, L, D)
        """
        assert u.dim() == 3, "Expected (B, L, D) input"
        bsz, seqlen, _ = u.shape
        conv_state, ssm_state = self.allocate_inference_cache(bsz, max_seqlen=1, dtype=u.dtype)
        outs = []
        for t in range(seqlen):
            out_t, conv_state, ssm_state = self.step(u[:, t:t+1, :], conv_state, ssm_state)
            outs.append(out_t)
        return torch.cat(outs, dim=1)

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (batch, D_in)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(  # noqa: N806
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = xBC
            xbc = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                xbc = xbc + self.conv1d.bias
            xbc = self.act(xbc).to(dtype=dtype)
        else:
            xbc = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, b_mat, c_mat = torch.split(xbc, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)  # noqa: N806

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))
            d_a = torch.exp(dt * A)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            d_bx = torch.einsum("bh,bn,bhp->bhpn", dt, b_mat, x)
            ssm_state.copy_(ssm_state * rearrange(d_a, "b h -> b h 1 1") + d_bx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), c_mat)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)
        else:
            a_full = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt_full = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            d_full = repeat(self.D, "h -> h p", p=self.headdim)
            b_full = rearrange(b_mat, "b (g n) -> b g n", g=self.ngroups)
            c_full = rearrange(c_mat, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt_full, a_full, b_full, c_full, d_full, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            # If rmsnorm is enabled, you can plug in an appropriate module here
            pass
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state
    def _get_states_from_cache(self, *args, **kwargs):
        raise NotImplementedError
