import torch.nn as nn
from torch import Tensor

from lerobot.policies.mact.configuration_mact import MACTConfig


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
            nn.Linear(512, dim_model),
        )

        self.multihead_attn = nn.MultiheadAttention(dim_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, is_causal: bool = False):
        """Cross-modal attention between visual history and robot state.

        Args:
            query: (B, L, D) visual history features
            key: (B, L, D) projected robot state
            value: (B, L, D) projected robot state
            is_causal: If True, use causal mask so position i only attends to [0:i+1]
        """
        attn_output, _ = self.multihead_attn(query, key, value, is_causal=is_causal)
        return self.norm(query + attn_output)
