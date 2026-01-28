import torch
import torch.nn.functional as F  # noqa: N812
from rotary_embedding_torch import RotaryEmbedding
from torch import Tensor, nn


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with RoPE applied after projection to per-head Q and K.

    This implements proper RoPE application:
    1. RoPE dimension matches per-head dimension (head_dim = embed_dim / num_heads)
    2. RoPE is applied AFTER Q/K projection (not before)
    3. RoPE is applied to both Q and K in every attention operation
    4. Each attention block applies RoPE independently (not carried through from encoder)
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        rope: RotaryEmbedding | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        """
        Args:
            query: (L, B, E) query tensor
            key: (S, B, E) key tensor
            value: (S, B, E) value tensor
            rope: RotaryEmbedding to apply to Q and K
            key_padding_mask: (B, S) mask where True means ignore that position
        Returns:
            (L, B, E) attention output, None (for compatibility with nn.MultiheadAttention)
        """
        L, B, E = query.shape
        S = key.shape[0]

        # Project Q, K, V
        Q = self.q_proj(query)  # (L, B, E)
        K = self.k_proj(key)    # (S, B, E)
        V = self.v_proj(value)  # (S, B, E)

        # Reshape to (L/S, B, num_heads, head_dim) then transpose to (B, num_heads, L/S, head_dim)
        Q = Q.view(L, B, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)  # (B, num_heads, L, head_dim)
        K = K.view(S, B, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)  # (B, num_heads, S, head_dim)
        V = V.view(S, B, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)  # (B, num_heads, S, head_dim)

        # Apply RoPE to Q and K (after projection, per-head)
        if rope is not None:
            # RoPE expects (batch, seq_len, heads, dim) or similar
            # Our Q/K are (B, num_heads, L/S, head_dim)
            # rotary-embedding-torch's rotate_queries_or_keys expects (..., seq_len, dim)
            # We need to reshape: (B, num_heads, L, head_dim) -> (B * num_heads, L, head_dim)
            Q_rope = Q.reshape(B * self.num_heads, L, self.head_dim)
            K_rope = K.reshape(B * self.num_heads, S, self.head_dim)

            Q_rope = rope.rotate_queries_or_keys(Q_rope)
            K_rope = rope.rotate_queries_or_keys(K_rope)

            Q = Q_rope.reshape(B, self.num_heads, L, self.head_dim)
            K = K_rope.reshape(B, self.num_heads, S, self.head_dim)

        # Compute attention scores: (B, num_heads, L, S)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask is (B, S), expand to (B, 1, 1, S) for broadcasting
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention to values: (B, num_heads, L, head_dim)
        output = torch.matmul(attn_weights, V)

        # Reshape back: (B, num_heads, L, head_dim) -> (L, B, E)
        output = output.transpose(1, 2).contiguous().view(B, L, E).transpose(0, 1)

        # Final projection
        output = self.out_proj(output)

        return output, None  # Return None for attention weights (compatibility)