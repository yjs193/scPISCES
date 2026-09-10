from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import DropPath, Mlp
except ImportError:  # timm<1.0 compatibility
    from timm.models.layers import DropPath, Mlp

try:
    from flash_attn import flash_attn_qkvpacked_func
except (ImportError, OSError):
    flash_attn_qkvpacked_func = None


class MultiheadSelfAttention(nn.Module):
    """Self-attention with optional FlashAttention and an SDPA fallback."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)
        self.attention_dropout = attention_dropout

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.num_heads, channels // self.num_heads
        )
        dropout_p = self.attention_dropout if self.training else 0.0

        can_use_flash = (
            flash_attn_qkvpacked_func is not None
            and padding_mask is None
            and qkv.is_cuda
            and qkv.dtype in (torch.float16, torch.bfloat16)
        )
        if can_use_flash:
            attended = flash_attn_qkvpacked_func(qkv, dropout_p=dropout_p)
        else:
            query = qkv[:, :, 0].transpose(1, 2)
            key = qkv[:, :, 1].transpose(1, 2)
            value = qkv[:, :, 2].transpose(1, 2)
            attention_mask = None
            if padding_mask is not None:
                attention_mask = (~padding_mask)[:, None, None, :]
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
            ).transpose(1, 2)

        output = attended.reshape(batch, length, channels)
        return self.proj_drop(self.proj(output))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.1,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiheadSelfAttention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=dropout,
        )

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), padding_mask))
        return x + self.drop_path(self.mlp(self.norm2(x)))
