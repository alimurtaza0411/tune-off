"""LLaMA-style Transformer (README: "Models"): RMSNorm pre-norm, RoPE attention, SwiGLU MLP."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mvt.core.config import ScaleConfig, TransformerConfig
from mvt.core.registry import register_model
from mvt.core.types import TaskSpec
from mvt.models.common import RMSNorm, SequenceModel


def rope_tables(length: int, head_dim: int, base: float, device: torch.device):
    """cos and sin tables of shape [L, head_dim / 2]."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    angles = torch.arange(length, device=device, dtype=torch.float32)[:, None] * inv_freq[None]
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (first half, second half) of the head dimension. x: [B, H, L, Dh]."""
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rope_base: float, causal: bool) -> None:
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.rope_base, self.causal = rope_base, causal
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)  # residual output projection

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, length, d = x.shape
        qkv = self.qkv(x).view(b, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each [B, H, L, Dh]
        cos, sin = rope_tables(length, self.head_dim, self.rope_base, x.device)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.causal:  # lm: causal mask, no padding
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:  # encoder: attend in both directions, never to pads (True = attend)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        return self.out(y.transpose(1, 2).reshape(b, length, d))


class SwiGLU(nn.Module):
    """W2 · (SiLU(W1 x) * W3 x), no biases."""

    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)  # residual output projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerLayer(nn.Module):
    def __init__(self, scale: ScaleConfig, cfg: TransformerConfig, causal: bool, dropout: float):
        super().__init__()
        d = scale.d_model
        self.norm1 = RMSNorm(d, cfg.norm_eps)
        self.attn = Attention(d, scale.n_heads, cfg.rope_base, causal)  # type: ignore[arg-type]
        self.norm2 = RMSNorm(d, cfg.norm_eps)
        self.mlp = SwiGLU(d, scale.mlp_hidden)  # type: ignore[arg-type]
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), mask))
        return x + self.drop(self.mlp(self.norm2(x)))


@register_model("transformer")
class TransformerModel(SequenceModel):
    def __init__(
        self, scale: ScaleConfig, model_cfg: TransformerConfig, spec: TaskSpec, dropout: float
    ) -> None:
        layers = [
            TransformerLayer(scale, model_cfg, spec.causal, dropout) for _ in range(scale.n_layers)
        ]
        super().__init__(spec, scale.d_model, layers, dropout, model_cfg.norm_eps)
        self._init_weights(model_cfg.init_std, scale.n_layers)
        self.tie_weights()

    def _init_weights(self, std: float, n_layers: int) -> None:
        """GPT-2 / LLaMA style: N(0, std); residual output projections scaled by 1/sqrt(2L)."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
        for layer in self.layers:
            for proj in (layer.attn.out, layer.mlp.w2):
                nn.init.normal_(proj.weight, mean=0.0, std=std / math.sqrt(2 * n_layers))
        self.init_embedding_and_head(std)
