"""Parts shared by both architectures (README §Models, shared rules).

token embedding -> dropout -> N residual layers -> final RMSNorm -> task head
"""

from __future__ import annotations

import torch
from torch import nn

from mvt.core.types import TaskSpec


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight).to(x.dtype)


class SequenceModel(nn.Module):
    """Embedding, residual layers, final norm and head. Subclasses supply the layers.

    Each layer is called as ``layer(x, attention_mask)`` and returns the new residual stream.
    """

    def __init__(
        self,
        spec: TaskSpec,
        d_model: int,
        layers: list[nn.Module],
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.embed = nn.Embedding(spec.vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(d_model, norm_eps)
        if spec.head == "lm":
            self.head = nn.Linear(d_model, spec.vocab_size, bias=False)  # tied in tie_weights()
        else:
            self.head = nn.Linear(d_model, spec.num_labels, bias=True)  # type: ignore[arg-type]

    def init_embedding_and_head(self, std: float) -> None:
        nn.init.normal_(self.embed.weight, mean=0.0, std=std)
        nn.init.normal_(self.head.weight, mean=0.0, std=std)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def tie_weights(self) -> None:
        """Call AFTER initialisation: the LM output projection shares the embedding matrix."""
        if self.spec.head == "lm":
            self.head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """seq_cls -> [B, C]; tok_cls -> [B, L, C]; lm -> [B, L, V]."""
        x = self.drop(self.embed(input_ids))
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        if self.spec.head == "seq_cls":  # masked mean pooling over real tokens
            m = attention_mask.unsqueeze(-1).to(x.dtype)
            x = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(x)

    def param_counts(self) -> dict[str, int]:
        """Non-embedding = everything except the embedding matrix and the task head."""
        total = sum(p.numel() for p in self.parameters())  # shared (tied) tensors counted once
        embedding = self.embed.weight.numel()
        tied = self.head.weight is self.embed.weight
        head = 0 if tied else sum(p.numel() for p in self.head.parameters())
        return {
            "total": total,
            "embedding": embedding,
            "head": head,
            "non_embedding": total - embedding - head,
        }
