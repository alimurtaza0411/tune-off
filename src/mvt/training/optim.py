"""AdamW with weight-decay groups and the warmup + cosine schedule (README §Training recipe)."""

from __future__ import annotations

import math

import torch
from torch import nn

from mvt.core.config import OptimizerConfig


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Decay matrices (ndim >= 2) except those flagged ``_no_weight_decay`` (Mamba A_log, D).

    Biases and norm gains (ndim < 2) are never decayed.
    """
    decay, no_decay = [], []
    for param in model.parameters():  # tied parameters are yielded once
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and not getattr(param, "_no_weight_decay", False):
            decay.append(param)
        else:
            no_decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def make_optimizer(
    model: nn.Module, lr: float, weight_decay: float, cfg: OptimizerConfig, device: torch.device
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        param_groups(model, weight_decay),
        lr=lr,
        betas=cfg.betas,
        eps=cfg.eps,
        fused=device.type == "cuda",
    )


def lr_multiplier(step: int, total_steps: int, warmup_fraction: float, final_ratio: float) -> float:
    """Linear warmup over ``warmup_fraction * total_steps``, then cosine to ``final_ratio``."""
    warmup = int(warmup_fraction * total_steps)
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def make_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_fraction: float, final_ratio: float
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_multiplier(step, total_steps, warmup_fraction, final_ratio)
    )
