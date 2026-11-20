"""Deterministic seeds (README §Seeds)."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def derive_seed(*parts: object) -> int:
    """Stable 32-bit seed from any parts, e.g. ``derive_seed("sst2-025-8m", "sobol", 3)``."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:8], 16)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and all GPUs) from one seed."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
