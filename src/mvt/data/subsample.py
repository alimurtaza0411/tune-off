"""Deterministic training-fraction subsets and hold-out splits."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def take_fraction(items: Sequence[T], fraction: float, seed: int) -> list[T]:
    """A fixed random subset of ``round(len * fraction)`` items, kept in original order."""
    if fraction >= 1.0:
        return list(items)
    n = int(round(len(items) * fraction))
    keep = np.sort(np.random.default_rng(seed).permutation(len(items))[:n])
    return [items[i] for i in keep]


def holdout_split(
    items: Sequence[T], holdout_fraction: float, seed: int
) -> tuple[list[T], list[T]]:
    """Split into (kept, held_out); e.g. SST-2's 90/10 train-pool / validation split."""
    perm = np.random.default_rng(seed).permutation(len(items))
    n_held = int(round(len(items) * holdout_fraction))
    held, kept = np.sort(perm[:n_held]), np.sort(perm[n_held:])
    return [items[i] for i in kept], [items[i] for i in held]
