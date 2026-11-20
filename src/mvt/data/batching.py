"""Batching shared by every task (README: "Rules every contributor must keep").

Every batch is a dict with ``input_ids`` [B, L], ``attention_mask`` [B, L] (True = real token,
real tokens always form a prefix), ``labels`` and ``n_tokens``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np
import torch

from mvt.core.types import ClsExample, Example, HeadType, LMChunk, TagExample

POOL_BATCHES = 50  # length-sorting happens inside pools of 50 batches
IGNORE_INDEX = -100


def plan_total_steps(epochs: int, n_units: int, batch_size: int) -> int:
    return epochs * math.ceil(n_units / batch_size)


def plan_epoch(
    n: int, batch_size: int, rng: np.random.Generator, lengths: np.ndarray | None = None
) -> list[np.ndarray]:
    """Index batches for one epoch.

    Shuffle; if ``lengths`` is given, sort by length inside pools of 50 batches (less padding);
    cut into batches of ``batch_size`` (only the very last may be smaller); shuffle batch order.
    """
    order = rng.permutation(n)
    if lengths is None:
        return [order[i : i + batch_size] for i in range(0, n, batch_size)]
    pool = POOL_BATCHES * batch_size
    batches: list[np.ndarray] = []
    for start in range(0, n, pool):
        idx = order[start : start + pool]
        idx = idx[np.argsort(lengths[idx], kind="stable")]
        batches += [idx[i : i + batch_size] for i in range(0, len(idx), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def collate(examples: Sequence[Example], head: HeadType, pad_id: int = 0) -> dict:
    """Right-pad a list of examples into one batch."""
    if head == "lm":
        ids = torch.tensor([e.input_ids for e in examples], dtype=torch.long)  # type: ignore[union-attr]
        labels = torch.tensor([e.targets for e in examples], dtype=torch.long)  # type: ignore[union-attr]
        mask = torch.ones_like(ids, dtype=torch.bool)
        return {"input_ids": ids, "attention_mask": mask, "labels": labels, "n_tokens": ids.numel()}

    batch, length = len(examples), max(len(e.input_ids) for e in examples)
    ids = torch.full((batch, length), pad_id, dtype=torch.long)
    mask = torch.zeros((batch, length), dtype=torch.bool)
    for b, e in enumerate(examples):
        ids[b, : len(e.input_ids)] = torch.tensor(e.input_ids)
        mask[b, : len(e.input_ids)] = True
    if head == "seq_cls":
        labels = torch.tensor([e.label for e in examples], dtype=torch.long)  # type: ignore[union-attr]
    else:
        labels = torch.full((batch, length), IGNORE_INDEX, dtype=torch.long)
        for b, e in enumerate(examples):
            labels[b, : len(e.labels)] = torch.tensor(e.labels)  # type: ignore[union-attr]
    return {"input_ids": ids, "attention_mask": mask, "labels": labels, "n_tokens": int(mask.sum())}


def _lengths(examples: Sequence[Example], head: HeadType) -> np.ndarray | None:
    if head == "lm":
        return None
    return np.array([len(e.input_ids) for e in examples])


def iterate_train_batches(
    examples: Sequence[Example], head: HeadType, batch_size: int, epochs: int, seed: int
) -> Iterator[dict]:
    """All training batches for a trial, in order. Epoch ``e`` is shuffled with (seed, e)."""
    lengths = _lengths(examples, head)
    for epoch in range(epochs):
        rng = np.random.default_rng([seed, epoch])
        for idx in plan_epoch(len(examples), batch_size, rng, lengths):
            yield collate([examples[i] for i in idx], head)


def iterate_eval_batches(
    examples: Sequence[Example], head: HeadType, batch_size: int
) -> Iterator[dict]:
    """Deterministic evaluation batches (length-sorted to reduce padding)."""
    lengths = _lengths(examples, head)
    order = np.arange(len(examples)) if lengths is None else np.argsort(lengths, kind="stable")
    for i in range(0, len(order), batch_size):
        yield collate([examples[j] for j in order[i : i + batch_size]], head)


__all__ = [
    "ClsExample",
    "LMChunk",
    "TagExample",
    "collate",
    "iterate_eval_batches",
    "iterate_train_batches",
    "plan_epoch",
    "plan_total_steps",
]
