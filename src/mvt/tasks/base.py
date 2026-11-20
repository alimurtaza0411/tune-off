"""The task contract (README §Task contract).

A task turns a dataset into ``TaskData``: a ``TaskSpec`` plus lists of examples. It does NOT
batch, pad or compute losses; the core does that identically for every task.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from mvt.core.config import TaskConfig
from mvt.core.types import Example, TaskData, TaskSpec
from mvt.data.subsample import take_fraction


class Task(ABC):
    """Abstract task. Register subclasses with ``@register_task("name")``."""

    #: Bump when processing changes; it is part of the data-cache key.
    PROCESSING_VERSION = 1
    #: Whether processed TaskData may be cached on disk (real tasks: yes).
    cacheable = True

    def __init__(self, config: TaskConfig) -> None:
        self.config = config

    @abstractmethod
    def build(self, fraction: float) -> TaskData:
        """Deterministic for a given config and fraction.

        Rules: val/test never depend on ``fraction``; the vocabulary comes from the full
        training pool; the training subset is drawn with ``config.subset_seed``.
        """

    @abstractmethod
    def compute_metric(self, predictions: Any, references: Any) -> tuple[float, float]:
        """Return ``(score_higher_is_better, raw_metric)``."""


class BaseTask(Task):
    """Task with helpers and a default metric per head.

    Default ``compute_metric`` inputs (produced by the trainer):
      * seq_cls: predictions / references = list[int]           -> accuracy
      * tok_cls: predictions / references = list[list[int]]     -> token accuracy
      * lm:      predictions = list[(sum_nll, n_tokens)]        -> (-mean NLL, perplexity)
    Override it when needed (e.g. CoNLL-2003 -> seqeval entity F1).
    """

    def make_spec(self, vocab_size: int, label_names: Sequence[str] | None = None) -> TaskSpec:
        c = self.config
        floor = c.floor if c.floor is not None else -math.log(vocab_size)  # lm: uniform guess
        return TaskSpec(
            name=c.name,
            head=c.head,
            causal=c.head == "lm",
            num_labels=c.num_labels,
            vocab_size=vocab_size,
            pad_id=0,
            max_len=c.max_len,
            metric_name=c.metric,
            floor=floor,
            epsilon=c.epsilon,
            label_names=tuple(label_names) if label_names else None,
        )

    def finalize(
        self,
        spec: TaskSpec,
        train_pool: Sequence[Example],
        val: Sequence[Example],
        test: Sequence[Example],
        fraction: float,
        stats: dict[str, int | float] | None = None,
    ) -> TaskData:
        """Subsample the training pool to ``fraction`` and collect basic stats."""
        train = take_fraction(train_pool, fraction, self.config.subset_seed)
        all_stats: dict[str, int | float] = {
            "n_train_pool": len(train_pool),
            "n_train_examples": len(train),
            "n_train_tokens": sum(len(e.input_ids) for e in train),
            "n_val": len(val),
            "n_test": len(test),
        }
        all_stats.update(stats or {})
        return TaskData(spec=spec, train=train, val=val, test=test, stats=all_stats)

    @staticmethod
    def truncate(ids: list[int], max_len: int) -> tuple[list[int], bool]:
        """Cut to ``max_len``; also report whether anything was cut (count these and log)."""
        return ids[:max_len], len(ids) > max_len

    def compute_metric(self, predictions: Any, references: Any) -> tuple[float, float]:
        head = self.config.head
        if head == "seq_cls":
            acc = sum(p == r for p, r in zip(predictions, references)) / max(len(references), 1)
            return acc, acc
        if head == "tok_cls":
            correct = sum(p == r for ps, rs in zip(predictions, references) for p, r in zip(ps, rs))
            total = sum(len(rs) for rs in references)
            acc = correct / max(total, 1)
            return acc, acc
        total_nll = sum(nll for nll, _ in predictions)
        n_tokens = sum(n for _, n in predictions)
        mean_nll = total_nll / max(n_tokens, 1)
        return -mean_nll, math.exp(min(mean_nll, 700.0))
