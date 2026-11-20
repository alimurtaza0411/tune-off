"""Synthetic debug tasks for tests and ``mvt smoke``: tiny, deterministic, no downloads."""

from __future__ import annotations

import numpy as np

from mvt.core.registry import register_task
from mvt.core.types import ClsExample, Example, LMChunk, TagExample, TaskData
from mvt.tasks.base import BaseTask

VOCAB_SIZE = 32  # ids 0 (<pad>) and 1 (<unk>) are reserved; real tokens are 2..31


class _DebugTask(BaseTask):
    cacheable = False  # fast to rebuild

    def build(self, fraction: float) -> TaskData:
        o = self.config.options
        sizes = {"train": o["n_train"], "val": o["n_val"], "test": o["n_test"]}
        splits = {
            name: self.make_examples(n, np.random.default_rng([o["data_seed"], k]))
            for k, (name, n) in enumerate(sizes.items())
        }
        spec = self.make_spec(vocab_size=VOCAB_SIZE)
        return self.finalize(spec, splits["train"], splits["val"], splits["test"], fraction)

    def make_examples(self, n: int, rng: np.random.Generator) -> list[Example]:
        raise NotImplementedError

    def _length(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.config.options["min_len"], self.config.max_len + 1))


@register_task("debug_cls")
class DebugCls(_DebugTask):
    """Label = 1 iff token 3 appears in the first half of the sequence."""

    def make_examples(self, n: int, rng: np.random.Generator) -> list[Example]:
        out: list[Example] = []
        for i in range(n):
            length = self._length(rng)
            ids = rng.integers(4, VOCAB_SIZE, size=length)
            label = i % 2
            if label:
                ids[rng.integers(0, length // 2)] = 3
            elif rng.random() < 0.5:  # distractor: token 3 in the second half
                ids[rng.integers(length // 2, length)] = 3
            out.append(ClsExample(input_ids=ids.tolist(), label=label))
        return out


@register_task("debug_tag")
class DebugTag(_DebugTask):
    """Tag of each token = (previous token id) mod 3; the first token gets tag 0."""

    def make_examples(self, n: int, rng: np.random.Generator) -> list[Example]:
        out: list[Example] = []
        for _ in range(n):
            ids = rng.integers(2, VOCAB_SIZE, size=self._length(rng)).tolist()
            labels = [0] + [prev % 3 for prev in ids[:-1]]
            out.append(TagExample(input_ids=ids, labels=labels))
        return out


@register_task("debug_lm")
class DebugLM(_DebugTask):
    """A token stream made of short random motifs, each repeated a few times."""

    def make_examples(self, n: int, rng: np.random.Generator) -> list[Example]:
        o, seq_len = self.config.options, self.config.max_len
        stream: list[int] = []
        while len(stream) < n * (seq_len + 1):
            motif = rng.integers(2, VOCAB_SIZE, size=o["motif_len"]).tolist()
            stream += motif * o["motif_repeats"]
        out: list[Example] = []
        for i in range(n):
            window = stream[i * (seq_len + 1) : (i + 1) * (seq_len + 1)]
            out.append(LMChunk(input_ids=window[:-1], targets=window[1:]))
        return out
