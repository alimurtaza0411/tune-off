"""Word-level vocabulary. ``<pad>`` = 0 and ``<unk>`` = 1 always."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1


class Vocab:
    def __init__(self, itos: Sequence[str]) -> None:
        if list(itos[:2]) != [PAD, UNK]:
            raise ValueError("The vocabulary must start with <pad>, <unk>")
        self.itos = list(itos)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    @classmethod
    def build(
        cls,
        sentences: Iterable[Sequence[str]],
        min_freq: int = 1,
        extra_specials: Sequence[str] = (),
    ) -> Vocab:
        """Build from the FULL training pool (never from a fraction).

        Words are ordered by frequency, then alphabetically, so the result is deterministic.
        """
        counts = Counter(tok for sent in sentences for tok in sent)
        specials = [PAD, UNK, *extra_specials]
        words = [w for w, c in counts.items() if c >= min_freq and w not in specials]
        words.sort(key=lambda w: (-counts[w], w))
        return cls(specials + words)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        return [self.stoi.get(tok, UNK_ID) for tok in tokens]

    def __len__(self) -> int:
        return len(self.itos)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.itos))

    @classmethod
    def load(cls, path: str | Path) -> Vocab:
        return cls(json.loads(Path(path).read_text()))
