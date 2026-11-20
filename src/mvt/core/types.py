"""Plain data types that define the contracts between modules.

See the README section "Rules every contributor must keep".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal

HeadType = Literal["seq_cls", "tok_cls", "lm"]
Phase = Literal["sobol", "tpe", "best", "default", "pilot", "smoke"]
Status = Literal["ok", "diverged", "failed"]

REAL_PHASES: tuple[str, ...] = ("sobol", "tpe", "best", "default")
PHASE_ORDER = {"sobol": 0, "tpe": 1, "best": 2, "default": 3, "pilot": 4, "smoke": 5}


def make_cell_id(task: str, fraction: float, scale: str) -> str:
    """Cell id such as ``sst2-025-8m`` (task, training-data percent, scale)."""
    return f"{task}-{round(fraction * 100):03d}-{scale}"


@dataclass(frozen=True)
class TaskSpec:
    """Everything a model and the trainer need to know about a task."""

    name: str
    head: HeadType
    causal: bool  # True iff head == "lm"
    num_labels: int | None  # classes / tags; None for lm
    vocab_size: int
    pad_id: int  # always 0
    max_len: int  # truncation length (seq_cls / tok_cls) or seq_len (lm)
    metric_name: str
    floor: float  # score of a uniform random predictor (score units)
    epsilon: float  # robustness margin (score units)
    label_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.causal != (self.head == "lm"):
            raise ValueError(f"{self.name}: causal must be True iff head == 'lm'")
        if (self.num_labels is None) != (self.head == "lm"):
            raise ValueError(f"{self.name}: num_labels must be None iff head == 'lm'")


@dataclass(frozen=True)
class ClsExample:
    """One sentence with one label (seq_cls)."""

    input_ids: list[int]
    label: int


@dataclass(frozen=True)
class TagExample:
    """One sentence with one label per token (tok_cls)."""

    input_ids: list[int]
    labels: list[int]


@dataclass(frozen=True)
class LMChunk:
    """A contiguous window of text and its next-token targets (lm)."""

    input_ids: list[int]
    targets: list[int]


Example = ClsExample | TagExample | LMChunk


@dataclass
class TaskData:
    """Output of ``Task.build``: the spec plus train (already subsampled), val and test."""

    spec: TaskSpec
    train: Sequence[Example]
    val: Sequence[Example]
    test: Sequence[Example]
    stats: dict[str, int | float] = field(default_factory=dict)


@dataclass(frozen=True)
class HParams:
    """The 6 tuned hyperparameters (batch size stored as index and resolved value)."""

    learning_rate: float
    warmup_fraction: float
    weight_decay: float
    dropout: float
    batch_size_index: int
    batch_size: int
    grad_clip: float  # 0.0 = no clipping

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class TrialRequest:
    """One training run to perform."""

    task: str
    fraction: float
    scale: str
    arch: str
    phase: Phase
    index: int  # index within phase (replicate index for best/default)
    sobol_index: int | None  # set for phase "sobol"
    hparams: HParams
    seed: int
    evaluate_test: bool  # True only for best/default

    @property
    def cell_id(self) -> str:
        return make_cell_id(self.task, self.fraction, self.scale)
