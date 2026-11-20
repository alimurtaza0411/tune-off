"""The 6-dimensional search space (README: "What gets tuned").

A point is either a unit vector ``u`` in [0, 1)^6 (Sobol) or an Optuna parameter dict (TPE).
Both are converted to ``HParams``. Batch size is searched as a choice INDEX (0-3), so a Sobol
point means the same relative setting on every task; the index is resolved per task.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from optuna.distributions import BaseDistribution, CategoricalDistribution, FloatDistribution

from mvt.core.config import ParamSpec, SearchSpaceConfig
from mvt.core.types import HParams

PARAM_ORDER = (
    "learning_rate",
    "warmup_fraction",
    "weight_decay",
    "dropout",
    "batch_size_index",
    "grad_clip",
)


class SearchSpace:
    def __init__(self, cfg: SearchSpaceConfig, batch_sizes: Sequence[int]) -> None:
        names = tuple(p.name for p in cfg.params)
        if names != PARAM_ORDER:
            raise ValueError(f"search_space.yaml params must be, in order: {PARAM_ORDER}")
        self.params = {p.name: p for p in cfg.params}
        self.batch_sizes = list(batch_sizes)

    @property
    def dim(self) -> int:
        return len(self.params)

    # ------------------------------------------------------------ conversions
    @staticmethod
    def _from_unit(p: ParamSpec, u: float) -> float | int:
        if p.type == "categorical":
            k = len(p.choices)  # type: ignore[arg-type]
            return p.choices[min(math.floor(u * k), k - 1)]  # type: ignore[index]
        lo, hi = p.low, p.high
        if p.log:
            return math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))  # type: ignore[arg-type]
        return lo + u * (hi - lo)  # type: ignore[operator]

    def _make(self, values: dict[str, float | int]) -> HParams:
        index = int(values["batch_size_index"])
        return HParams(
            learning_rate=float(values["learning_rate"]),
            warmup_fraction=float(values["warmup_fraction"]),
            weight_decay=float(values["weight_decay"]),
            dropout=float(values["dropout"]),
            batch_size_index=index,
            batch_size=self.batch_sizes[index],
            grad_clip=float(values["grad_clip"]),
        )

    def from_unit(self, u: Sequence[float]) -> HParams:
        """Map a Sobol point u in [0, 1)^6 to hyperparameters."""
        if len(u) != self.dim:
            raise ValueError(f"expected {self.dim} values, got {len(u)}")
        return self._make(
            {name: self._from_unit(p, ui) for (name, p), ui in zip(self.params.items(), u)}
        )

    def defaults(self) -> HParams:
        """lambda_0: the shared default recipe (a reference point, never a search start)."""
        return self._make({name: p.default for name, p in self.params.items()})

    def contains(self, hp: HParams) -> bool:
        values = self.to_optuna_params(hp)
        for name, p in self.params.items():
            v = values[name]
            if p.type == "categorical":
                if v not in p.choices:  # type: ignore[operator]
                    return False
            elif not (p.low <= v <= p.high):  # type: ignore[operator]
                return False
        return True

    # ------------------------------------------------------------ Optuna
    def distributions(self) -> dict[str, BaseDistribution]:
        dists: dict[str, BaseDistribution] = {}
        for name, p in self.params.items():
            if p.type == "categorical":
                dists[name] = CategoricalDistribution(list(p.choices))  # type: ignore[arg-type]
            else:
                dists[name] = FloatDistribution(p.low, p.high, log=p.log)  # type: ignore[arg-type]
        return dists

    def to_optuna_params(self, hp: HParams) -> dict[str, float | int]:
        return {
            "learning_rate": hp.learning_rate,
            "warmup_fraction": hp.warmup_fraction,
            "weight_decay": hp.weight_decay,
            "dropout": hp.dropout,
            "batch_size_index": hp.batch_size_index,
            "grad_clip": hp.grad_clip,
        }

    def from_optuna_params(self, params: dict[str, float | int]) -> HParams:
        return self._make(params)

    def from_dict(self, hp: dict[str, float | int]) -> HParams:
        """Rebuild HParams from a saved result's ``hparams`` field."""
        return self._make(hp)
