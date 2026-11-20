"""Models register themselves with ``@register_model("name")`` in their own file."""

from __future__ import annotations

from mvt.core.config import ModelConfig, ScaleConfig
from mvt.core.registry import MODELS, discover
from mvt.core.types import TaskSpec
from mvt.models.common import SequenceModel

__all__ = ["SequenceModel", "build_model", "count_params"]


def build_model(
    arch: str, scale: ScaleConfig, model_cfg: ModelConfig, spec: TaskSpec, dropout: float
) -> SequenceModel:
    discover("mvt.models")
    return MODELS.get(arch)(scale=scale, model_cfg=model_cfg, spec=spec, dropout=dropout)


def count_params(model: SequenceModel) -> dict[str, int]:
    return model.param_counts()
