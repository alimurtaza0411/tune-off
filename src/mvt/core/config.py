"""Typed configs loaded from ``configs/*.yaml`` (README §Config contract).

Every experimental constant lives in YAML. This module only parses and validates it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

REPO_ROOT = Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    return Path(os.environ.get("MVT_CONFIG_DIR", REPO_ROOT / "configs"))


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------- experiment.yaml
class PhaseCounts(_Strict):
    sobol: int
    tpe: int
    replicates: int

    def expected(self) -> dict[str, int]:
        """Expected number of results per (cell, arch, phase)."""
        r = self.replicates
        return {"sobol": self.sobol, "tpe": self.tpe, "best": r, "default": r}


class SmokeConfig(_Strict):
    tasks: list[str]
    fraction: float
    scale: str
    phases: PhaseCounts


class ExperimentConfig(_Strict):
    tasks: list[str]
    fractions: list[float]
    scales: list[str]
    archs: list[str]
    phases: PhaseCounts
    n_evals: int
    smoke: SmokeConfig


# ---------------------------------------------------------------- base.yaml
class OptimizerConfig(_Strict):
    betas: tuple[float, float]
    eps: float


class ScheduleConfig(_Strict):
    final_lr_ratio: float


class RuntimeConfig(_Strict):
    results_root: str
    data_cache_dir: str
    num_workers: int = 0
    device: str = "auto"


class BaseConfig(_Strict):
    optimizer: OptimizerConfig
    schedule: ScheduleConfig
    precision: Literal["fp16", "fp32"]
    runtime: RuntimeConfig


# ---------------------------------------------------------------- search_space.yaml
class ParamSpec(_Strict):
    name: str
    type: Literal["float", "categorical"]
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: list[int | float] | None = None
    default: int | float


class SobolConfig(_Strict):
    seed: int
    m: int
    table: str


class SearchSpaceConfig(_Strict):
    sobol: SobolConfig
    params: list[ParamSpec]


# ---------------------------------------------------------------- tasks/*.yaml
class DatasetConfig(_Strict):
    id: str
    config: str | None = None
    revision: str | None = None


class TaskConfig(_Strict):
    name: str
    head: Literal["seq_cls", "tok_cls", "lm"]
    num_labels: int | None
    metric: str
    floor: float | None  # None -> computed by the task (e.g. -ln V for lm)
    epsilon: float
    max_len: int
    min_freq: int = 1
    batch_sizes: list[int]
    epochs: dict[int, int]  # training-data percent -> epochs
    subset_seed: int
    holdout_seed: int | None = None
    holdout_fraction: float | None = None
    dataset: DatasetConfig | None = None
    options: dict[str, Any] = {}

    @field_validator("batch_sizes")
    @classmethod
    def _four_batch_sizes(cls, v: list[int]) -> list[int]:
        if len(v) != 4:
            raise ValueError("batch_sizes must list exactly 4 choices")
        return v

    def epochs_for(self, fraction: float) -> int:
        return self.epochs[round(fraction * 100)]


# ---------------------------------------------------------------- models/*.yaml, scales.yaml
class TransformerConfig(_Strict):
    rope_base: float
    norm_eps: float
    init_std: float


class MambaConfig(_Strict):
    d_state: int
    d_conv: int
    expand: int
    dt_min: float
    dt_max: float
    dt_init_floor: float
    conv_bias: bool
    bias: bool
    impl: Literal["auto", "cuda", "ref"]
    norm_eps: float
    init_std: float


ModelConfig = TransformerConfig | MambaConfig
MODEL_CONFIGS: dict[str, type[BaseModel]] = {"transformer": TransformerConfig, "mamba": MambaConfig}


class ScaleConfig(_Strict):
    """One scale resolved for one architecture."""

    name: str
    d_model: int
    n_layers: int
    n_heads: int | None = None
    mlp_hidden: int | None = None
    expected_non_embedding: int | None = None
    extension: bool = False


# ---------------------------------------------------------------- loaders
def _read_yaml(relative: str) -> Any:
    return yaml.safe_load((config_dir() / relative).read_text())


def load_experiment() -> ExperimentConfig:
    return ExperimentConfig(**_read_yaml("experiment.yaml"))


def load_base() -> BaseConfig:
    raw = _read_yaml("base.yaml")
    env = {
        "results_root": "MVT_RESULTS_ROOT",
        "data_cache_dir": "MVT_DATA_CACHE",
        "device": "MVT_DEVICE",
    }
    for key, var in env.items():
        if os.environ.get(var):
            raw["runtime"][key] = os.environ[var]
    return BaseConfig(**raw)


def load_search_space() -> SearchSpaceConfig:
    return SearchSpaceConfig(**_read_yaml("search_space.yaml"))


def load_task_config(name: str) -> TaskConfig:
    path = config_dir() / "tasks" / f"{name}.yaml"
    if path.exists():
        return TaskConfig(**_read_yaml(f"tasks/{name}.yaml"))
    debug = _read_yaml("tasks/debug.yaml")
    if name in debug:
        return TaskConfig(**debug[name])
    raise FileNotFoundError(f"No task config for '{name}' in {config_dir() / 'tasks'}")


def load_model_config(arch: str) -> ModelConfig:
    return MODEL_CONFIGS[arch](**_read_yaml(f"models/{arch}.yaml"))  # type: ignore[return-value]


def load_scale(name: str, arch: str) -> ScaleConfig:
    raw = _read_yaml("scales.yaml")[name]
    return ScaleConfig(
        name=name, d_model=raw["d_model"], extension=raw.get("extension", False), **raw[arch]
    )


@dataclass(frozen=True)
class ResolvedConfig:
    """Everything one trial needs: experiment + base + task + model + scale + search space."""

    task_name: str
    fraction: float
    arch: str
    experiment: ExperimentConfig
    base: BaseConfig
    task: TaskConfig
    model: ModelConfig
    scale: ScaleConfig
    search_space: SearchSpaceConfig

    def hash(self) -> str:
        return config_hash(self)


def load_resolved_config(task: str, fraction: float, scale: str, arch: str) -> ResolvedConfig:
    return ResolvedConfig(
        task_name=task,
        fraction=fraction,
        arch=arch,
        experiment=load_experiment(),
        base=load_base(),
        task=load_task_config(task),
        model=load_model_config(arch),
        scale=load_scale(scale, arch),
        search_space=load_search_space(),
    )


RUNTIME_ONLY_FIELDS = ("results_root", "data_cache_dir", "num_workers", "device")


def config_hash(cfg: ResolvedConfig) -> str:
    """First 12 hex chars of SHA-256 over the canonical JSON, without runtime-only fields.

    The hash identifies a CELL, so it covers both architectures' model and scale settings and
    is identical for the Transformer and Mamba trials of the same cell.
    """
    base = cfg.base.model_dump(mode="json")
    base["runtime"] = {k: v for k, v in base["runtime"].items() if k not in RUNTIME_ONLY_FIELDS}
    payload = {
        "task_name": cfg.task_name,
        "fraction": cfg.fraction,
        "experiment": cfg.experiment.model_dump(mode="json"),
        "base": base,
        "task": cfg.task.model_dump(mode="json"),
        "models": {a: load_model_config(a).model_dump(mode="json") for a in MODEL_CONFIGS},
        "scales": {a: load_scale(cfg.scale.name, a).model_dump(mode="json") for a in MODEL_CONFIGS},
        "search_space": cfg.search_space.model_dump(mode="json"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]
