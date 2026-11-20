"""Results contract, schema version 1 (README §Results contract).

One JSON file per trial at ``{root}/{cell_id}/{arch}/{phase}/{index:03d}.json``.
Files are written atomically and never overwritten unless ``force=True``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from mvt.core.types import PHASE_ORDER, REAL_PHASES

SCHEMA_VERSION = 1

HP_COLUMNS = [
    "learning_rate",
    "warmup_fraction",
    "weight_decay",
    "dropout",
    "batch_size_index",
    "batch_size",
    "grad_clip",
]
MODEL_COLUMNS = [
    "d_model",
    "n_layers",
    "n_heads",
    "mlp_hidden",
    "d_state",
    "expand",
    "d_conv",
    "impl",
]


class TrialResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    run_id: str
    cell_id: str
    task: str
    fraction: float
    scale: str
    arch: str
    phase: str
    index: int
    sobol_index: int | None
    seed: int
    hparams: dict[str, float | int]
    model: dict[str, Any]
    status: str  # ok | diverged | failed
    val_score: float | None = None
    val_metric: float | None = None
    test_score: float | None = None
    test_metric: float | None = None
    floor: float | None = None
    epsilon: float | None = None
    val_curve: list[float | None] = []
    train_loss_curve: list[float | None] = []
    epochs: int | None = None
    best_step: int | None = None
    total_steps: int | None = None
    tokens_seen: int | None = None
    diverged_at_step: int | None = None
    wall_time_s: float | None = None
    train_time_s: float | None = None
    peak_mem_mb: float | None = None
    n_params_total: int | None = None
    n_params_non_embedding: int | None = None
    precision: str | None = None
    device: str | None = None
    gpu_name: str | None = None
    mamba_impl: str | None = None
    git_commit: str
    git_dirty: bool
    code_version: str
    config_hash: str
    torch_version: str | None = None
    mamba_ssm_version: str | None = None
    owner: str
    started_at: str
    finished_at: str
    error: str | None = None


def floored_score(status: str, val_score: float | None, floor: float) -> float:
    """Analysis score: the floor for diverged runs, else max(val_score, floor)."""
    if status == "diverged" or val_score is None or val_score != val_score:  # NaN check
        return floor
    return max(val_score, floor)


# ---------------------------------------------------------------- file I/O
def result_path(root: str | Path, cell_id: str, arch: str, phase: str, index: int) -> Path:
    return Path(root) / cell_id / arch / phase / f"{index:03d}.json"


def write_result(result: TrialResult, root: str | Path, force: bool = False) -> Path:
    """Atomically write one result (temp file, then rename). Never overwrites unless forced."""
    path = result_path(root, result.cell_id, result.arch, result.phase, result.index)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists (use force=True to overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(result.model_dump_json(indent=1))
        os.replace(tmp, path)
    finally:
        if tmp.exists():  # interrupted before the rename: leave no partial file behind
            tmp.unlink()
    return path


def read_result(path: str | Path) -> TrialResult:
    return TrialResult.model_validate_json(Path(path).read_text())


def iter_results(root: str | Path, cell_id: str | None = None) -> Iterator[TrialResult]:
    pattern = f"{cell_id or '*'}/*/*/*.json"
    for path in sorted(Path(root).glob(pattern)):
        yield read_result(path)


# ---------------------------------------------------------------- tables
def results_to_frame(results: Iterable[TrialResult]) -> pd.DataFrame:
    """Flatten results to one row each: ``hparams`` -> ``hp_*``, ``model`` -> ``model_*``."""
    rows = []
    for r in results:
        row = r.model_dump()
        hp, model = row.pop("hparams"), row.pop("model")
        row.update({f"hp_{k}": hp.get(k) for k in HP_COLUMNS})
        row.update({f"model_{k}": model.get(k) for k in MODEL_COLUMNS})
        row["val_curve"] = json.dumps(row["val_curve"])
        row["train_loss_curve"] = json.dumps(row["train_loss_curve"])
        rows.append(row)
    columns = _column_order()
    return pd.DataFrame(rows, columns=columns)


def _column_order() -> list[str]:
    fields = list(TrialResult.model_fields)
    head = fields[: fields.index("hparams")]
    tail = [f for f in fields[fields.index("hparams") :] if f not in ("hparams", "model")]
    hp = [f"hp_{k}" for k in HP_COLUMNS]
    model = [f"model_{k}" for k in MODEL_COLUMNS]
    return head + hp + model + tail


def export_cell(root: str | Path, cell_id: str, out_dir: str | Path = "results") -> Path:
    """Write ``{out_dir}/{cell_id}.csv`` with both architectures and all real phases."""
    results = [r for r in iter_results(root, cell_id) if r.phase in REAL_PHASES]
    if not results:
        raise FileNotFoundError(f"No real-phase results for cell {cell_id} under {root}")
    results.sort(key=lambda r: (r.arch, PHASE_ORDER[r.phase], r.index))
    out = Path(out_dir) / f"{cell_id}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results_to_frame(results).to_csv(out, index=False)
    return out


def validate_frame(df: pd.DataFrame, expected: dict[str, int] | None = None) -> list[str]:
    """Return a list of problems (empty = valid). ``expected`` maps phase -> count per arch."""
    problems: list[str] = []
    missing = [c for c in _column_order() if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if (df["schema_version"] != SCHEMA_VERSION).any():
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    bad_status = set(df["status"]) - {"ok", "diverged", "failed"}
    if bad_status:
        problems.append(f"unknown status values: {bad_status}")

    for cell, g in df.groupby("cell_id"):
        if g["config_hash"].nunique() > 1:
            problems.append(f"{cell}: mixes config hashes {sorted(g['config_hash'].unique())}")
        families = {".".join(str(v).split(".")[:2]) for v in g["code_version"]}
        if len(families) > 1:
            problems.append(f"{cell}: mixes code versions {sorted(families)}")
        n_failed = int((g["status"] == "failed").sum())
        if n_failed:
            problems.append(f"{cell}: {n_failed} failed trial(s) must be re-run")
        dup = g.duplicated(["arch", "phase", "index"])
        if dup.any():
            problems.append(f"{cell}: {int(dup.sum())} duplicate (arch, phase, index) rows")
        if expected:
            for arch in sorted(g["arch"].unique()):
                for phase, n in expected.items():
                    got = int(((g["arch"] == arch) & (g["phase"] == phase)).sum())
                    if got != n:
                        problems.append(f"{cell}/{arch}/{phase}: {got} results, expected {n}")
        sob = g[g["phase"] == "sobol"]
        hp_cols = [f"hp_{k}" for k in HP_COLUMNS]
        for idx, s in sob.groupby("sobol_index"):
            if len(s[hp_cols].drop_duplicates()) > 1:
                problems.append(f"{cell}: sobol_index {idx} has different hparams across archs")
            if s["seed"].nunique() > 1:
                problems.append(f"{cell}: sobol_index {idx} has different seeds across archs")
    return problems
