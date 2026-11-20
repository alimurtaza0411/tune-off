"""The pilot (README §Pilot protocol). Pilot results are NEVER used in the analysis.

1. Learning-rate sweep at 8m: 7 log-spaced learning rates, other hparams at lambda_0.
2. Scale check at 2m: one run per (fraction, arch) at that arch's best sweep learning rate.
Then the pre-registered rules are evaluated and printed (applying them is a human decision).
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mvt.core.config import load_resolved_config
from mvt.core.results import TrialResult, floored_score, read_result, result_path, write_result
from mvt.core.seeding import derive_seed
from mvt.core.types import TrialRequest, make_cell_id
from mvt.search.space import SearchSpace
from mvt.search.trial import run_trial

log = logging.getLogger(__name__)
N_SWEEP = 7


def _run(request: TrialRequest, root: Path, owner: str) -> TrialResult:
    path = result_path(root, request.cell_id, request.arch, "pilot", request.index)
    if path.exists():
        return read_result(path)
    cfg = load_resolved_config(request.task, request.fraction, request.scale, request.arch)
    result = run_trial(request, cfg, owner)
    write_result(result, root)
    log.info(
        "pilot %s %s lr=%.2e -> %s",
        request.cell_id,
        request.arch,
        request.hparams.learning_rate,
        result.val_score,
    )
    return result


def _position(r: TrialResult) -> float:
    return (r.best_step or 0) / max(r.total_steps or 1, 1)


def run_pilot(
    task: str,
    fractions: list[float],
    archs: list[str],
    root: str | Path,
    owner: str,
    sweep_scale: str = "8m",
    check_scale: str = "2m",
) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for fraction in fractions:
        cfg = load_resolved_config(task, fraction, sweep_scale, archs[0])
        space = SearchSpace(cfg.search_space, cfg.task.batch_sizes)
        lr_spec = space.params["learning_rate"]
        lrs = np.geomspace(lr_spec.low, lr_spec.high, N_SWEEP)
        sweep_cell = make_cell_id(task, fraction, sweep_scale)
        check_cell = make_cell_id(task, fraction, check_scale)

        for arch in archs:
            sweep = []
            for i, lr in enumerate(lrs):
                hp = dataclasses.replace(space.defaults(), learning_rate=float(lr))
                req = TrialRequest(
                    task,
                    fraction,
                    sweep_scale,
                    arch,
                    "pilot",
                    i,
                    None,
                    hp,
                    derive_seed(sweep_cell, "pilot", i),
                    False,
                )
                sweep.append(_run(req, root, owner))
            scores = [floored_score(r.status, r.val_score, r.floor) for r in sweep]  # type: ignore[arg-type]
            best_i = int(np.argmax(scores))
            hp = dataclasses.replace(space.defaults(), learning_rate=float(lrs[best_i]))
            req = TrialRequest(
                task,
                fraction,
                check_scale,
                arch,
                "pilot",
                0,
                None,
                hp,
                derive_seed(check_cell, "pilot", 0),
                False,
            )
            check = _run(req, root, owner)
            best = sweep[best_i]
            rows.append(
                {
                    "fraction": fraction,
                    "arch": arch,
                    "best_lr": float(lrs[best_i]),
                    "lr_at_edge": best_i in (0, N_SWEEP - 1),
                    "best_val": scores[best_i],
                    "floor": best.floor,
                    "peak_pos_8m": _position(best),
                    "peak_pos_2m": _position(check),
                    "sec_per_run_8m": float(np.mean([r.wall_time_s or 0 for r in sweep])),
                    "sec_per_run_2m": check.wall_time_s,
                    "epochs": best.epochs,
                }
            )
    return pd.DataFrame(rows)


def rule_outcomes(summary: pd.DataFrame) -> list[str]:
    """Evaluate the pre-registered pilot rules on a summary from ``run_pilot``."""
    notes = []
    if summary["lr_at_edge"].any():
        edges = summary[summary["lr_at_edge"]][["fraction", "arch", "best_lr"]]
        notes.append(f"LR RANGE: best LR at an edge -> widen that end by one decade:\n{edges}")
    else:
        notes.append("LR RANGE: keep")
    for fraction, g in summary.groupby("fraction"):
        positions = list(g["peak_pos_8m"]) + list(g["peak_pos_2m"])
        if max(positions) > 0.8:
            verdict = "DOUBLE epochs (a good run is still improving in the last 20%)"
        elif max(positions) < 0.3:
            verdict = "HALVE epochs (all good runs peak in the first 30%)"
        else:
            verdict = "keep epochs"
        notes.append(f"TRAINING LENGTH fraction={fraction}: {verdict}; positions={positions}")
    weak = summary[summary["best_val"] <= summary["floor"]]
    notes.append("SANITY: ok" if weak.empty else f"SANITY: not learning -> investigate:\n{weak}")
    return notes
