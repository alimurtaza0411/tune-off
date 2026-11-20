"""Load exported results for analysis, applying the inclusion and scoring rules.

* Inputs: ``results/*.csv`` only.
* Phases: sobol, tpe, best, default (pilot and smoke are excluded).
* Statuses: ok and diverged; failed rows are dropped and counted.
* Score: ``s = floor`` for diverged runs, else ``max(val_score, floor)``  (column ``score``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from mvt.core.results import validate_frame
from mvt.core.types import REAL_PHASES

log = logging.getLogger(__name__)


def _expand(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files: list[Path] = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.csv")) if p.is_dir() else [p]
    return files


def load_results(
    paths: str | Path | Iterable[str | Path] = "results",
    expected: dict[str, int] | None = None,
    archs: list[str] | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    files = _expand(paths)
    if not files:
        raise FileNotFoundError(f"No result CSVs found in {paths}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    problems = [p for p in validate_frame(df, expected, archs) if "failed trial" not in p]
    if problems:
        message = "Result validation problems:\n  " + "\n  ".join(problems)
        if strict:
            raise ValueError(message)
        log.warning(message)

    n_failed = int((df["status"] == "failed").sum())
    if n_failed:
        log.warning("Dropping %d failed trial(s); they must be re-run", n_failed)
    df = df[df["phase"].isin(REAL_PHASES) & df["status"].isin(["ok", "diverged"])].copy()
    diverged = df["status"] == "diverged"
    df["score"] = df["val_score"].where(~diverged, df["floor"]).fillna(df["floor"])
    df["score"] = df[["score", "floor"]].max(axis=1)
    df.attrs["n_failed"] = n_failed
    return df.reset_index(drop=True)
