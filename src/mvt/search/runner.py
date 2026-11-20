"""Run one cell, resumably (README: "Running experiments").

Order inside a phase is interleaved: (index 0, transformer), (index 0, mamba), (index 1, ...),
so both architectures progress together on the same machine. Existing result files are
skipped, so re-running after a disconnect simply continues.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Iterator
from pathlib import Path

from mvt.core.config import PhaseCounts, load_resolved_config
from mvt.core.env import git_state, resolve_device
from mvt.core.results import (
    TrialResult,
    floored_score,
    iter_results,
    read_result,
    result_path,
    write_result,
)
from mvt.core.seeding import derive_seed
from mvt.core.types import PHASE_ORDER, REAL_PHASES, HParams, TrialRequest, make_cell_id
from mvt.search import tpe
from mvt.search.sobol import load_table
from mvt.search.space import SearchSpace
from mvt.search.trial import failed_result, load_task_data, mamba_impl_for, now_iso, run_trial

log = logging.getLogger(__name__)

PHASE_GROUPS = {
    "all": ("sobol", "tpe", "best", "default"),
    "sobol": ("sobol",),
    "tpe": ("tpe",),
    "replicates": ("best", "default"),
}


class CellRunner:
    def __init__(
        self,
        task: str,
        fraction: float,
        scale: str,
        archs: list[str],
        results_root: str | Path,
        owner: str,
        counts: PhaseCounts | None = None,
        enforce_rules: bool = True,
    ) -> None:
        self.task, self.fraction, self.scale, self.archs = task, fraction, scale, archs
        self.cell_id = make_cell_id(task, fraction, scale)
        self.root, self.owner = Path(results_root), owner
        self.cfgs = {a: load_resolved_config(task, fraction, scale, a) for a in archs}
        first = self.cfgs[archs[0]]
        self.counts = counts or first.experiment.phases
        self.enforce_rules = enforce_rules
        self.space = SearchSpace(first.search_space, first.task.batch_sizes)
        self.sobol_table = load_table(first.search_space)[: self.counts.sobol]
        self.n_run = 0

    # ------------------------------------------------------------ rules
    def check_rules(self) -> None:
        """Real phases need an owner and a clean git tree; never mix config hashes in a cell.

        Only real-phase results are compared: pilot results may come from an older config.
        """
        if self.enforce_rules:
            if not self.owner:
                raise RuntimeError("Set MVT_OWNER before running real phases")
            _, dirty = git_state()
            if dirty:
                raise RuntimeError("Uncommitted changes: real phases must run from a clean tree")
        for result in iter_results(self.root, self.cell_id):
            if result.phase not in REAL_PHASES:
                continue
            expected = self.cfgs[result.arch].hash() if result.arch in self.cfgs else None
            if expected and result.config_hash != expected:
                raise RuntimeError(
                    f"{self.cell_id} has results with config hash {result.config_hash}, "
                    f"but the current config hash is {expected}. Move the old results aside "
                    "before re-running this cell."
                )

    def preflight(self) -> None:
        """Fail fast, before any trial: load the task data once and check the Mamba kernels.

        Errors here are setup problems, so they must not be recorded as failed trials.
        """
        cfg = self.cfgs[self.archs[0]]
        load_task_data(cfg)
        if "mamba" in self.cfgs:
            mamba_impl_for(self.cfgs["mamba"], resolve_device(cfg.base.runtime.device))

    # ------------------------------------------------------------ public
    def run(self, phase: str = "all", limit: int | None = None, retry_failed: bool = False) -> int:
        """Run (or resume) the requested phases. Returns the number of trials executed."""
        self.check_rules()
        self.preflight()
        self.limit, self.retry_failed, self.n_run = limit, retry_failed, 0
        for name in PHASE_GROUPS[phase]:
            for request in self._requests(name):
                if self.limit is not None and self.n_run >= self.limit:
                    return self.n_run
                self._execute(request)
        return self.n_run

    # ------------------------------------------------------------ planning
    def _requests(self, phase: str) -> Iterator[TrialRequest]:
        if phase == "sobol":
            for i, u in enumerate(self.sobol_table):
                hp = self.space.from_unit(u)
                for arch in self.archs:
                    yield self._request(arch, "sobol", i, hp, derive_seed(self.cell_id, "sobol", i))
        elif phase == "tpe":
            for k in range(self.counts.tpe):
                for arch in self.archs:
                    if self._done(arch, "tpe", k):
                        continue
                    hp = tpe.suggest(
                        self.space,
                        self._history(arch, k),
                        seed=derive_seed(self.cell_id, arch, "tpe", k),
                    )
                    yield self._request(arch, "tpe", k, hp, derive_seed(self.cell_id, "tpe", k))
        else:  # best / default replicates, seeds shared between the two phases
            for r in range(self.counts.replicates):
                for arch in self.archs:
                    hp = self.best_hparams(arch) if phase == "best" else self.space.defaults()
                    seed = derive_seed(self.cell_id, "replicate", r)
                    yield self._request(arch, phase, r, hp, seed)

    def _request(self, arch: str, phase: str, index: int, hp: HParams, seed: int) -> TrialRequest:
        return TrialRequest(
            task=self.task,
            fraction=self.fraction,
            scale=self.scale,
            arch=arch,
            phase=phase,  # type: ignore[arg-type]
            index=index,
            sobol_index=index if phase == "sobol" else None,
            hparams=hp,
            seed=seed,
            evaluate_test=phase in ("best", "default"),
        )

    def _load(self, arch: str, phase: str, index: int) -> TrialResult | None:
        path = result_path(self.root, self.cell_id, arch, phase, index)
        return read_result(path) if path.exists() else None

    def _done(self, arch: str, phase: str, index: int) -> bool:
        result = self._load(arch, phase, index)
        return result is not None and not (self.retry_failed and result.status == "failed")

    def _completed(self, arch: str, phase: str, n: int) -> list[TrialResult]:
        results = [self._load(arch, phase, i) for i in range(n)]
        missing = [i for i, r in enumerate(results) if r is None or r.status == "failed"]
        if missing:
            raise RuntimeError(f"{self.cell_id}/{arch}/{phase} is incomplete: indices {missing}")
        return results  # type: ignore[return-value]

    def _history(self, arch: str, k: int) -> list[tuple[HParams, float]]:
        """All Sobol results plus TPE 0..k-1, as (hparams, floored score)."""
        results = self._completed(arch, "sobol", self.counts.sobol)
        results += self._completed(arch, "tpe", k)
        return [(self.space.from_dict(r.hparams), _score(r)) for r in results]

    def best_hparams(self, arch: str) -> HParams:
        """lambda*: highest floored score over sobol + tpe; ties -> earlier phase, lower index."""
        results = self._completed(arch, "sobol", self.counts.sobol)
        results += self._completed(arch, "tpe", self.counts.tpe)
        best = max(results, key=lambda r: (_score(r), -PHASE_ORDER[r.phase], -r.index))
        return self.space.from_dict(best.hparams)

    # ------------------------------------------------------------ execution
    def _execute(self, request: TrialRequest) -> None:
        arch, phase, index = request.arch, request.phase, request.index
        if self._done(arch, phase, index):
            return
        cfg = self.cfgs[arch]
        started = now_iso()
        result = None
        for attempt in (1, 2):  # infrastructure failures are retried once
            try:
                result = run_trial(request, cfg, self.owner)
                break
            except KeyboardInterrupt:
                raise
            except Exception:  # noqa: BLE001
                error = traceback.format_exc()
                log.error(
                    "%s %s/%s/%d attempt %d failed:\n%s",
                    self.cell_id,
                    arch,
                    phase,
                    index,
                    attempt,
                    error,
                )
        if result is None:
            result = failed_result(request, cfg, self.owner, error, started)
        write_result(result, self.root, force=self.retry_failed)
        self.n_run += 1
        log.info(
            "%s %-11s %-7s %03d  %-8s val=%s  %.1fs",
            self.cell_id,
            arch,
            phase,
            index,
            result.status,
            _fmt(result.val_score),
            result.wall_time_s or 0.0,
        )


def _score(result: TrialResult) -> float:
    return floored_score(result.status, result.val_score, result.floor)  # type: ignore[arg-type]


def _fmt(x: float | None) -> str:
    return "None" if x is None else f"{x:.4f}"
