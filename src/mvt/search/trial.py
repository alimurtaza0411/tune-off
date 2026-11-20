"""One trial: resolved config -> cached task data -> model -> train -> ``TrialResult``."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

import torch

from mvt import __version__
from mvt.core.config import ResolvedConfig
from mvt.core.env import git_state, gpu_name, mamba_ssm_version, optuna_version, resolve_device
from mvt.core.results import TrialResult
from mvt.core.seeding import seed_everything
from mvt.core.types import TaskData, TrialRequest
from mvt.data.cache import load_or_build
from mvt.models import build_model
from mvt.models.mamba import resolve_impl
from mvt.tasks import Task, get_task
from mvt.training.trainer import train_and_evaluate

_DATA_MEMO: dict[tuple, TaskData] = {}  # avoid re-reading the cache for every trial in a process


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_task_data(cfg: ResolvedConfig) -> tuple[Task, TaskData]:
    task = get_task(cfg.task.name, cfg.task)
    task_json = cfg.task.model_dump_json(exclude={"fp32_cells"})  # precision never changes data
    key = (task_json, cfg.fraction, task.PROCESSING_VERSION)
    if key not in _DATA_MEMO:
        if task.cacheable:
            parts = ["taskdata", *key]
            build = lambda: task.build(cfg.fraction)  # noqa: E731
            _DATA_MEMO[key] = load_or_build(parts, build, cfg.base.runtime.data_cache_dir)
        else:
            _DATA_MEMO[key] = task.build(cfg.fraction)
    return task, _DATA_MEMO[key]


def mamba_impl_for(cfg: ResolvedConfig, device: torch.device) -> str:
    """The Mamba mixer implementation this trial will use (raises on a GPU without kernels)."""
    return resolve_impl(cfg.model.impl, on_gpu=device.type == "cuda")  # type: ignore[union-attr]


def model_summary(cfg: ResolvedConfig, impl: str | None) -> dict:
    s, m = cfg.scale, cfg.model
    return {
        "d_model": s.d_model,
        "n_layers": s.n_layers,
        "n_heads": s.n_heads,
        "mlp_hidden": s.mlp_hidden,
        "d_state": getattr(m, "d_state", None),
        "expand": getattr(m, "expand", None),
        "d_conv": getattr(m, "d_conv", None),
        "impl": impl,
    }


def run_trial(request: TrialRequest, cfg: ResolvedConfig, owner: str) -> TrialResult:
    """Run one trial. Divergence is a normal result; infrastructure errors raise."""
    started, t0 = now_iso(), time.perf_counter()
    device = resolve_device(cfg.base.runtime.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    task, data = load_task_data(cfg)
    seed_everything(request.seed)

    model_cfg = cfg.model
    impl = None
    if request.arch == "mamba":
        impl = mamba_impl_for(cfg, device)
        model_cfg = model_cfg.model_copy(update={"impl": impl})
    model = build_model(request.arch, cfg.scale, model_cfg, data.spec, request.hparams.dropout)
    model.to(device)

    epochs = cfg.task.epochs_for(request.fraction)
    out = train_and_evaluate(
        model=model,
        data=data,
        task=task,
        hparams=request.hparams,
        epochs=epochs,
        seed=request.seed,
        device=device,
        precision=cfg.precision,
        optim_cfg=cfg.base.optimizer,
        final_lr_ratio=cfg.base.schedule.final_lr_ratio,
        n_evals=cfg.experiment.n_evals,
        evaluate_test=request.evaluate_test,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    counts = model.param_counts()
    commit, dirty = git_state()
    use_amp = device.type == "cuda" and cfg.precision == "fp16"
    return TrialResult(
        run_id=str(uuid.uuid4()),
        cell_id=request.cell_id,
        task=request.task,
        fraction=request.fraction,
        scale=request.scale,
        arch=request.arch,
        phase=request.phase,
        index=request.index,
        sobol_index=request.sobol_index,
        seed=request.seed,
        hparams=request.hparams.to_dict(),
        model=model_summary(cfg, impl),
        status=out.status,
        val_score=out.val_score,
        val_metric=out.val_metric,
        test_score=out.test_score,
        test_metric=out.test_metric,
        floor=data.spec.floor,
        epsilon=data.spec.epsilon,
        val_curve=out.val_curve,
        train_loss_curve=out.train_loss_curve,
        epochs=epochs,
        best_step=out.best_step,
        total_steps=out.total_steps,
        tokens_seen=out.tokens_seen,
        diverged_at_step=out.diverged_at_step,
        wall_time_s=time.perf_counter() - t0,
        train_time_s=out.train_time_s,
        peak_mem_mb=peak,
        n_params_total=counts["total"],
        n_params_non_embedding=counts["non_embedding"],
        precision="fp16" if use_amp else "fp32",
        device=device.type,
        gpu_name=gpu_name(device),
        mamba_impl=impl,
        git_commit=commit,
        git_dirty=dirty,
        code_version=__version__,
        config_hash=cfg.hash(),
        torch_version=torch.__version__,
        mamba_ssm_version=mamba_ssm_version(),
        optuna_version=optuna_version(),
        owner=owner,
        started_at=started,
        finished_at=now_iso(),
    )


def failed_result(
    request: TrialRequest, cfg: ResolvedConfig, owner: str, error: str, started: str
) -> TrialResult:
    """Record an infrastructure failure (excluded from analysis; must be re-run)."""
    commit, dirty = git_state()
    return TrialResult(
        run_id=str(uuid.uuid4()),
        cell_id=request.cell_id,
        task=request.task,
        fraction=request.fraction,
        scale=request.scale,
        arch=request.arch,
        phase=request.phase,
        index=request.index,
        sobol_index=request.sobol_index,
        seed=request.seed,
        hparams=request.hparams.to_dict(),
        model=model_summary(cfg, None),
        status="failed",
        git_commit=commit,
        git_dirty=dirty,
        code_version=__version__,
        config_hash=cfg.hash(),
        torch_version=torch.__version__,
        mamba_ssm_version=mamba_ssm_version(),
        optuna_version=optuna_version(),
        owner=owner,
        started_at=started,
        finished_at=now_iso(),
        error=error,
    )
