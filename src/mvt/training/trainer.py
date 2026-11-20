"""The training loop (README §Training recipe).

Runs exactly ``total_steps`` steps, evaluates on validation at 20 evenly spaced points and
keeps the best score. A non-finite training loss means the run DIVERGED: training stops and the
run is scored at the floor. That is a result, not an error, so no exception is raised.

Speed note: at this model size the per-step Python overhead dominates, so nothing in the step
loop calls ``.item()``. The divergence flag lives on the device and is read at eval points.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from mvt.core.config import OptimizerConfig
from mvt.core.types import Example, HParams, TaskData
from mvt.data.batching import (
    IGNORE_INDEX,
    iterate_eval_batches,
    iterate_train_batches,
    plan_total_steps,
)
from mvt.models.common import SequenceModel
from mvt.tasks.base import Task
from mvt.training.optim import make_optimizer, make_scheduler


@dataclass
class TrainOutput:
    status: str  # "ok" | "diverged"
    total_steps: int
    val_score: float | None = None
    val_metric: float | None = None
    test_score: float | None = None
    test_metric: float | None = None
    best_step: int | None = None
    diverged_at_step: int | None = None
    tokens_seen: int = 0
    train_time_s: float = 0.0
    val_curve: list[float | None] = field(default_factory=list)
    train_loss_curve: list[float | None] = field(default_factory=list)


def eval_points(total_steps: int, n_evals: int) -> list[int]:
    """Steps ceil(k * total / n), k = 1..n (fewer if total < n)."""
    return sorted({math.ceil(k * total_steps / n_evals) for k in range(1, n_evals + 1)})


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, head: str) -> torch.Tensor:
    if head == "seq_cls":
        return F.cross_entropy(logits.float(), labels)
    return F.cross_entropy(
        logits.float().flatten(0, 1), labels.flatten(), ignore_index=IGNORE_INDEX
    )


@torch.no_grad()
def evaluate(
    model: SequenceModel,
    examples: Sequence[Example],
    task: Task,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """Return ``(score, raw_metric)`` on a split via ``task.compute_metric``."""
    head = model.spec.head
    was_training = model.training
    model.eval()
    preds: list = []
    refs: list = []
    for batch in iterate_eval_batches(examples, head, batch_size):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(ids, mask)
        if head == "seq_cls":
            preds += logits.argmax(-1).tolist()
            refs += batch["labels"].tolist()
        elif head == "tok_cls":
            pred_ids = logits.argmax(-1).cpu()
            lengths = batch["attention_mask"].sum(1).tolist()
            for b, n in enumerate(lengths):
                preds.append(pred_ids[b, :n].tolist())
                refs.append(batch["labels"][b, :n].tolist())
        else:
            labels = batch["labels"].to(device)
            nll = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="sum")
            preds.append((float(nll), labels.numel()))
    model.train(was_training)
    return task.compute_metric(preds, refs)


def _finite_or_none(x: float) -> float | None:
    return x if math.isfinite(x) else None


def train_and_evaluate(
    model: SequenceModel,
    data: TaskData,
    task: Task,
    hparams: HParams,
    epochs: int,
    seed: int,
    device: torch.device,
    precision: str,
    optim_cfg: OptimizerConfig,
    final_lr_ratio: float,
    n_evals: int = 20,
    evaluate_test: bool = False,
) -> TrainOutput:
    head = data.spec.head
    use_amp = device.type == "cuda" and precision == "fp16"
    # Short sentences evaluate fast in big batches; LM batches are capped by the vocab softmax.
    eval_bs = hparams.batch_size if head == "lm" else max(hparams.batch_size, 256)
    total = plan_total_steps(epochs, len(data.train), hparams.batch_size)
    evals = eval_points(total, n_evals)
    eval_set = set(evals)

    optimizer = make_optimizer(
        model, hparams.learning_rate, hparams.weight_decay, optim_cfg, device
    )
    scheduler = make_scheduler(optimizer, total, hparams.warmup_fraction, final_lr_ratio)
    scaler = make_grad_scaler(use_amp)

    out = TrainOutput(status="ok", total_steps=total)
    best_score, best_state = -math.inf, None
    first_bad = torch.full((), -1, dtype=torch.long, device=device)  # first non-finite step
    loss_sum = torch.zeros((), device=device)
    n_losses = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    model.train()
    step = 0
    with warnings.catch_warnings():
        # GradScaler may skip the first steps; LambdaLR then warns about call order. Harmless.
        warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()`")
        batches = iterate_train_batches(data.train, head, hparams.batch_size, epochs, seed)
        for batch in batches:
            step += 1
            out.tokens_seen += batch["n_tokens"]
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(ids, mask)
            loss = compute_loss(logits, labels, head)

            bad_now = (first_bad < 0) & ~torch.isfinite(loss)
            first_bad = torch.where(bad_now, torch.full_like(first_bad, step), first_bad)
            loss_sum += loss.detach()
            n_losses += 1

            scaler.scale(loss).backward()
            if hparams.grad_clip > 0:
                scaler.unscale_(optimizer)  # unscale BEFORE clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if step not in eval_set:
                continue
            bad_step = int(first_bad.item())
            if bad_step >= 0:
                out.status, out.diverged_at_step = "diverged", bad_step
                break
            out.train_loss_curve.append(_finite_or_none(float(loss_sum) / n_losses))
            loss_sum.zero_()
            n_losses = 0
            score, metric = evaluate(model, data.val, task, eval_bs, device, use_amp)
            out.val_curve.append(_finite_or_none(score))
            if math.isfinite(score) and score > best_score:
                best_score, out.best_step = score, step
                out.val_score, out.val_metric = score, metric
                if evaluate_test:  # keep the best checkpoint only when it is needed
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }

    if device.type == "cuda":
        torch.cuda.synchronize()
    out.train_time_s = time.perf_counter() - start

    if out.status == "diverged":
        out.val_score = out.val_metric = None
        missing = len(evals) - len(out.val_curve)
        out.val_curve += [None] * missing
        out.train_loss_curve += [None] * missing
        return out
    if step != total:
        raise RuntimeError(f"Ran {step} steps but planned {total}")

    if evaluate_test:
        if best_state is not None:
            model.load_state_dict(best_state)
        out.test_score, out.test_metric = evaluate(model, data.test, task, eval_bs, device, use_amp)
    return out
