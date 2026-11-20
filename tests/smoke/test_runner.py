"""Trainer and runner behaviour on the tiny debug tasks (CPU, reference Mamba)."""

from __future__ import annotations

import dataclasses

import pytest

from mvt.core.config import PhaseCounts, load_resolved_config
from mvt.core.results import iter_results, results_to_frame, validate_frame, write_result
from mvt.core.types import TrialRequest
from mvt.search import runner as runner_module
from mvt.search.runner import CellRunner
from mvt.search.space import SearchSpace
from mvt.search.trial import run_trial

SMALL = PhaseCounts(sobol=2, tpe=1, replicates=1)


def request(arch: str, task: str = "debug_cls", **hp_overrides) -> tuple:
    cfg = load_resolved_config(task, 1.0, "tiny", arch)
    space = SearchSpace(cfg.search_space, cfg.task.batch_sizes)
    hp = dataclasses.replace(space.defaults(), **hp_overrides)
    req = TrialRequest(task, 1.0, "tiny", arch, "smoke", 0, None, hp, 123, False)
    return req, cfg


@pytest.mark.parametrize("arch", ["transformer", "mamba"])
def test_training_learns_debug_task(arch):
    req, cfg = request(arch, "debug_lm", learning_rate=3e-3)
    result = run_trial(req, cfg, "test")
    assert result.status == "ok"
    losses = result.train_loss_curve
    assert losses[-1] < losses[0]
    assert len(result.val_curve) == 20 and result.best_step <= result.total_steps
    assert result.n_params_non_embedding > 0 and result.config_hash == cfg.hash()


def test_divergence_is_a_result_not_an_error():
    req, cfg = request("transformer", learning_rate=1e3, grad_clip=0.0)
    result = run_trial(req, cfg, "test")
    assert result.status == "diverged"
    assert result.val_score is None and result.diverged_at_step is not None
    assert len(result.val_curve) == 20


def _runner(root, **kw) -> CellRunner:
    return CellRunner(
        "debug_cls",
        1.0,
        "tiny",
        ["transformer", "mamba"],
        root,
        "test",
        counts=SMALL,
        enforce_rules=False,
        **kw,
    )


def test_resume_pairing_and_validation(tmp_path):
    root = tmp_path / "runs"
    assert _runner(root).run(limit=3) == 3  # "interrupted" after 3 trials
    assert _runner(root).run() == 2 * (2 + 1 + 1 + 1) - 3  # resumes, no duplicates
    assert _runner(root).run() == 0
    df = results_to_frame(iter_results(root))
    assert validate_frame(df, SMALL.expected(), ["transformer", "mamba"]) == []
    sob = df[df["phase"] == "sobol"]
    for _, g in sob.groupby("sobol_index"):  # identical hparams and seed across architectures
        assert g["hp_learning_rate"].nunique() == 1 and g["seed"].nunique() == 1
    best = df[df["phase"] == "best"]
    rep = df[df["phase"] == "default"]
    assert set(best["seed"]) == set(rep["seed"])  # replicate seeds shared by best/default


def test_tpe_history_has_all_sobol_trials(tmp_path, monkeypatch):
    seen = []
    real = runner_module.tpe.suggest

    def spy(space, history, seed):
        seen.append(len(history))
        return real(space, history, seed)

    monkeypatch.setattr(runner_module.tpe, "suggest", spy)
    _runner(tmp_path / "runs").run("sobol")
    _runner(tmp_path / "runs").run("tpe")
    assert seen == [SMALL.sobol, SMALL.sobol]  # one TPE trial per architecture


def test_old_pilot_results_do_not_block_the_cell(tmp_path):
    """After the pilot changes epochs or the LR range, its old results must not block runs."""
    req, cfg = request("transformer")
    req = dataclasses.replace(req, phase="pilot")
    old = run_trial(req, cfg, "test").model_copy(update={"config_hash": "from-old-config"})
    write_result(old, tmp_path / "runs")
    _runner(tmp_path / "runs").check_rules()  # must not raise


def test_real_phases_refuse_dirty_tree_or_missing_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "git_state", lambda: ("abc", True))
    with pytest.raises(RuntimeError, match="clean tree"):
        CellRunner("debug_cls", 1.0, "tiny", ["mamba"], tmp_path, "me", counts=SMALL).run()
    monkeypatch.setattr(runner_module, "git_state", lambda: ("abc", False))
    with pytest.raises(RuntimeError, match="MVT_OWNER"):
        CellRunner("debug_cls", 1.0, "tiny", ["mamba"], tmp_path, "", counts=SMALL).run()
