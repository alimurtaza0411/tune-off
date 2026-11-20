from __future__ import annotations

import pandas as pd
import pytest

from mvt.core.config import load_resolved_config
from mvt.core.results import (
    TrialResult,
    floored_score,
    read_result,
    result_path,
    results_to_frame,
    validate_frame,
    write_result,
)
from mvt.core.seeding import derive_seed


def test_derive_seed_is_stable():
    assert derive_seed("sst2-025-8m", "sobol", 0) == derive_seed("sst2-025-8m", "sobol", 0)
    assert derive_seed("a", 1) != derive_seed("a", 2)
    assert derive_seed("x") == 0x2D711642  # sha256("x")[:8]; must never change


def test_config_hash_ignores_runtime_fields_and_arch(monkeypatch):
    t = load_resolved_config("debug_cls", 1.0, "tiny", "transformer")
    m = load_resolved_config("debug_cls", 1.0, "tiny", "mamba")
    assert t.hash() == m.hash()  # one hash per cell
    monkeypatch.setenv("MVT_RESULTS_ROOT", "/somewhere/else")
    monkeypatch.setenv("MVT_DATA_CACHE", "/another/place")
    assert load_resolved_config("debug_cls", 1.0, "tiny", "mamba").hash() == m.hash()
    assert load_resolved_config("debug_cls", 0.25, "tiny", "mamba").hash() != m.hash()


def make_result(**overrides) -> TrialResult:
    base = dict(
        run_id="r",
        cell_id="c-100-tiny",
        task="c",
        fraction=1.0,
        scale="tiny",
        arch="transformer",
        phase="sobol",
        index=0,
        sobol_index=0,
        seed=1,
        hparams={
            "learning_rate": 1e-3,
            "warmup_fraction": 0.05,
            "weight_decay": 0.1,
            "dropout": 0.1,
            "batch_size_index": 1,
            "batch_size": 16,
            "grad_clip": 1.0,
        },
        model={"d_model": 32},
        status="ok",
        val_score=0.7,
        floor=0.5,
        epsilon=0.01,
        git_commit="abc",
        git_dirty=False,
        code_version="0.1.0",
        config_hash="h",
        owner="me",
        started_at="t0",
        finished_at="t1",
    )
    base.update(overrides)
    return TrialResult(**base)


def test_result_round_trip_and_no_overwrite(tmp_path):
    r = make_result()
    path = write_result(r, tmp_path)
    assert path == result_path(tmp_path, "c-100-tiny", "transformer", "sobol", 0)
    assert read_result(path) == r
    with pytest.raises(FileExistsError):
        write_result(r, tmp_path)
    write_result(make_result(val_score=0.9), tmp_path, force=True)
    assert read_result(path).val_score == 0.9
    assert not list(tmp_path.rglob("*.tmp"))


def test_floored_score():
    assert floored_score("ok", 0.4, 0.5) == 0.5
    assert floored_score("ok", 0.7, 0.5) == 0.7
    assert floored_score("diverged", 0.9, 0.5) == 0.5
    assert floored_score("ok", None, 0.5) == 0.5


def _valid_frame() -> pd.DataFrame:
    rows = [
        make_result(arch=a, index=i, sobol_index=i)
        for a in ("transformer", "mamba")
        for i in range(2)
    ]
    return results_to_frame(rows)


def test_validate_accepts_valid_frame():
    assert validate_frame(_valid_frame(), {"sobol": 2}) == []


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda df: df.assign(config_hash=["a", "b", "a", "a"]), "config hashes"),
        (lambda df: df.assign(status=["failed", "ok", "ok", "ok"]), "failed"),
        (lambda df: pd.concat([df, df.iloc[:1]]), "duplicate"),
        (lambda df: df.iloc[:3], "expected 2"),
        (lambda df: df.assign(hp_dropout=[0.1, 0.1, 0.2, 0.1]), "different hparams"),
        (lambda df: df.assign(seed=[1, 1, 2, 1]), "different seeds"),
    ],
)
def test_validate_catches_each_violation(mutate, message):
    problems = validate_frame(mutate(_valid_frame()), {"sobol": 2})
    assert any(message in p for p in problems), problems
