from __future__ import annotations

import numpy as np

from mvt.core.config import load_search_space, load_task_config
from mvt.core.types import HParams
from mvt.search import tpe
from mvt.search.sobol import check_table, load_table
from mvt.search.space import SearchSpace


def space() -> SearchSpace:
    return SearchSpace(load_search_space(), load_task_config("sst2").batch_sizes)


def test_sobol_table_regenerates_identically():
    assert check_table(load_search_space())


def test_sobol_balance():
    u = load_table(load_search_space())
    assert u.shape == (32, 6)
    for d in range(6):  # exactly one point per 1/32 interval in every dimension
        assert sorted(np.floor(u[:, d] * 32).astype(int)) == list(range(32))
    s = space()
    hps = [s.from_unit(row) for row in u]
    for name in ("batch_size_index", "grad_clip"):
        _, counts = np.unique([getattr(h, name) for h in hps], return_counts=True)
        assert list(counts) == [8, 8, 8, 8]
    assert int(((u[:, 0] >= 0.25) & (u[:, 0] < 0.75)).sum()) == 16  # LR sub-range


def test_from_unit_bounds_and_defaults():
    s = space()
    assert s.contains(s.defaults())
    assert s.defaults().batch_size == 64
    for u in np.random.default_rng(0).random((200, 6)):
        assert s.contains(s.from_unit(u))
    lo = s.from_unit([0.0] * 6)
    assert np.isclose(lo.learning_rate, 1e-5) and lo.batch_size == 32 and lo.grad_clip == 0.25
    hi = s.from_unit([0.999999] * 6)
    assert hi.batch_size == 256 and hi.grad_clip == 0.0


def test_tpe_is_deterministic_and_in_bounds():
    s = space()
    u = load_table(load_search_space())
    history = [
        (s.from_unit(row), float(-abs(np.log10(s.from_unit(row).learning_rate) + 3))) for row in u
    ]
    a = tpe.suggest(s, history, seed=7)
    b = tpe.suggest(s, history, seed=7)
    assert a == b and isinstance(a, HParams) and s.contains(a)
