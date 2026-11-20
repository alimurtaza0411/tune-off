"""Task contract, run for EVERY registered task (a new task is tested automatically).

Real tasks download data, so they are marked ``data``; debug tasks run everywhere.
"""

from __future__ import annotations

import pytest

from mvt.core.config import load_task_config
from mvt.core.registry import TASKS, discover
from mvt.data.batching import iterate_train_batches
from mvt.tasks import get_task

discover("mvt.tasks")
PARAMS = [
    pytest.param(n, marks=[] if n.startswith("debug_") else [pytest.mark.data])
    for n in TASKS.names()
]


def build(name: str, fraction: float):
    return get_task(name, load_task_config(name)).build(fraction)


@pytest.mark.parametrize("name", PARAMS)
def test_spec_is_valid(name):
    spec = build(name, 1.0).spec
    assert spec.causal == (spec.head == "lm")
    assert spec.pad_id == 0 and spec.epsilon > 0
    assert spec.floor is not None


@pytest.mark.parametrize("name", PARAMS)
def test_build_is_deterministic(name):
    a, b = build(name, 0.25), build(name, 0.25)
    assert a.train == b.train and a.val == b.val and a.spec == b.spec


@pytest.mark.parametrize("name", PARAMS)
def test_fraction_rules(name):
    full, quarter = build(name, 1.0), build(name, 0.25)
    assert abs(len(quarter.train) - 0.25 * len(full.train)) <= max(1, 0.01 * len(full.train))
    full_set = {repr(e) for e in full.train}
    assert all(repr(e) in full_set for e in quarter.train)  # a subset of the pool
    assert quarter.val == full.val and quarter.test == full.test
    assert quarter.spec.vocab_size == full.spec.vocab_size  # vocab independent of fraction


@pytest.mark.parametrize("name", PARAMS)
def test_batches_follow_contract(name):
    data = build(name, 0.25)
    b = next(iterate_train_batches(data.train, data.spec.head, 8, 1, seed=0))
    ids, mask = b["input_ids"], b["attention_mask"]
    assert ids.shape == mask.shape and (ids[~mask] == 0).all()
    # real tokens form a prefix of each row
    assert (mask.int().diff(dim=1) <= 0).all()
    assert ids.max() < data.spec.vocab_size


@pytest.mark.parametrize("name", PARAMS)
def test_metric_is_max_on_perfect_predictions(name):
    task = get_task(name, load_task_config(name))
    data = task.build(1.0)
    head, val = data.spec.head, data.val
    if head == "seq_cls":
        refs = [e.label for e in val]
        score, _ = task.compute_metric(refs, refs)
        assert score == 1.0
    elif head == "tok_cls":
        refs = [e.labels for e in val]
        score, _ = task.compute_metric(refs, refs)
        assert score == 1.0
    else:
        score, ppl = task.compute_metric([(0.0, 100)], None)
        assert score == 0.0 and ppl == 1.0


def test_unimplemented_task_has_clear_message():
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        get_task("not_a_registered_task", load_task_config("debug_cls"))
