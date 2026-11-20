from __future__ import annotations

import math

import numpy as np
import torch

from mvt.core.types import ClsExample, LMChunk
from mvt.data.batching import collate, plan_epoch, plan_total_steps
from mvt.data.vocab import Vocab
from mvt.training.optim import lr_multiplier, param_groups


def test_plan_total_steps():
    assert plan_total_steps(3, 100, 32) == 12
    assert plan_total_steps(1, 64, 64) == 1


def test_bucketed_batches_cover_every_example_once():
    lengths = np.random.default_rng(0).integers(1, 50, size=1000)
    batches = plan_epoch(1000, 7, np.random.default_rng(1), lengths)
    flat = np.concatenate(batches)
    assert sorted(flat.tolist()) == list(range(1000))
    sizes = [len(b) for b in batches]
    assert len(batches) == math.ceil(1000 / 7)
    assert sizes.count(7) == len(batches) - 1  # only one smaller batch
    # length-bucketed: random batches would have a length spread near 40
    assert np.mean([np.ptp(lengths[b]) for b in batches]) < 5
    # deterministic for a fixed seed
    again = plan_epoch(1000, 7, np.random.default_rng(1), lengths)
    assert all((a == b).all() for a, b in zip(batches, again))


def test_collate_contract():
    ex = [ClsExample([5, 6, 7], 1), ClsExample([8], 0)]
    b = collate(ex, "seq_cls")
    assert b["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert b["attention_mask"].tolist() == [[True] * 3, [True, False, False]]
    assert b["labels"].tolist() == [1, 0] and b["n_tokens"] == 4
    lm = collate([LMChunk([1, 2], [2, 3])], "lm")
    assert lm["attention_mask"].all() and lm["labels"].tolist() == [[2, 3]]


def test_vocab():
    v = Vocab.build([["b", "a", "b"], ["c"]], extra_specials=["<eos>"])
    assert v.itos[:3] == ["<pad>", "<unk>", "<eos>"] and v.itos[3] == "b"
    assert v.encode(["b", "zzz"]) == [3, 1]


def test_weight_decay_groups():
    from mvt.models.mamba_ref import MambaMixerRef

    mixer = MambaMixerRef(16)
    decay, no_decay = param_groups(mixer, 0.1)
    decay_ids = {id(p) for p in decay["params"]}
    assert id(mixer.A_log) not in decay_ids and id(mixer.D) not in decay_ids
    assert id(mixer.in_proj.weight) in decay_ids and id(mixer.dt_proj.bias) not in decay_ids
    assert no_decay["weight_decay"] == 0.0


def test_schedule_shape():
    total = 100
    m = [lr_multiplier(s, total, 0.1, 0.1) for s in range(total)]
    assert m[0] == 0.1 and m[9] == 1.0  # linear warmup over 10 steps
    assert all(a >= b for a, b in zip(m[10:], m[11:]))  # cosine decays monotonically
    assert math.isclose(lr_multiplier(total, total, 0.1, 0.1), 0.1)
    assert lr_multiplier(0, total, 0.0, 0.1) == 1.0  # no warmup


def test_tied_lm_head_is_counted_once():
    from mvt.core.config import load_model_config, load_scale
    from mvt.core.types import TaskSpec
    from mvt.models import build_model

    spec = TaskSpec("x", "lm", True, None, 50, 0, 8, "m", -3.9, 0.05)
    model = build_model(
        "transformer",
        load_scale("tiny", "transformer"),
        load_model_config("transformer"),
        spec,
        0.0,
    )
    assert model.head.weight is model.embed.weight
    assert model.param_counts()["head"] == 0
    assert torch.equal(model.head.weight, model.embed.weight)
