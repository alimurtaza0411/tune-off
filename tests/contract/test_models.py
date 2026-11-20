"""Model contract: shapes, pad invariance, causality, gradients, determinism, exact sizes."""

from __future__ import annotations

import pytest
import torch

from mvt.core.config import load_model_config, load_scale
from mvt.core.types import ClsExample, TaskSpec
from mvt.data.batching import collate
from mvt.models import build_model

ARCHS = ["transformer", "mamba"]
SPECS = {
    "seq_cls": TaskSpec("t", "seq_cls", False, 3, 40, 0, 16, "accuracy", 1 / 3, 0.01),
    "tok_cls": TaskSpec("t", "tok_cls", False, 5, 40, 0, 16, "accuracy", 0.2, 0.01),
    "lm": TaskSpec("t", "lm", True, None, 40, 0, 16, "neg_log_ppl", -3.7, 0.05),
}


def make(arch: str, head: str, seed: int = 0):
    torch.manual_seed(seed)
    cfg = load_model_config(arch)
    if arch == "mamba":
        cfg = cfg.model_copy(update={"impl": "ref"})
    model = build_model(arch, load_scale("tiny", arch), cfg, SPECS[head], dropout=0.1)
    return model.eval()


def batch(lengths: list[int], extra_pad: int = 0, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    ex = [ClsExample(torch.randint(2, 40, (n,), generator=g).tolist(), 0) for n in lengths]
    b = collate(ex, "seq_cls")
    if extra_pad:
        pad = torch.zeros(len(lengths), extra_pad, dtype=torch.long)
        b["input_ids"] = torch.cat([b["input_ids"], pad], 1)
        b["attention_mask"] = torch.cat([b["attention_mask"], pad.bool()], 1)
    return b


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("head", ["seq_cls", "tok_cls", "lm"])
def test_output_shapes_and_finite_grads(arch, head):
    model = make(arch, head).train()
    b = batch([7, 7]) if head == "lm" else batch([7, 4])
    out = model(b["input_ids"], b["attention_mask"])
    expected = {"seq_cls": (2, 3), "tok_cls": (2, 7, 5), "lm": (2, 7, 40)}[head]
    assert tuple(out.shape) == expected
    out.float().pow(2).mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("head", ["seq_cls", "tok_cls"])
def test_pad_invariance(arch, head):
    """Adding extra padding must not change outputs at real positions."""
    model = make(arch, head)
    b, b_pad = batch([9, 5, 2]), batch([9, 5, 2], extra_pad=6)
    with torch.no_grad():
        out = model(b["input_ids"], b["attention_mask"])
        out_pad = model(b_pad["input_ids"], b_pad["attention_mask"])
    if head == "seq_cls":
        torch.testing.assert_close(out, out_pad, atol=1e-5, rtol=1e-5)
    else:
        for row, n in enumerate([9, 5, 2]):
            torch.testing.assert_close(out[row, :n], out_pad[row, :n], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("arch", ARCHS)
def test_lm_is_causal(arch):
    model = make(arch, "lm")
    b = batch([12])
    ids = b["input_ids"].clone()
    t = 6
    ids2 = ids.clone()
    ids2[0, t] = (ids[0, t] + 1) % 38 + 2
    with torch.no_grad():
        a = model(ids, b["attention_mask"])
        c = model(ids2, b["attention_mask"])
    torch.testing.assert_close(a[0, :t], c[0, :t], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(a[0, t:], c[0, t:])


@pytest.mark.parametrize("arch", ARCHS)
def test_seeded_determinism(arch):
    b = batch([6, 3])
    with torch.no_grad():
        a = make(arch, "tok_cls", seed=3)(b["input_ids"], b["attention_mask"])
        c = make(arch, "tok_cls", seed=3)(b["input_ids"], b["attention_mask"])
    torch.testing.assert_close(a, c)


@pytest.mark.parametrize("scale", ["2m", "8m"])
def test_exact_non_embedding_param_counts(scale):
    counts = {}
    for arch in ARCHS:
        for head in ("lm", "tok_cls"):  # causal and bidirectional share parameters
            cfg = load_model_config(arch)
            if arch == "mamba":
                cfg = cfg.model_copy(update={"impl": "ref"})
            sc = load_scale(scale, arch)
            model = build_model(arch, sc, cfg, SPECS[head], dropout=0.0)
            assert model.param_counts()["non_embedding"] == sc.expected_non_embedding
        counts[arch] = sc.expected_non_embedding
    t, m = counts["transformer"], counts["mamba"]
    assert abs(t - m) / max(t, m) < 0.03
