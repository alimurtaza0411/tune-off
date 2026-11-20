"""GPU checks (run on a Colab T4 with ``pytest -m gpu``).

These tests FAIL, never skip, when the GPU or the Mamba kernels are missing: a skipped GPU test
would hide exactly the problem it exists to catch.
"""

from __future__ import annotations

import pytest
import torch

from mvt.core.config import load_model_config
from mvt.core.env import mamba_ssm_import_error, mamba_ssm_version
from mvt.models.mamba import BiMixer, make_mixer

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _require_gpu_and_kernels():
    assert torch.cuda.is_available(), "pytest -m gpu needs a CUDA GPU"
    assert mamba_ssm_version(), f"mamba_ssm does not import: {mamba_ssm_import_error()}"


@pytest.mark.parametrize("bidirectional", [False, True])
def test_cuda_and_ref_mixers_agree(bidirectional):
    cfg = load_model_config("mamba")
    torch.manual_seed(0)
    cuda = make_mixer(64, cfg, "cuda").cuda()
    ref = make_mixer(64, cfg, "ref").cuda()
    ref.load_state_dict(cuda.state_dict())  # identical parameter names and shapes
    if bidirectional:
        cuda, ref = BiMixer(cuda), BiMixer(ref)
    x = torch.randn(3, 20, 64, device="cuda")
    lengths = torch.tensor([20, 13, 5], device="cuda")
    with torch.no_grad():
        a = cuda(x, lengths) if bidirectional else cuda(x)
        b = ref(x, lengths) if bidirectional else ref(x)
    torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("scale", ["2m", "8m"])
def test_debug_run_on_cuda(scale):
    from mvt.core.config import load_resolved_config
    from mvt.core.types import TrialRequest
    from mvt.search.space import SearchSpace
    from mvt.search.trial import run_trial

    cfg = load_resolved_config("debug_tag", 1.0, scale, "mamba")
    space = SearchSpace(cfg.search_space, cfg.task.batch_sizes)
    req = TrialRequest(
        "debug_tag", 1.0, scale, "mamba", "smoke", 0, None, space.defaults(), 1, False
    )
    result = run_trial(req, cfg, "gpu-test")
    assert result.mamba_impl == "cuda" and result.status == "ok"
    assert result.train_time_s > 0 and result.peak_mem_mb > 0
