from __future__ import annotations

import pytest
import torch

torch.set_num_threads(1)  # tiny CPU models run fastest single-threaded


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    """Never write runs or caches into the repository during tests."""
    monkeypatch.setenv("MVT_RESULTS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("MVT_DATA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MVT_DEVICE", "cpu")
