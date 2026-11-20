"""Environment facts recorded in every result: git state, versions, device, Mamba kernel."""

from __future__ import annotations

import importlib
import subprocess
from functools import lru_cache

import torch

from mvt import __version__
from mvt.core.config import REPO_ROOT


def git_state() -> tuple[str, bool]:
    """Return (commit, dirty). Untracked files do not count as dirty."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return commit, bool(status)
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


@lru_cache(maxsize=1)
def mamba_ssm_version() -> str | None:
    """Version of ``mamba_ssm`` if it imports, else None. Imported lazily on purpose."""
    try:
        module = importlib.import_module("mamba_ssm")
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return None
    return getattr(module, "__version__", "unknown")


def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def gpu_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"


def env_report() -> dict[str, object]:
    device = resolve_device()
    commit, dirty = git_state()
    cuda_ok = device.type == "cuda" and mamba_ssm_version() is not None
    return {
        "mvt_version": __version__,
        "torch_version": torch.__version__,
        "mamba_ssm_version": mamba_ssm_version(),
        "device": str(device),
        "gpu_name": gpu_name(device),
        "mamba_impl (auto)": "cuda" if cuda_ok else "ref",
        "git_commit": commit,
        "git_dirty": dirty,
    }
