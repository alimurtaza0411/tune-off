"""Environment facts recorded in every result: git state, versions, device, Mamba kernel."""

from __future__ import annotations

import importlib
import subprocess
from functools import lru_cache

import optuna
import torch

from mvt import __version__
from mvt.core.config import REPO_ROOT

# Paths whose changes affect results. Uncommitted or untracked files here make the tree "dirty".
CODE_PATHS = ("src", "configs", "pyproject.toml")


def git_state() -> tuple[str, bool]:
    """Return (commit, dirty). Dirty = any modified OR untracked file under ``CODE_PATHS``."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", *CODE_PATHS],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return commit, bool(status)
    except (OSError, subprocess.CalledProcessError):  # not a git repo, or no commit yet
        return "unknown", True


@lru_cache(maxsize=1)
def _import_mamba_ssm() -> tuple[str | None, str | None]:
    """(version, error). Imported lazily so CPU environments never need mamba_ssm."""
    try:
        module = importlib.import_module("mamba_ssm")
    except Exception as exc:  # noqa: BLE001 - report any import failure, never hide it
        return None, f"{type(exc).__name__}: {exc}"
    return getattr(module, "__version__", "unknown"), None


def mamba_ssm_version() -> str | None:
    return _import_mamba_ssm()[0]


def mamba_ssm_import_error() -> str | None:
    return _import_mamba_ssm()[1]


def optuna_version() -> str:
    return optuna.__version__


def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def gpu_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"


def env_report() -> dict[str, object]:
    device = resolve_device()
    commit, dirty = git_state()
    if device.type != "cuda":
        impl = "ref (no GPU)"
    elif mamba_ssm_version():
        impl = "cuda"
    else:
        impl = f"ERROR: mamba_ssm does not import ({mamba_ssm_import_error()})"
    return {
        "mvt_version": __version__,
        "torch_version": torch.__version__,
        "optuna_version": optuna_version(),
        "mamba_ssm_version": mamba_ssm_version(),
        "device": str(device),
        "gpu_name": gpu_name(device),
        "mamba_impl": impl,
        "git_commit": commit,
        "git_dirty": dirty,
    }
