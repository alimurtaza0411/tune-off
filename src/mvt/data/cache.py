"""Fingerprinted on-disk cache, so each trial loads processed data in milliseconds."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def fingerprint(key_parts: Any) -> str:
    canonical = json.dumps(key_parts, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def load_or_build(key_parts: Any, build_fn: Callable[[], T], cache_dir: str | Path) -> T:
    """Return the cached object for ``key_parts``, building and saving it on a miss."""
    path = Path(cache_dir) / f"{fingerprint(key_parts)}.pkl"
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    obj = build_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)
    return obj
