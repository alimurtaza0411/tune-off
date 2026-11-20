"""Name -> class registries. Tasks and models register themselves in their own files."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Any


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Any] = {}

    def register(self, name: str) -> Callable[[Any], Any]:
        def decorator(obj: Any) -> Any:
            old = self._items.get(name)
            if old is not None and old.__qualname__ != obj.__qualname__:
                raise ValueError(f"{self.kind} '{name}' is already registered by {old}")
            self._items[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> Any:
        if name not in self._items:
            raise KeyError(f"Unknown {self.kind} '{name}'. Registered: {self.names()}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items


TASKS = Registry("task")
MODELS = Registry("model")
register_task = TASKS.register
register_model = MODELS.register


def discover(package_name: str) -> None:
    """Import every module of a package (skipping names that start with ``_``)."""
    package = importlib.import_module(package_name)
    for module in pkgutil.iter_modules(package.__path__):
        if not module.name.startswith("_"):
            importlib.import_module(f"{package_name}.{module.name}")
