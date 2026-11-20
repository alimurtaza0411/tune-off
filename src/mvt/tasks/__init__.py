"""Tasks register themselves with ``@register_task("name")`` in their own file.

Modules in this package are discovered automatically (files starting with ``_`` are skipped).
"""

from __future__ import annotations

from mvt.core.config import TaskConfig
from mvt.core.registry import TASKS, discover
from mvt.tasks.base import BaseTask, Task

__all__ = ["BaseTask", "Task", "get_task"]


def get_task(name: str, config: TaskConfig) -> Task:
    discover("mvt.tasks")
    if name not in TASKS:
        raise NotImplementedError(
            f"Task '{name}' is not implemented yet — see tasks/_template.py "
            f'and README §"Adding a task"'
        )
    return TASKS.get(name)(config)
