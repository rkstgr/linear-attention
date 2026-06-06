"""Task abstraction and registry.

Importing a task module (e.g. ``linattn.tasks.mqar``) registers it in ``TASKS``.
This package root stays jax-free so the registry can be imported and tested
without the numeric stack; the task modules themselves pull jax as needed.
"""

from linattn.tasks.base import (
    TASKS,
    Split,
    Task,
    TaskConfig,
    build_task,
    register_task,
)

__all__ = [
    "TASKS",
    "Split",
    "Task",
    "TaskConfig",
    "build_task",
    "register_task",
]
