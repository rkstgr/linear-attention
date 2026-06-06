"""Task abstraction and registry.

A `Task` is the data side of a run: it knows its vocabulary, generates
supervised splits of `(tokens, targets, mask)`, and can describe one example
for debugging. Tasks are selected by name through `TASKS`, mirroring the mixer
registry in `linattn.models.factory`.

Configs are plain frozen dataclasses carrying a `name` discriminator;
`build_task(cfg)` dispatches on `cfg.name` and reconstructs the live task
in-worker. This module is deliberately jax-free so the registry can be imported
and unit-tested without the heavy numeric stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from jax import Array

    Split = tuple[Array, Array, Array]  # tokens, targets, mask; each (n, T)
else:
    Split = tuple  # runtime alias; kept jax-free so the registry imports light


@runtime_checkable
class TaskConfig(Protocol):
    """Plain-data task spec. The `name` field selects the task in `TASKS`."""

    name: str


class Task(Protocol):
    vocab_size: int
    sources: tuple[str, ...]  # source files that feed the executor's digest

    def make_split(self, key, n: int) -> "Split": ...

    def describe(self, model, key) -> None: ...


TASKS: dict[str, Callable[[Any], "Task"]] = {}


def register_task(name: str) -> Callable[[Callable[[Any], "Task"]], Callable[[Any], "Task"]]:
    """Register a task factory (usually the Task class) under `name`."""

    def decorator(factory: Callable[[Any], "Task"]) -> Callable[[Any], "Task"]:
        TASKS[name] = factory
        return factory

    return decorator


def build_task(cfg: "TaskConfig") -> "Task":
    """Reconstruct the live task for a config, dispatching on `cfg.name`."""
    try:
        factory = TASKS[cfg.name]
    except KeyError as exc:
        raise ValueError(
            f"unknown task {cfg.name!r}; choices: {sorted(TASKS)}"
        ) from exc
    return factory(cfg)
