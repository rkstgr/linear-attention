"""Central config shape for a training run.

A `RunConfig` is the single, plain-data contract between the orchestration
layer (the executor, which hashes it for content-addressing) and a single
training run (`linattn.train.fit`). It composes a model spec, a task spec, a
train spec, and a seed. Everything here is a frozen dataclass so the executor
can normalize and digest it.

The task spec is any `TaskConfig` (see `linattn.tasks.base`); it is
discriminated by its `name` field and rebuilt in-worker via `build_task`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from linattn.tasks.base import TaskConfig


@dataclass(frozen=True)
class ModelConfig:
    """Everything needed to build an LMModel from the mixer/ffn registries.

    `vocab_size` is intentionally duplicated with the task config: the model's
    embedding/lm_head must match the task vocabulary, and callers keep the two
    consistent (see `linattn.runner`).
    """

    mixer: str
    vocab_size: int
    dim: int
    n_heads: int
    n_layers: int
    mlp_mult: int
    ffn: str = "swiglu"
    mixer_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainConfig:
    """Training-loop knobs. Dataset sizes live on the task config, not here."""

    batch_size: int
    eval_batch_size: int
    max_epochs: int
    learning_rate: float
    target_acc: float
    patience_epochs: int


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    task: "TaskConfig"
    train: TrainConfig
    seed: int


# Debug-scale default. Full experiments (e.g. level1) carry their own
# TrainConfig rather than overriding this.
DEFAULT_TRAIN = TrainConfig(
    batch_size=64,
    eval_batch_size=64,
    max_epochs=16,
    learning_rate=1e-3,
    target_acc=0.99,
    patience_epochs=5,
)
