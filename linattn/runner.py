"""Executor glue: turn a RunConfig into a cached training step.

`default_train(name, run)` builds an `ExecutorStep` whose `fn` is `train_run`.
The executor hashes the `RunConfig` (model + task + train + seed) plus the
declared source files for content-addressing; `train_run` reconstructs the live
task and model in-worker and calls `fit`, writing `metrics.json` from the
returned `TrainResult`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import optax

from linattn.config import ModelConfig, RunConfig
from linattn.executor import ExecutorStep, SourceSet, this_output_path
from linattn.models.factory import build_lm_model
from linattn.tasks.base import build_task
from linattn.train import fit


CORE_SOURCES = SourceSet(
    "core",
    (
        "linattn/runner.py",
        "linattn/train.py",
        "linattn/config.py",
        "linattn/utils.py",
        "linattn/models/backbone.py",
        "linattn/models/factory.py",
        "linattn/models/ffn.py",
    ),
)

MIXER_SOURCES = {
    "transformer": SourceSet("mixer:transformer", ("linattn/models/attention.py",)),
    "linear_attention": SourceSet(
        "mixer:linear_attention", ("linattn/models/linear_attention.py",)
    ),
    "deltanet": SourceSet("mixer:deltanet", ("linattn/models/deltanet.py",)),
    "gated_deltanet": SourceSet("mixer:gated_deltanet", ("linattn/models/deltanet.py",)),
    "titans": SourceSet("mixer:titans", ("linattn/models/titans.py",)),
}


@dataclass(frozen=True)
class TrainRunConfig:
    run: RunConfig
    output_path: str


def task_sources(task_cfg) -> SourceSet:
    """Source files declared by a task, for the step digest."""
    task = build_task(task_cfg)
    return SourceSet(f"task:{task_cfg.name}", tuple(task.sources))


def mixer_sources(model: ModelConfig) -> SourceSet:
    try:
        return MIXER_SOURCES[model.mixer]
    except KeyError as exc:
        raise ValueError(
            f"unknown mixer {model.mixer!r}; choices: {sorted(MIXER_SOURCES)}"
        ) from exc


def default_train(name: str, run: RunConfig) -> ExecutorStep:
    return ExecutorStep(
        name=f"checkpoints/{name}",
        fn=train_run,
        config=TrainRunConfig(run=run, output_path=this_output_path()),
        sources=(CORE_SOURCES, task_sources(run.task), mixer_sources(run.model)),
    )


def train_run(config: TrainRunConfig) -> None:
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    run = config.run

    task = build_task(run.task)
    k_model, k_train = jax.random.split(jax.random.PRNGKey(run.seed))
    model = build_lm_model(
        run.model.mixer,
        vocab_size=run.model.vocab_size,
        dim=run.model.dim,
        n_heads=run.model.n_heads,
        n_layers=run.model.n_layers,
        mlp_mult=run.model.mlp_mult,
        key=k_model,
        ffn=run.model.ffn,
        **run.model.mixer_kwargs,
    )
    opt = optax.adamw(run.train.learning_rate)
    result = fit(model, task, run.train, k_train, opt=opt)

    best = max((h["test_acc"] for h in result.history), default=0.0)
    metrics = {
        "best": best,
        "history": result.history,
        "stop_info": result.stop_info,
        "run": asdict(run),
    }
    (output_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
