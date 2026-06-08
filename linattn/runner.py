"""Executor glue: turn a RunConfig into a cached training step.

`default_train(name, run)` builds an `ExecutorStep` whose `fn` is `train_run`.
The executor hashes the `RunConfig` (model + task + train + seed) plus the
declared source files for content-addressing; `train_run` reconstructs the live
task and model in-worker and calls `fit`, writing `metrics.json` from the
returned `TrainResult`.
"""

from __future__ import annotations

import json
import os
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

    wandb_run, reporter = _make_reporter(run)
    try:
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
        result = fit(model, task, run.train, k_train, opt=opt, reporter=reporter)

        headline = _headline_from_history(result.history)
        metrics = {
            # `best` is the test_partial_accuracy at the best-val epoch (val path)
            # or simply the max over test_partial_accuracy (legacy two-way path).
            # Same scalar previous callers read off `best`.
            "best": headline["test_partial_accuracy"],
            "headline": headline,
            "history": result.history,
            "stop_info": result.stop_info,
            "run": asdict(run),
        }
        (output_path / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def _make_reporter(run: "RunConfig"):
    """Return (wandb_run | None, Reporter).

    Always includes StdoutReporter. Adds WandbReporter when WANDB_PROJECT is
    set in the environment — project/group come from env vars so enabling or
    disabling W&B does not affect the step content hash.
    """
    from linattn.train import MultiReporter, StdoutReporter, WandbReporter

    project = os.environ.get("WANDB_PROJECT")
    if not project:
        return None, StdoutReporter()

    try:
        import wandb
    except ModuleNotFoundError:
        return None, StdoutReporter()

    group = os.environ.get("WANDB_RUN_GROUP", run.model.mixer)
    name = f"{group}-s{run.seed}"

    wandb_run = wandb.init(
        name=name,
        group=group,
        config={
            "model": asdict(run.model),
            "task":  asdict(run.task),
            "train": asdict(run.train),
            "seed":  run.seed,
        },
    )
    return wandb_run, MultiReporter([StdoutReporter(), WandbReporter(wandb_run)])


def _headline_from_history(history: list[dict]) -> dict:
    """Pick the test metric at the epoch that wins on the selection signal.

    With val: ``argmax`` over ``val_partial_accuracy`` (the same scalar
    early-stopping uses) and return the test metrics at that epoch — i.e.
    test-at-best-val, never selecting on test.

    Without val (legacy two-way path): selection happens on test, so the
    headline is ``argmax`` over ``test_partial_accuracy`` — byte-identical to
    the previous ``best`` scalar.

    Returns a dict with ``epoch``, ``test_accuracy`` (complete), and
    ``test_partial_accuracy`` plus the selection scalar used.
    """
    if not history:
        return {
            "epoch": -1,
            "selection": None,
            "selection_value": 0.0,
            "test_accuracy": 0.0,
            "test_partial_accuracy": 0.0,
        }
    selection_key = (
        "val_partial_accuracy"
        if "val_partial_accuracy" in history[0]
        else "test_partial_accuracy"
    )
    best = max(history, key=lambda h: h[selection_key])
    return {
        "epoch": int(best["epoch"]),
        "selection": selection_key,
        "selection_value": float(best[selection_key]),
        "test_accuracy": float(best["test_accuracy"]),
        "test_partial_accuracy": float(best["test_partial_accuracy"]),
    }
