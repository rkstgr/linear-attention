"""Default experiment step builders."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jax
import optax

from data import Config
from executor import ExecutorStep, SourceSet, this_output_path
from models.registry import build_lm_model
from train import train_and_eval


CORE_SOURCES = SourceSet(
    "core",
    (
        "experiments/defaults.py",
        "train.py",
        "data.py",
        "utils.py",
        "models/backbone.py",
        "models/registry.py",
        "models/ffn.py",
    ),
)

MIXER_SOURCES = {
    "transformer": SourceSet("mixer:transformer", ("models/attention.py",)),
    "linear_attention": SourceSet("mixer:linear_attention", ("models/linear_attention.py",)),
    "deltanet": SourceSet("mixer:deltanet", ("models/deltanet.py",)),
    "gated_deltanet": SourceSet("mixer:gated_deltanet", ("models/deltanet.py",)),
    "titans": SourceSet("mixer:titans", ("models/titans.py",)),
}


@dataclass(frozen=True)
class ModelConfig:
    mixer: str
    vocab_size: int
    dim: int
    n_heads: int
    n_layers: int
    mlp_mult: int
    mixer_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MQARConfig:
    vocab_size: int
    input_seq_len: int
    num_kv_pairs: int
    power_a: float
    n_train: int
    n_test: int


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    eval_batch_size: int
    max_epochs: int
    learning_rate: float
    target_acc: float
    patience_epochs: int


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    data: MQARConfig
    train: TrainConfig
    seed: int


@dataclass(frozen=True)
class TrainRunConfig:
    run: RunConfig
    output_path: str


def mqar_sources() -> SourceSet:
    return SourceSet("task:mqar", ("data.py",))


def mixer_sources(model: ModelConfig) -> SourceSet:
    try:
        return MIXER_SOURCES[model.mixer]
    except KeyError as exc:
        raise ValueError(
            f"unknown mixer {model.mixer!r}; choices: {sorted(MIXER_SOURCES)}"
        ) from exc


def config_from_data_config(cfg: Config) -> tuple[MQARConfig, TrainConfig]:
    return (
        MQARConfig(
            vocab_size=cfg.vocab_size,
            input_seq_len=cfg.input_seq_len,
            num_kv_pairs=cfg.num_kv_pairs,
            power_a=cfg.power_a,
            n_train=cfg.n_train,
            n_test=cfg.n_test,
        ),
        TrainConfig(
            batch_size=cfg.batch_size,
            eval_batch_size=cfg.eval_batch_size,
            max_epochs=cfg.max_epochs,
            learning_rate=cfg.learning_rate,
            target_acc=cfg.target_acc,
            patience_epochs=cfg.patience_epochs,
        ),
    )


def data_config(run: RunConfig) -> Config:
    return Config(
        vocab_size=run.data.vocab_size,
        input_seq_len=run.data.input_seq_len,
        num_kv_pairs=run.data.num_kv_pairs,
        power_a=run.data.power_a,
        n_train=run.data.n_train,
        n_test=run.data.n_test,
        batch_size=run.train.batch_size,
        eval_batch_size=run.train.eval_batch_size,
        max_epochs=run.train.max_epochs,
        learning_rate=run.train.learning_rate,
        target_acc=run.train.target_acc,
        patience_epochs=run.train.patience_epochs,
    )


def default_train(name: str, run: RunConfig) -> ExecutorStep:
    return ExecutorStep(
        name=f"checkpoints/{name}",
        fn=train_run,
        config=TrainRunConfig(run=run, output_path=this_output_path()),
        sources=(CORE_SOURCES, mqar_sources(), mixer_sources(run.model)),
    )


def train_run(config: TrainRunConfig) -> None:
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    cfg = data_config(config.run)
    k_model, k_train = jax.random.split(jax.random.PRNGKey(config.run.seed))
    model = build_lm_model(
        config.run.model.mixer,
        config.run.model.vocab_size,
        config.run.model.dim,
        config.run.model.n_heads,
        config.run.model.n_layers,
        config.run.model.mlp_mult,
        k_model,
        **config.run.model.mixer_kwargs,
    )
    opt = optax.adamw(cfg.learning_rate)
    _, history, stop_info = train_and_eval(model, cfg, k_train, opt=opt, return_info=True)
    best = max((h["test_acc"] for h in history), default=0.0)
    metrics = {
        "best": best,
        "history": history,
        "stop_info": stop_info,
        "run": asdict(config.run),
    }
    (output_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
