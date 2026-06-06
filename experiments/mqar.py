"""MQAR experiment configs shared by the quick CLIs and sweeps.

`toy` and `easy` are debug-scale presets (re-exported from the task module).
`level1` is a full experiment — Zoology's easiest training config — so it lives
here in the experiments layer rather than among the task presets. It pairs with
its own TrainConfig (more epochs than the debug default).
"""

from __future__ import annotations

import dataclasses

from linattn.config import DEFAULT_TRAIN, TrainConfig
from linattn.tasks.mqar import MQARConfig, easy, toy


# level1 — Zoology's easiest training config from `original_mqar_configs.py`
# (vocab=8192, seq=64, N_KV=4, power-law gaps). Numbers from this run are
# directly comparable to published Zoology curves.
level1 = MQARConfig(
    vocab_size=8192,
    input_seq_len=64,
    num_kv_pairs=4,
    power_a=0.01,
    n_train=100_000,
    n_test=1_000,
)

# Training params follow Zoology defaults except batch_size, which we shrink
# from 256 to 64 so a CPU run fits in memory; with early stopping the easy task
# usually solves in 1-3 epochs anyway.
LEVEL1_TRAIN = TrainConfig(
    batch_size=64,
    eval_batch_size=64,
    max_epochs=32,
    learning_rate=1e-3,
    target_acc=0.99,
    patience_epochs=5,
)


# Data configs the quick CLIs can select by name. Debug presets reuse
# DEFAULT_TRAIN; level1 carries its own train config.
MQAR_CONFIGS = {"toy": toy, "easy": easy, "level1": level1}

_DATA_FIELDS = {f.name for f in dataclasses.fields(MQARConfig)}
_TRAIN_FIELDS = {f.name for f in dataclasses.fields(TrainConfig)}


def train_for(name: str) -> TrainConfig:
    return LEVEL1_TRAIN if name == "level1" else DEFAULT_TRAIN


def resolve(name: str, **overrides) -> tuple[MQARConfig, TrainConfig]:
    """Return (data, train) configs for a named preset, applying non-None overrides.

    Overrides are routed by field name to the data or train config; `None`
    values (unset CLI flags) are ignored.
    """
    data = MQAR_CONFIGS[name]
    train = train_for(name)
    data_over = {
        k: v for k, v in overrides.items() if k in _DATA_FIELDS and v is not None
    }
    train_over = {
        k: v for k, v in overrides.items() if k in _TRAIN_FIELDS and v is not None
    }
    return (
        dataclasses.replace(data, **data_over),
        dataclasses.replace(train, **train_over),
    )
