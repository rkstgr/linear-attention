"""Generic W&B sweep entrypoint, arch- and task-parametric.

One sampled cell: pick an arch (``mixer``), a task (``task``), and the shape /
optimizer knobs from ``wandb.config``. Per-arch extra hyperparameters come from
``linattn.archs.ARCHS`` so adding an arch to a sweep means adding it to the
registry, not editing this file.

Replaces the Titans-hardcoded ``experiments/sweep_titans_toy.py`` (proposal
0004). Keeps the metric-grouping convention (``learning/``, ``runtime/``,
``stability/``, ``health/``, ``objective/``, ``sweep/``) and the
``objective/score`` selection metric W&B sweeps target.

Run one sampled config:
    uv run --group experiment python -m experiments.sweep

Create and launch a sweep:
    uv run --group experiment wandb sweep sweeps/titans_toy.yaml
    scripts/run_wandb_agent_cpu.sh <entity/project/sweep_id>
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

import equinox as eqx
import jax

import dataclasses

from experiments.mqar import MQAR_CONFIGS
from linattn.archs import ARCHS, mixer_build_kwargs
from linattn.config import TrainConfig
from linattn.models.factory import build_lm_model
from linattn.tasks.addition import AdditionConfig
from linattn.tasks.base import build_task
from linattn.tasks.mqar import MQARConfig
from linattn.train import WandbReporter, fit

NONFINITE_SCORE_PENALTY = 0.05
GPU_PLATFORMS = {"cuda", "gpu"}


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def jax_device_summary(devices) -> str:
    pieces = []
    for device in devices:
        device_id = getattr(device, "id", "?")
        device_kind = getattr(device, "device_kind", "")
        label = f"{device.platform}:{device_id}"
        if device_kind:
            label = f"{label}:{device_kind}"
        pieces.append(label)
    return ",".join(pieces)


def jax_runtime_info():
    try:
        backend = jax.default_backend()
        devices = jax.devices()
    except Exception as exc:
        if env_flag("REQUIRE_JAX_GPU"):
            raise SystemExit(
                "REQUIRE_JAX_GPU=1 but JAX could not initialize devices. "
                f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS', '')!r}. "
                "Check the CUDA driver and JAX CUDA dependency group."
            ) from exc
        raise
    has_gpu = backend in GPU_PLATFORMS or any(
        device.platform in GPU_PLATFORMS for device in devices
    )
    return backend, devices, has_gpu


def update_unlocked_config(run, current_config, values):
    """Avoid W&B sweep-lock warnings by only adding genuinely new config keys."""
    unlocked = {k: v for k, v in values.items() if k not in current_config}
    if unlocked:
        run.config.update(unlocked, allow_val_change=True)


def missing_to(value, default):
    return default if value is None else value


def count_params(model) -> int:
    """Exact trainable inexact-array parameter count."""
    leaves = [x for x in jax.tree.leaves(model) if eqx.is_inexact_array(x)]
    return int(sum(x.size for x in leaves))


# Task-config builders. Each builder reads its task's parametric knobs through
# a `getter(name, default)` callable so the entrypoint can layer CLI > W&B
# config > builtin defaults uniformly. Selecting a task is one config key
# (``task``); the knobs read off the same getter.

def _mqar_config(get) -> MQARConfig:
    # `config_name` is a Zoology-preset shortcut (toy/easy/level1). When given
    # it supplies the cell-shape defaults; individual knobs in the W&B config
    # still override per-field, so sweeps can tweak a preset on a single axis.
    preset_name = get("config_name", "")
    if preset_name:
        if preset_name not in MQAR_CONFIGS:
            raise SystemExit(
                f"unknown config_name {preset_name!r}; choices: {sorted(MQAR_CONFIGS)}"
            )
        base = MQAR_CONFIGS[preset_name]
        overrides = {
            f.name: get(f.name, getattr(base, f.name))
            for f in dataclasses.fields(MQARConfig)
            if f.name != "name"
        }
        return dataclasses.replace(base, **overrides)
    return MQARConfig(
        vocab_size=int(get("vocab_size", 512)),
        input_seq_len=int(get("input_seq_len", 64)),
        num_kv_pairs=int(get("num_kv_pairs", 4)),
        power_a=float(get("power_a", 0.01)),
        n_train=int(get("n_train", 50_000)),
        n_test=int(get("n_test", 2_000)),
        n_val=int(get("n_val", 2_000)),
    )


def _addition_config(get) -> AdditionConfig:
    return AdditionConfig(
        max_digits=int(get("max_digits", 3)),
        num_addends=int(get("num_addends", 2)),
        n_train=int(get("n_train", 50_000)),
        n_test=int(get("n_test", 2_000)),
        n_val=int(get("n_val", 2_000)),
    )


TASK_BUILDERS = {"mqar": _mqar_config, "addition": _addition_config}


def build_task_config(task_name: str, get) -> object:
    try:
        builder = TASK_BUILDERS[task_name]
    except KeyError as exc:
        raise ValueError(
            f"unknown task {task_name!r}; choices: {sorted(TASK_BUILDERS)}"
        ) from exc
    return builder(get)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixer", choices=sorted(ARCHS))
    parser.add_argument("--task", choices=sorted(TASK_BUILDERS))
    parser.add_argument("--memory-mult", type=int)
    parser.add_argument("--max-inner-lr", type=float)
    # CLI overrides for the most common knobs (anything missing falls back to
    # `wandb.config` then to a built-in default).
    for name in (
        "dim", "n_heads", "n_layers", "mlp_mult", "head_dim", "max_epochs",
        "batch_size", "eval_batch_size", "patience_epochs", "seed",
        "n_train", "n_val", "n_test", "vocab_size", "input_seq_len",
        "num_kv_pairs", "max_digits", "num_addends",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--target-acc", type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    jax_backend, jax_devices, jax_has_gpu = jax_runtime_info()
    jax_devices_str = jax_device_summary(jax_devices)
    if env_flag("REQUIRE_JAX_GPU") and not jax_has_gpu:
        raise SystemExit(
            "REQUIRE_JAX_GPU=1 but JAX did not initialize a GPU. "
            f"backend={jax_backend!r} devices={jax_devices_str!r}."
        )

    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "wandb is not installed. Run with: "
            "uv run --group experiment python -m experiments.sweep"
        ) from exc

    run = wandb.init()
    wc = wandb.config

    def value(name, default):
        cli_value = getattr(args, name.replace("-", "_"), None)
        if cli_value is not None:
            return cli_value
        return wc.get(name, default)

    mixer = str(value("mixer", "titans"))
    task_name = str(value("task", "mqar"))
    if mixer not in ARCHS:
        raise SystemExit(f"unknown mixer {mixer!r}; choices: {sorted(ARCHS)}")

    task_cfg = build_task_config(task_name, value)
    task = build_task(task_cfg)

    # Shape: one scale knob (`dim`), heads inferred from `head_dim` when given.
    dim = int(value("dim", 64))
    head_dim = int(wc.get("head_dim", 0)) or None
    if head_dim is not None:
        if dim % head_dim != 0:
            raise SystemExit(f"dim={dim} not divisible by head_dim={head_dim}")
        n_heads = dim // head_dim
    else:
        n_heads = int(value("n_heads", 4))
    n_layers = int(value("n_layers", 2))
    mlp_mult = int(value("mlp_mult", 4))

    train_cfg = TrainConfig(
        batch_size=int(value("batch_size", 64)),
        eval_batch_size=int(value("eval_batch_size", 64)),
        max_epochs=int(value("max_epochs", 16)),
        learning_rate=float(value("learning_rate", 1e-3)),
        target_acc=float(value("target_acc", 0.99)),
        patience_epochs=int(value("patience_epochs", 5)),
    )

    seed = int(value("seed", 1))
    key = jax.random.PRNGKey(seed)
    k_model, k_train = jax.random.split(key)

    extra_kwargs = mixer_build_kwargs(mixer, wc)
    model = build_lm_model(
        mixer,
        vocab_size=task_cfg.vocab_size,
        dim=dim,
        n_heads=n_heads,
        n_layers=n_layers,
        mlp_mult=mlp_mult,
        key=k_model,
        **extra_kwargs,
    )

    param_count = count_params(model)
    # FLOP estimator is a stub for the 0004 pilot (iso-FLOP deferred).
    forward_flops = ARCHS[mixer].flop_estimator(
        vocab_size=task_cfg.vocab_size,
        dim=dim, n_heads=n_heads, n_layers=n_layers,
        mlp_mult=mlp_mult, **extra_kwargs,
    )

    update_unlocked_config(
        run,
        wc,
        {
            "mixer": mixer,
            "task": task_name,
            "head_dim": dim // n_heads,
            "param_count": param_count,
            "forward_flops_per_example_est": forward_flops,
            "jax_backend": jax_backend,
            "jax_devices": jax_devices_str,
            "jax_platforms_env": os.environ.get("JAX_PLATFORMS", ""),
            "require_jax_gpu": env_flag("REQUIRE_JAX_GPU"),
            "nonfinite_score_penalty": NONFINITE_SCORE_PENALTY,
        },
    )
    run.log(
        {
            "model/params": param_count,
            "runtime/jax_backend_cpu": 1.0 if jax_backend == "cpu" else 0.0,
            "runtime/jax_has_gpu": 1.0 if jax_has_gpu else 0.0,
            "compute/forward_flops_per_example_est": forward_flops,
        },
        step=0,
    )

    result = fit(model, task, train_cfg, k_train, reporter=WandbReporter(run))
    history = result.history
    stop_info = result.stop_info

    # Selection follows the loop: val_partial_accuracy when val present, else
    # test_partial_accuracy (legacy). The objective is the test metric at the
    # epoch that wins on selection — never select on test.
    if not history:
        best_record = None
    else:
        sel_key = (
            "val_partial_accuracy"
            if "val_partial_accuracy" in history[0]
            else "test_partial_accuracy"
        )
        best_record = max(history, key=lambda h: h[sel_key])
    best_test_partial = best_record["test_partial_accuracy"] if best_record else 0.0
    best_test_complete = best_record["test_accuracy"] if best_record else 0.0
    best_epoch = best_record["epoch"] if best_record else -1
    final = history[-1] if history else {
        "test_accuracy": 0.0,
        "test_partial_accuracy": 0.0,
        "train_partial_accuracy": 0.0,
        "train_loss": float("nan"),
    }
    nonfinite = bool(stop_info["nonfinite"])
    objective_score = best_test_partial - (NONFINITE_SCORE_PENALTY if nonfinite else 0.0)

    run.log(
        {
            "objective/score": objective_score,
            "objective/best_test_partial_accuracy": best_test_partial,
            "objective/best_test_accuracy": best_test_complete,
            "objective/best_epoch": best_epoch,
            "objective/final_test_partial_accuracy": final["test_partial_accuracy"],
            "objective/final_test_accuracy": final["test_accuracy"],
            "objective/final_train_partial_accuracy": final["train_partial_accuracy"],
            "objective/final_train_loss": final["train_loss"],
            "objective/epochs_ran": len(history),
            "health/nonfinite": 1.0 if nonfinite else 0.0,
            "health/nonfinite_epoch": missing_to(stop_info["nonfinite_epoch"], -1),
            "health/nonfinite_step": missing_to(stop_info["nonfinite_step"], -1),
            "health/nonfinite_global_step": missing_to(
                stop_info["nonfinite_global_step"], -1
            ),
            "health/nonfinite_loss": missing_to(stop_info["nonfinite_loss"], 0.0),
            "health/nonfinite_grad_norm": missing_to(
                stop_info["nonfinite_grad_norm"], 0.0
            ),
            "sweep/score": objective_score,
            "sweep/best_test_partial_accuracy": best_test_partial,
            "sweep/best_test_accuracy": best_test_complete,
            "sweep/best_epoch": best_epoch,
            "sweep/epochs_ran": len(history),
        }
    )
    run.summary["health/stop_reason"] = stop_info["stop_reason"]
    run.finish()


if __name__ == "__main__":
    main()
