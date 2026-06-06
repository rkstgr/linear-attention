"""W&B sweep entrypoint for the Titans MQAR curriculum.

Run one sampled config:
    uv run --group experiment python -m experiments.sweep_titans_toy

Create and launch the sweep:
    uv run --group experiment wandb sweep sweeps/titans_toy.yaml
    scripts/run_wandb_agent_cpu.sh <entity/project/sweep_id>
    scripts/run_wandb_agent_cuda.sh <entity/project/sweep_id> --count 30
"""

import os
import argparse

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

import jax
import equinox as eqx

from experiments.mqar import MQAR_CONFIGS, resolve
from linattn.models.factory import build_lm_model
from linattn.tasks.base import build_task
from linattn.train import WandbReporter, fit

CONV_SIZE = 4
TRAIN_FLOP_MULTIPLIER = 3.0
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


def estimate_forward_flops_per_example(
    *,
    vocab_size: int,
    seq_len: int,
    dim: int,
    n_heads: int,
    n_layers: int,
    mlp_mult: int,
    memory_mult: int,
) -> int:
    """Rough forward FLOPs for one sequence.

    Counts dense matmuls as 2*m*n*k multiply-add FLOPs. Elementwise activations,
    norms, softmax-free nonlinearities, indexing, and optimizer bookkeeping are
    either omitted or approximated. The estimate is meant for comparing sweep
    runs, not hardware accounting.
    """
    head_dim = dim // n_heads
    memory_hidden = memory_mult * head_dim
    block_hidden = mlp_mult * dim

    qkvo_proj = 4 * 2 * seq_len * dim * dim
    conv = 3 * 2 * seq_len * CONV_SIZE * dim
    gates = 3 * 2 * seq_len * dim * n_heads

    # Per head/token: two memory MLP reads, manual L2-gradient construction,
    # and momentum/retention updates over W1/W2.
    fast_memory = n_heads * seq_len * 22 * head_dim * memory_hidden

    block_mlp = 6 * seq_len * dim * block_hidden
    per_block = qkvo_proj + conv + gates + fast_memory + block_mlp
    lm_head = 2 * seq_len * dim * vocab_size
    return int(n_layers * per_block + lm_head)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", choices=sorted(MQAR_CONFIGS))
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience-epochs", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    jax_backend, jax_devices, jax_has_gpu = jax_runtime_info()
    jax_devices_str = jax_device_summary(jax_devices)
    if env_flag("REQUIRE_JAX_GPU") and not jax_has_gpu:
        raise SystemExit(
            "REQUIRE_JAX_GPU=1 but JAX did not initialize a GPU. "
            f"backend={jax_backend!r} devices={jax_devices_str!r}. "
            "Run the CUDA agent from a Linux x86_64 GPU VM after "
            "`uv sync --group cuda --group experiment`."
        )

    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "wandb is not installed. Run with: "
            "uv run --group experiment python -m experiments.sweep_titans_toy"
        ) from exc

    run = wandb.init()
    wc = wandb.config

    def value(name, default):
        cli_value = getattr(args, name.replace("-", "_"), None)
        if cli_value is not None:
            return cli_value
        return wc.get(name, default)

    config_name = value("config_name", "toy")
    base_data, base_train = resolve(config_name)
    data_cfg, train_cfg = resolve(
        config_name,
        n_train=int(value("n_train", base_data.n_train)),
        n_test=int(value("n_test", base_data.n_test)),
        batch_size=int(value("batch_size", base_train.batch_size)),
        eval_batch_size=int(value("eval_batch_size", base_train.eval_batch_size)),
        max_epochs=int(value("max_epochs", base_train.max_epochs)),
        learning_rate=float(value("learning_rate", base_train.learning_rate)),
        patience_epochs=int(value("patience_epochs", base_train.patience_epochs)),
    )
    task = build_task(data_cfg)

    seed = int(value("seed", 1))
    key = jax.random.PRNGKey(seed)
    k_model, k_train = jax.random.split(key)

    dim = int(wc.get("dim", 64))
    n_heads = int(wc.get("n_heads", 4))
    n_layers = int(wc.get("n_layers", 2))
    mlp_mult = int(wc.get("mlp_mult", 4))
    memory_mult = int(value("memory_mult", 4))
    max_inner_lr = float(value("max_inner_lr", 0.005))

    model = build_lm_model(
        "titans",
        vocab_size=data_cfg.vocab_size,
        dim=dim,
        n_heads=n_heads,
        n_layers=n_layers,
        mlp_mult=mlp_mult,
        key=k_model,
        memory_mult=memory_mult,
        max_inner_lr=max_inner_lr,
    )

    param_count = count_params(model)
    forward_flops = estimate_forward_flops_per_example(
        vocab_size=data_cfg.vocab_size,
        seq_len=data_cfg.input_seq_len,
        dim=dim,
        n_heads=n_heads,
        n_layers=n_layers,
        mlp_mult=mlp_mult,
        memory_mult=memory_mult,
    )
    steps_per_epoch = data_cfg.n_train // train_cfg.batch_size
    train_flops_per_step = int(TRAIN_FLOP_MULTIPLIER * forward_flops * train_cfg.batch_size)
    update_unlocked_config(
        run,
        wc,
        {
            "vocab_size": data_cfg.vocab_size,
            "input_seq_len": data_cfg.input_seq_len,
            "num_kv_pairs": data_cfg.num_kv_pairs,
            "jax_backend": jax_backend,
            "jax_devices": jax_devices_str,
            "jax_platforms_env": os.environ.get("JAX_PLATFORMS", ""),
            "require_jax_gpu": env_flag("REQUIRE_JAX_GPU"),
            "param_count": param_count,
            "forward_flops_per_example": forward_flops,
            "forward_flops_per_token": forward_flops / data_cfg.input_seq_len,
            "train_flops_per_step_est": train_flops_per_step,
            "train_flop_multiplier_est": TRAIN_FLOP_MULTIPLIER,
            "nonfinite_score_penalty": NONFINITE_SCORE_PENALTY,
        },
    )
    run.log(
        {
            "model/params": param_count,
            "runtime/jax_backend_cpu": 1.0 if jax_backend == "cpu" else 0.0,
            "runtime/jax_backend_gpu": 1.0 if jax_has_gpu else 0.0,
            "runtime/jax_has_gpu": 1.0 if jax_has_gpu else 0.0,
            "runtime/has_gpu": 1.0 if jax_has_gpu else 0.0,
            "compute/forward_flops_per_example": forward_flops,
            "compute/forward_flops_per_token": forward_flops / data_cfg.input_seq_len,
            "compute/train_flops_per_step_est": train_flops_per_step,
        },
        step=0,
    )

    result = fit(model, task, train_cfg, k_train, reporter=WandbReporter(run))
    history = result.history
    train_info = result.stop_info

    best_record = max(history, key=lambda h: h["test_acc"], default=None)
    best_test_acc = best_record["test_acc"] if best_record is not None else 0.0
    best_epoch = best_record["epoch"] if best_record is not None else -1
    final = history[-1] if history else {
        "test_acc": 0.0,
        "train_acc": 0.0,
        "train_loss": float("nan"),
    }
    total_train_flops = train_flops_per_step * steps_per_epoch * len(history)
    nonfinite = bool(train_info["nonfinite"])
    objective_score = best_test_acc - (NONFINITE_SCORE_PENALTY if nonfinite else 0.0)
    run.log(
        {
            "objective/score": objective_score,
            "objective/best_test_acc": best_test_acc,
            "objective/best_epoch": best_epoch,
            "objective/final_test_acc": final["test_acc"],
            "objective/final_train_acc": final["train_acc"],
            "objective/final_train_loss": final["train_loss"],
            "objective/epochs_ran": len(history),
            "health/nonfinite": 1.0 if nonfinite else 0.0,
            "health/nonfinite_epoch": missing_to(train_info["nonfinite_epoch"], -1),
            "health/nonfinite_step": missing_to(train_info["nonfinite_step"], -1),
            "health/nonfinite_global_step": missing_to(
                train_info["nonfinite_global_step"], -1
            ),
            "health/nonfinite_loss": missing_to(train_info["nonfinite_loss"], 0.0),
            "health/nonfinite_grad_norm": missing_to(
                train_info["nonfinite_grad_norm"], 0.0
            ),
            "health/nonfinite_update_norm": missing_to(
                train_info["nonfinite_update_norm"], 0.0
            ),
            "health/nonfinite_param_norm": missing_to(
                train_info["nonfinite_param_norm"], 0.0
            ),
            "sweep/best_test_acc": best_test_acc,
            "sweep/score": objective_score,
            "sweep/best_epoch": best_epoch,
            "sweep/final_test_acc": final["test_acc"],
            "sweep/final_train_acc": final["train_acc"],
            "sweep/final_train_loss": final["train_loss"],
            "sweep/epochs_ran": len(history),
            "compute/total_train_flops_est": total_train_flops,
            "compute/total_train_tflops_est": total_train_flops / 1e12,
        }
    )
    run.summary["health/stop_reason"] = train_info["stop_reason"]
    run.finish()


if __name__ == "__main__":
    main()
