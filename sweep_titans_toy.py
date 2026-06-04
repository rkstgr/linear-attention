"""W&B sweep entrypoint for the Titans MQAR curriculum.

Run one sampled config:
    uv run --with wandb python sweep_titans_toy.py

Create and launch the sweep:
    uv run --with wandb wandb sweep sweeps/titans_toy.yaml
    uv run --with wandb wandb agent <entity/project/sweep_id>
"""

import dataclasses
import os
import argparse

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

import jax
import equinox as eqx

from data import CONFIGS
from titans import Transformer, make_wandb_logger
from train import train_and_eval

CONV_SIZE = 4
TRAIN_FLOP_MULTIPLIER = 3.0


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
    parser.add_argument("--config-name", choices=sorted(CONFIGS))
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience-epochs", type=int)
    parser.add_argument("--diagnostics", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "wandb is not installed. Run with: "
            "uv run --with wandb python sweep_titans_toy.py"
        ) from exc

    run = wandb.init()
    wc = wandb.config

    def value(name, default):
        cli_value = getattr(args, name.replace("-", "_"), None)
        if cli_value is not None:
            return cli_value
        return wc.get(name, default)

    config_name = value("config_name", "toy")
    base_cfg = CONFIGS[config_name]
    cfg = dataclasses.replace(
        base_cfg,
        n_train=int(value("n_train", base_cfg.n_train)),
        n_test=int(value("n_test", base_cfg.n_test)),
        batch_size=int(value("batch_size", base_cfg.batch_size)),
        eval_batch_size=int(value("eval_batch_size", base_cfg.eval_batch_size)),
        max_epochs=int(value("max_epochs", base_cfg.max_epochs)),
        learning_rate=float(value("learning_rate", base_cfg.learning_rate)),
        patience_epochs=int(value("patience_epochs", base_cfg.patience_epochs)),
    )

    seed = int(value("seed", 1))
    key = jax.random.PRNGKey(seed)
    k_model, k_train = jax.random.split(key)

    dim = int(wc.get("dim", 64))
    n_heads = int(wc.get("n_heads", 4))
    n_layers = int(wc.get("n_layers", 2))
    mlp_mult = int(wc.get("mlp_mult", 4))
    memory_mult = int(value("memory_mult", 4))
    max_inner_lr = float(value("max_inner_lr", 0.005))

    model = Transformer(
        vocab_size=cfg.vocab_size,
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
        vocab_size=cfg.vocab_size,
        seq_len=cfg.input_seq_len,
        dim=dim,
        n_heads=n_heads,
        n_layers=n_layers,
        mlp_mult=mlp_mult,
        memory_mult=memory_mult,
    )
    steps_per_epoch = cfg.n_train // cfg.batch_size
    train_flops_per_step = int(TRAIN_FLOP_MULTIPLIER * forward_flops * cfg.batch_size)
    run.config.update(
        {
            "config_name": config_name,
            "vocab_size": cfg.vocab_size,
            "input_seq_len": cfg.input_seq_len,
            "num_kv_pairs": cfg.num_kv_pairs,
            "param_count": param_count,
            "forward_flops_per_example": forward_flops,
            "forward_flops_per_token": forward_flops / cfg.input_seq_len,
            "train_flops_per_step_est": train_flops_per_step,
            "train_flop_multiplier_est": TRAIN_FLOP_MULTIPLIER,
        },
        allow_val_change=True,
    )
    run.log(
        {
            "model/params": param_count,
            "compute/forward_flops_per_example": forward_flops,
            "compute/forward_flops_per_token": forward_flops / cfg.input_seq_len,
            "compute/train_flops_per_step_est": train_flops_per_step,
        },
        step=0,
    )

    diagnostics = value("diagnostics", True)
    log_fn = make_wandb_logger(
        run,
        cfg,
        diagnostics_enabled=bool(diagnostics),
    )
    model, history = train_and_eval(model, cfg, k_train, log_fn=log_fn)

    best_test_acc = max(h["test_acc"] for h in history)
    best_epoch = max(history, key=lambda h: h["test_acc"])["epoch"]
    final = history[-1]
    total_train_flops = train_flops_per_step * steps_per_epoch * len(history)
    run.log(
        {
            "sweep/best_test_acc": best_test_acc,
            "sweep/best_epoch": best_epoch,
            "sweep/final_test_acc": final["test_acc"],
            "sweep/final_train_acc": final["train_acc"],
            "sweep/final_train_loss": final["train_loss"],
            "sweep/epochs_ran": len(history),
            "compute/total_train_flops_est": total_train_flops,
            "compute/total_train_tflops_est": total_train_flops / 1e12,
        }
    )
    run.finish()


if __name__ == "__main__":
    main()
