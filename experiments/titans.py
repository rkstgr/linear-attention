"""Titans MQAR experiment entrypoint.

Run:
    uv run python -m experiments.titans
"""

import argparse
import dataclasses
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

import jax

from data import CONFIGS
from models.registry import build_lm_model
from train import inspect_example, train_and_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=sorted(CONFIGS), default="toy")
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--num-kv-pairs", type=int)
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--target-acc", type=float)
    parser.add_argument("--patience-epochs", type=int)
    parser.add_argument("--max-inner-lr", type=float, default=0.05)
    parser.add_argument("--memory-mult", type=int, default=4)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="linear-attention")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    return parser.parse_args()


def config_from_args(args):
    cfg = CONFIGS[args.config]
    overrides = {}
    arg_to_field = {
        "vocab_size": "vocab_size",
        "seq_len": "input_seq_len",
        "num_kv_pairs": "num_kv_pairs",
        "n_train": "n_train",
        "n_test": "n_test",
        "batch_size": "batch_size",
        "eval_batch_size": "eval_batch_size",
        "max_epochs": "max_epochs",
        "learning_rate": "learning_rate",
        "target_acc": "target_acc",
        "patience_epochs": "patience_epochs",
    }
    for arg_name, field_name in arg_to_field.items():
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = value
    return dataclasses.replace(cfg, **overrides)


def init_wandb(args, cfg):
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "wandb is not installed. Run: "
            "uv run --group experiment python -m experiments.titans --wandb"
        ) from exc

    config = {
        **dataclasses.asdict(cfg),
        "model": "titans_recurrent",
        "config_preset": args.config,
        "memory_mult": args.memory_mult,
        "max_inner_lr": args.max_inner_lr,
    }
    kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_name,
        "config": config,
    }
    if args.wandb_mode is not None:
        kwargs["mode"] = args.wandb_mode
    return wandb.init(**kwargs)


def main():
    args = parse_args()
    cfg = config_from_args(args)
    print(
        "--- MQAR (Titans recurrent, "
        f"config={args.config}, vocab={cfg.vocab_size}, "
        f"T={cfg.input_seq_len}, N_KV={cfg.num_kv_pairs}) ---"
    )
    k_model, k_train, k_inspect = jax.random.split(jax.random.PRNGKey(1), 3)

    model = build_lm_model(
        "titans",
        vocab_size=cfg.vocab_size,
        dim=64,
        n_heads=4,
        n_layers=2,
        mlp_mult=4,
        key=k_model,
        memory_mult=args.memory_mult,
        max_inner_lr=args.max_inner_lr,
    )

    wandb_run = init_wandb(args, cfg) if args.wandb else None
    log_fn = None
    if wandb_run is not None:

        def log_fn(metrics, model):
            payload = {k: v for k, v in metrics.items() if k != "kind"}
            wandb_run.log(payload, step=metrics["global_step"])

    model, _ = train_and_eval(model, cfg, k_train, log_fn=log_fn)
    inspect_example(model, k_inspect, cfg)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
