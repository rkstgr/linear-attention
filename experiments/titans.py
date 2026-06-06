"""Titans MQAR experiment entrypoint.

Run:
    uv run python -m experiments.titans
"""

import argparse
import dataclasses
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

import jax

from experiments.mqar import MQAR_CONFIGS, resolve
from linattn.models.factory import build_lm_model
from linattn.tasks.base import build_task
from linattn.train import MultiReporter, StdoutReporter, WandbReporter, fit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=sorted(MQAR_CONFIGS), default="toy")
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


def configs_from_args(args):
    overrides = {
        "vocab_size": args.vocab_size,
        "input_seq_len": args.seq_len,
        "num_kv_pairs": args.num_kv_pairs,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "target_acc": args.target_acc,
        "patience_epochs": args.patience_epochs,
    }
    return resolve(args.config, **overrides)


def init_wandb(args, data_cfg, train_cfg):
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "wandb is not installed. Run: "
            "uv run --group experiment python -m experiments.titans --wandb"
        ) from exc

    config = {
        **dataclasses.asdict(data_cfg),
        **dataclasses.asdict(train_cfg),
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
    data_cfg, train_cfg = configs_from_args(args)
    task = build_task(data_cfg)
    print(
        "--- MQAR (Titans recurrent, "
        f"config={args.config}, vocab={data_cfg.vocab_size}, "
        f"T={data_cfg.input_seq_len}, N_KV={data_cfg.num_kv_pairs}) ---"
    )
    k_model, k_train, k_inspect = jax.random.split(jax.random.PRNGKey(1), 3)

    model = build_lm_model(
        "titans",
        vocab_size=data_cfg.vocab_size,
        dim=64,
        n_heads=4,
        n_layers=2,
        mlp_mult=4,
        key=k_model,
        memory_mult=args.memory_mult,
        max_inner_lr=args.max_inner_lr,
    )

    wandb_run = init_wandb(args, data_cfg, train_cfg) if args.wandb else None
    if wandb_run is not None:
        reporter = MultiReporter([StdoutReporter(), WandbReporter(wandb_run)])
    else:
        reporter = StdoutReporter()

    result = fit(model, task, train_cfg, k_train, reporter=reporter)
    task.describe(result.model, k_inspect)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
