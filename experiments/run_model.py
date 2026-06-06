"""Run one registry model on MQAR and inspect a held-out example.

Examples:
    uv run python -m experiments.run_model transformer
    uv run python -m experiments.run_model linear_attention
    uv run python -m experiments.run_model deltanet
    uv run python -m experiments.run_model gated_deltanet
"""

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

import jax

from experiments.mqar import MQAR_CONFIGS, resolve
from linattn.models.factory import MIXERS, build_lm_model
from linattn.tasks.base import build_task
from linattn.train import fit

DISPLAY_NAMES = {
    "transformer": "Transformer",
    "linear_attention": "Linear Attention",
    "deltanet": "DeltaNet",
    "gated_deltanet": "Gated DeltaNet",
    "titans": "Titans",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mixer", choices=sorted(MIXERS))
    parser.add_argument("--config", choices=sorted(MQAR_CONFIGS), default="level1")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--mlp-mult", type=int, default=4)
    parser.add_argument("--memory-mult", type=int, default=4)
    parser.add_argument("--max-inner-lr", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    data_cfg, train_cfg = resolve(args.config)
    task = build_task(data_cfg)
    name = DISPLAY_NAMES.get(args.mixer, args.mixer)
    print(f"--- MQAR ({name}, config={args.config}, vocab={data_cfg.vocab_size}) ---")
    k_model, k_train, k_inspect = jax.random.split(jax.random.PRNGKey(1), 3)
    mixer_kwargs = {}
    if args.mixer == "titans":
        mixer_kwargs = {
            "memory_mult": args.memory_mult,
            "max_inner_lr": args.max_inner_lr,
        }

    model = build_lm_model(
        args.mixer,
        vocab_size=data_cfg.vocab_size,
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_mult=args.mlp_mult,
        key=k_model,
        **mixer_kwargs,
    )
    result = fit(model, task, train_cfg, k_train)
    task.describe(result.model, k_inspect)


if __name__ == "__main__":
    main()
