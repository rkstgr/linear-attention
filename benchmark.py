"""Benchmark all models on a single MQAR difficulty level.

Trains each model from scratch with the same hyperparams (dim=64, n_heads=4,
n_layers=2, mlp_mult=4) on the chosen level, then prints a markdown table
of final-epoch metrics matching the format used in README.md.

Run:
    uv run python benchmark.py                          # default --level 1
    uv run python benchmark.py --level easy
    uv run python benchmark.py --level medium --models transformer,deltanet
"""

import sys

import jax

from data import get_level
from train import train_and_eval

# Importing each model file is safe — their __main__ blocks are guarded.
from transformer import Transformer as TransformerModel
from linear_attention import Transformer as LinearAttentionModel
from deltanet import Transformer as DeltaNetModel


# slug -> (display name, builder(V, key) -> model)
MODELS = {
    "transformer":      ("Transformer",      lambda V, k: TransformerModel(V, 64, 4, 2, 4, k)),
    "linear_attention": ("Linear Attention", lambda V, k: LinearAttentionModel(V, 64, 4, 2, 4, k)),
    "deltanet":         ("DeltaNet",         lambda V, k: DeltaNetModel(V, 64, 4, 2, 4, k, gated=False)),
    "gated_deltanet":   ("Gated DeltaNet",   lambda V, k: DeltaNetModel(V, 64, 4, 2, 4, k, gated=True)),
}


def main():
    cfg = get_level(sys.argv)

    if "--models" in sys.argv:
        names = sys.argv[sys.argv.index("--models") + 1].split(",")
        unknown = [n for n in names if n not in MODELS]
        if unknown:
            raise ValueError(f"unknown --models {unknown}; available: {list(MODELS)}")
        chosen = [(n, *MODELS[n]) for n in names]
    else:
        chosen = [(n, *v) for n, v in MODELS.items()]

    print(
        f"benchmark: vocab={cfg.vocab_size}  T={cfg.input_seq_len}  "
        f"N_KV={cfg.num_kv_pairs}  n_train={cfg.n_train}  "
        f"max_epochs={cfg.max_epochs}"
    )

    rows = []
    for slug, label, build in chosen:
        print(f"\n=== {label} ===")
        k_model, k_train = jax.random.split(jax.random.PRNGKey(1))
        model = build(cfg.vocab_size, k_model)
        _, history = train_and_eval(model, cfg, k_train)
        last = history[-1]
        rows.append((label, last["epoch"], last["train_loss"], last["train_acc"], last["test_acc"]))

    print("\n")
    print("| model            | epoch | train_loss | train_acc | test_acc |")
    print("| ---------------- | ----: | ---------: | --------: | -------: |")
    for label, epoch, train_loss, train_acc, test_acc in rows:
        print(
            f"| {label:<16} | {epoch:>5d} | {train_loss:>10.4f} | "
            f"{train_acc:>9.3f} | {test_acc:>8.3f} |"
        )


if __name__ == "__main__":
    main()
