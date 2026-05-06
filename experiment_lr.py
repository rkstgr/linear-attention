"""LR sweep at the most informative single N_KV.

Follow-up to experiment_recall.py: at N_KV=16 our default lr=1e-3 has
Linear Attention at 0.927, DeltaNet at 0.896, Gated DeltaNet at 0.002
(stuck in the init plateau). Zoology takes the max over an LR grid per
arch+difficulty, so a fixed LR may be hiding the published ordering.

This sweeps lr in {1e-3, 3e-3, 1e-2} for each of the three sub-quadratic
mixers at one fixed N_KV. Everything else matches experiment_recall.py.
"""

import dataclasses

import jax
import optax

from data import level1
from train import train_and_eval
from linear_attention import Transformer as LinearAttention
from deltanet import Transformer as DeltaNet


N_KV = 16
LRS = [1e-3, 3e-3, 1e-2]

cfg = dataclasses.replace(
    level1,
    input_seq_len=128,
    num_kv_pairs=N_KV,
    n_train=100_000,
    n_test=1_000,
    max_epochs=48,
    patience_epochs=999,
)

MODELS = [
    ("Linear Attention", lambda V, k: LinearAttention(V, 64, 4, 2, 4, k)),
    ("DeltaNet",         lambda V, k: DeltaNet(V, 64, 4, 2, 4, k, gated=False)),
    ("Gated DeltaNet",   lambda V, k: DeltaNet(V, 64, 4, 2, 4, k, gated=True)),
]


def main():
    rows = []
    for label, build in MODELS:
        for lr in LRS:
            print(f"\n=== {label}  (N_KV={N_KV}, lr={lr:g}) ===")
            k_model, k_train = jax.random.split(jax.random.PRNGKey(1))
            model = build(cfg.vocab_size, k_model)
            opt = optax.adamw(lr)
            _, history = train_and_eval(model, cfg, k_train, opt=opt)
            last = history[-1]
            best = max(h["test_acc"] for h in history)
            rows.append(
                (label, lr, last["epoch"], last["train_loss"],
                 last["train_acc"], last["test_acc"], best)
            )

    print("\n\n## Summary  (N_KV={})\n".format(N_KV))
    print("| model            |    lr | epoch | train_loss | train_acc | test_acc | best_acc |")
    print("| ---------------- | ----: | ----: | ---------: | --------: | -------: | -------: |")
    for label, lr, epoch, tl, ta, te, bt in rows:
        print(
            f"| {label:<16} | {lr:>5.0e} | {epoch:>5d} | {tl:>10.4f} | "
            f"{ta:>9.3f} | {te:>8.3f} | {bt:>8.3f} |"
        )


if __name__ == "__main__":
    main()
