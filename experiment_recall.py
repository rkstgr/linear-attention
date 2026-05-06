"""Controlled state-capacity sweep on MQAR.

Question: do we reproduce (a) DeltaNet > Linear Attention as N_KV grows past
head_dim, and (b) Gated DeltaNet >= DeltaNet on top of that?

Hypothesis: Linear Attention's additive write s += k v^T saturates the rank-Dh
state once N_KV exceeds head_dim; DeltaNet's delta rule overwrites in the key
direction and keeps working; gating adds controlled forgetting that should help
when query gaps span more noise.

Held constant across every run:
    vocab=8192  T=128  dim=64  n_heads=4  n_layers=2  mlp_mult=4
    n_train=100k  max_epochs=48  patience=disabled  lr=1e-3  batch=64
    (DeltaNet has a long flat init phase at N_KV >= 16 — train_loss descends
    but test_acc stays at 0 for many epochs, so patience-based early stop
    kills it before signal emerges. We let every run consume the full epoch
    budget so the comparison is fair.)
    same PRNGKey(1) per N_KV (so each model sees the same train/test split)

Only knob: num_kv_pairs in {8, 16, 24, 32}. With T=128 the data generator
caps N_KV at 32 (requires T >= 4*N_KV). Per-head state rank = head_dim = 16,
so the sweep brackets that limit (8 well below, 16 at it, 24/32 above it).
"""

import dataclasses
import sys

import jax
import optax

from data import level1
from train import train_and_eval
from linear_attention import Transformer as LinearAttention
from deltanet import Transformer as DeltaNet


def make_cfg(num_kv_pairs: int):
    return dataclasses.replace(
        level1,
        input_seq_len=128,
        num_kv_pairs=num_kv_pairs,
        n_train=100_000,
        n_test=1_000,
        max_epochs=48,
        patience_epochs=999,  # disable patience: DeltaNet has a long flat
                              # init phase at high N_KV (test_acc stays at 0
                              # while train_loss descends), so patience-based
                              # early stop kills it before signal emerges.
                              # Rely on max_epochs / target_acc instead.
    )


MODELS = [
    ("Linear Attention", lambda V, k: LinearAttention(V, 64, 4, 2, 4, k)),
    ("DeltaNet",         lambda V, k: DeltaNet(V, 64, 4, 2, 4, k, gated=False)),
    ("Gated DeltaNet",   lambda V, k: DeltaNet(V, 64, 4, 2, 4, k, gated=True)),
]


def main():
    n_kvs = [8, 16, 24, 32]
    if "--n_kvs" in sys.argv:
        n_kvs = [int(x) for x in sys.argv[sys.argv.index("--n_kvs") + 1].split(",")]
    lr = 1e-3
    if "--lr" in sys.argv:
        lr = float(sys.argv[sys.argv.index("--lr") + 1])

    rows = []
    for n_kv in n_kvs:
        cfg = make_cfg(n_kv)
        print(
            f"\n##### N_KV={n_kv}  T={cfg.input_seq_len}  "
            f"head_dim={64 // 4}  n_train={cfg.n_train}  lr={lr:g} #####"
        )
        for label, build in MODELS:
            print(f"\n=== {label}  (N_KV={n_kv}, lr={lr:g}) ===")
            k_model, k_train = jax.random.split(jax.random.PRNGKey(1))
            model = build(cfg.vocab_size, k_model)
            opt = optax.adamw(lr)
            _, history = train_and_eval(model, cfg, k_train, opt=opt)
            last = history[-1]
            best = max(h["test_acc"] for h in history)
            rows.append(
                (n_kv, label, last["epoch"], last["train_loss"],
                 last["train_acc"], last["test_acc"], best)
            )

    print(f"\n\n## Summary  (lr={lr:g})\n")
    print("| N_KV | model            | epoch | train_loss | train_acc | test_acc | best_acc |")
    print("| ---: | ---------------- | ----: | ---------: | --------: | -------: | -------: |")
    for n_kv, label, epoch, tl, ta, te, bt in rows:
        print(
            f"| {n_kv:>4d} | {label:<16} | {epoch:>5d} | {tl:>10.4f} | "
            f"{ta:>9.3f} | {te:>8.3f} | {bt:>8.3f} |"
        )


if __name__ == "__main__":
    main()
