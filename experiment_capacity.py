"""Minimal experiment: capacity ceiling separates LinAttn from DeltaNet.

Mechanism: LA's additive write `S += v k^T` saturates the rank-d_k state once
N_KV exceeds head_dim. DN's delta rule overwrites along the key direction and
keeps working past the ceiling.

Two cells per model — one well below capacity, one well above. Below is the
sanity check (both should solve); above is the demonstration (LA fails, DN
holds).

Held constant: vocab=256, T=128, dim=64, head_dim=16, lr=3e-3, n_train=20k,
max_epochs=16, batch=64. ~5 minutes on CPU.

Cells are cached by (config + relevant source-file bytes) under
`.experiment_cache/`. Pass `--rerun` to force recomputation. Pass
`--parallel N` to run up to N cells concurrently (defaults to 1 = serial).
Pair `--parallel 4` with running both experiment scripts at once for 8-way
total concurrency on an 8+-core Mac.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

import dataclasses
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from data import Config, level1


# d_head = dim / n_heads = 64 / 4 = 16 → capacity ceiling at N_KV ≈ 16.
CFG = dataclasses.replace(
    level1,
    vocab_size=256,        # 30× cheaper lm_head than vocab=8192; mechanism unchanged
    input_seq_len=128,     # T ≥ 4·N_KV at N_KV=32
    n_train=20_000,
    max_epochs=16,
    patience_epochs=999,   # DN has a long flat init phase at high N_KV
    learning_rate=3e-3,
)

N_KVS = [4, 32]            # below ceiling, above ceiling

# Model specs: pure data so they pickle cleanly across spawn boundaries.
MODEL_SPECS = [
    {"label": "Linear Attention", "mixer": "linear_attention"},
    {"label": "DeltaNet",         "mixer": "deltanet"},
]

SHARED_SOURCES = [
    "train.py",
    "data.py",
    "utils.py",
    "models/backbone.py",
    "models/registry.py",
    "models/ffn.py",
]
MIXER_SOURCES = {
    "linear_attention": "models/linear_attention.py",
    "deltanet": "models/deltanet.py",
}
ARCH = "dim=64,n_heads=4,n_layers=2,mlp_mult=4"


def _worker(spec):
    """Run one cell. Top-level + picklable so ProcessPoolExecutor can dispatch.

    Each spawned worker re-imports JAX fresh (own device init, own JIT cache).
    """
    import os as _os
    _os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import jax
    import optax

    from cache import cached
    from models.registry import build_lm_model
    from train import train_and_eval

    cfg = Config(**spec["cfg_kwargs"])
    cache_key = {
        "label": spec["label"], "n_kv": spec["n_kv"],
        "cfg": spec["cfg_kwargs"], "model": ARCH,
    }
    sources = [MIXER_SOURCES[spec["mixer"]], *SHARED_SOURCES]
    hit, save = cached(cache_key, sources, rerun=spec["rerun"])
    if hit is not None:
        return (spec["n_kv"], spec["label"], hit["best"], "cached")

    k_model, k_train = jax.random.split(jax.random.PRNGKey(1))
    model = build_lm_model(spec["mixer"], cfg.vocab_size, 64, 4, 2, 4, k_model)
    opt = optax.adamw(cfg.learning_rate)
    _, history = train_and_eval(model, cfg, k_train, opt=opt)
    best = max(h["test_acc"] for h in history)
    save({"best": best, "history": history})
    return (spec["n_kv"], spec["label"], best, "fresh")


def _make_specs(rerun: bool):
    specs = []
    for n_kv in N_KVS:
        cfg = dataclasses.replace(CFG, num_kv_pairs=n_kv)
        cfg_kwargs = dataclasses.asdict(cfg)
        for m in MODEL_SPECS:
            specs.append({
                "label": m["label"], "mixer": m["mixer"],
                "cfg_kwargs": cfg_kwargs, "n_kv": n_kv, "rerun": rerun,
            })
    return specs


def main():
    rerun = "--rerun" in sys.argv
    parallel = 1
    if "--parallel" in sys.argv:
        parallel = int(sys.argv[sys.argv.index("--parallel") + 1])

    specs = _make_specs(rerun)
    rows = []

    if parallel <= 1:
        for spec in specs:
            print(f"\n=== {spec['label']}  (N_KV={spec['n_kv']}) ===", flush=True)
            n_kv, label, best, status = _worker(spec)
            print(f"[{status}] {label}  N_KV={n_kv}  best={best:.3f}", flush=True)
            rows.append((n_kv, label, best))
    else:
        ctx = get_context("spawn")
        print(f"running {len(specs)} cells with --parallel {parallel} "
              f"(stdout from workers will interleave)\n", flush=True)
        with ProcessPoolExecutor(max_workers=parallel, mp_context=ctx) as pool:
            futures = {pool.submit(_worker, s): s for s in specs}
            for fut in as_completed(futures):
                n_kv, label, best, status = fut.result()
                print(f"[{status}] {label}  N_KV={n_kv}  best={best:.3f}",
                      flush=True)
                rows.append((n_kv, label, best))

    rows.sort(key=lambda r: (r[0], r[1]))
    print("\n\n## Summary\n")
    print("| N_KV | model            | best_acc |")
    print("| ---: | ---------------- | -------: |")
    for n_kv, label, best in rows:
        print(f"| {n_kv:>4d} | {label:<16} | {best:>8.3f} |")


if __name__ == "__main__":
    main()
