"""Minimal experiment: gating helps when streaming noise pollutes the state.

Mechanism: every non-write token still passes through W_k and writes a rank-1
perturbation into the recurrence. As T grows at fixed N_KV, this noise
accumulates. Gating's `α < 1` decays it; plain DeltaNet has no mechanism to.

N_KV=4 is held well below capacity (head_dim=16) so the *capacity* axis is
not the variable — the only thing that changes between cells is how much
noise the recurrence has to absorb.

Two T points per model. T=64 is short enough that noise is negligible (DN
should match GDN). T=512 is long enough that noise dominates (GDN should
hold while DN decays).

Held constant: vocab=1024, N_KV=4, dim=64, head_dim=16, lr=3e-3,
n_train=20k, max_epochs=16, batch=64. ~5 minutes on CPU.

Cells are cached by (config + relevant source-file bytes) under
`.experiment_cache/`. Pass `--rerun` to force recomputation. Pass
`--parallel N` to run up to N cells concurrently (defaults to 1 = serial).
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

import dataclasses
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from linattn.data import Config, level1


# vocab > T at the largest T (assertion in data.py). 1024 > 512.
CFG = dataclasses.replace(
    level1,
    vocab_size=1024,
    num_kv_pairs=4,        # well below head_dim=16; capacity is not the variable
    n_train=20_000,
    max_epochs=16,
    patience_epochs=999,
    learning_rate=3e-3,
)

TS = [64, 512]             # short → noise negligible; long → noise dominates

MODEL_SPECS = [
    {"label": "DeltaNet",       "mixer": "deltanet"},
    {"label": "Gated DeltaNet", "mixer": "gated_deltanet"},
]

SHARED_SOURCES = [
    "linattn/train.py",
    "linattn/data.py",
    "linattn/utils.py",
    "linattn/models/backbone.py",
    "linattn/models/factory.py",
    "linattn/models/ffn.py",
]
MIXER_SOURCES = {
    "deltanet": "linattn/models/deltanet.py",
    "gated_deltanet": "linattn/models/deltanet.py",
}
ARCH = "dim=64,n_heads=4,n_layers=2,mlp_mult=4"


def _worker(spec):
    """Run one cell. Top-level + picklable for ProcessPoolExecutor."""
    import os as _os
    _os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import jax
    import optax

    from linattn.cache import cached
    from linattn.models.factory import build_lm_model
    from linattn.train import train_and_eval

    cfg = Config(**spec["cfg_kwargs"])
    cache_key = {
        "label": spec["label"], "T": spec["T"],
        "cfg": spec["cfg_kwargs"], "model": ARCH,
    }
    sources = [MIXER_SOURCES[spec["mixer"]], *SHARED_SOURCES]
    hit, save = cached(cache_key, sources, rerun=spec["rerun"])
    if hit is not None:
        return (spec["T"], spec["label"], hit["best"], "cached")

    k_model, k_train = jax.random.split(jax.random.PRNGKey(1))
    model = build_lm_model(spec["mixer"], cfg.vocab_size, 64, 4, 2, 4, k_model)
    opt = optax.adamw(cfg.learning_rate)
    _, history = train_and_eval(model, cfg, k_train, opt=opt)
    best = max(h["test_acc"] for h in history)
    save({"best": best, "history": history})
    return (spec["T"], spec["label"], best, "fresh")


def _make_specs(rerun: bool):
    specs = []
    for T in TS:
        cfg = dataclasses.replace(CFG, input_seq_len=T)
        cfg_kwargs = dataclasses.asdict(cfg)
        for m in MODEL_SPECS:
            specs.append({
                "label": m["label"], "mixer": m["mixer"],
                "cfg_kwargs": cfg_kwargs, "T": T, "rerun": rerun,
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
            print(f"\n=== {spec['label']}  (T={spec['T']}) ===", flush=True)
            T, label, best, status = _worker(spec)
            print(f"[{status}] {label}  T={T}  best={best:.3f}", flush=True)
            rows.append((T, label, best))
    else:
        ctx = get_context("spawn")
        print(f"running {len(specs)} cells with --parallel {parallel} "
              f"(stdout from workers will interleave)\n", flush=True)
        with ProcessPoolExecutor(max_workers=parallel, mp_context=ctx) as pool:
            futures = {pool.submit(_worker, s): s for s in specs}
            for fut in as_completed(futures):
                T, label, best, status = fut.result()
                print(f"[{status}] {label}  T={T}  best={best:.3f}", flush=True)
                rows.append((T, label, best))

    rows.sort(key=lambda r: (r[0], r[1]))
    print("\n\n## Summary\n")
    print("|    T | model            | best_acc |")
    print("| ---: | ---------------- | -------: |")
    for T, label, best in rows:
        print(f"| {T:>4d} | {label:<16} | {best:>8.3f} |")


if __name__ == "__main__":
    main()
