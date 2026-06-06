"""Minimal experiment: capacity ceiling separates LinAttn from DeltaNet.

Mechanism: LA's additive write `S += v k^T` saturates the rank-d_k state once
N_KV exceeds head_dim. DN's delta rule overwrites along the key direction and
keeps working past the ceiling.

Two cells per model — one well below capacity, one well above. Below is the
sanity check (both should solve); above is the demonstration (LA fails, DN
holds).

Held constant: vocab=256, T=128, dim=64, head_dim=16, lr=3e-3, n_train=20k,
max_epochs=16, batch=64. ~5 minutes on CPU.

Cells run through `executor.py`, which caches each step under
`.experiment_cache/steps/` and writes launch manifests under
`.experiment_cache/runs/`. Pass `--rerun` to force recomputation. Pass
`--parallel N` to run up to N cells concurrently (defaults to 1 = serial).
Pair `--parallel 4` with running both experiment scripts at once for 8-way
total concurrency on an 8+-core Mac.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

import argparse
import dataclasses
import json
from pathlib import Path

from linattn.data import level1
from linattn.executor import executor_main
from experiments.defaults import (
    ModelConfig,
    RunConfig,
    config_from_data_config,
    default_train,
)


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

ARCH_KWARGS = {"dim": 64, "n_heads": 4, "n_layers": 2, "mlp_mult": 4}


def _make_specs():
    specs = []
    for n_kv in N_KVS:
        cfg = dataclasses.replace(CFG, num_kv_pairs=n_kv)
        for m in MODEL_SPECS:
            specs.append({
                "label": m["label"], "mixer": m["mixer"],
                "cfg": cfg, "n_kv": n_kv,
            })
    return specs


def _make_step(spec):
    data_cfg, train_cfg = config_from_data_config(spec["cfg"])
    model_cfg = ModelConfig(
        mixer=spec["mixer"],
        vocab_size=spec["cfg"].vocab_size,
        **ARCH_KWARGS,
    )
    run = RunConfig(model=model_cfg, data=data_cfg, train=train_cfg, seed=1)
    return default_train(f"capacity/{spec['mixer']}/nkv{spec['n_kv']}", run)


def _read_best(output_path: str) -> float:
    metrics = json.loads((Path(output_path) / "metrics.json").read_text(encoding="utf-8"))
    return float(metrics["best"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()

    specs = _make_specs()
    steps = [_make_step(spec) for spec in specs]
    results = executor_main(
        steps,
        parallel=args.parallel,
        rerun=args.rerun,
        experiment_name="capacity",
    )

    rows = []
    for spec, result in zip(specs, results):
        best = _read_best(result.output_path)
        print(
            f"[{result.cache_status}] {spec['label']}  "
            f"N_KV={spec['n_kv']}  best={best:.3f}",
            flush=True,
        )
        rows.append((spec["n_kv"], spec["label"], best))

    rows.sort(key=lambda r: (r[0], r[1]))
    print("\n\n## Summary\n")
    print("| N_KV | model            | best_acc |")
    print("| ---: | ---------------- | -------: |")
    for n_kv, label, best in rows:
        print(f"| {n_kv:>4d} | {label:<16} | {best:>8.3f} |")


if __name__ == "__main__":
    main()
