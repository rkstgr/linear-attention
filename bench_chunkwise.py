"""Scaling benchmark: parallel vs recurrent vs chunkwise linear attention.

Single-head, batch=1, head_dim=16, fp32. One figure for the blog post —
pedagogical clarity, not benchmark rigor.

Run once per device (the JAX MPS / CUDA plugins must be selected at import time):

    uv run python bench_chunkwise.py --device cpu
    uv run python bench_chunkwise.py --device mps
    uv run python bench_chunkwise.py --device cuda           # if available
    uv run python bench_chunkwise.py --device cpu --c-sweep  # C sweep at T=2048
    uv run python bench_chunkwise.py --device cpu --rerun    # force re-time

Results accumulate in `bench_chunkwise.csv` (one row per impl/device/T/mode/C).
Existing rows are skipped on re-run unless --rerun.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
parser.add_argument("--c-sweep", action="store_true")
parser.add_argument("--rerun", action="store_true")
args = parser.parse_args()

os.environ["JAX_PLATFORMS"] = args.device  # must precede `import jax`

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

# Force true fp32 matmuls. On Ampere+ (A100, H100) JAX defaults to TF32
# which truncates inputs to ~10-bit mantissas — the parallel form's big
# Q @ K.T then disagrees with the per-token recurrent by ~1e-2, breaking
# the correctness check and giving parallel an unfair speed edge.
jax.config.update("jax_default_matmul_precision", "highest")


HEAD_DIM = 16
# At head_dim=16 the parallel form is so cheap that on a modern GPU the
# T x T attention matrix stays in cache up to T~4096 — chunkwise can't beat
# it there. Pushing T to 8192/16384 makes the mask big enough to strain HBM
# (256 MB / 1 GB), which is where parallel finally loses to chunkwise.
T_SWEEP = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
C_FIXED = 64
# Two C-sweeps so the chunk-size sweet spot is visible on dispatch-light
# accelerators too: small T leans on intra-chunk T² waste, big T amplifies
# dispatch overhead at small C and shifts the optimum upward.
C_SWEEPS = [
    (2048, [16, 32, 64, 128, 256]),
    (8192, [32, 64, 128, 256, 512, 1024]),
]
CSV_PATH = Path(__file__).resolve().parent / "bench_chunkwise.csv"


# --- implementations --------------------------------------------------------
# Recurrence: S_t = S_{t-1} + k_t v_t^T, o_t = S_t^T q_t.

def parallel(Q, K, V):
    """O = (Q K^T ⊙ M) V — matmul-only, O(T²·d) compute / O(T²) memory."""
    T = Q.shape[0]
    mask = jnp.tril(jnp.ones((T, T), dtype=Q.dtype))
    return (Q @ K.T * mask) @ V


def recurrent(Q, K, V):
    """Per-token scan with carry S : (d, d). O(T·d²), no quadratic memory."""
    d = Q.shape[1]

    def step(S, qkv):
        q, k, v = qkv
        S = S + jnp.outer(k, v)  # k_t v_t^T
        o = S.T @ q              # o_t = S_t^T q_t
        return S, o

    S0 = jnp.zeros((d, d), dtype=Q.dtype)
    _, O = jax.lax.scan(step, S0, (Q, K, V))
    return O


def chunkwise_impl(Q, K, V, C):
    """Scan over chunks of size C; parallel form within, recurrent carry between."""
    T, d = Q.shape
    n = T // C
    Qc = Q.reshape(n, C, d)
    Kc = K.reshape(n, C, d)
    Vc = V.reshape(n, C, d)
    mask = jnp.tril(jnp.ones((C, C), dtype=Q.dtype))

    def step(S, chunk):
        q, k, v = chunk
        o_inter = q @ S                  # carry contribution: (C, d)
        o_intra = (q @ k.T * mask) @ v   # local parallel form: (C, d)
        S_next = S + k.T @ v             # update carry: (d, d)
        return S_next, o_inter + o_intra

    S0 = jnp.zeros((d, d), dtype=Q.dtype)
    _, Oc = jax.lax.scan(step, S0, (Qc, Kc, Vc))
    return Oc.reshape(T, d)


def make_chunkwise(C):
    def fn(Q, K, V):
        return chunkwise_impl(Q, K, V, C)
    fn.__name__ = f"chunkwise_C{C}"
    return fn


# --- correctness ------------------------------------------------------------

def correctness_check():
    T, C = 64, 16
    kQ, kK, kV = jax.random.split(jax.random.PRNGKey(0), 3)
    Q = jax.random.normal(kQ, (T, HEAD_DIM))
    K = jax.random.normal(kK, (T, HEAD_DIM))
    V = jax.random.normal(kV, (T, HEAD_DIM))

    o_par = jax.block_until_ready(parallel(Q, K, V))
    o_rec = jax.block_until_ready(recurrent(Q, K, V))
    o_chk = jax.block_until_ready(chunkwise_impl(Q, K, V, C))

    d_rec = float(jnp.max(jnp.abs(o_par - o_rec)))
    d_chk = float(jnp.max(jnp.abs(o_par - o_chk)))
    if d_rec > 1e-4 or d_chk > 1e-4:
        raise AssertionError(
            f"correctness failed at T={T}: "
            f"|parallel-recurrent|={d_rec:.2e}, |parallel-chunkwise|={d_chk:.2e}"
        )
    print(f"correctness OK at T={T}: "
          f"|parallel-recurrent|={d_rec:.2e}, |parallel-chunkwise|={d_chk:.2e}")


# --- timing -----------------------------------------------------------------

def time_one(fn, fn_args, n_warmup=3, n_runs=7):
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*fn_args))
    samples = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*fn_args))
        samples.append((time.perf_counter() - t0) * 1000)
    return float(np.median(samples))


def make_fn(impl, mode, C):
    if impl == "parallel":
        base = parallel
    elif impl == "recurrent":
        base = recurrent
    elif impl == "chunkwise":
        base = make_chunkwise(C)
    else:
        raise ValueError(impl)
    if mode == "bwd":
        fn = jax.grad(lambda Q, K, V: base(Q, K, V).sum(), argnums=(0, 1, 2))
    else:
        fn = base
    return jax.jit(fn)


# --- csv cache --------------------------------------------------------------

FIELDS = ["impl", "device", "T", "mode", "C", "median_ms", "tokens_per_s"]


def row_key(r):
    c = r["C"]
    return (r["impl"], r["device"], int(r["T"]), r["mode"],
            int(c) if c not in ("", None) else None)


def load_rows():
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


def save_rows(rows):
    tmp = CSV_PATH.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    tmp.replace(CSV_PATH)


# --- run plans --------------------------------------------------------------

def t_sweep_plan(device):
    cells = []
    for T in T_SWEEP:
        for mode in ("fwd", "bwd"):
            for impl in ("parallel", "recurrent", "chunkwise"):
                c = C_FIXED if impl == "chunkwise" else None
                cells.append({"impl": impl, "device": device, "T": T, "mode": mode, "C": c})
    return cells


def c_sweep_plan(device):
    cells = []
    for T, Cs in C_SWEEPS:
        for C in Cs:
            for mode in ("fwd", "bwd"):
                cells.append({"impl": "chunkwise", "device": device,
                              "T": T, "mode": mode, "C": C})
    return cells


# --- main -------------------------------------------------------------------

DEVICE_ALIASES = {"cpu": {"cpu"}, "mps": {"metal", "mps"}, "cuda": {"gpu", "cuda"}}


def assert_device(requested):
    actual = jax.devices()[0].platform.lower()
    if actual not in DEVICE_ALIASES[requested]:
        raise RuntimeError(
            f"requested device={requested!r} but jax.devices()[0].platform={actual!r}; "
            f"is the {requested} plugin installed?"
        )
    print(f"device: {requested} ({actual})  jax {jax.__version__}")


def main():
    assert_device(args.device)
    correctness_check()

    rows = load_rows()
    present = {row_key(r) for r in rows}
    plan = c_sweep_plan(args.device) if args.c_sweep else t_sweep_plan(args.device)

    n_done = n_skip = 0
    for cell in plan:
        key = (cell["impl"], cell["device"], cell["T"], cell["mode"], cell["C"])
        if key in present and not args.rerun:
            n_skip += 1
            continue

        T = cell["T"]
        kQ, kK, kV = jax.random.split(jax.random.PRNGKey(T), 3)
        Q = jax.random.normal(kQ, (T, HEAD_DIM))
        K = jax.random.normal(kK, (T, HEAD_DIM))
        V = jax.random.normal(kV, (T, HEAD_DIM))
        fn = make_fn(cell["impl"], cell["mode"], cell["C"])
        ms = time_one(fn, (Q, K, V))

        row = {
            "impl": cell["impl"],
            "device": cell["device"],
            "T": str(cell["T"]),
            "mode": cell["mode"],
            "C": str(cell["C"]) if cell["C"] is not None else "",
            "median_ms": f"{ms:.4f}",
            "tokens_per_s": f"{T / (ms / 1000):.1f}",
        }
        rows = [r for r in rows if row_key(r) != key]
        rows.append(row)
        save_rows(rows)
        present.add(key)
        n_done += 1
        c_str = f"{cell['C']:>4d}" if cell["C"] is not None else "    "
        print(f"  {cell['impl']:>10} {cell['device']:>4} "
              f"T={T:>4} mode={cell['mode']:>3} C={c_str} -> {ms:8.3f} ms")

    print(f"\ndone: {n_done} timed, {n_skip} cached. CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
