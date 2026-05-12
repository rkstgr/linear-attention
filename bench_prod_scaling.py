"""Production-scale scaling benchmark: Figures 1 & 2 for Part 3.

Single layer of vanilla linear attention at production head_dim=128, fp32.
Sweeps T ∈ {4K, 8K, 32K, 128K} on CUDA (CPU/MPS work but the story only
lands on A100/H100). One row per (impl, batch, T, mode, C) in the CSV.

Figure 1: parallel + recurrent — show the memory wall (parallel) and the
launch-overhead floor (recurrent).
Figure 2: same plot + chunkwise at C ∈ {64, 512} — show the synthesis.

OOM cells are recorded with status="oom" and empty median_ms so the plotter
can mark them. Recurrent at T ≥ 32K is slow (kernel-launch dominated); use
--skip-large-recurrent to skip if you only want the parallel/chunkwise lines.

    uv run python bench_prod_scaling.py
    uv run python bench_prod_scaling.py --batch-size 4
    uv run python bench_prod_scaling.py --skip-large-recurrent
    uv run python bench_prod_scaling.py --rerun
"""

import argparse
import csv
import gc
import os
import time
import traceback
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cuda")
parser.add_argument("--batch-size", type=int, default=1,
                    help="Batch size. Default 1; tweak to test multi-sample regimes.")
parser.add_argument("--head-dim", type=int, default=128,
                    help="Head dimension (d_k=d_v). 128 is production; 16/64 are toy.")
parser.add_argument("--rerun", action="store_true",
                    help="Re-time cells already present in the CSV.")
parser.add_argument("--skip-large-recurrent", action="store_true",
                    help="Skip recurrent for T >= 32K (kernel-launch dominated, slow).")
parser.add_argument("--n-warmup", type=int, default=3)
parser.add_argument("--n-runs", type=int, default=7)
args = parser.parse_args()

os.environ["JAX_PLATFORMS"] = args.device  # must precede `import jax`

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

# Force true fp32 — on Ampere+ JAX defaults to TF32 which truncates inputs
# to ~10-bit mantissas. The parallel form's big Q @ K^T then disagrees with
# the per-token recurrent by ~1e-2 and breaks the correctness check.
jax.config.update("jax_default_matmul_precision", "highest")


BATCH = args.batch_size
HEAD_DIM = args.head_dim
# Production-scale sweep: pretraining (4K, 8K), finetuning (32K), long-context (128K).
T_SWEEP = [4096, 8192, 32768, 131072]
# Figure 2 chunk sizes — small (C=64) hits dispatch overhead, large (C=512)
# pushes toward parallel-form behavior. Both must divide every T in T_SWEEP.
C_VALUES = [64, 512]
CSV_PATH = Path(__file__).resolve().parent / "bench_prod_scaling.csv"


# --- implementations (all batched: Q, K, V : (B, T, d)) --------------------
# Convention here matches bench_chunkwise.py: S in R^{d_k x d_v},
# S_t = S_{t-1} + k_t v_t^T, o_t = S_t^T q_t.

def parallel_batched(Q, K, V):
    """O = (Q K^T ⊙ M) V — one big batched matmul. O(B T² d_v + B T² d_k)."""
    T = Q.shape[1]
    mask = jnp.tril(jnp.ones((T, T), dtype=Q.dtype))
    scores = jnp.einsum("btd,bsd->bts", Q, K) * mask
    return jnp.einsum("bts,bsd->btd", scores, V)


def _recurrent_single(Q, K, V):
    d = Q.shape[1]
    def step(S, qkv):
        q, k, v = qkv
        S = S + jnp.outer(k, v)  # k_t v_t^T
        o = S.T @ q              # S^T q_t
        return S, o
    S0 = jnp.zeros((d, d), dtype=Q.dtype)
    _, O = jax.lax.scan(step, S0, (Q, K, V))
    return O


def recurrent_batched(Q, K, V):
    return jax.vmap(_recurrent_single)(Q, K, V)


def _chunkwise_single(Q, K, V, C):
    T, d = Q.shape
    n = T // C
    Qc = Q.reshape(n, C, d)
    Kc = K.reshape(n, C, d)
    Vc = V.reshape(n, C, d)
    mask = jnp.tril(jnp.ones((C, C), dtype=Q.dtype))

    def step(S, chunk):
        q, k, v = chunk
        o_inter = q @ S
        o_intra = (q @ k.T * mask) @ v
        S_next = S + k.T @ v
        return S_next, o_inter + o_intra

    S0 = jnp.zeros((d, d), dtype=Q.dtype)
    _, Oc = jax.lax.scan(step, S0, (Qc, Kc, Vc))
    return Oc.reshape(T, d)


def make_chunkwise_batched(C):
    def fn(Q, K, V):
        return jax.vmap(lambda q, k, v: _chunkwise_single(q, k, v, C))(Q, K, V)
    fn.__name__ = f"chunkwise_C{C}"
    return fn


# --- theoretical memory (independent of measurement noise) ------------------

def theoretical_peak_mb(impl, B, T, d, C=None, mode="fwd"):
    """Back-of-envelope peak live tensors in fp32. Not counting JAX overhead.
    Useful as a sanity check against the measured peak."""
    bpf = 4  # fp32
    qkv = 3 * B * T * d * bpf  # Q, K, V inputs
    out = B * T * d * bpf       # output

    if impl == "parallel":
        scores = B * T * T * bpf
        body = qkv + out + scores
    elif impl == "recurrent":
        state = B * d * d * bpf
        body = qkv + out + state
    elif impl == "chunkwise":
        state = B * d * d * bpf
        chunk_scores = B * C * C * bpf
        body = qkv + out + state + chunk_scores
    else:
        raise ValueError(impl)

    # Backward roughly doubles activations (T-sized tensors must be saved).
    if mode == "bwd":
        body += B * T * d * bpf * 2

    return body / (1024 ** 2)


# --- correctness check ------------------------------------------------------

def correctness_check():
    B, T, d = 1, 256, HEAD_DIM
    C = 64
    kQ, kK, kV = jax.random.split(jax.random.PRNGKey(0), 3)
    Q = jax.random.normal(kQ, (B, T, d))
    K = jax.random.normal(kK, (B, T, d))
    V = jax.random.normal(kV, (B, T, d))

    o_par = jax.block_until_ready(parallel_batched(Q, K, V))
    o_rec = jax.block_until_ready(recurrent_batched(Q, K, V))
    o_chk = jax.block_until_ready(make_chunkwise_batched(C)(Q, K, V))

    d_rec = float(jnp.max(jnp.abs(o_par - o_rec)))
    d_chk = float(jnp.max(jnp.abs(o_par - o_chk)))
    # Looser tolerance at d=128 than the d=16 toy benchmark: more fp accumulation.
    tol = 1e-3
    if d_rec > tol or d_chk > tol:
        raise AssertionError(
            f"correctness failed at T={T}, d={d}: "
            f"|par-rec|={d_rec:.2e}, |par-chk|={d_chk:.2e}"
        )
    print(f"correctness OK (B={B}, T={T}, d={d}): "
          f"|par-rec|={d_rec:.2e}, |par-chk|={d_chk:.2e}")


# --- memory & OOM utilities -------------------------------------------------

def memory_peak_mb():
    """Peak device bytes in MB. JAX's peak is monotonic per-process; use this
    as a high-water mark and pair it with theoretical_peak_mb for the figure."""
    try:
        stats = jax.devices()[0].memory_stats()
        peak = stats.get("peak_bytes_in_use")
        return peak / (1024 ** 2) if peak is not None else None
    except Exception:
        return None


def is_oom(exc):
    msg = str(exc).lower()
    needles = (
        "out of memory",
        "oom",
        "resource_exhausted",
        "ran out of memory",
        "memory exhausted",
        "cuda_error_out_of_memory",
    )
    return any(s in msg for s in needles)


def time_one(fn, fn_args, n_warmup, n_runs):
    """Returns (median_ms, peak_mb). Caller catches OOM."""
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*fn_args))
    samples = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*fn_args))
        samples.append((time.perf_counter() - t0) * 1000)
    return float(np.median(samples)), memory_peak_mb()


def make_fn(impl, mode, C):
    if impl == "parallel":
        base = parallel_batched
    elif impl == "recurrent":
        base = recurrent_batched
    elif impl == "chunkwise":
        base = make_chunkwise_batched(C)
    else:
        raise ValueError(impl)
    if mode == "bwd":
        fn = jax.grad(lambda Q, K, V: base(Q, K, V).sum(), argnums=(0, 1, 2))
    else:
        fn = base
    return jax.jit(fn)


# --- csv cache --------------------------------------------------------------

FIELDS = [
    "impl", "device", "batch", "T", "mode", "C",
    "median_ms", "peak_mb", "theoretical_mb",
    "tokens_per_s", "status",
]


def row_key(r):
    c = r["C"]
    return (
        r["impl"],
        r["device"],
        int(r["batch"]),
        int(r["T"]),
        r["mode"],
        int(c) if c not in ("", None) else None,
    )


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


# --- run plan ---------------------------------------------------------------

def t_sweep_plan(device, batch):
    cells = []
    # Run smallest T first so JAX's monotonic peak gives the most accurate
    # reading for each successive cell.
    for T in T_SWEEP:
        for mode in ("fwd", "bwd"):
            cells.append({"impl": "parallel", "device": device, "batch": batch,
                          "T": T, "mode": mode, "C": None})
            if not (args.skip_large_recurrent and T >= 32768):
                cells.append({"impl": "recurrent", "device": device, "batch": batch,
                              "T": T, "mode": mode, "C": None})
            for C in C_VALUES:
                if T % C != 0:
                    continue
                cells.append({"impl": "chunkwise", "device": device, "batch": batch,
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
    print(f"device: {requested} ({actual})  jax {jax.__version__}  "
          f"batch={BATCH}  head_dim={HEAD_DIM}")


def main():
    assert_device(args.device)
    correctness_check()

    rows = load_rows()
    present = {row_key(r) for r in rows}
    plan = t_sweep_plan(args.device, BATCH)

    n_done = n_skip = n_oom = 0
    for cell in plan:
        key = (cell["impl"], cell["device"], cell["batch"],
               cell["T"], cell["mode"], cell["C"])
        if key in present and not args.rerun:
            n_skip += 1
            continue

        T = cell["T"]
        theo_mb = theoretical_peak_mb(
            cell["impl"], cell["batch"], T, HEAD_DIM, cell["C"], cell["mode"]
        )

        try:
            kQ, kK, kV = jax.random.split(jax.random.PRNGKey(T * cell["batch"]), 3)
            Q = jax.random.normal(kQ, (cell["batch"], T, HEAD_DIM))
            K = jax.random.normal(kK, (cell["batch"], T, HEAD_DIM))
            V = jax.random.normal(kV, (cell["batch"], T, HEAD_DIM))
            fn = make_fn(cell["impl"], cell["mode"], cell["C"])
            ms, peak_mb = time_one(fn, (Q, K, V), args.n_warmup, args.n_runs)
            status = "ok"
        except Exception as exc:
            if is_oom(exc):
                ms, peak_mb, status = None, None, "oom"
                n_oom += 1
                # Free anything we managed to allocate before the OOM.
                gc.collect()
            else:
                print(f"  ERROR on {cell}: {exc}")
                traceback.print_exc()
                raise

        row = {
            "impl": cell["impl"],
            "device": cell["device"],
            "batch": str(cell["batch"]),
            "T": str(cell["T"]),
            "mode": cell["mode"],
            "C": str(cell["C"]) if cell["C"] is not None else "",
            "median_ms": f"{ms:.4f}" if ms is not None else "",
            "peak_mb": f"{peak_mb:.1f}" if peak_mb is not None else "",
            "theoretical_mb": f"{theo_mb:.1f}",
            "tokens_per_s": (
                f"{T * cell['batch'] / (ms / 1000):.1f}" if ms is not None else ""
            ),
            "status": status,
        }
        rows = [r for r in rows if row_key(r) != key]
        rows.append(row)
        save_rows(rows)
        present.add(key)
        n_done += 1

        c_str = f"{cell['C']:>4d}" if cell["C"] is not None else "    "
        ms_str = f"{ms:>10.3f} ms" if ms is not None else "       OOM"
        mem_str = f"{peak_mb:>7.0f} MB" if peak_mb is not None else "      —"
        theo_str = f"~{theo_mb:>7.0f} MB theo"
        print(f"  {cell['impl']:>10} B={cell['batch']} T={T:>6} "
              f"mode={cell['mode']:>3} C={c_str} -> {ms_str}  "
              f"peak {mem_str}  {theo_str}")

    print(f"\ndone: {n_done} runs ({n_oom} OOM), {n_skip} cached. CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
