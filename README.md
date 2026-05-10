# Code scratchpad

Minimal implementations of softmax / linear attention / DeltaNet / Gated
DeltaNet on MQAR for the linear-attention post. One file per architecture;
the only line that changes is `self.attn = ...`.

## Two minimal experiments

Each isolates one axis. ~5 min each on CPU.

```
uv run python experiment_capacity.py
uv run python experiment_retention.py
```

Cells are cached by (config + source-file bytes) under `.experiment_cache/`.
Edit one model file → only its cells recompute. `--rerun` forces, `--parallel N`
dispatches N cells across worker processes. Pair with two terminals
(`OMP_NUM_THREADS=2 ... --parallel 4` each) for 8-way concurrency.

### Capacity ceiling — LinAttn vs DeltaNet

LA's additive `S += v k^T` saturates at `N_KV ~ d_k`. DN's delta rule
overwrites in the key direction and degrades past the ceiling rather than
collapsing.

vocab=256, T=128, dim=64, head_dim=16, lr=3e-3, n_train=20k. `N_KV ∈ {4, 32}`
brackets the ceiling.

| N_KV | model            | best_acc |
| ---: | ---------------- | -------: |
|    4 | Linear Attention |    0.966 |
|    4 | DeltaNet         |    0.994 |
|   32 | Linear Attention |    0.049 |
|   32 | DeltaNet         |    0.771 |

Below the ceiling: both solve. At 2×: LA collapses to ~1/N_KV (random
pick from the value set), DN holds at 0.77 — the JL soft-cap past `d_k`,
not a hard cliff.

### Streaming noise — DeltaNet vs Gated DeltaNet

Every non-write token still writes a rank-1 perturbation through `W_k`.
At fixed N_KV, this noise accumulates with T. Gating's `α < 1` decays it;
plain DN can't.

vocab=1024, N_KV=4 (well below the ceiling), dim=64, lr=3e-3, n_train=20k.
`T ∈ {64, 512}`. Gate parameterised Mamba2-style:
`α = exp(-softplus(dt_logit) · σ(x·W_α + b_α))` with `dt_logit` init −10
so α ≈ 1 at step 0 and stays in (0, 1].

|    T | model            | best_acc |
| ---: | ---------------- | -------: |
|   64 | DeltaNet         |    0.985 |
|   64 | Gated DeltaNet   |    0.984 |
|  512 | DeltaNet         |    0.909 |
|  512 | Gated DeltaNet   |    0.939 |

T=64: tied — no streaming noise to decay. T=512: GDN +0.03. The bounded
init keeps the gap small — `softplus` saturates at very negative input,
so `dt_logit` moves off −10 slowly. Less aggressive init (e.g. −3) would
trade `α ≈ 1` fidelity for faster activation.

### Scaling — parallel vs recurrent vs chunkwise

Three single-head impls of `S_t = S_{t-1} + k_t v_t^T` (no delta, no gating):
parallel `(Q K^T ⊙ M) V`, per-token `lax.scan`, and chunkwise (scan over
chunks of size C, parallel form within). fp32, head_dim=16, batch=1.

```
uv run python bench_chunkwise.py --device {cpu,mps,cuda}
uv run python bench_chunkwise.py --device {cpu,mps,cuda} --c-sweep
uv run python bench_chunkwise_plot.py
```

**MacBook Pro M5** (10-core, 16 GB)

![M5 scaling](figures/bench_chunkwise.png)

![M5 C sweet spot](figures/bench_chunkwise_C_sweep.png)

**A100** (40 GB, single GPU)

![A100 scaling](figures/a100/bench_chunkwise.png)

![A100 C sweet spot](figures/a100/bench_chunkwise_C_sweep.png)


## Setup

```
uv sync                  # CPU only
uv sync --group cuda     # + NVIDIA GPU (Linux x86_64)
```

## Run a single model

```
uv run python transformer.py
uv run python linear_attention.py
uv run python deltanet.py            # plain
uv run python deltanet.py --gated    # gated
```

Each trains on `level1` (vocab=8192, T=64, N_KV=4 — Zoology's easiest)
and prints predictions on a held-out example.

## Layout

- `data.py` — MQAR generator + `level1` Config.
- `train.py` — shared training loop.
- `utils.py` — RMSNorm, SwiGLU, RoPE.
- `transformer.py` / `linear_attention.py` / `deltanet.py` — one mixer each.
- `experiment_capacity.py` / `experiment_retention.py` — minimal sweeps.
- `cache.py` — content-addressed cell cache.

## Shape conventions

```
T  = sequence length
D  = model dim
H  = number of heads
Dh = head dim  (= D / H)
V  = vocab size
B  = batch size
```

## Sources

- [HazyResearch/zoology](https://github.com/HazyResearch/zoology)
- [Zoology paper (arXiv 2312.04927)](https://arxiv.org/html/2312.04927v1)
- [Zoology blogpost — Measuring and Improving Recall](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis)
