# Code scratchpad

Minimal implementations of softmax attention / linear attention / DeltaNet
/ Gated DeltaNet on MQAR (Multi-Query Associative Recall, Zoology 2023) for
the linear-attention post.

One file per architecture. The only line that meaningfully changes between
them is `self.attn = ...`. Everything else (projections, RoPE, training
loop, data) is shared.

## What separates the four models

Two minimal experiments, each isolating one axis of variation. Run on CPU
in ~5 minutes:

```
uv run python experiment_capacity.py                  # serial, default
uv run python experiment_retention.py --rerun         # recompute everything
uv run python experiment_capacity.py --parallel 4     # 4 cells concurrent
```

Cells are cached by (config + bytes of dependent source files) under
`.experiment_cache/`. Re-running the script after editing one model file
recomputes only that model's cells; everything else loads from cache.
Delete `.experiment_cache/` to clear.

`--parallel N` dispatches up to N cells across spawn'd worker processes
(each with its own JAX init). Pair with two terminals to run both
experiments concurrently for up to 8-way parallelism on an 8+-core Mac:

```
# terminal A
OMP_NUM_THREADS=2 uv run python experiment_capacity.py --parallel 4
# terminal B
OMP_NUM_THREADS=2 uv run python experiment_retention.py --parallel 4
```

Worker stdout interleaves on the terminal in parallel mode — the parent
prints one `[cached]` / `[fresh]` line per cell as it completes, plus a
sorted summary at the end.

### Capacity ceiling — LinAttn vs DeltaNet

**Claim.** LA's additive write `S += v k^T` saturates the rank-`d_k` state
once `N_KV` exceeds `head_dim`. DN's delta rule overwrites along the key
direction and keeps working past the ceiling.

**Setup.** vocab=256, T=128, dim=64, head_dim=16, lr=3e-3, n_train=20k,
max_epochs=16. One cell per (model, N_KV) — `N_KV ∈ {4, 32}` brackets the
ceiling at 16.

| N_KV | model            | best_acc |
| ---: | ---------------- | -------: |
|    4 | Linear Attention |    0.966 |
|    4 | DeltaNet         |    0.994 |
|   32 | Linear Attention |    0.049 |
|   32 | DeltaNet         |    0.771 |

Below the ceiling, both solve. At 2× the ceiling, LA collapses to roughly
1/N_KV = 1/32 ≈ 0.03 (random pick from the value set) while DN degrades
gracefully to 0.77 — the JL soft-cap past `d_k`, not a hard cliff.

### Streaming noise — DeltaNet vs Gated DeltaNet

**Claim.** Every non-write token still passes through `W_k` and writes a
rank-1 perturbation into the state. As T grows at fixed N_KV, this noise
accumulates. Gating's `α < 1` decays it; plain DN has no mechanism to.

**Setup.** vocab=1024, N_KV=4 (well below the head_dim=16 ceiling so
capacity isn't the variable), dim=64, lr=3e-3, n_train=20k, max_epochs=16.
One cell per (model, T) — `T ∈ {64, 512}`. Gate parameterised Mamba2-style
as `α = exp(-softplus(dt_logit) · σ(x·W_α + b_α))` with `dt_logit` init
−10 → α ≈ 1 at step 0. α is bounded in (0, 1] by construction so the
recurrence is NaN-stable even when dt grows during training.

|    T | model            | best_acc |
| ---: | ---------------- | -------: |
|   64 | DeltaNet         |    0.985 |
|   64 | Gated DeltaNet   |    0.984 |
|  512 | DeltaNet         |    0.909 |
|  512 | Gated DeltaNet   |    0.939 |

At T=64 both solve (gap inside seed noise — GDN matches DN, gate barely
activates because there's no streaming noise to decay). At T=512 the
recurrence absorbs ~500 noise tokens between writes and queries; DN drops
to 0.91, GDN holds at 0.94. The +0.03 gap at T=512 is the gating mechanism
doing what it's there for.

_The gap is dampened by the bounded-init parameterisation: `dt_logit`
inits at −10 (so α ≈ 1 ⇒ GDN starts equivalent to DN), but `softplus`
saturates at very negative inputs so the gradient pulling `dt_logit`
toward 0 is small early in training. A less aggressive init (e.g. −3)
would activate gating faster at the cost of starting α slightly below 1.
Tradeoff worth a follow-up if the gap matters more than NaN-stability._

## Setup

```
uv sync                  # default — CPU only
uv sync --group cuda     # + NVIDIA GPU (Linux x86_64)
```

CPU is the default JAX backend (set in `data.py`). The recurrence-form
DeltaNet is per-token sequential and runs faster on CPU than on MPS or
small CUDA matmuls — kernel-launch overhead dominates on accelerators
for tiny ops. To opt into GPU explicitly:

```
JAX_PLATFORMS=cuda,cpu uv run python <script>.py
```

## Run a single model

Each model file is `imports + Mixer + Block + Transformer + __main__`,
self-contained, trains on `level1` (vocab=8192, T=64, N_KV=4 — Zoology's
easiest config) and prints predictions on a held-out example.

```
uv run python transformer.py
uv run python linear_attention.py
uv run python deltanet.py            # plain DeltaNet
uv run python deltanet.py --gated    # Gated DeltaNet
```

## Layout

- `data.py` — MQAR generator + `level1` Config.
- `train.py` — shared training loop with early stopping.
- `utils.py` — RMSNorm, SwiGLU, RoPE.
- `transformer.py` / `linear_attention.py` / `deltanet.py` — one mixer each.
- `experiment_capacity.py` / `experiment_retention.py` — minimal sweeps.
- `cache.py` — content-addressed cache for experiment cells.

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
