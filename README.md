# Code scratchpad

One file per model. No framework, no configs beyond `data.level1`. Every file
has a `__main__` that trains on MQAR and prints predictions on a held-out
example.

## Results
Solved means >99% test acc.
### Level 0
Attention
Linear Attention: solved in x epochs. xms per step 
Deltanet: solved in 3 epochs, 60ms per step.
Gated Deltanet:

### Level 1

| model             | epoch | train_loss | train_acc | test_acc |
| ----------------- | ----: | ---------: | --------: | -------: |
| Transformer       |     2 |     1.3187 |     0.795 |    0.993 |
| Deltanet          |    31 |     0.0050 |     0.999 |    0.983 |
| Gated Deltanet    |    31 |     0.0047 |     0.999 |    0.976 |
| Linear Attention  |    21 |     0.0062 |     0.998 |    0.971 |

## Setup

```
uv sync                  # CPU + Apple Silicon (Metal)
uv sync --group cuda     # + NVIDIA GPU (Linux x86_64)
```

**Apple Silicon GPU.** The project auto-selects JAX's Metal backend
(`jax-mps` plugin) on macOS arm64 — no env vars needed. Step time is ~3.3×
faster than CPU on an M-series chip.
**Caveat**: Many small kernels that need to be launched (deltanet) leads to slower perf than using *cpu*.

**NVIDIA GPU.** Opt-in dependency group (the CUDA wheels are ~2 GB, so
they're not in the default install). After `uv sync --group cuda`,
`data.py` sets `JAX_PLATFORMS=cuda,cpu` so JAX prefers the GPU and silently
falls back to CPU if no usable device is present.

To force CPU on either platform (e.g. for debugging or comparing numbers),
set `JAX_PLATFORMS=cpu`.

## Run

```
uv run python transformer.py          # softmax baseline (level1)
uv run python linear_attention.py     # linear attention
uv run python deltanet.py             # plain DeltaNet
uv run python gated_deltanet.py       # DeltaNet with state-decay gate
```

Levels: `--level 0` (vocab=256 dev, ~30s/run), `--level 1` (default,
Zoology-comparable: vocab=8192, T=64, N_KV=4), `--level easy` (T=128, N_KV=8),
`--level medium` (T=512, N_KV=64), `--level hard` (T=1024, N_KV=128).

```
uv run python transformer.py --level 0
uv run python deltanet.py --level easy --gated
```

To run all four models on the same level and get a comparison table:

```
uv run python benchmark.py --level medium
uv run python benchmark.py --level easy --models transformer,deltanet
```

## Layout

- `utils.py` — shared building blocks: RMSNorm, SwiGLU, RoPE.
- `data.py` — MQAR data generator. Defines `Config` and `level1` (Zoology's
  easiest difficulty: vocab=8192, seq=64, N_KV=4, power-law gaps).
- `train.py` — shared training loop with early stopping. Pre-generates a
  fixed train/test split per `cfg`, runs AdamW, evaluates each epoch, stops
  when test accuracy crosses `cfg.target_acc` or stalls.
- `transformer.py` — decoder-only softmax transformer (baseline).
- `linear_attention.py` — Katharopoulos et al. 2020.
- `deltanet.py` — Yang et al. 2024.
- `gated_deltanet.py` — wrapper that runs `deltanet.py --gated`.
- `benchmark.py` — train all four models on one level, print a summary table.

Every model file is `imports + Mixer class + Block + Transformer + __main__`.
The only line that meaningfully changes between files is `self.attn = ...`.

## Configs

- `data.level0` — small-vocab dev config (vocab=256, n_train=50k). Use for
  quick correctness checks.
- `data.level1` — Zoology's smallest training config from
  `original_mqar_configs.py` (vocab=8192, n_train=100k). Numbers from this
  run compare directly to published Zoology curves.
- `data.easy` / `data.medium` / `data.hard` — fixed-vocab (8192) difficulty
  ladder for benchmarking. `(N_KV, T) = (8, 128)` / `(64, 512)` / `(128, 1024)`.
  Easy: softmax wins; sub-quadratic models keep up. Medium: DeltaNet/Gated
  start trailing softmax. Hard: continuation of the progression.

### Training schedule

Following Zoology's `original_mqar_configs.py`, `max_epochs=32` and
`patience_epochs=5` are held constant across the difficulty ladder. Harder
tasks scale `n_train` *down* (100k → 40k → 20k for easy/medium/hard) rather
than scaling training time up — keeps per-config compute roughly bounded so
results are comparable to published Zoology curves. Zoology also sweeps
learning rate over `np.logspace(-3, -1.5, 4)` per arch+difficulty; we use a
single `lr=1e-3` to keep the benchmark minimal.

Sources:
- [HazyResearch/zoology](https://github.com/HazyResearch/zoology)
- [zoology/usage.md](https://github.com/HazyResearch/zoology/blob/main/usage.md)
- [Zoology paper (arXiv 2312.04927)](https://arxiv.org/html/2312.04927v1)
- [Zoology blogpost — Measuring and Improving Recall](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis)

To run a different difficulty, build your own `Config` (e.g.
`dataclasses.replace(level1, num_kv_pairs=16, input_seq_len=256)`) and pass it
to `train_and_eval`.

## Shape conventions

```
T  = sequence length
D  = model dim
H  = number of heads
Dh = head dim  (= D / H)
V  = vocab size
B  = batch size
```
