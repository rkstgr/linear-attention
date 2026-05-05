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

## Setup

```
uv sync
```

**Apple Silicon GPU.** The project auto-selects JAX's Metal backend
(`jax-mps` plugin) on macOS arm64 — no env vars needed. Step time is ~3.3×
faster than CPU on an M-series chip. To force CPU (e.g. for debugging or
comparing numbers), set `JAX_PLATFORMS=cpu`.
**Caveat**: Many small kernels that need to be launched (deltanet) leads to slower perf than using *cpu*.

## Run

```
uv run python transformer.py          # softmax baseline (level1)
uv run python linear_attention.py     # linear attention
uv run python deltanet.py             # plain DeltaNet
uv run python gated_deltanet.py       # DeltaNet with state-decay gate
```

Pass `--level 0` for fast dev (vocab=256, ~30s/run), `--level 1` (default) for
the Zoology-comparable config (vocab=8192, slower but matches published curves).

```
uv run python transformer.py --level 0
uv run python deltanet.py --level 0 --gated
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

Every model file is `imports + Mixer class + Block + Transformer + __main__`.
The only line that meaningfully changes between files is `self.attn = ...`.

## Configs

- `data.level0` — small-vocab dev config (vocab=256, n_train=50k). Use for
  quick correctness checks.
- `data.level1` — Zoology's smallest training config from
  `original_mqar_configs.py` (vocab=8192, n_train=100k). Numbers from this
  run compare directly to published Zoology curves.

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
