# Code scratchpad

One file per model. No framework, no configs. Every file has a `__main__`
that trains on a toy task in under a minute on CPU.

## Setup

```
uv sync
```

## Run

```
uv run python transformer.py
```

## Layout

- `modern.py` — shared building blocks: RMSNorm, SwiGLU, RoPE.
- `data.py` — toy datasets (currently: MQAR).
- `transformer.py` — decoder-only softmax transformer (baseline).

Shape conventions used in every file:

```
T  = sequence length
D  = model dim
H  = number of heads
Dh = head dim  (= D / H)
V  = vocab size
```
