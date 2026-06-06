# Proposal 0001: parity-gated scaffold extraction

> Foundational infra — does **not** answer a [DIRECTION.md](../../DIRECTION.md)
> question directly. The **parity equality** (params + forward outputs identical
> before/after) is the claim.

Path: **Full** (Sprinter → Skeptic → settle), even though the change is
mechanical — it carries real design decisions and a behavior-lock gate worth a
Skeptic pass.

---

## Goal

Extract the duplicated `Block` + LM wrapper into one **mixer-agnostic backbone**
plus **mixer/FFN registries**, so each arch file is *only its mixer* — with **no
change** to any model's parameters or forward outputs.

## Non-goals

- No new task, metric, sweep, or measurement.
- No executor / redun work (0002); no `Task` / `Config` split (0003).
- No directory move of `data.py` / `train.py` into `tasks/` / `experiments/` —
  **`models/` only** this PR.
- No new architecture. PE is **removed as dead plumbing**, not made configurable
  — the PE knob is 0006.
- No numeric improvement. If any output changes, the PR is wrong.

## Sprinter proposal

### What the four arch files actually share

`Block` + `Transformer` are byte-identical across `transformer.py`,
`linear_attention.py`, `deltanet.py`, `titans.py` **except** for two things:

1. **Mixer-specific kwargs threaded through** `Block`/`Transformer`: `gated`
   (deltanet), `memory_mult` + `max_inner_lr` (titans). Attention / LA have none.
2. **`cos`/`sin` is dead for 3 of 4 mixers** — only `Attention` calls
   `apply_rope`. LA / DeltaNet / Titans take `(x, cos, sin)` and ignore the last
   two.

Titans' `Transformer` additionally carries a Titans-only `diagnostics()` method
(plus `trace()` on the mixer and the `summarize_trace()` free function).

### Target layout (this PR = `models/` only)

```
models/
  backbone.py          # Block + LMModel (was Transformer), mixer-agnostic
  registry.py          # MIXERS[name] -> factory(dim, n_heads, key); FFNS[name]
  ffn.py               # SwiGLU (the one FFN today)
  attention.py         # Attention mixer only — owns its RoPE internally
  linear_attention.py  # mixer only
  deltanet.py          # mixer; registers "deltanet" AND "gated_deltanet"
  titans.py            # mixer + trace()/diagnostics() as free fns on LMModel
```

`utils.py` keeps `RMSNorm` + `rope_freqs` / `apply_rope`; `SwiGLU` moves to
`models/ffn.py`.

### Three non-mechanical decisions (Sprinter leans, for the Skeptic pass)

1. **How mixer kwargs are carried.** *Lean:* a registry entry is a factory
   `(dim, n_heads, key) -> mixer`; `Block` takes the factory and knows nothing
   about `gated` / `memory_*`. `gated_deltanet` = `partial(DeltaNet, gated=True)`.
   Backbone stays arch-agnostic.
2. **Where Titans diagnostics live.** *Lean:* move `diagnostics()` /
   `summarize_trace()` **out** of the shared model into `titans.py` as free
   functions taking a generic `LMModel`; keep `trace()` on the Titans mixer.
   Keeps `LMModel` clean.
3. **PE removal.** Mixer signature drops to `__call__(self, x)`. `Attention`
   computes `rope_freqs(head_dim, T)` internally → identical result; the other
   three never used it → no change. This is what makes the refactor parity-safe.

### Steps

1. **Record the parity snapshot first** (before touching anything): for all 5
   variants (transformer, linear_attention, deltanet, gated_deltanet, titans),
   capture param count + forward logits on a fixed key + fixed token input.
2. Build `backbone.py` + `registry.py` + `ffn.py` from the transformer copy.
3. Convert each arch file to mixer-only; register it.
4. Repoint construction sites (arch `__main__`s, `experiment_capacity.py`,
   `experiment_retention.py`, `sweep_titans_toy.py`, `bench_*.py`) to the
   registry; rename `Transformer → LMModel`.
5. Delete the four duplicated `Block` / `Transformer` copies.
6. **Parity assert:** identical param count + logits for all 5.

## Skeptic review

> **Pending** — to be completed by the Skeptic reviewer (per
> [METHOD.md](../process/METHOD.md), a *different* model/voice, so the tension is
> real). Candidate objections Sprinter has already surfaced for that pass:

- **The parity gate is hostage to RNG key-consumption order.** Init values depend
  on the exact `jax.random.split` structure: `split(key, n_layers+2)`,
  `tok_emb=keys[0]`, blocks=`keys[1:-1]`, `lm_head=keys[-1]`, and inside a block
  `k_attn, k_mlp = split(key)` with the mixer consuming `k_attn` identically. The
  backbone must reproduce this **exactly**, or params drift and parity fails. This
  is the test's whole job.
- **What does "identical" mean?** Lean: **exact** equality on CPU, because the op
  graph is preserved (moving `rope_freqs` into `Attention` reorders no math). If
  any unavoidable reorder forces tolerance, state the `atol`/`rtol` and justify —
  don't silently loosen.
- **Diagnostics relocation is the one genuinely non-mechanical move.** Confirm
  Titans `diagnostics()` reproduces its prior numbers on a fixed input after
  moving off the shared class.
- **Float determinism / platform.** Parity is asserted on CPU (`JAX_PLATFORMS=cpu`,
  as the arch files already pin); note that cross-device equality is not claimed.

## Final settlement

> **Pending review.**

## Budget

**~0 compute.** No training. Validation is one forward pass per arch (5 variants)
on CPU plus a param-count / pytree-structure check — seconds, no GPU.

## Files expected to change

- **New:** `models/backbone.py`, `models/registry.py`, `models/ffn.py`,
  `models/attention.py`, `models/linear_attention.py`, `models/deltanet.py`,
  `models/titans.py`, `tests/test_parity.py`.
- **Modified:** `utils.py` (`SwiGLU` → `ffn.py`), `experiment_capacity.py`,
  `experiment_retention.py`, `sweep_titans_toy.py`, `bench_chunkwise.py`,
  `bench_prod_scaling.py`, and any other `Transformer(...)` construction site.
- **Deleted (moved into `models/`):** root `transformer.py`,
  `linear_attention.py`, `deltanet.py`, `titans.py`.

## Validation

A parity test (`tests/test_parity.py`), recorded from current code and asserted
against the refactor: for each of the 5 variants, with a fixed PRNG key and fixed
token input, assert (a) identical parameter count + pytree structure, and (b)
forward logits identical (exact equality on CPU). The existing per-arch
`__main__` MQAR smoke runs remain green as a secondary check.

## Claim (pre-registered)

**Infrastructure only.** The claim *is* the parity equality: param count + forward
logits are unchanged for all five variants. There is no metric, null, or regime
because there is no measurement. Nothing about capacity / retention / external
validity is asserted or affected by this PR.

## Follow-ups

- 0002 executor (redun); 0003 `Task` + `Config` split — the directory moves into
  `tasks/` / `experiments/` land there, not here.
- PE as a `ModelConfig` knob → 0006.
