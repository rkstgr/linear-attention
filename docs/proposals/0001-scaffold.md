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
- No diagnostics API, callback system, or generic model observability hook.
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
  __init__.py
  backbone.py          # Block + LMModel (was Transformer), mixer-agnostic
  registry.py          # MIXERS[name] -> factory(dim, n_heads, key); FFNS[name]
  ffn.py               # SwiGLU (the one FFN today)
  attention.py         # Attention mixer only — owns its RoPE internally
  linear_attention.py  # mixer only
  deltanet.py          # mixer; registers "deltanet" AND "gated_deltanet"
  titans.py            # mixer + trace(); diagnostics helper stays arch-specific
```

`utils.py` keeps `RMSNorm` + `rope_freqs` / `apply_rope`; `SwiGLU` moves to
`models/ffn.py`.

### Three non-mechanical decisions (Sprinter leans, for the Skeptic pass)

1. **How mixer kwargs are carried.** *Lean:* a registry entry is a factory
   `(dim, n_heads, key) -> mixer`; `Block` takes the factory and knows nothing
   about `gated` / `memory_*`. `gated_deltanet` = `partial(DeltaNet, gated=True)`.
   Backbone stays arch-agnostic.
2. **Where Titans diagnostics live.** *Lean:* do **not** solve the diagnostics
   API here. Keep any Titans diagnostic helper code in `models/titans.py` as
   architecture-specific, unused code; do not wire it into `train.py`, W&B, or
   `LMModel`. A later PR decides whether diagnostics are callbacks/hooks or
   explicit sweep helpers.
3. **PE removal.** Mixer signature drops to `__call__(self, x)`. `Attention`
   computes/applies RoPE internally because it is the only mixer that uses it;
   the other three never receive `cos`/`sin`. The backbone stays PE-agnostic.

### Steps

1. **Record the parity snapshot first** (before touching anything): for all 5
   variants (transformer, linear_attention, deltanet, gated_deltanet, titans),
   capture trainable parameter leaf shape/dtype/bytes plus forward logits on the
   minimal fixed fixture described in Validation.
2. Build `backbone.py` + `registry.py` + `ffn.py` from the transformer copy.
3. Convert each arch file to mixer-only; register it.
4. Repoint construction sites (arch `__main__`s, `experiment_capacity.py`,
   `experiment_retention.py`, `sweep_titans_toy.py`) to the registry; rename
   `Transformer → LMModel`; update commands to `python -m models.<arch>`.
5. Delete the four duplicated `Block` / `Transformer` copies.
6. **Parity assert:** identical trainable leaf bytes + logits for all 5.

## Skeptic review

Blocking objections and guardrails:

- **The parity gate is hostage to RNG key-consumption order.** Init values depend
  on the exact `jax.random.split` structure: `split(key, n_layers+2)`,
  `tok_emb=keys[0]`, blocks=`keys[1:-1]`, `lm_head=keys[-1]`, and inside a block
  `k_attn, k_mlp = split(key)` with the mixer consuming `k_attn` identically. The
  backbone must reproduce this **exactly**, or params drift and parity fails.
- **Param count + one logits tensor is too weak.** The parity oracle must compare
  trainable parameter leaves by shape, dtype, and raw bytes, plus exact logits,
  on one tiny locked CPU fixture. Keep this minimal; do not grow it into a broad
  model test suite.
- **Raw PyTree structure is the wrong equality.** Class/module identity will
  change when `Transformer` becomes `LMModel`, so exact PyTreeDef equality would
  reject the intended refactor. Compare semantic trainable leaf order/signature
  instead.
- **Public launch surfaces must move deliberately.** Root model scripts are
  deleted, so documented commands and sweep/import entrypoints must move to
  `python -m models.<arch>`.
- **Old experiment caches may go stale; stale future hits are not okay.** It is
  acceptable for this PR to invalidate previous `.experiment_cache/` entries.
  After the refactor, cache source lists must include the new model implementation
  files so edits to `models/backbone.py`, `models/registry.py`, mixers, or
  `models/ffn.py` are part of future cache keys.
- **Diagnostics are a separate API decision.** Do not put `diagnostics()` on the
  generic `LMModel` and do not wire Titans diagnostics into training/W&B in this
  PR. Leaving the helper code in `models/titans.py` is fine.
- **RoPE belongs to the mixer that uses it.** The shared backbone should not
  precompute `cos`/`sin`; softmax `Attention` owns RoPE internally, and the
  recurrent mixers remain PE-free.
- **Float determinism / platform.** Parity is asserted on CPU (`JAX_PLATFORMS=cpu`,
  as the arch files already pin); cross-device equality is not claimed.

## Final settlement

- Add `models/` with a PE-agnostic `LMModel` backbone, mixer registry, FFN
  module, and mixer-only architecture modules. `Attention` owns RoPE internally;
  Linear Attention, DeltaNet, Gated DeltaNet, and Titans do not receive
  positional-encoding tensors.
- Preserve exact initialization key order and mixer key consumption.
- Convert construction sites to the registry and update model commands to:
  `uv run python -m models.transformer`,
  `uv run python -m models.linear_attention`,
  `uv run python -m models.deltanet --gated`, and
  `uv run python -m models.titans`.
- Keep `data.py`, `train.py`, `tasks/`, and `experiments/` directory moves out of
  scope. Only update imports/construction where needed for the registry and cache
  source lists.
- Keep any Titans diagnostics helper in `models/titans.py` as unused
  architecture-specific code; do not add `LMModel.diagnostics()` and do not
  introduce callbacks/hooks here.
- Validate with the minimal parity fixture below before any broader cleanup.

## Budget

**~0 compute.** No training. Validation is one tiny forward pass per arch
(5 variants) on CPU plus trainable leaf byte checks — seconds, no GPU.

## Files expected to change

- **New:** `models/__init__.py`, `models/backbone.py`, `models/registry.py`,
  `models/ffn.py`, `models/attention.py`, `models/linear_attention.py`,
  `models/deltanet.py`, `models/titans.py`, `tests/test_parity.py`.
- **Modified:** `utils.py` (`SwiGLU` → `ffn.py`), `experiment_capacity.py`,
  `experiment_retention.py`, `sweep_titans_toy.py`, `README.md`, and any other
  `Transformer(...)` construction site or model-source cache list.
- **Deleted (moved into `models/`):** root `transformer.py`,
  `linear_attention.py`, `deltanet.py`, `titans.py`.

## Validation

A minimal parity test (`tests/test_parity.py`), recorded from current code and
asserted against the refactor:

- fixture: `vocab_size=17`, `dim=8`, `n_heads=2`, `n_layers=2`, `mlp_mult=2`,
  one fixed PRNG key, one fixed token sequence, CPU only;
- variants: transformer, linear_attention, deltanet, gated_deltanet, titans;
- Titans uses non-default kwargs once: `memory_mult=2`, `max_inner_lr=0.0125`;
- assert exact trainable parameter leaf shape/dtype/bytes and exact logits
  shape/dtype/bytes;
- do not require raw PyTreeDef/class identity equality.

The module entrypoints above should import and construct models successfully.
The existing per-arch MQAR smoke runs remain green as a secondary check.

## Claim (pre-registered)

**Infrastructure only.** The claim *is* the parity equality: trainable parameter
leaf bytes and forward logits are unchanged for all five variants on the locked
CPU fixture. There is no metric, null, or regime because there is no measurement.
Nothing about capacity / retention / external validity is asserted or affected by
this PR.

## Follow-ups

- 0002 executor (redun); 0003 `Task` + `Config` split — the directory moves into
  `tasks/` / `experiments/` land there, not here.
- PE as a `ModelConfig` knob → 0006.
- Decide diagnostics handling in a later PR: architecture-specific callbacks/hooks
  versus explicit per-sweep helper calls. If logged to W&B, diagnostics can use
  grouped names such as `diagnostics/titans/w1_norm`, but they should stay out of
  sweep selection and pre-registered claims unless explicitly promoted.
