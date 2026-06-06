# Design: modular, multi-task, multi-architecture refactor

Status: **adopted anchor.** This document owns the *what / why* of the refactor;
the ordered proposals that implement it (one goal = one proposal = one PR) live
in [`docs/proposals/ROADMAP.md`](docs/proposals/ROADMAP.md). Decisions still
marked **OPEN** are resolved in the proposal that depends on them.

## 1. Why this exists

The repo started as a single-narrative scratchpad ("one file per architecture,
the only line that changes is `self.attn = ...`"). We now want it to be a small
but real comparison harness:

1. Remove the scaffold duplication across architectures.
2. Keep one file per architecture, but make that file *only the mixer*.
3. Make tasks modular — add an `addition` task beside MQAR (params: max digit
   size, how many numbers), built from day one to probe **generalization**.
4. Run the same sweeps we ran on Titans for every architecture, so results are
   **comparable**.
5. Make experiments modular — every run is a content-addressed `(model, task,
   train)` config dispatched through one executor, so the work above plugs into a
   single backbone instead of bespoke scripts.

This is a behavior-preserving refactor first, and a feature addition second. The
existing Titans W&B results must remain reproducible. Modularization has a model
side (1–2), a task side (3), and an experiment side (5); the executor lands
**early** — right after the scaffold (Phase 1) — because the task and sweep work
plug into it.

## 2. Current state (grounded in the code)

- **Shared stack:** `data.py` (MQAR generator + `Config`), `train.py` (the real
  workhorse: masked cross-entropy loop, early stopping, finite-checks, a
  W&B-shaped `log_fn`), `utils.py` (RMSNorm / SwiGLU / RoPE).
- **Four mixers, each in its own file:** `transformer.py`, `linear_attention.py`,
  `deltanet.py` (plain + gated via a flag), `titans.py`.
- **Duplication:** `Block` and the LM wrapper (named `Transformer` everywhere,
  even for DeltaNet/Titans) are copy-pasted near-verbatim across all four files.
  The only real difference is the mixer constructed inside `Block`.
- **Sweep system is Titans-only:** `sweep_titans_toy.py` + `sweeps/titans_{toy,
  easy,level1}.yaml`. The FLOP estimator and diagnostics are Titans-specific.
- **A separate Flax prototype (`addition_transformer.py`, not in this repo) is a
  different world:** Flax (not Equinox), its own tokenizer / train loop / model
  (LayerNorm + GELU + learned positional embeddings + tied weights), plus a MoE
  section and a Chinchilla IsoFLOP scaling-law harness. It motivates §5 and §10
  but is not adopted.

### Three couplings the refactor has to break

1. **Scaffold lives inside each mixer file.** The shared `Block` + LM wrapper
   should live in one place; each arch file should contain only its mixer.
   The non-trivial part: mixers do **not** share a constructor signature —
   `(dim, n_heads, key)` for softmax/linear-attn, `(..., gated=False)` for
   DeltaNet, `(..., memory_mult, max_inner_lr)` for Titans.
2. **Task is baked into `data.py` and `train.py`.** `Config` mixes task params
   (`vocab_size`, `num_kv_pairs`, `power_a`) with training params (`lr`,
   `batch_size`, `epochs`, ...); `train.py` imports `mqar_example` / `make_split`
   directly; `inspect_example` assumes kv-pairs. Addition breaks the `Config`
   assumptions: its `vocab_size` is fixed by the tokenizer and its `seq_len` is
   *derived* from `(max_digits, n_numbers)`.
3. **The sweep harness is Titans-shaped.** `estimate_forward_flops_per_example`,
   `make_wandb_logger`, `diagnostics`, `trace`, `summarize_trace` all assume
   Titans' fast-memory internals.

## 3. Design principles

- **One shared training loop.** Architectures differ only in the mixer; tasks
  differ only in the data + eval metric. `train.py`'s masked-CE loop already
  fits both MQAR and addition — keep it as the single protocol.
- **Every task feeds the *existing* Equinox models.** Do **not** adopt the Flax
  `AdditionTransformer`. Using a different model for addition would make it
  non-comparable with the MQAR results and with the Titans data already in W&B.
  Addition is "another `(tokens, targets, mask)` generator," nothing more.
- **Config is data.** Each run is fully described by a `(model, task, train)`
  config; results are content-addressed and cached so reruns are cheap. We
  already have the kernel of this in `cache.py`.
- **The refactor is gated by a golden test.** Param count + a fixed-seed forward
  per arch must be identical before and after the scaffold extraction.

## 4. Target architecture

### 4.1 Proposed module layout (names provisional)

```
utils.py                 # unchanged: RMSNorm, SwiGLU, RoPE
models/
  scaffold.py            # Block, LMModel, build_lm(...)  — the shared wrapper
  registry.py            # MIXERS: name -> (cls, default_kwargs); FFNS likewise
  attention.py           # softmax mixer            (from transformer.py)
  linear_attention.py    # mixer only
  deltanet.py            # mixer only (plain + gated)
  titans.py              # mixer + optional diagnostics hook
  ffn.py                 # SwiGLU now; MoE later
tasks/
  base.py                # Task interface
  registry.py            # TASKS: name -> Task factory
  mqar.py                # current data.py logic, wrapped
  addition.py            # new
train.py                 # task-driven loop (largely unchanged)
sweep.py                 # arch-agnostic sweep entrypoint (from sweep_titans_toy.py)
experiments/             # experiment-as-python-file (see §7)
cache.py                 # grows into a Step/Experiment layer (see §7)
sweeps/*.yaml            # gain `arch` and `task` keys
```

### 4.2 Scaffold + mixer/FFN registries

The shared `Block` takes a **mixer factory** and an **FFN factory**, not a
concrete class:

- `MIXERS["titans"] = (Titans, {"memory_mult": 4, "max_inner_lr": 0.05})`
- `FFNS["swiglu"] = (SwiGLU, {})` — and later `FFNS["moe"] = (MoEFFN, {...})`.

This kills the duplication and makes two future moves cheap: MoE becomes an FFN
swap (not a new scaffold), and "DeltaNet vs Gated DeltaNet" stays a single file.

Two cleanups while we're in here:

- Rename the LM wrapper `Transformer -> LMModel` (it is not a transformer for
  DeltaNet/Titans).
- Remove the **dead `cos`/`sin` plumbing**. Only the softmax mixer applies RoPE
  today; the linear/recurrent mixers accept `cos`/`sin` and ignore them
  (effectively NoPE). Make positional encoding an explicit per-mixer choice
  rather than threading unused tables (this also matters for §6).

### 4.3 Task interface

```python
class Task(Protocol):
    vocab_size: int
    seq_len: int                          # max sequence length the model sees
    def make_split(self, key, n) -> tuple[Array, Array, Array]   # tokens, targets, mask
    def eval_metric(self, logits, targets, mask) -> float        # owns its metric
    def inspect(self, model, key) -> None                        # human-readable probe
```

- **MQAR** wraps the current `mqar_example` / `make_split` / `inspect_example`
  with no behavior change; its metric is the existing per-token masked accuracy.
- **Addition** (see §5) reports **per-example exact-match** instead — getting
  every answer digit right, the metric that matters for arithmetic.

### 4.4 Config split

`Config` splits into `ModelConfig` (dim, n_heads, n_layers, mlp_mult, + arch
kwargs), `TaskConfig` (per-task), and `TrainConfig` (lr, batch_size, epochs,
patience, target_acc). A run = one of each. This is also what makes the sweep
and the eventual executor clean.

## 5. The addition task

### 5.1 Tokenizer, vocab, sequence length

- Vocab: digits `0-9`, `+`, `=`, plus a reserved `PAD` (and optionally `EOS`).
  ~13-14 tokens, fixed — so `vocab_size` is a property of the task, not a knob.
- `seq_len` is derived to bound the **largest** problem in the eval grid (see
  §5.3), not just the train distribution, so longer OOD sequences still fit.
  Shorter sequences are left-padded; loss is masked to the answer span.

### 5.2 Targets / mask alignment (the one care-point)

MQAR overrides targets at query positions. Addition is plain causal next-token
on the answer span: `targets[p] = tokens[p+1]` and `mask[p] = 1` for positions
`p` whose next token is an answer digit (i.e. after `=`). This fits the existing
`(tokens, targets, mask)` contract; the shift just has to be exactly right, and
it gets a golden/smoke test.

### 5.3 Generalization (the actually-interesting axis)

We want to train on one region and test on bigger digits / more numbers. So the
task carries **two distributions**: a train distribution and an **eval grid**
over `(max_digits, n_numbers)` that includes out-of-distribution cells.

- **Reporting:** a 2D heatmap of exact-match accuracy over
  `(n_digits, n_numbers)` with the train region marked, logged per arch — the
  artifact that shows *what each mechanism needs to generalize*. The eval-grid
  plumbing ships with the addition task (Phase 3); the heatmap becomes a *claim*
  only once PE is controlled (Phase 5).
- **Positional encoding is the dominant confound here.** Whether a model
  extrapolates to longer/larger problems is driven more by PE (RoPE vs NoPE vs
  index-hints / abacus-style) than by the mixer. Today the softmax model uses
  RoPE and the recurrent models use NoPE — an uncontrolled difference.
  **Decision (§11):** addition (Phase 3) makes an *in-distribution* claim only;
  PE becomes a controlled `ModelConfig` knob in Phase 5, and length-generalization
  claims land with it. (Another reason not to use the Flax learned-pos model: it
  cannot run at a longer `T` than it was trained on.)

## 6. Comparability: what "comparable" means here

Agreed two-phase protocol:

- **Phase A (first):** pin model size per task tier, run a per-arch HP search
  over a shared space, and report **accuracy + steps/FLOPs-to-solve**. This
  isolates the mechanism and matches the spirit of the existing
  capacity/retention experiments.
- **Phase B (later):** sweep model sizes and plot accuracy vs params (and FLOPs)
  per arch to compare **scaling behavior per mechanism**.

### Specific issues in the current sweep to fix when generalizing it

1. **Model-size knobs live inside the Bayes objective.** The Titans YAMLs sweep
   `dim / n_heads / n_layers / mlp_mult` *and* maximize accuracy in one search,
   conflating capacity with mechanism. Phase A fixes size and searches only
   optimization HPs.
2. **`target_acc=0.99` saturates the objective.** With early stop at the target,
   every config that solves the task ties at ~0.99, so `objective/score` is
   nearly flat across solvers — the search mostly learns *whether* it solved.
   Use **steps/FLOPs-to-target** (or accuracy at a fixed small budget) as the
   objective instead.
3. **Hyperband is effectively a no-op.** `early_terminate` bands on
   `objective/score`, but that metric is logged exactly once at the end of the
   run. Band on a per-epoch metric (e.g. `learning/test_acc`).
4. **FLOP estimator is Titans-only.** `estimate_forward_flops_per_example`
   hardcodes the fast-memory cost. Make it arch-aware (per-mixer FLOP function),
   since "to-FLOPs" is a comparison axis.
5. **Per-arch knobs and diagnostics.** `max_inner_lr` exists only for Titans;
   the shared search space is `(lr, size, seed)` plus a small per-arch
   extension, with an **equal trial budget** per arch. Titans' `memory/*`
   diagnostics become an optional per-arch hook; the generic
   `objective/learning/stability/health/runtime/compute` metrics from `train.py`
   are the comparable core.

### OPEN — "pin model size" means iso-width or iso-param?

At fixed `(dim, layers)`, param counts are **not** equal across mixers (softmax <
linear-attn (carries convs) < DeltaNet (+gates) < Titans (+fast-memory) < MoE
(+experts)). For Phase A I propose fixing `(dim, depth)` and *reporting* the
param/FLOP delta; in Phase B params is the x-axis so it resolves itself. Confirm
before Phase A.

## 7. Experiment orchestration (executor) — decided: build on redun, lands early

Requirement: a config-driven system that **tracks every step**, caches results,
and records provenance. We already have the kernel — `cache.py`'s
`cached(key, sources)` is content-addressed caching keyed by config + source
bytes, and `experiment_capacity.py` is a hand-rolled step-DAG with parallel
dispatch. The minimal interface:

```python
step = Step(name, fn, config)     # fn(config) -> result, cached by hash(config + source)
exp  = Experiment([step, ...])    # a python file that declares steps; W&B is one sink
```

**Decision: build on [redun](https://github.com/insitro/redun)** — Python-native,
pip-only, local-first — rather than vendor Marin or hand-roll a DAG. It already
gives us the two things we'd otherwise write by hand: a content-addressed
step-DAG and queryable provenance (a SQLite call-graph). It hashes each task's
source + args as the cache key; `@task(version=...)` pins a version to suppress
reruns on cosmetic edits.

*Why automatic hashing, not Marin's manual versioning.* Marin's executor targets
**expensive** steps (tokenize a web corpus, train 8B), so it trusts the author to
declare the few config fields that count and accepts that a *forgotten* field can
silently collide. Our steps are **cheap** (toy/level1 sweeps run in minutes),
which inverts the trade-off — a silent false cache hit costs more than a 2-minute
rerun. So we keep redun's automatic source+args hashing as the correctness
default and borrow only Marin's two scale-independent ideas as thin conveniences:
**readable `name-hash` output paths** (greppable, not an opaque blob) and
**pseudo-dependencies** (version-but-don't-block, for resuming from an
in-progress checkpoint).

*Why early (Phase 1, right after the scaffold).* The executor is the
experiment-side of the same modularization. Landing it before tasks and sweeps
means each of those plugs into one backbone instead of a bespoke script, and the
hand-rolled DAG in `experiment_capacity.py` is *replaced*, not extended. This
stays lean, not lock-in: every run is still a `(model, task, train)` config, so
swapping redun out later remains cheap.

## 8. Testing

- **Golden no-op (gates Phase 0):** snapshot param count + a fixed-seed forward
  per arch before the scaffold extraction; assert identical after.
- **Smoke test per (arch x task):** a handful of steps; assert right shapes and
  finite loss. This is also the addition target/mask alignment check.
- **(Later) metric regression:** re-run a couple of Titans sweep cells and check
  the numbers match the W&B history, so the task/sweep refactor is provably
  faithful.

## 9. Staged plan

Each phase is one goal = one proposal = one PR; the proposal map (numbers,
dependencies, pre-registered claims) lives in
[`docs/proposals/ROADMAP.md`](docs/proposals/ROADMAP.md) and the Sprinter/Auditor
workflow in [`docs/process/METHOD.md`](docs/process/METHOD.md). The executor
(Phase 1) is pulled ahead of tasks and sweeps because they plug into it.

- **Phase 0 — golden gate + scaffold extraction** (proposal 0001). Pure dedup,
  zero behavior change: shared `Block` + `LMModel`, mixer + FFN registries,
  rename wrapper, drop dead RoPE plumbing. Exit: golden test green for all four
  archs.
- **Phase 1 — executor on redun** (proposal 0002). Grow `cache.py` into the
  redun-backed `Step`/`Experiment` layer (§7) and re-express
  `experiment_capacity.py` through it. Exit: the capacity experiment reproduces
  its cached numbers via the executor.
- **Phase 2 — task interface + Config split** (proposal 0003). `Task` protocol;
  wrap MQAR with no behavior change; split `Config` into model/task/train; runs
  declared as `(model, task, train)` on the executor. Exit: MQAR smoke + a
  capacity cell unchanged.
- **Phase 3 — addition task** (proposal 0004). `tasks/addition.py` with the
  tokenizer, target/mask alignment (§5.2), train + eval-grid split, and
  per-example exact-match. **In-distribution claim only** — length-generalization
  waits for the PE knob in Phase 5 (§5.3). Exit: addition trains on the existing
  models; smoke tests green.
- **Phase 4 — comparable per-arch HP search** (proposal 0005). Arch-agnostic
  sweep as an executor experiment; fixed size per tier; steps/FLOPs-to-target
  objective; arch-aware FLOPs; hyperband fix; per-arch diagnostics hook. Resolves
  the iso-width/param and objective OPENs (§6, §11). Exit: one sweep per
  (arch x task tier) with matched budget; Titans numbers reproduced.
- **Phase 5 — scaling frontier + PE knob** (proposal 0006). Sweep sizes;
  acc-vs-params/FLOPs per arch; make PE a `ModelConfig` knob; finalize the
  PE-controlled generalization heatmap (§5.3). Resolves the PE OPEN (§11).
- **Phase 6 (later) — MoE arch + IsoFLOP harness** (proposal 0007) onto the
  generic Task + arch registry.

## 10. Explicitly deferred / not adopted

- **MoE architecture** — comes in as an FFN swap (`FFNS["moe"]`), not a new
  scaffold; needs `jax.lax.ragged_dot`. Phase 6.
- **IsoFLOP / Chinchilla scaling-law harness** — port onto the generic Task +
  arch registry once the core phases land. Phase 6.
- **Flax `AdditionTransformer` (LayerNorm/GELU/learned-pos/tied)** — not
  adopted; addition reuses the existing Equinox models for comparability.

## 11. Open decisions (rollup)

Resolved since the first draft:

- **Executor approach** (§7) → **build on redun**, and pulled early to Phase 1.
- **Positional-encoding knob** (§5.3) → addition (Phase 3) ships eval-grid
  infrastructure and an *in-distribution* claim only; the PE knob and any
  length-generalization claim land together in Phase 5.

Still open, each locked in the proposal that needs it:

1. **Iso-width vs iso-param** for "pinned" size — locked in the Phase 4 sweep
   proposal (§6); current lean: fix `(dim, depth)`, report the param/FLOP delta.
2. **Sweep objective details** — steps-to-target vs FLOPs-to-target vs
   accuracy-at-budget, and the censoring convention for configs that never solve
   — locked in the Phase 4 sweep proposal (§6).
