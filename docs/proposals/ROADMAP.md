# Backlog (todo DAG)

A dependency-linked list of **open todos**, not a fixed plan.
[`DIRECTION.md`](../../DIRECTION.md) owns the living **what / why** (questions and
bets); [`../process/METHOD.md`](../process/METHOD.md) owns **how** (the
Sprinter/Skeptic workflow). This file is the **how-much-is-ready**: it turns the
current direction into concrete todos with explicit dependencies, and is
**re-derived every time an experiment lands**.

**Read it as a graph, not a march.** Numbers are stable IDs (see
[`README.md`](README.md)), **not** an order. "What's next" is the highest-value
**ready** todo — the load-bearing question in `DIRECTION.md` breaks ties — never
"the lowest open number". 0002 (the executor) sits behind 0003/0004 even though
it is numerically earlier, because nothing forces it yet.

**One todo = one open question = one goal = one PR.** If a todo wants an "and", it
is two todos. When we start one, we draft `docs/proposals/000N-<goal>.md` from
[`../process/PROPOSAL_TEMPLATE.md`](../process/PROPOSAL_TEMPLATE.md) (Sprinter →
Skeptic → settle), implement it on `pr/000N-<short-goal>`, and record the outcome
in [`../experiments/`](../experiments/) against the pre-registered claim — which
then updates `DIRECTION.md` and re-derives this backlog.

## Status legend

- ✅ **done** — landed in `main`.
- ▶️ **ready** — all code deps met and any unblock-condition holds; pick the next
  PR from here.
- ⏳ **blocked** — waiting on a dependency or a stated condition.
- 💤 **deferred** — real, but parked until the program needs it.

## How to use this

1. A todo is **ready** when its code deps are done *and* its unblock-condition (if
   any) holds. Only ▶️ todos are eligible to start.
2. Pick the next PR from the ▶️ set by **value**, not by number. The load-bearing
   `DIRECTION.md` question (external validity) wins ties.
3. Keep one goal per PR. A title that needs "and" is two todos — split it.
4. Every landed experiment re-derives this file: a dep clears, a condition flips,
   a todo splits or drops. The plan is disposable; the ledger is the memory.

## The graph

```
0001 ✅ scaffold + registries
  │
  ├─► 0003 ▶️ Task interface ─► 0004 addition ─► 0005 per-arch sweep ─► 0006 scaling + PE
  │      + Config split         (ext. validity)        │   └────────────► 0007 MoE + IsoFLOP
  │                                                     │
  │                              cache.py felt-pain ────┘  pulls in ↓
  ├─► 0002a ▶️ executor design study ──────────────────────► 0002 ⏳ light executor
  │         (study redun + marin)                              (self-built; gated)
  │
  └─► 0008 ▶️ parity CI + cache-source hardening (Lite, optional)
```

The spine `0003 → 0004 → 0005 → 0006` walks straight at the load-bearing
question (do synthetic rankings predict a real task, and survive scale?). The
executor is a **side track**: a research spike (0002a) is ready now, but the
implementation (0002) only unblocks when a sweep makes `cache.py` actually hurt.

## The backlog

| ID | Goal | Depends on | Unblock when | Resolves OPEN | Compute | Status |
|---|---|---|---|---|---|---|
| [0001](#0001--parity-gated-scaffold-extraction) | parity-gated scaffold + registries | — | done | — | ~0 | ✅ |
| [0003](#0003--task-interface--config-split) | `Task` protocol + `Config` split (MQAR unchanged) | 0001 | **now** | — | ~0 | ▶️ **next** |
| [0002a](#0002a--executor-design-study) | executor design study (redun + marin) | 0001 | **now** | executor approach | ~0 | ▶️ |
| [0002](#0002--light-self-built-executor) | light self-built executor | 0002a, 0003 | study settled **and** `cache.py` hurts | executor → light/self-built | ~0 | ⏳ |
| [0004](#0004--addition-task) | addition task + exact-match | 0003 | 0003 lands | PE (infra only) | low | ⏳ |
| [0005](#0005--comparable-per-arch-hp-search) | comparable per-arch sweep | 0003, 0004 | 0004 lands | iso-width/param · objective | **high** | ⏳ |
| [0006](#0006--scaling-frontier--pe-knob) | scaling frontier + PE knob | 0005 | 0005 lands | PE (knob + claim) | **high** | ⏳ |
| [0007](#0007--moe--isoflop-later) | MoE arch + IsoFLOP port | 0001, 0005 | 0005 lands | — | high | 💤 |
| [0008](#0008--parity-ci--cache-source-hardening-lite) | parity CI + cache-source hardening | 0001 | **now** (optional) | — | ~0 | ▶️ |

Compute only enters at **0004+**; 0001–0003, 0002a, and 0008 are
behavior-preserving infra with no new measurement. Per-todo **budgets** (runs ×
seeds × trials, equal across archs) are fixed in each proposal when it is drafted
— compute is the scarce resource, not LOC.

## Ready now → what's next

Three todos are ▶️ **ready**: **0003** (Task interface), **0002a** (executor
design study), and **0008** (parity CI, optional cleanup).

- **Next PR: 0003 — Task interface + `Config` split.** It is the spine toward the
  load-bearing external-validity question and is ~0 compute. *Re-derived
  dependency:* 0003 now depends on **0001 only**, not the executor — `cache.py`
  already content-addresses config dicts, so the `Task` protocol and `Config`
  split land on `cache.py`; routing through an executor is a later convenience
  (0002), not a prerequisite.
- **In parallel (separate track): 0002a — executor design study.** A research
  spike, no code beyond reading: study redun's content-addressed caching + lazy
  expression DAG and marin's executor / step model, then decide the **minimal**
  subset we want in a *light, self-built* layer grown from `cache.py`. The
  executor implementation (0002) waits on this study **and** on felt pain.

---

## 0001 — parity-gated scaffold extraction ✅

- **Goal** — extract the duplicated `Block` + LM wrapper into one shared backbone
  plus mixer/FFN registries; each arch file becomes *only its mixer*.
  Behavior-preserving.
- **Outcome** — landed in `main` (PR #4). `models/backbone.py` (`Block` +
  `LMModel`), `models/registry.py` (`MIXERS`/`FFNS` + `build_lm_model`),
  `models/ffn.py`, mixer-only arch files, and `tests/test_parity.py` (trainable
  leaf bytes + logits, CPU). All construction sites repoint to `build_lm_model`.
- **Claim** — infra only; the parity equality *was* the claim (no new
  measurement). Recorded in [`../experiments/`](../experiments/) when the entry
  lands.
- **Follow-ups it spun off** — the parity test is local-only (no CI) and cache
  source-lists are duplicated per script → **0008**.

## 0003 — Task interface + Config split  ▶️ **next**

- **Goal** — introduce the `Task` protocol and split `Config` into
  `ModelConfig` / `TaskConfig` / `TrainConfig`; wrap MQAR with **zero** behavior
  change; runs become `(model, task, train)` configs.
- **Direction** — infra; the task abstraction that lets one arch face many tasks,
  on the path to the external-validity question.
- **Depends on** — **0001 only** (registry → a config names an arch). *Not* the
  executor: `cache.py` keys on config dicts today, so the split lands on
  `cache.py` and is re-routed through 0002 later if/when it exists.
- **Key files** — `tasks/base.py`, `tasks/registry.py`, `tasks/mqar.py` (wraps
  today's `data.py`), config dataclasses, task-driven `train.py`.
- **Validation / exit** — wrapped-MQAR `(tokens, targets, mask)` are **byte-
  identical** to the current generator for a fixed seed (golden), and the
  capacity/retention cells reproduce unchanged through `cache.py`.
- **Claim** — infra only (MQAR unchanged under the abstraction).

## 0002a — executor design study  ▶️

- **Goal** — a short research spike that settles **what light executor we
  actually need**, before writing one. Output is a settled design note (and a
  draft `0002` proposal), not code.
- **Direction** — infra; forces the *executor* open decision (see
  [`DIRECTION.md`](../../DIRECTION.md) → Open decisions) toward a concrete,
  minimal shape.
- **Depends on** — 0001. No felt-pain gate: research is cheap and de-risks 0002.
- **What to study**
  - **redun** — content-addressed caching, lazy expression DAG, task hashing,
    provenance/call-graph queries. Which of these does `cache.py` already give us
    (content hash of key + source bytes) and which are missing (call-graph,
    automatic source tracking)?
  - **marin** — the executor / `Step` model for reproducible experiment pipelines
    (versioned step outputs, dependency wiring, caching). What is the smallest
    "step + experiment" surface we'd port `experiment_capacity.py` onto?
  - **our pain** — the concrete `cache.py` smells: manual per-script source-lists
    (a missed file = a **stale hit**, the Skeptic's exact worry) and no
    call-graph. Decide which the light layer must fix.
- **Exit** — a one-page design note in `docs/` (or a settled `0002` proposal)
  that names: the minimal `Step`/`Experiment` API, what we borrow vs. skip from
  redun/marin, whether we take a dependency or stay hand-rolled, and the felt-pain
  trigger that flips 0002 to ▶️.
- **Resolves OPEN** — executor approach → **light, self-built** (shape fixed here;
  see [`DIRECTION.md`](../../DIRECTION.md) → Open decisions).
- **Claim** — none (research/design only).

## 0002 — light self-built executor  ⏳

- **Goal** — grow `cache.py` into a **light, self-built** `Step`/`Experiment`
  layer (redun + marin-inspired, scope fixed by 0002a) and re-express
  `experiment_capacity.py` through it. Behavior-preserving.
- **Direction** — infra; pays down the `cache.py` pain (manual source-lists, no
  provenance) once it is real.
- **Depends on** — 0002a (design settled) + 0003 (a config can name an
  arch/task/train, so the executor dispatches the 3-way config).
- **Unblock condition** — start only when `cache.py` **actually hurts** (e.g. the
  0005 sweep multiplies `arch × task` cells and the manual source-lists bite),
  not on schedule. This is the standing Sprinter guardrail.
- **Key files** — `executor.py` (or grown `cache.py`); `experiments/capacity.py`
  (port); `pyproject.toml` if a pinned dependency is taken (decided in 0002a).
- **Validation / exit** — the capacity experiment reproduces its previously
  cached numbers via the executor; a no-op rerun is a cache **hit**, and changing
  one source byte is a cache **miss** (content-addressing shown, not assumed);
  provenance (call-graph) is queryable.
- **Claim** — infra only (reproduces existing capacity numbers).
- **Skeptic guardrail** — if a dependency is taken, pin it; record git hash +
  executor/lib version in run provenance.

## 0004 — addition task  ⏳

- **Goal** — add `tasks/addition.py`: tokenizer, target/mask alignment, train +
  eval-grid split, per-example exact-match — on the **existing Equinox models**.
- **Direction** — opens the **external-validity** question: does a real task
  (addition) behave, and would the synthetic ranking predict it? (in-distribution
  only here).
- **Depends on** — 0003 (`Task` interface + `Config` split).
- **Key files** — `tasks/addition.py`; register in `tasks/registry.py`; a smoke
  test that is also the alignment check (target/mask shifted exactly right).
- **Validation / exit** — smoke green (target/mask shift exactly right; finite
  loss; right shapes); addition trains on the softmax model and ≥1 recurrent
  model. **Overfit a single batch to ~0 loss before any budget** (METHOD).
- **Claim (pre-registered)** — on a **locked in-distribution** eval set, addition
  reaches per-example exact-match ≥ τ on the existing models. **In-distribution
  only.** The `(n_digits, n_numbers)` eval grid is plumbing here and is marked
  *exploratory*; no cross-cell / length-generalization claim is made.
- **PE decision (settled)** — length-generalization is confounded by positional
  encoding, so the PE knob **and** any length-gen claim land together in
  **0006**, not here. Lock the in-distribution eval set + seed before training;
  do not read OOD cells as a result.

## 0005 — comparable per-arch HP search  ⏳

- **Goal** — an **arch-agnostic** sweep, run as an experiment, with fixed size per
  tier, a steps/FLOPs-to-target objective, arch-aware FLOPs, a working hyperband
  band, equal trial budget per arch, and an optional per-arch diagnostics hook.
  Reproduce the Titans numbers.
- **Direction** — the **capacity / retention** bets, now at *matched budget*
  across archs.
- **Depends on** — 0003 + 0004 (tasks; runs per `arch × task`). **This is the
  felt-pain puller for 0002**: many `arch × task` cells are where the manual
  `cache.py` source-lists start to bite.
- **Key files** — `sweep.py` (from `sweep_titans_toy.py`); per-mixer FLOP
  functions; `sweeps/*.yaml` gain `arch`/`task` keys; Titans diagnostics hook.
- **Diagnostics decision** — deferred from 0001: decide whether architecture-
  specific observability is exposed as callbacks/hooks or explicit per-sweep
  helper calls. Keep diagnostics out of sweep objectives unless promoted by a
  proposal.
- **Validation / exit** — one sweep per `(arch × task tier)` at **matched
  budget**; the Titans cells reproduce W&B history (metric regression).
- **Resolves OPEN** — iso-width vs iso-param (lean: fix `(dim, depth)`, report
  Δparams/ΔFLOPs); sweep objective + the censoring convention for configs that
  never solve.
- **Claim (pre-registered)** — per-arch accuracy + steps/FLOPs-to-target on locked
  task tiers, at equal budget. Selection on validation; test reserved.
- **Split watch** — this is the densest todo. If the settlement finds two goals
  (generic sweep runner vs. arch-aware FLOPs + hyperband fix), split into
  0005a/0005b. Flag at draft time; do not pre-split here.

## 0006 — scaling frontier + PE knob  ⏳

- **Goal** — sweep model sizes for accuracy-vs-params/FLOPs per arch; make PE a
  `ModelConfig` knob; finalize the **PE-controlled** generalization heatmap for
  addition.
- **Direction** — the **survives-scale** question + the PE / length-gen confound.
- **Depends on** — 0005.
- **Resolves OPEN** — PE knob (and unlocks the deferred 0004 length-gen claim).
- **Claim (pre-registered)** — scaling behavior per mechanism (acc-vs-params/
  FLOPs) and, now confound-controlled, length-generalization per arch.

## 0007 — MoE + IsoFLOP (later)  💤

- **Goal** — MoE as `FFNS["moe"]` (needs `jax.lax.ragged_dot`); port the IsoFLOP /
  Chinchilla harness onto the generic Task + arch registry.
- **Direction** — extension: a new mechanism family (MoE) through the same
  registry.
- **Depends on** — 0001 (FFN registry) + 0005 (sweep harness).
- **Claim** — set at the gate.

## 0008 — parity CI + cache-source hardening (Lite)  ▶️

- **Goal** — make the 0001 safety net durable: run `tests/test_parity.py` in CI so
  future "behavior-preserving" PRs (0003, 0005) are actually guarded, and remove
  the duplicated per-script cache source-lists so a forgotten file can't cause a
  **stale hit**.
- **Direction** — infra hygiene flagged by the 0001 review; optional, low
  priority, but cheap and closes two real gaps.
- **Depends on** — 0001. Ready now.
- **Files** — a minimal CI workflow under `.github/workflows/`; a shared
  source-list helper (or fold the source-list concern into 0002, which would
  content-address the call graph automatically — decide in 0002a).
- **Validation** — CI runs the parity test on push; the experiment scripts still
  cache-hit on a no-op rerun.
- **Claim** — none (infra hygiene).

---

## Open-decision routing

The decisions themselves (and current leans) live in
[`DIRECTION.md`](../../DIRECTION.md); this table is just which todo is expected to
force each one:

| OPEN decision | Decided in |
|---|---|
| Executor approach (shape) | 0002a — light, self-built; redun + marin-inspired |
| Executor adoption (when) | 0002 — only when `cache.py` hurts |
| PE knob / length-gen | 0004 (in-dist claim) → 0006 (knob + length-gen claim) |
| Iso-width vs iso-param | 0005 |
| Sweep objective + censoring | 0005 |

## Next action

**0003 — Task interface + `Config` split** is the next PR (▶️, depends on 0001
only). Draft `docs/proposals/0003-task-interface.md` from
[`../process/PROPOSAL_TEMPLATE.md`](../process/PROPOSAL_TEMPLATE.md) (Sprinter →
Skeptic → settle), then implement on `pr/0003-task-interface`. In parallel,
**0002a — executor design study** is ready as a separate research track.
