# Roadmap

The ordered proposal backlog that implements [`DESIGN.md`](../../DESIGN.md).
`DESIGN.md` owns **what / why**; [`../process/METHOD.md`](../process/METHOD.md)
owns **how** (the Sprinter/Auditor workflow). This file is the bridge: it turns
the staged plan into a numbered queue of proposals.

**One proposal = one DESIGN phase = one goal = one PR.** If a proposal wants an
"and", it is two proposals. When we start one, we draft
`docs/proposals/000N-<goal>.md` from
[`../process/PROPOSAL_TEMPLATE.md`](../process/PROPOSAL_TEMPLATE.md) (Sprinter →
Auditor → settle), implement it on `pr/000N-<short-goal>`, and record the outcome
in [`../experiments/`](../experiments/) against the pre-registered claim.

Status: 🔜 next · ⬜ planned · ✅ done

## Priority change vs the first DESIGN draft

The original draft deferred the **executor** to the last phase and tried to do
several things per phase. Two corrections shape this queue:

1. **Executor moves up to Phase 1** (proposal 0002), right after the scaffold and
   *before* tasks and sweeps — it is the experiment-side of the same
   modularization, so the task/config/sweep work plugs into one backbone instead
   of bespoke scripts. The kernel already exists (`cache.py` + the hand-rolled DAG
   in `experiment_capacity.py`), so it lands cheaply and behavior-preserving.
2. **One goal per proposal.** The draft's "task abstraction + addition" splits
   into 0003 (abstraction, MQAR unchanged) and 0004 (the new addition task), so
   each PR proves one thing.

The order is otherwise **dependency-locked**, not a free ranking: everything
needs the scaffold; tasks precede sweeps (sweeps run per `arch × task`); scaling
needs the sweep harness; MoE is deferred.

## The queue

| # | Phase · goal | Depends on | Resolves OPEN | Compute | Status |
|---|---|---|---|---|---|
| [0001](#0001--golden-gate--scaffold-extraction) | P0 · golden-gated scaffold + registries | — | — | ~0 | 🔜 |
| [0002](#0002--executor-on-redun) | P1 · executor on redun | 0001 | executor → redun | ~0 | ⬜ |
| [0003](#0003--task-interface--config-split) | P2 · `Task` protocol + `Config` split (MQAR unchanged) | 0002 | — | ~0 | ⬜ |
| [0004](#0004--addition-task) | P3 · addition task + exact-match | 0003 | PE (infra only) | low | ⬜ |
| [0005](#0005--comparable-per-arch-hp-search) | P4 · comparable per-arch sweep | 0002, 0004 | iso-width/param · objective | **high** | ⬜ |
| [0006](#0006--scaling-frontier--pe-knob) | P5 · scaling frontier + PE knob | 0005 | PE (knob + claim) | **high** | ⬜ |
| [0007](#0007--moe--isoflop-later) | P6 · MoE arch + IsoFLOP port | 0001, 0005 | — | high | ⬜ |

```
0001 ─► 0002 ─► 0003 ─► 0004 ─► 0005 ─► 0006
scaffold executor task   add    sweep   scaling+PE
                                   └► 0007 MoE + IsoFLOP  (also needs 0001)
```

Compute only enters at **0004+**; 0001–0003 are behavior-preserving infra with no
new measurement. Per-proposal **budgets** (runs × seeds × trials, equal across
archs) are fixed in each proposal — compute is the scarce resource, not LOC.

---

## 0001 — golden gate + scaffold extraction

- **Goal** — extract the duplicated `Block` + LM wrapper into one shared scaffold
  plus mixer/FFN registries; each arch file becomes *only its mixer*.
  Behavior-preserving.
- **DESIGN** — Phase 0; §3, §4.1, §4.2, §8.
- **Depends on** — nothing (foundation).
- **Key files** — new `models/scaffold.py`, `models/registry.py`, `models/ffn.py`;
  mixer-only `models/{attention,linear_attention,deltanet,titans}.py`; a golden
  test; delete the four copy-pasted scaffolds. Also: rename `Transformer →
  LMModel`; drop the dead `cos`/`sin` plumbing (PE becomes an explicit per-mixer
  choice).
- **Validation / exit** — **write the golden snapshot first** (param count +
  fixed-seed forward per arch, against current code), then refactor under it;
  assert identical for all four archs.
- **Claim** — infra only; the golden equality *is* the claim (no new measurement).
- **Why first** — lowest-risk, highest-leverage: a safety net then pure dedup,
  and the registry is the prerequisite for "a config names an arch".

## 0002 — executor on redun

- **Goal** — grow `cache.py` into a redun-backed `Step`/`Experiment` layer and
  re-express `experiment_capacity.py` through it. Behavior-preserving.
- **DESIGN** — Phase 1; §3 ("config is data"), §7.
- **Depends on** — 0001 (registry → a config can name an arch/FFN).
- **Key files** — `executor.py` (or grown `cache.py`); `experiments/capacity.py`
  (port); `pyproject.toml` (+`redun`, pinned, in the experiment group).
- **Validation / exit** — the capacity experiment reproduces its previously
  cached numbers via the executor; a no-op rerun is a cache **hit**, and changing
  one source byte is a cache **miss** (content-addressing shown, not assumed);
  provenance (call-graph) is queryable.
- **Resolves OPEN** — executor approach → **redun** (DESIGN §7).
- **Claim** — infra only (reproduces existing capacity numbers).
- **Auditor guardrail** — pin the redun version; record git hash + redun version
  in run provenance.

## 0003 — task interface + Config split

- **Goal** — introduce the `Task` protocol and split `Config` into
  `ModelConfig` / `TaskConfig` / `TrainConfig`; wrap MQAR with **zero** behavior
  change; runs become `(model, task, train)` configs on the executor.
- **DESIGN** — Phase 2; §3, §4.3, §4.4.
- **Depends on** — 0002 (the executor dispatches the 3-way config).
- **Key files** — `tasks/base.py`, `tasks/registry.py`, `tasks/mqar.py` (wraps
  today's `data.py`), config dataclasses, task-driven `train.py`.
- **Validation / exit** — wrapped-MQAR `(tokens, targets, mask)` are **byte-
  identical** to the current generator for a fixed seed (golden), and the 0002
  capacity cell reproduces unchanged.
- **Claim** — infra only (MQAR unchanged under the abstraction).

## 0004 — addition task

- **Goal** — add `tasks/addition.py`: tokenizer, target/mask alignment, train +
  eval-grid split, per-example exact-match — on the **existing Equinox models**.
- **DESIGN** — Phase 3; §5 (esp. §5.2 alignment care-point), §4.3.
- **Depends on** — 0003 (`Task` interface + `Config` split).
- **Key files** — `tasks/addition.py`; register in `tasks/registry.py`; a smoke
  test that is also the §5.2 alignment check.
- **Validation / exit** — smoke green (target/mask shift exactly right; finite
  loss; right shapes); addition trains on the softmax model and ≥1 recurrent
  model.
- **Claim (pre-registered)** — on a **locked in-distribution** eval set, addition
  reaches per-example exact-match ≥ τ on the existing models. **In-distribution
  only.** The `(n_digits, n_numbers)` eval grid is plumbing here and is marked
  *exploratory*; no cross-cell / length-generalization claim is made.
- **PE decision (settled)** — length-generalization is confounded by positional
  encoding (§5.3), so the PE knob **and** any length-gen claim land together in
  **0006**, not here. Lock the in-distribution eval set + seed before training;
  do not read OOD cells as a result.

## 0005 — comparable per-arch HP search

- **Goal** — an **arch-agnostic** sweep, run as an executor experiment, with
  fixed size per tier, a steps/FLOPs-to-target objective, arch-aware FLOPs, a
  working hyperband band, equal trial budget per arch, and an optional per-arch
  diagnostics hook. Reproduce the Titans numbers.
- **DESIGN** — Phase 4; §6.
- **Depends on** — 0002 (executor) + 0004 (tasks; runs per `arch × task`).
- **Key files** — `sweep.py` (from `sweep_titans_toy.py`); per-mixer FLOP
  functions; `sweeps/*.yaml` gain `arch`/`task` keys; Titans diagnostics hook.
- **Validation / exit** — one sweep per `(arch × task tier)` at **matched
  budget**; the Titans cells reproduce W&B history (metric regression).
- **Resolves OPEN** — iso-width vs iso-param (lean: fix `(dim, depth)`, report
  Δparams/ΔFLOPs); sweep objective + the censoring convention for configs that
  never solve.
- **Claim (pre-registered)** — per-arch accuracy + steps/FLOPs-to-target on locked
  task tiers, at equal budget. Selection on validation; test reserved.
- **Split watch** — this is the densest phase. If the settlement finds two goals
  (generic sweep runner vs. arch-aware FLOPs + hyperband fix), split into
  0005a/0005b. Flag at draft time; do not pre-split here.

## 0006 — scaling frontier + PE knob

- **Goal** — sweep model sizes for accuracy-vs-params/FLOPs per arch; make PE a
  `ModelConfig` knob; finalize the **PE-controlled** generalization heatmap for
  addition.
- **DESIGN** — Phase 5; §5.3, §6 (Phase B).
- **Depends on** — 0005.
- **Resolves OPEN** — PE knob (and unlocks the deferred 0004 length-gen claim).
- **Claim (pre-registered)** — scaling behavior per mechanism (acc-vs-params/
  FLOPs) and, now confound-controlled, length-generalization per arch.

## 0007 — MoE + IsoFLOP (later)

- **Goal** — MoE as `FFNS["moe"]` (needs `jax.lax.ragged_dot`); port the IsoFLOP /
  Chinchilla harness onto the generic Task + arch registry.
- **DESIGN** — Phase 6; §10.
- **Depends on** — 0001 (FFN registry) + 0005 (sweep harness).
- **Claim** — set at the gate.

---

## Open-decision routing

| OPEN decision | DESIGN § | Locked in | Current lean |
|---|---|---|---|
| Executor approach | §7 | 0002 | **redun** (decided) |
| PE knob / length-gen | §5.3, §11 | 0004 (infra) → 0006 (knob + claim) | in-dist claim now; PE + gen in 0006 (decided) |
| Iso-width vs iso-param | §6, §11 | 0005 | fix `(dim, depth)`, report Δ |
| Sweep objective + censoring | §6, §11 | 0005 | steps/FLOPs-to-target |

## Next action

Draft **0001 — golden gate + scaffold extraction** from the template
(Sprinter → Auditor → settle), then implement on `pr/0001-golden-scaffold`.
