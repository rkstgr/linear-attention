# Proposal 0004: scaling-recipe study (= cross-task tradeoff pilot)

Status: proposed on branch `claude/main-experiments-discussion-kIqbU`.

## Goal

Pick one goal: **freeze the depth/width scaling recipe** that every later
budget-matched experiment depends on, and in the same run get the **first read
on whether MQAR and addition favor different architectures**.

These are the same experiment. To choose a fair depth/width ratio we must train
every architecture on both tasks at a small budget; doing so simultaneously
tells us whether the recall-best and algorithm-best architectures diverge. One
cheap run, three deliverables.

This is the forcing function for the foundation cleanup: it cannot run until the
harness is arch-parametric *and* task-parametric, has a real validation split,
and can match a parameter budget across architectures.

## Claim

This is a setup/pilot proposal, not a headline result. Two falsifiable pieces:

1. **Recipe stability.** There is a depth/width ratio (at fixed `head_dim`,
   ~500k params) that is Pareto-best *averaged across architectures* on each
   task, and it does not drift between ~500k and ~1M params. Null: the best
   ratio is architecture-specific or moves with scale, in which case a fixed
   recipe is illegitimate and we need a scaling rule instead.

2. **Tradeoff gate.** Either the architecture that tops MQAR also tops addition
   (no frontier — single-task headline holds) or they diverge (cross-task Pareto
   becomes the headline and both tasks scale symmetrically). The pilot's job is
   only to detect which world we are in, not to resolve the frontier.

## Tasks and cells

Both tasks parametric; the pilot fixes one solvable cell each (capacity/length
are held below failure so depth/width shape is the only variable).

- **MQAR**: `N_KV=4, T=64, vocab=512`. Level1's capacity/length shape at a modest
  vocab so the parameter budget lives in the mixer, not the embedding table.
  `N_KV=4` is below any sensible `head_dim` ceiling, so all five archs should
  solve it — capacity is not the variable here.
- **Addition**: 3-digit operands, 2 addends. In-distribution cell only; OOD
  length-generalization is deferred (see Deferred).

Offline fixed pool per task, split into **train / val / test** (disjoint RNG
keys). Proposed sizes `n_train=50_000`, `n_val=2_000`, `n_test=2_000` (knob).

## Model family and scale knob

One scale knob (`dim`); other shape hyperparameters are tied to it:

- `head_dim` **fixed** (proposed 16) — it is the capacity lever, not a free
  knob; `n_heads = dim / head_dim` follows from the knob.
- `mlp_mult` fixed at the SwiGLU-adjusted convention; conv width, norm
  placement, optimizer family inherited from current defaults.
- Token-embedding/lm_head **tying held constant across all five archs** (it is a
  real slice of a 500k budget at `vocab=512`); verify the backbone's current
  setting during the build and fix it as a shared setting.
- Titans `memory_mult` tied to `head_dim` (paper default) for the pilot.

Shapes under test: **1L / 2L / 4L**, each width-solved to ~**500k params** at the
fixed `head_dim`. Stability re-check repeats the *chosen* recipe at ~**1M**.

## Budget control

**Iso-parameter** for the pilot (all shapes hit the same param target per scale).
Iso-train-FLOP is deferred to the main grids; the pilot only needs a single
control to compare shapes.

## Metric and selection

- Score: masked-position accuracy (MQAR) / exact-match accuracy (addition).
- **Select and early-stop on val; report test at the best-val checkpoint.**
  Never select on test.
- Optimization hyperparameters (lr, and any per-arch knob) tuned by a small
  fixed grid with **equal budget and equal search-space structure per arch**,
  selected on val. The recipe/stability numbers use the per-(arch, shape, task)
  val-selected lr so shape, not lr, is the variable.

## Compute budget and guardrails

Staged to keep the equal-HPO-budget discipline cheap and explicit:

1. **lr selection**: 5 archs x 3 shapes x 2 tasks x 3 lr x 1 seed ~ 90 runs.
2. **recipe + seeds**: 5 x 3 x 2 x 3 seeds at chosen lr ~ 90 runs.
3. **stability**: chosen recipe only, 2nd budget: 5 x 1 x 2 x 3 seeds ~ 30 runs.

~210 toy-scale runs total (500k params, 50k examples, CPU-minutes each). Cap:
runs through the executor with provenance; abort and re-scope if a single cell
exceeds a few CPU-minutes or total wall-clock blows past a single workday with
`--parallel`. Everything labeled toy-scale.

## Validation plan

- Recipe legitimacy is checked by the stability re-run, not assumed.
- The lr grid guards against shape differences being lr artifacts.
- Three seeds give a spread; report mean +/- range, never a single seed.
- Falsified if the best ratio is arch-specific or scale-dependent (then: scaling
  rule, not fixed ratio) or if lr-selection ties make shapes indistinguishable
  (then: widen the lr grid or budget before trusting the recipe).

## Deliverables and decision gate

1. Frozen per-task depth/width recipe (or a scaling rule if unstable).
2. Stability verdict across the two budgets.
3. Cross-task ranking signal -> the gate: same arch wins both (single-task
   headline) vs divergence (cross-task Pareto headline, scale both symmetrically).

Recorded as a `docs/experiments/` entry after the run: budget spent, metric vs
null, result (incl. negative/inconclusive plainly), regime, verdict, and what
changes in `AGENTS.md`.

## Decisions and rationale (distilled from design discussion)

- **Both tasks parametric from day one** (MQAR `N_KV,T`; addition `digits,
  addends`): the interesting result may be a cross-task tradeoff, so the harness
  must be task-symmetric or the comparison is confounded by an MQAR-fit protocol.
- **Fix `head_dim`**: it is the capacity ceiling (`N_KV ~ head_dim`), so it is a
  deliberate benchmark design pick, not a swept knob.
- **One shared recipe, measured jointly across archs**: fairness beats
  optimality — a slightly suboptimal ratio applied identically is fine; an
  arch-flattering ratio is fatal. Per-task ratios allowed; per-arch ratios not.
- **Equal HPO budget per arch**: "best found" is only defensible if every arch
  got the same number of trials and the same search-space structure.
- **Dual budget controls** (iso-param + iso-FLOP) are the standard for the main
  grids; a separation that survives both is real, one that flips is a budget
  artifact. The pilot uses iso-param only.
- **Select on val, report test at best-val**: the current code selects on test
  (early-stops and scores on `test_acc`); that is the methodology bug this fixes.
- **Offline fixed pool now**: keeps MQAR comparable to Zoology and is fine for a
  recipe study that does not measure sample efficiency.

## Deferred (designed-for, not built now)

- **Iso-train-FLOP** matching and per-model FLOP estimators (main grids).
- **Full `(N_KV, T)` and `(digits, addends)` grids** -> per-arch heatmaps.
- **Online/streaming generation** and **OOD length-generalization splits**,
  required for the addition sample-efficiency and extrapolation readouts; not
  for this pilot. The harness should be designed knowing they are coming.
- **muP / lr scale-transfer**: start with coarse per-scale lr re-tune; adopt muP
  only if transfer proves unstable across the scale range we use.

## Implementation surface (to be scoped next, one goal per PR)

- Addition task (registry entry, offline splits, exact-match metric).
- Train/val/test split + val-based selection in `fit`.
- Arch- and task-parametric run/sweep entrypoint (replace the Titans-hardcoded
  `experiments/sweep_titans_toy.py`).
- Iso-parameter budget-matching helper (solve `dim` for a param target at fixed
  `head_dim` and shape).
- Foundation cleanup that this unblocks: retention -> executor port, `cache.py`
  decision, CI.
