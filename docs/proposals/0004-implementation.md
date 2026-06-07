# 0004 Implementation Runbook

Tracks the build for proposal `0004-scaling-recipe.md`. Parts were prepared in a
**no-network session** (PyPI blocked, so JAX could not be installed and the
jitted tests could not run). Pure logic was validated with stdlib reimplementations.
This runbook is the handoff: what is done, what is unverified, and the exact spec
for what remains — to be finished in a **session with network access** where every
step is immediately testable.

## Environment setup (network session)

```sh
uv sync
uv sync --group experiment
uv run python -m unittest discover -v
```

Everything below assumes CPU (`JAX_PLATFORMS=cpu`, the repo default).

## Status

| Item | State | Validate |
| --- | --- | --- |
| A. Parametric addition task | prepared (logic-checked) | `uv run python -m unittest tests.test_addition -v` |
| B. Shared complete/partial metrics | prepared (logic-checked) | same as A (`MetricsTest`) |
| C. Draw-once-then-partition splits (both tasks, de-duplicated) | prepared (logic-checked) | `tests.test_addition`, `tests.test_tasks` |
| D. Val-based early stopping in `fit` | prepared | `uv run python -m unittest tests.test_train -v` |
| E. Iso-parameter budget matcher | prepared | `uv run python -m unittest tests.test_budget -v` |
| F. Dual metrics into history + W&B | TODO (spec below) | — |
| G. Test-at-best-val reporting | TODO (spec below) | — |
| H. Arch+task-parametric sweep entrypoint | TODO (spec below) | — |
| I. `experiments/recipe.py` driver | TODO (spec below) | — |
| J. Retention -> executor port; delete `cache.py`; CI | TODO | — |

First action in the network session: **run the full suite**. The prepared items
(A-E) should pass as-is; if not, they are small and self-contained.

## Prepared this session (A-E)

- **A. Addition task** — `linattn/tasks/addition.py`. Causal next-token, scored on
  answer digits, parametric on `(max_digits, num_addends)`; ported into the repo's
  position-aligned `(tokens, targets, mask)` contract.
- **B. Metrics** — `linattn/metrics.py`. `accuracy` (complete / exact-match) and
  `partial_accuracy` (per scored position), shared by both tasks.
- **C. Splits** — `make_splits` on both tasks draws one pool *without redrawing*
  then partitions into disjoint train/val/test. Addition de-duplicates on operand
  tuples; MQAR de-duplicates on token sequences (a near no-op given its space).
  `n_val=0` omits val. Added `n_val` to both configs and the `Task` protocol.
- **D. Val early stopping** — `fit` uses `task.make_splits` and early-stops /
  tracks best on **val** accuracy when `n_val>0`; otherwise the legacy two-way
  path runs with byte-identical keying so existing cached runs (capacity,
  retention) are unchanged. `val_acc` is added to history.
- **E. Budget matcher** — `linattn/budget.py`. `solve_dim_for_params` builds and
  counts models exactly to find the `dim` (multiple of `head_dim`) closest to a
  param target. CLI prints the recipe-shape dims:
  `uv run python -m linattn.budget --targets 500000 1000000 --layers 1 2 4 --vocab 512 --head-dim 16`.
  Use its output to fix the per-(arch, shape) dims for the recipe study.

Backbone embeddings are **untied** (`tok_emb` + separate `lm_head`), identical
across all mixers, so the "embedding share held constant across archs" fairness
condition holds by construction.

## TODO specs

### F. Dual complete/partial metrics into history + W&B

Goal: log `{train,val,test}_accuracy` (complete) and `{train,val,test}_partial_accuracy`.

- Add `evaluate_metrics(model, data, batch_size) -> {"accuracy", "partial_accuracy"}`
  in `train.py`, accumulating exact numerator/denominator across batches (reuse
  `metrics.accuracies`). Keep `evaluate` (partial-only) for the legacy path.
- In the epoch loop compute metrics for train(eval)/val/test; write all six keys
  to `history`.
- Reporter signature change: `on_epoch(... )` currently takes `test_acc`. Widen to
  pass the metric dict (or per-split kwargs) and update `StdoutReporter`,
  `WandbReporter`, `MultiReporter`, and the W&B keys in `experiments/sweep_titans_toy.py`.
- **Decision to confirm:** selection metric. Current prepared code early-stops on
  val **partial** accuracy (smooth, matches the legacy `target_acc` semantics).
  Complete accuracy can sit at 0 then jump, which is poor for patience. Recommend:
  select on val partial, report complete as the headline. Flip by changing the
  scalar fed to `stopper.update`.

### G. Test-at-best-val reporting

Report the test metric at the epoch with the best **val** (not best test, not
final). Two options: snapshot the best-val model during the loop (memory cost), or
post-hoc pick `argmax val_acc` from history and read its test metric. Post-hoc is
enough for the pilot; do it in `runner.train_run` when writing `metrics.json`.

### H. Arch + task-parametric sweep entrypoint

Replace the Titans-hardcoded `experiments/sweep_titans_toy.py` with one entrypoint
driven by an **arch registry**: `mixer -> {extra_hyperparams, build_kwargs, flop_estimator}`.
For the pilot only `lr` (+ Titans `memory_mult`, `max_inner_lr`) is searched. The
FLOP estimator is **deferred** (iso-FLOP is not in the pilot); stub it for now.
Keep the W&B metric grouping and the `objective/score` convention.

### I. `experiments/recipe.py` driver (the pilot)

Executor-backed (like `experiments/capacity.py`). Cells = arch x shape x task x
seed x lr. Stages, to keep equal-HPO-budget cheap:

1. lr selection: 5 archs x 3 shapes x 2 tasks x 3 lr x 1 seed, select lr per
   (arch, shape, task) on **val**.
2. recipe + seeds: best lr, 3 seeds; pick depth/width recipe per task (Pareto-best
   averaged across archs) + read the cross-task tradeoff gate.
3. stability: chosen recipe only, at the 1M budget, 3 seeds.

Use `budget.solve_dim_for_params` to fix dims; `head_dim=16`; cells MQAR
`N_KV=4,T=64,vocab=512` and addition 3-digit/2-addend; sizes `n_train=50k`,
`n_val=2k`, `n_test=2k` (knobs). Emit a summary table and a heatmap-ready JSON.

### J. Foundation cleanup

- Port `experiments/retention.py` to the executor (mirror `capacity.py`), removing
  the last `linattn/cache.py` consumer, then delete `cache.py`.
- Add CI (GitHub Actions) running `uv run python -m unittest discover` on push.

## Open decisions to confirm

- Selection metric: val **partial** accuracy (prepared default) vs val complete.
- `head_dim=16`, split sizes `50k/2k/2k` — current defaults, easily changed.
- Embeddings stay **untied** (current backbone) — uniform across archs.
- De-duplicated draw-once-then-partition: **decided** (this session), both tasks.

## Validation checklist (network session)

1. `uv sync && uv sync --group experiment`.
2. `uv run python -m unittest discover -v` — A-E green.
3. `uv run python -m linattn.budget ...` — sanity-check solved dims, fix recipe shapes.
4. Implement F-G, re-run tests, confirm capacity/retention numbers unchanged.
5. Implement H-I, smoke-run `experiments/recipe.py` at tiny sizes.
6. Run the pilot; record results in `docs/experiments/` per the ledger schema.
