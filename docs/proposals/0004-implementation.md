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
| F. Dual metrics into history + W&B | done | `uv run python -m unittest tests.test_train -v` |
| G. Test-at-best-val reporting | done | `headline.test_*` in `metrics.json` |
| H. Arch+task-parametric sweep entrypoint | done | `experiments/sweep.py`, `linattn/archs.py` |
| I. `experiments/recipe.py` driver | done (smoke-tested) | `uv run python -m experiments.recipe --stage lr_selection --smoke` |
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

## Built this session (F-I)

- **F. Dual metrics** — `linattn/train.py` gains `evaluate_metrics(model, data,
  batch_size)` returning `{"accuracy", "partial_accuracy"}`, accumulating exact
  numerator/denominator across batches. `evaluate` (partial-only) is retained
  for callers that only need partial. The epoch loop computes metrics for
  train(eval)/val/test and writes six history keys
  (`{train,val,test}_accuracy`, `{train,val,test}_partial_accuracy`). Reporter
  signatures widened — `on_epoch(*, epoch, global_step, train_loss, metrics)`
  takes the per-split dict; `StdoutReporter`/`WandbReporter`/`MultiReporter`
  updated. W&B logs `learning/<split>_accuracy` and
  `learning/<split>_partial_accuracy`. Selection metric: val partial when
  present, else test partial (legacy).
- **G. Test-at-best-val** — `runner._headline_from_history` post-hoc picks the
  epoch with the highest selection metric (`val_partial_accuracy` when val
  exists, `test_partial_accuracy` otherwise) and records its test metrics under
  `metrics["headline"]`. `metrics["best"]` stays the same partial test scalar
  capacity/retention callers already read.
- **H. Generic sweep entrypoint** — `linattn/archs.py` is the arch registry
  (`mixer -> ArchSpec{extra_hyperparams, flop_estimator stub}`); Titans has
  `{memory_mult=4, max_inner_lr=5e-3}` as the searched extras. `experiments/sweep.py`
  replaces the deleted `sweep_titans_toy.py`: reads `mixer`, `task`
  (`mqar`/`addition`), `dim`/`n_heads`/`head_dim`/`n_layers`/`mlp_mult`,
  `learning_rate`, `seed`, per-task knobs, and (for MQAR) the `config_name`
  preset shortcut. The three `sweeps/titans_*.yaml` files now point at
  `experiments.sweep` with `mixer: titans, task: mqar` injected.
- **I. Pilot driver** — `experiments/recipe.py` runs the three stages
  executor-backed: `lr_selection`, `recipe`, `stability`. Cells solved via
  `budget.solve_dim_for_params` per (arch, n_layers, vocab, target). Defaults:
  `head_dim=16`, MQAR cell `N_KV=4/T=64/vocab=512`, addition 3-digit/2-addend,
  `n_train=50k/n_val=2k/n_test=2k`, lr grid `(3e-4, 1e-3, 3e-3)`, seeds
  `(1,2,3)`, targets 500k (pilot) and 1M (stability). `--smoke` shrinks
  task/train sizes and the param target to 20k for orchestration checks.
  Stage handoff goes through `.experiment_cache/runs/recipe_summary_*.json`
  (best lr lookup, picked recipe). Each stage prints a markdown table; the
  recipe stage also prints the cross-task tradeoff ranking.

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

1. `uv sync && uv sync --group experiment`. **done**
2. `uv run python -m unittest discover -v` — 31 tests, A-E green. **done**
3. `uv run python -m linattn.budget --targets 500000 1000000 --layers 1 2 4 --vocab 512 --head-dim 16`
   — sanity-check solved dims, fix recipe shapes.
4. F-G implemented; tests green; capacity/retention numbers unchanged (legacy
   path keys via `headline.test_partial_accuracy`).
5. H-I implemented; orchestration smoke-tested (4-cell mini-smoke at
   `target_params=20k`, 2 epochs, n_train=512). End-to-end JIT compile,
   headlines, and aggregation verified.
6. Run the pilot — `uv run python -m experiments.recipe --stage all --parallel <N>`.
   Then record results in `docs/experiments/` per the ledger schema.
