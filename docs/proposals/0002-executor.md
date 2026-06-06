# Proposal 0002: local executor

Status: landed in `main` via PR #6.

## Goal

Replace per-script cache plumbing with a small local executor for
content-addressed experiment steps.

## Claim

No research claim.

Infrastructure claim: for a fixed step function, config, dependency graph, source
set, seed, and git checkout, the executor derives a stable content address and
reuses the cached step output. Changing any hashed config value, dependency
digest, or source byte causes a cache miss.

## Final Shape

Implemented:

- `executor.py` with `SourceSet`, `ExecutorStep`, `this_output_path`,
  `output_path_of`, `executor_main`, and run manifests;
- local output directories under `.experiment_cache/steps/`;
- run manifests under `.experiment_cache/runs/`;
- source-sensitive digests;
- dependency digest flow;
- duplicate-step dedupe;
- spawned-process parallel execution;
- local lock files;
- `experiments/defaults.py` with partial config dataclasses and default train
  step builder;
- `experiments.capacity` ported to the executor;
- `tests/test_executor.py`.

Not implemented:

- a general `Task` protocol;
- central config module;
- retention port;
- W&B sweep integration through the executor;
- distributed execution or remote storage.

## Validation

Current local validation:

```sh
uv run python -m unittest discover -v
```

Executor tests cover:

- stable digest for identical inputs;
- digest change when config changes;
- digest change when a source byte changes;
- dependency digest flow;
- cache hit and rerun behavior;
- duplicate-step dedupe;
- output path resolution;
- parallel independent steps;
- manifest provenance.

## Follow-Ups

See `../../AGENTS.md` for the current active queue.
