# Agent Instructions

Last reconciled with owner input: 2026-06-06.

This file is the compact operating guide for the repo. Keep it current.

## North Star

Build a mechanism research workbench for sequence-mixing architectures.

The repo should make it easy to compare different architectures like softmax attention, 
linear attention, DeltaNet, gated DeltaNet, Titans, and future variants with enough rigor that a
W&B comparison is believable, but not so much process that iteration slows down.

## Current Goal

Clean the foundation before the next research push.

Foundation means:

- shared model scaffold and registries;
- local executor and provenance;
- task abstraction;
- central config shape;
- clear README/setup/test commands;
- cleanup or deletion of legacy paths.

Done: scaffold/registries, executor, capacity executor port, parity tests,
executor tests.

Not done: task abstraction, central config module, README/setup polish, retention
executor port, CI, and legacy `cache.py` decision.

## Research Context

Recent work: Titans sweeps on toy/easy/level1 landed in W&B.

Next research target: run comparable W&B sweeps for the other architectures and
compare them against Titans.

Do not start that comparison until the foundation is clean enough that arch/task
config, provenance, and test coverage are not a source of doubt.

## Minimal Process

Move fast, with structure.

For ordinary engineering work:

1. pick one goal;
2. make the smallest inspectable change;
3. run the smallest useful validation;
4. update this file, `DIRECTION.md`, `README.md`, or `docs/proposals/ROADMAP.md`
   only when reality changed.

Use a proposal only when work affects a research claim, metric, task definition,
training/eval protocol, W&B sweep, or compute budget. A proposal can be short:
goal, claim, metric/null, budget, guardrails, validation.

For experiments:

- record config, seeds, git commit, W&B links, metric, null/baseline, and regime;
- select on validation, report final numbers on held-out test;
- keep toy-scale results labeled toy-scale;
- every new task must overfit one batch before sweeps.

## Active Queue

1. Task/config foundation.
2. README/setup/test cleanup.
3. Retention executor port and `cache.py` decision.
4. CI for parity and executor tests.
5. Record existing Titans W&B sweeps and historical toy results.
6. Comparable W&B sweeps for non-Titans architectures.

Keep one goal per PR. Use `pr/<short-goal>` for foundation work and
`pr/<number>-<short-goal>` when a numbered proposal exists.

Do not push directly to `main` unless the user explicitly asks for it.
