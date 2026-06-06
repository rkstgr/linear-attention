# Proposal NNNN: <name>

> Answers [DIRECTION.md](../../DIRECTION.md) question <Q> (or: standalone).

Two paths. Use **Lite** for mechanical changes that touch no experiment, metric,
claim, or evaluation. Use **Full** for anything that affects what we measure or
claim.

---

## Lite

- **Goal** — one sentence.
- **Files** — the handful that change.
- **Validation** — the command or test that proves it.

If you find yourself needing a confound, a budget, or a claim, it is not Lite.
Use Full.

---

## Full

### Goal

One sentence; the single goal.

### Non-goals

What this explicitly will not do.

### Sprinter proposal

Smallest useful implementation: main idea, expected files, fastest validation
path, and any shortcuts acceptable for this phase.

### Skeptic review

Failure modes and required guardrails: possible confounds, the **null** each
metric must beat, the **regime** the claim lives in (and what would falsify it at
~10× size/length), reproducibility requirements (seeds, git hash, provenance),
and the claims this would **not** yet support. At least one concrete objection —
or "no blocking objection, because ...".

### Final settlement

The concrete plan after Sprinter folds in the minimal guardrails.

### Budget

The compute this authorizes: number of runs, seeds, and rough wall-clock or
GPU-hours. Equal budgets across architectures unless stated otherwise.

### Files expected to change

- `path/to/file.py`

### Validation

How the change is checked: unit test, smoke run, golden forward comparison, W&B
dry run, or a manual inspection command. **Any new task must overfit a single
batch to ~0 loss before it touches a compute budget.**

### Claim (pre-registered)

The single claim this is allowed to support, written **before** running it, plus:

- the **metric** and the locked eval set that decides it;
- the **null** it must beat (chance / baseline — a number without its null is not
  a result);
- the **regime** it holds in (size, length, task tier) and the smallest
  experiment that would falsify it at ~10× — so a toy result is never mistaken
  for a mechanism law.

If this only prepares infrastructure, say so. The result is recorded against this
claim in [`../experiments/`](../experiments/); do not upgrade the claim after
seeing the numbers.

### Follow-ups

Related work that stays out of this PR.
