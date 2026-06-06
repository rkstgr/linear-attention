# Sprinter/Auditor Method

This repo uses a lightweight proposal workflow before implementation. The goal
is to preserve research speed while making architecture comparisons defensible.

## Roles

- Sprinter owns iteration speed.
- Auditor owns empirical rigor.

The selected plan is the smallest implementation that lets both roles do their
job.

## Workflow

1. State one goal or requirement.
2. Sprinter writes the smallest viable proposal.
3. Auditor reviews the proposal for misleading comparisons, hidden confounds,
   reproducibility gaps, and missing controls.
4. Sprinter incorporates the minimal guardrails required by the Auditor.
5. The final settlement becomes the implementation plan.
6. One PR implements exactly that settlement.

Use this rule of thumb:

```text
Personas debate proposals.
PRs implement settlements.
```

## PR Scope

Every PR must have one clear goal. If a PR title needs "and", it is probably
two PRs.

Good PR goals:

- Extract shared model scaffold.
- Add task interface for MQAR.
- Add addition task.
- Add generic W&B sweep runner.
- Add validation/test split.

Bad PR goal:

- Refactor repo and add addition and improve sweeps.

## Required Proposal Sections

Each proposal should cover:

- Goal
- Non-goals
- Sprinter proposal
- Auditor review
- Final settlement
- Expected files
- Validation
- Claim enabled

Use [PROPOSAL_TEMPLATE.md](PROPOSAL_TEMPLATE.md).

## Git And Worktrees

Git is the boundary between ideas.

- One branch/PR implements one settled proposal.
- W&B runs and cached results should record the git commit hash.
- Use worktrees when multiple agents, experiments, or PRs run in parallel.

Suggested branch names:

```text
codex/0000-process
codex/0001-golden-tests
codex/0002-model-scaffold
```

Suggested worktree pattern:

```sh
git worktree add ../linear-attention-0001 -b codex/0001-golden-tests
```

Do not use worktrees for every tiny edit. Use them when there is real
parallelism or a PR boundary.

## Parallel Work

Proposals may be drafted in parallel after the process docs exist. Implementation
PRs should respect dependencies:

```text
process
  -> proposals
  -> golden tests
  -> model scaffold / registry
  -> task interface
  -> addition + validation protocol
  -> generic sweeps
  -> comparison runs
  -> error analysis
```

Implementation can run in parallel only when branches do not depend on each
other's code.

## Decision Rule

For each Auditor objection, ask:

```text
What is the smallest added structure that prevents this failure?
```

For each Sprinter objection, ask:

```text
What common workflow did this make slower or harder?
```

The final plan should keep modules simple and make experiment launch strict.

