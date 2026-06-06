# Experiments

The back-matter of the process. Proposals pre-register a claim; this directory
records what actually happened, with the same discipline.

One file per settled proposal that runs something:

```text
0003-addition-task.md
0004-generic-sweeps.md
```

## Each entry records

- **Proposal / PR** — links back to the pre-registered claim.
- **Setup** — git commit hash, config(s), seeds, W&B run URLs, and the budget
  actually spent.
- **Result** — the numbers on the locked eval / test set, **each next to the null
  it beats** (chance / baseline), against the pre-registered claim.
- **Regime** — the size / length / task tier these numbers live in, and whether
  the pre-registered ~10× falsifier was checked — and what it showed.
- **Verdict** — supported / not supported / inconclusive. Negative results are
  first-class: "this architecture lost, here is the evidence" is a complete,
  valuable entry.
- **Confounds hit** — anything the Skeptic flagged that actually bit, and what
  it means for the claim.
- **Direction update** — the one-line change this makes to
  [`DIRECTION.md`](../../DIRECTION.md) (a bet moves status, a regime widens, a
  question splits) and how it reorders the backlog. Closing this loop is what the
  entry is *for*.
- **Decision** — what we do next (keep, drop, re-run with X).

## Rules

- Report the metric and eval set fixed in the proposal. If you changed them, the
  claim changed: note it and treat the result as exploratory, not confirmatory.
- Every number is reported with its null; an accuracy alone is not a result.
- A claim holds only in its stated regime; do not generalize past it without the
  falsifier.
- Test set is for final reporting only; selection happens on validation.
- A clean negative result is a success of the process, not a failure of the work.
