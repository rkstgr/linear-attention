# Sprinter/Auditor Method

A lightweight proposal workflow used before implementation. The goal is to keep
research fast while making architecture comparisons defensible.

Two reviewers, in tension by design:

- **Sprinter** owns iteration speed.
- **Auditor** owns empirical rigor.

The chosen plan is the smallest implementation that lets both do their job.

> Personas debate proposals. PRs implement settlements.

## Where this fits

- `DESIGN.md` (the technical roadmap) owns **what / why** — the architecture
  decisions, phases, and open questions.
- This process owns **how** — turning one phase into one PR.
- Each proposal implements **one** DESIGN phase, so proposal numbers track the
  phase order (`0001` = Phase 0, ...). One ordering, one source of truth.

## Workflow

1. State one goal (usually "do Phase N of DESIGN.md").
2. Sprinter writes the smallest viable proposal.
3. Auditor reviews for misleading comparisons, hidden confounds, reproducibility
   gaps, and missing controls, and files **at least one concrete objection** —
   or explicitly records "no blocking objection, because ...".
4. Sprinter folds in the minimal guardrails required.
5. The settlement becomes the implementation plan.
6. One PR implements exactly that settlement.
7. After the run, the result is recorded in [`../experiments/`](../experiments/)
   against the claim the proposal pre-registered.

### Keep the tension real

The method is only worth the paper if the two roles actually disagree. When
agents do the work, assign the roles to **different** models (one drafts as
Sprinter, another reviews as Auditor) rather than letting one voice rubber-stamp
itself. A review with zero objections should be the exception, and should say
why.

## The two roles

### Sprinter — research-velocity reviewer

Mission: keep the repo small, editable, and close to the model; new ideas should
be cheap to try.

Pushes back on:

- Framework creep and abstractions that hide model mechanics.
- Too many files touched for a simple idea.
- Slow feedback loops; config ceremony before a result can be inspected.
- Changes that make ad hoc ablations painful.

Asks:

- Can a researcher understand the change from the touched files?
- If we add an architecture tomorrow, how many files must change?
- Can we inspect an example and debug a failure quickly?
- Is this solving the current goal, or preparing for too many future ones?

### Auditor — empirical-rigor reviewer

Mission: make comparisons defensible — reproducible, fairly budgeted, and clear
about the claim they support.

Pushes back on:

- Optimizing on the test set; selection on anything but validation.
- Hidden architecture-specific advantages; incomparable HP / compute budgets.
- Missing seeds or git hashes; unclear W&B provenance.
- Metrics that saturate and hide differences.
- Sweep objectives that reward capacity when the goal is mechanism.
- Claims unsupported by the protocol.

Asks:

- What misleading result could this produce?
- Are model size, training budget, and HP-search budget comparable?
- Is validation used for selection and test reserved for final reporting?
- Are task-specific error cases observable?
- Could another researcher reproduce this from the PR alone?

## Decision rule

For each **Auditor** objection: *what is the smallest added structure that
prevents this failure?* For each **Sprinter** objection: *what common workflow
did this make slower?* The settlement keeps modules simple and experiment launch
strict.

When the two genuinely conflict and neither yields (e.g. five seeds x four archs
vs. iteration speed), the human breaks the tie and the trade-off is recorded in
the settlement. State the **budget** explicitly — compute is the real scarce
resource, not lines of code.

## PR scope

One clear goal per PR. If the title needs "and", it is probably two PRs.

- Good: "Extract shared model scaffold." / "Add addition task." / "Add generic
  W&B sweep runner."
- Bad: "Refactor repo and add addition and improve sweeps."

## Proposals and results

Settled proposals (the front-matter: intent and pre-registered claim) live in
[`../proposals/`](../proposals/); recorded results (the back-matter: what
happened) live in [`../experiments/`](../experiments/). Use
[PROPOSAL_TEMPLATE.md](PROPOSAL_TEMPLATE.md); mechanical changes may use its Lite
path.

## Git

Git is the boundary between ideas.

- One branch / PR per settled proposal; branch names track proposal numbers
  (`codex/0001-golden-tests`).
- W&B runs and cached results record the git commit hash.
- Use a worktree only when there is real parallelism (multiple agents or PRs at
  once), not for every edit.

## Dependencies

Proposals may be drafted in parallel once this process exists. Implementation
respects the DESIGN phase order:

```text
process -> proposals -> golden tests -> scaffold / registry -> task interface
        -> addition + validation protocol -> generic sweeps -> comparison runs
        -> error analysis
```

Branches can run in parallel only when they do not depend on each other's code.
