# Sprinter/Skeptic Method

A lightweight proposal workflow used before implementation. The goal is to keep
research fast while making architecture comparisons hard to fool yourself with.

The question being answered comes from [`DIRECTION.md`](../../DIRECTION.md) — the
researcher owns *what is worth asking*. Two reviewers then pull the answer into
shape, in tension by design:

- **Sprinter** strives for elegance and simplicity — the smallest, most
  inspectable change that answers the question.
- **Skeptic** strives for truth — that the result is real, not an artifact and
  not self-deception. Reproducibility is a *means* to that end, not the end.

The chosen plan is the smallest implementation that lets both do their job.

> The researcher picks the question. Personas debate the answer. PRs implement
> the settlement; experiments update the direction.

## Operating principles

1. **Lite by default; Full when you make a claim.** Stay fast and loose while
   exploring. Invoke the heavy ritual (pre-registration, equal budgets) only when
   a PR is about to *assert* something — not for every edit.
2. **Make it work → make it right → make it real.** Get a dumb baseline running,
   then make it correct, then push it to a regime someone cares about. "Fast"
   (kernel optimization) is premature until a mechanism has earned it.
3. **Overfit one batch before you sweep.** Any new task must drive a single batch
   to ~0 loss before it touches a compute budget — the cheapest possible
   alignment/plumbing check.
4. **Results reorder the plan.** The experiment ledger updates `DIRECTION.md` and
   re-derives the backlog. Compute is the scarce resource, not lines of code.

## Where this fits

- [`DIRECTION.md`](../../DIRECTION.md) holds the living **what / why** — the open
  questions and current bets. It is not a spec; it changes as experiments land.
- This process owns **how** — turning one open question into one PR.
- Each proposal answers **one** open question. Proposal numbers are monotonic
  ids, **not** a master plan: the next number is whatever the experiment ledger
  says is worth doing next, not a pre-locked phase.

## Workflow

1. State one goal — usually "answer the next open question in `DIRECTION.md`".
2. Sprinter writes the smallest viable proposal.
3. Skeptic reviews for misleading comparisons, hidden confounds, missing
   controls, a metric with no stated null, and claims that quietly assume scale,
   and files **at least one concrete objection** — or explicitly records "no
   blocking objection, because ...".
4. Sprinter folds in the minimal guardrails required.
5. The settlement becomes the implementation plan.
6. One PR implements exactly that settlement.
7. After the run, the result is recorded in [`../experiments/`](../experiments/)
   against the claim the proposal pre-registered.
8. The experiment entry updates [`DIRECTION.md`](../../DIRECTION.md) (a bet moves,
   a regime widens) and the backlog is re-derived. This loop — not the proposal
   ritual — is what makes the process iterative rather than a small-step
   waterfall.

### Keep the tension real

The method is only worth the paper if the two roles actually disagree. When
agents do the work, assign the roles to **different** models (one drafts as
Sprinter, another reviews as Skeptic) rather than letting one voice rubber-stamp
itself. A review with zero objections should be the exception, and should say
why.

## The two roles

### Sprinter — elegance and simplicity

Mission: the smallest, most inspectable change that answers the question; keep
the repo close to the model so new ideas are cheap to try.

Pushes back on:

- Framework creep and abstractions that hide model mechanics.
- Adopting a tool *on a schedule* rather than when the hand-rolled version
  actually hurts — let felt pain pull the dependency in.
- Too many files touched for a simple idea.
- Slow feedback loops; config ceremony before a result can be inspected.
- Changes that make ad hoc ablations painful.

Asks:

- Can a researcher understand the change from the touched files?
- If we add an architecture tomorrow, how many files must change?
- Can we inspect an example and debug a failure quickly?
- Is this solving the current goal, or preparing for too many future ones?

### Skeptic — truth (do not fool yourself)

Mission: make sure the result is *real* — not an artifact, not self-deception,
not a toy-scale coincidence. "The first principle is that you must not fool
yourself, and you are the easiest person to fool." Reproducibility, fair budgets,
and provenance are the *means*; truth is the end.

Pushes back on:

- A number reported with no **null**: every metric needs the chance/baseline it
  beats (e.g. "~`1/N_KV` random pick"). Reproducible ≠ true.
- A claim that quietly assumes **scale**: a toy-regime result stated as a
  mechanism law, with no named regime and no falsifier at larger size/length.
- Optimizing on the test set; selection on anything but validation.
- Hidden architecture-specific advantages; incomparable HP / compute budgets.
- Missing seeds or git hashes; unclear W&B provenance.
- Metrics that saturate and hide differences.
- Sweep objectives that reward capacity when the goal is mechanism.

Asks:

- What is the null, and does the number actually beat it?
- In what **regime** is this true, and what is the smallest experiment that would
  falsify it at ~10× width or length?
- What misleading result could this produce?
- Are model size, training budget, and HP-search budget comparable?
- Is validation used for selection and test reserved for final reporting?
- Are task-specific error cases observable, and did you look at one by hand?
- Could another researcher reproduce this from the PR alone?

## Decision rule

For each **Skeptic** objection: *what is the smallest added structure that
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
  (`pr/0001-golden-tests`).
- W&B runs and cached results record the git commit hash.
- Use a worktree only when there is real parallelism (multiple agents or PRs at
  once), not for every edit.

## Dependencies

Proposals may be drafted in parallel once this process exists. Implementation
respects **dependency constraints**, not a fixed phase order — a later question
can jump the queue if the ledger says it matters more, as long as its code
prerequisites exist:

```text
scaffold / registry ──► task interface ──► sweeps ──► comparison runs
        (everything downstream needs the scaffold; sweeps need a task)
```

Branches can run in parallel only when they do not depend on each other's code.
