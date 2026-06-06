# Proposals

Settled Sprinter/Auditor proposals — one goal each, one PR each. Proposals are
the **front-matter** (what we intend to do and claim); recorded results live in
[`../experiments/`](../experiments/) (what happened).

Each proposal implements one phase of `DESIGN.md`, so numbers track phase order.

## Naming

```text
0000-process.md
0001-golden-tests-and-scaffold-extraction.md
0002-architecture-registry.md
```

## Lifecycle

1. Draft from a single goal (usually "Phase N of DESIGN.md").
2. Add the Sprinter proposal.
3. Add the Auditor review (>= 1 concrete objection, or why none).
4. Settle: fold in the minimal guardrails.
5. Implement in one PR.
6. Record the result in [`../experiments/`](../experiments/) against the
   pre-registered claim.

## Scope rule

One goal per proposal. If it wants to solve two independent problems, split it.
Mechanical changes can use the Lite path in
[`../process/PROPOSAL_TEMPLATE.md`](../process/PROPOSAL_TEMPLATE.md).
