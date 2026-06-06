# Proposals

Proposals are for research-affecting work, not every edit. See `../../AGENTS.md`
for the current process.

Use a proposal when a change can affect a claim, metric, task, benchmark,
training protocol, evaluation protocol, W&B sweep, or compute budget. Otherwise
use the default path in `AGENTS.md`.

## Naming

```text
NNNN-short-goal.md
```

Numbers are historical IDs, not execution order.

## Lifecycle

1. Write the smallest proposal that fixes the goal, claim, budget, and guardrails.
2. Implement one goal on one branch.
3. Record experimental results in `docs/experiments/` when a claim is tested.
4. Update `AGENTS.md` when the active queue or repo direction changes.

Historical proposals may be more verbose than the current standard.
