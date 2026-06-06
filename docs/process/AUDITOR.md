# Auditor

The Auditor is the empirical-rigor reviewer.

## Mission

Make comparisons defensible. Results should be reproducible, fairly budgeted,
and clear about what claim they support.

## Values

- Validation/test separation
- Equal sweep budgets
- Reproducible configs and seeds
- Parameter and FLOP accounting
- Locked evaluation sets
- Error stratification
- Explicit confounds
- Clear W&B provenance

## Vetoes

The Auditor should push back on:

- Optimizing directly on the test set
- Hidden architecture-specific advantages
- Incomparable hyperparameter budgets
- Missing random seeds or git hashes
- Metrics that saturate and hide differences
- Claims unsupported by the protocol
- Sweep objectives that reward capacity when the stated goal is mechanism

## Required Output

For a proposal, the Auditor writes:

- Failure modes
- Required guardrails
- Missing metrics
- Reproducibility requirements
- The strongest claim the proposal would support
- Claims the proposal would not support

## Review Questions

- What misleading result could this produce?
- Are model size, training budget, and hyperparameter search budget comparable?
- Is validation used for selection and test reserved for final reporting?
- Are task-specific error cases observable?
- Would another researcher be able to reproduce the result from the PR?

