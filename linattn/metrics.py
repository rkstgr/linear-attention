"""Shared accuracy metrics over masked (scored) positions.

Two notions, both computed only at positions where ``mask == 1``:

- ``partial_accuracy`` — fraction of individual scored positions predicted
  correctly (micro-averaged over all scored positions). For MQAR this is the
  fraction of kv-query positions recalled; for addition the fraction of answer
  digits correct. This is the metric the training loop has always used.
- ``complete_accuracy`` — fraction of *examples* whose every scored position is
  correct (exact match). For addition this is "the whole answer is right"; for
  MQAR "every query in the sequence was recalled". This is the honest
  algorithmic readout.

Inputs are position-aligned with the model's logits: ``preds[..., t]`` is scored
against ``targets[..., t]`` wherever ``mask[..., t] == 1``. Shapes may be a
single example ``(T,)`` or a batch ``(B, T)``; reductions for
``complete_accuracy`` are over the last (time) axis only.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def partial_accuracy(preds: Array, targets: Array, mask: Array) -> Array:
    """Micro-averaged accuracy over all scored positions."""
    correct = (preds == targets).astype(jnp.float32) * mask
    return correct.sum() / jnp.maximum(mask.sum(), 1.0)


def complete_accuracy(preds: Array, targets: Array, mask: Array) -> Array:
    """Fraction of examples with *every* scored position correct (exact match).

    Examples with no scored positions are excluded from the denominator.
    """
    correct = (preds == targets).astype(jnp.float32) * mask
    scored = mask.sum(-1)
    per_example = (correct.sum(-1) == scored) & (scored > 0)
    denom = jnp.maximum((scored > 0).sum(), 1.0)
    return per_example.astype(jnp.float32).sum() / denom


def accuracies(logits: Array, targets: Array, mask: Array) -> dict[str, Array]:
    """Both metrics from logits. Keys: ``accuracy`` (complete), ``partial_accuracy``."""
    preds = jnp.argmax(logits, axis=-1)
    return {
        "accuracy": complete_accuracy(preds, targets, mask),
        "partial_accuracy": partial_accuracy(preds, targets, mask),
    }
