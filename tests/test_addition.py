"""Addition task generation and the shared complete/partial accuracy metrics."""

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

import jax
import jax.numpy as jnp
import numpy as np

from linattn.metrics import accuracies, complete_accuracy, partial_accuracy
from linattn.tasks.addition import (
    PAD_ID,
    VOCAB_SIZE,
    AdditionConfig,
    addition_example,
    decode,
    encode,
    seq_len_for,
)
from linattn.tasks.base import build_task

CFG = AdditionConfig(max_digits=1, num_addends=2, n_train=8, n_test=4)


class AdditionTaskTest(unittest.TestCase):
    def test_build_task_dispatch(self):
        task = build_task(CFG)
        self.assertEqual(task.vocab_size, VOCAB_SIZE)
        self.assertEqual(task.n_train, 8)
        self.assertEqual(task.sources, ("linattn/tasks/addition.py",))

    def test_seq_len_worst_case(self):
        # "9+9=18" -> 1+1+1+1+2 = 6
        self.assertEqual(seq_len_for(1, 2), 6)
        # "9999+9999+9999=29997" -> 20
        self.assertEqual(seq_len_for(4, 3), 20)

    def test_split_shapes(self):
        task = build_task(CFG)
        tokens, targets, mask = task.make_split(jax.random.PRNGKey(0), task.n_train)
        T = seq_len_for(1, 2)
        self.assertEqual(tokens.shape, (8, T))
        self.assertEqual(targets.shape, (8, T))
        self.assertEqual(mask.shape, (8, T))
        # every example scores at least one answer digit
        self.assertTrue(bool((mask.sum(axis=1) >= 1).all()))

    def test_alignment_known_example(self):
        # Force the problem "2+3=5": tokens left-padded to width 6.
        class FixedRng:
            def integers(self, low, high, size):
                return np.array([2, 3])

        tokens, targets, mask = addition_example(FixedRng(), CFG)
        self.assertEqual(decode(tokens.tolist()), "_2+3=5".replace("_", decode([PAD_ID])))
        # The scored position is the '=' (logit there predicts the answer '5').
        scored = np.nonzero(mask)[0]
        self.assertEqual(scored.tolist(), [4])
        self.assertEqual(targets[4], encode("5")[0])

    def test_deterministic_and_disjoint_keys(self):
        task = build_task(CFG)
        a = task.make_split(jax.random.PRNGKey(0), 8)
        b = task.make_split(jax.random.PRNGKey(0), 8)
        c = task.make_split(jax.random.PRNGKey(1), 8)
        self.assertTrue(bool((a[0] == b[0]).all()))  # same key -> same data
        self.assertFalse(bool((a[0] == c[0]).all()))  # different key -> different


class MetricsTest(unittest.TestCase):
    def test_complete_vs_partial(self):
        # Two examples, two scored positions each.
        # Ex 0: both correct -> complete. Ex 1: one of two correct -> not complete.
        targets = jnp.array([[5, 7], [1, 2]])
        preds = jnp.array([[5, 7], [1, 9]])
        mask = jnp.array([[1.0, 1.0], [1.0, 1.0]])
        self.assertAlmostEqual(float(partial_accuracy(preds, targets, mask)), 3 / 4)
        self.assertAlmostEqual(float(complete_accuracy(preds, targets, mask)), 1 / 2)

    def test_mask_excludes_unscored(self):
        # A wrong prediction at an unscored position must not count.
        targets = jnp.array([[5, 7]])
        preds = jnp.array([[9, 7]])
        mask = jnp.array([[0.0, 1.0]])
        self.assertAlmostEqual(float(partial_accuracy(preds, targets, mask)), 1.0)
        self.assertAlmostEqual(float(complete_accuracy(preds, targets, mask)), 1.0)

    def test_accuracies_from_logits(self):
        # logits argmax -> [[1, 0]]; targets [[1, 1]], both scored.
        logits = jnp.array([[[0.1, 0.9], [0.8, 0.2]]])
        targets = jnp.array([[1, 1]])
        mask = jnp.array([[1.0, 1.0]])
        out = accuracies(logits, targets, mask)
        self.assertAlmostEqual(float(out["partial_accuracy"]), 1 / 2)
        self.assertAlmostEqual(float(out["accuracy"]), 0.0)


if __name__ == "__main__":
    unittest.main()
