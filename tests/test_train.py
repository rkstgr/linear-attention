"""Unit tests for the decomposed training pieces."""

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

from linattn.train import EarlyStopping


class EarlyStoppingTest(unittest.TestCase):
    def test_target_acc_stops_immediately(self):
        stopper = EarlyStopping(target_acc=0.9, patience_epochs=5)
        self.assertEqual(stopper.update(0.95), "target_acc")

    def test_patience_stops_after_no_improvement(self):
        stopper = EarlyStopping(target_acc=0.99, patience_epochs=2)
        self.assertIsNone(stopper.update(0.5))  # improves 0 -> 0.5
        self.assertIsNone(stopper.update(0.5))  # no improvement, patience 1
        self.assertEqual(stopper.update(0.5), "patience")  # patience 2 -> stop

    def test_improvement_resets_patience(self):
        stopper = EarlyStopping(target_acc=0.99, patience_epochs=2)
        self.assertIsNone(stopper.update(0.5))
        self.assertIsNone(stopper.update(0.5))  # patience 1
        self.assertIsNone(stopper.update(0.6))  # improvement resets
        self.assertIsNone(stopper.update(0.6))  # patience 1
        self.assertEqual(stopper.update(0.6), "patience")


if __name__ == "__main__":
    unittest.main()
