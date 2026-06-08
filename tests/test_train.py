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


class FitValPathTest(unittest.TestCase):
    """`fit` selects on val when the task defines one, else stays legacy."""

    def _run(self, n_val):
        import jax

        from linattn.config import TrainConfig
        from linattn.models.factory import build_lm_model
        from linattn.tasks.base import build_task
        from linattn.tasks.mqar import MQARConfig
        from linattn.train import Reporter, fit

        cfg = MQARConfig(
            vocab_size=64, input_seq_len=16, num_kv_pairs=1, power_a=0.01,
            n_train=16, n_val=n_val, n_test=8,
        )
        task = build_task(cfg)
        train = TrainConfig(
            batch_size=8, eval_batch_size=8, max_epochs=1,
            learning_rate=1e-3, target_acc=1.1, patience_epochs=5,
        )
        model = build_lm_model("linear_attention", 64, 16, 2, 1, 4, jax.random.PRNGKey(0))
        return fit(model, task, train, jax.random.PRNGKey(1), reporter=Reporter())

    def test_val_present_logs_val_metrics(self):
        result = self._run(n_val=8)
        self.assertIn("val_partial_accuracy", result.history[0])
        self.assertIn("val_accuracy", result.history[0])
        # both train and test metrics are always present in dual-metric mode
        self.assertIn("train_accuracy", result.history[0])
        self.assertIn("test_accuracy", result.history[0])

    def test_no_val_is_legacy(self):
        result = self._run(n_val=0)
        self.assertNotIn("val_partial_accuracy", result.history[0])
        # test/train still dual-metric in the two-way path
        self.assertIn("test_partial_accuracy", result.history[0])


if __name__ == "__main__":
    unittest.main()
