"""Task registry and MQAR generation tests."""

import os
import unittest
from dataclasses import dataclass

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

import jax

from linattn.tasks.base import build_task
from linattn.tasks.mqar import MQARConfig, MQARTask

CFG = MQARConfig(
    vocab_size=64,
    input_seq_len=16,
    num_kv_pairs=1,
    power_a=0.01,
    n_train=8,
    n_test=4,
)


class TaskRegistryTest(unittest.TestCase):
    def test_build_task_dispatch(self):
        task = build_task(CFG)
        self.assertIsInstance(task, MQARTask)
        self.assertEqual(task.vocab_size, 64)
        self.assertEqual(task.n_train, 8)
        self.assertEqual(task.n_test, 4)
        self.assertEqual(task.sources, ("linattn/tasks/mqar.py",))

    def test_unknown_task_raises(self):
        @dataclass(frozen=True)
        class Bad:
            name: str = "nope"

        with self.assertRaises(ValueError):
            build_task(Bad())


class MQARSplitTest(unittest.TestCase):
    def test_make_split_shapes_and_mask(self):
        task = build_task(CFG)
        tokens, targets, mask = task.make_split(jax.random.PRNGKey(0), task.n_train)
        self.assertEqual(tokens.shape, (8, 16))
        self.assertEqual(targets.shape, (8, 16))
        self.assertEqual(mask.shape, (8, 16))
        # exactly num_kv_pairs query positions scored per example
        self.assertTrue(bool((mask.sum(axis=1) == CFG.num_kv_pairs).all()))

    def test_make_split_deterministic(self):
        task = build_task(CFG)
        a = task.make_split(jax.random.PRNGKey(0), 8)
        b = task.make_split(jax.random.PRNGKey(0), 8)
        self.assertTrue(bool((a[0] == b[0]).all()))


class MQARSplitsTest(unittest.TestCase):
    CFG = MQARConfig(
        vocab_size=64,
        input_seq_len=16,
        num_kv_pairs=1,
        power_a=0.01,
        n_train=20,
        n_val=8,
        n_test=8,
    )

    def _rows(self, split):
        return {tuple(int(x) for x in row) for row in split[0].tolist()}

    def test_splits_disjoint_and_sized(self):
        splits = build_task(self.CFG).make_splits(jax.random.PRNGKey(0))
        self.assertEqual(set(splits), {"train", "val", "test"})
        self.assertEqual(splits["train"][0].shape[0], 20)
        self.assertEqual(splits["val"][0].shape[0], 8)
        self.assertEqual(splits["test"][0].shape[0], 8)
        tr, va, te = (self._rows(splits[k]) for k in ("train", "val", "test"))
        self.assertEqual(tr & va, set())
        self.assertEqual(tr & te, set())
        self.assertEqual(va & te, set())

    def test_no_val_when_zero(self):
        cfg = MQARConfig(
            vocab_size=64, input_seq_len=16, num_kv_pairs=1, power_a=0.01,
            n_train=20, n_test=8,
        )
        splits = build_task(cfg).make_splits(jax.random.PRNGKey(0))
        self.assertEqual(set(splits), {"train", "test"})


if __name__ == "__main__":
    unittest.main()
