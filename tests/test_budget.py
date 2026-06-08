"""Iso-parameter budget matcher: monotonicity and closest-to-target solving."""

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

from linattn.budget import count_for_shape, solve_dim_for_params

COMMON = dict(vocab_size=512, head_dim=16, n_layers=2, mlp_mult=4)


class BudgetTest(unittest.TestCase):
    def test_count_monotone_in_dim(self):
        counts = [
            count_for_shape(mixer="transformer", dim=d, **COMMON)
            for d in (16, 32, 64, 128)
        ]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(len(set(counts)), len(counts))  # strictly increasing

    def test_solve_hits_target_closely(self):
        target = 500_000
        r = solve_dim_for_params(target, mixer="transformer", **COMMON)
        self.assertEqual(r["dim"] % COMMON["head_dim"], 0)
        self.assertEqual(r["n_heads"], r["dim"] // COMMON["head_dim"])
        # closest multiple-of-head_dim dim should be within one step's worth
        self.assertLess(abs(r["rel_err"]), 0.25)

    def test_all_mixers_solve(self):
        for mixer in ("transformer", "linear_attention", "deltanet", "gated_deltanet"):
            r = solve_dim_for_params(500_000, mixer=mixer, **COMMON)
            self.assertGreater(r["dim"], 0)


if __name__ == "__main__":
    unittest.main()
