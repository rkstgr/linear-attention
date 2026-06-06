"""Parity gate for proposal 0001."""

import hashlib
import json
import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

import equinox as eqx
import jax
import jax.numpy as jnp

from models.registry import build_lm_model

FIXTURE = {
    "vocab_size": 17,
    "dim": 8,
    "n_heads": 2,
    "n_layers": 2,
    "mlp_mult": 2,
}
TOKENS = jnp.array([0, 3, 5, 7, 11, 13, 16], dtype=jnp.int32)
KEY = jax.random.PRNGKey(123)

EXPECTED = {
    "transformer": {
        "digest": "a82479b3b7f62ab3ed11eebf29b5ba4bbc9e3e34c26376d73e057331b5918b43",
        "logits_dtype": "float32",
        "logits_shape": [7, 17],
        "n_params": 21,
    },
    "linear_attention": {
        "digest": "9cba68bbe0dd5d4daa12390878d64decb77216418c840b55a0235aa4628a8bf6",
        "logits_dtype": "float32",
        "logits_shape": [7, 17],
        "n_params": 27,
    },
    "deltanet": {
        "digest": "ade312c51edc91ce49b75f4b9a010a07f7d504fb134f63a922c22104ac31e08a",
        "logits_dtype": "float32",
        "logits_shape": [7, 17],
        "n_params": 35,
    },
    "gated_deltanet": {
        "digest": "66a70e7f3b302d4cd7c52864bc6e02caa3e234b097a3188b06f8c11e2e5cd361",
        "logits_dtype": "float32",
        "logits_shape": [7, 17],
        "n_params": 35,
    },
    "titans": {
        "digest": "6fb0f16fcdcd613b3821573d46a4d160bb1a32acb7065fb7ba9e0611fbef6dfa",
        "logits_dtype": "float32",
        "logits_shape": [7, 17],
        "n_params": 43,
    },
}

VARIANT_KWARGS = {
    "titans": {"memory_mult": 2, "max_inner_lr": 0.0125},
}


def _update_array(hasher, kind: str, x):
    meta = {"kind": kind, "shape": list(x.shape), "dtype": str(x.dtype)}
    hasher.update(json.dumps(meta, sort_keys=True).encode())
    hasher.update(b"\0")
    hasher.update(x.tobytes())
    hasher.update(b"\0")


def _semantic_digest(params, logits) -> str:
    hasher = hashlib.sha256()
    for leaf in params:
        _update_array(hasher, "param", leaf)
    _update_array(hasher, "logits", logits)
    return hasher.hexdigest()


class ParityTest(unittest.TestCase):
    def test_trainable_leaf_and_logits_parity(self):
        for mixer, expected in EXPECTED.items():
            with self.subTest(mixer=mixer):
                model = build_lm_model(
                    mixer,
                    **FIXTURE,
                    key=KEY,
                    **VARIANT_KWARGS.get(mixer, {}),
                )
                params = [
                    x for x in jax.tree.leaves(model) if eqx.is_inexact_array(x)
                ]
                logits = model(TOKENS)

                self.assertEqual(len(params), expected["n_params"])
                self.assertEqual(list(logits.shape), expected["logits_shape"])
                self.assertEqual(str(logits.dtype), expected["logits_dtype"])
                self.assertEqual(_semantic_digest(params, logits), expected["digest"])


if __name__ == "__main__":
    unittest.main()
