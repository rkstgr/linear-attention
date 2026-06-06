"""Linear attention (Katharopoulos et al. 2020).

Run MQAR smoke experiments via `uv run python -m experiments.run_model linear_attention`.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

CONV_SIZE = 4


def causal_dwconv(x: Array, w: Array) -> Array:
    """Depthwise causal 1D conv."""

    T = x.shape[0]
    K = w.shape[0]
    x_pad = jnp.pad(x, ((K - 1, 0), (0, 0)))
    return sum(w[k] * x_pad[K - 1 - k : K - 1 - k + T] for k in range(K))


class LinearAttention(eqx.Module):
    """Linear attention with SiLU feature map."""

    Wq: Array
    Wk: Array
    Wv: Array
    Wo: Array
    Cq: Array
    Ck: Array
    Cv: Array
    n_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, dim: int, n_heads: int, key):
        assert dim % n_heads == 0
        keys = jax.random.split(key, 7)
        s = 1.0 / jnp.sqrt(dim)
        sc = 1.0 / jnp.sqrt(CONV_SIZE)
        self.Wq = jax.random.normal(keys[0], (dim, dim)) * s
        self.Wk = jax.random.normal(keys[1], (dim, dim)) * s
        self.Wv = jax.random.normal(keys[2], (dim, dim)) * s
        self.Wo = jax.random.normal(keys[3], (dim, dim)) * s
        self.Cq = jax.random.normal(keys[4], (CONV_SIZE, dim)) * sc
        self.Ck = jax.random.normal(keys[5], (CONV_SIZE, dim)) * sc
        self.Cv = jax.random.normal(keys[6], (CONV_SIZE, dim)) * sc
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

    def __call__(self, x: Array) -> Array:
        T, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        q = (
            jax.nn.silu(causal_dwconv(x @ self.Wq, self.Cq))
            .reshape(T, H, Dh)
            .transpose(1, 0, 2)
        )
        k = (
            jax.nn.silu(causal_dwconv(x @ self.Wk, self.Ck))
            .reshape(T, H, Dh)
            .transpose(1, 0, 2)
        )
        v = (
            jax.nn.silu(causal_dwconv(x @ self.Wv, self.Cv))
            .reshape(T, H, Dh)
            .transpose(1, 0, 2)
        )

        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        out = (q @ k.transpose(0, 2, 1) * mask) @ v

        out = out.transpose(1, 0, 2).reshape(T, D)
        return out @ self.Wo
