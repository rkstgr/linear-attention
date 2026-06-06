"""Softmax attention mixer."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from utils import apply_rope, rope_freqs


class Attention(eqx.Module):
    """Standard multi-head causal softmax attention."""

    Wq: Array
    Wk: Array
    Wv: Array
    Wo: Array
    n_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, dim: int, n_heads: int, key):
        assert dim % n_heads == 0
        k1, k2, k3, k4 = jax.random.split(key, 4)
        s = 1.0 / jnp.sqrt(dim)
        self.Wq = jax.random.normal(k1, (dim, dim)) * s
        self.Wk = jax.random.normal(k2, (dim, dim)) * s
        self.Wv = jax.random.normal(k3, (dim, dim)) * s
        self.Wo = jax.random.normal(k4, (dim, dim)) * s
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

    def __call__(self, x: Array) -> Array:
        T, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        q = (x @ self.Wq).reshape(T, H, Dh).transpose(1, 0, 2)
        k = (x @ self.Wk).reshape(T, H, Dh).transpose(1, 0, 2)
        v = (x @ self.Wv).reshape(T, H, Dh).transpose(1, 0, 2)

        cos, sin = rope_freqs(Dh, T)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        scores = q @ k.transpose(0, 2, 1) / jnp.sqrt(Dh)
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        scores = jnp.where(mask, scores, -jnp.inf)
        probs = jax.nn.softmax(scores, axis=-1)
        out = probs @ v

        out = out.transpose(1, 0, 2).reshape(T, D)
        return out @ self.Wo
