"""Feed-forward modules shared by model backbones."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


class SwiGLU(eqx.Module):
    """Gated feed-forward: SiLU(x W_g) * (x W_u) * W_d."""

    Wg: Array
    Wu: Array
    Wd: Array

    def __init__(self, dim: int, hidden: int, key):
        k1, k2, k3 = jax.random.split(key, 3)
        s_in = 1.0 / jnp.sqrt(dim)
        s_out = 1.0 / jnp.sqrt(hidden)
        self.Wg = jax.random.normal(k1, (dim, hidden)) * s_in
        self.Wu = jax.random.normal(k2, (dim, hidden)) * s_in
        self.Wd = jax.random.normal(k3, (hidden, dim)) * s_out

    def __call__(self, x: Array) -> Array:
        return (jax.nn.silu(x @ self.Wg) * (x @ self.Wu)) @ self.Wd
