"""Modern transformer utilities: RMSNorm and RoPE.

Shared by every model in this directory.
"""

import equinox as eqx
import jax.numpy as jnp
from jax import Array


class RMSNorm(eqx.Module):
    weight: Array
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-6):
        self.weight = jnp.ones(dim)
        self.eps = eps

    def __call__(self, x: Array) -> Array:
        # x: (..., D)
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.weight


def rope_freqs(head_dim: int, seq_len: int, base: float = 10000.0):
    """Precompute (cos, sin) tables for rotary position embedding.

    Returns cos, sin of shape (T, Dh/2).
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    # freqs: (Dh/2,)  θ_i = base^(-2i/Dh)
    freqs = 1.0 / (base ** (jnp.arange(0, head_dim, 2) / head_dim))
    # angles: (T, Dh/2)
    angles = jnp.outer(jnp.arange(seq_len), freqs)
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x: Array, cos: Array, sin: Array) -> Array:
    """Rotate pairs of features in x using RoPE.

    x:   (..., T, Dh)
    cos: (T, Dh/2)
    sin: (T, Dh/2)

    Uses the interleaved convention: pairs are (x[0], x[1]), (x[2], x[3]), ...
    """
    x1 = x[..., ::2]   # even indices
    x2 = x[..., 1::2]  # odd indices
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    # interleave back: stack along a new trailing axis, then flatten last two dims
    return jnp.stack([out1, out2], axis=-1).reshape(x.shape)
