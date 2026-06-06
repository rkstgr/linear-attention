"""DeltaNet and Gated DeltaNet mixers.

Run MQAR smoke experiments via `uv run python -m experiments.run_model deltanet`
or `uv run python -m experiments.run_model gated_deltanet`.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)

CONV_SIZE = 4


def causal_dwconv(x: Array, w: Array) -> Array:
    """Depthwise causal 1D conv."""

    T = x.shape[0]
    K = w.shape[0]
    x_pad = jnp.pad(x, ((K - 1, 0), (0, 0)))
    return sum(w[k] * x_pad[K - 1 - k : K - 1 - k + T] for k in range(K))


class DeltaNet(eqx.Module):
    """DeltaNet mixer."""

    Wq: Array
    Wk: Array
    Wv: Array
    Wbeta: Array
    Walpha: Array
    balpha: Array
    dt_logit: Array
    Wo: Array
    Cq: Array
    Ck: Array
    Cv: Array
    n_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    gated: bool = eqx.field(static=True)

    def __init__(self, dim: int, n_heads: int, key, gated: bool = False):
        assert dim % n_heads == 0
        keys = jax.random.split(key, 9)
        s = 1.0 / jnp.sqrt(dim)
        sc = 1.0 / jnp.sqrt(CONV_SIZE)
        self.Wq = jax.random.normal(keys[0], (dim, dim)) * s
        self.Wk = jax.random.normal(keys[1], (dim, dim)) * s
        self.Wv = jax.random.normal(keys[2], (dim, dim)) * s
        self.Wbeta = jax.random.normal(keys[3], (dim, n_heads)) * s
        self.Walpha = jax.random.normal(keys[4], (dim, n_heads)) * s
        self.balpha = jnp.zeros((n_heads,))
        self.dt_logit = jnp.full((n_heads,), -10.0)
        self.Wo = jax.random.normal(keys[5], (dim, dim)) * s
        self.Cq = jax.random.normal(keys[6], (CONV_SIZE, dim)) * sc
        self.Ck = jax.random.normal(keys[7], (CONV_SIZE, dim)) * sc
        self.Cv = jax.random.normal(keys[8], (CONV_SIZE, dim)) * sc
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.gated = gated

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

        beta = 2.0 * jax.nn.sigmoid(x @ self.Wbeta).transpose(1, 0)
        if self.gated:
            g = jax.nn.sigmoid(x @ self.Walpha + self.balpha)
            dt = jax.nn.softplus(self.dt_logit)
            alpha = jnp.exp(-dt * g).transpose(1, 0)
        else:
            alpha = jnp.ones_like(beta)

        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)

        def per_head(q, k, v, beta, alpha):
            def step(S, inputs):
                q_t, k_t, v_t, beta_t, alpha_t = inputs
                Sk = S @ k_t
                S_new = alpha_t * (
                    S - beta_t * jnp.outer(Sk, k_t)
                ) + beta_t * jnp.outer(v_t, k_t)
                o_t = S_new @ q_t
                return S_new, o_t

            S0 = jnp.zeros((Dh, Dh))
            _, out = jax.lax.scan(step, S0, (q, k, v, beta, alpha))
            return out

        out = jax.vmap(per_head)(q, k, v, beta, alpha)
        out = out.transpose(1, 0, 2).reshape(T, D)
        return out @ self.Wo
