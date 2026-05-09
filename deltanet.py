"""DeltaNet (Yang et al. 2024, arXiv:2406.06484) and Gated DeltaNet
(Yang, Kautz, Hatamizadeh 2024, arXiv:2412.06464) in one file.

DeltaNet recurrence:
    S_t = S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T

Gated DeltaNet adds a scalar state-decay gate alpha_t in (0, 1):
    S_t = alpha_t * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T

beta scales the rank-1 perturbation along k_t (surgical overwrite at the
key direction). alpha multiplies the *entire* prior state uniformly —
exponential decay of stale information regardless of direction. The two
gates handle complementary failure modes: beta is for direction collisions,
alpha is for accumulated noise / stale prefix.

Run:
    uv run python deltanet.py            # plain DeltaNet
    uv run python deltanet.py --gated    # Gated DeltaNet
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

import sys

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from data import level1
from train import inspect_example, train_and_eval
from utils import RMSNorm, SwiGLU, rope_freqs

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)

CONV_SIZE = 4  # short conv kernel; standard FLA default


def causal_dwconv(x: Array, w: Array) -> Array:
    """Depthwise causal 1D conv.

    x: (T, D), w: (K, D). Returns (T, D) with
        out[t, d] = sum_{k=0}^{K-1} w[k, d] * x[t-k, d]   (x[t<0] = 0)

    "Depthwise" = no cross-channel mixing; each channel d has its own
    K-tap FIR filter w[:, d]. Total params per conv = K * D.
    "Causal" = position t depends only on positions <= t (left-pad).

    Why we need this in DeltaNet: at any given token, q/k/v are computed
    from the last K tokens, not just the current one. Lets the model see
    "I'm at a query position, the last few tokens were noise" — context
    that pure recurrence can't peek ahead to.
    """
    T = x.shape[0]
    K = w.shape[0]
    x_pad = jnp.pad(x, ((K - 1, 0), (0, 0)))
    return sum(w[k] * x_pad[K - 1 - k : K - 1 - k + T] for k in range(K))


class DeltaNet(eqx.Module):
    """DeltaNet mixer.

    Compared to LinearAttention, two extra ingredients:

    - A per-token, per-head scalar gate beta_t in (0, 1), produced by a
      sigmoid on a learned linear projection of x. Controls the step size
      of the inner gradient update. beta_t = 0 means "ignore this token";
      beta_t = 1 means "fully overwrite the k_t direction".

    - L2 normalization of k after the feature map. The rank-1 transition
      I - beta_t k_t k_t^T is well-conditioned only when ||k_t|| is bounded;
      L2-normalizing forces ||k_t|| = 1 so the eigenvalue along k_t lands
      cleanly in [1 - beta_t, 1].
    """

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
        # short causal depthwise conv weights, one filter bank per stream
        self.Cq = jax.random.normal(keys[6], (CONV_SIZE, dim)) * sc
        self.Ck = jax.random.normal(keys[7], (CONV_SIZE, dim)) * sc
        self.Cv = jax.random.normal(keys[8], (CONV_SIZE, dim)) * sc
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.gated = gated

    def __call__(self, x: Array, cos: Array, sin: Array) -> Array:
        # x: (T, D)
        T, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        # project: (T, D) -> (T, D), then short causal conv across time, then SiLU.
        # Order matches FLA reference: project -> dwconv -> SiLU -> reshape -> norm.
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

        # gates per (head, time): (T, H) -> (H, T). Computed from raw x, not post-conv.
        beta = 2.0 * jax.nn.sigmoid(x @ self.Wbeta).transpose(1, 0)
        if self.gated:
            # alpha = exp(-softplus(dt_logit) * sigmoid(x @ Walpha + balpha))
            # Bounded in (0, 1] by construction; see __init__ for init rationale.
            g = jax.nn.sigmoid(x @ self.Walpha + self.balpha)  # (T, H)
            dt = jax.nn.softplus(self.dt_logit)  # (H,)
            alpha = jnp.exp(-dt * g).transpose(1, 0)  # (H, T)
        else:
            alpha = jnp.ones_like(beta)  # alpha=1 → no decay → plain DeltaNet

        # L2 normalize k so the rank-1 perturbation I - beta * k k^T has
        # bounded spectrum. (q is unnormalized — read magnitude carries info.)
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)

        def per_head(q, k, v, beta, alpha):
            def step(S, inputs):
                q_t, k_t, v_t, beta_t, alpha_t = inputs
                # S @ (I - beta k k^T) = S - beta (S k) k^T — matvec + outer,
                # avoids the Dh×Dh @ Dh×Dh matmul (O(Dh^3) -> O(Dh^2)).
                Sk = S @ k_t
                S_new = alpha_t * (
                    S - beta_t * jnp.outer(Sk, k_t)
                ) + beta_t * jnp.outer(v_t, k_t)
                o_t = S_new @ q_t
                return S_new, o_t

            S0 = jnp.zeros((Dh, Dh))
            _, out = jax.lax.scan(step, S0, (q, k, v, beta, alpha))
            return out

        out = jax.vmap(per_head)(q, k, v, beta, alpha)  # (H, T, Dh)
        # ------------------------------------------------------------------

        # merge heads: (H, T, Dh) -> (T, D)
        out = out.transpose(1, 0, 2).reshape(T, D)
        return out @ self.Wo


# ---------------------------------------------------------------------------
# Block + Transformer — identical scaffolding, mixer is DeltaNet.
# ---------------------------------------------------------------------------


class Block(eqx.Module):
    norm_attn: RMSNorm
    attn: DeltaNet
    norm_mlp: RMSNorm
    mlp: SwiGLU

    def __init__(self, dim: int, n_heads: int, mlp_mult: int, key, gated: bool = False):
        k_attn, k_mlp = jax.random.split(key)
        self.norm_attn = RMSNorm(dim)
        self.attn = DeltaNet(dim, n_heads, k_attn, gated=gated)
        self.norm_mlp = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_mult * dim, k_mlp)

    def __call__(self, x: Array, cos: Array, sin: Array) -> Array:
        x = x + self.attn(self.norm_attn(x), cos, sin)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class Transformer(eqx.Module):
    tok_emb: Array
    blocks: list
    final_norm: RMSNorm
    lm_head: Array
    head_dim: int = eqx.field(static=True)

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_heads: int,
        n_layers: int,
        mlp_mult: int,
        key,
        gated: bool = False,
    ):
        keys = jax.random.split(key, n_layers + 2)
        s = 1.0 / jnp.sqrt(dim)
        self.tok_emb = jax.random.normal(keys[0], (vocab_size, dim)) * s
        self.blocks = [
            Block(dim, n_heads, mlp_mult, k, gated=gated) for k in keys[1:-1]
        ]
        self.final_norm = RMSNorm(dim)
        self.lm_head = jax.random.normal(keys[-1], (dim, vocab_size)) * s
        self.head_dim = dim // n_heads

    def __call__(self, tokens: Array) -> Array:
        T = tokens.shape[0]
        x = self.tok_emb[tokens]
        cos, sin = rope_freqs(self.head_dim, T)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.final_norm(x)
        return x @ self.lm_head


# ---------------------------------------------------------------------------
# MQAR (level1 — Zoology baseline: vocab=8192, seq=64, N_KV=4, power-law gaps).
# DeltaNet should reach >99% — surgical overwrite handles the 4 keys cleanly.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    GATED = "--gated" in sys.argv
    name = "Gated DeltaNet" if GATED else "DeltaNet"
    cfg = level1
    print(f"--- MQAR ({name}, vocab={cfg.vocab_size}) ---")
    k_model, k_train, k_inspect = jax.random.split(jax.random.PRNGKey(1), 3)

    model = Transformer(
        vocab_size=cfg.vocab_size,
        dim=64,
        n_heads=4,
        n_layers=2,
        mlp_mult=4,
        key=k_model,
        gated=GATED,
    )
    model, _ = train_and_eval(model, cfg, k_train)
    inspect_example(model, k_inspect, cfg)
