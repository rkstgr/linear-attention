"""Linear attention (Katharopoulos et al. 2020).

Drop-in replacement for `Attention` in transformer.py. The mixer body is
yours to fill in — the rest (projections, head plumbing, RoPE, SiLU,
training loop) is identical to the transformer baseline so the only
variable is the recurrence itself.

Run:
    uv run python linear_attention.py

Pass/fail: should reach >= 0.95 MQAR accuracy at N_KV=4, SEQ_LEN=32,
VOCAB=64 within ~1500 steps (the same config the softmax baseline solves).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from data import get_level
from train import inspect_example, train_and_eval
from utils import RMSNorm, SwiGLU, apply_rope, rope_freqs


CONV_SIZE = 4  # short conv kernel; matches FLA / Based recipe


def causal_dwconv(x: Array, w: Array) -> Array:
    """Depthwise causal 1D conv.

    x: (T, D), w: (K, D). Returns (T, D) with
        out[t, d] = sum_{k=0}^{K-1} w[k, d] * x[t-k, d]   (x[t<0] = 0)

    Each channel d has its own K-tap filter w[:, d]. Causal via left-pad.
    Same helper used in deltanet.py.
    """
    T = x.shape[0]
    K = w.shape[0]
    x_pad = jnp.pad(x, ((K - 1, 0), (0, 0)))
    return sum(w[k] * x_pad[K - 1 - k : K - 1 - k + T] for k in range(K))


class LinearAttention(eqx.Module):
    """Linear attention with SiLU feature map.

    Per head, the mixer must compute the recurrence

        S_t = S_{t-1} + v_t k_t^T          (state update)
        o_t = S_t q_t                       (read out)

    where S_t is a (Dh, Dh) matrix carrying associative memory across
    the sequence. q, k pass through SiLU (cheap positive feature map);
    no softmax, no normalizer.

    The body of __call__ below sets up q, k, v with the same projections,
    RoPE, and SiLU as DeltaNet / Gated DeltaNet will use, then hands off
    to a `mix(q, k, v)` you have to implement.
    """

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

    def __call__(self, x: Array, cos: Array, sin: Array) -> Array:
        # x: (T, D)
        T, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        # project -> short causal conv -> SiLU -> reshape to heads.
        # Conv adds local-context features (Zoology / Based recipe).
        # No RoPE: keeping the same recipe as deltanet.py for fair comparison.
        q = jax.nn.silu(causal_dwconv(x @ self.Wq, self.Cq)).reshape(T, H, Dh).transpose(1, 0, 2)
        k = jax.nn.silu(causal_dwconv(x @ self.Wk, self.Ck)).reshape(T, H, Dh).transpose(1, 0, 2)
        v = jax.nn.silu(causal_dwconv(x @ self.Wv, self.Cv)).reshape(T, H, Dh).transpose(1, 0, 2)

        # ------------------------------------------------------------------
        # YOUR CODE HERE
        #
        # Compute `out` of shape (H, T, Dh) implementing
        #
        #   o_t = sum_{s <= t}  v_s  *  (k_s^T q_t)
        #
        # Two natural ways:
        #   (a) parallel form  — build the (T, T) score matrix q @ k^T,
        #       apply a causal lower-triangular mask, multiply by v.
        #       Mirrors transformer.py exactly except: no scale, no softmax.
        #
        #   (b) recurrent form — jax.lax.scan over time with carry
        #       S_t = S_{t-1} + v_t k_t^T, output S_t q_t at each step.
        #       Use jax.vmap to scan each head independently.
        #
        # Both should give the same numbers. (a) is shorter and faster on
        # GPU; (b) is what the post's recurrence framing actually says.
        # Either is fine — pick one, write ~5 lines.
        # ------------------------------------------------------------------

        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        out = (q @ k.transpose(0,2,1) * mask) @ v
        # ------------------------------------------------------------------

        # merge heads: (H, T, Dh) -> (T, D)
        out = out.transpose(1, 0, 2).reshape(T, D)
        return out @ self.Wo


# ---------------------------------------------------------------------------
# Block + Transformer — identical to transformer.py, repeated here so this
# file is self-contained. The only line that differs is `self.attn = ...`.
# ---------------------------------------------------------------------------


class Block(eqx.Module):
    norm_attn: RMSNorm
    attn: LinearAttention
    norm_mlp: RMSNorm
    mlp: SwiGLU

    def __init__(self, dim: int, n_heads: int, mlp_mult: int, key):
        k_attn, k_mlp = jax.random.split(key)
        self.norm_attn = RMSNorm(dim)
        self.attn = LinearAttention(dim, n_heads, k_attn)
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
    ):
        keys = jax.random.split(key, n_layers + 2)
        s = 1.0 / jnp.sqrt(dim)
        self.tok_emb = jax.random.normal(keys[0], (vocab_size, dim)) * s
        self.blocks = [Block(dim, n_heads, mlp_mult, k) for k in keys[1:-1]]
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
# MQAR — pass `--level 0` for fast dev (vocab=256), default level1 matches
# Zoology (vocab=8192, seq=64, N_KV=4, power-law gaps). Linear attention
# should reach >99% on either: capacity d_k = 16 >> N_KV = 4.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    cfg = get_level(sys.argv)
    print(f"--- MQAR (linear attention, vocab={cfg.vocab_size}) ---")
    k_model, k_train, k_inspect = jax.random.split(jax.random.PRNGKey(1), 3)

    model = Transformer(
        vocab_size=cfg.vocab_size,
        dim=64,
        n_heads=4,
        n_layers=2,
        mlp_mult=4,
        key=k_model,
    )
    model, _ = train_and_eval(model, cfg, k_train)
    inspect_example(model, k_inspect, cfg)
