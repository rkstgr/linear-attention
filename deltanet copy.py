"""DeltaNet (Yang et al. 2024, arXiv:2406.06484).

Drop-in replacement for `Attention`. The mixer is the delta rule:

    S_t = S_{t-1} + beta_t * (v_t - S_{t-1} k_t) k_t^T          (state update)
    o_t = S_t q_t                                                (read out)

This is one step of online gradient descent on  L_t = (1/2) ||v_t - S k_t||^2
with step size beta_t. Equivalently:

    S_t = S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T

i.e. a rank-1 perturbation that *erases* prior writes overlapping k_t and
inserts the new (k_t, v_t) pair. Surgical overwrite — same state size as
linear attention, but the bits get reused efficiently.

Run:
    uv run python deltanet.py

Pass/fail: should reach >= 0.95 MQAR accuracy at N_KV=4, SEQ_LEN=32,
VOCAB=64 within ~1500 steps. Should comfortably beat linear attention.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from data import mqar_batch, mqar_example
from utils import RMSNorm, SwiGLU, apply_rope, rope_freqs


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
    Wo: Array
    n_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, dim: int, n_heads: int, key):
        assert dim % n_heads == 0
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        s = 1.0 / jnp.sqrt(dim)
        self.Wq = jax.random.normal(k1, (dim, dim)) * s
        self.Wk = jax.random.normal(k2, (dim, dim)) * s
        self.Wv = jax.random.normal(k3, (dim, dim)) * s
        # one scalar beta per head per token: project (T, D) -> (T, H)
        self.Wbeta = jax.random.normal(k4, (dim, n_heads)) * s
        self.Wo = jax.random.normal(k5, (dim, dim)) * s
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

    def __call__(self, x: Array, cos: Array, sin: Array) -> Array:
        # x: (T, D)
        T, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        # project and split heads: (T, D) -> (H, T, Dh)
        q = (x @ self.Wq).reshape(T, H, Dh).transpose(1, 0, 2)
        k = (x @ self.Wk).reshape(T, H, Dh).transpose(1, 0, 2)
        v = (x @ self.Wv).reshape(T, H, Dh).transpose(1, 0, 2)

        # beta per (head, time): (T, H) -> (H, T)
        beta = jax.nn.sigmoid(x @ self.Wbeta).transpose(1, 0)

        # RoPE on q, k
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # SiLU feature map on q, k
        q = jax.nn.silu(q)
        k = jax.nn.silu(k)

        # L2 normalize k so the rank-1 perturbation I - beta * k k^T has
        # bounded spectrum. (q is unnormalized — read magnitude carries info.)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)

        # ------------------------------------------------------------------
        # YOUR CODE HERE
        #
        # Compute `out` of shape (H, T, Dh) by running the delta-rule
        # recurrence for each head:
        #
        #   S_t = S_{t-1} + beta_t * (v_t - S_{t-1} k_t) k_t^T
        #   o_t = S_t q_t
        #
        # No clean parallel form exists without WY (out of scope for §5).
        # Use jax.lax.scan over time, jax.vmap over heads. Skeleton:
        #
        #   def per_head(q, k, v, beta):       # shapes (T, Dh) / (T, Dh) / (T, Dh) / (T,)
        #       def step(S, inputs):
        #           q_t, k_t, v_t, beta_t = inputs
        #           # ... compute S_new and o_t from the equations above ...
        #           return S_new, o_t
        #       S0 = jnp.zeros((Dh, Dh))
        #       _, out = jax.lax.scan(step, S0, (q, k, v, beta))
        #       return out                     # (T, Dh)
        #
        #   out = jax.vmap(per_head)(q, k, v, beta)   # (H, T, Dh)
        #
        # Two design choices to be deliberate about:
        #   (i)  S has shape (Dh, Dh) — d_v on rows, d_k on columns.
        #        S @ k is shape (Dh,). Outer product (v - S k) @ k^T uses
        #        jnp.outer.
        #   (ii) Output uses the *post-update* state (S_t, not S_{t-1}).
        #        This is the same convention as transformer.py's softmax
        #        attention (causal mask includes the diagonal: position t
        #        attends to itself).
        # ------------------------------------------------------------------
        #
        def per_head(q,k,v,beta):
            def step(S, inputs):
                q_t, k_t, v_t, beta_t = inputs
                o_t = S @ q_t
                S_new = S @ (jnp.eyes((Dh, Dh)) - beta*jnp.outer(k_t,k_t)) + beta*jnp.outer(v_t,k_t)
                return S_new, o_t
            S0 = jnp.zeros((Dh, Dh))
            _, out = jax.lax.scan(step, S0, (q,k,v,beta))
            return out


        out = jax.vmap(per_head)(q, k, v, beta)  # (H, T, Dh)
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

    def __init__(self, dim: int, n_heads: int, mlp_mult: int, key):
        k_attn, k_mlp = jax.random.split(key)
        self.norm_attn = RMSNorm(dim)
        self.attn = DeltaNet(dim, n_heads, k_attn)
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
# MQAR training — same config softmax solved.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- MQAR (DeltaNet) ---")
    VOCAB = 64
    N_KV = 4
    SEQ_LEN = 32
    BATCH = 32

    model = Transformer(
        vocab_size=VOCAB,
        dim=64,
        n_heads=4,
        n_layers=2,
        mlp_mult=4,
        key=jax.random.PRNGKey(1),
    )

    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, 1e-3, transition_steps=100),
            optax.constant_schedule(1e-3),
        ],
        boundaries=[100],
    )
    opt = optax.adamw(schedule)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    def mqar_loss_and_acc(model, tokens, targets, mask):
        logits = jax.vmap(model)(tokens)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        loss = (losses * mask).sum() / mask.sum()
        preds = jnp.argmax(logits, axis=-1)
        acc = ((preds == targets).astype(jnp.float32) * mask).sum() / mask.sum()
        return loss, acc

    @eqx.filter_jit
    def mqar_step(model, opt_state, tokens, targets, mask):
        (loss, acc), grads = eqx.filter_value_and_grad(
            mqar_loss_and_acc, has_aux=True
        )(model, tokens, targets, mask)
        updates, opt_state = opt.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, acc

    k_data = jax.random.PRNGKey(42)
    for i in range(1500):
        k_data, sub = jax.random.split(k_data)
        tokens, targets, mask = mqar_batch(sub, BATCH, N_KV, SEQ_LEN, VOCAB)
        model, opt_state, loss, acc = mqar_step(
            model, opt_state, tokens, targets, mask
        )
        if i % 50 == 0:
            print(f"step {i:4d}  loss {float(loss):.4f}  acc {float(acc):.3f}")

    tokens, targets, mask = mqar_example(jax.random.PRNGKey(999), N_KV, SEQ_LEN, VOCAB)
    preds = jnp.argmax(model(tokens), axis=-1)
    q_idx = jnp.nonzero(mask, size=N_KV)[0]
    print(f"\n  tokens   : {tokens.tolist()}")
    print(f"  kv pairs : {tokens[: 2 * N_KV].tolist()}")
    print(f"  q pos    : {q_idx.tolist()}")
    print(f"  queries  : {tokens[q_idx].tolist()}")
    print(f"  expected : {targets[q_idx].tolist()}")
    print(f"  predicted: {preds[q_idx].tolist()}")
