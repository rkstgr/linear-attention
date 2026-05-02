"""Decoder-only transformer with modern defaults (RMSNorm, RoPE, SwiGLU).

This is the softmax baseline. Every later model in this directory is a
drop-in replacement for `Attention` that changes how the sequence mixes
tokens. Normalization, MLP, and residual structure stay the same.

Run:
    uv run python transformer.py
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from data import mqar_batch, mqar_example
from utils import RMSNorm, SwiGLU, apply_rope, rope_freqs


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

    def __call__(self, x: Array, cos: Array, sin: Array) -> Array:
        # x: (T, D)
        T, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        # project and split heads: (T, D) -> (H, T, Dh)
        q = (x @ self.Wq).reshape(T, H, Dh).transpose(1, 0, 2)
        k = (x @ self.Wk).reshape(T, H, Dh).transpose(1, 0, 2)
        v = (x @ self.Wv).reshape(T, H, Dh).transpose(1, 0, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # scaled dot-product attention per head
        scores = q @ k.transpose(0, 2, 1) / jnp.sqrt(Dh)       # (H, T, T)
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        scores = jnp.where(mask, scores, -jnp.inf)
        probs = jax.nn.softmax(scores, axis=-1)                # (H, T, T)
        out = probs @ v                                        # (H, T, Dh)

        # merge heads: (H, T, Dh) -> (T, D)
        out = out.transpose(1, 0, 2).reshape(T, D)
        return out @ self.Wo


class Block(eqx.Module):
    """Pre-norm residual block: attention + MLP."""
    norm_attn: RMSNorm
    attn: Attention
    norm_mlp: RMSNorm
    mlp: SwiGLU

    def __init__(self, dim: int, n_heads: int, mlp_mult: int, key):
        k_attn, k_mlp = jax.random.split(key)
        self.norm_attn = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, k_attn)
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
        # tokens: (T,) int -> logits: (T, V)
        T = tokens.shape[0]
        x = self.tok_emb[tokens]                       # (T, D)
        cos, sin = rope_freqs(self.head_dim, T)        # (T, Dh/2) each
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.final_norm(x)
        return x @ self.lm_head                        # (T, V)


# ---------------------------------------------------------------------------
# Toy training: character LM on a repeated sentence. Should overfit fast.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    text = "the quick brown fox jumps over the lazy dog. " * 64
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = jnp.array([stoi[c] for c in text], dtype=jnp.int32)

    seq_len = 64
    key = jax.random.PRNGKey(0)
    k_model, k_data = jax.random.split(key)

    model = Transformer(
        vocab_size=vocab_size,
        dim=64,
        n_heads=4,
        n_layers=2,
        mlp_mult=4,
        key=k_model,
    )

    opt = optax.adamw(5e-4)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    def loss_fn(model, x, y):
        logits = model(x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    @eqx.filter_jit
    def step(model, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = opt.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    for i in range(500):
        k_data, sub = jax.random.split(k_data)
        start = jax.random.randint(sub, (), 0, len(data) - seq_len - 1)
        x = jax.lax.dynamic_slice(data, (start,), (seq_len,))
        y = jax.lax.dynamic_slice(data, (start + 1,), (seq_len,))
        model, opt_state, loss = step(model, opt_state, x, y)
        if i % 50 == 0:
            print(f"step {i:4d}  loss {float(loss):.4f}")

    @eqx.filter_jit
    def forward(model, tokens):
        return model(tokens)

    prompt = "the quick "
    gen_len = 60
    total_len = len(prompt) + gen_len
    prompt_tok = jnp.array([stoi[c] for c in prompt], dtype=jnp.int32)
    tokens = jnp.zeros((total_len,), dtype=jnp.int32).at[: len(prompt)].set(prompt_tok)
    for i in range(gen_len):
        pos = len(prompt) + i
        logits = forward(model, tokens)[pos - 1]
        tokens = tokens.at[pos].set(jnp.argmax(logits))
    print("\nsample:", "".join(chars[int(t)] for t in tokens))

    # -----------------------------------------------------------------------
    # MQAR — Zoology formulation (Arora et al. 2023).
    # Softmax attention should hit >95% accuracy in a few hundred steps.
    # -----------------------------------------------------------------------
    print("\n--- MQAR ---")
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
        # all inputs: (B, T)
        logits = jax.vmap(model)(tokens)                          # (B, T, V)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        loss = (losses * mask).sum() / mask.sum()
        preds = jnp.argmax(logits, axis=-1)
        acc = ((preds == targets).astype(jnp.float32) * mask).sum() / mask.sum()
        return loss, acc

    @eqx.filter_jit
    def mqar_step(model, opt_state, tokens, targets, mask):
        (loss, acc), grads = eqx.filter_value_and_grad(mqar_loss_and_acc, has_aux=True)(
            model, tokens, targets, mask
        )
        updates, opt_state = opt.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, acc

    k_data = jax.random.PRNGKey(42)
    for i in range(1400):
        k_data, sub = jax.random.split(k_data)
        tokens, targets, mask = mqar_batch(sub, BATCH, N_KV, SEQ_LEN, VOCAB)
        model, opt_state, loss, acc = mqar_step(model, opt_state, tokens, targets, mask)
        if i % 50 == 0:
            print(f"step {i:4d}  loss {float(loss):.4f}  acc {float(acc):.3f}")

    # inspect one example: show kv pairs, query positions, predicted vs true values
    tokens, targets, mask = mqar_example(jax.random.PRNGKey(999), N_KV, SEQ_LEN, VOCAB)
    preds = jnp.argmax(model(tokens), axis=-1)
    q_idx = jnp.nonzero(mask, size=N_KV)[0]  # (N_KV,) query positions
    print(f"\n  tokens   : {tokens.tolist()}")
    print(f"  kv pairs : {tokens[: 2 * N_KV].tolist()}")
    print(f"  q pos    : {q_idx.tolist()}")
    print(f"  queries  : {tokens[q_idx].tolist()}")
    print(f"  expected : {targets[q_idx].tolist()}")
    print(f"  predicted: {preds[q_idx].tolist()}")
