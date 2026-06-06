"""Tiny character-model Transformer demo.

Run:
    uv run python -m experiments.char_transformer
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from models.registry import build_lm_model


def main():
    text = "the quick brown fox jumps over the lazy dog. " * 64
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = jnp.array([stoi[c] for c in text], dtype=jnp.int32)

    seq_len = 64
    key = jax.random.PRNGKey(0)
    k_model, k_data = jax.random.split(key)

    model = build_lm_model(
        "transformer",
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


if __name__ == "__main__":
    main()
