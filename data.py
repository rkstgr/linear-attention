"""Toy datasets for sub-quadratic sequence models.

MQAR (Multi-Query Associative Recall) — Arora et al., "Zoology" (2023).
The benchmark that empirically distinguishes softmax from linear attention.

Format (matches Zoology's `multiquery_ar.py`):

    [ k1 v1 k2 v2 ... kN vN | <query region of length L - 2N> ]
                              ^ each of the N keys appears once,
                                placed at a random even slot;
                                all other slots are noise.

At each query position P the model must predict the value associated with that
key. Crucially, the *input* token at position P+1 is noise, not the value —
the model never sees the value in its input, it must look it up from the kv
section via content-based attention.

Vocab convention:
    keys   ∈ [1, V/2)
    values ∈ [V/2, V)
    token 0 is reserved (never appears as key or value; may appear as noise)
"""

import jax
import jax.numpy as jnp
from jax import Array


def mqar_example(key, num_kv_pairs: int, input_seq_len: int, vocab_size: int):
    """Generate one MQAR sequence.

    Returns:
        tokens:  (T,) int32    input sequence, T = input_seq_len
        targets: (T,) int32    desired model output at each position
        mask:    (T,) float32  1.0 at query positions (the only positions scored)
    """
    assert input_seq_len % 2 == 0
    assert vocab_size > input_seq_len, "vocab_size > input_seq_len per Zoology"
    context_size = 2 * num_kv_pairs
    query_region = input_seq_len - context_size
    assert query_region % 2 == 0 and query_region // 2 >= num_kv_pairs

    V_half = vocab_size // 2
    k_keys, k_vals, k_gaps, k_noise = jax.random.split(key, 4)

    # unique keys in [1, V/2); unique values in [V/2, V)
    keys = jax.random.permutation(k_keys, V_half - 1)[:num_kv_pairs] + 1
    values = jax.random.permutation(k_vals, V_half)[:num_kv_pairs] + V_half

    # kv section: interleaved [k1, v1, k2, v2, ...]
    kv = jnp.zeros(context_size, dtype=jnp.int32)
    kv = kv.at[0::2].set(keys)
    kv = kv.at[1::2].set(values)

    # query region: random noise, then overwrite N even slots with the keys
    space = query_region // 2
    gaps = jax.random.permutation(k_gaps, space)[:num_kv_pairs]
    q_pos_in_region = 2 * gaps                                   # (N,) even positions
    noise = jax.random.randint(k_noise, (query_region,), 0, vocab_size)
    qregion = noise.at[q_pos_in_region].set(keys)

    tokens = jnp.concatenate([kv, qregion])                       # (T,)

    # target at logit-index P: the value associated with the key at tokens[P].
    # logits[P] predicts position P+1, but here we override: at a query
    # position P, the target *at that logit* is the value (ignoring the actual
    # input at P+1, which is noise).
    q_pos_full = context_size + q_pos_in_region                   # (N,) in [2N, T)
    targets = jnp.zeros(input_seq_len, dtype=jnp.int32)
    mask = jnp.zeros(input_seq_len, dtype=jnp.float32)
    targets = targets.at[q_pos_full].set(values)
    mask = mask.at[q_pos_full].set(1.0)

    return tokens, targets, mask


def mqar_batch(key, batch_size: int, num_kv_pairs: int, input_seq_len: int, vocab_size: int):
    """Vectorized mqar_example. Returns tokens, targets, mask all of shape (B, T)."""
    keys = jax.random.split(key, batch_size)
    return jax.vmap(mqar_example, in_axes=(0, None, None, None))(
        keys, num_kv_pairs, input_seq_len, vocab_size
    )
