"""Toy datasets for sub-quadratic sequence models.

MQAR (Multi-Query Associative Recall) — Arora et al., "Zoology" (2023).
The benchmark that empirically distinguishes softmax from linear attention.

Format (matches Zoology's `multiquery_ar.py`):

    [ k1 v1 k2 v2 ... kN vN | <query region of length L - 2N> ]
                              ^ each of the N keys appears once,
                                placed at a power-law-distributed even slot
                                (gaps biased toward small values, mimicking
                                bigram-repetition statistics in real text);
                                all other slots are uniform noise drawn from
                                the full vocab — distractors *can* collide
                                with active keys/values, on purpose.

At each query position P the model must predict the value associated with that
key. The *input* token at position P+1 is noise, not the value — the model
never sees the value during the query phase, it must look it up from the kv
section via content-based attention.

Vocab convention:
    keys   ∈ [1, V/2)
    values ∈ [V/2, V)
    token 0 is reserved (never appears as key or value; may appear as noise)
"""

import os as _os

# Default to CPU. DeltaNet's per-token `lax.scan` is dominated by per-step
# kernel launch overhead on MPS (and small-matmul overhead on CUDA), so CPU
# is fastest for the small models in this repo. Set `JAX_PLATFORMS=cuda,cpu`
# explicitly when you actually want GPU (e.g. for the large softmax baseline).
# Must run before any jax import anywhere; data.py is the first import in
# every entrypoint, so the env var lands in time.
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Config:
    """A complete MQAR experiment configuration: data + training."""
    vocab_size: int
    input_seq_len: int
    num_kv_pairs: int
    power_a: float
    n_train: int
    n_test: int
    batch_size: int
    eval_batch_size: int
    max_epochs: int
    learning_rate: float
    target_acc: float
    patience_epochs: int


# toy — first proof test for new recurrent mechanisms.
#
# With one association there is no key disambiguation yet. Passing this says
# the model can carry a prefix value to a query position at all; failing this
# points at write/read plumbing or inner-update stability.
toy = Config(
    vocab_size=64,
    input_seq_len=16,
    num_kv_pairs=1,
    power_a=0.01,
    n_train=5_000,
    n_test=256,
    batch_size=64,
    eval_batch_size=64,
    max_epochs=16,
    learning_rate=1e-3,
    target_acc=0.99,
    patience_epochs=5,
)


# easy — debug curriculum for new recurrent mechanisms.
#
# This is intentionally easier than Zoology level1: small vocab, short context,
# and only two associations. It still requires content-based lookup, but the
# model gets a low-entropy first rung before level1's 4096-way value choice.
easy = Config(
    vocab_size=128,
    input_seq_len=32,
    num_kv_pairs=2,
    power_a=0.01,
    n_train=20_000,
    n_test=512,
    batch_size=64,
    eval_batch_size=64,
    max_epochs=16,
    learning_rate=1e-3,
    target_acc=0.99,
    patience_epochs=5,
)


# level1 — Zoology's easiest training config from `original_mqar_configs.py`
# (vocab=8192, seq=64, N_KV=4, power-law gaps). Numbers from this run are
# directly comparable to published Zoology curves.
#
# Training params follow Zoology defaults except batch_size, which we shrink
# from 256 to 64 so a CPU run fits in memory; with early stopping the easy
# task usually solves in 1–3 epochs anyway.
level1 = Config(
    vocab_size=8192,
    input_seq_len=64,
    num_kv_pairs=4,
    power_a=0.01,
    n_train=100_000,
    n_test=1_000,
    batch_size=64,
    eval_batch_size=64,
    max_epochs=32,
    learning_rate=1e-3,
    target_acc=0.99,
    patience_epochs=5,
)


CONFIGS = {
    "toy": toy,
    "easy": easy,
    "level1": level1,
}


def mqar_example(
    key,
    num_kv_pairs: int,
    input_seq_len: int,
    vocab_size: int,
    power_a: float = 0.01,
):
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

    # power-law gap sampling: p[i] ∝ (i+1)^(power_a - 1) over i ∈ [0, space).
    # With power_a = 0.01 this concentrates probability at small gaps,
    # matching the bigram-repetition distribution Zoology was designed to mimic.
    space = query_region // 2
    weights = power_a * jnp.arange(1, space + 1, dtype=jnp.float32) ** (power_a - 1)
    weights = weights / weights.sum()
    gaps = jax.random.choice(k_gaps, space, shape=(num_kv_pairs,), replace=False, p=weights)
    q_pos_in_region = 2 * gaps

    # query region: noise drawn from the full vocab (Zoology default —
    # distractors can equal active keys/values, the model must disambiguate
    # role from position). N keys placed at the sampled even slots.
    noise = jax.random.randint(k_noise, (query_region,), 0, vocab_size)
    qregion = noise.at[q_pos_in_region].set(keys)

    tokens = jnp.concatenate([kv, qregion])

    # target at logit-index P: the value associated with the key at tokens[P].
    # logits[P] predicts position P+1, but here we override: at a query
    # position P, the target *at that logit* is the value (ignoring the actual
    # input at P+1, which is noise).
    q_pos_full = context_size + q_pos_in_region
    targets = jnp.zeros(input_seq_len, dtype=jnp.int32)
    mask = jnp.zeros(input_seq_len, dtype=jnp.float32)
    targets = targets.at[q_pos_full].set(values)
    mask = mask.at[q_pos_full].set(1.0)

    return tokens, targets, mask


def mqar_batch(
    key,
    batch_size: int,
    num_kv_pairs: int,
    input_seq_len: int,
    vocab_size: int,
    power_a: float = 0.01,
):
    """Vectorized mqar_example. Returns tokens, targets, mask all of shape (B, T)."""
    keys = jax.random.split(key, batch_size)
    return jax.vmap(mqar_example, in_axes=(0, None, None, None, None))(
        keys, num_kv_pairs, input_seq_len, vocab_size, power_a,
    )


def make_split(key, n: int, cfg: Config):
    """Pre-generate n MQAR examples per cfg. Returns arrays of shape (n, T).

    Zoology pre-builds and caches train/test splits so every epoch sees the
    same data; this matches their protocol.
    """
    keys = jax.random.split(key, n)
    return jax.vmap(mqar_example, in_axes=(0, None, None, None, None))(
        keys, cfg.num_kv_pairs, cfg.input_seq_len, cfg.vocab_size, cfg.power_a,
    )
