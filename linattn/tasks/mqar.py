"""MQAR (Multi-Query Associative Recall) task — Arora et al., "Zoology" (2023).

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
# Must run before any jax import.
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from linattn.tasks.base import Split, register_task


@dataclass(frozen=True)
class MQARConfig:
    """MQAR data spec. `name` discriminates the task in the registry."""

    vocab_size: int
    input_seq_len: int
    num_kv_pairs: int
    power_a: float
    n_train: int
    n_test: int
    n_val: int = 0
    name: str = "mqar"


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


def make_split(key, n: int, cfg: MQARConfig) -> Split:
    """Pre-generate n MQAR examples per cfg. Returns arrays of shape (n, T).

    Zoology pre-builds and caches train/test splits so every epoch sees the
    same data; this matches their protocol.
    """
    keys = jax.random.split(key, n)
    return jax.vmap(mqar_example, in_axes=(0, None, None, None, None))(
        keys, cfg.num_kv_pairs, cfg.input_seq_len, cfg.vocab_size, cfg.power_a,
    )


def _draw_unique_pool(key, total: int, cfg: MQARConfig, max_rounds: int = 16) -> Split:
    """Draw `total` examples and keep only distinct token sequences.

    MQAR's space is enormous so a single draw is essentially already unique; the
    de-duplication loop just guarantees it. Top-up rounds use fresh sub-keys.
    """
    rows = []
    seen: set[bytes] = set()
    rk = key
    for _ in range(max_rounds):
        need = total - len(rows)
        if need <= 0:
            break
        rk, sub = jax.random.split(rk)
        tk, tg, mk = make_split(sub, need, cfg)
        tk, tg, mk = np.asarray(tk), np.asarray(tg), np.asarray(mk)
        for i in range(tk.shape[0]):
            b = tk[i].tobytes()
            if b in seen:
                continue
            seen.add(b)
            rows.append((tk[i], tg[i], mk[i]))
            if len(rows) == total:
                break
    if len(rows) < total:
        raise ValueError(
            f"could not draw {total} unique MQAR examples in {max_rounds} rounds"
        )
    toks, tgts, masks = zip(*rows)
    return (
        jnp.asarray(np.stack(toks)),
        jnp.asarray(np.stack(tgts)),
        jnp.asarray(np.stack(masks)),
    )


def make_splits(key, cfg: MQARConfig) -> dict[str, Split]:
    """Draw one de-duplicated pool, then partition into disjoint train/val/test.

    Partitioning a single unique pool — rather than drawing each split from an
    independent key — guarantees no example appears in two splits. This matches
    the addition task's protocol exactly (sample once without redrawing, then
    split). ``val`` is omitted when ``cfg.n_val == 0``.
    """
    total = cfg.n_train + cfg.n_val + cfg.n_test
    tokens, targets, mask = _draw_unique_pool(key, total, cfg)

    def sl(a: int, b: int) -> Split:
        return (tokens[a:b], targets[a:b], mask[a:b])

    out = {
        "train": sl(0, cfg.n_train),
        "test": sl(cfg.n_train + cfg.n_val, total),
    }
    if cfg.n_val > 0:
        out["val"] = sl(cfg.n_train, cfg.n_train + cfg.n_val)
    return out


@register_task("mqar")
class MQARTask:
    """Live MQAR task built from an MQARConfig."""

    sources: tuple[str, ...] = ("linattn/tasks/mqar.py",)

    def __init__(self, cfg: MQARConfig):
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.n_train = cfg.n_train
        self.n_test = cfg.n_test
        self.n_val = cfg.n_val

    def make_split(self, key, n: int) -> Split:
        return make_split(key, n, self.cfg)

    def make_splits(self, key) -> dict[str, Split]:
        return make_splits(key, self.cfg)

    def describe(self, model, key) -> None:
        """Print kv pairs, query positions, and predicted vs true values."""
        cfg = self.cfg
        tokens, targets, mask = mqar_example(
            key, cfg.num_kv_pairs, cfg.input_seq_len, cfg.vocab_size, cfg.power_a
        )
        preds = jnp.argmax(model(tokens), axis=-1)
        q_idx = jnp.nonzero(mask, size=cfg.num_kv_pairs)[0]
        print(f"\n  kv pairs : {tokens[: 2 * cfg.num_kv_pairs].tolist()}")
        print(f"  q pos    : {q_idx.tolist()}")
        print(f"  queries  : {tokens[q_idx].tolist()}")
        print(f"  expected : {targets[q_idx].tolist()}")
        print(f"  predicted: {preds[q_idx].tolist()}")


# Debug/smoke-scale presets. level1 is a full experiment, not a preset — it
# lives as a complete RunConfig in the experiments layer.

# toy — first proof test for new recurrent mechanisms.
#
# With one association there is no key disambiguation yet. Passing this says
# the model can carry a prefix value to a query position at all; failing this
# points at write/read plumbing or inner-update stability.
toy = MQARConfig(
    vocab_size=64,
    input_seq_len=16,
    num_kv_pairs=1,
    power_a=0.01,
    n_train=5_000,
    n_test=256,
)

# easy — debug curriculum for new recurrent mechanisms.
#
# Intentionally easier than Zoology level1: small vocab, short context, and
# only two associations. It still requires content-based lookup, but the model
# gets a low-entropy first rung before level1's 4096-way value choice.
easy = MQARConfig(
    vocab_size=128,
    input_seq_len=32,
    num_kv_pairs=2,
    power_a=0.01,
    n_train=20_000,
    n_test=512,
)

PRESETS = {
    "toy": toy,
    "easy": easy,
}
