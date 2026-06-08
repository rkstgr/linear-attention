"""Multi-operand integer addition — an algorithmic task (vs MQAR's recall).

Task definition ported from a standalone reference implementation. A sequence is
the character string of an addition problem, left-padded to a fixed width so the
answer is right-aligned:

    "  2+3=5"          (max_digits=1, num_addends=2)
    "9999+9999+9999=29997"  (max_digits=4, num_addends=3)

The model is a causal LM and is scored only on the answer tokens (everything
strictly after ``=``). Unlike MQAR — which overrides the target at the query
*position* — this is ordinary next-token prediction: the logit at position ``t``
predicts ``tokens[t+1]``, scored where ``tokens[t+1]`` is an answer digit. We
emit position-aligned ``(tokens, targets, mask)`` so the shift is baked in and
the training/eval contract (``loss_and_acc``) is shared with MQAR unchanged.

Parametric axes: ``max_digits`` (per operand) and ``num_addends``. These are the
addition analogue of MQAR's ``(N_KV, T)`` — the difficulty knobs a phase diagram
sweeps.

Vocab (faithful to the reference): digits ``0-9``, ``+``, ``=``, a space, plus a
dedicated pad id.

    keys   : characters in ``VOCAB`` (ids 0..12)
    pad    : id 13 (left-padding only; never a target)

Note on offline splits: examples are sampled i.i.d. with replacement. For small
problem spaces (e.g. 3-digit/2-addend is only ~1e6 distinct problems) this leaks
examples between train and a held-out split. Cross-split disjointness needs the
splits generated together with de-duplication; that belongs with the
train/val/test work, not this port. Documented here so it is not forgotten.
"""

import os as _os

_os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before any jax import

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from linattn.tasks.base import Split, register_task

VOCAB = "0123456789+= "  # 13 chars: digits, '+', '=', space
PAD_ID = len(VOCAB)  # 13
VOCAB_SIZE = PAD_ID + 1  # 14

_CHAR_TO_ID = {c: i for i, c in enumerate(VOCAB)}


def encode(s: str) -> list[int]:
    return [_CHAR_TO_ID[c] for c in s]


def decode(ids) -> str:
    return "".join(VOCAB[i] if i < len(VOCAB) else "_" for i in ids)


def seq_len_for(max_digits: int, num_addends: int) -> int:
    """Fixed sequence width: the longest possible problem string.

    Worst case is every operand at full ``max_digits`` width and the sum at its
    full width, e.g. ``9999+9999+9999=29997``. Shorter problems are left-padded.
    """
    max_operand = 10**max_digits - 1
    max_sum = num_addends * max_operand
    operands = num_addends * max_digits
    pluses = num_addends - 1
    return operands + pluses + 1 + len(str(max_sum))  # +1 for '='


@dataclass(frozen=True)
class AdditionConfig:
    """Addition data spec. ``name`` discriminates the task in the registry."""

    max_digits: int
    num_addends: int
    n_train: int
    n_test: int
    n_val: int = 0
    name: str = "addition"

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    @property
    def seq_len(self) -> int:
        return seq_len_for(self.max_digits, self.num_addends)


def _build_example(numbers, cfg: AdditionConfig):
    """Build position-aligned numpy arrays from explicit operands.

    Returns:
        tokens:  (T,) int32    left-padded "a+b+...=s" character ids
        targets: (T,) int32    next-token labels (tokens shifted left by one)
        mask:    (T,) float32  1.0 where the predicted next token is an answer digit
    """
    T = cfg.seq_len
    total = int(sum(int(n) for n in numbers))
    s = "+".join(str(int(n)) for n in numbers) + "=" + str(total)
    ids = encode(s)
    pad = T - len(ids)
    assert pad >= 0, f"problem '{s}' (len {len(ids)}) exceeds seq_len {T}"

    tokens = np.full(T, PAD_ID, dtype=np.int32)
    tokens[pad:] = ids

    # Answer tokens are the right-aligned tail (the sum digits).
    answer_pos = np.zeros(T, dtype=np.float32)
    answer_pos[T - len(str(total)) :] = 1.0  # 1.0 on answer-digit positions

    # Position-aligned next-token form: logit[t] predicts tokens[t+1].
    targets = np.zeros(T, dtype=np.int32)
    targets[:-1] = tokens[1:]
    mask = np.zeros(T, dtype=np.float32)
    mask[:-1] = answer_pos[1:]
    return tokens, targets, mask


def _sample_numbers(rng: np.random.Generator, cfg: AdditionConfig):
    return rng.integers(0, 10**cfg.max_digits, size=cfg.num_addends)


def addition_example(rng: np.random.Generator, cfg: AdditionConfig):
    """Sample one addition example as position-aligned numpy arrays."""
    return _build_example(_sample_numbers(rng, cfg), cfg)


def _seed_from_key(key) -> int:
    """Deterministic numpy seed from a jax key (reproducible offline pools)."""
    return int(jax.random.randint(key, (), 0, np.iinfo(np.int32).max))


def _stack(rows):
    toks, tgts, masks = zip(*rows)
    return (
        jnp.asarray(np.stack(toks)),
        jnp.asarray(np.stack(tgts)),
        jnp.asarray(np.stack(masks)),
    )


def make_split(key, n: int, cfg: AdditionConfig) -> Split:
    """Pre-generate n addition examples (i.i.d., with replacement). Shape (n, T).

    The jax ``key`` seeds a numpy generator deterministically, so a fixed key and
    config reproduce identical bytes (matching MQAR's offline-pool protocol and
    the executor's content-addressing). Use ``make_splits`` for disjoint
    train/val/test pools.
    """
    rng = np.random.default_rng(_seed_from_key(key))
    return _stack([addition_example(rng, cfg) for _ in range(n)])


def make_splits(key, cfg: AdditionConfig) -> dict[str, Split]:
    """Disjoint train/val/test pools via de-duplicated sampling.

    Examples are unique operand tuples, so no problem appears in more than one
    split — the leakage that i.i.d. sampling causes on small problem spaces (a
    real concern at e.g. 3-digit/2-addend, ~1e6 problems) is removed. ``val`` is
    omitted when ``cfg.n_val == 0``.
    """
    n_val = cfg.n_val
    total = cfg.n_train + n_val + cfg.n_test
    space = (10**cfg.max_digits) ** cfg.num_addends
    if total > space:
        raise ValueError(
            f"requested {total} unique examples but only {space} exist for "
            f"max_digits={cfg.max_digits}, num_addends={cfg.num_addends}"
        )
    rng = np.random.default_rng(_seed_from_key(key))
    seen: set[tuple[int, ...]] = set()
    tuples: list = []
    while len(tuples) < total:
        nums = _sample_numbers(rng, cfg)
        t = tuple(int(x) for x in nums)
        if t in seen:
            continue
        seen.add(t)
        tuples.append(nums)

    def build(rows):
        return _stack([_build_example(nums, cfg) for nums in rows])

    out = {
        "train": build(tuples[: cfg.n_train]),
        "test": build(tuples[cfg.n_train + n_val :]),
    }
    if n_val > 0:
        out["val"] = build(tuples[cfg.n_train : cfg.n_train + n_val])
    return out


@register_task("addition")
class AdditionTask:
    """Live addition task built from an AdditionConfig."""

    sources: tuple[str, ...] = ("linattn/tasks/addition.py",)

    def __init__(self, cfg: AdditionConfig):
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.seq_len = cfg.seq_len
        self.n_train = cfg.n_train
        self.n_test = cfg.n_test
        self.n_val = cfg.n_val

    def make_split(self, key, n: int) -> Split:
        return make_split(key, n, self.cfg)

    def make_splits(self, key) -> dict[str, Split]:
        return make_splits(key, self.cfg)

    def describe(self, model, key) -> None:
        """Print one problem with the model's predicted vs true answer digits."""
        seed = int(jax.random.randint(key, (), 0, np.iinfo(np.int32).max))
        tokens, targets, mask = addition_example(np.random.default_rng(seed), self.cfg)
        preds = jnp.argmax(model(jnp.asarray(tokens)), axis=-1)
        q_idx = np.nonzero(np.asarray(mask))[0]
        print(f"\n  problem  : {decode(tokens.tolist())!r}")
        print(f"  expected : {decode(np.asarray(targets)[q_idx].tolist())!r}")
        print(f"  predicted: {decode(np.asarray(preds)[q_idx].tolist())!r}")


# Debug-scale preset for smoke tests. Full experiment cells carry their own
# AdditionConfig in the experiments layer (mirroring MQAR's level1).
toy = AdditionConfig(max_digits=1, num_addends=2, n_train=5_000, n_test=512)

PRESETS = {"toy": toy}
