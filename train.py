"""Shared training loop for MQAR.

Every model trains the same way: pre-generate train/test splits per the
Config, run AdamW, evaluate after each epoch, stop early when test accuracy
crosses `target_acc` or fails to improve for `patience_epochs`. Models differ
only in architecture, not in protocol — so the loop lives here and each
model's __main__ is a one-liner.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from data import Config, make_split, mqar_example


def loss_and_acc(model, tokens, targets, mask):
    """Cross-entropy and accuracy at masked (query) positions only."""
    logits = jax.vmap(model)(tokens)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    loss = (losses * mask).sum() / mask.sum()
    preds = jnp.argmax(logits, axis=-1)
    acc = ((preds == targets).astype(jnp.float32) * mask).sum() / mask.sum()
    return loss, acc


@eqx.filter_jit
def _eval_batch(model, tokens, targets, mask):
    return loss_and_acc(model, tokens, targets, mask)


def evaluate(model, data, batch_size: int) -> float:
    """Mean masked-position accuracy over the test set."""
    tokens, targets, mask = data
    n = tokens.shape[0]
    n_full = (n // batch_size) * batch_size
    accs = []
    for i in range(0, n_full, batch_size):
        _, acc = _eval_batch(
            model,
            tokens[i : i + batch_size],
            targets[i : i + batch_size],
            mask[i : i + batch_size],
        )
        accs.append(acc)
    return float(jnp.mean(jnp.stack(accs)))


def train_and_eval(model, cfg: Config, key, opt=None):
    """Train `model` on MQAR per `cfg` with early stopping.

    Pre-generates a fixed train and test split (Zoology-style caching), runs
    AdamW for at most cfg.max_epochs, stops early when test_acc >= target_acc
    or test_acc has not improved by 1e-3 for patience_epochs epochs.

    If `opt` is None, uses plain AdamW(cfg.learning_rate). Pass an explicit
    optax optimizer (e.g. with warmup or decay) to override.

    Returns (trained_model, history) where history is a list of per-epoch dicts.
    """
    k_train, k_test, k_loop = jax.random.split(key, 3)
    train_data = make_split(k_train, cfg.n_train, cfg)
    test_data = make_split(k_test, cfg.n_test, cfg)
    train_tokens, train_targets, train_mask = train_data

    if opt is None:
        opt = optax.adamw(cfg.learning_rate)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(model, opt_state, x, y, m):
        (loss, acc), grads = eqx.filter_value_and_grad(loss_and_acc, has_aux=True)(
            model, x, y, m
        )
        updates, opt_state = opt.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, acc

    n_full = (cfg.n_train // cfg.batch_size) * cfg.batch_size
    n_batches = n_full // cfg.batch_size
    log_every = max(1, n_batches // 10)  # ~10 intra-epoch progress lines
    best_acc = 0.0
    patience = 0
    history = []

    for epoch in range(cfg.max_epochs):
        k_loop, k_perm = jax.random.split(k_loop)
        perm = jax.random.permutation(k_perm, cfg.n_train)[:n_full]
        idx = perm.reshape(-1, cfg.batch_size)

        losses, accs = [], []
        for i, batch_idx in enumerate(idx):
            x = train_tokens[batch_idx]
            y = train_targets[batch_idx]
            m = train_mask[batch_idx]
            model, opt_state, loss, acc = step(model, opt_state, x, y, m)
            losses.append(loss)
            accs.append(acc)
            # always log step 1 (JIT compile done) + every log_every after
            if i == 0 or (i + 1) % log_every == 0:
                print(
                    f"  epoch {epoch:3d} step {i + 1:4d}/{n_batches}  "
                    f"loss {float(loss):.4f}  acc {float(acc):.3f}"
                )

        train_loss = float(jnp.mean(jnp.stack(losses)))
        train_acc = float(jnp.mean(jnp.stack(accs)))
        test_acc = evaluate(model, test_data, cfg.eval_batch_size)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
        })
        print(
            f"epoch {epoch:3d}  train_loss {train_loss:.4f}  "
            f"train_acc {train_acc:.3f}  test_acc {test_acc:.3f}"
        )

        if test_acc >= cfg.target_acc:
            print(f"early stop @ epoch {epoch}: test_acc {test_acc:.3f} >= {cfg.target_acc}")
            break
        if test_acc > best_acc + 1e-3:
            best_acc = test_acc
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience_epochs:
                print(
                    f"early stop @ epoch {epoch}: no improvement for "
                    f"{cfg.patience_epochs} epochs (best {best_acc:.3f})"
                )
                break

    return model, history


def inspect_example(model, key, cfg: Config):
    """Print kv pairs, query positions, and predicted vs true values."""
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
