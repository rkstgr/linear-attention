"""Shared training loop for MQAR.

Every model trains the same way: pre-generate train/test splits per the
Config, run AdamW, evaluate after each epoch, stop early when test accuracy
crosses `target_acc` or fails to improve for `patience_epochs`. Models differ
only in architecture, not in protocol — so the loop lives here and each
model's __main__ is a one-liner.
"""

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from data import Config, make_split, mqar_example


def tree_l2_norm(tree):
    """L2 norm over all inexact array leaves in a pytree."""
    leaves = [x for x in jax.tree.leaves(tree) if eqx.is_inexact_array(x)]
    if not leaves:
        return jnp.array(0.0)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


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


def train_and_eval(
    model,
    cfg: Config,
    key,
    opt=None,
    log_fn=None,
    *,
    fail_fast_nonfinite: bool = True,
    return_info: bool = False,
):
    """Train `model` on MQAR per `cfg` with early stopping.

    Pre-generates a fixed train and test split (Zoology-style caching), runs
    AdamW for at most cfg.max_epochs, stops early when test_acc >= target_acc
    or test_acc has not improved by 1e-3 for patience_epochs epochs.

    If `opt` is None, uses plain AdamW(cfg.learning_rate). Pass an explicit
    optax optimizer (e.g. with warmup or decay) to override. If `log_fn` is
    passed, it receives `(metrics, model)` at the same cadence as stdout.
    With `fail_fast_nonfinite=True`, the loop stops at the first non-finite
    loss/optimizer/model norm and returns the last known finite model.

    Returns (trained_model, history) where history is a list of per-epoch dicts.
    If `return_info=True`, also returns a stop-info dict.
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
        grad_norm = tree_l2_norm(grads)
        updates, opt_state = opt.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        update_norm = tree_l2_norm(updates)
        model = eqx.apply_updates(model, updates)
        param_norm = tree_l2_norm(model)
        all_finite = jnp.all(
            jnp.isfinite(jnp.stack([loss, acc, grad_norm, update_norm, param_norm]))
        )
        return model, opt_state, loss, acc, grad_norm, update_norm, param_norm, all_finite

    n_full = (cfg.n_train // cfg.batch_size) * cfg.batch_size
    n_batches = n_full // cfg.batch_size
    if n_batches == 0:
        raise ValueError(
            f"n_train={cfg.n_train} is smaller than batch_size={cfg.batch_size}; "
            "increase n_train or reduce batch_size."
        )
    log_every = max(1, n_batches // 20)  # ~10 intra-epoch progress lines
    best_acc = 0.0
    patience = 0
    history = []
    global_step = 0
    stop_info = {
        "stop_reason": "max_epochs",
        "nonfinite": False,
        "nonfinite_epoch": None,
        "nonfinite_step": None,
        "nonfinite_global_step": None,
        "nonfinite_loss": None,
        "nonfinite_grad_norm": None,
        "nonfinite_update_norm": None,
        "nonfinite_param_norm": None,
    }
    should_stop = False

    for epoch in range(cfg.max_epochs):
        k_loop, k_perm = jax.random.split(k_loop)
        perm = jax.random.permutation(k_perm, cfg.n_train)[:n_full]
        idx = perm.reshape(-1, cfg.batch_size)

        losses, accs = [], []
        # Track wall-clock between log points. The first reading includes JIT
        # compile; non-finite checks synchronize once per step.
        last_log_t = time.perf_counter()
        last_log_step = 0
        for i, batch_idx in enumerate(idx):
            x = train_tokens[batch_idx]
            y = train_targets[batch_idx]
            m = train_mask[batch_idx]
            prev_model = model
            prev_opt_state = opt_state
            (
                model,
                opt_state,
                loss,
                acc,
                grad_norm,
                update_norm,
                param_norm,
                all_finite,
            ) = step(
                model, opt_state, x, y, m
            )
            global_step += 1
            all_finite_b = bool(all_finite)
            loss_f = float(loss)
            acc_f = float(acc)
            grad_norm_f = float(grad_norm)
            update_norm_f = float(update_norm)
            param_norm_f = float(param_norm)

            if fail_fast_nonfinite and not all_finite_b:
                model = prev_model
                opt_state = prev_opt_state
                now = time.perf_counter()
                ms = (now - last_log_t) / max((i + 1) - last_log_step, 1) * 1000
                stop_info.update(
                    {
                        "stop_reason": "nonfinite",
                        "nonfinite": True,
                        "nonfinite_epoch": epoch,
                        "nonfinite_step": i + 1,
                        "nonfinite_global_step": global_step,
                        "nonfinite_loss": loss_f,
                        "nonfinite_grad_norm": grad_norm_f,
                        "nonfinite_update_norm": update_norm_f,
                        "nonfinite_param_norm": param_norm_f,
                    }
                )
                print(
                    f"non-finite stop @ epoch {epoch} step {i + 1}/{n_batches}: "
                    f"loss {loss_f}  grad_norm {grad_norm_f}  "
                    f"update_norm {update_norm_f}  param_norm {param_norm_f}"
                )
                if log_fn is not None:
                    log_fn(
                        {
                            "kind": "nonfinite",
                            "epoch": epoch,
                            "step": i + 1,
                            "global_step": global_step,
                            "learning/train_loss": loss_f,
                            "learning/train_acc": acc_f,
                            "runtime/ms_per_step": ms,
                            "stability/grad_norm": grad_norm_f,
                            "stability/update_norm": update_norm_f,
                            "stability/param_norm": param_norm_f,
                            "health/all_finite": 0.0,
                            "health/nonfinite": 1.0,
                            "health/nonfinite_epoch": epoch,
                            "health/nonfinite_step": i + 1,
                            "health/nonfinite_global_step": global_step,
                        },
                        model,
                    )
                should_stop = True
                break

            losses.append(loss)
            accs.append(acc)
            # always log step 1 (JIT compile done) + every log_every after
            if i == 0 or (i + 1) % log_every == 0:
                now = time.perf_counter()
                ms = (now - last_log_t) / ((i + 1) - last_log_step) * 1000
                print(
                    f"  epoch {epoch:3d} step {i + 1:4d}/{n_batches}  "
                    f"loss {loss_f:.4f}  acc {acc_f:.3f}  "
                    f"{ms:7.1f} ms/step"
                )
                if log_fn is not None:
                    log_fn(
                        {
                            "kind": "train",
                            "epoch": epoch,
                            "step": i + 1,
                            "global_step": global_step,
                            "learning/train_loss": loss_f,
                            "learning/train_acc": acc_f,
                            "runtime/ms_per_step": ms,
                            "stability/grad_norm": grad_norm_f,
                            "stability/update_norm": update_norm_f,
                            "stability/param_norm": param_norm_f,
                            "health/all_finite": 1.0,
                            "health/nonfinite": 0.0,
                        },
                        model,
                    )
                last_log_t = now
                last_log_step = i + 1

        if should_stop:
            break

        train_loss = float(jnp.mean(jnp.stack(losses)))
        train_acc = float(jnp.mean(jnp.stack(accs)))
        test_acc = evaluate(model, test_data, cfg.eval_batch_size)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
            }
        )
        print(
            f"epoch {epoch:3d}  train_loss {train_loss:.4f}  "
            f"train_acc {train_acc:.3f}  test_acc {test_acc:.3f}"
        )
        if log_fn is not None:
            log_fn(
                {
                    "kind": "epoch",
                    "epoch": epoch,
                    "global_step": global_step,
                    "learning/epoch_train_loss": train_loss,
                    "learning/epoch_train_acc": train_acc,
                    "learning/test_acc": test_acc,
                    "health/nonfinite": 0.0,
                },
                model,
            )

        if test_acc >= cfg.target_acc:
            print(
                f"early stop @ epoch {epoch}: test_acc {test_acc:.3f} >= {cfg.target_acc}"
            )
            stop_info["stop_reason"] = "target_acc"
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
                stop_info["stop_reason"] = "patience"
                break

    if return_info:
        return model, history, stop_info
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
