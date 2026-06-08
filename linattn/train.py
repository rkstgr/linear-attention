"""Training run, decomposed into composable pieces.

`fit` orchestrates; each responsibility is a named, independently testable unit:

- `train_step`   — one jitted optimizer step, returning `StepStats`;
- `evaluate`     — mean masked-position accuracy over a split;
- `EarlyStopping`— the target-acc / patience policy;
- `Reporter`     — where step/epoch/stop events go (stdout, W&B, ...);
- `TrainResult`  — the model plus history and stop diagnostics.

Every task trains the same way: pre-generate fixed train/test splits, run
AdamW, evaluate after each epoch, stop early on target accuracy or no
improvement. Tasks differ only in their data (`task.make_split`), models only
in architecture — so the loop lives here once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from linattn.config import TrainConfig
from linattn.tasks.base import Task


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
    """Mean masked-position (partial) accuracy over a split.

    Retained for callers that only need the partial metric; the training loop
    uses ``evaluate_metrics`` so complete and partial come from the same pass.
    """
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


@eqx.filter_jit
def _eval_batch_metrics(model, tokens, targets, mask):
    """Per-batch sums for exact across-batch accumulation of both metrics."""
    logits = jax.vmap(model)(tokens)
    preds = jnp.argmax(logits, axis=-1)
    correct = (preds == targets).astype(jnp.float32) * mask
    correct_positions = correct.sum()
    total_positions = mask.sum()
    scored_per_example = mask.sum(-1)
    per_example_complete = (correct.sum(-1) == scored_per_example) & (
        scored_per_example > 0
    )
    complete_examples = per_example_complete.astype(jnp.float32).sum()
    scored_examples = (scored_per_example > 0).astype(jnp.float32).sum()
    return correct_positions, total_positions, complete_examples, scored_examples


def evaluate_metrics(model, data, batch_size: int) -> dict[str, float]:
    """Complete and partial accuracy over a split.

    Accumulates per-batch numerator/denominator across the full pass so the
    result is identical to a single-batch computation regardless of
    ``batch_size``. Returns ``{"accuracy": <complete>, "partial_accuracy": <partial>}``.
    """
    tokens, targets, mask = data
    n = tokens.shape[0]
    n_full = (n // batch_size) * batch_size
    cp = tp = ce = se = 0.0
    for i in range(0, n_full, batch_size):
        cp_i, tp_i, ce_i, se_i = _eval_batch_metrics(
            model,
            tokens[i : i + batch_size],
            targets[i : i + batch_size],
            mask[i : i + batch_size],
        )
        cp += float(cp_i)
        tp += float(tp_i)
        ce += float(ce_i)
        se += float(se_i)
    partial = cp / max(tp, 1.0)
    complete = ce / max(se, 1.0)
    return {"accuracy": complete, "partial_accuracy": partial}


class StepStats(NamedTuple):
    """Per-step diagnostics produced by `train_step`."""

    loss: Array
    acc: Array
    grad_norm: Array
    update_norm: Array
    param_norm: Array
    all_finite: Array


@eqx.filter_jit
def train_step(model, opt_state, batch, opt):
    """One AdamW-style step. `opt` is a static (non-array) argument.

    Returns the updated model, optimizer state, and a `StepStats` with loss,
    accuracy, and grad/update/param norms plus an all-finite flag.
    """
    x, y, m = batch
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
    return model, opt_state, StepStats(
        loss, acc, grad_norm, update_norm, param_norm, all_finite
    )


@dataclass
class EarlyStopping:
    """Target-accuracy / patience stopping policy.

    `update(test_acc)` returns a stop reason (`"target_acc"` or `"patience"`)
    or None to continue. Mirrors the original loop: target check first, then
    best/patience bookkeeping.
    """

    target_acc: float
    patience_epochs: int
    best_acc: float = 0.0
    _patience: int = field(default=0, repr=False)

    def update(self, test_acc: float) -> str | None:
        if test_acc >= self.target_acc:
            return "target_acc"
        if test_acc > self.best_acc + 1e-3:
            self.best_acc = test_acc
            self._patience = 0
        else:
            self._patience += 1
            if self._patience >= self.patience_epochs:
                return "patience"
        return None


class Reporter:
    """No-op reporter. Override the hooks you care about.

    The training loop emits structured events; formatting and sinks (stdout,
    W&B, ...) are the reporter's concern, not the loop's.
    """

    def on_step(self, stats, *, epoch, step, n_batches, global_step, ms):
        pass

    def on_epoch(self, *, epoch, global_step, train_loss, metrics):
        """`metrics` maps split name to ``{accuracy, partial_accuracy}`` dicts."""
        pass

    def on_nonfinite(self, stats, *, epoch, step, n_batches, global_step, ms):
        pass

    def on_stop(self, *, reason, epoch, select_acc, best_acc):
        pass


class StdoutReporter(Reporter):
    """Reproduces the original stdout training log."""

    def on_step(self, stats, *, epoch, step, n_batches, global_step, ms):
        print(
            f"  epoch {epoch:3d} step {step:4d}/{n_batches}  "
            f"loss {float(stats.loss):.4f}  acc {float(stats.acc):.3f}  "
            f"{ms:7.1f} ms/step"
        )

    def on_epoch(self, *, epoch, global_step, train_loss, metrics):
        train_p = metrics["train"]["partial_accuracy"]
        test_p = metrics["test"]["partial_accuracy"]
        test_c = metrics["test"]["accuracy"]
        line = (
            f"epoch {epoch:3d}  train_loss {train_loss:.4f}  "
            f"train_acc {train_p:.3f}  test_acc {test_p:.3f}  "
            f"test_complete {test_c:.3f}"
        )
        if "val" in metrics:
            val_p = metrics["val"]["partial_accuracy"]
            val_c = metrics["val"]["accuracy"]
            line += f"  val_acc {val_p:.3f}  val_complete {val_c:.3f}"
        print(line)

    def on_nonfinite(self, stats, *, epoch, step, n_batches, global_step, ms):
        print(
            f"non-finite stop @ epoch {epoch} step {step}/{n_batches}: "
            f"loss {float(stats.loss)}  grad_norm {float(stats.grad_norm)}  "
            f"update_norm {float(stats.update_norm)}  "
            f"param_norm {float(stats.param_norm)}"
        )

    def on_stop(self, *, reason, epoch, select_acc, best_acc):
        if reason == "target_acc":
            print(
                f"early stop @ epoch {epoch}: select_acc {select_acc:.3f} (target reached)"
            )
        elif reason == "patience":
            print(f"early stop @ epoch {epoch}: no improvement (best {best_acc:.3f})")


class WandbReporter(Reporter):
    """Logs the same metric keys the old `log_fn` did to a W&B run."""

    def __init__(self, run):
        self.run = run

    def on_step(self, stats, *, epoch, step, n_batches, global_step, ms):
        self.run.log(
            {
                "epoch": epoch,
                "step": step,
                "learning/train_loss": float(stats.loss),
                "learning/train_acc": float(stats.acc),
                "runtime/ms_per_step": ms,
                "stability/grad_norm": float(stats.grad_norm),
                "stability/update_norm": float(stats.update_norm),
                "stability/param_norm": float(stats.param_norm),
                "health/all_finite": 1.0,
                "health/nonfinite": 0.0,
            },
            step=global_step,
        )

    def on_epoch(self, *, epoch, global_step, train_loss, metrics):
        payload = {
            "epoch": epoch,
            "learning/epoch_train_loss": train_loss,
            "health/nonfinite": 0.0,
        }
        for split, m in metrics.items():
            payload[f"learning/{split}_accuracy"] = float(m["accuracy"])
            payload[f"learning/{split}_partial_accuracy"] = float(m["partial_accuracy"])
        self.run.log(payload, step=global_step)

    def on_nonfinite(self, stats, *, epoch, step, n_batches, global_step, ms):
        self.run.log(
            {
                "epoch": epoch,
                "step": step,
                "learning/train_loss": float(stats.loss),
                "learning/train_acc": float(stats.acc),
                "runtime/ms_per_step": ms,
                "stability/grad_norm": float(stats.grad_norm),
                "stability/update_norm": float(stats.update_norm),
                "stability/param_norm": float(stats.param_norm),
                "health/all_finite": 0.0,
                "health/nonfinite": 1.0,
                "health/nonfinite_epoch": epoch,
                "health/nonfinite_step": step,
                "health/nonfinite_global_step": global_step,
            },
            step=global_step,
        )


class MultiReporter(Reporter):
    """Fan an event out to several reporters."""

    def __init__(self, reporters):
        self.reporters = list(reporters)

    def on_step(self, *args, **kwargs):
        for r in self.reporters:
            r.on_step(*args, **kwargs)

    def on_epoch(self, *args, **kwargs):
        for r in self.reporters:
            r.on_epoch(*args, **kwargs)

    def on_nonfinite(self, *args, **kwargs):
        for r in self.reporters:
            r.on_nonfinite(*args, **kwargs)

    def on_stop(self, *args, **kwargs):
        for r in self.reporters:
            r.on_stop(*args, **kwargs)


@dataclass
class TrainResult:
    model: object
    history: list[dict]
    stop_info: dict


def _initial_stop_info() -> dict:
    return {
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


def fit(
    model,
    task: Task,
    train: TrainConfig,
    key,
    *,
    opt=None,
    reporter: Reporter | None = None,
) -> TrainResult:
    """Train `model` on `task` per `train` with early stopping.

    Pre-generates fixed splits (Zoology-style caching). When the task defines a
    validation split (`n_val > 0`), selection and early stopping run on
    validation accuracy and the test split is held out for reporting; otherwise
    it falls back to the legacy two-way path and selects on test. Runs AdamW for
    at most `train.max_epochs`, stopping early on target accuracy or
    `patience_epochs` without improvement. The loop fails fast on the first
    non-finite step and returns the last known finite model.

    If `opt` is None, uses plain AdamW(`train.learning_rate`). Pass an explicit
    optax optimizer to override. Events go to `reporter` (default: stdout).
    """
    if reporter is None:
        reporter = StdoutReporter()

    if getattr(task, "n_val", 0) > 0:
        # Three-way split: select/early-stop on val, report final on test.
        k_data, k_loop = jax.random.split(key)
        splits = task.make_splits(k_data)
        train_data, val_data, test_data = splits["train"], splits["val"], splits["test"]
    else:
        # Legacy two-way path with identical keying, so existing cached runs
        # (capacity/retention, no val split) are byte-for-byte unchanged.
        k_train, k_test, k_loop = jax.random.split(key, 3)
        train_data = task.make_split(k_train, task.n_train)
        test_data = task.make_split(k_test, task.n_test)
        val_data = None
    train_tokens, train_targets, train_mask = train_data

    if opt is None:
        opt = optax.adamw(train.learning_rate)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    n_full = (task.n_train // train.batch_size) * train.batch_size
    n_batches = n_full // train.batch_size
    if n_batches == 0:
        raise ValueError(
            f"n_train={task.n_train} is smaller than batch_size={train.batch_size}; "
            "increase n_train or reduce batch_size."
        )
    log_every = max(1, n_batches // 20)  # ~10 intra-epoch progress lines

    stopper = EarlyStopping(train.target_acc, train.patience_epochs)
    history: list[dict] = []
    stop_info = _initial_stop_info()
    global_step = 0

    for epoch in range(train.max_epochs):
        k_loop, k_perm = jax.random.split(k_loop)
        perm = jax.random.permutation(k_perm, task.n_train)[:n_full]
        idx = perm.reshape(-1, train.batch_size)

        losses, accs = [], []
        # Track wall-clock between log points. The first reading includes JIT
        # compile; the non-finite check synchronizes once per step.
        last_log_t = time.perf_counter()
        last_log_step = 0
        nonfinite = False
        for i, batch_idx in enumerate(idx):
            batch = (
                train_tokens[batch_idx],
                train_targets[batch_idx],
                train_mask[batch_idx],
            )
            prev_model, prev_opt_state = model, opt_state
            model, opt_state, stats = train_step(model, opt_state, batch, opt)
            global_step += 1

            if not bool(stats.all_finite):
                model, opt_state = prev_model, prev_opt_state
                now = time.perf_counter()
                ms = (now - last_log_t) / max((i + 1) - last_log_step, 1) * 1000
                stop_info.update(
                    {
                        "stop_reason": "nonfinite",
                        "nonfinite": True,
                        "nonfinite_epoch": epoch,
                        "nonfinite_step": i + 1,
                        "nonfinite_global_step": global_step,
                        "nonfinite_loss": float(stats.loss),
                        "nonfinite_grad_norm": float(stats.grad_norm),
                        "nonfinite_update_norm": float(stats.update_norm),
                        "nonfinite_param_norm": float(stats.param_norm),
                    }
                )
                reporter.on_nonfinite(
                    stats,
                    epoch=epoch,
                    step=i + 1,
                    n_batches=n_batches,
                    global_step=global_step,
                    ms=ms,
                )
                nonfinite = True
                break

            losses.append(stats.loss)
            accs.append(stats.acc)
            # always log step 1 (JIT compile done) + every log_every after
            if i == 0 or (i + 1) % log_every == 0:
                now = time.perf_counter()
                ms = (now - last_log_t) / ((i + 1) - last_log_step) * 1000
                reporter.on_step(
                    stats,
                    epoch=epoch,
                    step=i + 1,
                    n_batches=n_batches,
                    global_step=global_step,
                    ms=ms,
                )
                last_log_t = now
                last_log_step = i + 1

        if nonfinite:
            break

        train_loss = float(jnp.mean(jnp.stack(losses)))
        train_acc = float(jnp.mean(jnp.stack(accs)))
        # Dual (complete + partial) metrics, evaluated once per split per epoch.
        # `train_metrics` uses the eval model on train data — a slight cost paid
        # so the headline is reported on the same protocol as val/test.
        train_metrics = evaluate_metrics(model, train_data, train.eval_batch_size)
        test_metrics = evaluate_metrics(model, test_data, train.eval_batch_size)
        metrics: dict[str, dict[str, float]] = {
            "train": train_metrics,
            "test": test_metrics,
        }
        if val_data is not None:
            metrics["val"] = evaluate_metrics(model, val_data, train.eval_batch_size)

        # Selection signal: val partial when available, else test partial
        # (legacy). Partial is the smooth signal, matching the legacy target_acc
        # semantics; complete accuracy is reported but not selected on (it sits
        # at 0 then jumps, which is poor for patience).
        select_split = "val" if "val" in metrics else "test"
        select_acc = metrics[select_split]["partial_accuracy"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_step_acc": train_acc,  # streaming train_step partial acc
        }
        for split, m in metrics.items():
            record[f"{split}_accuracy"] = float(m["accuracy"])
            record[f"{split}_partial_accuracy"] = float(m["partial_accuracy"])
        history.append(record)
        reporter.on_epoch(
            epoch=epoch,
            global_step=global_step,
            train_loss=train_loss,
            metrics=metrics,
        )

        reason = stopper.update(select_acc)
        if reason is not None:
            stop_info["stop_reason"] = reason
            reporter.on_stop(
                reason=reason,
                epoch=epoch,
                select_acc=select_acc,
                best_acc=stopper.best_acc,
            )
            break

    return TrainResult(model=model, history=history, stop_info=stop_info)
