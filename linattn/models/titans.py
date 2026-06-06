"""Titans-style MLP memory, recurrent reference implementation.

Run MQAR experiments via `uv run python -m experiments.titans`.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)

CONV_SIZE = 4


def causal_dwconv(x: Array, w: Array) -> Array:
    """Depthwise causal 1D conv."""

    T = x.shape[0]
    K = w.shape[0]
    x_pad = jnp.pad(x, ((K - 1, 0), (0, 0)))
    return sum(w[k] * x_pad[K - 1 - k : K - 1 - k + T] for k in range(K))


def silu_grad(x: Array) -> Array:
    """Derivative of SiLU(x) = x * sigmoid(x)."""

    s = jax.nn.sigmoid(x)
    return s + x * s * (1.0 - s)


class Titans(eqx.Module):
    """Titans mixer with a two-layer MLP as per-head fast memory."""

    Wq: Array
    Wk: Array
    Wv: Array
    Wbeta: Array
    Wnu: Array
    bnu: Array
    Walpha: Array
    balpha: Array
    dt_logit: Array
    Wo: Array
    Cq: Array
    Ck: Array
    Cv: Array
    mem_W1: Array
    mem_W2: Array
    n_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    memory_hidden: int = eqx.field(static=True)
    max_inner_lr: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        n_heads: int,
        key,
        memory_mult: int = 4,
        max_inner_lr: float = 0.05,
    ):
        assert dim % n_heads == 0
        keys = jax.random.split(key, 12)
        s = 1.0 / jnp.sqrt(dim)
        sc = 1.0 / jnp.sqrt(CONV_SIZE)
        self.Wq = jax.random.normal(keys[0], (dim, dim)) * s
        self.Wk = jax.random.normal(keys[1], (dim, dim)) * s
        self.Wv = jax.random.normal(keys[2], (dim, dim)) * s
        self.Wbeta = jax.random.normal(keys[3], (dim, n_heads)) * s
        self.Wnu = jax.random.normal(keys[4], (dim, n_heads)) * s
        self.bnu = jnp.ones((n_heads,))
        self.Walpha = jax.random.normal(keys[5], (dim, n_heads)) * s
        self.balpha = jnp.zeros((n_heads,))
        self.dt_logit = jnp.full((n_heads,), -10.0)
        self.Wo = jax.random.normal(keys[6], (dim, dim)) * s
        self.Cq = jax.random.normal(keys[7], (CONV_SIZE, dim)) * sc
        self.Ck = jax.random.normal(keys[8], (CONV_SIZE, dim)) * sc
        self.Cv = jax.random.normal(keys[9], (CONV_SIZE, dim)) * sc

        head_dim = dim // n_heads
        memory_hidden = memory_mult * head_dim
        s1 = 1.0 / jnp.sqrt(head_dim)
        s2 = 1.0 / jnp.sqrt(memory_hidden)
        self.mem_W1 = jax.random.normal(
            keys[10], (n_heads, head_dim, memory_hidden)
        ) * s1
        self.mem_W2 = jax.random.normal(
            keys[11], (n_heads, memory_hidden, head_dim)
        ) * s2

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.memory_hidden = memory_hidden
        self.max_inner_lr = max_inner_lr

    def _project(self, x: Array):
        T, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        q = (
            jax.nn.silu(causal_dwconv(x @ self.Wq, self.Cq))
            .reshape(T, H, Dh)
            .transpose(1, 0, 2)
        )
        k = (
            jax.nn.silu(causal_dwconv(x @ self.Wk, self.Ck))
            .reshape(T, H, Dh)
            .transpose(1, 0, 2)
        )
        v = (
            jax.nn.silu(causal_dwconv(x @ self.Wv, self.Cv))
            .reshape(T, H, Dh)
            .transpose(1, 0, 2)
        )

        beta = self.max_inner_lr * jax.nn.sigmoid(x @ self.Wbeta).transpose(1, 0)
        nu = jax.nn.sigmoid(x @ self.Wnu + self.bnu).transpose(1, 0)

        g = jax.nn.sigmoid(x @ self.Walpha + self.balpha)
        dt = jax.nn.softplus(self.dt_logit)
        alpha = jnp.exp(-dt * g).transpose(1, 0)

        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
        return q, k, v, beta, nu, alpha

    def __call__(self, x: Array) -> Array:
        T, D = x.shape
        q, k, v, beta, nu, alpha = self._project(x)

        def mlp_apply(W1, W2, z):
            return jax.nn.silu(z @ W1) @ W2

        def per_head(q, k, v, beta, nu, alpha, W1_init, W2_init):
            def step(carry, inputs):
                W1, W2, mW1, mW2 = carry
                q_t, k_t, v_t, beta_t, nu_t, alpha_t = inputs

                pre = k_t @ W1
                h = jax.nn.silu(pre)
                pred = h @ W2
                residual = v_t - pred

                grad_W2 = -jnp.outer(h, residual)
                hidden_err = (W2 @ residual) * silu_grad(pre)
                grad_W1 = -jnp.outer(k_t, hidden_err)

                mW1 = nu_t * mW1 + grad_W1
                mW2 = nu_t * mW2 + grad_W2
                W1 = alpha_t * W1 - beta_t * mW1
                W2 = alpha_t * W2 - beta_t * mW2

                o_t = mlp_apply(W1, W2, q_t)
                return (W1, W2, mW1, mW2), o_t

            mW1_0 = jnp.zeros_like(W1_init)
            mW2_0 = jnp.zeros_like(W2_init)
            carry0 = (W1_init, W2_init, mW1_0, mW2_0)
            _, out = jax.lax.scan(step, carry0, (q, k, v, beta, nu, alpha))
            return out

        out = jax.vmap(per_head)(
            q, k, v, beta, nu, alpha, self.mem_W1, self.mem_W2
        )

        out = out.transpose(1, 0, 2).reshape(T, D)
        return out @ self.Wo

    def trace(self, x: Array):
        """Return mixer output plus per-token fast-memory diagnostics."""

        T, D = x.shape
        q, k, v, beta, nu, alpha = self._project(x)

        def mlp_apply(W1, W2, z):
            return jax.nn.silu(z @ W1) @ W2

        def per_head_trace(q, k, v, beta, nu, alpha, W1_init, W2_init):
            def step(carry, inputs):
                W1, W2, mW1, mW2 = carry
                q_t, k_t, v_t, beta_t, nu_t, alpha_t = inputs

                o_pre = mlp_apply(W1, W2, q_t)
                pre = k_t @ W1
                h = jax.nn.silu(pre)
                pred = h @ W2
                residual = v_t - pred

                grad_W2 = -jnp.outer(h, residual)
                hidden_err = (W2 @ residual) * silu_grad(pre)
                grad_W1 = -jnp.outer(k_t, hidden_err)

                mW1 = nu_t * mW1 + grad_W1
                mW2 = nu_t * mW2 + grad_W2
                dW1 = -beta_t * mW1
                dW2 = -beta_t * mW2
                W1 = alpha_t * W1 + dW1
                W2 = alpha_t * W2 + dW2

                o_post = mlp_apply(W1, W2, q_t)
                metrics = {
                    "residual_norm": jnp.linalg.norm(residual),
                    "pred_norm": jnp.linalg.norm(pred),
                    "read_delta_norm": jnp.linalg.norm(o_post - o_pre),
                    "o_pre_norm": jnp.linalg.norm(o_pre),
                    "o_post_norm": jnp.linalg.norm(o_post),
                    "W1_norm": jnp.linalg.norm(W1),
                    "W2_norm": jnp.linalg.norm(W2),
                    "mW1_norm": jnp.linalg.norm(mW1),
                    "mW2_norm": jnp.linalg.norm(mW2),
                    "grad_W1_norm": jnp.linalg.norm(grad_W1),
                    "grad_W2_norm": jnp.linalg.norm(grad_W2),
                    "dW1_norm": jnp.linalg.norm(dW1),
                    "dW2_norm": jnp.linalg.norm(dW2),
                }
                return (W1, W2, mW1, mW2), (o_post, metrics)

            mW1_0 = jnp.zeros_like(W1_init)
            mW2_0 = jnp.zeros_like(W2_init)
            carry0 = (W1_init, W2_init, mW1_0, mW2_0)
            _, (out, metrics) = jax.lax.scan(
                step, carry0, (q, k, v, beta, nu, alpha)
            )
            return out, metrics

        out, traces = jax.vmap(per_head_trace)(
            q, k, v, beta, nu, alpha, self.mem_W1, self.mem_W2
        )
        traces["beta"] = beta
        traces["nu"] = nu
        traces["alpha"] = alpha
        out = out.transpose(1, 0, 2).reshape(T, D)
        return out @ self.Wo, traces


def diagnostics(
    model,
    tokens: Array,
    targets: Array,
    mask: Array,
    num_kv_pairs: int,
    vocab_size: int,
) -> dict[str, float]:
    """Summarize lookup behavior and first-block fast-memory dynamics."""

    logits = model(tokens)
    probs = jax.nn.softmax(logits, axis=-1)
    preds = jnp.argmax(logits, axis=-1)
    query_mask = mask.astype(jnp.float32)
    query_count = jnp.maximum(query_mask.sum(), 1.0)

    true_prob = jnp.take_along_axis(probs, targets[:, None], axis=-1)[:, 0]
    prefix_values = tokens[1 : 2 * num_kv_pairs : 2]
    prefix_value_probs = probs[:, prefix_values]
    prefix_value_mass = prefix_value_probs.sum(axis=-1)
    best_prefix_value = prefix_values[jnp.argmax(prefix_value_probs, axis=-1)]

    def qmean(x):
        return (x * query_mask).sum() / query_count

    metrics = {
        "lookup/query_acc": qmean((preds == targets).astype(jnp.float32)),
        "lookup/query_true_prob": qmean(true_prob),
        "lookup/query_prefix_value_mass": qmean(prefix_value_mass),
        "lookup/query_best_prefix_value_acc": qmean(
            (best_prefix_value == targets).astype(jnp.float32)
        ),
        "lookup/query_pred_is_value_half": qmean(
            (preds >= (vocab_size // 2)).astype(jnp.float32)
        ),
        "lookup/query_logit_true": qmean(
            jnp.take_along_axis(logits, targets[:, None], axis=-1)[:, 0]
        ),
        "lookup/query_logit_max": qmean(jnp.max(logits, axis=-1)),
    }

    x = model.tok_emb[tokens]
    first_block = model.blocks[0]
    _, trace = first_block.mixer.trace(first_block.norm_attn(x))
    metrics.update(summarize_trace(trace, query_mask, num_kv_pairs))
    return {k: float(v) for k, v in metrics.items()}


def summarize_trace(trace: dict[str, Array], query_mask: Array, num_kv_pairs: int):
    """Aggregate first-block Titans traces by token role."""

    T = query_mask.shape[0]
    pos = jnp.arange(T)
    prefix_key_mask = ((pos < 2 * num_kv_pairs) & (pos % 2 == 0)).astype(jnp.float32)
    prefix_value_mask = ((pos < 2 * num_kv_pairs) & (pos % 2 == 1)).astype(jnp.float32)
    query_mask = query_mask.astype(jnp.float32)
    noise_mask = 1.0 - jnp.clip(prefix_key_mask + prefix_value_mask + query_mask, 0, 1)

    def mean_all(x):
        return jnp.mean(x)

    def mean_role(x, role_mask):
        denom = jnp.maximum(role_mask.sum() * x.shape[0], 1.0)
        return (x * role_mask[None, :]).sum() / denom

    metrics = {}
    for name in ("beta", "nu", "alpha"):
        x = trace[name]
        metrics[f"memory/{name}/mean"] = mean_all(x)
        metrics[f"memory/{name}/min"] = jnp.min(x)
        metrics[f"memory/{name}/max"] = jnp.max(x)
        metrics[f"memory/{name}/query_mean"] = mean_role(x, query_mask)
        metrics[f"memory/{name}/noise_mean"] = mean_role(x, noise_mask)

    for name in (
        "residual_norm",
        "read_delta_norm",
        "o_pre_norm",
        "o_post_norm",
        "grad_W1_norm",
        "grad_W2_norm",
        "dW1_norm",
        "dW2_norm",
    ):
        x = trace[name]
        metrics[f"memory/{name}/mean"] = mean_all(x)
        metrics[f"memory/{name}/max"] = jnp.max(x)
        metrics[f"memory/{name}/prefix_key_mean"] = mean_role(x, prefix_key_mask)
        metrics[f"memory/{name}/prefix_value_mean"] = mean_role(x, prefix_value_mask)
        metrics[f"memory/{name}/query_mean"] = mean_role(x, query_mask)
        metrics[f"memory/{name}/noise_mean"] = mean_role(x, noise_mask)

    for name in ("W1_norm", "W2_norm", "mW1_norm", "mW2_norm"):
        x = trace[name]
        metrics[f"memory/{name}/final"] = jnp.mean(x[:, -1])
        metrics[f"memory/{name}/max"] = jnp.max(x)

    all_finite = jnp.array(1.0)
    for x in trace.values():
        all_finite = all_finite * jnp.all(jnp.isfinite(x)).astype(jnp.float32)
    metrics["health/trace_all_finite"] = all_finite
    return metrics

