"""Iso-parameter budget matching: solve model width for a target param count.

A budget-matched comparison fixes a parameter target and finds, per architecture
and per shape (``n_layers`` at fixed ``head_dim``), the ``dim`` that lands closest
to it. Param count is read exactly off a built model — no analytic approximation
that could differ per mixer.

Embeddings are untied in this backbone (separate ``tok_emb`` and ``lm_head``), so
both count toward the budget; this is identical across all mixers, so the
"embedding share held constant across archs" fairness condition holds by
construction.

CLI: print the solved dims for the proposal-0004 recipe shapes, e.g.

    uv run python -m linattn.budget --targets 500000 1000000 --layers 1 2 4 \
        --vocab 512 --head-dim 16
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before any jax import

import argparse

import equinox as eqx
import jax

from linattn.models.factory import MIXERS, build_lm_model


def param_count(model) -> int:
    """Total trainable (inexact-array) parameter count of a built model."""
    return int(sum(x.size for x in jax.tree.leaves(model) if eqx.is_inexact_array(x)))


def count_for_shape(
    *,
    mixer: str,
    vocab_size: int,
    dim: int,
    head_dim: int,
    n_layers: int,
    mlp_mult: int,
    key=None,
    ffn: str = "swiglu",
    **mixer_kwargs,
) -> int:
    """Build one model at the given shape and count its parameters exactly."""
    assert dim % head_dim == 0, f"dim {dim} not a multiple of head_dim {head_dim}"
    if key is None:
        key = jax.random.PRNGKey(0)
    model = build_lm_model(
        mixer,
        vocab_size,
        dim,
        dim // head_dim,
        n_layers,
        mlp_mult,
        key,
        ffn=ffn,
        **mixer_kwargs,
    )
    return param_count(model)


def solve_dim_for_params(
    target: int,
    *,
    mixer: str,
    vocab_size: int,
    head_dim: int,
    n_layers: int,
    mlp_mult: int,
    key=None,
    ffn: str = "swiglu",
    max_dim: int = 4096,
    **mixer_kwargs,
) -> dict:
    """Return the ``dim`` (a multiple of ``head_dim``) closest to ``target`` params.

    Param count is monotone increasing in ``dim``, so we scan multiples of
    ``head_dim`` and stop once we pass the target, keeping the closest of the two
    straddling candidates. Returns the chosen dim, n_heads, and exact count.
    """
    best = None  # (abs_err, dim, count)
    dim = head_dim
    while dim <= max_dim:
        n = count_for_shape(
            mixer=mixer,
            vocab_size=vocab_size,
            dim=dim,
            head_dim=head_dim,
            n_layers=n_layers,
            mlp_mult=mlp_mult,
            key=key,
            ffn=ffn,
            **mixer_kwargs,
        )
        cand = (abs(n - target), dim, n)
        if best is None or cand < best:
            best = cand
        if n >= target:  # monotone: closest straddling candidate now known
            break
        dim += head_dim
    _, dim, n = best
    return {
        "mixer": mixer,
        "n_layers": n_layers,
        "head_dim": head_dim,
        "dim": dim,
        "n_heads": dim // head_dim,
        "param_count": n,
        "target": target,
        "rel_err": (n - target) / target,
    }


def _main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", type=int, nargs="+", default=[500_000, 1_000_000])
    p.add_argument("--layers", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--vocab", type=int, default=512)
    p.add_argument("--head-dim", type=int, default=16)
    p.add_argument("--mlp-mult", type=int, default=4)
    p.add_argument("--mixers", nargs="+", default=sorted(MIXERS))
    args = p.parse_args()

    print(
        f"vocab={args.vocab} head_dim={args.head_dim} mlp_mult={args.mlp_mult}\n"
        "| target | mixer | L | dim | heads | params | rel_err |\n"
        "| -----: | ----- | -: | --: | ----: | -----: | ------: |"
    )
    for target in args.targets:
        for mixer in args.mixers:
            for n_layers in args.layers:
                r = solve_dim_for_params(
                    target,
                    mixer=mixer,
                    vocab_size=args.vocab,
                    head_dim=args.head_dim,
                    n_layers=n_layers,
                    mlp_mult=args.mlp_mult,
                )
                print(
                    f"| {target:>6d} | {mixer:<16} | {n_layers} | {r['dim']:>4d} "
                    f"| {r['n_heads']:>5d} | {r['param_count']:>7d} | {r['rel_err']:+.3f} |"
                )


if __name__ == "__main__":
    _main()
