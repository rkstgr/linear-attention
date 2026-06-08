"""Arch registry: per-mixer HPO knobs, build kwargs, FLOP estimator.

A single source of truth for "what hyperparameters does this architecture have
beyond ``(dim, n_heads, n_layers, mlp_mult, lr)``" — so a sweep entrypoint can
pick the right per-arch knobs from a flat config dict, and a recipe driver
knows the equal-search-space-structure shape per arch.

The FLOP estimator is a stub for the 0004 pilot (iso-FLOP is deferred to the
main grids); included here so the interface is in place when iso-FLOP lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class ArchSpec:
    """Per-mixer recipe for sweep/recipe drivers.

    - ``extra_hyperparams`` are the arch-specific HPO knobs (name -> default).
      The sweep entrypoint reads them out of a flat config dict; the recipe
      driver uses them to lay out per-(arch, shape, task) cells.
    - ``flop_estimator`` returns a forward-FLOP estimate per example. Stub for
      now (returns 0); iso-FLOP comes online in a later proposal.
    """

    mixer: str
    extra_hyperparams: Mapping[str, Any] = field(default_factory=dict)
    flop_estimator: Callable[..., int] = lambda **kw: 0


def _zero_flops(**_: Any) -> int:
    return 0


ARCHS: dict[str, ArchSpec] = {
    "transformer": ArchSpec("transformer", flop_estimator=_zero_flops),
    "linear_attention": ArchSpec("linear_attention", flop_estimator=_zero_flops),
    "deltanet": ArchSpec("deltanet", flop_estimator=_zero_flops),
    "gated_deltanet": ArchSpec("gated_deltanet", flop_estimator=_zero_flops),
    "titans": ArchSpec(
        "titans",
        extra_hyperparams={"memory_mult": 4, "max_inner_lr": 5e-3},
        flop_estimator=_zero_flops,
    ),
}


def mixer_build_kwargs(mixer: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Pull a mixer's extra build kwargs out of a flat config dict.

    Uses the arch spec's defaults where the config does not provide a value, so
    a sweep config that omits e.g. ``memory_mult`` still gets a sane Titans
    default rather than an error.
    """
    try:
        spec = ARCHS[mixer]
    except KeyError as exc:
        raise ValueError(
            f"unknown mixer {mixer!r}; choices: {sorted(ARCHS)}"
        ) from exc
    return {name: config.get(name, default) for name, default in spec.extra_hyperparams.items()}
