"""Registries for mixers and feed-forward modules."""

from functools import partial

from models.attention import Attention
from models.backbone import LMModel
from models.deltanet import DeltaNet
from models.ffn import SwiGLU
from models.linear_attention import LinearAttention
from models.titans import Titans

MIXERS = {
    "transformer": Attention,
    "linear_attention": LinearAttention,
    "deltanet": partial(DeltaNet, gated=False),
    "gated_deltanet": partial(DeltaNet, gated=True),
    "titans": Titans,
}

FFNS = {
    "swiglu": SwiGLU,
}


def build_lm_model(
    mixer: str,
    vocab_size: int,
    dim: int,
    n_heads: int,
    n_layers: int,
    mlp_mult: int,
    key,
    *,
    ffn: str = "swiglu",
    **mixer_kwargs,
) -> LMModel:
    """Construct an LMModel from registry names."""

    try:
        mixer_base = MIXERS[mixer]
    except KeyError as exc:
        raise ValueError(f"unknown mixer {mixer!r}; choices: {sorted(MIXERS)}") from exc
    try:
        ffn_factory = FFNS[ffn]
    except KeyError as exc:
        raise ValueError(f"unknown FFN {ffn!r}; choices: {sorted(FFNS)}") from exc

    def mixer_factory(dim, n_heads, key):
        return mixer_base(dim, n_heads, key, **mixer_kwargs)

    return LMModel(
        vocab_size=vocab_size,
        dim=dim,
        n_heads=n_heads,
        n_layers=n_layers,
        mlp_mult=mlp_mult,
        key=key,
        mixer_factory=mixer_factory,
        ffn_factory=ffn_factory,
    )
