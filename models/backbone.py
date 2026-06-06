"""Mixer-agnostic decoder LM backbone."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from models.ffn import SwiGLU
from utils import RMSNorm

MixerFactory = Callable[[int, int, Array], eqx.Module]
FFNFactory = Callable[[int, int, Array], eqx.Module]


class Block(eqx.Module):
    """Pre-norm residual block: mixer + MLP."""

    norm_attn: RMSNorm
    mixer: eqx.Module
    norm_mlp: RMSNorm
    mlp: eqx.Module

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_mult: int,
        key,
        mixer_factory: MixerFactory,
        ffn_factory: FFNFactory = SwiGLU,
    ):
        k_attn, k_mlp = jax.random.split(key)
        self.norm_attn = RMSNorm(dim)
        self.mixer = mixer_factory(dim, n_heads, k_attn)
        self.norm_mlp = RMSNorm(dim)
        self.mlp = ffn_factory(dim, mlp_mult * dim, k_mlp)

    def __call__(self, x: Array) -> Array:
        x = x + self.mixer(self.norm_attn(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class LMModel(eqx.Module):
    """Decoder-only LM with a registry-selected sequence mixer."""

    tok_emb: Array
    blocks: list
    final_norm: RMSNorm
    lm_head: Array

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_heads: int,
        n_layers: int,
        mlp_mult: int,
        key,
        mixer_factory: MixerFactory,
        ffn_factory: FFNFactory = SwiGLU,
    ):
        keys = jax.random.split(key, n_layers + 2)
        s = 1.0 / jnp.sqrt(dim)
        self.tok_emb = jax.random.normal(keys[0], (vocab_size, dim)) * s
        self.blocks = [
            Block(
                dim,
                n_heads,
                mlp_mult,
                k,
                mixer_factory=mixer_factory,
                ffn_factory=ffn_factory,
            )
            for k in keys[1:-1]
        ]
        self.final_norm = RMSNorm(dim)
        self.lm_head = jax.random.normal(keys[-1], (dim, vocab_size)) * s

    def __call__(self, tokens: Array) -> Array:
        x = self.tok_emb[tokens]
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return x @ self.lm_head
