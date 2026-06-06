"""Mixer-backed language models."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

from models.backbone import LMModel
from models.registry import FFNS, MIXERS, build_lm_model

__all__ = ["FFNS", "LMModel", "MIXERS", "build_lm_model"]
