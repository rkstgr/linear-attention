"""Mixer-backed language models."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

from linattn.models.backbone import LMModel
from linattn.models.factory import FFNS, MIXERS, build_lm_model

__all__ = ["FFNS", "LMModel", "MIXERS", "build_lm_model"]
