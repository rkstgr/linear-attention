"""Shared presets for experiment files."""

from linattn.config import TrainConfig
from linattn.tasks.mqar import MQARConfig

# ── Seeds ──────────────────────────────────────────────────────────────────────

SEEDS = (1, 2, 3)

# ── Task presets ───────────────────────────────────────────────────────────────

MQAR_SMALL = MQARConfig(
    vocab_size=512, input_seq_len=64, num_kv_pairs=4, power_a=0.01,
    n_train=10_000, n_val=1_000, n_test=1_000,
)

# ── Training presets ───────────────────────────────────────────────────────────

DEFAULT_TRAIN = TrainConfig(
    batch_size=256, eval_batch_size=256, max_epochs=20,
    learning_rate=1e-3, target_acc=1.01, patience_epochs=5,
)
