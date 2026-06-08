"""Hypothesis: DeltaNet learns MQAR at small scale."""

from experiments.defaults import DEFAULT_TRAIN, MQAR_SMALL, SEEDS
from linattn.config import ModelConfig, RunConfig
from linattn.executor import executor_main
from linattn.runner import default_train

task = MQAR_SMALL

model = ModelConfig(
    mixer="deltanet",
    vocab_size=512,
    dim=64,     # 200k params at L=2, head_dim=16, mlp_mult=4
    n_heads=4,
    n_layers=2,
    mlp_mult=4,
)

train = DEFAULT_TRAIN

steps = [
    default_train(f"mqar-deltanet-L2-200k/s{seed}", RunConfig(model=model, task=task, train=train, seed=seed))
    for seed in SEEDS
]

if __name__ == "__main__":
    executor_main(steps=steps)
