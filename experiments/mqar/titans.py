"""Hypothesis: Titans learns MQAR at small scale."""

from experiments.defaults import DEFAULT_TRAIN, MQAR_SMALL, SEEDS
from linattn.config import ModelConfig, RunConfig
from linattn.executor import executor_main
from linattn.runner import default_train

task = MQAR_SMALL

model = ModelConfig(
    mixer="titans",
    vocab_size=512,
    dim=64,     # 216k params at L=2, head_dim=16, mlp_mult=4
    n_heads=4,
    n_layers=2,
    mlp_mult=4,
    mixer_kwargs={"memory_mult": 4, "max_inner_lr": 5e-3},
)

train = DEFAULT_TRAIN

steps = [
    default_train(f"mqar-titans-L2-200k/s{seed}", RunConfig(model=model, task=task, train=train, seed=seed))
    for seed in SEEDS
]

if __name__ == "__main__":
    executor_main(steps=steps)
