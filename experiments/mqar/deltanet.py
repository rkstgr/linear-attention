"""Hypothesis: DeltaNet learns MQAR at small scale."""

from experiments.defaults import SEEDS
from linattn.config import ModelConfig, RunConfig, TrainConfig
from linattn.executor import executor_main
from linattn.runner import default_train
from linattn.tasks.mqar import MQARConfig

task = MQARConfig(
    vocab_size=512, input_seq_len=64, num_kv_pairs=4, power_a=0.01,
    n_train=20_000, n_val=1_000, n_test=1_000,
)

model = ModelConfig(
    mixer="deltanet",
    vocab_size=512,
    dim=64,     # 200k params at L=2, head_dim=16, mlp_mult=4
    n_heads=2,
    n_layers=2,
    mlp_mult=4,
)

train = TrainConfig(
    batch_size=128, eval_batch_size=128, max_epochs=20,
    learning_rate=0.0045, target_acc=1.01, patience_epochs=3,
)

steps = [
    default_train(f"mqar-deltanet-L2/s{seed}", RunConfig(model=model, task=task, train=train, seed=seed))
    for seed in [1]
]

if __name__ == "__main__":
    executor_main(steps=steps)
