# linear-attention

Small research workbench for comparing sequence-mixing mechanisms:

- softmax attention;
- linear attention;
- DeltaNet and gated DeltaNet;
- Titans.

The compact operating guide and active queue are in `AGENTS.md`.

Current mode: foundation cleanup before comparable W&B sweeps for the
non-Titans architectures.

## Setup

```sh
uv sync
uv sync --group experiment
uv sync --group cuda --group experiment
```

Use the CUDA group only on Linux x86_64 NVIDIA GPU machines.

## Tests

```sh
uv run python -m unittest discover -v
```

Current tests cover model parity and the local executor.

## Core Experiments

```sh
uv run python -m experiments.capacity
uv run python -m experiments.retention
```

`experiments.capacity` uses the local executor in `executor.py`.
`experiments.retention` still uses the legacy `cache.py` path and is scheduled
for cleanup.

Executor outputs live under:

```text
.experiment_cache/steps/
.experiment_cache/runs/
```

Useful flags:

```sh
uv run python -m experiments.capacity --rerun
uv run python -m experiments.capacity --parallel 4
```

## Run One Model

```sh
uv run python -m experiments.run_model transformer
uv run python -m experiments.run_model linear_attention
uv run python -m experiments.run_model deltanet
uv run python -m experiments.run_model gated_deltanet
uv run python -m experiments.titans
```

## W&B Sweeps

Recent context: Titans toy/easy/level1 sweeps exist in W&B. The next research
push is to run comparable sweeps for the other architectures after config/task
and provenance cleanup.

```sh
uv run --group experiment wandb sweep sweeps/titans_level1.yaml
scripts/run_wandb_agent_cpu.sh <entity/project/sweep_id> --count 30
scripts/run_wandb_agent_cuda.sh <entity/project/sweep_id> --count 30
```

See `sweeps/README.md` for metric groups and launch details.

## Benchmarks

```sh
uv run python bench_chunkwise.py --device cpu
uv run python bench_chunkwise.py --device cpu --c-sweep
uv run python bench_chunkwise_plot.py
```

Existing plots are under `figures/`.

## Layout

- `models/` - shared LM scaffold, registries, FFN, and mixers.
- `data.py` - MQAR generator and current MQAR configs.
- `train.py` - shared MQAR training loop.
- `executor.py` - local content-addressed step executor.
- `experiments/defaults.py` - default executor training recipe and partial
  config dataclasses.
- `experiments/capacity.py` - executor-backed capacity toy experiment.
- `experiments/retention.py` - legacy-cache retention toy experiment.
- `sweeps/` - W&B sweep configs and docs.
- `tests/` - parity and executor tests.
- `AGENTS.md` - operating guide and active queue.
- `docs/experiments/` - result ledger.

## Sources

- [HazyResearch/zoology](https://github.com/HazyResearch/zoology)
- [Zoology paper](https://arxiv.org/html/2312.04927v1)
- [Zoology blogpost](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis)
