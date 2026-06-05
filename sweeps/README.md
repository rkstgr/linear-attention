# Titans W&B Sweeps

The sweep YAML files are hardware-agnostic. Pick CPU vs CUDA at the agent
launcher layer, then every sampled run uses the same project environment via
`uv run --no-sync`.

Create a sweep:

```sh
uv run --group experiment wandb sweep sweeps/titans_level1.yaml
```

Run a CPU agent:

```sh
scripts/run_wandb_agent_cpu.sh <entity/project/sweep_id> --count 30
```

Run a CUDA agent on a Linux x86_64 GPU VM:

```sh
scripts/run_wandb_agent_cuda.sh <entity/project/sweep_id> --count 30
```

CUDA agents export `JAX_PLATFORMS=cuda,cpu`, disable XLA memory preallocation
by default, and set `REQUIRE_JAX_GPU=1`. If JAX falls back to CPU, the run exits
before training. W&B records `jax_backend`, `jax_devices`,
`runtime/jax_has_gpu`, and FLOP estimates for comparability.
