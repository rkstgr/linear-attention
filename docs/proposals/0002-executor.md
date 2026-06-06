# Proposal 0002: light executor design

> Resolves the executor-approach open decision in
> [DIRECTION.md](../../DIRECTION.md). This is the Sprinter design draft for the
> light executor that runs experiment steps after the `Task` interface and config
> split have their own minimal shape.

Path: **Full**, because this shapes experiment provenance and cache validity
even though it should not authorize a new scientific claim.

Status: **Sprinter draft**. Per [METHOD.md](../process/METHOD.md), a separate
Skeptic pass is required before this becomes a settlement.

---

## Goal

Define the smallest self-built executor that replaces per-script cache plumbing
with content-addressed `ExecutorStep` execution, provenance, and config-shaped
experiment launches, without importing redun or Marin.

## Non-goals

- No new task, metric, sweep, or model result.
- No adoption of redun as a dependency.
- No vendoring Marin's executor.
- No distributed runner, cloud object store, GCS mirroring, Fray/Iris, or remote
  resource scheduler.
- No lazy expression DSL or decorated task language.
- No selective config versioning in the first implementation. Hash the full
  normalized config until that becomes painful.
- No implementation of the `Task` protocol itself. The executor must accept the
  split config shape, but task semantics live outside the executor.
- No separate `steps/` package. Follow Marin's convention: reusable experiment
  recipes live in `experiments/defaults.py`.

## Sources reviewed

- redun docs: [overview](https://insitro.github.io/redun/),
  [design](https://insitro.github.io/redun/design.html),
  [tasks](https://insitro.github.io/redun/tasks.html),
  [scheduler](https://insitro.github.io/redun/scheduler.html),
  [executors](https://insitro.github.io/redun/executors.html).
- Marin docs: [executor explanation](https://marin.readthedocs.io/en/latest/explanations/executor/),
  [executor tutorial](https://marin.readthedocs.io/en/latest/tutorials/executor-101/),
  [executor API](https://marin.readthedocs.io/en/latest/references/executor-api/).
- Marin source: [`types.py`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/execution/types.py),
  [`executor.py`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/execution/executor.py),
  [`step_spec.py`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/execution/step_spec.py),
  [`step_runner.py`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/execution/step_runner.py),
  [`executor_step_status.py`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/execution/executor_step_status.py).

## Sprinter proposal

### Read of redun

redun's useful idea is not "use redun"; it is the discipline around task
identity:

- A task is treated as effectively pure: same concrete inputs and same relevant
  code should mean the same result.
- Cache keys include task identity, task version/source hash, and arguments.
- The scheduler evaluates a lazy DAG and can reuse intermediate results while
  still noticing changed downstream task versions.
- Provenance is a call graph, not just a final metric.
- Local/process/cloud executors are pluggable, but the executor machinery brings
  config files, a database, task decorators, value types, and a broader workflow
  language than this repo needs.

Borrow: content-addressed task calls, explicit purity contract, in-flight
dedupe/CSE for identical cells, provenance records, and source-sensitive cache
misses.

Skip: lazy expressions, task decorators, `.redun/redun.ini`, central DB, File
value layer, remote executors, code packaging, scheduler configuration surface,
and recursive graph reduction.

### Read of Marin

Marin's useful idea is a filesystem-first experiment step plus default recipe
builders:

- `ExecutorStep` has a name, function, dataclass config, and optional
  dependencies.
- Configs can refer to previous step outputs; execution resolves those into
  concrete paths.
- A step's output path is derived from its name, versioned config values, and
  dependency versions.
- Each step writes machine-readable info/status next to its output.
- A runner executes steps in topological order and skips already-successful
  outputs.
- Experiment scripts usually call default helpers such as `default_train(...)`
  rather than writing the training recipe inline.
- The production source includes a lot we do not want yet: distributed leases,
  heartbeats, GCS region inference, mirroring, Fray/Iris resources, pseudo-deps,
  and remote environments.

Borrow: named `ExecutorStep`, dataclass configs, dependency-aware digest,
deterministic output directory, status/metadata files, topological local execution,
`this_output_path()` / `output_path_of(...)` placeholders, a small
`executor_main(...)` entrypoint, and `experiments/defaults.py` as the home for
standard recipes like `default_train(...)`.

Skip: selective `VersionedValue`, pseudo-dependencies, distributed locks, remote
resources, mirroring, regional placement, and broad pipeline materialization.

### Our current pain

`cache.py` is good because it is inspectable. It is bad in two precise ways:

- Source lists are manual in every experiment script. A forgotten file can
  produce a stale cache hit.
- The cache stores values but not an experiment-level manifest: no step graph,
  source digest list, hit/fresh ledger, git commit, command, or normalized
  config record.

The executor should solve those and little else.

### Proposed API

Add `executor.py` with four small concepts:

```python
@dataclass(frozen=True)
class SourceSet:
    name: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class ExecutorStep:
    name: str
    fn: Callable[[Any], None]
    config: Any
    sources: tuple[SourceSet | str, ...] = ()
    deps: tuple["ExecutorStep", ...] = ()
    version: str = "1"
    description: str | None = None


@dataclass(frozen=True)
class OutputPath:
    step: ExecutorStep | None
    name: str | None = None


def this_output_path() -> OutputPath:
    return OutputPath(step=None)


def output_path_of(step: ExecutorStep, name: str | None = None) -> OutputPath:
    return OutputPath(step=step, name=name)
```

`executor_main(steps, *, prefix=".experiment_cache", parallel=1, rerun=False)`:

1. Normalizes each dataclass/dict/list config into stable JSON.
2. Replaces `this_output_path()` with the current step output directory and
   `output_path_of(step)` with the dependency's output directory.
3. Hashes `schema_version`, `step.name`, `step.version`, normalized config,
   dependency digests, and source file bytes.
4. Creates `.experiment_cache/steps/<step-name>-<digest>/`.
5. If `_SUCCESS` plus `status.json` says success and `rerun=False`, returns a
   cache hit.
6. Otherwise calls `fn(resolved_config)`. The step function owns writing its
   artifacts under the resolved output path.
7. Writes `.experiment_cache/runs/<experiment-name>-<timestamp>-<digest>.json`
   with git commit, argv, cwd, full step metadata, source digests, and hit/fresh
   status.

This is deliberately closer to Marin's `ExecutorStep` than to redun's task
decorator. A researcher should be able to understand a launch by reading one
experiment file, `experiments/defaults.py`, and `executor.py`.

### Source sets

Centralize the source lists that are currently duplicated:

```python
CORE_SOURCES = SourceSet(
    "core",
    (
        "configs.py",
        "experiments/defaults.py",
        "tasks/registry.py",
        "utils.py",
        "models/backbone.py",
        "models/registry.py",
        "models/ffn.py",
    ),
)

MIXER_SOURCES = {
    "transformer": SourceSet("mixer:transformer", ("models/attention.py",)),
    "linear_attention": SourceSet("mixer:linear_attention", ("models/linear_attention.py",)),
    "deltanet": SourceSet("mixer:deltanet", ("models/deltanet.py",)),
    "gated_deltanet": SourceSet("mixer:gated_deltanet", ("models/deltanet.py",)),
    "titans": SourceSet("mixer:titans", ("models/titans.py",)),
}
```

Task-specific files are not part of `CORE_SOURCES`; they come from
`task_sources(run.task)`. That is the bridge: tasks own data semantics; the
executor owns cache/provenance mechanics.

### Default train recipe

Training should be exposed to experiments as a default step builder, not as an
inline loop in every experiment. Add `experiments/defaults.py`:

```python
@dataclass(frozen=True)
class ModelConfig:
    mixer: str
    vocab_size: int
    dim: int
    n_heads: int
    n_layers: int
    mlp_mult: int
    mixer_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    eval_batch_size: int
    max_epochs: int
    learning_rate: float
    target_metric: str
    target_value: float
    patience_epochs: int


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    task: TaskConfig
    train: TrainConfig
    seed: int


@dataclass(frozen=True)
class TrainRunConfig:
    run: RunConfig
    output_path: str


def default_train(name: str, run: RunConfig) -> ExecutorStep:
    return ExecutorStep(
        name=f"checkpoints/{name}",
        fn=train_run,
        config=TrainRunConfig(run=run, output_path=this_output_path()),
        sources=(CORE_SOURCES, task_sources(run.task), mixer_sources(run.model)),
    )
```

`train_run(...)` is the shared training recipe: build the task from
`run.task`, build the model from `run.model`, run the optimization loop, evaluate
the task metric, and write `metrics.json` / optional artifacts into
`output_path`. There is no public `train_and_eval(...)` requirement in this
proposal; helper functions can stay private inside `experiments/defaults.py`
until reuse pressure appears.

The executor remains config-agnostic. It does not know MQAR, addition, model
registries, W&B, or training semantics. It only resolves output paths, hashes
steps, runs functions, and records provenance.

An experiment that trains Titans on addition should look like:

```python
from executor import executor_main
from experiments.defaults import default_train
from configs import ModelConfig, RunConfig, TrainConfig
from tasks.addition import AdditionConfig

task = AdditionConfig(n_digits=2, n_numbers=2, train_size=50_000, test_size=2_000)
model = ModelConfig(
    mixer="titans",
    vocab_size=task.vocab_size,
    dim=64,
    n_heads=4,
    n_layers=2,
    mlp_mult=4,
    mixer_kwargs={"memory_mult": 2, "max_inner_lr": 0.0125},
)
train = TrainConfig(
    batch_size=64,
    eval_batch_size=128,
    max_epochs=32,
    learning_rate=3e-4,
    target_metric="exact_match",
    target_value=0.99,
    patience_epochs=5,
)

step = default_train(
    "addition/titans/2digit-2number",
    RunConfig(model=model, task=task, train=train, seed=1),
)

if __name__ == "__main__":
    executor_main([step])
```

### First port

Port only `experiment_capacity.py` first.

Before:

- Build a list of worker specs.
- Each worker hand-builds a cache key.
- Each worker hand-builds source paths.
- `ProcessPoolExecutor` runs cells.

After:

- `experiment_capacity.py` declares four `ExecutorStep`s, one per cell.
- Each `ExecutorStep.config` names the mixer, architecture, data config,
  learning rate, seed, and cell label.
- Each `ExecutorStep.sources` is `(CORE_SOURCES, task_sources(task),
  mixer_sources(model))`.
- `executor_main(steps, parallel=args.parallel, rerun=args.rerun)` handles cache,
  process pool, manifest, and summary statuses.
- The step function stays top-level and picklable so JAX still runs in spawned
  worker processes.

Do not port `experiment_retention.py` in the same PR unless the capacity port
shows the API is wrong. One experiment is enough to validate the executor shape.

### Cache behavior

Use a local-only status file, not a distributed lease:

```text
.experiment_cache/
  steps/
    capacity-linear_attention-nkv4-<digest>/
      metrics.json
      artifacts/
      metadata.json
      status.json
      _SUCCESS
      lock
  runs/
    capacity-20260606T120000-<digest>.json
```

`status.json` records `RUNNING`, `SUCCESS`, or `FAILED`, timestamps, exception
summary, git commit, source digests, normalized config, dependency digests, and
executor schema version.

The lock can be a simple local atomic file create (`O_CREAT | O_EXCL`) with PID
and timestamp. If a second process hits the same step while it is running, it
waits and then reads the result. No stale-lock recovery beyond a clear error in
the first version.

### Parallelism

Keep the runner local:

- `parallel=1` runs serially for debugging.
- `parallel>1` uses `ProcessPoolExecutor` with `spawn`, matching the current
  JAX-friendly scripts.
- Steps with unmet deps wait; independent ready steps can run concurrently.
- Identical digests in the same run are deduped before launch.

No threads for JAX training cells in the first version. Threads are fine for
cache metadata, but the execution unit should remain a process.

### Why not redun or Marin directly?

redun is a real workflow engine. That is the wrong center of gravity for a repo
whose experiments are still small, local, and meant to be read by changing a
couple files.

Marin's executor has the right shape, but its real implementation is optimized
for shared foundation-model pipelines: path placeholders, remote resources,
region pinning, status leases, and cloud mirroring. Copying that code would make
the first task/config split harder to inspect.

The light layer should be roughly "current `cache.py` plus named steps, output
directories, centralized source sets, topological execution, default recipe
builders, and manifests."

## Skeptic review

Pending separate Skeptic review.

The review should especially attack:

- whether automatic `fn` source inclusion plus explicit source sets is enough to
  prevent stale cache hits;
- whether hashing the full normalized config causes noisy misses, and whether
  that is acceptable at this scale;
- whether process-pool execution can preserve current JAX behavior and stdout
  ergonomics;
- whether filesystem artifacts (`metrics.json`, optional `artifacts/`) are
  enough and avoid opaque long-lived `result.pkl` records;
- whether local locks are sufficient if two scripts share a cell;
- whether the executor implementation should wait until the task/config split has
  landed, while this proposal only settles the executor shape.

## Final settlement

Pending Skeptic review.

Sprinter's proposed settlement is: implement a local, self-built executor by
growing `cache.py` into `executor.py`; borrow Marin's named-step/output-dir
shape, `experiments/defaults.py` recipe builders, and redun's source-sensitive
task identity; skip both projects' general-purpose workflow machinery; port only
the capacity experiment first.

## Budget

~0 compute for the executor PR.

Validation should use unit tests with dummy steps and a tiny no-training smoke
step. If the capacity port is run for parity, it should reuse existing seeds and
configs; no new mechanism result is authorized.

## Files expected to change

- `executor.py` or a grown `cache.py`
- `experiments/defaults.py`
- `configs.py` if the split config classes do not already exist
- `experiments/capacity.py` or `experiment_capacity.py`
- `tests/test_executor.py`
- `README.md` if launch commands or cache docs change

Optional only if the implementation makes it obvious:

- `experiment_retention.py`

## Validation

- Unit: identical `ExecutorStep` config/source/deps produces the same digest.
- Unit: changing one config field changes the digest.
- Unit: changing one source byte changes the digest.
- Unit: dependency digest changes flow into downstream step digest.
- Unit: no-op rerun is a cache hit and does not call the step function.
- Unit: `rerun=True` recomputes and overwrites the result.
- Unit: duplicate identical steps in one experiment execute once.
- Unit: `this_output_path()` resolves to the current step output directory.
- Unit: `output_path_of(upstream)` resolves to the upstream step output
  directory and makes the downstream digest depend on the upstream digest.
- Smoke: the capacity entrypoint still prints the same four capacity cells and
  summary shape.
- Smoke: a second no-op run reports cache hits.
- Manual: inspect the run manifest and confirm it includes git commit, argv,
  normalized config, source digests, step status, and hit/fresh state.

## Claim (pre-registered)

Infrastructure only.

For a fixed step function, source set, dependency graph, config, seed, and git
checkout, the executor derives a stable content address and reuses the cached
step output. Changing any hashed config value, dependency digest, or source file
byte causes a cache miss. Porting the capacity experiment through the executor
does not change its cell configs, seeds, model construction, training recipe, or
summary format.

No accuracy, capacity, retention, scaling, or task-generalization claim is made.
There is no metric/null/regime because this proposal does not measure model
behavior.

## Follow-ups

- The task/config split should introduce `Task`, `ModelConfig`, `TaskConfig`,
  `TrainConfig`, and `RunConfig` without depending on the executor
  implementation.
- Later port `experiment_retention.py` only after the capacity port validates the
  API.
- Add W&B sweep integration later, after task/config split makes `arch x task`
  cells concrete.
- Consider selective versioning only if full-config hashing causes real noisy
  recomputation.
- Consider richer file/path outputs only when a multi-step data pipeline exists.
  Until then, each train step writes `metrics.json` plus optional artifacts under
  its output directory.
