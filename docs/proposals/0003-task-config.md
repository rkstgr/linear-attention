# Proposal 0003: task abstraction, central config, and `fit` decomposition

Status: proposed. Branch: `claude/refactoring-state-C3UO3`.

## Goal

Finish the foundation items that block the cross-architecture W&B comparison:
a `Task` abstraction, a central config module, and a training entry point that
does one job each. Collapse the two parallel config representations into one,
move the code into a real source package, and keep the executor as the only
orchestration layer.

## Claim

No research claim.

Infrastructure claim: after this change there is a single config shape
(`RunConfig = model + task + train + seed`), tasks are selected through a
registry exactly like mixers, and `fit` composes small, independently testable
pieces (`train_step`, `evaluate`, `EarlyStopping`, `Reporter`, `TrainResult`).
Model construction stays byte-for-byte identical (parity digests unchanged) and
MQAR data generation is a pure relocation (identical bytes for a fixed key and
config).

## Decisions (settled with owner)

- **Scope:** full migration. Delete `data.Config` and the
  `config_from_data_config` / `data_config` bridges; migrate every entrypoint.
- **Package:** move core modules into a `linattn/` source package now (one
  import-rewrite pass, since the migration already rewrites imports broadly).
- **vocab_size:** kept on both `ModelConfig` and the task config; callers keep
  them consistent (preserves current behavior).
- **Task dispatch:** discriminator field — every `TaskConfig` carries `name`;
  `build_task(cfg)` dispatches on it.
- **n_train / n_test:** live on the task config (data sizing).
- **Presets vs experiments:** `toy` and `easy` are debug/smoke task presets in
  `linattn/tasks/mqar.py`. `level1` is a full experiment, not a preset, and
  lives as a complete `RunConfig` in the experiments layer alongside
  `capacity` and `retention`.
- **`train_and_eval` -> `fit`:** full decomposition with a `Reporter` protocol
  (Stdout / Wandb / Json).
- **Registry rename:** `models/registry.py` -> `linattn/models/factory.py`
  (the builder `build_lm_model` is the headline export; "registry" oversells two
  static dicts).

## Target structure

```
linattn/
  __init__.py
  config.py          # ModelConfig, TrainConfig, RunConfig, DEFAULT_TRAIN
  executor.py        # moved from root, unchanged
  train.py           # fit + train_step + StepStats + EarlyStopping
                     #   + Reporter protocol + TrainResult + evaluate
  runner.py          # default_train (ExecutorStep builder) + train_run entry
  utils.py           # moved from root
  cache.py           # moved from root (legacy; retention still uses it)
  models/            # moved from root; registry.py -> factory.py
    __init__.py backbone.py factory.py attention.py linear_attention.py
    deltanet.py titans.py transformer.py ffn.py
  tasks/
    __init__.py
    base.py          # Split, TaskConfig, Task, TASKS, build_task
    mqar.py          # MQARConfig, MQARTask, toy/easy presets, MQAR generation

experiments/         # capacity.py, retention.py, titans.py, run_model.py,
                     #   char_transformer.py, sweep_titans_toy.py
                     #   (defaults.py is absorbed into linattn/runner.py)
tests/  sweeps/  scripts/  bench_*.py  docs/   # stay outside the package
```

`data.py` is deleted; its MQAR generation and docstring move to
`linattn/tasks/mqar.py`. No `[build-system]` is required — the repo runs flat
via `uv run` with the root on `sys.path`, so `import linattn` resolves the same
way the current flat modules do.

## Interfaces

### `linattn/config.py`

```python
@dataclass(frozen=True)
class ModelConfig:
    mixer: str
    vocab_size: int
    dim: int; n_heads: int; n_layers: int; mlp_mult: int
    ffn: str = "swiglu"
    mixer_kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class TrainConfig:
    batch_size: int; eval_batch_size: int
    max_epochs: int; learning_rate: float
    target_acc: float; patience_epochs: int

@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    task: TaskConfig          # discriminated by task.name
    train: TrainConfig
    seed: int

DEFAULT_TRAIN = TrainConfig(...)   # debug-scale default
```

### `linattn/tasks/base.py`

```python
Split = tuple[Array, Array, Array]   # tokens, targets, mask; each (n, T)

class TaskConfig(Protocol):
    name: str

class Task(Protocol):
    vocab_size: int
    def make_split(self, key, n) -> Split: ...
    def describe(self, model, key) -> None: ...   # replaces inspect_example
    sources: tuple[str, ...]                       # source files for the digest

TASKS: dict[str, Callable[[TaskConfig], Task]] = {"mqar": MQARTask}

def build_task(cfg: TaskConfig) -> Task:           # dispatch on cfg.name
    ...
```

### `linattn/tasks/mqar.py`

`MQARConfig(name="mqar", vocab_size, input_seq_len, num_kv_pairs, power_a,
n_train, n_test)`, `MQARTask` (owns the relocated `mqar_example` / `mqar_batch`
/ `make_split`), and the `toy` / `easy` presets. `MQARTask.sources` declares
`("linattn/tasks/mqar.py",)`, replacing `mqar_sources()`.

### `linattn/train.py` — `fit`, decomposed

```python
class StepStats(NamedTuple):
    loss: Array; acc: Array
    grad_norm: Array; update_norm: Array; param_norm: Array
    all_finite: Array

@eqx.filter_jit
def train_step(model, opt_state, batch, opt) -> tuple[Model, OptState, StepStats]: ...

def evaluate(model, data, batch_size) -> float: ...   # kept as-is

@dataclass
class EarlyStopping:
    target_acc: float; patience_epochs: int
    def update(self, test_acc: float) -> str | None: ...   # stop reason or None

class Reporter(Protocol):
    def on_step(self, stats, *, epoch, step, n_batches, ms): ...
    def on_epoch(self, *, epoch, train_loss, train_acc, test_acc): ...
    def on_nonfinite(self, stats, *, epoch, step): ...
# StdoutReporter (default), WandbReporter, JsonReporter, MultiReporter

@dataclass
class TrainResult:
    model: Model
    history: list[dict]
    stop_info: dict        # stop_reason + non-finite diagnostics

def fit(model, task: Task, train: TrainConfig, key, *,
        opt=None, reporter: Reporter = StdoutReporter()) -> TrainResult: ...
```

`fit` reads `n_train` / `n_test` from the task config, builds train/test splits
via `task.make_split`, and threads model/opt_state through `train_step`. The
non-finite fail-fast + rollback lives in the loop, driven by `stats.all_finite`,
and is surfaced through `reporter.on_nonfinite`. The `return_info` boolean flag
is gone — `fit` always returns a `TrainResult`.

## Alignment with the executor (Marin / Levanter framing)

The repo already mirrors Marin's two-layer split, and this change keeps it:

- **Executor = orchestration / provenance / caching** (Marin's executor). It
  decides *which* runs happen and content-addresses their outputs.
- **`fit` + `Reporter` + `Task` = one run** (Levanter's `Trainer` + `tracker`).
  Levanter independently arrived at the same seams — jitted train step, hooks
  for eval/stop, a pluggable `tracker` for wandb/tensorboard/noop. Our
  `Reporter` is that tracker; `fit` is a miniature `Trainer`.
- **`RunConfig` is the plain-data contract** between the layers. The executor
  hashes it; `linattn/runner.py:train_run` reconstructs `build_lm_model` +
  `build_task` in-worker and calls `fit` with a `JsonReporter(this_output_path())`.

**Cache-vs-W&B rule:** the `JsonReporter` artifact (`metrics.json`) is the
cached source of truth. The `WandbReporter` fires only on a fresh run; a cache
hit replays `metrics.json` rather than re-logging, so a green dashboard is never
silently stale.

## Migration plan (smallest inspectable commits)

1. This proposal.
2. Create `linattn/` and move modules in (`executor`, `train`, `utils`, `cache`,
   `models/` with `registry.py` -> `factory.py`); fix imports; suite green.
3. `linattn/config.py`.
4. `linattn/tasks/` (`base.py` + `mqar.py`); relocate MQAR generation with an
   identity check against the old `data.make_split` output.
5. `linattn/train.py` decomposition (`fit` and the five seams).
6. `linattn/runner.py` (`default_train` + `train_run` with `JsonReporter`);
   delete the `defaults.py` bridges.
7. Migrate entrypoints: `capacity`, `titans` (-> `WandbReporter`), `run_model`,
   `sweep_titans_toy`; `level1` becomes a full `RunConfig` experiment.
   `retention` gets the minimal config/task swap to stay green (its executor
   port stays queue item 3).
8. Delete `data.py`.
9. Tests: unit `EarlyStopping`, `build_task`, and a `make_split` identity test;
   run the full suite.
10. Update `AGENTS.md` active queue.

## Guardrails

- `tests/test_parity.py` passes with identical digests (model construction is
  behavior-preserving).
- MQAR `make_split` is byte-identical to the pre-move implementation for a fixed
  key and config.
- Every entrypoint still runs at toy scale.
- The `WandbReporter` never logs on a cache hit.

## Validation

```sh
uv run python -m unittest discover -v
```

Plus a toy-scale smoke run of `run_model` and `capacity` to confirm the
`fit` / executor / reporter wiring end to end.

## Follow-ups

- Retention executor port and `cache.py` decision (active queue item 3).
- W&B sweep integration through the executor (active queue item 6 territory).
- A `Checkpointer` reporter/hook (Levanter has one; we do not yet).
