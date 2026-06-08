"""Proposal 0004 pilot: depth/width recipe + cross-task tradeoff.

Three stages, all executor-backed (cells cached by RunConfig + source bytes,
re-runnable via ``--rerun``):

  1. ``lr_selection`` — 1 seed per (arch, shape, task, lr); pick best lr per
     (arch, shape, task) on val.
  2. ``recipe`` — 3 seeds at the per-(arch, shape, task) best lr; pick the
     depth/width recipe per task (Pareto-best averaged across archs) and read
     the cross-task tradeoff gate.
  3. ``stability`` — chosen recipe only at the 1M budget, 3 seeds.

Iso-parameter is the pilot's only budget control (iso-FLOP is deferred). Dims
are solved exactly per (arch, shape, task) via ``budget.solve_dim_for_params``
so the same param target lands the same regardless of mixer or vocab.

Usage:
    uv run python -m experiments.recipe --stage lr_selection
    uv run python -m experiments.recipe --stage recipe
    uv run python -m experiments.recipe --stage stability
    uv run python -m experiments.recipe --stage all  # default
    uv run python -m experiments.recipe --smoke      # tiny sizes for a smoke run

Mirror runs into W&B (one run per cell, in addition to the local cache):
    uv run --group experiment python -m experiments.recipe --stage all \\
        --wandb --wandb-project linear-attention

Each stage prints a markdown summary; ``recipe`` and ``stability`` also emit a
heatmap-ready JSON under ``.experiment_cache/runs/recipe-<stage>-<digest>.json``.
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must run before `import jax`

import argparse
import dataclasses
import json
import statistics
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from linattn.archs import ARCHS
from linattn.budget import solve_dim_for_params
from linattn.config import ModelConfig, RunConfig, TrainConfig
from linattn.executor import executor_main
from linattn.runner import default_train
from linattn.tasks.addition import AdditionConfig
from linattn.tasks.mqar import MQARConfig

RECIPE_ARCHS = ("transformer", "linear_attention", "deltanet", "gated_deltanet", "titans")
HEAD_DIM = 16
MLP_MULT = 4
SHAPES_N_LAYERS = (1, 2, 4)
TARGET_PARAMS_PILOT = 500_000
TARGET_PARAMS_STABILITY = 1_000_000
LR_GRID = (3e-4, 1e-3, 3e-3)
SEEDS = (1, 2, 3)

# Pilot cells (proposal 0004): both tasks at a single solvable cell.
MQAR_CELL = MQARConfig(
    vocab_size=512, input_seq_len=64, num_kv_pairs=4, power_a=0.01,
    n_train=50_000, n_val=2_000, n_test=2_000,
)
ADDITION_CELL = AdditionConfig(
    max_digits=3, num_addends=2,
    n_train=50_000, n_val=2_000, n_test=2_000,
)
TASK_CELLS: dict[str, object] = {"mqar": MQAR_CELL, "addition": ADDITION_CELL}

# Training defaults shared across cells. target_acc set above 1 disables the
# target stop; the pilot selects on val, so patience is the only stop signal
# besides max_epochs.
TRAIN_DEFAULTS = dict(
    batch_size=64,
    eval_batch_size=64,
    max_epochs=20,
    target_acc=1.01,
    patience_epochs=5,
)

SMOKE_OVERRIDES = dict(
    n_train=512, n_val=64, n_test=64,
    target_params=20_000,
    train=dict(batch_size=32, eval_batch_size=32, max_epochs=2, patience_epochs=2),
)


@dataclass(frozen=True)
class Cell:
    """One executor cell: arch x shape x task x lr x seed at a budget."""

    mixer: str
    n_layers: int
    task: str
    lr: float
    seed: int
    target_params: int


@lru_cache(maxsize=None)
def _solve_dim_cached(mixer: str, n_layers: int, vocab_size: int, target: int) -> int:
    """Cache the (small) exact-build solve so repeated cell construction is cheap."""
    out = solve_dim_for_params(
        target,
        mixer=mixer,
        vocab_size=vocab_size,
        head_dim=HEAD_DIM,
        n_layers=n_layers,
        mlp_mult=MLP_MULT,
    )
    return int(out["dim"])


def _scaled_task_cell(task_name: str, *, n_train: int, n_val: int, n_test: int):
    """Override the (n_train, n_val, n_test) of a base task config."""
    base = TASK_CELLS[task_name]
    return replace(base, n_train=n_train, n_val=n_val, n_test=n_test)


def _cell_step(cell: Cell, *, task_cfg, train_cfg):
    dim = _solve_dim_cached(
        cell.mixer, cell.n_layers, task_cfg.vocab_size, cell.target_params
    )
    extra = dict(ARCHS[cell.mixer].extra_hyperparams)
    model_cfg = ModelConfig(
        mixer=cell.mixer,
        vocab_size=task_cfg.vocab_size,
        dim=dim,
        n_heads=dim // HEAD_DIM,
        n_layers=cell.n_layers,
        mlp_mult=MLP_MULT,
        mixer_kwargs=extra,
    )
    train_cfg = replace(train_cfg, learning_rate=cell.lr)
    run = RunConfig(model=model_cfg, task=task_cfg, train=train_cfg, seed=cell.seed)
    # Step name encodes the cell so cached lookups stay legible on disk.
    name = (
        f"recipe/{cell.task}/{cell.mixer}/L{cell.n_layers}/"
        f"p{cell.target_params}/lr{cell.lr:.0e}/s{cell.seed}"
    )
    return default_train(name, run)


def _build_steps(cells: Iterable[Cell], *, smoke: bool):
    """Build executor steps for a list of cells, sharing task/train resolution."""
    task_cfgs = {}
    for cell in cells:
        if cell.task not in task_cfgs:
            if smoke:
                task_cfgs[cell.task] = _scaled_task_cell(
                    cell.task, n_train=SMOKE_OVERRIDES["n_train"],
                    n_val=SMOKE_OVERRIDES["n_val"], n_test=SMOKE_OVERRIDES["n_test"],
                )
            else:
                task_cfgs[cell.task] = TASK_CELLS[cell.task]
    train_defaults = dict(TRAIN_DEFAULTS)
    if smoke:
        train_defaults.update(SMOKE_OVERRIDES["train"])
    train_cfg = TrainConfig(**{**train_defaults, "learning_rate": 1e-3})
    return [
        _cell_step(cell, task_cfg=task_cfgs[cell.task], train_cfg=train_cfg)
        for cell in cells
    ]


def _read_headline(output_path: str) -> dict:
    return json.loads(
        (Path(output_path) / "metrics.json").read_text(encoding="utf-8")
    )["headline"]


def _read_metrics(output_path: str) -> dict:
    return json.loads(
        (Path(output_path) / "metrics.json").read_text(encoding="utf-8")
    )


# --- optional W&B logging --------------------------------------------------
#
# Decoupled from the executor: we read each cell's `metrics.json` and replay
# its history into one W&B run per cell. Cached cells get logged the same way
# fresh ones do (the headline does not change between runs), so re-invoking
# with --wandb after a prior local run mirrors everything into W&B without
# recomputing.

_EPOCH_LOG_KEYS = (
    "train_accuracy", "train_partial_accuracy",
    "val_accuracy", "val_partial_accuracy",
    "test_accuracy", "test_partial_accuracy",
)


def _log_cell_to_wandb(cell, result, *, project, entity, stage, smoke):
    import wandb

    metrics = _read_metrics(result.output_path)
    run_cfg = metrics["run"]
    headline = metrics["headline"]
    name = (
        f"{stage}/{cell.task}/{cell.mixer}/L{cell.n_layers}/"
        f"p{cell.target_params // 1000}k/lr{cell.lr:.0e}/s{cell.seed}"
    )
    config = {
        "stage": stage,
        "smoke": smoke,
        "mixer": cell.mixer,
        "task": cell.task,
        "n_layers": cell.n_layers,
        "lr": cell.lr,
        "seed": cell.seed,
        "target_params": cell.target_params,
        "head_dim": HEAD_DIM,
        "mlp_mult": MLP_MULT,
        "dim": run_cfg["model"]["dim"],
        "n_heads": run_cfg["model"]["n_heads"],
        "n_train": run_cfg["task"]["n_train"],
        "n_val": run_cfg["task"]["n_val"],
        "n_test": run_cfg["task"]["n_test"],
        "cache_status": result.cache_status,
        "step_digest": result.digest,
    }
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=f"recipe-{stage}",
        job_type=stage,
        tags=[stage, cell.mixer, cell.task, f"L{cell.n_layers}",
              f"p{cell.target_params // 1000}k"],
        config=config,
        reinit=True,
    )
    try:
        for h in metrics["history"]:
            payload = {"epoch": h["epoch"]}
            if "train_loss" in h:
                payload["learning/epoch_train_loss"] = h["train_loss"]
            for k in _EPOCH_LOG_KEYS:
                if k in h:
                    payload[f"learning/{k}"] = h[k]
            run.log(payload, step=h["epoch"])
        # Headline (test-at-best-val) and the sweep-style objective scalar so
        # W&B comparisons pick the same metric the local stages do.
        run.summary["headline/test_accuracy"] = headline["test_accuracy"]
        run.summary["headline/test_partial_accuracy"] = headline["test_partial_accuracy"]
        run.summary["headline/selection"] = headline["selection"]
        run.summary["headline/selection_value"] = headline["selection_value"]
        run.summary["headline/epoch"] = headline["epoch"]
        run.summary["objective/score"] = headline["test_partial_accuracy"]
        run.summary["health/stop_reason"] = metrics["stop_info"]["stop_reason"]
    finally:
        run.finish()


def _make_wandb_callback(args, cells: list, steps: list, stage: str):
    """Return an executor `on_step_complete` that logs each cell to W&B inline.

    Cells finish in any order under ``--parallel``; the callback uses a
    step-name -> cell map built from the actual ``ExecutorStep`` objects so it
    tracks the real step naming even if that format ever changes.
    """
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "wandb is not installed. Run with: "
            "uv run --group experiment python -m experiments.recipe ..."
        ) from exc
    name_to_cell = {step.name: cell for cell, step in zip(cells, steps)}

    def _callback(result):
        cell = name_to_cell.get(result.name)
        if cell is None:
            return  # not one of ours (shouldn't happen, but be defensive)
        _log_cell_to_wandb(
            cell, result,
            project=args.wandb_project,
            entity=args.wandb_entity,
            stage=stage,
            smoke=args.smoke,
        )

    return _callback


def _print_table(title: str, header: list[str], rows: list[list]) -> None:
    print(f"\n\n## {title}\n")
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join("---:" for _ in header) + " |")
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")


# --- stage 1: lr selection ------------------------------------------------

def _pilot_target(args) -> int:
    return SMOKE_OVERRIDES["target_params"] if args.smoke else TARGET_PARAMS_PILOT


def _stability_target(args) -> int:
    return SMOKE_OVERRIDES["target_params"] if args.smoke else TARGET_PARAMS_STABILITY


def _lr_selection_cells(target: int) -> list[Cell]:
    return [
        Cell(arch, n_l, t, lr, seed=1, target_params=target)
        for arch in RECIPE_ARCHS
        for n_l in SHAPES_N_LAYERS
        for t in TASK_CELLS
        for lr in LR_GRID
    ]


def _best_lr_per_axis(cells: list[Cell], headlines: list[dict]) -> dict:
    """Pick the lr that maximizes the selection metric per (arch, n_layers, task)."""
    best: dict[tuple[str, int, str], tuple[float, float]] = {}
    for cell, h in zip(cells, headlines):
        key = (cell.mixer, cell.n_layers, cell.task)
        score = float(h.get("selection_value", 0.0))
        prev = best.get(key)
        if prev is None or score > prev[0]:
            best[key] = (score, cell.lr)
    return {k: lr for k, (_, lr) in best.items()}


def stage_lr_selection(args) -> dict:
    cells = _lr_selection_cells(_pilot_target(args))
    steps = _build_steps(cells, smoke=args.smoke)
    results = executor_main(
        steps, parallel=args.parallel, rerun=args.rerun,
        experiment_name="recipe-lr_selection",
        on_step_complete=_make_wandb_callback(args, cells, steps, "lr_selection"),
    )
    headlines = [_read_headline(r.output_path) for r in results]
    rows = [
        [c.task, c.mixer, c.n_layers, f"{c.lr:.0e}",
         f"{h['selection_value']:.3f}", f"{h['test_partial_accuracy']:.3f}",
         f"{h['test_accuracy']:.3f}"]
        for c, h in zip(cells, headlines)
    ]
    rows.sort()
    _print_table(
        "Stage 1 — lr selection (1 seed)",
        ["task", "mixer", "L", "lr", "val_partial", "test_partial", "test_complete"],
        rows,
    )
    best_lr = _best_lr_per_axis(cells, headlines)
    _print_table(
        "Best lr per (arch, shape, task)",
        ["task", "mixer", "L", "best_lr"],
        sorted([[t, m, L, f"{lr:.0e}"] for (m, L, t), lr in best_lr.items()]),
    )
    return {
        "stage": "lr_selection",
        "best_lr": {f"{m}|{L}|{t}": lr for (m, L, t), lr in best_lr.items()},
    }


# --- stage 2: recipe + seeds ----------------------------------------------

def _recipe_cells(best_lr: dict, target: int) -> list[Cell]:
    cells = []
    for arch in RECIPE_ARCHS:
        for n_l in SHAPES_N_LAYERS:
            for t in TASK_CELLS:
                lr = best_lr[(arch, n_l, t)]
                for seed in SEEDS:
                    cells.append(Cell(arch, n_l, t, lr, seed, target))
    return cells


def _aggregate(cells: list[Cell], headlines: list[dict]) -> dict:
    """Group by (task, mixer, n_layers); return per-cell list of test metrics across seeds."""
    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for c, h in zip(cells, headlines):
        grouped.setdefault((c.task, c.mixer, c.n_layers), []).append(h)
    out: dict[tuple[str, str, int], dict] = {}
    for k, hs in grouped.items():
        partials = [h["test_partial_accuracy"] for h in hs]
        completes = [h["test_accuracy"] for h in hs]
        out[k] = {
            "n_seeds": len(hs),
            "partial_mean": statistics.fmean(partials),
            "partial_range": max(partials) - min(partials),
            "complete_mean": statistics.fmean(completes),
            "complete_range": max(completes) - min(completes),
        }
    return out


def _pick_recipe(agg: dict) -> dict[str, int]:
    """Per task, pick the n_layers with the highest mean across archs."""
    pick: dict[str, int] = {}
    for task in TASK_CELLS:
        per_shape = {n_l: [] for n_l in SHAPES_N_LAYERS}
        for (t, m, L), v in agg.items():
            if t == task:
                per_shape[L].append(v["partial_mean"])
        scored = {L: statistics.fmean(v) for L, v in per_shape.items() if v}
        pick[task] = max(scored, key=scored.get)
    return pick


def _tradeoff_gate(agg: dict, recipe: dict[str, int]) -> dict:
    """At the chosen recipe per task, rank archs and report whether the rankings agree."""
    ranking_by_task: dict[str, list[tuple[str, float]]] = {}
    for task, L in recipe.items():
        scores = [
            (m, agg[(task, m, L)]["partial_mean"])
            for m in RECIPE_ARCHS
            if (task, m, L) in agg
        ]
        scores.sort(key=lambda x: -x[1])
        ranking_by_task[task] = scores
    if "mqar" in ranking_by_task and "addition" in ranking_by_task:
        mqar_top = ranking_by_task["mqar"][0][0]
        add_top = ranking_by_task["addition"][0][0]
        diverge = mqar_top != add_top
    else:
        diverge = None
    return {"rankings": ranking_by_task, "diverge": diverge}


def stage_recipe(args, best_lr_lookup: dict | None = None) -> dict:
    if best_lr_lookup is None:
        best_lr_lookup = _load_best_lr_lookup(args)
    cells = _recipe_cells(best_lr_lookup, _pilot_target(args))
    steps = _build_steps(cells, smoke=args.smoke)
    results = executor_main(
        steps, parallel=args.parallel, rerun=args.rerun,
        experiment_name="recipe-recipe",
        on_step_complete=_make_wandb_callback(args, cells, steps, "recipe"),
    )
    headlines = [_read_headline(r.output_path) for r in results]
    agg = _aggregate(cells, headlines)
    rows = sorted(
        [[t, m, L, v["n_seeds"], f"{v['partial_mean']:.3f}",
          f"{v['partial_range']:.3f}", f"{v['complete_mean']:.3f}"]
         for (t, m, L), v in agg.items()]
    )
    _print_table(
        f"Stage 2 — recipe + seeds @ {_pilot_target(args)//1000}k params",
        ["task", "mixer", "L", "n_seeds", "test_partial_mean",
         "partial_range", "test_complete_mean"],
        rows,
    )
    recipe = _pick_recipe(agg)
    gate = _tradeoff_gate(agg, recipe)
    print("\n### Picked recipe (per task, Pareto-best L averaged across archs)")
    for t, L in recipe.items():
        print(f"  {t}: L = {L}")
    print("\n### Tradeoff gate")
    for t, ranking in gate["rankings"].items():
        ranks = ", ".join(f"{m}={s:.3f}" for m, s in ranking)
        print(f"  {t}: {ranks}")
    print(f"  diverge: {gate['diverge']}")
    return {
        "stage": "recipe",
        "aggregate": {
            f"{t}|{m}|{L}": v for (t, m, L), v in agg.items()
        },
        "recipe": recipe,
        "tradeoff_gate": {
            "rankings": {t: [list(r) for r in v]
                         for t, v in gate["rankings"].items()},
            "diverge": gate["diverge"],
        },
    }


# --- stage 3: stability ---------------------------------------------------

def _stability_cells(best_lr_lookup: dict, recipe: dict[str, int], target: int) -> list[Cell]:
    cells = []
    for arch in RECIPE_ARCHS:
        for task, n_l in recipe.items():
            lr = best_lr_lookup[(arch, n_l, task)]
            for seed in SEEDS:
                cells.append(Cell(arch, n_l, task, lr, seed, target))
    return cells


def stage_stability(args, best_lr_lookup=None, recipe=None) -> dict:
    if best_lr_lookup is None:
        best_lr_lookup = _load_best_lr_lookup(args)
    if recipe is None:
        recipe = _load_recipe(args)
    cells = _stability_cells(best_lr_lookup, recipe, _stability_target(args))
    steps = _build_steps(cells, smoke=args.smoke)
    results = executor_main(
        steps, parallel=args.parallel, rerun=args.rerun,
        experiment_name="recipe-stability",
        on_step_complete=_make_wandb_callback(args, cells, steps, "stability"),
    )
    headlines = [_read_headline(r.output_path) for r in results]
    agg = _aggregate(cells, headlines)
    rows = sorted(
        [[t, m, L, v["n_seeds"], f"{v['partial_mean']:.3f}",
          f"{v['partial_range']:.3f}", f"{v['complete_mean']:.3f}"]
         for (t, m, L), v in agg.items()]
    )
    _print_table(
        f"Stage 3 — stability @ {_stability_target(args)//1000}k params",
        ["task", "mixer", "L", "n_seeds", "test_partial_mean",
         "partial_range", "test_complete_mean"],
        rows,
    )
    return {
        "stage": "stability",
        "aggregate": {f"{t}|{m}|{L}": v for (t, m, L), v in agg.items()},
    }


# --- inter-stage handoff via .experiment_cache/runs/ ---------------------

def _runs_dir() -> Path:
    return Path(".experiment_cache") / "runs"


def _save_stage_summary(summary: dict, name: str) -> Path:
    out_dir = _runs_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"recipe_summary_{name}.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_best_lr_lookup(args) -> dict:
    path = _runs_dir() / "recipe_summary_lr_selection.json"
    if not path.exists():
        raise SystemExit(
            f"lr selection summary not found at {path}; run --stage lr_selection first"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))["best_lr"]
    out: dict[tuple[str, int, str], float] = {}
    for k, v in raw.items():
        m, L, t = k.split("|")
        out[(m, int(L), t)] = float(v)
    return out


def _load_recipe(args) -> dict[str, int]:
    path = _runs_dir() / "recipe_summary_recipe.json"
    if not path.exists():
        raise SystemExit(
            f"recipe summary not found at {path}; run --stage recipe first"
        )
    return {t: int(L) for t, L in json.loads(path.read_text(encoding="utf-8"))["recipe"].items()}


# --- CLI ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["lr_selection", "recipe", "stability", "all"],
        default="all",
    )
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true",
        help="tiny task/train sizes for a smoke run (cells still build all archs/shapes)",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="log each cell as one W&B run (in addition to the local executor cache)",
    )
    parser.add_argument(
        "--wandb-project", default="linear-attention",
        help="W&B project name (default: linear-attention)",
    )
    parser.add_argument(
        "--wandb-entity", default=None,
        help="W&B entity (team/user); defaults to your wandb login default",
    )
    args = parser.parse_args()

    if args.stage in ("lr_selection", "all"):
        summary = stage_lr_selection(args)
        _save_stage_summary(summary, "lr_selection")
        best_lr_lookup: dict[tuple[str, int, str], float] = {}
        for k, lr in summary["best_lr"].items():
            m, L, t = k.split("|")
            best_lr_lookup[(m, int(L), t)] = float(lr)
    else:
        best_lr_lookup = None

    if args.stage in ("recipe", "all"):
        summary = stage_recipe(args, best_lr_lookup=best_lr_lookup)
        _save_stage_summary(summary, "recipe")
        recipe = summary["recipe"]
    else:
        recipe = None

    if args.stage in ("stability", "all"):
        summary = stage_stability(args, best_lr_lookup=best_lr_lookup, recipe=recipe)
        _save_stage_summary(summary, "stability")


if __name__ == "__main__":
    main()
