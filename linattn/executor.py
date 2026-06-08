"""Small local executor for content-addressed experiment steps.

The executor is intentionally filesystem-first: each step resolves to an output
directory under `.experiment_cache/steps/`, writes status next to artifacts, and
each launch writes one manifest under `.experiment_cache/runs/`.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SourceSet:
    name: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class OutputPath:
    step: "ExecutorStep | None"
    name: str | None = None


@dataclass(frozen=True)
class ExecutorStep:
    name: str
    fn: Callable[[Any], Any]
    config: Any
    sources: tuple[SourceSet | str, ...] = ()
    deps: tuple["ExecutorStep", ...] = ()
    version: str = "1"
    description: str | None = None


@dataclass(frozen=True)
class StepResult:
    name: str
    digest: str
    output_path: str
    cache_status: str
    status: str
    status_path: str


@dataclass(frozen=True)
class _StepPlan:
    step: ExecutorStep
    digest: str
    output_path: Path
    normalized_config: Any
    resolved_config: Any
    source_digests: dict[str, str]
    dependency_digests: dict[str, str]


def this_output_path() -> OutputPath:
    return OutputPath(step=None)


def output_path_of(step: ExecutorStep, name: str | None = None) -> OutputPath:
    return OutputPath(step=step, name=name)


def step_digest(step: ExecutorStep) -> str:
    """Return the content digest for a step and its transitive dependencies."""
    planner = _Planner(Path.cwd(), Path(".experiment_cache"))
    return planner.plan(step).digest


def executor_main(
    steps: list[ExecutorStep] | tuple[ExecutorStep, ...],
    *,
    prefix: str | Path = ".experiment_cache",
    parallel: int = 1,
    rerun: bool = False,
    experiment_name: str | None = None,
    on_step_complete: Callable[[StepResult], None] | None = None,
) -> list[StepResult]:
    """Execute steps locally and return one result per requested step.

    `parallel=1` is serial and easiest to debug. `parallel>1` uses spawned
    worker processes so JAX-heavy cells behave like the old experiment scripts.

    ``on_step_complete`` fires once per cell as soon as it finishes (cached or
    fresh), with the cell's ``StepResult``. Use this to stream side-effects
    (e.g. W&B logging) without waiting for the whole batch.
    """

    if parallel < 1:
        raise ValueError(f"parallel must be >= 1, got {parallel}")

    root = Path.cwd()
    prefix_path = Path(prefix)
    if not prefix_path.is_absolute():
        prefix_path = root / prefix_path
    (prefix_path / "steps").mkdir(parents=True, exist_ok=True)
    (prefix_path / "runs").mkdir(parents=True, exist_ok=True)

    planner = _Planner(root, prefix_path)
    plans_by_digest: dict[str, _StepPlan] = {}
    requested_digests: list[str] = []
    for step in steps:
        plan = planner.plan(step)
        requested_digests.append(plan.digest)
    for plan in planner.plans():
        plans_by_digest.setdefault(plan.digest, plan)

    ordered_plans = _topological_unique(plans_by_digest.values())

    def _on_done(plan: _StepPlan, status: str) -> None:
        if on_step_complete is None:
            return
        on_step_complete(
            StepResult(
                name=plan.step.name,
                digest=plan.digest,
                output_path=str(plan.output_path),
                cache_status=status,
                status="SUCCESS",
                status_path=str(plan.output_path / "status.json"),
            )
        )

    if parallel == 1:
        statuses = _run_serial(ordered_plans, rerun, _on_done)
    else:
        statuses = _run_parallel(ordered_plans, parallel, rerun, _on_done)

    results_by_digest = {
        digest: StepResult(
            name=plans_by_digest[digest].step.name,
            digest=digest,
            output_path=str(plans_by_digest[digest].output_path),
            cache_status=statuses[digest],
            status="SUCCESS",
            status_path=str(plans_by_digest[digest].output_path / "status.json"),
        )
        for digest in statuses
    }
    results = [results_by_digest[digest] for digest in requested_digests]
    _write_manifest(
        prefix_path,
        experiment_name or _default_experiment_name(steps),
        root,
        ordered_plans,
        statuses,
        results,
    )
    return results


class _Planner:
    def __init__(self, root: Path, prefix: Path):
        self.root = root
        self.prefix = prefix
        self._plans_by_id: dict[int, _StepPlan] = {}
        self._resolving: set[int] = set()

    def plan(self, step: ExecutorStep) -> _StepPlan:
        step_id = id(step)
        if step_id in self._plans_by_id:
            return self._plans_by_id[step_id]
        if step_id in self._resolving:
            raise ValueError(f"cycle detected while planning {step.name!r}")

        self._resolving.add(step_id)
        dep_steps = _effective_deps(step)
        dep_plans = [self.plan(dep) for dep in dep_steps]
        dependency_digests = {dep.step.name: dep.digest for dep in dep_plans}
        source_digests = _source_digests(self.root, step)
        normalized_config = _normalize_value(
            step.config,
            current_step=step,
            dep_plans={id(dep.step): dep for dep in dep_plans},
            for_digest=True,
        )
        digest_payload = {
            "schema_version": SCHEMA_VERSION,
            "step": step.name,
            "version": step.version,
            "config": normalized_config,
            "dependencies": dependency_digests,
            "sources": source_digests,
        }
        digest = _sha256_json(digest_payload)
        output_path = self.prefix / "steps" / f"{_slug(step.name)}-{digest}"
        plan = _StepPlan(
            step=step,
            digest=digest,
            output_path=output_path,
            normalized_config=normalized_config,
            resolved_config=_resolve_value(
                step.config,
                current_output_path=output_path,
                dep_plans={id(dep.step): dep for dep in dep_plans},
            ),
            source_digests=source_digests,
            dependency_digests=dependency_digests,
        )
        self._plans_by_id[step_id] = plan
        self._resolving.remove(step_id)
        return plan

    def plans(self) -> list[_StepPlan]:
        return list(self._plans_by_id.values())


def _run_serial(
    plans: list[_StepPlan],
    rerun: bool,
    on_done: Callable[[_StepPlan, str], None],
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for plan in plans:
        statuses[plan.digest] = _run_one(plan, rerun)
        print(f"[{statuses[plan.digest]}] {plan.step.name}", flush=True)
        on_done(plan, statuses[plan.digest])
    return statuses


def _run_parallel(
    plans: list[_StepPlan],
    parallel: int,
    rerun: bool,
    on_done: Callable[[_StepPlan, str], None],
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    remaining = {plan.digest: plan for plan in plans}
    running = {}
    ctx = get_context("spawn")
    print(
        f"running {len(plans)} executor steps with --parallel {parallel} "
        "(stdout from workers will interleave)\n",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=parallel, mp_context=ctx) as pool:
        while remaining or running:
            ready = [
                plan
                for plan in remaining.values()
                if all(dep_digest in statuses for dep_digest in plan.dependency_digests.values())
            ]
            while ready and len(running) < parallel:
                plan = ready.pop(0)
                remaining.pop(plan.digest)
                running[pool.submit(_run_one, plan, rerun)] = plan
            if not running:
                blocked = ", ".join(plan.step.name for plan in remaining.values())
                raise RuntimeError(f"no ready steps; dependency cycle near: {blocked}")
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                plan = running.pop(fut)
                statuses[plan.digest] = fut.result()
                print(f"[{statuses[plan.digest]}] {plan.step.name}", flush=True)
                on_done(plan, statuses[plan.digest])
    return statuses


def _run_one(plan: _StepPlan, rerun: bool) -> str:
    if _is_success(plan.output_path) and not rerun:
        return "cached"

    plan.output_path.mkdir(parents=True, exist_ok=True)
    lock_fd = _acquire_or_wait(plan, rerun)
    if lock_fd is None:
        return "cached"

    status_path = plan.output_path / "status.json"
    try:
        _clean_output_dir(plan.output_path)
        _write_json(status_path, _status_payload(plan, "RUNNING"))
        plan.step.fn(plan.resolved_config)
        _write_json(status_path, _status_payload(plan, "SUCCESS"))
        (plan.output_path / "_SUCCESS").write_text("", encoding="utf-8")
        return "fresh"
    except BaseException as exc:
        _write_json(
            status_path,
            _status_payload(
                plan,
                "FAILED",
                error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            ),
        )
        raise
    finally:
        os.close(lock_fd)
        try:
            (plan.output_path / "lock").unlink()
        except FileNotFoundError:
            pass


def _acquire_or_wait(plan: _StepPlan, rerun: bool) -> int | None:
    lock_path = plan.output_path / "lock"
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(
                fd,
                json.dumps(
                    {"pid": os.getpid(), "created_at": _now(), "step": plan.step.name},
                    sort_keys=True,
                ).encode(),
            )
            return fd
        except FileExistsError:
            if _is_success(plan.output_path) and not rerun:
                return None
            status = _read_status(plan.output_path)
            if status and status.get("state") == "FAILED":
                raise RuntimeError(
                    f"step {plan.step.name!r} has a failed locked output at "
                    f"{plan.output_path}; pass --rerun after clearing the lock"
                )
            if not status or status.get("state") != "RUNNING":
                raise RuntimeError(
                    f"step {plan.step.name!r} is locked but not RUNNING at "
                    f"{plan.output_path}; remove stale lock manually"
                )
            time.sleep(1.0)


def _clean_output_dir(output_path: Path) -> None:
    for child in output_path.iterdir():
        if child.name == "lock":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _is_success(output_path: Path) -> bool:
    status = _read_status(output_path)
    return (output_path / "_SUCCESS").exists() and bool(status and status.get("state") == "SUCCESS")


def _read_status(output_path: Path) -> dict[str, Any] | None:
    status_path = output_path / "status.json"
    if not status_path.exists():
        return None
    return json.loads(status_path.read_text(encoding="utf-8"))


def _status_payload(
    plan: _StepPlan,
    state: str,
    *,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "timestamp": _now(),
        "git_commit": _git_commit(),
        "step": {
            "name": plan.step.name,
            "version": plan.step.version,
            "description": plan.step.description,
            "digest": plan.digest,
            "output_path": str(plan.output_path),
            "normalized_config": plan.normalized_config,
            "dependency_digests": plan.dependency_digests,
            "source_digests": plan.source_digests,
        },
    }
    if error is not None:
        payload["error"] = error
    return payload


def _write_manifest(
    prefix: Path,
    experiment_name: str,
    root: Path,
    plans: list[_StepPlan],
    statuses: dict[str, str],
    results: list[StepResult],
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": experiment_name,
        "timestamp": _now(),
        "git_commit": _git_commit(),
        "argv": sys.argv,
        "cwd": str(root),
        "steps": [
            {
                "name": plan.step.name,
                "version": plan.step.version,
                "description": plan.step.description,
                "digest": plan.digest,
                "output_path": str(plan.output_path),
                "normalized_config": plan.normalized_config,
                "dependency_digests": plan.dependency_digests,
                "source_digests": plan.source_digests,
                "cache_status": statuses[plan.digest],
            }
            for plan in plans
        ],
        "requested_steps": [dataclasses.asdict(result) for result in results],
    }
    run_digest = _sha256_json(payload)[:16]
    timestamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%S")
    path = prefix / "runs" / f"{_slug(experiment_name)}-{timestamp}-{run_digest}.json"
    _write_json(path, payload)


def _topological_unique(plans: list[_StepPlan] | Any) -> list[_StepPlan]:
    by_digest = {plan.digest: plan for plan in plans}
    ordered: list[_StepPlan] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(plan: _StepPlan) -> None:
        if plan.digest in visited:
            return
        if plan.digest in visiting:
            raise ValueError(f"cycle detected near {plan.step.name!r}")
        visiting.add(plan.digest)
        for dep_digest in plan.dependency_digests.values():
            if dep_digest in by_digest:
                visit(by_digest[dep_digest])
        visiting.remove(plan.digest)
        visited.add(plan.digest)
        ordered.append(plan)

    for plan in by_digest.values():
        visit(plan)
    return ordered


def _effective_deps(step: ExecutorStep) -> tuple[ExecutorStep, ...]:
    seen: set[int] = set()
    deps: list[ExecutorStep] = []

    def add(dep: ExecutorStep) -> None:
        if id(dep) not in seen:
            seen.add(id(dep))
            deps.append(dep)

    for dep in step.deps:
        add(dep)
    for dep in _referenced_output_steps(step.config):
        add(dep)
    return tuple(deps)


def _referenced_output_steps(value: Any) -> list[ExecutorStep]:
    refs: list[ExecutorStep] = []

    def walk(x: Any) -> None:
        if isinstance(x, OutputPath):
            if x.step is not None:
                refs.append(x.step)
            return
        if dataclasses.is_dataclass(x) and not isinstance(x, type):
            for field in dataclasses.fields(x):
                walk(getattr(x, field.name))
            return
        if isinstance(x, dict):
            for key, val in x.items():
                walk(key)
                walk(val)
            return
        if isinstance(x, (list, tuple)):
            for item in x:
                walk(item)

    walk(value)
    return refs


def _normalize_value(
    value: Any,
    *,
    current_step: ExecutorStep,
    dep_plans: dict[int, _StepPlan],
    for_digest: bool,
) -> Any:
    if isinstance(value, OutputPath):
        if value.step is None:
            return {"__output_path__": "this", "name": value.name}
        dep_plan = dep_plans[id(value.step)]
        if for_digest:
            return {
                "__output_path__": "dependency",
                "step": value.step.name,
                "digest": dep_plan.digest,
                "name": value.name,
            }
        path = dep_plan.output_path if value.name is None else dep_plan.output_path / value.name
        return str(path)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _normalize_value(
                getattr(value, field.name),
                current_step=current_step,
                dep_plans=dep_plans,
                for_digest=for_digest,
            )
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): _normalize_value(
                val,
                current_step=current_step,
                dep_plans=dep_plans,
                for_digest=for_digest,
            )
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, tuple):
        return [
            _normalize_value(
                item,
                current_step=current_step,
                dep_plans=dep_plans,
                for_digest=for_digest,
            )
            for item in value
        ]
    if isinstance(value, list):
        return [
            _normalize_value(
                item,
                current_step=current_step,
                dep_plans=dep_plans,
                for_digest=for_digest,
            )
            for item in value
        ]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot normalize {type(value).__name__} in step {current_step.name!r}")


def _resolve_value(value: Any, *, current_output_path: Path, dep_plans: dict[int, _StepPlan]) -> Any:
    if isinstance(value, OutputPath):
        if value.step is None:
            path = current_output_path if value.name is None else current_output_path / value.name
        else:
            dep_plan = dep_plans[id(value.step)]
            path = dep_plan.output_path if value.name is None else dep_plan.output_path / value.name
        return str(path)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        kwargs = {
            field.name: _resolve_value(
                getattr(value, field.name),
                current_output_path=current_output_path,
                dep_plans=dep_plans,
            )
            for field in dataclasses.fields(value)
        }
        return type(value)(**kwargs)
    if isinstance(value, dict):
        return {
            key: _resolve_value(val, current_output_path=current_output_path, dep_plans=dep_plans)
            for key, val in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _resolve_value(item, current_output_path=current_output_path, dep_plans=dep_plans)
            for item in value
        )
    if isinstance(value, list):
        return [
            _resolve_value(item, current_output_path=current_output_path, dep_plans=dep_plans)
            for item in value
        ]
    return value


def _source_digests(root: Path, step: ExecutorStep) -> dict[str, str]:
    paths: list[Path] = []
    fn_source = inspect.getsourcefile(step.fn)
    if fn_source is not None:
        paths.append(Path(fn_source))
    for source in step.sources:
        if isinstance(source, SourceSet):
            paths.extend(Path(path) for path in source.paths)
        else:
            paths.append(Path(source))

    digests: dict[str, str] = {}
    for path in paths:
        resolved = path if path.is_absolute() else root / path
        resolved = resolved.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"source file for step {step.name!r} does not exist: {path}")
        label = _source_label(root, resolved)
        digests[label] = _sha256_bytes(resolved.read_bytes())
    return dict(sorted(digests.items()))


def _source_label(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    return slug or "step"


def _default_experiment_name(steps: list[ExecutorStep] | tuple[ExecutorStep, ...]) -> str:
    if not steps:
        return "experiment"
    return steps[0].name.split("/", 1)[0].split("-", 1)[0] or "experiment"
