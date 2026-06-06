import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from linattn.executor import (
    ExecutorStep,
    SourceSet,
    executor_main,
    output_path_of,
    step_digest,
    this_output_path,
)


@dataclass(frozen=True)
class DummyConfig:
    output_path: str
    value: int
    log_path: str
    upstream_path: str | None = None


def write_dummy(config: DummyConfig) -> None:
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = Path(config.log_path)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{config.value}\n")
    payload = {"value": config.value, "upstream_path": config.upstream_path}
    if config.upstream_path is not None:
        payload["upstream_text"] = Path(config.upstream_path).read_text(encoding="utf-8")
    (output_path / "value.json").write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_path / "value.txt").write_text(str(config.value), encoding="utf-8")


class ExecutorTest(unittest.TestCase):
    def test_digest_is_stable_and_changes_with_config_or_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            log = tmp_path / "calls.log"
            step = ExecutorStep(
                name="dummy/stable",
                fn=write_dummy,
                config=DummyConfig(this_output_path(), 1, str(log)),
                sources=(SourceSet("tmp", (str(source),)),),
            )
            same = ExecutorStep(
                name="dummy/stable",
                fn=write_dummy,
                config=DummyConfig(this_output_path(), 1, str(log)),
                sources=(SourceSet("tmp", (str(source),)),),
            )
            changed_config = ExecutorStep(
                name="dummy/stable",
                fn=write_dummy,
                config=DummyConfig(this_output_path(), 2, str(log)),
                sources=(SourceSet("tmp", (str(source),)),),
            )

            self.assertEqual(step_digest(step), step_digest(same))
            self.assertNotEqual(step_digest(step), step_digest(changed_config))

            before = step_digest(step)
            source.write_text("VALUE = 2\n", encoding="utf-8")
            self.assertNotEqual(before, step_digest(step))

    def test_cache_hit_rerun_and_duplicate_dedupe(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            log = tmp_path / "calls.log"
            prefix = tmp_path / "cache"
            step = ExecutorStep(
                name="dummy/cache",
                fn=write_dummy,
                config=DummyConfig(this_output_path(), 1, str(log)),
                sources=(str(source),),
            )

            first = executor_main([step], prefix=prefix, experiment_name="unit")
            self.assertEqual(first[0].cache_status, "fresh")
            self.assertEqual(log.read_text(encoding="utf-8").splitlines(), ["1"])
            self.assertTrue((Path(first[0].output_path) / "_SUCCESS").exists())

            second = executor_main([step], prefix=prefix, experiment_name="unit")
            self.assertEqual(second[0].cache_status, "cached")
            self.assertEqual(log.read_text(encoding="utf-8").splitlines(), ["1"])

            third = executor_main([step], prefix=prefix, rerun=True, experiment_name="unit")
            self.assertEqual(third[0].cache_status, "fresh")
            self.assertEqual(log.read_text(encoding="utf-8").splitlines(), ["1", "1"])

            duplicate_log = tmp_path / "duplicate.log"
            duplicate = ExecutorStep(
                name="dummy/duplicate",
                fn=write_dummy,
                config=DummyConfig(this_output_path(), 3, str(duplicate_log)),
                sources=(str(source),),
            )
            results = executor_main([duplicate, duplicate], prefix=prefix, experiment_name="unit")
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].digest, results[1].digest)
            self.assertEqual(duplicate_log.read_text(encoding="utf-8").splitlines(), ["3"])

    def test_output_paths_and_dependency_digest_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            log = tmp_path / "calls.log"
            prefix = tmp_path / "cache"
            upstream = ExecutorStep(
                name="dummy/upstream",
                fn=write_dummy,
                config=DummyConfig(this_output_path(), 7, str(log)),
                sources=(str(source),),
            )
            downstream = ExecutorStep(
                name="dummy/downstream",
                fn=write_dummy,
                config=DummyConfig(
                    this_output_path(),
                    9,
                    str(log),
                    output_path_of(upstream, "value.txt"),
                ),
                sources=(str(source),),
            )
            changed_upstream = ExecutorStep(
                name="dummy/upstream",
                fn=write_dummy,
                config=DummyConfig(this_output_path(), 8, str(log)),
                sources=(str(source),),
            )
            changed_downstream = ExecutorStep(
                name="dummy/downstream",
                fn=write_dummy,
                config=DummyConfig(
                    this_output_path(),
                    9,
                    str(log),
                    output_path_of(changed_upstream, "value.txt"),
                ),
                sources=(str(source),),
            )

            self.assertNotEqual(step_digest(downstream), step_digest(changed_downstream))
            result = executor_main([downstream], prefix=prefix, experiment_name="unit")[0]
            payload = json.loads(
                (Path(result.output_path) / "value.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["upstream_text"], "7")
            self.assertEqual(log.read_text(encoding="utf-8").splitlines(), ["7", "9"])

    def test_parallel_independent_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            prefix = tmp_path / "cache"
            steps = [
                ExecutorStep(
                    name=f"dummy/parallel/{value}",
                    fn=write_dummy,
                    config=DummyConfig(
                        this_output_path(),
                        value,
                        str(tmp_path / f"calls-{value}.log"),
                    ),
                    sources=(str(source),),
                )
                for value in (1, 2)
            ]

            results = executor_main(steps, prefix=prefix, parallel=2, experiment_name="unit")

            self.assertEqual([result.cache_status for result in results], ["fresh", "fresh"])
            for value in (1, 2):
                self.assertEqual(
                    (tmp_path / f"calls-{value}.log").read_text(encoding="utf-8").splitlines(),
                    [str(value)],
                )

    def test_manifest_contains_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            prefix = tmp_path / "cache"
            step = ExecutorStep(
                name="dummy/manifest",
                fn=write_dummy,
                config=DummyConfig(this_output_path(), 1, str(tmp_path / "calls.log")),
                sources=(str(source),),
            )

            executor_main([step], prefix=prefix, experiment_name="unit")
            manifests = sorted((prefix / "runs").glob("unit-*.json"))
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))

            self.assertEqual(manifest["experiment_name"], "unit")
            self.assertIn("git_commit", manifest)
            self.assertIn("argv", manifest)
            self.assertEqual(manifest["steps"][0]["cache_status"], "fresh")
            self.assertIn("normalized_config", manifest["steps"][0])
            self.assertIn(str(source.resolve()), manifest["steps"][0]["source_digests"])


if __name__ == "__main__":
    unittest.main()
