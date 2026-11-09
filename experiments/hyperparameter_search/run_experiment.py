from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import socket
import sqlite3
import sys
import traceback
import warnings
import subprocess
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from time import perf_counter, time
from typing import Any, Iterator

import joblib
import numpy as np
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.hyperparameter_search.experiment_setup import (
    atomic_json,
    describe,
    prepare_task,
    search_space,
)
from experiments.hyperparameter_search.plan_search_experiment import FAMILIES, SearchPlan
from experiments.hyperparameter_search.search_execution import SearchRun

ROOT = Path(__file__).resolve().parent
METHODS = ("GA", "PDPGA", "SelfCGA", "SelfCSHAGA", "PDPSHAGA", "SHAGA", "Random", "Grid")


@contextmanager
def directory_lock(output: Path) -> Iterator[None]:
    path = output / "experiment.lock"
    with path.open("a+b") as stream:
        if sys.platform == "win32":
            import msvcrt

            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("Another process is using this experiment directory.") from error
        else:
            import fcntl

            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another process is using this experiment directory.") from error
        try:
            yield
        finally:
            if sys.platform == "win32":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class Experiment:
    def __init__(
        self,
        output: Path,
        n_jobs: int,
        seeds: tuple[int, ...],
        tasks: list[str],
        families: list[str],
        methods: list[str],
        budgets: tuple[int, ...],
        samples: int,
        noise: float,
        data_seed: int,
        max_samples: int,
        smoke: bool,
        progress: bool,
        batch_size: int,
    ) -> None:
        self.output = output
        self.n_jobs = n_jobs
        self.plan = SearchPlan(seeds, n_jobs=n_jobs)
        self.plan.tasks = [task for task in self.plan.tasks if task["id"] in tasks]
        self.plan.budgets = budgets
        self.plan.configurations = [c for c in self.plan.configurations if c["method"] in methods]
        if smoke:
            selected: dict[str, dict[str, Any]] = {}
            for configuration in self.plan.configurations:
                selected.setdefault(configuration["method"], configuration)
            self.plan.configurations = list(selected.values())
        self.configurations = {c["id"]: c for c in self.plan.configurations}
        self.families = families
        self.methods = methods
        self.samples, self.noise, self.data_seed = samples, noise, data_seed
        self.max_samples = max_samples
        self.smoke, self.progress, self.batch_size = smoke, progress, batch_size
        self.total = (
            len(self.plan.tasks)
            * len(families)
            * (
                len(seeds) * len(budgets) * (len(self.configurations) + int("Random" in methods))
                + int("Grid" in methods)
            )
        )

    def jobs(self) -> Iterator[dict[str, Any]]:
        for job in self.plan.jobs():
            if job["family"] in self.families and job["method"] in self.methods:
                yield job

    def _manifest(self) -> dict[str, Any]:
        from thefittest import optimizers

        package = Path(optimizers.__file__).resolve().parents[1]
        source_files = {
            "experiment/"
            + p.name: hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest()
            for p in ROOT.glob("*.py")
            if not p.name.startswith("test_")
        }
        source_files.update(
            {
                "thefittest/"
                + str(p.relative_to(package))
                .replace("\\", "/"): hashlib.sha256(p.read_text(encoding="utf-8").encode())
                .hexdigest()
                for p in package.rglob("*.py")
                if "tests" not in p.parts
            }
        )
        return describe(
            {
                "schema": 1,
                "spec_sha256": hashlib.sha256(
                    (ROOT / "EXPERIMENT_SPEC.md").read_text(encoding="utf-8").encode()
                ).hexdigest(),
                "source_files": source_files,
                "versions": {
                    name: version(name)
                    for name in [
                        "thefittest",
                        "scikit-learn",
                        "numpy",
                        "numba",
                        "joblib",
                        "scipy",
                        "tqdm",
                    ]
                },
                "python": platform.python_version(),
                "tasks": self.plan.tasks,
                "families": self.families,
                "methods": self.methods,
                "seeds": self.plan.seeds,
                "budgets": self.plan.budgets,
                "optimizer_configurations": self.plan.configurations,
                "generation": {
                    "samples": self.samples,
                    "noise_fraction": self.noise,
                    "data_seed": self.data_seed,
                    "max_samples": self.max_samples,
                },
                "duplicate_policy": "drop identical X,y rows; group remaining identical X across splits",
                "split_seed": 42,
                "test_fraction": 0.25,
                "cv_folds": 3,
                "smoke": self.smoke,
                "batch_size": self.batch_size,
                "total_searches": self.total,
                "resume_mode": "deterministic_seed_replay_with_fold_cache",
                "fitness_transform": "2 + (2/pi)*atan(primary_cv_score); nonfinite -> 0",
                "backend": "loky",
                "inner_max_num_threads": 1,
            }
        )

    def _prepare(self) -> None:
        for directory in ["data", "cases", "runs"]:
            (self.output / directory).mkdir(parents=True, exist_ok=True)
        manifest = self._manifest()
        path = self.output / "manifest.json"
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                raise ValueError(
                    "Experiment settings, specification or code changed; select a different --output."
                )
        else:
            atomic_json(path, manifest)
        with (self.output / "sessions.jsonl").open("a", encoding="utf-8") as stream:
            try:
                commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
                ).strip()
                dirty = subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
                )
            except (OSError, subprocess.CalledProcessError):
                commit, dirty = None, None
            stream.write(
                json.dumps(
                    {
                        "started": time(),
                        "host": socket.gethostname(),
                        "platform": platform.platform(),
                        "logical_cpus": os.cpu_count(),
                        "n_jobs": self.n_jobs,
                        "argv": sys.argv,
                        "git_commit": commit,
                        "git_status": dirty,
                    }
                )
                + "\n"
            )
        if not (self.output / "jobs.jsonl").exists():
            temporary = self.output / "jobs.tmp"
            count = 0
            with temporary.open("w", encoding="utf-8") as stream:
                for job in self.jobs():
                    stream.write(json.dumps(job) + "\n")
                    count += 1
            if count != self.total:
                raise RuntimeError("Generated job count differs from the manifest.")
            temporary.replace(self.output / "jobs.jsonl")
        for task in tqdm(self.plan.tasks, desc="Prepare datasets", disable=not self.progress):
            data = self._task(task)
            for family in self.families:
                path = self.output / "cases" / f"{task['id']}__{family}.joblib"
                if path.exists() and path.with_suffix(".json").exists():
                    metadata = json.loads(path.with_suffix(".json").read_text())
                    if joblib.hash(joblib.load(path)) != metadata["signature"]:
                        raise ValueError(f"Search-space checkpoint checksum mismatch: {path}")
                    continue
                estimator, grid = search_space(
                    family, data.X.shape[1], data.classification, self.smoke
                )
                count = len(ParameterGrid(grid))
                temporary = path.with_suffix(".tmp")
                joblib.dump((estimator, grid), temporary)
                temporary.replace(path)
                atomic_json(
                    path.with_suffix(".json"),
                    {
                        "task": task["id"],
                        "family": family,
                        "space_size": count,
                        "estimator": estimator,
                        "grid": grid,
                        "signature": joblib.hash((estimator, grid)),
                    },
                )

    def _task(self, task: dict[str, str]) -> Any:
        return prepare_task(
            task, self.output / "data", self.samples, self.noise, self.data_seed, self.max_samples
        )

    def _csv(self, connection: sqlite3.Connection) -> None:
        fields = [
            "id",
            "task",
            "family",
            "method",
            "configuration",
            "seed",
            "pop_size",
            "iters",
            "status",
            "space_size",
            "unique_candidates",
            "proposals",
            "cache_hits",
            "coverage",
            "cv_fits",
            "best_cv",
            "test_score",
            "seconds",
            "incomplete_prior_attempts",
        ]
        temporary = self.output / "summary.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for (payload,) in connection.execute(
                "SELECT result FROM runs WHERE result IS NOT NULL ORDER BY rowid"
            ):
                writer.writerow(json.loads(payload))
        temporary.replace(self.output / "summary.csv")

    def _append_csv(self, result: dict[str, Any]) -> None:
        with (self.output / "summary.csv").open("r", encoding="utf-8") as stream:
            fields = next(csv.reader(stream))
        with (self.output / "summary.csv").open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore").writerow(result)

    def run(self) -> dict[str, int]:
        self.output.mkdir(parents=True, exist_ok=True)
        with directory_lock(self.output):
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(message)s",
                handlers=[logging.FileHandler(self.output / "run.log", encoding="utf-8")],
                force=True,
            )
            self._prepare()
            connection = sqlite3.connect(self.output / "results.sqlite")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, status TEXT, result TEXT)"
            )
            connection.commit()
            self._csv(connection)
            complete = connection.execute("SELECT count(*) FROM runs WHERE status='ok'").fetchone()[
                0
            ]
            try:
                self._execute(connection, complete)
            finally:
                self._csv(connection)
                connection.close()
            with sqlite3.connect(self.output / "results.sqlite") as results:
                counts = dict(results.execute("SELECT status,count(*) FROM runs GROUP BY status"))
            logging.info("Finished %s", counts)
            return counts

    def _execute(self, connection: sqlite3.Connection, complete: int) -> None:
        task_lookup = {task["id"]: task for task in self.plan.tasks}
        active_case = None
        data = estimator = grid = None
        with tqdm(
            total=self.total,
            initial=complete,
            desc="Searches",
            unit="search",
            position=0,
            disable=not self.progress,
        ) as bar:
            for job in self.jobs():
                previous = connection.execute(
                    "SELECT status FROM runs WHERE id=?", (job["id"],)
                ).fetchone()
                if previous is not None and previous[0] == "ok":
                    continue
                case = (job["task"], job["family"])
                if case != active_case:
                    data = self._task(task_lookup[job["task"]])
                    estimator, grid = joblib.load(
                        self.output / "cases" / f"{job['task']}__{job['family']}.joblib"
                    )
                    active_case = case
                run_dir = self.output / "runs" / hashlib.sha256(job["id"].encode()).hexdigest()[:24]
                run_dir.mkdir(exist_ok=True)
                atomic_json(run_dir / "job.json", job)
                logging.info("START %s", job["id"])
                bar.set_postfix_str(f"{job['task']} / {job['family']} / {job['method']}")
                search = None
                status = "failed"
                start = perf_counter()
                with warnings.catch_warnings(record=True) as messages:
                    warnings.simplefilter("always")
                    try:
                        search = SearchRun(
                            job,
                            estimator,
                            grid,
                            data,
                            self.configurations.get(job.get("configuration")),
                            run_dir,
                            self.n_jobs,
                            self.progress,
                            self.batch_size,
                        )
                        result = search.execute()
                        status = "ok"
                    except KeyboardInterrupt:
                        status = "interrupted"
                        logging.info(
                            "INTERRUPTED %s; rerun the same command to continue", job["id"]
                        )
                        raise
                    except Exception:
                        result = {
                            **job,
                            "status": "failed",
                            "seconds": perf_counter() - start,
                            "error": traceback.format_exc(),
                        }
                        logging.exception("FAILED %s", job["id"])
                    finally:
                        if search is not None:
                            search.close(status)
                        if messages:
                            with (run_dir / "warnings.jsonl").open("a", encoding="utf-8") as stream:
                                for warning in messages:
                                    stream.write(
                                        json.dumps(
                                            {
                                                "category": warning.category.__name__,
                                                "message": str(warning.message),
                                            }
                                        )
                                        + "\n"
                                    )
                result["directory"] = str(run_dir.relative_to(self.output))
                atomic_json(run_dir / "result.json", result)
                payload = json.dumps(describe(result), ensure_ascii=False)
                connection.execute(
                    "INSERT OR REPLACE INTO runs VALUES (?,?,?)",
                    (job["id"], result["status"], payload),
                )
                connection.commit()
                self._append_csv(describe(result))
                logging.info(
                    "DONE %s status=%s CV=%s test=%s seconds=%.3f",
                    job["id"],
                    result["status"],
                    result.get("best_cv"),
                    result.get("test_score"),
                    result["seconds"],
                )
                bar.update(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or resume the complete hyperparameter search experiment."
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--families", nargs="+", choices=list(FAMILIES))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--budgets", type=int, nargs="+")
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.jobs != -1 and args.jobs < 1:
        parser.error("jobs must be -1 or a positive integer.")
    plan = SearchPlan(tuple(args.seeds), n_jobs=args.jobs)
    available = {task["id"] for task in plan.tasks}
    tasks = args.tasks or (["IrisDataset", "Diabetes"] if args.smoke else list(available))
    if not set(tasks) <= available:
        parser.error("Unknown task; use the identifiers in EXPERIMENT_SPEC.md.")
    budgets = tuple(args.budgets or ([10] if args.smoke else plan.budgets))
    if any(q < 10 for q in budgets) or len(set(budgets)) != len(budgets):
        parser.error("Budgets must be unique and >= 10.")
    if (
        args.samples < 120
        or not np.isfinite(args.noise)
        or args.noise < 0
        or args.max_samples < 0
        or args.batch_size < 1
    ):
        parser.error("Require samples >=120, noise >=0, max-samples >=0 and batch-size >=1.")
    if args.max_samples and args.max_samples < 120:
        parser.error("max-samples must be 0 or >=120.")
    if args.data_seed < 0 or args.data_seed + 17 >= 2**32:
        parser.error("Invalid data seed.")
    for name, values in [("tasks", tasks), ("families", args.families), ("methods", args.methods)]:
        if values is not None and len(set(values)) != len(values):
            parser.error(f"{name} must be unique.")
    experiment = Experiment(
        output=args.output or ROOT / "output" / ("smoke" if args.smoke else "experiment"),
        n_jobs=args.jobs,
        seeds=tuple(args.seeds),
        tasks=tasks,
        families=args.families or list(FAMILIES),
        methods=args.methods,
        budgets=budgets,
        samples=120 if args.smoke else args.samples,
        noise=args.noise,
        data_seed=args.data_seed,
        max_samples=120 if args.smoke else args.max_samples,
        smoke=args.smoke,
        progress=not args.no_progress,
        batch_size=args.batch_size,
    )
    try:
        counts = experiment.run()
    except KeyboardInterrupt:
        print(
            "Interrupted. Rerun the same command to resume completed CV evaluations.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    print(json.dumps(counts, indent=2))
    if counts.get("failed", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
