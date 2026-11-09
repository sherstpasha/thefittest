from __future__ import annotations

import json
import pickle
import sqlite3
import zlib
from pathlib import Path
from time import perf_counter, time
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

import thefittest.optimizers as optimizers
from thefittest.model_selection._search import _ParameterGridEncoder

from .experiment_setup import TaskData, describe
from .grid_cache import GridEvaluationCache, GridIndexEncoder
from .search_runtime import CandidateEvaluator, search_scoring


def _pack(value: Any) -> bytes:
    return zlib.compress(pickle.dumps(value, protocol=4))


def _unpack(value: bytes) -> Any:
    return pickle.loads(zlib.decompress(value))


class SearchRun:
    def __init__(
        self,
        job: dict[str, Any],
        estimator: BaseEstimator,
        grid: list[dict[str, list]],
        data: TaskData,
        configuration: Optional[dict[str, Any]],
        output: Path,
        n_jobs: int,
        progress: bool,
        batch_size: int,
        cache: Optional[GridEvaluationCache] = None,
    ) -> None:
        self.job = job
        self.grid = ParameterGrid(grid)
        self.data = data
        self.configuration = configuration
        self.output = output
        self.batch_size = batch_size
        self.cache = cache
        if cache is not None and job["method"] != "Grid" and not cache.is_complete():
            raise ValueError("Complete the shared Grid cache before running other searches.")
        self.primary = "accuracy" if data.classification else "rmse"
        self.evaluator = CandidateEvaluator(
            estimator,
            data.X[data.train],
            data.y[data.train],
            data.splits,
            search_scoring(data.classification),
            output,
            n_jobs=n_jobs,
            progress=progress,
            progress_position=2,
        )
        self.connection = sqlite3.connect(output / "search.sqlite")
        self.connection.executescript(
            "CREATE TABLE IF NOT EXISTS candidates ("
            "seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, parameters BLOB, "
            "first_batch INTEGER, first_position INTEGER, first_seen REAL, score REAL, metrics TEXT);"
            "CREATE TABLE IF NOT EXISTS generations ("
            "batch INTEGER, kind TEXT, payload BLOB, PRIMARY KEY(batch, kind));"
            "CREATE TABLE IF NOT EXISTS attempts ("
            "id INTEGER PRIMARY KEY, started REAL, seconds REAL, status TEXT);"
        )
        self.connection.commit()
        self._visited: np.ndarray = np.zeros(len(self.grid), dtype=bool)
        self._best_score = -np.inf
        if cache is not None:
            for key, score in self.connection.execute("SELECT id,score FROM candidates"):
                self._visited[int(key)] = True
                if score is not None:
                    self._best_score = max(self._best_score, score)
        self.attempt = self.connection.execute(
            "INSERT INTO attempts (started,status) VALUES (?, 'running')", (time(),)
        ).lastrowid
        self.connection.commit()
        self.start = perf_counter()
        self.batch = 0
        self.genotypes: Optional[np.ndarray] = None
        self.optimizer: Any = None
        total = len(self.grid) if job["method"] == "Grid" else job["nominal_budget"]
        if job["method"] == "Random":
            total = min(total, len(self.grid))
        self.bar = tqdm(
            total=total,
            desc=job["method"],
            unit="candidate",
            position=1,
            leave=False,
            disable=not progress,
        )
        self.elapsed_before = self.connection.execute(
            "SELECT COALESCE(sum(seconds),0) FROM attempts WHERE id<>?", (self.attempt,)
        ).fetchone()[0]

    def close(self, status: str) -> None:
        self.connection.execute(
            "UPDATE attempts SET seconds=?,status=? WHERE id=?",
            (perf_counter() - self.start, status, self.attempt),
        )
        self.connection.commit()
        self.bar.close()
        self.evaluator.close()
        self.connection.close()

    def _state(self) -> dict[str, Any]:
        if self.optimizer is None:
            return {}
        names = [
            "_population_g_i",
            "_fitness_i",
            "_selection_proba",
            "_crossover_proba",
            "_mutation_proba",
            "_selection_operators",
            "_crossover_operators",
            "_mutation_operators",
            "_H_MR",
            "_H_CR",
            "_MR",
            "_CR",
            "_k",
            "_success_i",
        ]
        state = {
            name: getattr(self.optimizer, name) for name in names if hasattr(self.optimizer, name)
        }
        state["numpy_random_state"] = np.random.get_state()
        return state

    def _snapshot(self, kind: str, value: dict[str, Any]) -> None:
        previous = self.connection.execute(
            "SELECT payload FROM generations WHERE batch=? AND kind=?", (self.batch, kind)
        ).fetchone()
        if previous is not None:
            if kind == "proposals":
                old = _unpack(previous[0])
                if old["candidate_ids"] != value["candidate_ids"] or not np.array_equal(
                    old["genotypes"], value["genotypes"]
                ):
                    raise RuntimeError(
                        "Replay differs from checkpoint; use matching code and environment."
                    )
            return
        self.connection.execute(
            "INSERT INTO generations VALUES (?, ?, ?)", (self.batch, kind, _pack(value))
        )
        self.connection.commit()

    def evaluate(self, candidates: list[dict[str, Any]]) -> np.ndarray:
        self.batch += 1
        keys = [joblib.hash(parameters) for parameters in candidates]
        for position, (key, parameters) in enumerate(zip(keys, candidates)):
            self.connection.execute(
                "INSERT OR IGNORE INTO candidates "
                "(id,parameters,first_batch,first_position,first_seen) VALUES (?,?,?,?,?)",
                (key, _pack(parameters), self.batch, position, time()),
            )
        self.connection.commit()
        results = self.evaluator.evaluate(candidates)
        scores = np.array([r["mean_test_scores"][self.primary] for r in results], dtype=float)
        for key, result, score in zip(keys, results, scores):
            self.connection.execute(
                "UPDATE candidates SET score=?,metrics=? WHERE id=?",
                (
                    float(score) if np.isfinite(score) else None,
                    json.dumps(
                        describe({name: value for name, value in result.items() if name != "folds"})
                    ),
                    key,
                ),
            )
        self.connection.commit()
        return self._record_proposals(keys, scores)

    def _evaluate_indices(self, indices: list[int]) -> np.ndarray:
        assert self.cache is not None
        self.batch += 1
        records = self.cache.evaluate(indices)
        primary = self.cache.metrics.index(self.primary)
        scores = records["test_scores"].mean(axis=1)[:, primary]
        for position, (index, score) in enumerate(zip(indices, scores)):
            if self._visited[index]:
                continue
            self.connection.execute(
                "INSERT INTO candidates "
                "(id,first_batch,first_position,first_seen,score) VALUES (?,?,?,?,?)",
                (
                    str(index),
                    self.batch,
                    position,
                    time(),
                    float(score) if np.isfinite(score) else None,
                ),
            )
            self._visited[index] = True
        self.connection.commit()
        if np.isfinite(scores).any():
            self._best_score = max(self._best_score, np.max(scores[np.isfinite(scores)]))
        return self._record_proposals([str(index) for index in indices], scores)

    def _record_proposals(self, keys: list[str], scores: np.ndarray) -> np.ndarray:
        fitness = 2.0 + (2.0 / np.pi) * np.arctan(scores)
        fitness[~np.isfinite(scores)] = 0.0
        unique_genotypes = (
            len(np.unique(self.genotypes, axis=0)) if self.genotypes is not None else None
        )
        self._snapshot(
            "proposals",
            {
                "candidate_ids": keys,
                "genotypes": self.genotypes,
                "fitness": fitness,
                "scores": scores,
                "unique_genotypes": unique_genotypes,
                "unique_phenotypes": len(set(keys)),
                "elapsed_seconds": self.elapsed_before + perf_counter() - self.start,
                "state": self._state(),
            },
        )
        self.bar.update(len(keys))
        best = (
            (self._best_score if np.isfinite(self._best_score) else None)
            if self.cache is not None
            else self.connection.execute("SELECT max(score) FROM candidates").fetchone()[0]
        )
        if best is not None:
            self.bar.set_postfix(best=round(best if self.data.classification else -best, 6))
        return fitness

    def _decode(self, population: np.ndarray, encoder: _ParameterGridEncoder) -> np.ndarray:
        self.genotypes = population.copy()
        return encoder.transform(population)

    def _generation(self, optimizer: Any) -> None:
        self._snapshot("population", self._state())

    def execute(self) -> dict[str, Any]:
        if self.job["method"] in ("Grid", "Random"):
            if self.job["method"] == "Grid":
                indices = range(len(self.grid))
            else:
                rng = np.random.RandomState(self.job["seed"])
                indices = rng.permutation(len(self.grid))[
                    : min(len(self.grid), self.job["nominal_budget"])
                ]
            for start in range(0, len(indices), self.batch_size):
                batch = [int(i) for i in indices[start : start + self.batch_size]]
                if self.cache is None:
                    self.evaluate([self.grid[i] for i in batch])
                else:
                    self._evaluate_indices(batch)
        else:
            if self.configuration is None:
                raise ValueError("An evolutionary search requires an optimizer configuration.")
            encoder = (
                _ParameterGridEncoder(self.grid.param_grid)
                if self.cache is None
                else GridIndexEncoder(self.grid.param_grid)
            )
            evaluate = self.evaluate if self.cache is None else self._evaluate_indices
            if encoder.str_len == 0:
                evaluate(encoder.transform(np.empty((1, 0), dtype=np.int8)).tolist())
            else:
                optimizer_type = getattr(optimizers, self.configuration["optimizer"])
                self.optimizer = optimizer_type(
                    fitness_function=lambda population: evaluate(population.tolist()),
                    genotype_to_phenotype=lambda population: self._decode(population, encoder),
                    str_len=encoder.str_len,
                    iters=self.job["iters"],
                    pop_size=self.job["pop_size"],
                    random_state=self.job["seed"],
                    n_jobs=1,
                    keep_history=False,
                    optimal_value=None,
                    no_increase_num=None,
                    on_generation=self._generation,
                    **self.configuration["args"],
                )
                self.optimizer.fit()
                self._snapshot("population", self._state())
        best = self.connection.execute(
            "SELECT id,parameters,score,metrics FROM candidates WHERE score IS NOT NULL "
            "ORDER BY score DESC, seq ASC LIMIT 1"
        ).fetchone()
        if best is None:
            raise RuntimeError(
                "All candidates failed; inspect evaluations.sqlite and warnings.jsonl."
            )
        parameters = _unpack(best[1]) if self.cache is None else self.grid[int(best[0])]
        self.bar.set_description("Refit best")
        refit = self.evaluator.refit(
            parameters, self.data.X, self.data.y, self.data.train, self.data.test
        )
        if refit["error"]:
            raise RuntimeError("Best candidate refit failed: " + refit["error"])
        if refit["test_scores"][self.primary] is None:
            raise RuntimeError("Best candidate has a non-finite holdout score.")
        count = self.connection.execute("SELECT count(*) FROM candidates").fetchone()[0]
        proposals = self.connection.execute(
            "SELECT payload FROM generations WHERE kind='proposals'"
        )
        total_proposals = sum(len(_unpack(row[0])["candidate_ids"]) for row in proposals)
        with sqlite3.connect(self.output / "evaluations.sqlite") as evaluations:
            cv_fits = evaluations.execute("SELECT count(*) FROM folds").fetchone()[0]
            model_fits = {"started": 0, "completed": 0}
            for (result,) in evaluations.execute("SELECT result FROM folds"):
                fits = json.loads(result).get("model_fits", {})
                for kind in model_fits:
                    model_fits[kind] += sum(fits.get(kind, {}).values())
        cv_seconds_estimate = 0.0
        cv_metrics = json.loads(best[3]) if self.cache is None else {}
        logical_cv_fits = cv_fits
        if self.cache is not None:
            logical_cv_fits = count * len(self.data.splits)
            cv_fits = logical_cv_fits if self.job["method"] == "Grid" else 0
            model_fits = {"started": 0, "completed": 0}
            visited = np.flatnonzero(self._visited)
            for start in range(0, len(visited), self.batch_size):
                records = self.cache.evaluate(visited[start : start + self.batch_size])
                cv_seconds_estimate += float(np.nansum(records["fit_seconds"]))
                cv_seconds_estimate += float(np.nansum(records["score_seconds"]))
                if self.job["method"] == "Grid":
                    for row, kind in enumerate(("started", "completed")):
                        model_fits[kind] += int(records["model_fits"][:, :, row].sum())
            selected = self.cache.evaluate([int(best[0])])
            cv_metrics = {
                f"{stat}_{split}_scores": {
                    name: float(getattr(selected[f"{split}_scores"][0, :, metric], stat)())
                    for metric, name in enumerate(self.cache.metrics)
                }
                for stat in ("mean", "std")
                for split in ("train", "test")
            }
        sign = 1 if self.data.classification else -1
        return {
            **self.job,
            "status": "ok",
            "metric": self.primary,
            "space_size": len(self.grid),
            "unique_candidates": count,
            "proposal_batches": self.batch,
            "completed_generations": self.batch if self.configuration is not None else None,
            "proposals": total_proposals,
            "cache_hits": total_proposals - count,
            "coverage": count / len(self.grid),
            "cv_fits": cv_fits,
            "logical_cv_fits": logical_cv_fits,
            "cv_seconds_estimate": cv_seconds_estimate if self.cache is not None else None,
            "evaluation_mode": "shared_grid_cache" if self.cache is not None else "independent_cv",
            "cv_model_fits": model_fits,
            "refit_model_fits": refit["model_fits"],
            "best_cv": sign * best[2],
            "cv_metrics": cv_metrics,
            "test_score": sign * refit["test_scores"][self.primary],
            "test_metrics": refit["test_scores"],
            "best_parameters": describe(parameters),
            "candidate_id": best[0],
            "refit_seconds": refit["seconds"],
            "seconds": self.elapsed_before + perf_counter() - self.start,
            "incomplete_prior_attempts": self.connection.execute(
                "SELECT count(*) FROM attempts WHERE id<>? AND seconds IS NULL", (self.attempt,)
            ).fetchone()[0],
            "resume_mode": "deterministic_seed_replay_with_fold_cache",
        }
