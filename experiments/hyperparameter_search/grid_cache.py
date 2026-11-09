from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import scipy
import sklearn
from joblib import Parallel, delayed, parallel_config
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm
from thefittest.model_selection._search import _ParameterGridEncoder

from .experiment_setup import atomic_json
from .search_runtime import _evaluate_fold


class GridIndexEncoder(_ParameterGridEncoder):
    """Decode the same binary blocks as the search encoder into ParameterGrid indices."""

    def __init__(self, param_grid: list[dict[str, list]]) -> None:
        super().__init__(param_grid)
        sizes = [len(ParameterGrid(branch)) for branch in param_grid]
        self._offsets = np.cumsum([0, *sizes[:-1]])

    def transform(self, population: np.ndarray) -> np.ndarray:
        if population.ndim != 2 or population.shape[1] != self.str_len:
            raise ValueError(f"population must have shape (n_individuals, {self.str_len}).")
        if not np.all((population == 0) | (population == 1)):
            raise ValueError("population must contain only binary values.")
        indices: np.ndarray = np.empty(len(population), dtype=np.int64)
        for row, individual in enumerate(population):
            branch = self._decode_index(individual[: self._selector_bits], len(self._values))
            start, index = self._starts[branch], 0
            for name, values in self._values[branch].items():
                stop = start + self._bits[branch][name]
                index = index * len(values) + self._decode_index(
                    individual[start:stop], len(values)
                )
                start = stop
            indices[row] = self._offsets[branch] + index
        return indices


class GridEvaluationCache:
    """Persist CV fold results by grid index for one dataset and model family.

    Only completed folds are reused. Data, estimator, grid, scoring and CV splits
    must match when reopening the cache. Returned records follow request order,
    including repeated indices; search histories belong to the caller. The caller
    must serialize writers to the same directory.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: list[dict[str, list]],
        X: np.ndarray,
        y: np.ndarray,
        splits: list[tuple[np.ndarray, np.ndarray]],
        scoring: dict[str, str],
        output: Path,
        n_jobs: int = 1,
        progress: bool = False,
    ) -> None:
        if not splits or not scoring:
            raise ValueError("Provide CV splits and scoring metrics.")
        if n_jobs != -1 and n_jobs < 1:
            raise ValueError("n_jobs must be -1 or a positive integer.")
        self.grid = ParameterGrid(param_grid)
        self.metrics = tuple(scoring)
        self._estimator, self._X, self._y = estimator, X, y
        self._splits, self._scoring = splits, dict(scoring)
        self._n_jobs = n_jobs
        self._progress = progress
        self.last_evaluated_folds = 0
        components = [estimator]
        for branch in param_grid:
            for values in branch.values():
                components.extend(value for value in values if isinstance(value, BaseEstimator))
        models = [
            item
            for component in components
            for item in [component, *component.get_params(deep=True).values()]
            if isinstance(item, BaseEstimator) and hasattr(item, "predict")
        ]
        self.model_names = tuple(
            sorted({type(model).__module__ + "." + type(model).__name__ for model in models})
        )
        dtype = np.dtype(
            [
                ("train_scores", "<f8", (len(scoring),)),
                ("test_scores", "<f8", (len(scoring),)),
                ("fit_seconds", "<f8"),
                ("score_seconds", "<f8"),
                ("seconds", "<f8"),
                ("started_at", "<f8"),
                ("finished_at", "<f8"),
                ("model_fits", "<i8", (2, len(self.model_names))),
            ]
        )
        signature = joblib.hash(
            (
                1,
                sklearn.__version__,
                np.__version__,
                scipy.__version__,
                estimator,
                param_grid,
                X,
                y,
                splits,
                scoring,
                self.metrics,
            )
        )
        output.mkdir(parents=True, exist_ok=True)
        manifest = output / "manifest.json"
        shape = (len(self.grid), len(splits))
        if manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8"))["signature"] != signature:
                raise ValueError(
                    "Cache belongs to a different grid, estimator, data, CV or scoring."
                )
            self._values = np.load(output / "folds.npy", mmap_mode="r+")
            self._done = np.load(output / "status.npy", mmap_mode="r+")
            if self._values.shape != shape or self._values.dtype != dtype:
                raise ValueError("Invalid fold cache layout.")
            if self._done.shape != shape or self._done.dtype != np.dtype("uint8"):
                raise ValueError("Invalid fold status layout.")
        else:
            self._values = np.lib.format.open_memmap(
                output / "folds.npy", mode="w+", dtype=dtype, shape=shape
            )
            self._done = np.lib.format.open_memmap(
                output / "status.npy", mode="w+", dtype="uint8", shape=shape
            )
            self._done[:] = 0
            self._done.flush()
            joblib.dump((estimator, param_grid), output / "space.joblib")
            atomic_json(
                manifest,
                {
                    "signature": signature,
                    "metrics": self.metrics,
                    "model_names": self.model_names,
                    "shape": shape,
                    "bytes_per_candidate": (dtype.itemsize + 1) * len(splits),
                },
            )
        self._diagnostics = sqlite3.connect(output / "diagnostics.sqlite")
        self._diagnostics.execute(
            "CREATE TABLE IF NOT EXISTS diagnostics ("
            "candidate INTEGER, fold INTEGER, error TEXT, warnings TEXT, "
            "PRIMARY KEY(candidate,fold))"
        )
        self._diagnostics.commit()
        self._complete = False

    def is_complete(self) -> bool:
        if not self._complete:
            self._complete = bool(np.all(self._done != 0))
        return self._complete

    def close(self) -> None:
        self._diagnostics.close()
        self._values._mmap.close()
        self._done._mmap.close()

    def _store(self, index: int, fold: int, result: dict[str, Any]) -> None:
        record = self._values[index, fold]
        for split in ("train", "test"):
            record[f"{split}_scores"] = [result[f"{split}_scores"][m] for m in self.metrics]
        for field in ("fit_seconds", "score_seconds", "seconds", "started_at", "finished_at"):
            record[field] = np.nan if result[field] is None else result[field]
        counts = result["model_fits"]
        for row, kind in enumerate(("started", "completed")):
            record["model_fits"][row] = [counts[kind].get(name, 0) for name in self.model_names]
        if result["error"] or result["warnings"]:
            self._diagnostics.execute(
                "INSERT OR REPLACE INTO diagnostics VALUES (?,?,?,?)",
                (index, fold, result["error"], json.dumps(result["warnings"], ensure_ascii=False)),
            )

    def evaluate(self, indices: Sequence[int]) -> np.ndarray:
        requested = np.asarray(indices)
        if requested.ndim != 1 or (requested.size and requested.dtype.kind not in "iu"):
            raise ValueError("Provide a one-dimensional sequence of integer grid indices.")
        if np.any(requested < 0) or np.any(requested >= len(self.grid)):
            raise ValueError("Grid index out of range.")
        requested = requested.astype(np.int64)
        self.last_evaluated_folds = 0
        if self._complete:
            return self._values[requested]
        pending = [
            (int(index), fold)
            for index in np.unique(requested)
            for fold in range(len(self._splits))
            if self._done[index, fold] == 0
        ]
        if not pending:
            return self._values[requested]
        completed = []
        try:
            with parallel_config(backend="loky", inner_max_num_threads=1):
                with (
                    closing(
                        Parallel(
                            n_jobs=self._n_jobs,
                            return_as="generator_unordered",
                            pre_dispatch="n_jobs",
                        )(
                            delayed(_evaluate_fold)(
                                self._estimator,
                                self.grid[index],
                                self._X,
                                self._y,
                                self._splits[fold][0],
                                self._splits[fold][1],
                                self._scoring,
                                str(index),
                                fold,
                            )
                            for index, fold in pending
                        )
                    ) as results,
                    tqdm(
                        total=len(pending),
                        desc="CV folds",
                        unit="fit",
                        position=2,
                        leave=False,
                        disable=not self._progress,
                    ) as bar,
                ):
                    for result in results:
                        index, fold = int(result["candidate"]), result["fold"]
                        self._store(index, fold, result)
                        completed.append((index, fold, 2 if result["error"] else 1))
                        self.last_evaluated_folds += 1
                        bar.update(1)
        finally:
            self._values.flush()
            self._diagnostics.commit()
            for index, fold, status in completed:
                self._done[index, fold] = status
            self._done.flush()
        return self._values[requested]

    def populate(self, batch_size: int = 32) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        for start in range(0, len(self.grid), batch_size):
            self.evaluate(range(start, min(start + batch_size, len(self.grid))))
