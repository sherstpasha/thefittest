from __future__ import annotations

import io
import json
import sqlite3
import traceback
import warnings
from contextlib import contextmanager, redirect_stderr
from functools import wraps
from pathlib import Path
from time import perf_counter, time
from typing import Any, Iterator, Optional

import numpy as np
import sklearn
from joblib import Parallel, delayed, hash as joblib_hash, parallel_config
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import check_scoring, get_scorer
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


class FeatureGroupCV(BaseCrossValidator):
    def __init__(self, classification: bool, n_splits: int = 3, random_state: int = 42) -> None:
        self.classification = classification
        self.n_splits = n_splits
        self.random_state = random_state

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(self, X: np.ndarray, y: Any = None, groups: Any = None) -> Iterator[Any]:
        if groups is None:
            _, groups = np.unique(X, axis=0, return_inverse=True)
        if len(np.unique(groups)) < len(X):
            splitter = (
                StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=self.random_state)
                if self.classification
                else GroupKFold(self.n_splits)
            )
            return splitter.split(X, y, groups)
        splitter_type = StratifiedKFold if self.classification else KFold
        return splitter_type(self.n_splits, shuffle=True, random_state=self.random_state).split(
            X, y
        )


def _warning_records(messages: list[warnings.WarningMessage]) -> list[dict[str, Any]]:
    return [
        {
            "category": item.category.__name__,
            "message": str(item.message),
            "filename": item.filename,
            "lineno": item.lineno,
        }
        for item in messages
    ]


@contextmanager
def track_model_fits(model: BaseEstimator) -> Iterator[dict[str, dict[str, int]]]:
    counts: dict[str, dict[str, int]] = {"started": {}, "completed": {}}
    containers = (Pipeline, TransformedTargetRegressor, StackingClassifier, StackingRegressor)
    components = [model, *model.get_params(deep=True).values()]
    classes = {
        type(item)
        for item in components
        if isinstance(item, BaseEstimator)
        and hasattr(item, "predict")
        and not isinstance(item, containers)
    }
    originals = {cls: cls.fit for cls in classes}

    def counted(original: Any, name: str) -> Any:
        @wraps(original)
        def fit(*args: Any, **kwargs: Any) -> Any:
            counts["started"][name] = counts["started"].get(name, 0) + 1
            result = original(*args, **kwargs)
            counts["completed"][name] = counts["completed"].get(name, 0) + 1
            return result

        return fit

    try:
        for cls, original in originals.items():
            cls.fit = counted(original, cls.__module__ + "." + cls.__name__)
        yield counts
    finally:
        for cls, original in originals.items():
            cls.fit = original


def _evaluate_fold(
    estimator: BaseEstimator,
    parameters: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    scoring: dict[str, str],
    candidate: str,
    fold: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "candidate": candidate,
        "fold": fold,
        "train_scores": {},
        "test_scores": {},
        "fit_seconds": None,
        "score_seconds": None,
        "error": None,
        "started_at": time(),
    }
    start = perf_counter()
    diagnostics = io.StringIO()
    with (
        warnings.catch_warnings(record=True) as messages,
        threadpool_limits(limits=1),
        redirect_stderr(diagnostics),
    ):
        warnings.simplefilter("always")
        try:
            model = clone(estimator).set_params(**clone(parameters, safe=False))
            with track_model_fits(model) as counts:
                model.fit(X[train], y[train])
            result["fit_seconds"] = perf_counter() - start
            start_score = perf_counter()
            scorers = check_scoring(model, scoring=scoring)
            for split, indices in [("train", train), ("test", test)]:
                for name, score in scorers(model, X[indices], y[indices]).items():
                    value = float(score)
                    result[f"{split}_scores"][name] = value if np.isfinite(value) else None
            result["score_seconds"] = perf_counter() - start_score
        except Exception:
            result["error"] = traceback.format_exc()
            result["train_scores"] = dict.fromkeys(scoring)
            result["test_scores"] = dict.fromkeys(scoring)
    result["warnings"] = _warning_records(messages)
    if diagnostics.getvalue():
        result["warnings"].append(
            {
                "category": "StderrDiagnostic",
                "message": diagnostics.getvalue(),
                "filename": "",
                "lineno": 0,
            }
        )
    result["model_fits"] = counts if "counts" in locals() else {"started": {}, "completed": {}}
    result["seconds"] = perf_counter() - start
    result["finished_at"] = time()
    return result


class CandidateEvaluator:
    """Evaluate candidates on fixed CV folds and persist completed fold evaluations."""

    def __init__(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        splits: list[tuple[np.ndarray, np.ndarray]],
        scoring: dict[str, str],
        output: Path,
        n_jobs: int = 1,
        progress: bool = True,
        progress_position: int = 0,
    ) -> None:
        if n_jobs != -1 and n_jobs < 1:
            raise ValueError("n_jobs must be -1 or a positive integer.")
        if not splits or not scoring:
            raise ValueError("Provide CV splits and at least one scoring metric.")
        for name in scoring.values():
            get_scorer(name)
        self._estimator = estimator
        self._X = X
        self._y = y
        self._splits = splits
        self._scoring = dict(scoring)
        self._output = output
        self._n_jobs = n_jobs
        self._progress = progress
        self._progress_position = progress_position
        self.last_stats: dict[str, int] = {}
        signature = joblib_hash(
            (
                2,
                sklearn.__version__,
                np.__version__,
                estimator,
                X,
                y,
                splits,
                scoring,
            )
        )
        output.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(output / "evaluations.sqlite")
        try:
            self._connection.execute("CREATE TABLE IF NOT EXISTS context (signature TEXT NOT NULL)")
            previous = self._connection.execute("SELECT signature FROM context").fetchone()
            if previous is not None and previous[0] != signature:
                raise ValueError("Checkpoint belongs to different data, CV, estimator or scoring.")
            if previous is None:
                self._connection.execute("INSERT INTO context VALUES (?)", (signature,))
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS folds ("
                "candidate TEXT, fold INTEGER, result TEXT NOT NULL, "
                "PRIMARY KEY (candidate, fold))"
            )
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS refits ("
                "candidate TEXT PRIMARY KEY, signature TEXT, result TEXT)"
            )
            self._connection.commit()
        except BaseException:
            self._connection.close()
            raise

    def close(self) -> None:
        self._connection.close()

    def refit(
        self,
        parameters: dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        train: np.ndarray,
        test: np.ndarray,
    ) -> dict[str, Any]:
        key = joblib_hash(parameters)
        signature = joblib_hash((X, y, train, test))
        previous = self._connection.execute(
            "SELECT signature, result FROM refits WHERE candidate=?", (key,)
        ).fetchone()
        if previous is not None:
            if previous[0] != signature:
                raise ValueError("Refit checkpoint belongs to a different holdout split.")
            return json.loads(previous[1])
        result = _evaluate_fold(
            self._estimator, parameters, X, y, train, test, self._scoring, key, -1
        )
        self._connection.execute(
            "INSERT INTO refits VALUES (?, ?, ?)",
            (key, signature, json.dumps(result, allow_nan=False)),
        )
        self._connection.commit()
        self._log_warnings(result["warnings"], key, -1)
        return result

    def evaluate(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        keys = [joblib_hash(parameters) for parameters in candidates]
        unique = dict(zip(keys, candidates))
        completed = {}
        pending = []
        for key, parameters in unique.items():
            for fold, (train, test) in enumerate(self._splits):
                previous = self._connection.execute(
                    "SELECT result FROM folds WHERE candidate=? AND fold=?", (key, fold)
                ).fetchone()
                if previous is not None:
                    completed[key, fold] = json.loads(previous[0])
                else:
                    pending.append((key, parameters, fold, train, test))
        self.last_stats = {
            "proposals": len(candidates),
            "unique_candidates": len(unique),
            "cached_folds": len(completed),
            "evaluated_folds": 0,
        }
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always")
            try:
                with (
                    tqdm(
                        total=len(unique) * len(self._splits),
                        initial=len(completed),
                        desc="CV folds",
                        unit="fit",
                        disable=not self._progress,
                        position=self._progress_position,
                        leave=False,
                    ) as bar,
                    parallel_config(backend="loky", inner_max_num_threads=1),
                ):
                    results = Parallel(
                        n_jobs=self._n_jobs, return_as="generator_unordered", pre_dispatch="n_jobs"
                    )(
                        delayed(_evaluate_fold)(
                            self._estimator,
                            parameters,
                            self._X,
                            self._y,
                            train,
                            test,
                            self._scoring,
                            key,
                            fold,
                        )
                        for key, parameters, fold, train, test in pending
                    )
                    for result in results:
                        key, fold = result["candidate"], result["fold"]
                        self._connection.execute(
                            "INSERT INTO folds VALUES (?, ?, ?)",
                            (key, fold, json.dumps(result, allow_nan=False)),
                        )
                        self._connection.commit()
                        completed[key, fold] = result
                        self.last_stats["evaluated_folds"] += 1
                        self._log_warnings(result["warnings"], key, fold)
                        bar.update(1)
            finally:
                self._log_warnings(_warning_records(messages), None, None)
        summaries = {}
        for key in unique:
            folds = [completed[key, fold] for fold in range(len(self._splits))]
            summaries[key] = {
                "candidate": key,
                "folds": folds,
                "mean_test_scores": {
                    name: float(
                        np.asarray(
                            [fold["test_scores"][name] for fold in folds], dtype=float
                        ).mean()
                    )
                    for name in self._scoring
                },
                "std_test_scores": {
                    name: float(
                        np.asarray([fold["test_scores"][name] for fold in folds], dtype=float).std()
                    )
                    for name in self._scoring
                },
                "mean_train_scores": {
                    name: float(
                        np.asarray(
                            [fold["train_scores"][name] for fold in folds], dtype=float
                        ).mean()
                    )
                    for name in self._scoring
                },
                "std_train_scores": {
                    name: float(
                        np.asarray(
                            [fold["train_scores"][name] for fold in folds], dtype=float
                        ).std()
                    )
                    for name in self._scoring
                },
            }
        return [summaries[key] for key in keys]

    def _log_warnings(
        self, records: list[dict[str, Any]], candidate: Optional[str], fold: Optional[int]
    ) -> None:
        if records:
            with (self._output / "warnings.jsonl").open("a", encoding="utf-8") as stream:
                for warning in records:
                    stream.write(
                        json.dumps(
                            {"candidate": candidate, "fold": fold, **warning}, ensure_ascii=False
                        )
                        + "\n"
                    )
