from __future__ import annotations

import argparse
import json
import sqlite3
import traceback
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

import numpy as np
import sklearn
from joblib import Parallel, delayed, hash as joblib_hash, parallel_config
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import get_scorer
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


def _warning_records(messages: list[warnings.WarningMessage]) -> list[dict[str, Any]]:
    return [
        {"category": item.category.__name__, "message": str(item.message),
         "filename": item.filename, "lineno": item.lineno}
        for item in messages
    ]


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
    result = {
        "candidate": candidate, "fold": fold, "train_scores": {}, "test_scores": {},
        "fit_seconds": None, "score_seconds": None, "error": None,
    }
    start = perf_counter()
    with warnings.catch_warnings(record=True) as messages, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        try:
            model = clone(estimator).set_params(**clone(parameters, safe=False))
            model.fit(X[train], y[train])
            result["fit_seconds"] = perf_counter() - start
            start_score = perf_counter()
            for name, scorer_name in scoring.items():
                scorer = get_scorer(scorer_name)
                for split, indices in [("train", train), ("test", test)]:
                    value = float(scorer(model, X[indices], y[indices]))
                    result[f"{split}_scores"][name] = value if np.isfinite(value) else None
            result["score_seconds"] = perf_counter() - start_score
        except Exception:
            result["error"] = traceback.format_exc()
            result["train_scores"] = dict.fromkeys(scoring)
            result["test_scores"] = dict.fromkeys(scoring)
    result["warnings"] = _warning_records(messages)
    result["seconds"] = perf_counter() - start
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
        self.last_stats: dict[str, int] = {}
        signature = joblib_hash((
            1, sklearn.__version__, np.__version__, estimator, X, y, splits, scoring,
        ))
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
            self._connection.commit()
        except BaseException:
            self._connection.close()
            raise

    def close(self) -> None:
        self._connection.close()

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
                        total=len(unique) * len(self._splits), initial=len(completed),
                        desc="CV folds", unit="fit", disable=not self._progress,
                    ) as bar,
                    parallel_config(backend="loky", inner_max_num_threads=1),
                ):
                    results = Parallel(
                        n_jobs=self._n_jobs, return_as="generator_unordered", pre_dispatch="n_jobs"
                    )(
                        delayed(_evaluate_fold)(
                            self._estimator, parameters, self._X, self._y, train, test,
                            self._scoring, key, fold,
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
                    name: float(np.asarray(
                        [fold["test_scores"][name] for fold in folds], dtype=float
                    ).mean())
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
                    stream.write(json.dumps(
                        {"candidate": candidate, "fold": fold, **warning}, ensure_ascii=False
                    ) + "\n")


def _smoke(output: Path, n_jobs: int, progress: bool) -> None:
    from sklearn.datasets import load_iris
    from sklearn.model_selection import ParameterGrid, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X, y = load_iris(return_X_y=True)
    splits = list(StratifiedKFold(3, shuffle=True, random_state=42).split(X, y))
    evaluator = CandidateEvaluator(
        make_pipeline(StandardScaler(), SVC()), X, y, splits,
        {"accuracy": "accuracy", "f1_macro": "f1_macro"}, output, n_jobs, progress,
    )
    try:
        candidates = list(ParameterGrid({"svc__C": [0.1, 1.0, 10.0], "svc__gamma": [0.001, 0.01]}))
        results = evaluator.evaluate(candidates)
        print(json.dumps({
            **evaluator.last_stats,
            "best_cv_accuracy": max(row["mean_test_scores"]["accuracy"] for row in results),
        }, indent=2))
    finally:
        evaluator.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check resumable CV evaluation on a small workload."
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "output" / "runtime_smoke",
    )
    args = parser.parse_args()
    _smoke(args.output, args.jobs, not args.no_progress)
