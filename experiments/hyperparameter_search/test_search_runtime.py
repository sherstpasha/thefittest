import json
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold

from experiments.hyperparameter_search import search_runtime


class WarningRegressor(DummyRegressor):
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: object = None) -> "WarningRegressor":
        warnings.warn("Training warning for the journal", UserWarning)
        return super().fit(X, y, sample_weight=sample_weight)


def evaluator(output: Path, n_jobs: int = 1) -> search_runtime.CandidateEvaluator:
    X = np.arange(60, dtype=float).reshape(30, 2)
    y = np.sin(X[:, 0])
    return search_runtime.CandidateEvaluator(
        WarningRegressor(),
        X,
        y,
        list(KFold(3, shuffle=True, random_state=42).split(X)),
        {"score": "neg_root_mean_squared_error"},
        output,
        n_jobs=n_jobs,
        progress=False,
    )


def test_resume_after_interruption_and_silent_warning_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = search_runtime._evaluate_fold
    calls = 0

    def interrupt(*args: object, **kwargs: object) -> dict:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr(search_runtime, "_evaluate_fold", interrupt)
    candidates = [{"strategy": "mean"}, {"strategy": "median"}]
    first = evaluator(tmp_path)
    try:
        with pytest.raises(KeyboardInterrupt):
            first.evaluate(candidates)
    finally:
        first.close()
    with sqlite3.connect(tmp_path / "evaluations.sqlite") as connection:
        assert connection.execute("SELECT count(*) FROM folds").fetchone()[0] == 2

    monkeypatch.setattr(search_runtime, "_evaluate_fold", original)
    resumed = evaluator(tmp_path, n_jobs=2)
    try:
        with warnings.catch_warnings(record=True) as visible:
            results = resumed.evaluate(candidates)
        assert not visible
        assert resumed.last_stats["cached_folds"] == 2
        assert resumed.last_stats["evaluated_folds"] == 4
        assert all(np.isfinite(row["mean_test_scores"]["score"]) for row in results)
        assert all(fold["warnings"] for row in results for fold in row["folds"])
        assert all(fold["error"] is None for row in results for fold in row["folds"])
        assert resumed.evaluate(candidates) == results
        assert resumed.last_stats["cached_folds"] == 6
        assert resumed.last_stats["evaluated_folds"] == 0
        X = np.arange(60, dtype=float).reshape(30, 2)
        y = np.sin(X[:, 0])
        first_refit = resumed.refit(candidates[0], X, y, np.arange(20), np.arange(20, 30))
        assert first_refit["model_fits"]["completed"]
        assert resumed.refit(candidates[0], X, y, np.arange(20), np.arange(20, 30)) == first_refit
    finally:
        resumed.close()
    records = [json.loads(line) for line in (tmp_path / "warnings.jsonl").read_text().splitlines()]
    assert sum(r["message"] == "Training warning for the journal" for r in records) == 7


def test_cache_deduplicates_candidates_and_rejects_changed_context(tmp_path: Path) -> None:
    current = evaluator(tmp_path)
    try:
        results = current.evaluate([{"strategy": "mean"}, {"strategy": "mean"}])
        assert results[0] == results[1]
        assert current.last_stats["evaluated_folds"] == 3
        X = np.arange(60, dtype=float).reshape(30, 2)
        with pytest.raises(ValueError, match="different data"):
            search_runtime.CandidateEvaluator(
                WarningRegressor(),
                X,
                np.zeros(30),
                list(KFold(3).split(X)),
                {"score": "neg_root_mean_squared_error"},
                tmp_path,
            )
    finally:
        current.close()


def test_fit_errors_are_saved(tmp_path: Path) -> None:
    current = evaluator(tmp_path)
    try:
        result = current.evaluate([{"strategy": "invalid"}])[0]
        assert np.isnan(result["mean_test_scores"]["score"])
        assert all(fold["error"] for fold in result["folds"])
        current.evaluate([{"strategy": "invalid"}])
        assert current.last_stats["evaluated_folds"] == 0
    finally:
        current.close()
