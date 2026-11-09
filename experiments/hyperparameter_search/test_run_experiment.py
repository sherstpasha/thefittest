import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    ParameterGrid,
    StratifiedKFold,
    train_test_split,
)

from experiments.hyperparameter_search.experiment_setup import (
    TaskData,
    describe,
    prepare_task,
    search_space,
)
from experiments.hyperparameter_search.plan_search_experiment import SearchPlan
from experiments.hyperparameter_search.run_experiment import Experiment, directory_lock
from experiments.hyperparameter_search.search_execution import SearchRun
from experiments.hyperparameter_search.search_runtime import FeatureGroupCV


@pytest.mark.parametrize("classification", [True, False])
def test_full_spaces_and_serialization(classification: bool) -> None:
    expected = {
        "svm": 393216,
        "forest": 829440 if classification else 414720,
        "boosting": 829440,
        "mlp": 442368,
        "stacking": 167936,
    }
    for family, count in expected.items():
        estimator, grid = search_space(family, 4, classification)
        parameters = ParameterGrid(grid)
        assert len(parameters) == count
        for i in [0, len(parameters) // 2, len(parameters) - 1]:
            from sklearn.base import clone

            clone(estimator).set_params(**parameters[i])
        json.dumps(describe({"estimator": estimator, "grid": grid}), allow_nan=False)
    estimator, grid = search_space("svm", 1, classification)
    assert len(grid) == 1
    assert len(ParameterGrid(grid)) == 32768


def test_all_30_task_snapshots(tmp_path: Path) -> None:
    for task in SearchPlan((42,)).tasks:
        data = prepare_task(task, tmp_path, 120, 0.0, 42, 120)
        assert np.isfinite(data.X).all() and np.isfinite(data.y).all()
        assert not set(data.train) & set(data.test)
        for train, test in data.splits:
            assert not set(train) & set(test)
        restored = prepare_task(task, tmp_path, 120, 0.0, 42, 120)
        np.testing.assert_array_equal(data.X, restored.X)
        np.testing.assert_array_equal(data.y, restored.y)


def test_identical_features_stay_in_one_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.hyperparameter_search import experiment_setup

    class ConflictingData:
        def get_X(self) -> np.ndarray:
            return np.repeat(np.arange(60, dtype=float).reshape(-1, 1), 2, axis=0)

        def get_y(self) -> np.ndarray:
            return np.tile([0, 1], 60)

    monkeypatch.setattr(
        experiment_setup.benchmarks, "ConflictingData", ConflictingData, raising=False
    )
    data = prepare_task(
        {"id": "ConflictingData", "kind": "classification", "source": "test"},
        tmp_path,
        120,
        0,
        42,
        0,
    )
    assert not set(data.X[data.train, 0]) & set(data.X[data.test, 0])
    X, y = data.X[data.train], data.y[data.train]
    for train, test in [*data.splits, *FeatureGroupCV(True).split(X, y)]:
        assert not set(X[train, 0]) & set(X[test, 0])


def _data(classification: bool) -> TaskData:
    X, y = load_iris(return_X_y=True) if classification else load_diabetes(return_X_y=True)
    train, test = train_test_split(
        np.arange(len(y)), test_size=0.25, random_state=42, stratify=y if classification else None
    )
    cv = StratifiedKFold if classification else KFold
    return TaskData(
        X,
        y,
        train,
        test,
        list(cv(3, shuffle=True, random_state=42).split(X[train], y[train])),
        classification,
    )


@pytest.mark.parametrize("classification", [True, False])
def test_grid_results_match_sklearn(tmp_path: Path, classification: bool) -> None:
    data = _data(classification)
    estimator, grid = search_space("svm", data.X.shape[1], classification, smoke=True)
    job = {"id": "grid", "method": "Grid", "task": "example", "family": "svm"}
    run = SearchRun(job, estimator, grid, data, None, tmp_path, 2, False, 4)
    try:
        result = run.execute()
    finally:
        run.close("ok")
    scoring = "accuracy" if classification else "neg_root_mean_squared_error"
    reference = GridSearchCV(estimator, grid, scoring=scoring, cv=data.splits, n_jobs=1).fit(
        data.X[data.train], data.y[data.train]
    )
    sign = 1 if classification else -1
    assert result["best_cv"] == pytest.approx(sign * reference.best_score_)
    assert result["test_score"] == pytest.approx(
        sign * reference.score(data.X[data.test], data.y[data.test])
    )
    assert result["unique_candidates"] == len(ParameterGrid(grid))
    assert result["cv_fits"] == 3 * result["unique_candidates"]


@pytest.mark.parametrize("method", ["GA", "SHAGA", "PDPGA", "SelfCGA", "SelfCSHAGA", "PDPSHAGA"])
@pytest.mark.parametrize("classification", [True, False])
def test_evolution_resume_replays_without_retraining(
    tmp_path: Path, method: str, classification: bool
) -> None:
    data = _data(classification)
    estimator, grid = search_space("svm", data.X.shape[1], classification, smoke=True)
    configuration = next(c for c in SearchPlan((42,)).configurations if c["method"] == method)
    job = {
        "id": "ga",
        "method": method,
        "task": "example",
        "family": "svm",
        "nominal_budget": 50,
        "pop_size": 10,
        "iters": 5,
        "seed": 42,
    }
    first = SearchRun(job, estimator, grid, data, configuration, tmp_path, 1, False, 4)
    original = first._generation

    def stop(optimizer: object) -> None:
        original(optimizer)
        raise KeyboardInterrupt

    first._generation = stop
    try:
        with pytest.raises(KeyboardInterrupt):
            first.execute()
    finally:
        first.close("interrupted")
    resumed = SearchRun(job, estimator, grid, data, configuration, tmp_path, 2, False, 4)
    try:
        result = resumed.execute()
    finally:
        resumed.close("ok")
    clean = SearchRun(job, estimator, grid, data, configuration, tmp_path / "clean", 1, False, 4)
    try:
        expected = clean.execute()
    finally:
        clean.close("ok")
    assert result["best_cv"] == expected["best_cv"]
    assert result["proposals"] == expected["proposals"] == 50
    assert result["unique_candidates"] == expected["unique_candidates"]
    assert result["cv_fits"] == expected["cv_fits"]
    assert result["best_parameters"] == expected["best_parameters"]


def test_experiment_one_command_and_skip_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = dict(
        output=tmp_path,
        n_jobs=2,
        seeds=(42,),
        tasks=["IrisDataset", "Diabetes"],
        families=["svm"],
        methods=["GA", "Random", "Grid"],
        budgets=(10,),
        samples=120,
        noise=0,
        data_seed=42,
        max_samples=120,
        smoke=True,
        progress=False,
        batch_size=4,
    )
    assert Experiment(**settings).run() == {"ok": 6}
    with sqlite3.connect(tmp_path / "results.sqlite") as connection:
        results = [
            json.loads(row[0])
            for row in connection.execute("SELECT result FROM runs ORDER BY rowid")
        ]
    for task in settings["tasks"]:
        assert [r["method"] for r in results if r["task"] == task] == ["Grid", "GA", "Random"]
    assert len({r["worker_pid"] for r in results}) == 2
    assert all(r["cv_n_jobs"] == 1 for r in results)
    for result in results:
        assert result["logical_cv_fits"] == 3 * result["unique_candidates"]
        assert result["cv_fits"] == (result["logical_cv_fits"] if result["method"] == "Grid" else 0)
        assert result["cache_hits"] == result["proposals"] - result["unique_candidates"]
        assert result["cv_seconds_estimate"] > 0

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Completed search must not be rerun")

    monkeypatch.setattr(SearchRun, "execute", forbidden)
    settings["n_jobs"] = 1
    assert Experiment(**settings).run() == {"ok": 6}
    assert len(list((tmp_path / "runs").glob("*/result.json"))) == 6
    with directory_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another process"):
            with directory_lock(tmp_path):
                pass
    settings["noise"] = 0.1
    with pytest.raises(ValueError, match="settings"):
        Experiment(**settings).run()


@pytest.mark.parametrize("classification", [True, False])
@pytest.mark.parametrize(
    "method", ["GA", "SHAGA", "PDPGA", "SelfCGA", "SelfCSHAGA", "PDPSHAGA", "Random"]
)
def test_shared_cache_search_matches_independent_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, classification: bool, method: str
) -> None:
    import joblib
    from experiments.hyperparameter_search import grid_cache
    from experiments.hyperparameter_search.search_execution import _unpack

    data = _data(classification)
    estimator, grid = search_space("svm", data.X.shape[1], classification, smoke=True)
    configuration = next(
        (c for c in SearchPlan((42,)).configurations if c["method"] == method), None
    )
    job = dict(id="compare", method=method, seed=42, pop_size=10, iters=5, nominal_budget=50)
    direct = SearchRun(job, estimator, grid, data, configuration, tmp_path / "direct", 1, False, 4)
    try:
        expected = direct.execute()
        cache = grid_cache.GridEvaluationCache(
            estimator,
            grid,
            data.X[data.train],
            data.y[data.train],
            data.splits,
            direct.evaluator._scoring,
            tmp_path / "cache",
            n_jobs=2,
        )
    finally:
        direct.close("ok")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("A search must only read the prepared CV cache")

    try:
        cache.populate()
        monkeypatch.setattr(grid_cache, "_evaluate_fold", forbidden)
        search = SearchRun(
            job, estimator, grid, data, configuration, tmp_path / "cached", 1, False, 4, cache=cache
        )
        try:
            actual = search.execute()
        finally:
            search.close("ok")
        for field in (
            "best_cv",
            "test_score",
            "best_parameters",
            "unique_candidates",
            "proposals",
            "coverage",
            "cache_hits",
        ):
            assert actual[field] == expected[field]
        assert actual["cv_fits"] == 0
        assert actual["logical_cv_fits"] == expected["cv_fits"]
        traces = []
        for directory in ("direct", "cached"):
            with sqlite3.connect(tmp_path / directory / "search.sqlite") as connection:
                traces.append(
                    [
                        _unpack(row[0])
                        for row in connection.execute(
                            "SELECT payload FROM generations WHERE kind='proposals' ORDER BY batch"
                        )
                    ]
                )
        for reference, cached in zip(*traces):
            np.testing.assert_array_equal(reference["scores"], cached["scores"])
            np.testing.assert_array_equal(reference["fitness"], cached["fitness"])
            np.testing.assert_array_equal(reference["genotypes"], cached["genotypes"])
            assert reference["candidate_ids"] == [
                joblib.hash(cache.grid[int(index)]) for index in cached["candidate_ids"]
            ]
    finally:
        cache.close()


def test_shared_search_resumes_its_own_history(tmp_path: Path) -> None:
    from experiments.hyperparameter_search.grid_cache import GridEvaluationCache

    data = _data(True)
    estimator, grid = search_space("svm", 4, True, smoke=True)
    configuration = next(c for c in SearchPlan((42,)).configurations if c["method"] == "SelfCSHAGA")
    job = dict(id="resume", method="SelfCSHAGA", seed=42, pop_size=10, iters=5, nominal_budget=50)
    cache = GridEvaluationCache(
        estimator,
        grid,
        data.X[data.train],
        data.y[data.train],
        data.splits,
        {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro"},
        tmp_path / "cache",
    )

    def stop(optimizer: object) -> None:
        raise KeyboardInterrupt

    try:
        cache.populate()
        first = SearchRun(
            job, estimator, grid, data, configuration, tmp_path / "run", 1, False, 4, cache=cache
        )
        first._generation = stop
        try:
            with pytest.raises(KeyboardInterrupt):
                first.execute()
        finally:
            first.close("interrupted")
        results = []
        for directory in ("run", "clean"):
            run = SearchRun(
                job,
                estimator,
                grid,
                data,
                configuration,
                tmp_path / directory,
                1,
                False,
                4,
                cache=cache,
            )
            try:
                results.append(run.execute())
            finally:
                run.close("ok")
        for field in (
            "best_cv",
            "best_parameters",
            "proposals",
            "unique_candidates",
            "cache_hits",
            "cv_seconds_estimate",
        ):
            assert results[0][field] == results[1][field]
        assert results[0]["proposals"] == 50
        assert results[0]["coverage"] < 1
    finally:
        cache.close()


def test_case_parallelism_preserves_results(tmp_path: Path) -> None:
    results = []
    for workers in (1, 2):
        output = tmp_path / str(workers)
        experiment = Experiment(
            output=output,
            n_jobs=workers,
            seeds=(42,),
            tasks=["IrisDataset", "Diabetes"],
            families=["svm"],
            methods=["Random"],
            budgets=(10,),
            samples=120,
            noise=0,
            data_seed=42,
            max_samples=120,
            smoke=True,
            progress=False,
            batch_size=4,
        )
        assert experiment.run() == {"ok": 4}
        with sqlite3.connect(output / "results.sqlite") as connection:
            rows = {
                json.loads(row[0])["id"]: json.loads(row[0])
                for row in connection.execute("SELECT result FROM runs")
            }
        assert len({row["worker_pid"] for row in rows.values()}) == 2
        assert all(row["cv_n_jobs"] == 1 for row in rows.values())
        log = (output / "run.log").read_text()
        assert f"active={workers}" in log
        results.append(rows)
    assert results[0].keys() == results[1].keys()
    for job_id in results[0]:
        for field in (
            "best_cv",
            "test_score",
            "best_parameters",
            "unique_candidates",
            "proposals",
            "coverage",
            "cache_hits",
            "cv_fits",
            "logical_cv_fits",
        ):
            assert results[0][job_id][field] == results[1][job_id][field]
