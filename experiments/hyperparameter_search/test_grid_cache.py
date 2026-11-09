from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from experiments.hyperparameter_search import grid_cache
from experiments.hyperparameter_search.search_runtime import CandidateEvaluator
from thefittest.optimizers import GeneticAlgorithm, PDPGA, PDPSHAGA, SelfCGA, SelfCSHAGA, SHAGA


def settings(classification: bool) -> dict[str, Any]:
    X, y = load_iris(return_X_y=True) if classification else load_diabetes(return_X_y=True)
    cv = StratifiedKFold if classification else KFold
    model = DecisionTreeClassifier if classification else DecisionTreeRegressor
    return dict(
        estimator=model(random_state=0),
        param_grid=[{"max_depth": [1, 2, 3], "min_samples_leaf": [1, 3]}],
        X=X,
        y=y,
        splits=list(cv(3, shuffle=True, random_state=42).split(X, y)),
        scoring={"primary": "accuracy" if classification else "neg_root_mean_squared_error"},
    )


@pytest.mark.parametrize("classification", [True, False])
@pytest.mark.parametrize(
    "optimizer_type", [GeneticAlgorithm, PDPGA, PDPSHAGA, SelfCGA, SelfCSHAGA, SHAGA]
)
def test_grid_then_cached_search_matches_direct_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, classification: bool, optimizer_type: type
) -> None:
    config = settings(classification)
    cache = grid_cache.GridEvaluationCache(**config, output=tmp_path / "shared", n_jobs=2)
    direct_config = {key: value for key, value in config.items() if key != "param_grid"}
    direct = CandidateEvaluator(**direct_config, output=tmp_path / "direct", progress=False)
    space = ParameterGrid(config["param_grid"])
    traces: list[list[np.ndarray]] = [[], []]

    def decode(population: np.ndarray) -> np.ndarray:
        return ((population @ np.array([4, 2, 1], dtype=np.int64)) % len(space)).astype(np.int64)

    def reference(indices: np.ndarray) -> np.ndarray:
        rows = direct.evaluate([space[int(index)] for index in indices])
        scores = np.array([row["mean_test_scores"]["primary"] for row in rows])
        traces[0].append(scores.copy())
        return 2 + (2 / np.pi) * np.arctan(scores)

    def cached(indices: np.ndarray) -> np.ndarray:
        rows = cache.evaluate(indices)
        assert cache.last_evaluated_folds == 0
        scores = rows["test_scores"].mean(axis=1)[:, 0]
        traces[1].append(scores.copy())
        return 2 + (2 / np.pi) * np.arctan(scores)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("A populated cache must not train another model")

    try:
        cache.populate(batch_size=2)
        expected = direct.evaluate(list(space))
        rows = cache.evaluate(range(len(space)))
        for split in ("train", "test"):
            actual = rows[f"{split}_scores"][:, :, 0]
            target = [[fold[f"{split}_scores"]["primary"] for fold in r["folds"]] for r in expected]
            np.testing.assert_array_equal(actual, target)
        first = optimizer_type(reference, 5, 10, 3, genotype_to_phenotype=decode, random_state=42)
        first.fit()
        monkeypatch.setattr(grid_cache, "_evaluate_fold", forbidden)
        second = optimizer_type(cached, 5, 10, 3, genotype_to_phenotype=decode, random_state=42)
        second.fit()
        np.testing.assert_array_equal(traces[0], traces[1])
        np.testing.assert_array_equal(
            first.get_fittest()["genotype"], second.get_fittest()["genotype"]
        )
        assert first.get_calls() == second.get_calls() == 50
        assert np.all(rows["fit_seconds"] >= 0)
        assert np.all(rows["model_fits"] == 1)
    finally:
        direct.close()
        cache.close()


def test_interrupted_population_resumes_completed_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = settings(True)
    original = grid_cache._evaluate_fold
    calls = 0

    def interrupt(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    first = grid_cache.GridEvaluationCache(**config, output=tmp_path)
    monkeypatch.setattr(grid_cache, "_evaluate_fold", interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            first.evaluate([0, 1])
        assert first.last_evaluated_folds == 2
    finally:
        first.close()
    monkeypatch.setattr(grid_cache, "_evaluate_fold", original)
    resumed = grid_cache.GridEvaluationCache(**config, output=tmp_path, n_jobs=2)
    try:
        records = resumed.evaluate([1, 0, 1])
        assert resumed.last_evaluated_folds == 4
        np.testing.assert_array_equal(records[0], records[2])
        assert np.isfinite(records["test_scores"]).all()
    finally:
        resumed.close()
    config["y"] = np.zeros_like(config["y"])
    with pytest.raises(ValueError, match="different"):
        grid_cache.GridEvaluationCache(**config, output=tmp_path)


def test_failed_folds_are_cached_with_diagnostics(tmp_path: Path) -> None:
    import sqlite3

    config = settings(True)
    config["param_grid"] = [{"max_depth": [0]}]
    cache = grid_cache.GridEvaluationCache(**config, output=tmp_path)
    try:
        records = cache.evaluate([0, 0])
        assert cache.last_evaluated_folds == 3
        assert np.isnan(records["test_scores"]).all()
        cache.evaluate([0])
        assert cache.last_evaluated_folds == 0
        with sqlite3.connect(tmp_path / "diagnostics.sqlite") as connection:
            errors = connection.execute("SELECT error FROM diagnostics").fetchall()
        assert len(errors) == 3 and all(error for (error,) in errors)
        for indices in ([-1], [1], [0.5]):
            with pytest.raises(ValueError):
                cache.evaluate(indices)
    finally:
        cache.close()


@pytest.mark.parametrize("classification", [True, False])
def test_index_encoder_preserves_conditional_grid_mapping(classification: bool) -> None:
    import joblib
    from experiments.hyperparameter_search.experiment_setup import search_space
    from thefittest.model_selection._search import _ParameterGridEncoder

    for family in ("svm", "forest", "boosting", "mlp", "stacking"):
        for p in (1, 3, 4):
            _, grid = search_space(family, p, classification)
            reference = _ParameterGridEncoder(grid)
            encoder = grid_cache.GridIndexEncoder(grid)
            population = np.random.RandomState(42).randint(0, 2, (32, encoder.str_len))
            indices = encoder.transform(population)
            assert [joblib.hash(row) for row in reference.transform(population)] == [
                joblib.hash(encoder_grid)
                for encoder_grid in (ParameterGrid(grid)[int(i)] for i in indices)
            ]
