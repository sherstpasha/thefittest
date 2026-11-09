import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ..model_selection import GeneticSearchCV
from ..model_selection._search import _ParameterGridEncoder
from ..model_selection._search import _CandidateEvaluator
from ..model_selection._search import _CachedCV
from ..optimizers import (
    DifferentialEvolution,
    GeneticAlgorithm,
    GeneticProgramming,
    NSGAII,
    PDPGA,
    PDPSHAGA,
    SelfCGA,
    SelfCSHAGA,
    SHAGA,
)


@pytest.mark.parametrize(
    "optimizer", [GeneticAlgorithm, SelfCGA, PDPGA, SHAGA, SelfCSHAGA, PDPSHAGA]
)
def test_optimizer_settings_survive_clone(optimizer: type) -> None:
    arguments = {"iters": 3, "pop_size": 4}
    search = GeneticSearchCV(
        DecisionTreeClassifier(),
        {"max_depth": [1, 2]},
        optimizer=optimizer,
        optimizer_args=arguments,
        random_state=42,
    )
    copied = clone(search)
    copied._validate_params()
    assert copied.optimizer is optimizer
    assert copied.optimizer_args == arguments
    assert copied.random_state == 42
    assert search.optimizer_args is arguments


@pytest.mark.parametrize("optimizer", [DifferentialEvolution, GeneticProgramming, NSGAII, None])
def test_rejects_unsupported_optimizer_after_set_params(optimizer: object) -> None:
    search = GeneticSearchCV(DecisionTreeClassifier(), {"max_depth": [1, 2]})
    search.set_params(optimizer=optimizer)
    with pytest.raises(ValueError, match="single-objective genetic algorithm"):
        search._validate_params()


@pytest.mark.parametrize(
    "parameter",
    [
        "fitness_function",
        "fitness_function_args",
        "genotype_to_phenotype",
        "genotype_to_phenotype_args",
        "str_len",
        "init_population",
        "minimization",
        "n_jobs",
        "random_state",
    ],
)
def test_rejects_reserved_optimizer_arguments(parameter: str) -> None:
    search = GeneticSearchCV(DecisionTreeClassifier(), {"max_depth": [1, 2]})
    search.set_params(optimizer_args={parameter: None})
    with pytest.raises(AssertionError, match=parameter):
        search._validate_params()


def test_grid_encoder_decodes_indices_and_wraps_unused_codes() -> None:
    encoder = _ParameterGridEncoder(
        {"depth": [3, 5, None], "criterion": ["gini", "entropy"], "rate": [0.01]}
    )
    assert encoder.str_len == 3
    population = np.array([[0, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=np.int8)
    assert encoder.transform(population).tolist() == [
        {"criterion": "gini", "depth": 3, "rate": 0.01},
        {"criterion": "entropy", "depth": None, "rate": 0.01},
        {"criterion": "gini", "depth": 3, "rate": 0.01},
    ]


def test_grid_encoder_accepts_numpy_choices() -> None:
    values = np.linspace(0.1, 0.9, 3)
    encoder = _ParameterGridEncoder({"rate": values})
    decoded = encoder.transform(np.array([[0, 1]], dtype=np.int8))
    assert decoded[0]["rate"] == values[1]


@pytest.mark.parametrize("grid", [{}, {"depth": [None]}])
def test_grid_encoder_constant_space_needs_no_bits(grid: dict) -> None:
    encoder = _ParameterGridEncoder(grid)
    assert encoder.str_len == 0
    expected = {name: values[0] for name, values in grid.items()}
    assert encoder.transform(np.empty((2, 0), dtype=np.int8)).tolist() == [expected, expected]


@pytest.mark.parametrize("grid", [{"depth": []}, {"depth": 3}, {"depth": "auto"}])
def test_grid_encoder_rejects_invalid_choices(grid: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        _ParameterGridEncoder(grid)


@pytest.mark.parametrize("population", [np.array([0, 1]), np.zeros((1, 2)), np.array([[2]])])
def test_grid_encoder_rejects_invalid_population(population: np.ndarray) -> None:
    encoder = _ParameterGridEncoder({"depth": [1, 2]})
    with pytest.raises(ValueError):
        encoder.transform(population)


def test_grid_encoder_selects_conditional_space_and_ignores_inactive_bits() -> None:
    encoder = _ParameterGridEncoder(
        [
            {"kernel": ["linear"], "C": [1, 10]},
            {"kernel": ["rbf"], "C": [1, 10], "gamma": [0.01, 0.1]},
        ]
    )
    assert encoder.str_len == 4
    population = np.array(
        [[0, 1, 0, 0], [0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 1, 0]], dtype=np.int8
    )
    assert encoder.transform(population).tolist() == [
        {"C": 10, "kernel": "linear"},
        {"C": 10, "kernel": "linear"},
        {"C": 10, "gamma": 0.01, "kernel": "rbf"},
        {"C": 10, "gamma": 0.01, "kernel": "rbf"},
    ]


def test_grid_encoder_wraps_selector_and_accepts_empty_branch() -> None:
    encoder = _ParameterGridEncoder([{}, {"depth": [1]}, {"depth": [2]}])
    assert encoder.str_len == 2
    population = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)
    assert encoder.transform(population).tolist() == [{}, {"depth": 1}, {"depth": 2}, {}]


def test_grid_encoder_single_grid_list_matches_dict() -> None:
    grid = {"depth": [1, 2, 3]}
    population = np.array([[0, 1], [1, 1]], dtype=np.int8)
    assert _ParameterGridEncoder([grid]).transform(population).tolist() == (
        _ParameterGridEncoder(grid).transform(population).tolist()
    )


@pytest.mark.parametrize("grid", [[], [{"depth": []}], [{"depth": [1]}, None]])
def test_grid_encoder_rejects_invalid_grid_list(grid: list) -> None:
    with pytest.raises((TypeError, ValueError)):
        _ParameterGridEncoder(grid)


@pytest.mark.parametrize("refit", [True, False])
def test_search_matches_grid_search_on_evaluated_candidates(refit: bool) -> None:
    X, y = load_iris(return_X_y=True)
    estimator = Pipeline([("model", DecisionTreeClassifier(random_state=0))])
    search = GeneticSearchCV(
        estimator,
        [
            {"model__max_depth": [1, 2]},
            {"model__max_depth": [3], "model__criterion": ["entropy"]},
        ],
        optimizer_args={"iters": 2, "pop_size": 6},
        random_state=42,
        scoring="accuracy",
        cv=3,
        refit=refit,
        return_train_score=True,
    )
    assert search.fit(X, y) is search
    candidates = search.cv_results_["params"]
    reference = GridSearchCV(
        estimator,
        [{name: [value] for name, value in parameters.items()} for parameters in candidates],
        scoring="accuracy",
        cv=3,
        refit=refit,
        return_train_score=True,
    ).fit(X, y)
    np.testing.assert_allclose(
        search.cv_results_["mean_test_score"], reference.cv_results_["mean_test_score"]
    )
    np.testing.assert_allclose(
        search.cv_results_["mean_train_score"], reference.cv_results_["mean_train_score"]
    )
    assert search.best_params_ == reference.best_params_
    assert search.best_score_ == reference.best_score_
    if refit:
        np.testing.assert_array_equal(search.predict(X), reference.predict(X))
    else:
        assert not hasattr(search, "predict")


def test_candidate_evaluator_reuses_decoded_parameters() -> None:
    batches = []

    def evaluate_candidates(candidates: list[dict]) -> dict:
        batches.append(candidates)
        return {"mean_test_score": [0.25, 0.75][: len(candidates)]}

    evaluator = _CandidateEvaluator(evaluate_candidates)
    population = np.array([{"depth": 1}, {"depth": 2}, {"depth": 1}], dtype=object)
    expected = 2 + (2 / np.pi) * np.arctan([0.25, 0.75, 0.25])
    np.testing.assert_allclose(evaluator.evaluate(population), expected)
    np.testing.assert_allclose(evaluator.evaluate(population[::-1]), expected)
    assert batches == [[{"depth": 1}, {"depth": 2}]]


@pytest.mark.parametrize("grid", [{}, {"max_depth": [2]}])
def test_search_constant_grid(grid: dict) -> None:
    X, y = load_iris(return_X_y=True)
    search = GeneticSearchCV(DecisionTreeClassifier(random_state=0), grid, cv=2).fit(X, y)
    assert len(search.cv_results_["params"]) == 1
    assert search.predict(X).shape == y.shape


@pytest.mark.parametrize(
    "results, message",
    [
        ({"mean_test_score": [np.inf]}, "infinite"),
        ({"mean_test_accuracy": [0.5]}, "Set fitness_metric"),
    ],
)
def test_candidate_evaluator_rejects_unsupported_scores(results: dict, message: str) -> None:
    def evaluate_candidates(candidates: list[dict]) -> dict:
        return results

    evaluator = _CandidateEvaluator(evaluate_candidates)
    with pytest.raises(ValueError, match=message):
        evaluator.evaluate(np.array([{"depth": 1}], dtype=object))


def test_nan_scores_have_worse_fitness_and_are_cached() -> None:
    calls = []

    def evaluate_candidates(candidates: list[dict]) -> dict:
        calls.append(candidates)
        return {"mean_test_score": [np.nan, -1e100, -10.0, 0.0, 10.0]}

    evaluator = _CandidateEvaluator(evaluate_candidates)
    population = np.array([{"index": index} for index in range(5)], dtype=object)
    fitness = evaluator.evaluate(population)
    assert np.all(np.isfinite(fitness))
    assert np.all(np.diff(fitness) > 0)
    np.testing.assert_array_equal(evaluator.evaluate(population), fitness)
    assert len(calls) == 1


def test_cached_cv_reuses_random_splits() -> None:
    X = np.arange(40).reshape(20, 2)
    random_state = np.random.RandomState(42)
    cached = _CachedCV(ShuffleSplit(n_splits=3, random_state=random_state))
    first = list(cached.split(X))
    state = random_state.get_state()
    second = list(cached.split(X))
    for (train, test), (other_train, other_test) in zip(first, second):
        np.testing.assert_array_equal(train, other_train)
        np.testing.assert_array_equal(test, other_test)
    np.testing.assert_array_equal(state[1], random_state.get_state()[1])
    assert state[2:] == random_state.get_state()[2:]


def test_search_preserves_nan_in_results_and_refits_valid_candidate() -> None:
    X, y = load_iris(return_X_y=True)
    search = GeneticSearchCV(
        DecisionTreeClassifier(random_state=0),
        {"max_depth": [0, 2]},
        optimizer_args={"iters": 2, "pop_size": 6},
        random_state=42,
        cv=2,
    )
    with pytest.warns((UserWarning, FitFailedWarning)):
        search.fit(X, y)
    assert np.isnan(search.cv_results_["mean_test_score"]).sum() == 1
    assert search.best_params_ == {"max_depth": 2}
    assert search.predict(X).shape == y.shape


def test_search_restores_cv_after_failure() -> None:
    X, y = load_iris(return_X_y=True)
    cv = ShuffleSplit(n_splits=2, random_state=42)
    search = GeneticSearchCV(
        DecisionTreeClassifier(), {"max_depth": [0]}, cv=cv, error_score="raise"
    )
    with pytest.raises(ValueError):
        search.fit(X, y)
    assert search.cv is cv


def test_search_passes_groups_and_refreshes_cv_cache_between_fits() -> None:
    X, y = load_iris(return_X_y=True)
    groups = np.arange(len(y)) % 3
    cv = GroupKFold(n_splits=3)
    search = GeneticSearchCV(DecisionTreeClassifier(random_state=0), {}, cv=cv)
    search.fit(X, y, groups=groups)
    assert search.cv is cv
    first_scores = search.cv_results_["mean_test_score"].copy()
    search.fit(X, y, groups=groups)
    np.testing.assert_array_equal(search.cv_results_["mean_test_score"], first_scores)


def test_search_calls_random_splitter_once_per_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y = load_iris(return_X_y=True)
    cv = ShuffleSplit(n_splits=2, random_state=np.random.RandomState(42))
    original_split = cv.split
    calls = []

    def split(X: np.ndarray, y: np.ndarray, **params: object) -> object:
        calls.append(1)
        return original_split(X, y, **params)

    monkeypatch.setattr(cv, "split", split)
    search = GeneticSearchCV(
        DecisionTreeClassifier(random_state=0),
        {"max_depth": list(range(1, 33))},
        optimizer_args={"iters": 3, "pop_size": 6},
        random_state=42,
        cv=cv,
    )
    search.fit(X, y)
    assert len(search.cv_results_["params"]) > 6
    assert len(calls) == 1
    search.fit(X, y)
    assert len(calls) == 2
    assert search.cv is cv


@pytest.mark.parametrize("scoring_kind", ["dict", "list", "callable"])
@pytest.mark.parametrize("refit_kind", ["metric", "disabled", "callable"])
def test_search_multiple_metrics(scoring_kind: str, refit_kind: str) -> None:
    X, y = load_iris(return_X_y=True)

    def scorer(estimator: DecisionTreeClassifier, X: np.ndarray, y: np.ndarray) -> dict:
        prediction = estimator.predict(X)
        return {
            "accuracy": accuracy_score(y, prediction),
            "balanced_accuracy": balanced_accuracy_score(y, prediction),
        }

    def select_best(results: dict) -> int:
        return int(np.argmax(results["mean_test_accuracy"]))

    scoring = {
        "dict": {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy"},
        "list": ["accuracy", "balanced_accuracy"],
        "callable": scorer,
    }[scoring_kind]
    refit = {"metric": "accuracy", "disabled": False, "callable": select_best}[refit_kind]
    fitness_metric = None if refit_kind == "metric" else "accuracy"
    estimator = DecisionTreeClassifier(random_state=0)
    search = GeneticSearchCV(
        estimator,
        {"max_depth": [1, 2, 3]},
        scoring=scoring,
        refit=refit,
        fitness_metric=fitness_metric,
        optimizer_args={"iters": 2, "pop_size": 6},
        random_state=42,
        cv=3,
    ).fit(X, y)
    reference = GridSearchCV(
        estimator,
        [{"max_depth": [p["max_depth"]]} for p in search.cv_results_["params"]],
        scoring=scoring,
        refit=refit,
        cv=3,
    ).fit(X, y)
    assert search.multimetric_
    for name in ["accuracy", "balanced_accuracy"]:
        np.testing.assert_allclose(
            search.cv_results_[f"mean_test_{name}"], reference.cv_results_[f"mean_test_{name}"]
        )
    assert clone(search).fitness_metric == fitness_metric
    if refit_kind != "disabled":
        assert search.best_params_ == reference.best_params_
        np.testing.assert_array_equal(search.predict(X), reference.predict(X))
    else:
        assert not hasattr(search, "best_estimator_")


def test_candidate_evaluator_uses_selected_metric() -> None:
    def evaluate_candidates(candidates: list[dict]) -> dict:
        return {"mean_test_first": [0.1, 0.9], "mean_test_second": [0.8, 0.2]}

    evaluator = _CandidateEvaluator(evaluate_candidates, fitness_metric="second")
    fitness = evaluator.evaluate(np.array([{"depth": 1}, {"depth": 2}], dtype=object))
    assert fitness[0] > fitness[1]


@pytest.mark.parametrize(
    "fitness_metric, refit, message",
    [(None, False, "Set fitness_metric"), ("missing", "accuracy", "not present in scoring")],
)
def test_search_rejects_missing_fitness_metric(
    fitness_metric: object, refit: object, message: str
) -> None:
    X, y = load_iris(return_X_y=True)
    search = GeneticSearchCV(
        DecisionTreeClassifier(),
        {},
        scoring=["accuracy", "balanced_accuracy"],
        fitness_metric=fitness_metric,
        refit=refit,
    )
    with pytest.raises(ValueError, match=message):
        search.fit(X, y)


def test_parallel_search_can_be_pickled_and_fitted_again() -> None:
    X, y = load_iris(return_X_y=True)
    search = GeneticSearchCV(
        DecisionTreeClassifier(random_state=0),
        {"max_depth": [1, 2, 3]},
        optimizer_args={"iters": 2, "pop_size": 6},
        random_state=42,
        cv=2,
        n_jobs=2,
    ).fit(X, y)
    restored = pickle.loads(pickle.dumps(search))
    np.testing.assert_array_equal(restored.predict(X), search.predict(X))
    expected = search.cv_results_["mean_test_score"].copy()
    search.fit(X, y)
    np.testing.assert_array_equal(search.cv_results_["mean_test_score"], expected)
