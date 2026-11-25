from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from ..base import Tree
from ..base._gp import tree_complexity
from ..optimizers import GeneticProgramming, NSGPII
from ..regressors import GeneticProgrammingRegressor


class KnownFrontGP(NSGPII):
    def fit(self) -> KnownFrontGP:
        terminal = self._uniset._terminal_set[0]
        add, mul = self._uniset._functional_set[2]
        trees = np.array(
            [Tree([add, mul, terminal, terminal, terminal]), Tree([terminal])], dtype=object
        )
        self.front = {
            "genotype": deepcopy(trees),
            "phenotype": trees,
            "fitness": self._fitness_function(trees, **self._fitness_function_args),
            "indices": np.arange(2),
        }
        return self

    def get_pareto_front(self) -> dict:
        return deepcopy(self.front)


@pytest.mark.parametrize("complexity", ["n_nodes", "n_levels", "n_functions", "n_terminals"])
def test_real_multiobjective_regressor_fits_and_predicts(complexity: str) -> None:
    x = np.linspace(-1, 1, 20)[:, None]
    y = x[:, 0] ** 2 + x[:, 0]
    model = GeneticProgrammingRegressor(
        n_iter=3,
        pop_size=10,
        functional_set_names=("add", "mul"),
        optimizer=NSGPII,
        optimizer_args={"init_level": 3, "max_level": 5, "keep_history": True},
        complexity=complexity,
        random_state=42,
    )
    assert model.fit(x, y) is model
    front = model.get_pareto_front()
    assert front["fitness"].shape == (len(front["phenotype"]), 2)
    for tree, (error, cost) in zip(front["phenotype"], front["fitness"]):
        assert error == pytest.approx(np.sqrt(np.mean((tree() - y) ** 2)))
        assert cost == tree_complexity(tree, complexity)
    errors, costs = front["fitness"].T
    assert model.selected_tree_index_ == np.lexsort((costs, errors))[0]
    prediction = model.predict(x)
    assert prediction.shape == y.shape
    assert np.sqrt(np.mean((prediction - y) ** 2)) == pytest.approx(
        front["fitness"][model.selected_tree_index_, 0]
    )
    assert len(model.get_stats()["fitness"]) == 3
    assert all(fitness.shape == (10, 2) for fitness in model.get_stats()["fitness"])
    assert model._fitness_cache
    assert all(np.isscalar(value) for value in model._fitness_cache.values())
    assert clone(model).get_params(deep=False) == model.get_params(deep=False)


@pytest.mark.parametrize("selection, expected_index", [("best_error", 0), ("simplest", 1)])
def test_model_selection_and_manual_switch(selection: str, expected_index: int) -> None:
    x = np.linspace(-1, 1, 20)[:, None]
    y = x[:, 0] ** 2 + x[:, 0]
    model = GeneticProgrammingRegressor(
        n_iter=1,
        pop_size=2,
        functional_set_names=("add", "mul"),
        optimizer=KnownFrontGP,
        model_selection=selection,
        use_fitness_cache=False,
    ).fit(x, y)
    assert model.selected_tree_index_ == expected_index
    new_x = np.array([[-0.5], [0.5], [2.0]])
    expected = new_x[:, 0] ** 2 + new_x[:, 0] if expected_index == 0 else new_x[:, 0]
    np.testing.assert_allclose(model.predict(new_x), expected)
    alternative = 1 - expected_index
    assert model.select_tree(np.int64(alternative)) is model
    assert model.selected_tree_index_ == alternative
    expected = new_x[:, 0] ** 2 + new_x[:, 0] if alternative == 0 else new_x[:, 0]
    np.testing.assert_allclose(model.predict(new_x), expected)
    front = model.get_pareto_front()
    front["fitness"][:] = -1
    front["phenotype"][0]._nodes.clear()
    assert len(model.pareto_front_["phenotype"][0]) == 5
    assert np.all(model.pareto_front_["fitness"] >= 0)
    model.get_tree()._nodes.clear()
    assert len(model.pareto_front_["phenotype"][alternative]) > 0


@pytest.mark.parametrize(
    "index, error", [(-1, IndexError), (2, IndexError), (0.5, TypeError), (True, TypeError)]
)
def test_manual_selection_rejects_invalid_index(index: object, error: type[Exception]) -> None:
    x = np.linspace(-1, 1, 10)[:, None]
    model = GeneticProgrammingRegressor(
        optimizer=KnownFrontGP, n_iter=1, pop_size=2, functional_set_names=("add", "mul")
    ).fit(x, x[:, 0] ** 2 + x[:, 0])
    previous = model.selected_tree_index_
    with pytest.raises(error):
        model.select_tree(index)
    assert model.selected_tree_index_ == previous


@pytest.mark.parametrize("option", [{"complexity": "unknown"}, {"model_selection": "unknown"}])
def test_multiobjective_regressor_rejects_invalid_options(option: dict) -> None:
    x = np.linspace(-1, 1, 10)[:, None]
    with pytest.raises(ValueError):
        GeneticProgrammingRegressor(optimizer=NSGPII, n_iter=1, pop_size=2, **option).fit(
            x, x[:, 0]
        )


def test_single_objective_refit_clears_pareto_state_and_ignores_new_options() -> None:
    x = np.linspace(-1, 1, 10)[:, None]
    y = x[:, 0] ** 2 + x[:, 0]
    model = GeneticProgrammingRegressor(
        optimizer=KnownFrontGP,
        n_iter=2,
        pop_size=4,
        functional_set_names=("add", "mul"),
        optimizer_args={"init_level": 3, "max_level": 5},
        random_state=42,
    ).fit(x, y)
    model.set_params(optimizer=GeneticProgramming, complexity="ignored", model_selection="ignored")
    model.fit(x, y)
    assert isinstance(model.get_tree(), Tree)
    assert model.predict(x).shape == y.shape
    assert not hasattr(model, "pareto_front_")
    assert not hasattr(model, "selected_tree_index_")
    with pytest.raises(ValueError, match="Pareto front"):
        model.get_pareto_front()
    with pytest.raises(ValueError, match="Pareto front"):
        model.select_tree(0)


def test_front_methods_require_fitting() -> None:
    model = GeneticProgrammingRegressor(optimizer=NSGPII)
    with pytest.raises(NotFittedError):
        model.get_pareto_front()
    with pytest.raises(NotFittedError):
        model.select_tree(0)
