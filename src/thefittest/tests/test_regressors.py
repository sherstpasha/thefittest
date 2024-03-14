import numpy as np

import pytest

from sklearn.utils.estimator_checks import check_estimator

from ..optimizers import SelfCGA
from ..optimizers import SelfCGP
from ..regressors import GeneticProgrammingNeuralNetRegressor
from ..regressors import GeneticProgrammingRegressor
from ..regressors import MLPEARegressor


def test_GeneticProgrammingNeuralNetRegressor():
    def problem(x):
        return np.sin(x[:, 0])

    iters = 10
    pop_size = 50

    function = problem
    left_border = -4.5
    right_border = 4.5
    sample_size = 300
    n_dimension = 1

    X = np.array(
        [np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]
    ).T
    y = function(X)

    model = GeneticProgrammingNeuralNetRegressor(
        n_iter=3,
        pop_size=10,
        optimizer=GeneticProgramming,
        optimizer_args={
            "tour_size": 15,
            "show_progress_each": 1,
            "mutation_rate": 0.07,
            "parents_num": 1,
            "elitism": False,
        },
        weights_optimizer=DifferentialEvolution,
        weights_optimizer_args={"iters": 25, "pop_size": 25, "CR": 0.9},
        max_hidden_block_size=2,
        offset=False,
        test_sample_ratio=0.3,
        net_size_penalty=0.01,
    )

    model.fit(X, y)
    model.predict(X)

    net = model.get_net()
    stats = model.get_stats()
    tree = model.get_tree()

    model = GeneticProgrammingNeuralNetRegressor(
        n_iter=3,
        pop_size=10,
        optimizer=GeneticProgramming,
        optimizer_args={
            "iters": 15,
            "tour_size": 15,
            "show_progress_each": 1,
            "mutation_rate": 0.07,
            "parents_num": 1,
            "elitism": False,
        },
        weights_optimizer=DifferentialEvolution,
        weights_optimizer_args={"iters": 25, "pop_size": 25, "CR": 0.9},
    )

    with pytest.raises(AssertionError):
        model.fit(X, y)

    model = GeneticProgrammingNeuralNetRegressor(
        n_iter=3,
        pop_size=10,
        optimizer=GeneticProgramming,
        optimizer_args={
            "tour_size": 15,
            "show_progress_each": 1,
            "mutation_rate": 0.07,
            "parents_num": 1,
            "elitism": False,
        },
        weights_optimizer=DifferentialEvolution,
        weights_optimizer_args={"iters": 25, "pop_size": 25, "CR": 0.9, "fitness_function": sum},
    )

    with pytest.raises(AssertionError):
        model.fit(X, y)

    model = GeneticProgrammingNeuralNetRegressor(
        n_iter=3,
        pop_size=10,
        optimizer_args={"show_progress_each": 1},
    )

    check_estimator(model)


def test_SymbolicRegressionGP():

    def problem(x):
        return np.sin(x[:, 0])

    iters = 10
    pop_size = 50

    function = problem
    left_border = -4.5
    right_border = 4.5
    sample_size = 300
    n_dimension = 1

    X = np.array(
        [np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]
    ).T
    y = function(X)

    model = GeneticProgrammingRegressor(
        n_iter=15,
        pop_size=15,
        functional_set_names=("cos", "sin"),
        optimizer=SelfCGP,
        optimizer_args={"keep_history": True, "show_progress_each": 10, "elitism": True, "K": 2.5},
    )

    model.fit(X, y)
    model.predict(X)

    model = GeneticProgrammingRegressor(
        n_iter=15,
        pop_size=15,
        functional_set_names=("cos", "sin"),
        optimizer=SelfCGP,
        optimizer_args={
            "pop_size": 100,
            "keep_history": True,
            "show_progress_each": 10,
            "elitism": True,
            "K": 2.5,
        },
    )

    with pytest.raises(AssertionError):
        model.fit(X, y)

    model = GeneticProgrammingRegressor(
        n_iter=15,
        pop_size=15,
        functional_set_names=("cos2", "sin"),
        optimizer=SelfCGP,
        optimizer_args={
            "keep_history": True,
            "show_progress_each": 10,
            "elitism": True,
            "K": 2.5,
        },
    )

    with pytest.raises(ValueError):
        model.fit(X, y)

    model = GeneticProgrammingRegressor(n_iter=100, pop_size=100)

    check_estimator(model)


def test_MLPEARegressor():
    def problem(x):
        return np.sin(x[:, 0])

    iters = 10
    pop_size = 50

    function = problem
    left_border = -4.5
    right_border = 4.5
    sample_size = 300
    n_dimension = 1

    X = np.array(
        [np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]
    ).T
    y = function(X)

    model = MLPEARegressor(
        n_iter=15,
        pop_size=15,
        hidden_layers=(5, 3, 25),
        weights_optimizer=SelfCGA,
        weights_optimizer_args={"K": 0.9},
        activation="relu",
        offset=False,
    )

    model.fit(X, y)
    model.predict(X)

    net = model.get_net()
    stat = model.get_stats()

    model = MLPEARegressor(
        n_iter=15,
        pop_size=15,
        hidden_layers=(5, 3, 25),
        weights_optimizer=SelfCGA,
        weights_optimizer_args={"iters": 100, "CR": 0.9},
        activation="relu",
        offset=False,
    )

    with pytest.raises(AssertionError):
        model.fit(X, y)

    model = MLPEARegressor(hidden_layers=(0,))
    check_estimator(model)
