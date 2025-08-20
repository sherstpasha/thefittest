import pytest

from sklearn.utils.estimator_checks import check_estimator

from ..benchmarks import BanknoteDataset
from ..benchmarks import IrisDataset
from ..classifiers import GeneticProgrammingClassifier
from ..classifiers import GeneticProgrammingNeuralNetClassifier
from ..classifiers import MLPEAClassifier
from ..optimizers import DifferentialEvolution
from ..optimizers import GeneticProgramming
from ..optimizers import SelfCGA
from ..optimizers import SelfCGP
from ..utils.transformations import minmax_scale


def test_GeneticProgrammingNeuralNetClassifier():
    data = IrisDataset()
    X = minmax_scale(data.get_X())
    y = data.get_y()

    model = GeneticProgrammingNeuralNetClassifier(
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

    model = GeneticProgrammingNeuralNetClassifier(
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

    model = GeneticProgrammingNeuralNetClassifier(
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

    model = GeneticProgrammingNeuralNetClassifier(
        n_iter=1, pop_size=10, weights_optimizer_args={"iters": 25, "pop_size": 25}
    )

    check_estimator(model)


def test_MLPEAClassifier():
    data = IrisDataset()
    X = minmax_scale(data.get_X())
    y = data.get_y()

    model = MLPEAClassifier(
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

    model = MLPEAClassifier(
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

    model = MLPEAClassifier(hidden_layers=(0,), weights_optimizer_args={"no_increase_num": 30})

def test_GPClassifier():
    data = BanknoteDataset()
    X = minmax_scale(data.get_X())
    y = data.get_y()

    model = GeneticProgrammingClassifier(
        n_iter=15,
        pop_size=15,
        functional_set_names=("cos", "sin"),
        optimizer=SelfCGP,
        optimizer_args={"keep_history": True, "show_progress_each": 10, "elitism": True, "K": 2.5},
    )

    model.fit(X, y)
    model.predict(X)

    model = GeneticProgrammingClassifier(
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

    model = GeneticProgrammingClassifier(
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

    model = GeneticProgrammingClassifier(
        n_iter=100, pop_size=100, optimizer_args={"no_increase_num": 30}
    )

    check_estimator(model)
