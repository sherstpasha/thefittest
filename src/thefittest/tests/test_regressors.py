import numpy as np

from ..base import EphemeralNode
from ..base import FunctionalNode
from ..base import TerminalNode
from ..base import UniversalSet
from ..optimizers import DifferentialEvolution
from ..optimizers import GeneticProgramming
from ..optimizers import SelfCGA
from ..optimizers import SelfCGP
from ..regressors import GeneticProgrammingNeuralNetRegressor
from ..regressors import MLPEARegressor
from ..regressors import SymbolicRegressionGP
from ..base._tree import Add
from ..base._tree import Div
from ..base._tree import Mul
from ..base._tree import Neg
from ..utils.crossovers import uniform_crossoverGP

from sklearn.utils.estimator_checks import check_estimator


# def test_SymbolicRegressionGP():
#     def problem(x):
#         return np.sin(x[:, 0])

#     def generator1():
#         return np.round(np.random.uniform(0, 10), 4)

#     def generator2():
#         return np.random.randint(0, 10)

#     iters = 10
#     pop_size = 50

#     function = problem
#     left_border = -4.5
#     right_border = 4.5
#     sample_size = 300
#     n_dimension = 1

#     X = np.array(
#         [np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]
#     ).T
#     y = function(X)

#     functional_set = [
#         FunctionalNode(Add()),
#         FunctionalNode(Mul()),
#         FunctionalNode(Neg()),
#         FunctionalNode(Div()),
#     ]

#     terminal_set = [TerminalNode(X[:, i], f"x{i}") for i in range(n_dimension)]
#     terminal_set.extend([EphemeralNode(generator1), EphemeralNode(generator2)])
#     uniset = UniversalSet(functional_set, terminal_set)

#     optimizer = GeneticProgramming

#     optimizer_args = {
#         "tour_size": 15,
#         "show_progress_each": 10,
#         "optimal_value": 1.1,
#         "crossover": "gp_uniform_k",
#         "parents_num": 6,
#     }

#     model = SymbolicRegressionGP(
#         iters=iters,
#         pop_size=pop_size,
#         uniset=uniset,
#         optimizer=optimizer,
#         optimizer_args=optimizer_args,
#     )

#     model.fit(X, y)

#     optimizer = model.get_optimizer()

#     assert optimizer._iters == iters
#     assert optimizer._pop_size == pop_size
#     assert optimizer._show_progress_each == 10
#     assert optimizer._crossover_pool[optimizer._specified_crossover][1] == 6
#     assert optimizer._crossover_pool[optimizer._specified_crossover][0] == uniform_crossoverGP

#     model = SymbolicRegressionGP(
#         iters=iters,
#         pop_size=pop_size,
#     )

#     model.fit(X, y)

#     model.predict(X)


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

    optimizer = GeneticProgramming

    optimizer_args = {"tour_size": 15, "show_progress_each": 1}

    iters = 3
    pop_size = 10

    weights_optimizer = DifferentialEvolution
    weights_optimizer_args = {"iters": 25, "pop_size": 25, "CR": 0.9}

    model = GeneticProgrammingNeuralNetRegressor(
        n_iter=iters,
        pop_size=pop_size,
        optimizer=optimizer,
        optimizer_args=optimizer_args,
        weights_optimizer=weights_optimizer,
        weights_optimizer_args=weights_optimizer_args,
    )

    model.fit(X, y)

    model.predict(X)

    weights_optimizer = SelfCGA
    optimizer = SelfCGP
    weights_optimizer_args = {"iters": 25, "pop_size": 25, "K": 0.33}

    model = GeneticProgrammingNeuralNetRegressor(
        n_iter=iters,
        pop_size=pop_size,
        optimizer=optimizer,
        optimizer_args=optimizer_args,
        weights_optimizer=weights_optimizer,
        weights_optimizer_args=weights_optimizer_args,
    )

    model.fit(X, y)

    model.predict(X)

    model = GeneticProgrammingNeuralNetRegressor(
        n_iter=iters,
        pop_size=pop_size,
    )

    model.fit(X, y)

    model.predict(X)

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

    iters = 3
    pop_size = 10

    weights_optimizer = DifferentialEvolution
    weights_optimizer_args = {"CR": 0.9}

    model = MLPEARegressor(
        n_iter=iters,
        pop_size=pop_size,
        hidden_layers=(10, 3),
        weights_optimizer=weights_optimizer,
        weights_optimizer_args=weights_optimizer_args,
    )

    model.fit(X, y)

    model.predict(X)

    weights_optimizer = SelfCGA
    weights_optimizer_args = {"K": 0.33}

    model = MLPEARegressor(
        n_iter=iters,
        pop_size=pop_size,
        hidden_layers=(0,),
        weights_optimizer=weights_optimizer,
        weights_optimizer_args=weights_optimizer_args,
    )

    model.fit(X, y)

    model.predict(X)

    model = MLPEARegressor(n_iter=500, hidden_layers=(0,))

    model.fit(X, y)

    model.predict(X)

    check_estimator(model)
