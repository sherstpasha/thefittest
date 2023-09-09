import numpy as np

from ..benchmarks import BanknoteDataset
from ..benchmarks import BreastCancerDataset
from ..benchmarks import CreditRiskDataset
from ..benchmarks import DigitsDataset
from ..benchmarks import IrisDataset
from ..benchmarks import UserKnowladgeDataset
from ..benchmarks import WineDataset
from ..benchmarks.symbolicregression17 import (
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    F13,
    F14,
    F15,
    F16,
    F17,
)

from ..benchmarks.CEC2005 import problems_dict
from ..optimizers import DifferentialEvolution
from ..tools.random import float_population


def test_IrisDataset():
    data = IrisDataset()

    X = data.get_X()
    y = data.get_y()

    assert X.shape == (150, 4)
    assert y.shape == (150,)

    X_names = data.get_X_names()
    y_names = data.get_y_names()

    assert len(X_names) == 4
    assert len(y_names) == 3


def test_WineDataset():
    data = WineDataset()

    X = data.get_X()
    y = data.get_y()

    assert X.shape == (178, 13)
    assert y.shape == (178,)

    X_names = data.get_X_names()
    y_names = data.get_y_names()

    assert len(X_names) == 13
    assert len(y_names) == 3


def test_BreastCancerDataset():
    data = BreastCancerDataset()

    X = data.get_X()
    y = data.get_y()

    assert X.shape == (569, 30)
    assert y.shape == (569,)

    X_names = data.get_X_names()
    y_names = data.get_y_names()

    assert len(X_names) == 30
    assert len(y_names) == 2


def test_DigitsDataset():
    data = DigitsDataset()

    X = data.get_X()
    y = data.get_y()

    assert X.shape == (5620, 64)
    assert y.shape == (5620,)

    X_names = data.get_X_names()
    y_names = data.get_y_names()

    assert len(X_names) == 64
    assert len(y_names) == 10


def test_CreditRiskDataset():
    data = CreditRiskDataset()

    X = data.get_X()
    y = data.get_y()

    assert X.shape == (2000, 3)
    assert y.shape == (2000,)

    X_names = data.get_X_names()
    y_names = data.get_y_names()

    assert len(X_names) == 3
    assert len(y_names) == 2


def test_UserKnowladgeDataset():
    data = UserKnowladgeDataset()

    X = data.get_X()
    y = data.get_y()

    assert X.shape == (403, 5)
    assert y.shape == (403,)

    X_names = data.get_X_names()
    y_names = data.get_y_names()

    assert len(X_names) == 5
    assert len(y_names) == 4


def test_BanknoteDataset():
    data = BanknoteDataset()

    X = data.get_X()
    y = data.get_y()

    assert X.shape == (1372, 4)
    assert y.shape == (1372,)

    X_names = data.get_X_names()
    y_names = data.get_y_names()

    assert len(X_names) == 4
    assert len(y_names) == 2


def test_symbolicregression17():
    left_border = -4.5
    right_border = 4.5
    sample_size = 30
    n_dimension = 1

    X = np.array(
        [np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)],
        dtype=np.float64,
    ).T

    F1(X)
    F2(X)
    F16(X)
    F17(X)

    n_dimension = 2

    X = np.array(
        [np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)],
        dtype=np.float64,
    ).T

    F3(X)
    F4(X)
    F5(X)
    F6(X)
    F7(X)
    F8(X)
    F9(X)
    F10(X)
    F11(X)
    F12(X)
    F13(X)
    F14(X)
    F15(X)


def test_CEC2005():
    iters = 10
    pop_size = 10

    for problem in problems_dict.values():
        function = problem["function"]

        for dim in problem["dimentions"]:
            print(problem, dim)
            left_scalar = problem["bounds"][0]
            right_scalar = problem["bounds"][1]

            left = np.full(shape=dim, fill_value=left_scalar, dtype=np.float64)
            right = np.full(shape=dim, fill_value=right_scalar, dtype=np.float64)

            model = DifferentialEvolution(
                fitness_function=function(),
                iters=iters,
                pop_size=pop_size,
                left=left,
                right=right,
                minimization=True,
            )

            if "init_bounds" in problem.keys():
                init_left_scalar = problem["init_bounds"][1]
                init_left = np.full(shape=dim, fill_value=init_left_scalar, dtype=np.float64)
                init_right_scalar = problem["init_bounds"][1]
                init_right = np.full(shape=dim, fill_value=init_right_scalar, dtype=np.float64)

                initial_population = float_population(
                    pop_size=pop_size, left=init_left, right=init_right
                )

                model.set_strategy(initial_population=initial_population)

            model.fit()
