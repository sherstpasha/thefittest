from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from ..base._tree import init_symbolic_regression_uniset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted


from ..optimizers import GeneticProgramming

from ..base import UniversalSet
from ..optimizers import SelfCGP
from ..utils._metrics import categorical_crossentropy3d
from ..utils._metrics import root_mean_square_error2d
from ..utils._metrics import coefficient_determination
from ..utils.random import check_random_state
from ..utils.random import randint
from ..utils.random import uniform
from ..utils import array_like_to_numpy_X_y


def fitness_function(trees: NDArray, y: NDArray[np.float64]) -> NDArray[np.float64]:
    fitness = []
    for tree in trees:
        y_pred = tree() * np.ones(len(y))
        fitness.append(coefficient_determination(y, y_pred))
    return np.array(fitness, dtype=np.float64)


class BaseGP(BaseEstimator, metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        *,
        n_iter: int = 50,
        pop_size: int = 500,
        uniset: Optional[UniversalSet] = None,
        optimizer: Union[Type[SelfCGP], Type[GeneticProgramming]] = SelfCGP,
        optimizer_args: Optional[dict[str, Any]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        self.n_iter = n_iter
        self.pop_size = pop_size
        self.uniset = uniset
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.random_state = random_state

    def get_optimizer(
        self,
    ) -> Union[
        GeneticProgramming,
        SelfCGP,
    ]:
        return self.trained_optimizer_

    def generator1(self) -> float:
        value = np.round(uniform(0, 10, 1)[0], 4)
        return value

    def generator2(self) -> int:
        value = randint(0, 10, 1)[0]
        return value

    def check_optimizer_args(self) -> dict:
        if self.optimizer_args is None:
            optimizer_args = {}
        else:
            optimizer_args = self.optimizer_args.copy()
            for arg in (
                "iters",
                "uniset",
                "pop_size",
            ):
                assert (
                    arg not in optimizer_args
                ), f"Do not set '{arg}' in 'optimizer_args'. Instead, use the arguments of the class."
            for arg in (
                "fitness_function",
                "fitness_function_args",
                "genotype_to_phenotype",
                "genotype_to_phenotype_args",
                "minimization",
                "init_population",
                "optimal_value",
            ):
                assert (
                    arg not in optimizer_args
                ), f"Do not set '{arg}' to 'optimizer_args'. It is defined automatically."

        return optimizer_args

    def fit(self, X: ArrayLike, y: ArrayLike):

        optimizer_args = self.check_optimizer_args()
        check_random_state(self.random_state)

        if isinstance(self, ClassifierMixin):
            pass
        else:
            X, y = self._validate_data(X, y, y_numeric=True, reset=True)

        X, y = array_like_to_numpy_X_y(X, y)

        # в отдельную функцию
        if self.uniset is None:
            uniset = init_symbolic_regression_uniset(
                X, ephemeral_node_generators=(self.generator1, self.generator2)
            )
        else:
            uniset = self.uniset

        optimizer_args["iters"] = self.n_iter
        optimizer_args["pop_size"] = self.pop_size
        optimizer_args["uniset"] = uniset

        if isinstance(self, ClassifierMixin):
            pass

        else:
            optimizer_args["fitness_function"] = fitness_function
            optimizer_args["fitness_function_args"] = {"y": y}

        self.trained_optimizer_ = self.optimizer(**optimizer_args)
        self.trained_optimizer_.fit()

        self.tree_ = self.trained_optimizer_.get_fittest()["phenotype"]

        return self

    def predict(self, X: NDArray[np.float64]):

        check_is_fitted(self)

        X = check_array(X)
        n_features = X.shape[1]

        if self.n_features_in_ != n_features:
            raise ValueError(
                "Number of features of the model must match the "
                f"input. Model n_features is {self.n_features_in_} and input "
                f"n_features is {n_features}."
            )

        tree_for_predict = self.tree_.set_terminals(**{f"x{i}": X[:, i] for i in range(n_features)})

        y_predict = tree_for_predict() * np.ones(len(X))

        return y_predict
