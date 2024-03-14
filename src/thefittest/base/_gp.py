from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted


from ..base import Tree
from ..base._ea import Statistics
from ..base._mlp import check_optimizer_args
from ..base._tree import init_symbolic_regression_uniset
from ..optimizers import GeneticProgramming
from ..optimizers import SelfCGP
from ..utils import array_like_to_numpy_X_y
from ..utils._metrics import coefficient_determination
from ..utils._metrics import f1_score
from ..utils.random import check_random_state
from ..utils.random import randint
from ..utils.random import uniform


def fitness_function_gp(
    trees: NDArray,
    y: NDArray[np.float64],
    task_type: str = "regression",
) -> NDArray[np.float64]:
    fitness = []
    if task_type == "classification":
        for tree in trees:
            tree_output = tree() * np.ones(len(y))
            y_pred = (tree_output > 0).astype(np.int64)
            fitness.append(f1_score(y, y_pred))
    elif task_type == "regression":
        for tree in trees:
            y_pred = tree() * np.ones(len(y))
            fitness.append(coefficient_determination(y, y_pred))
    else:
        raise ValueError("task_type must be 'classification' or 'regression'")
    return np.array(fitness, dtype=np.float64)


class BaseGP(BaseEstimator, metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        *,
        n_iter: int = 50,
        pop_size: int = 500,
        functional_set_names: Tuple[str, ...] = ("cos", "sin", "add", "inv", "neg", "mul"),
        optimizer: Union[Type[SelfCGP], Type[GeneticProgramming]] = SelfCGP,
        optimizer_args: Optional[dict[str, Any]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        self.n_iter = n_iter
        self.pop_size = pop_size
        self.functional_set_names = functional_set_names
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.random_state = random_state

    def get_tree(self) -> Tree:
        return self.tree_

    def get_stats(self) -> Statistics:
        return self.optimizer_stats_

    def generator1(self) -> float:
        value = np.round(uniform(0, 10, 1)[0], 4)
        return value

    def generator2(self) -> int:
        value = randint(0, 10, 1)[0]
        return value

    def fit(self, X: ArrayLike, y: ArrayLike):

        if self.optimizer_args is not None:
            optimizer_args = self.optimizer_args.copy()
            check_optimizer_args(
                optimizer_args,
                args_auto_defined=[
                    "fitness_function",
                    "fitness_function_args",
                    "genotype_to_phenotype",
                    "genotype_to_phenotype_args",
                    "init_population",
                    "minimization",
                ],
                args_in_class=["iters", "pop_size"],
            )
        else:
            optimizer_args = {}

        check_random_state(self.random_state)

        if isinstance(self, ClassifierMixin):
            X, y = self._validate_data(X, y, y_numeric=False, reset=True)
            check_classification_targets(y)
            self._label_encoder = LabelEncoder()
            self._one_hot_encoder = OneHotEncoder(
                sparse_output=False, categories="auto", dtype=np.float64
            )
            y = self._label_encoder.fit_transform(y)
            self.classes_ = self._label_encoder.classes_
            self.n_classes_ = len(self.classes_)
            if self.n_classes_ != 2:
                raise ValueError(
                    f"This classifier is intended for binary classification. Expected 2 classes, but found {self.n_classes_} classes."
                )
            X, y = array_like_to_numpy_X_y(X, y, y_numeric=False)
        else:
            X, y = self._validate_data(X, y, y_numeric=True, reset=True)
            X, y = array_like_to_numpy_X_y(X, y, y_numeric=True)

        if self.functional_set_names is not None:
            uniset = init_symbolic_regression_uniset(
                X=X,
                functional_set_names=self.functional_set_names,
                ephemeral_node_generators=(self.generator1, self.generator2),
            )
        else:
            uniset = init_symbolic_regression_uniset(
                X=X,
                ephemeral_node_generators=(self.generator1, self.generator2),
            )

        optimizer_args["fitness_function"] = fitness_function_gp
        optimizer_args["iters"] = self.n_iter
        optimizer_args["pop_size"] = self.pop_size
        optimizer_args["uniset"] = uniset

        if isinstance(self, ClassifierMixin):
            optimizer_args["fitness_function_args"] = {"y": y, "task_type": "classification"}
        else:
            optimizer_args["fitness_function_args"] = {"y": y, "task_type": "regression"}

        optimizer_ = self.optimizer(**optimizer_args)
        optimizer_.fit()

        self.tree_ = optimizer_.get_fittest()["phenotype"]
        self.optimizer_stats_ = optimizer_.get_stats()

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
        tree_output = tree_for_predict() * np.ones(len(X))

        if isinstance(self, ClassifierMixin):
            indeces = (tree_output > 0).astype(np.int64)
            y_predict = self._label_encoder.inverse_transform(indeces)
        else:
            y_predict = tree_output

        return y_predict
