from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import warnings

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y

from ..base import Tree
from ..base._ea import Statistics
from ..base._mlp import check_optimizer_args
from ..base._tree import init_symbolic_regression_uniset
from ..optimizers import GeneticProgramming
from ..optimizers import SelfCGP
from ..utils import array_like_to_numpy_X_y
from ..utils._metrics import coefficient_determination
from ..utils._metrics import categorical_crossentropy
from ..utils._metrics import root_mean_square_error
from ..utils.random import check_random_state
from ..utils.random import generator1
from ..utils.random import generator2


def safe_sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def fitness_function_gp(
    trees: NDArray,
    y: NDArray[np.float64],
    task_type: str = "regression",
    fitness_cache: Optional[dict[str, float]] = None,
) -> NDArray[np.float64]:
    fitness = []
    for tree in trees:
        sig = tree.signature()
        if fitness_cache is not None and sig in fitness_cache:
            fitness.append(fitness_cache[sig])
            continue

        if task_type == "classification":
            tree_output = tree() * np.ones(len(y))
            proba = safe_sigmoid(tree_output)
            proba_predict = np.vstack([1 - proba, proba]).T
            fit_val = categorical_crossentropy(y, proba_predict)
        elif task_type == "regression":
            y_pred = tree() * np.ones(len(y))
            fit_val = root_mean_square_error(y, y_pred)
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")

        if fitness_cache is not None:
            fitness_cache[sig] = fit_val
        fitness.append(fit_val)

    return np.array(fitness, dtype=np.float64)

class BaseGP(BaseEstimator, metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        *,
        n_iter: int = 500,
        pop_size: int = 1000,
        functional_set_names: Tuple[str, ...] = ("cos", "sin", "add", "sub", "mul", "div"),
        optimizer: Union[Type[SelfCGP], Type[GeneticProgramming]] = SelfCGP,
        optimizer_args: Optional[dict[str, Any]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        use_fitness_cache: bool = False,
    ):
        self.n_iter = n_iter
        self.pop_size = pop_size
        self.functional_set_names = functional_set_names
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.random_state = random_state
        self.use_fitness_cache = use_fitness_cache

    def get_tree(self) -> Tree:
        return self.tree_

    def get_stats(self) -> Statistics:
        return self.optimizer_stats_

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
            X, y = check_X_y(X, y)
            check_classification_targets(y)

            self._label_encoder = LabelEncoder()
            self._one_hot_encoder = OneHotEncoder(
                sparse_output=False, categories="auto", dtype=np.float64
            )

            numeric_labels = self._label_encoder.fit_transform(y)
            y = self._one_hot_encoder.fit_transform(np.array(numeric_labels).reshape(-1, 1))
            self.classes_ = self._label_encoder.classes_
            self.n_classes_ = len(self.classes_)

            # sklearn-проверки иногда подают 3+ класса. Не падаем — включаем OVR fallback.
            if self.n_classes_ > 2:
                warnings.warn(
                    f"This GP classifier is binary by design; enabling One-vs-Rest fallback for {self.n_classes_} classes.",
                    RuntimeWarning,
                )
        else:
            X, y = check_X_y(X, y)

        X, y = array_like_to_numpy_X_y(X, y, y_numeric=True)
        self.n_features_in_ = X.shape[1]
        if self.functional_set_names is not None:
            uniset = init_symbolic_regression_uniset(
                X=X,
                functional_set_names=self.functional_set_names,
                ephemeral_node_generators=(generator1, generator2),
            )
        else:
            uniset = init_symbolic_regression_uniset(
                X=X,
                ephemeral_node_generators=(generator1, generator2),
            )

        if self.use_fitness_cache:
            self._fitness_cache: dict[str, float] = {}
        else:
            self._fitness_cache = None

        optimizer_args["fitness_function"] = fitness_function_gp
        optimizer_args["iters"] = self.n_iter
        optimizer_args["pop_size"] = self.pop_size
        optimizer_args["uniset"] = uniset
        optimizer_args["minimization"] = True

        if isinstance(self, ClassifierMixin):
            if self.n_classes_ <= 2:
                optimizer_args["fitness_function_args"] = {"y": y, "task_type": "classification", "fitness_cache": self._fitness_cache}
                optimizer_ = self.optimizer(**optimizer_args)
                optimizer_.fit()
                self.tree_ = optimizer_.get_fittest()["phenotype"]
                self.optimizer_stats_ = optimizer_.get_stats()
            else:
                # --- OVR fallback ---
                self.trees_ = []
                last_stats = None
                for k in range(self.n_classes_):
                    if self.use_fitness_cache:
                        self._fitness_cache = {}

                    y_bin = (numeric_labels == k).astype(np.int64)
                    y_bin_oh = np.stack([1 - y_bin, y_bin], axis=1).astype(np.float64)

                    args_k = optimizer_args.copy()
                    args_k["fitness_function_args"] = {"y": y_bin_oh, "task_type": "classification", "fitness_cache": self._fitness_cache}

                    opt_k = self.optimizer(**args_k)
                    opt_k.fit()
                    self.trees_.append(opt_k.get_fittest()["phenotype"])
                    last_stats = opt_k.get_stats()

                self.tree_ = self.trees_[0]
                self.optimizer_stats_ = last_stats
        else:
            optimizer_args["fitness_function_args"] = {"y": y, "task_type": "regression", "fitness_cache": self._fitness_cache}
            optimizer_ = self.optimizer(**optimizer_args)
            optimizer_.fit()
            self.tree_ = optimizer_.get_fittest()["phenotype"]
            self.optimizer_stats_ = optimizer_.get_stats()

        return self
