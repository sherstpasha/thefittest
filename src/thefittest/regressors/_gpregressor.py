from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
from numpy.typing import NDArray

from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from ..base._gp import BaseGP
from ..optimizers import GeneticProgramming
from ..optimizers import SelfCGP


class GeneticProgrammingRegressor(RegressorMixin, BaseGP):
    def __init__(
        self,
        *,
        n_iter: int = 300,
        pop_size: int = 1000,
        functional_set_names: Tuple[str, ...] = ("cos", "sin", "add", "inv", "neg", "mul"),
        optimizer: Union[Type[SelfCGP], Type[GeneticProgramming]] = SelfCGP,
        optimizer_args: Optional[dict[str, Any]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        super().__init__(
            n_iter=n_iter,
            pop_size=pop_size,
            functional_set_names=functional_set_names,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            random_state=random_state,
        )

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
