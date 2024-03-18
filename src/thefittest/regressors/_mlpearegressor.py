from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray

from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from ..base._mlp import BaseMLPEA
from ..classifiers._gpnnclassifier import weights_type_optimizer_alias
from ..optimizers import SHADE


class MLPEARegressor(RegressorMixin, BaseMLPEA):
    def __init__(
        self,
        *,
        n_iter: int = 200,
        pop_size: int = 500,
        hidden_layers: Tuple[int, ...] = (100,),
        activation: str = "sigma",
        offset: bool = True,
        weights_optimizer: weights_type_optimizer_alias = SHADE,
        weights_optimizer_args: Optional[dict[str, Any]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        super().__init__(
            n_iter=n_iter,
            pop_size=pop_size,
            hidden_layers=hidden_layers,
            activation=activation,
            offset=offset,
            weights_optimizer=weights_optimizer,
            weights_optimizer_args=weights_optimizer_args,
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

        if self.offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        output = self.net_.forward(X)[0]
        y_predict = output[:, 0]

        return y_predict
