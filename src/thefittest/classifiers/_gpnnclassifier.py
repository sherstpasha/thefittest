from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
from numpy.typing import NDArray

from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from ..base._gpnn import BaseGPNN
from ..base._mlp import weights_type_optimizer_alias
from ..optimizers import GeneticProgramming
from ..optimizers import SHADE
from ..optimizers import SelfCGP


class GeneticProgrammingNeuralNetClassifier(ClassifierMixin, BaseGPNN):
    def __init__(
        self,
        *,
        n_iter: int = 15,
        pop_size: int = 50,
        input_block_size: int = 1,
        max_hidden_block_size: int = 9,
        offset: bool = True,
        test_sample_ratio: float = 0.5,
        optimizer: Union[Type[SelfCGP], Type[GeneticProgramming]] = SelfCGP,
        optimizer_args: Optional[dict[str, Any]] = None,
        weights_optimizer: weights_type_optimizer_alias = SHADE,
        weights_optimizer_args: Optional[dict[str, Any]] = None,
        net_size_penalty: float = 0.0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        super().__init__(
            n_iter=n_iter,
            pop_size=pop_size,
            input_block_size=input_block_size,
            max_hidden_block_size=max_hidden_block_size,
            offset=offset,
            test_sample_ratio=test_sample_ratio,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            weights_optimizer=weights_optimizer,
            weights_optimizer_args=weights_optimizer_args,
            net_size_penalty=net_size_penalty,
            random_state=random_state,
        )

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
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

        proba = self.net_.forward(X)[0]
        return proba

    def predict(self, X: NDArray[np.float64]):

        proba = self.predict_proba(X)

        indeces = np.argmax(proba, axis=1)
        y_predict = self._label_encoder.inverse_transform(indeces)

        return y_predict
