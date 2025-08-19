from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

import warnings

import numpy as np
from numpy.typing import NDArray
from numpy.typing import ArrayLike

from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted

import torch

from ..base._mlp import BaseMLPEA
from ..base._mlp import weights_type_optimizer_alias
from ..optimizers import SHADE


class MLPEAClassifier(ClassifierMixin, BaseMLPEA):
    def __init__(
        self,
        *,
        n_iter: int = 200,
        pop_size: int = 500,
        hidden_layers: Tuple[int, ...] = (0,),
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

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self)

        X = self._validate_data(X, reset=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Number of features of the model must match the "
                f"input. Model n_features is {self.n_features_in_} and input "
                f"n_features is {X.shape[1]}."
            )

        if self.offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        device = torch.device(self.device)
        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)

        with torch.no_grad():
            proba_t = self.net_.forward(X_t)

        return proba_t.detach().cpu().numpy()

    def predict(self, X: ArrayLike):
        proba = self.predict_proba(X)
        indeces = np.argmax(proba, axis=1)

        return self._label_encoder.inverse_transform(indeces)
