from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray

from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

import torch

from ..base._mlp import BaseMLPEA
from ..base._mlp import weights_type_optimizer_alias
from ..optimizers import SHADE
from ..utils import (
    _snapshot_tensor_meta,
    _to_numpy_for_validation,
    _back_to_torch,
)


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

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:

        check_is_fitted(self)

        Xm = _snapshot_tensor_meta(X)
        X_np = _to_numpy_for_validation(X)
        X_np = check_array(X_np)

        X_np = check_array(X_np)
        n_features = X_np.shape[1] + 1

        if self.n_features_in_ != n_features:
            raise ValueError(
                "Number of features of the model must match the "
                f"input. Model n_features is {self.n_features_in_} and input "
                f"n_features is {n_features}."
            )

        X_t = _back_to_torch(X_np.astype(np.float32, copy=False), Xm, dtype=torch.float32)

        if self.offset:
            ones = torch.ones((X_t.shape[0], 1), dtype=X_t.dtype, device=X_t.device)
            X_t = torch.cat([X_t, ones], dim=1)

        with torch.no_grad():
            out = self.net_.forward(X_t)
        return out

    def predict(self, X: NDArray[np.float64]):

        out_t = self.predict_proba(X)
        indices = torch.argmax(out_t, dim=1).cpu().numpy()
        y_predict = self._label_encoder.inverse_transform(indices)

        return y_predict
