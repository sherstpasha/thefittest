from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..base._net import Net
from ..classifiers import MLPEAClassifier
from ..optimizers import SHADE
from ..tools.metrics import root_mean_square_error2d
from ..classifiers._gpnnclassifier import weights_type_optimizer_alias


class MLPEARegressor(MLPEAClassifier):
    def __init__(
        self,
        iters: int,
        pop_size: int,
        hidden_layers: Tuple[int, ...],
        activation: str = "sigma",
        output_activation: str = "sigma",
        offset: bool = True,
        weights_optimizer: weights_type_optimizer_alias = SHADE,
        weights_optimizer_args: Optional[dict[str, Any]] = None,
    ):
        MLPEAClassifier.__init__(
            self,
            iters=iters,
            pop_size=pop_size,
            hidden_layers=hidden_layers,
            activation=activation,
            output_activation=output_activation,
            offset=offset,
            weights_optimizer=weights_optimizer,
            weights_optimizer_args=weights_optimizer_args,
        )

    def _evaluate_nets(
        self: MLPEARegressor,
        weights: NDArray[np.float64],
        net: Net,
        X: NDArray[np.float64],
        targets: NDArray[Union[np.float64, np.int64]],
    ) -> NDArray[np.float64]:
        output2d = net.forward(X, weights)[:, :, 0]
        error = root_mean_square_error2d(targets, output2d)
        return error

    def _fitness_function(
        self: MLPEARegressor,
        population: NDArray,
        X: NDArray[np.float64],
        targets: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        output2d = np.array([net.forward(X)[0] for net in population], dtype=np.float64)[:, :, 0]
        fitness = root_mean_square_error2d(targets, output2d)
        return fitness

    def _fit(
        self: MLPEARegressor, X: NDArray[np.float64], y: NDArray[Union[np.float64, np.int64]]
    ) -> MLPEARegressor:
        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        n_inputs: int = X.shape[1]
        n_outputs: int = len(set(y))

        self._net = self._defitne_net(n_inputs, n_outputs)

        self._net._weights = self._train_net(self._net, X, y.astype(np.float64))
        return self

    def _predict(
        self: MLPEARegressor, X: NDArray[np.float64]
    ) -> NDArray[Union[np.float64, np.int64]]:
        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        output = self._net.forward(X)[0, :, 0]
        y_pred = output
        return y_pred
