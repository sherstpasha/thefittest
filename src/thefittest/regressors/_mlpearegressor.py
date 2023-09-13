from __future__ import annotations

from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..base._net import Net
from ..classifiers import MLPClassifierEA
from ..optimizers import OptimizerStringType
from ..optimizers import SHADE
from ..tools.metrics import root_mean_square_error2d


class MLPRegressorrEA(MLPClassifierEA):
    def __init__(
        self,
        iters: int,
        pop_size: int,
        hidden_layers: Tuple,
        activation: str = "sigma",
        output_activation: str = "sigma",
        offset: bool = True,
        no_increase_num: Optional[int] = None,
        show_progress_each: Optional[int] = None,
        keep_history: bool = False,
        optimizer_weights: OptimizerStringType = SHADE,
        optimizer_weights_bounds: tuple = (-10, 10),
        optimizer_weights_n_bit: int = 16,
    ):
        MLPClassifierEA.__init__(
            self,
            iters=iters,
            pop_size=pop_size,
            hidden_layers=hidden_layers,
            activation=activation,
            output_activation=output_activation,
            offset=offset,
            no_increase_num=no_increase_num,
            show_progress_each=show_progress_each,
            keep_history=keep_history,
            optimizer_weights=optimizer_weights,
            optimizer_weights_bounds=optimizer_weights_bounds,
            optimizer_weights_n_bit=optimizer_weights_n_bit,
        )

    def _evaluate_nets(
        self,
        weights: NDArray[np.float64],
        net: Net,
        X: NDArray[np.float64],
        targets: NDArray[Union[np.float64, np.int64]],
    ) -> NDArray[np.float64]:
        output2d = net.forward(X, weights)[:, :, 0]
        error = root_mean_square_error2d(targets, output2d)
        return error

    def _fitness_function(
        self, population: NDArray, X: NDArray[np.float64], targets: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        output2d = np.array([net.forward(X)[0] for net in population], dtype=np.float64)[:, :, 0]
        fitness = root_mean_square_error2d(targets, output2d)
        return fitness

    def _fit(
        self, X: NDArray[np.float64], y: NDArray[Union[np.float64, np.int64]]
    ) -> MLPRegressorrEA:
        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        n_inputs = X.shape[1]
        n_outputs = len(set(y))

        self._net = self._defitne_net(n_inputs, n_outputs)

        self._net._weights = self._train_net(self._net, X, y)
        return self

    def _predict(self, X: NDArray[np.float64]) -> NDArray[Union[np.float64, np.int64]]:
        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        output = self._net.forward(X)[0, :, 0]
        y_pred = output
        return y_pred
