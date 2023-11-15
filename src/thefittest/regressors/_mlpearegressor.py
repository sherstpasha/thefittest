from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..base._net import Net
from ..classifiers import MLPEAClassifier
from ..classifiers._gpnnclassifier import weights_type_optimizer_alias
from ..optimizers import DifferentialEvolution
from ..optimizers import SHADE
from ..optimizers import jDE
from ..tools.metrics import root_mean_square_error2d
from ..tools.random import float_population
from ..tools.transformations import GrayCode


def fitness_function(
    weights: NDArray[np.float64],
    net: Net,
    X: NDArray[np.float64],
    targets: NDArray[Union[np.float64, np.int64]],
) -> NDArray[np.float64]:
    output2d = net.forward(X, weights)[:, :, 0]
    error = root_mean_square_error2d(targets, output2d)
    return error


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

    def _train_net(
        self: MLPEAClassifier,
        net: Net,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self._weights_optimizer_args is not None:
            for arg in (
                "fitness_function",
                "left",
                "right",
                "str_len",
                "genotype_to_phenotype",
                "minimization",
            ):
                assert (
                    "iters" not in self._weights_optimizer_args.keys()
                    and "pop_size" not in self._weights_optimizer_args.keys()
                ), """Do not set the "iters" or "pop_size", or "uniset" in the "optimizer_args".
                  Instead, use the "MLPClassifierEA" arguments"""
                assert (
                    arg not in self._weights_optimizer_args.keys()
                ), f"""Do not set the "{arg}"
              to the "weights_optimizer_args". It is defined automatically"""
            weights_optimizer_args = self._weights_optimizer_args.copy()
        else:
            weights_optimizer_args = {}

        weights_optimizer_args["iters"] = self._iters
        weights_optimizer_args["pop_size"] = self._pop_size
        left: NDArray[np.float64] = np.full(
            shape=len(net._weights), fill_value=-10, dtype=np.float64
        )
        right: NDArray[np.float64] = np.full(
            shape=len(net._weights), fill_value=10, dtype=np.float64
        )
        initial_population: Union[NDArray[np.float64], NDArray[np.byte]] = float_population(
            weights_optimizer_args["pop_size"], left, right
        )
        initial_population[0] = net._weights.copy()

        weights_optimizer_args["fitness_function"] = fitness_function
        weights_optimizer_args["fitness_function_args"] = {
            "net": net,
            "X": X_train,
            "targets": y_train,
        }

        if self._weights_optimizer_class in (SHADE, DifferentialEvolution, jDE):
            weights_optimizer_args["left"] = left
            weights_optimizer_args["right"] = right
        else:
            parts: NDArray[np.int64] = np.full(
                shape=len(net._weights), fill_value=16, dtype=np.int64
            )
            genotype_to_phenotype = GrayCode(fit_by="parts").fit(left, right, parts)
            weights_optimizer_args["str_len"] = np.sum(parts)
            weights_optimizer_args["genotype_to_phenotype"] = genotype_to_phenotype.transform

        weights_optimizer_args["minimization"] = True
        optimizer = self._weights_optimizer_class(**weights_optimizer_args)
        optimizer.fit()

        self._weights_optimizer = optimizer

        phenotype = optimizer.get_fittest()["phenotype"]

        return phenotype

    def _fit(
        self: MLPEARegressor, X: NDArray[np.float64], y: NDArray[Union[np.float64, np.int64]]
    ) -> MLPEARegressor:
        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        n_inputs: int = X.shape[1]
        n_outputs: int = 1

        self._net = self._defitne_net(n_inputs, n_outputs)

        self._net._weights = self._train_net(self._net, X, y)
        return self

    def _predict(
        self: MLPEARegressor, X: NDArray[np.float64]
    ) -> NDArray[Union[np.float64, np.int64]]:
        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        output = self._net.forward(X)[0, :, 0]
        y_pred = output
        return y_pred
