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
from ..tools.metrics import root_mean_square_error2d, root_mean_square_error3d
from ..tools.random import float_population
from ..tools.transformations import GrayCode
from ..base._net import ACTIV_NAME_INV


def fitness_function(
    weights: NDArray[np.float32],
    net: Net,
    X: NDArray[np.float32],
    targets: NDArray[Union[np.float32, np.int64]],
) -> NDArray[np.float32]:
    output3d = net.forward(X, weights)
    # print(output2d.shape, targets.shape)
    error = root_mean_square_error3d(targets, output3d)
    return error


class MLPEARegressorMO(MLPEAClassifier):
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

    def _defitne_net(self: MLPEAClassifier, n_inputs: int, n_outputs: int) -> Net:
        start = 0
        end = n_inputs
        inputs_id = set(range(start, end))

        net = Net(inputs=inputs_id)

        for n_layer in self._hidden_layers:
            start = end
            end = end + n_layer
            inputs_id = {(n_inputs - 1)}
            hidden_id = set(range(start, end))
            activs = dict(zip(hidden_id, [ACTIV_NAME_INV[self._activation]] * len(hidden_id)))

            if self._offset:
                layer_net = Net(inputs=inputs_id) > Net(hidden_layers=[hidden_id], activs=activs)
            else:
                layer_net = Net(hidden_layers=[hidden_id], activs=activs)

            net = net > layer_net

        start = end
        end = end + n_outputs
        inputs_id = {(n_inputs - 1)}
        output_id = set(range(start, end))
        activs = dict(zip(output_id, [ACTIV_NAME_INV[self._output_activation]] * len(output_id)))

        if self._offset:
            layer_net = Net(inputs=inputs_id) > Net(outputs=output_id, activs=activs)
        else:
            layer_net = Net(outputs=output_id, activs=activs)

        net = net > layer_net
        net._offset = self._offset
        return net

    def _train_net(
        self: MLPEAClassifier,
        net: Net,
        X_train: NDArray[np.float32],
        y_train: NDArray[np.float32],
    ) -> NDArray[np.float32]:
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
        left: NDArray[np.float32] = np.full(
            shape=len(net._weights), fill_value=-10, dtype=np.float32
        )
        right: NDArray[np.float32] = np.full(
            shape=len(net._weights), fill_value=10, dtype=np.float32
        )
        initial_population: Union[NDArray[np.float32], NDArray[np.byte]] = float_population(
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
        self: MLPEARegressorMO, X: NDArray[np.float32], y: NDArray[Union[np.float32, np.int64]]
    ) -> MLPEARegressorMO:
        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        n_inputs: int = X.shape[1]
        n_outputs: int = y.shape[1]

        self._net = self._defitne_net(n_inputs, n_outputs)

        print(self._net)

        self._net._weights = self._train_net(self._net, X, y.astype(np.float32))
        return self

    def _predict(
        self: MLPEARegressorMO, X: NDArray[np.float32]
    ) -> NDArray[Union[np.float32, np.int64]]:
        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        output = self._net.forward(X)[0]
        y_pred = output
        return y_pred
