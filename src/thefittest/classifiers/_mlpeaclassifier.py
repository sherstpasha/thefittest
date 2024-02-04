from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
from numpy.typing import NDArray

from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import check_classification_targets

from ..base import Net
from ..base._model import Model
from ..base._net import ACTIV_NAME_INV
from ..optimizers import DifferentialEvolution
from ..optimizers import GeneticAlgorithm
from ..optimizers import GeneticProgramming
from ..optimizers import SHADE
from ..optimizers import SHAGA
from ..optimizers import SelfCGA
from ..optimizers import SelfCGP
from ..optimizers import jDE
from ..utils._metrics import categorical_crossentropy3d
from ..utils.transformations import GrayCode


weights_type_optimizer_alias = Union[
    Type[DifferentialEvolution],
    Type[jDE],
    Type[SHADE],
    Type[GeneticAlgorithm],
    Type[SelfCGA],
    Type[SHAGA],
]
weights_optimizer_alias = Union[DifferentialEvolution, jDE, SHADE, GeneticAlgorithm, SelfCGA, SHAGA]


def fitness_function(
    weights: NDArray[np.float64],
    net: Net,
    X: NDArray[np.float64],
    targets: NDArray[Union[np.float64, np.int64]],
) -> NDArray[np.float64]:
    output3d = net.forward(X, weights)
    error = categorical_crossentropy3d(targets, output3d)
    return error


class MLPEAClassifier(Model):
    def __init__(
        self,
        iters: int,
        pop_size: int,
        hidden_layers: Tuple[int, ...],
        activation: str = "sigma",
        output_activation: str = "softmax",
        offset: bool = True,
        weights_optimizer: weights_type_optimizer_alias = SHADE,
        weights_optimizer_args: Optional[dict[str, Any]] = None,
    ):
        Model.__init__(self)

        self._iters: int = iters
        self._pop_size: int = pop_size
        self._hidden_layers: Tuple[int, ...] = hidden_layers
        self._activation: str = activation
        self._output_activation: str = output_activation
        self._offset: bool = offset
        self._weights_optimizer: weights_optimizer_alias
        self._weights_optimizer_class: weights_type_optimizer_alias = weights_optimizer
        self._weights_optimizer_args: Optional[dict[str, Any]] = weights_optimizer_args
        self._net: Net

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
        X_train: NDArray[np.float64],
        proba_train: NDArray[np.float64],
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
        initial_population: Union[NDArray[np.float64], NDArray[np.byte]] = (
            DifferentialEvolution.float_population(weights_optimizer_args["pop_size"], left, right)
        )
        initial_population[0] = net._weights.copy()

        weights_optimizer_args["fitness_function"] = fitness_function
        weights_optimizer_args["fitness_function_args"] = {
            "net": net,
            "X": X_train,
            "targets": proba_train,
        }

        if self._weights_optimizer_class in (SHADE, DifferentialEvolution, jDE):
            weights_optimizer_args["left"] = left
            weights_optimizer_args["right"] = right
        else:
            parts: NDArray[np.int64] = np.full(
                shape=len(net._weights), fill_value=16, dtype=np.int64
            )
            genotype_to_phenotype = GrayCode().fit(
                left_border=-10.0,
                right_border=10.0,
                num_variables=len(net._weights),
                bits_per_variable=16,
            )
            weights_optimizer_args["str_len"] = np.sum(genotype_to_phenotype._bits_per_variable)
            weights_optimizer_args["genotype_to_phenotype"] = genotype_to_phenotype.transform

        weights_optimizer_args["minimization"] = True
        optimizer = self._weights_optimizer_class(**weights_optimizer_args)
        optimizer.fit()

        self._weights_optimizer = optimizer

        phenotype = optimizer.get_fittest()["phenotype"]

        return phenotype

    def get_optimizer(
        self: MLPEAClassifier,
    ) -> Union[
        DifferentialEvolution,
        GeneticAlgorithm,
        GeneticProgramming,
        jDE,
        SelfCGA,
        SelfCGP,
        SHADE,
        SHAGA,
    ]:
        return self._weights_optimizer

    def get_net(self: MLPEAClassifier) -> Net:
        return self._net

    def _fit(
        self: MLPEAClassifier, X: NDArray[np.float64], y: NDArray[Union[np.float64, np.int64]]
    ) -> MLPEAClassifier:

        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        n_inputs: int = X.shape[1]
        n_outputs: int = len(set(y))
        eye: NDArray[np.float64] = np.eye(n_outputs, dtype=np.float64)

        proba: NDArray[np.float64] = eye[y]

        self._net = self._defitne_net(n_inputs, n_outputs)

        self._net._weights = self._train_net(self._net, X, proba)
        return self

    def _predict(
        self: MLPEAClassifier, X: NDArray[np.float64]
    ) -> NDArray[Union[np.float64, np.int64]]:
        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        output = self._net.forward(X)[0]
        y_pred = np.argmax(output, axis=1)
        return y_pred


from thefittest.base._model import BaseMLPEA
from sklearn.base import ClassifierMixin


class MLPClassifierEA2(ClassifierMixin, BaseMLPEA):
    def __init__(
        self,
        *,
        iters: int,
        pop_size: int,
        hidden_layers: Tuple[int, ...],
        activation: str = "sigma",
        offset: bool = True,
        weights_optimizer: weights_type_optimizer_alias = SHADE,
        weights_optimizer_args: Optional[dict[str, Any]] = None,
    ):

        super().__init__(
            iters=iters,
            pop_size=pop_size,
            hidden_layers=hidden_layers,
            activation=activation,
            offset=offset,
            weights_optimizer=weights_optimizer,
            weights_optimizer_args=weights_optimizer_args,
        )

    def fit(
        self: MLPClassifierEA2, X: NDArray[np.float64], y: NDArray[np.int64]
    ) -> MLPEAClassifier:
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        if self.offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        n_inputs: int = X.shape[1]
        n_outputs: int = len(set(y))
        eye: NDArray[np.float64] = np.eye(n_outputs, dtype=np.float64)
        proba: NDArray[np.float64] = eye[y]

        self.net = self._defitne_net(n_inputs, n_outputs)
        self.net._weights = self._train_net(self.net, X, proba)
        return self

    # def predict(
    #     self: MLPEAClassifier, X: NDArray[np.float64]
    # ) -> NDArray[Union[np.float64, np.int64]]:
    #     if self._offset:
    #         X = np.hstack([X, np.ones((X.shape[0], 1))])

    #     output = self.net.forward(X)[0]
    #     y_pred = np.argmax(output, axis=1)
    #     return y_pred
