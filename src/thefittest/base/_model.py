from __future__ import annotations

from abc import ABCMeta, abstractmethod

from typing import Any
from typing import Union

from sklearn.base import BaseEstimator

import numpy as np
from numpy.typing import NDArray

from ..base import Net
from ..base._net import ACTIV_NAME_INV
from ..optimizers import DifferentialEvolution
from ..optimizers import GeneticAlgorithm
from ..optimizers import GeneticProgramming
from ..optimizers import SHADE
from ..optimizers import SHAGA
from ..optimizers import SelfCGA
from ..optimizers import SelfCGP
from ..optimizers import jDE

class Model:
    def _fit(
        self,
        X: np.typing.NDArray[np.float64],
        y: NDArray[Union[np.float64, np.int64]],
    ) -> Any:
        pass

    def _predict(self, X: NDArray[np.float64]) -> Any:
        pass

    def get_optimizer(
        self: Model,
    ) -> Any:
        pass

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[Union[np.float64, np.int64]],
    ) -> Any:
        assert np.all(np.isfinite(X))
        assert np.all(np.isfinite(y))
        return self._fit(X, y)

    def predict(self, X: NDArray[np.float64]) -> NDArray[Union[np.float64, np.int64]]:
        return self._predict(X)


class BaseMLPEA(BaseEstimator, metaclass=ABCMeta):

    @abstractmethod
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
        self.iters = iters
        self.pop_size = pop_size
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.offset = offset
        self.weights_optimizer = weights_optimizer
        self.weights_optimizer_args = weights_optimizer_args

    def _defitne_net(self: BaseEstimator, n_inputs: int, n_outputs: int) -> Net:
        start = 0
        end = n_inputs
        inputs_id = set(range(start, end))

        net = Net(inputs=inputs_id)

        for n_layer in self.hidden_layers:
            start = end
            end = end + n_layer
            inputs_id = {(n_inputs - 1)}
            hidden_id = set(range(start, end))
            activs = dict(zip(hidden_id, [ACTIV_NAME_INV[self.activation]] * len(hidden_id)))

            if self.offset:
                layer_net = Net(inputs=inputs_id) > Net(hidden_layers=[hidden_id], activs=activs)
            else:
                layer_net = Net(hidden_layers=[hidden_id], activs=activs)

            net = net > layer_net

        start = end
        end = end + n_outputs
        inputs_id = {(n_inputs - 1)}
        output_id = set(range(start, end))
        activs = dict(zip(output_id, [ACTIV_NAME_INV["softmax"]] * len(output_id)))

        if self.offset:
            layer_net = Net(inputs=inputs_id) > Net(outputs=output_id, activs=activs)
        else:
            layer_net = Net(outputs=output_id, activs=activs)

        net = net > layer_net
        net._offset = self.offset
        return net

    def _train_net(
        self: MLPEAClassifier,
        net: Net,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self.weights_optimizer_args is not None:
            for arg in (
                "fitness_function",
                "left",
                "right",
                "str_len",
                "genotype_to_phenotype",
                "minimization",
            ):
                assert (
                    "iters" not in self.weights_optimizer_args.keys()
                    and "pop_size" not in self.weights_optimizer_args.keys()
                ), """Do not set the "iters" or "pop_size", or "uniset" in the "optimizer_args".
                  Instead, use the "MLPClassifierEA" arguments"""
                assert (
                    arg not in self.weights_optimizer_args.keys()
                ), f"""Do not set the "{arg}"
              to the "weights_optimizer_args". It is defined automatically"""
            weights_optimizer_args = self.weights_optimizer_args.copy()
        else:
            weights_optimizer_args = {}

        weights_optimizer_args["iters"] = self_iters
        weights_optimizer_args["pop_size"] = self.pop_size
        left: NDArray[np.float64] = np.full(
            shape=len(net._weights), fill_value=-10, dtype=np.float64
        )
        right: NDArray[np.float64] = np.full(
            shape=len(net._weights), fill_value=10, dtype=np.float64
        )
        initial_population: Union[
            NDArray[np.float64], NDArray[np.byte]
        ] = DifferentialEvolution.float_population(weights_optimizer_args["pop_size"], left, right)
        initial_population[0] = net._weights.copy()

        weights_optimizer_args["fitness_function"] = fitness_function
        weights_optimizer_args["fitness_function_args"] = {
            "net": net,
            "X": X_train,
            "targets": y_train,
        }

        if self.weights_optimizer_class in (SHADE, DifferentialEvolution, jDE):
            weights_optimizer_args["left"] = left
            weights_optimizer_args["right"] = right
        else:
            parts: NDArray[np.int64] = np.full(
                shape=len(net._weights), fill_value=16, dtype=np.int64
            )
            genotype_to_phenotype = GrayCode().fit(left_border=-10., right_border=10.0,
         num_variables=len(net._weights), bits_per_variable=16)
            weights_optimizer_args["str_len"] = np.sum(genotype_to_phenotype._bits_per_variable)
            weights_optimizer_args["genotype_to_phenotype"] = genotype_to_phenotype.transform

        weights_optimizer_args["minimization"] = True
        optimizer = self.weights_optimizer_class(**weights_optimizer_args)
        optimizer.fit()

        self.weights_optimizer = optimizer

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
        return self.weights_optimizer