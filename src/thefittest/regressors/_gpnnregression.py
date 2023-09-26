from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..base import UniversalSet
from ..base._net import Net
from ..classifiers import GeneticProgrammingNeuralNetClassifier
from ..optimizers import GeneticProgramming
from ..optimizers import SHADE
from ..optimizers import SelfCGP
from ..tools.metrics import root_mean_square_error2d
from ..tools.random import train_test_split
from ..classifiers._gpnnclassifier import weights_type_optimizer_alias


class GeneticProgrammingNeuralNetRegressor(GeneticProgrammingNeuralNetClassifier):
    def __init__(
        self,
        iters: int,
        pop_size: int,
        input_block_size: int = 1,
        max_hidden_block_size: int = 9,
        offset: bool = True,
        output_activation: str = "sigma",
        test_sample_ratio: float = 0.5,
        optimizer: Union[Type[SelfCGP], Type[GeneticProgramming]] = SelfCGP,
        optimizer_args: Optional[dict[str, Any]] = None,
        weights_optimizer: weights_type_optimizer_alias = SHADE,
        weights_optimizer_args: Optional[dict[str, Any]] = None,
        cache: bool = True,
    ):
        GeneticProgrammingNeuralNetClassifier.__init__(
            self,
            iters=iters,
            pop_size=pop_size,
            input_block_size=input_block_size,
            max_hidden_block_size=max_hidden_block_size,
            offset=offset,
            output_activation=output_activation,
            test_sample_ratio=test_sample_ratio,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            weights_optimizer=weights_optimizer,
            weights_optimizer_args=weights_optimizer_args,
            cache=cache,
        )

    def _evaluate_nets(
        self: GeneticProgrammingNeuralNetRegressor,
        weights: NDArray[np.float64],
        net: Net,
        X: NDArray[np.float64],
        targets: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        output2d = net.forward(X, weights)[:, :, 0]
        error = root_mean_square_error2d(targets, output2d)
        return error

    def _fitness_function(
        self: GeneticProgrammingNeuralNetRegressor,
        population: NDArray,
        X: NDArray[np.float64],
        targets: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        output2d = np.array([net.forward(X)[0] for net in population], dtype=np.float64)[:, :, 0]
        fitness = root_mean_square_error2d(targets, output2d)
        return fitness

    def _fit(
        self: GeneticProgrammingNeuralNetClassifier,
        X: NDArray[np.float64],
        y: NDArray[Union[np.float64, np.int64]],
    ) -> GeneticProgrammingNeuralNetClassifier:
        optimizer_args: dict[str, Any]

        if self._offset:
            X = np.hstack([X.copy(), np.ones((X.shape[0], 1))])

        n_outputs: int = 1

        X_train, X_test, y_train, y_test = train_test_split(X, y, self._test_sample_ratio)

        if self._optimizer_args is not None:
            assert (
                "iters" not in self._optimizer_args.keys()
                and "pop_size" not in self._optimizer_args.keys()
            ), """Do not set the "iters" or "pop_size" in the "optimizer_args". Instead,
              use the "SymbolicRegressionGP" arguments"""
            for arg in (
                "fitness_function",
                "uniset",
                "minimization",
            ):
                assert (
                    arg not in self._optimizer_args.keys()
                ), f"""Do not set the "{arg}"
                to the "optimizer_args". It is defined automatically"""
            optimizer_args = self._optimizer_args.copy()

        else:
            optimizer_args = {}

        uniset: UniversalSet = self._get_uniset(X)

        optimizer_args["fitness_function"] = lambda population: self._fitness_function(
            population, X_test, y_test
        )
        optimizer_args["genotype_to_phenotype"] = lambda trees: self._genotype_to_phenotype(
            X_train, y_train, trees, n_outputs
        )

        optimizer_args["iters"] = self._iters
        optimizer_args["pop_size"] = self._pop_size
        optimizer_args["uniset"] = uniset
        optimizer_args["minimization"] = True

        self._optimizer = self._optimizer_class(**optimizer_args)
        self._optimizer.fit()

        return self

    def _prepare_output(
        self: GeneticProgrammingNeuralNetClassifier, output: NDArray[np.float64]
    ) -> Union[NDArray[np.float64], NDArray[np.int64]]:
        return output[:, 0]
