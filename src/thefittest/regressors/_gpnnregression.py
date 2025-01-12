from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..base import UniversalSet
from ..classifiers import GeneticProgrammingNeuralNetClassifier
from ..classifiers._gpnnclassifier import genotype_to_phenotype_tree
from ..classifiers._gpnnclassifier import train_net
from ..classifiers._gpnnclassifier import weights_type_optimizer_alias
from ..optimizers import GeneticProgramming
from ..optimizers import SHADE
from ..optimizers import SelfCGP
from ..regressors._mlpearegressor import fitness_function as evaluate_nets
from ..tools.metrics import root_mean_square_error2d
from ..tools.random import train_test_split


def fitness_function(
    population: NDArray,
    X: NDArray[np.float32],
    targets: NDArray[np.float32],
    net_size_penalty: float,
) -> NDArray[np.float32]:
    output2d = np.array([net.forward(X)[0] for net in population], dtype=np.float32)[:, :, 0]
    lens = np.array(list(map(len, population)))
    print(output2d.shape)
    fitness = root_mean_square_error2d(targets, output2d) + net_size_penalty * lens
    return fitness


def genotype_to_phenotype(
    population_g: NDArray,
    n_outputs: int,
    X_train: NDArray[np.float32],
    proba_train: NDArray[np.float32],
    weights_optimizer_args: Dict,
    weights_optimizer_class: weights_type_optimizer_alias,
    output_activation: str,
    offset: bool,
) -> NDArray:
    n_variables: int = X_train.shape[1]

    population_ph = np.array(
        [
            train_net(
                genotype_to_phenotype_tree(
                    individ_g, n_variables, n_outputs, output_activation, offset
                ),
                X_train=X_train,
                proba_train=proba_train,
                weights_optimizer_args=weights_optimizer_args,
                weights_optimizer_class=weights_optimizer_class,
                fitness_function=evaluate_nets,
            )
            for individ_g in population_g
        ],
        dtype=object,
    )

    return population_ph


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
        net_size_penalty: float = 0.0,
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
            net_size_penalty=net_size_penalty,
        )

    def _fit(
        self: GeneticProgrammingNeuralNetClassifier,
        X: NDArray[np.float32],
        y: NDArray[Union[np.float32, np.int64]],
    ) -> GeneticProgrammingNeuralNetClassifier:
        if self._offset:
            X = np.hstack([X.copy(), np.ones((X.shape[0], 1))])

        n_outputs: int = 1

        X_train, X_test, y_train, y_test = train_test_split(X, y, self._test_sample_ratio)

        uniset: UniversalSet = self._get_uniset(X)

        self._optimizer = self._define_optimizer(
            uniset=uniset,
            n_outputs=n_outputs,
            X_train=X_train,
            target_train=y_train,
            X_test=X_test,
            target_test=y_test,
            fitness_function=fitness_function,
            evaluate_nets=evaluate_nets,
        )

        self._optimizer.fit()

        return self

    def _prepare_output(
        self: GeneticProgrammingNeuralNetClassifier, output: NDArray[np.float32]
    ) -> Union[NDArray[np.float32], NDArray[np.int64]]:
        return output[:, 0]
