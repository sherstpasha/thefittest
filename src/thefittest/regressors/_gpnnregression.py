from typing import Optional

import numpy as np

from ..classifiers import GeneticProgrammingNeuralNetClassifier
from ..optimizers import OptimizerStringType
from ..optimizers import OptimizerTreeType
from ..optimizers import SHADE
from ..optimizers import SelfCGP
from ..tools.metrics import root_mean_square_error2d
from ..tools.random import train_test_split


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
        no_increase_num: Optional[int] = None,
        show_progress_each: Optional[int] = None,
        keep_history: bool = False,
        optimizer: OptimizerTreeType = SelfCGP,
        optimizer_weights: OptimizerStringType = SHADE,
        optimizer_weights_bounds: tuple = (-2, 2),
        optimizer_weights_eval_num: int = 10000,
        optimizer_weights_n_bit: int = 16,
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
            no_increase_num=no_increase_num,
            show_progress_each=show_progress_each,
            keep_history=keep_history,
            optimizer=optimizer,
            optimizer_weights=optimizer_weights,
            optimizer_weights_bounds=optimizer_weights_bounds,
            optimizer_weights_eval_num=optimizer_weights_eval_num,
            optimizer_weights_n_bit=optimizer_weights_n_bit,
            cache=cache,
        )

    def _evaluate_nets(self, weights: np.ndarray, net, X: np.ndarray, targets: np.ndarray) -> float:
        output2d = net.forward(X, weights)[:, :, 0]
        error = root_mean_square_error2d(targets, output2d)
        return error

    def _fitness_function(self, population: np.ndarray, X, targets: np.ndarray) -> np.ndarray:
        output2d = np.array([net.forward(X)[0] for net in population], dtype=np.float64)[:, :, 0]
        fitness = root_mean_square_error2d(targets, output2d)
        return fitness

    def _fit(self, X: np.ndarray, y: np.ndarray):
        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        n_outputs = 1

        X_train, X_test, y_train, y_test = train_test_split(X, y, self._test_sample_ratio)

        self.optimizer._fitness_function = lambda population: self._fitness_function(
            population, X_test, y_test
        )

        self.optimizer._genotype_to_phenotype = lambda trees: self._genotype_to_phenotype(
            X_train, y_train, trees, n_outputs
        )

        self.optimizer._uniset = self._define_uniset(X)
        self.optimizer.fit()

        return self

    def _predict(self, X):
        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        fittest = self.optimizer.get_fittest().get()

        output = fittest["phenotype"].forward(X)[0, :, 0]
        y_pred = output
        return y_pred
