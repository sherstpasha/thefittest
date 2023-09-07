from typing import Tuple
from typing import Optional
import numpy as np
from ..optimizers import SHADE
from ..optimizers import OptimizerAnyType
from ..optimizers import optimizer_binary_coded
from ..tools.metrics import root_mean_square_error2d
from ..classifiers import MLPClassifierEA


class MLPRegressorrEA(MLPClassifierEA):
    def __init__(
            self,
            iters: int,
            pop_size: int,
            hidden_layers: Tuple,
            activation: str = 'sigma',
            output_activation: str = 'sigma',
            offset: bool = True,
            no_increase_num: Optional[int] = None,
            show_progress_each: Optional[int] = None,
            keep_history: bool = False,
            optimizer_weights: OptimizerAnyType = SHADE,
            optimizer_weights_bounds: tuple = (-10, 10),
            optimizer_weights_n_bit: int = 16):

        MLPClassifierEA.__init__(self,
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
                                 optimizer_weights_n_bit=optimizer_weights_n_bit)

    def _evaluate_nets(self,
                       weights: np.ndarray,
                       net,
                       X: np.ndarray,
                       targets: np.ndarray) -> float:

        output2d = net.forward(X, weights)[:, :, 0]
        error = root_mean_square_error2d(targets, output2d)
        return error

    def _fitness_function(self,
                          population: np.ndarray,
                          X,
                          targets: np.ndarray) -> np.ndarray:
        output2d = np.array([net.forward(X)[0]
                            for net in population], dtype=np.float64)[:, :, 0]
        fitness = root_mean_square_error2d(targets, output2d)
        return fitness

    def _fit(self, X, y):

        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        if self.optimizer_weights in optimizer_binary_coded:
            self._train_func = self._train_net_bit
        else:
            self._train_func = self._train_net

        n_inputs = X.shape[1]
        n_outputs = len(set(y))

        self._defitne_net(n_inputs, n_outputs)

        self._net._weights = self._train_func(self._net, X, y)
        return self

    def _predict(self, X):
        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        output = self._net.forward(X)[0,:,0]
        y_pred = output
        return y_pred
