from typing import Tuple
from typing import List
from typing import Optional
from typing import Union
import numpy as np
from ..tools import donothing
from ..tools.metrics import categorical_crossentropy
from ..optimizers import SHADE
from ..base._net import Net
from ..optimizers import OptimizerAnyType
from ..optimizers import optimizer_binary_coded
from ..tools.random import float_population
from ..tools.metrics import categorical_crossentropy3d
from ..tools.transformations import GrayCode
from ..base._model import Model
from ..base._net import ACTIV_NAME_INV



class MLPClassifierEA(Model):
    def __init__(
            self,
            iters: int,
            pop_size: int,
            hidden_layers: Tuple,
            activation: str = 'sigma',
            output_activation: str = 'softmax',
            offset: bool = True,
            no_increase_num: Optional[int] = None,
            show_progress_each: Optional[int] = None,
            keep_history: bool = False,
            optimizer_weights: OptimizerAnyType = SHADE,
            optimizer_weights_bounds: tuple = (-10, 10),
            optimizer_weights_n_bit: int = 16):

        Model.__init__(self)
        self._iters = iters
        self._pop_size = pop_size
        self._hidden_layers = hidden_layers
        self._activation = activation
        self._output_activation = output_activation
        self._offset = offset
        self._no_increase_num = no_increase_num
        self._show_progress_each = show_progress_each
        self._keep_history = keep_history
        self._optimizer_weights = optimizer_weights
        self._optimizer_weights_bounds = optimizer_weights_bounds
        self._optimizer_weights_n_bit = optimizer_weights_n_bit
        self._train_func: Union[self._train_net, self._train_net_bit]

        self._net: Net


    def _defitne_net(self, n_inputs, n_outputs):
        start = 0
        end = n_inputs
        inputs_id = set(range(start, end))

        net = Net(inputs=inputs_id)

        for n_layer in self._hidden_layers:
            start = end
            end = end + n_layer
            inputs_id = set([n_inputs-1])
            hidden_id = set(range(start, end))
            activs = dict(zip(
                hidden_id, [ACTIV_NAME_INV[self._activation]]*len(hidden_id)))

            if self._offset:
                layer_net = Net(inputs=inputs_id) > Net(
                    hidden_layers=[hidden_id], activs=activs)
            else:
                layer_net = Net(hidden_layers=[hidden_id], activs=activs)

            net = net > layer_net

        start = end
        end = end + n_outputs
        inputs_id = set([n_inputs-1])
        output_id = set(range(start, end))
        activs = dict(
            zip(output_id, [ACTIV_NAME_INV[self._output_activation]]*len(output_id)))

        if self._offset:
            layer_net = Net(inputs=inputs_id) > Net(
                outputs=output_id, activs=activs)
        else:
            layer_net = Net(outputs=output_id, activs=activs)

        net = net > layer_net

        self._net = net
        self._net._get_order()

    def _evaluate_nets(self,
                       weights: np.ndarray,
                       net,
                       X: np.ndarray,
                       targets: np.ndarray) -> float:

        output3d = net.forward(X, weights)
        error = categorical_crossentropy3d(targets, output3d)
        return error

    def _train_net(self, net, X_train, proba_train):

        def fitness_function(population): return self._evaluate_nets(
            population, net, X_train, proba_train)

        left = np.full(shape=len(net._weights),
                       fill_value=self._optimizer_weights_bounds[0],
                       dtype=np.float64)
        right = np.full(shape=len(net._weights),
                        fill_value=self._optimizer_weights_bounds[1],
                        dtype=np.float64)

        self._optimizer_weights = self._optimizer_weights(fitness_function=fitness_function,
                                                          genotype_to_phenotype=donothing,
                                                          iters=self._iters,
                                                          pop_size=self._pop_size,
                                                          left=left,
                                                          right=right,
                                                          minimization=True,
                                                          no_increase_num=self._no_increase_num,
                                                          keep_history=self._keep_history,
                                                          show_progress_each=self._show_progress_each)

        initial_population = float_population(self._pop_size, left, right)
        initial_population[0] = net._weights.copy()

        self._optimizer_weights.set_strategy(
            initial_population=initial_population)
        self._optimizer_weights.fit()
        fittest = self._optimizer_weights.get_fittest()
        genotype, phenotype, fitness = fittest.get()

        return phenotype

    def _train_net_bit(self, net, X_train, proba_train):

        def fitness_function(population): return self._evaluate_nets(
            population, net, X_train, proba_train)

        left = np.full(shape=len(net._weights),
                       fill_value=self._optimizer_weights_bounds[0],
                       dtype=np.float64)
        right = np.full(shape=len(net._weights),
                        fill_value=self._optimizer_weights_bounds[1],
                        dtype=np.float64)
        parts = np.full(shape=len(net._weights),
                        fill_value=self._optimizer_weights_n_bit,
                        dtype=np.int64)

        genotype_to_phenotype = GrayCode(
            fit_by='parts').fit(left, right, parts)

        str_len = np.sum(parts)

        self._optimizer_weights = self._optimizer_weights(
            fitness_function=fitness_function,
            genotype_to_phenotype=genotype_to_phenotype.transform,
            iters=self._iters,
            pop_size=self._pop_size,
            str_len=str_len,
            minimization=True,
            no_increase_num=self._no_increase_num,
            keep_history=self._keep_history,
            show_progress_each=self._show_progress_each)

        initial_population = float_population(self._pop_size, left, right)
        initial_population[0] = net._weights.copy()

        initial_population_bit = genotype_to_phenotype.inverse_transform(
            initial_population)
        self._optimizer_weights.set_strategy(
            initial_population=initial_population_bit)
        self._optimizer_weights.fit()
        fittest = self._optimizer_weights.get_fittest()
        genotype, phenotype, fitness = fittest.get()

        return phenotype

    def _fit(self, X, y):

        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        if self._optimizer_weights in optimizer_binary_coded:
            self._train_func = self._train_net_bit
        else:
            self._train_func = self._train_net

        n_inputs = X.shape[1]
        n_outputs = len(set(y))
        eye = np.eye(n_outputs)
        target_probas = eye[y]

        self._defitne_net(n_inputs, n_outputs)

        self._net._weights = self._train_func(self._net, X, target_probas)
        return self

    def _predict(self, X):
        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        output = self._net.forward(X)[0]
        y_pred = np.argmax(output, axis=1)
        return y_pred