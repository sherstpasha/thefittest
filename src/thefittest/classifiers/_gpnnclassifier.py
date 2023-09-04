from typing import Optional
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..base import EphemeralNode
from ..base import FunctionalNode
from ..base import TerminalNode
from ..base import Tree
from ..base import UniversalSet
from ..base._model import Model
from ..base._net import ACTIV_NAME_INV
from ..base._net import HiddenBlock
from ..base._net import Net
from ..optimizers import OptimizerAnyType
from ..optimizers import OptimizerTreeType
from ..optimizers import SHADE
from ..optimizers import SelfCGP
from ..optimizers import optimizer_binary_coded
from ..tools import donothing
from ..tools.metrics import categorical_crossentropy3d
from ..tools.operators import Add
from ..tools.operators import More
from ..tools.random import float_population
from ..tools.random import train_test_split_stratified
from ..tools.transformations import GrayCode


class GeneticProgrammingNeuralNetClassifier(Model):
    def __init__(self,
                 iters: int,
                 pop_size: int,
                 input_block_size: int = 1,
                 max_hidden_block_size: int = 9,
                 offset: bool = True,
                 output_activation: str = 'softmax',
                 test_sample_ratio: float = 0.5,
                 no_increase_num: Optional[int] = None,
                 show_progress_each: Optional[int] = None,
                 keep_history: bool = False,
                 optimizer: OptimizerTreeType = SelfCGP,
                 optimizer_weights: OptimizerAnyType = SHADE,
                 optimizer_weights_bounds: tuple = (-10, 10),
                 optimizer_weights_eval_num: int = 10000,
                 optimizer_weights_n_bit: int = 16,
                 cache: bool = True):

        Model.__init__(self)
        self._input_block_size = input_block_size
        self._max_hidden_block_size = max_hidden_block_size
        self._offset = offset
        self._output_activation = output_activation
        self._test_sample_ratio = test_sample_ratio
        self._optimizer = optimizer(fitness_function=donothing,
                                    uniset=UniversalSet,
                                    iters=iters,
                                    pop_size=pop_size,
                                    optimal_value=None,
                                    no_increase_num=no_increase_num,
                                    show_progress_each=show_progress_each,
                                    keep_history=keep_history,
                                    minimization=True)
        self._optimizer_weights = optimizer_weights
        self._optimizer_weights_bounds = optimizer_weights_bounds
        self._optimizer_weights_eval_num = optimizer_weights_eval_num
        self._optimizer_weights_n_bit = optimizer_weights_n_bit
        self._cache_condition = cache
        self._cache = {}
        self._train_func: Union[self._train_net, self._train_net_bit]

        Model.__init__(self)

    def _genotype_to_phenotype_tree(self,
                                    n_variables: int,
                                    n_outputs: int,
                                    tree: Tree) -> Net:
        pack = []
        n = n_variables
        for node in reversed(tree._nodes):
            args = []
            for _ in range(node._n_args):
                args.append(pack.pop())
            if node.is_functional():
                pack.append(node._value(*args))
            else:
                if type(node) is TerminalNode:
                    unit = Net(inputs=node._value)
                else:
                    end = n + node._value._size
                    hidden_id = set(range(n, end))
                    activs = dict(
                        zip(hidden_id, [node._value._activ] * len(hidden_id)))
                    n = end
                    unit = Net(hidden_layers=[hidden_id], activs=activs)
                pack.append(unit)
        end = n + n_outputs
        output_id = set(range(n, end))
        activs = dict(
            zip(output_id, [ACTIV_NAME_INV[self._output_activation]] * len(output_id)))
        to_return = pack[0] > Net(outputs=output_id, activs=activs)
        to_return = to_return._fix(set(range(n_variables)))
        return to_return

    def _evaluate_nets(self,
                       weights: np.ndarray,
                       net,
                       X: np.ndarray,
                       targets: np.ndarray) -> float:

        output3d = net.forward(X, weights)
        error = categorical_crossentropy3d(targets, output3d)
        return error

    def _train_net(self, net, X_train, proba_train):

        def fitness_function(population):
            return self._evaluate_nets(population, net, X_train, proba_train)

        left = np.full(shape=len(net._weights),
                       fill_value=self._optimizer_weights_bounds[0],
                       dtype=np.float64)
        right = np.full(shape=len(net._weights),
                        fill_value=self._optimizer_weights_bounds[1],
                        dtype=np.float64)
        iters = int(np.sqrt(self._optimizer_weights_eval_num))
        pop_size = int(self._optimizer_weights_eval_num / iters)

        optimizer_weights = self._optimizer_weights(fitness_function=fitness_function,
                                                    iters=iters,
                                                    pop_size=pop_size,
                                                    left=left,
                                                    right=right,
                                                    minimization=True)

        initial_population = float_population(pop_size, left, right)
        initial_population[0] = net._weights.copy()

        optimizer_weights.set_strategy(initial_population=initial_population)
        optimizer_weights.fit()
        fittest = optimizer_weights.get_fittest()
        genotype, phenotype, fitness = fittest.get()
        return phenotype

    def _train_net_bit(self, net, X_train, proba_train):

        def fitness_function(population):
            return self._evaluate_nets(population, net, X_train, proba_train)

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

        iters = int(np.sqrt(self._optimizer_weights_eval_num))
        pop_size = int(self._optimizer_weights_eval_num / iters)
        str_len = np.sum(parts)

        optimizer_weights = self._optimizer_weights(
            fitness_function=fitness_function,
            genotype_to_phenotype=genotype_to_phenotype.transform,
            iters=iters,
            pop_size=pop_size,
            str_len=str_len,
            minimization=True)

        initial_population = float_population(pop_size, left, right)
        initial_population[0] = net._weights.copy()

        initial_population_bit = genotype_to_phenotype.inverse_transform(
            initial_population)
        optimizer_weights.set_strategy(
            initial_population=initial_population_bit)
        optimizer_weights.fit()
        fittest = optimizer_weights.get_fittest()
        genotype, phenotype, fitness = fittest.get()

        return phenotype

    def _genotype_to_phenotype(self,
                               X_train: NDArray[np.float64],
                               proba_train: NDArray[np.float64],
                               population_g: NDArray,
                               n_outputs) -> NDArray:
        n_variables = X_train.shape[1]

        population_ph = np.empty(shape=len(population_g), dtype=object)
        need_to_train_cond = np.full_like(
            population_ph, fill_value=False, dtype=bool)

        neet_to_train_list = []
        neet_to_train_list_str = []

        for i, individ_g in enumerate(population_g):
            str_tree = str(individ_g)
            if str_tree in self._cache.keys():
                population_ph[i] = self._cache[str_tree].copy()
            else:
                net = self._genotype_to_phenotype_tree(
                    n_variables, n_outputs, individ_g)

                neet_to_train_list.append(net)
                neet_to_train_list_str.append(str_tree)
                need_to_train_cond[i] = True

        trained_weights = list(map(
            lambda net: self._train_func(net, X_train, proba_train), neet_to_train_list))

        for net, weight in zip(neet_to_train_list, trained_weights):
            net._weights = weight
        population_ph[need_to_train_cond] = neet_to_train_list

        new_cache = dict(zip(neet_to_train_list_str, neet_to_train_list))
        self._cache = dict(list(self._cache.items()) + list(new_cache.items()))

        return population_ph

    def _define_uniset(self,
                       X: NDArray[np.float64]) -> UniversalSet:
        if self._offset:
            n_dimension = X.shape[1] - 1
        else:
            n_dimension = X.shape[1]

        cut_id = np.arange(self._input_block_size,
                           n_dimension,
                           self._input_block_size)
        variables_pool = np.split(np.arange(n_dimension), cut_id)

        functional_set = (FunctionalNode(Add()),
                          FunctionalNode(More()))

        def random_hidden_block():
            return HiddenBlock(self._max_hidden_block_size)

        terminal_set = [TerminalNode(set(variables), 'in{}'.format(i))
                        for i, variables in enumerate(variables_pool)]
        if self._offset:
            terminal_set.append(TerminalNode({[n_dimension], }),
                                'in{}'.format(len(variables_pool)))
        terminal_set.append(EphemeralNode(random_hidden_block))

        uniset = UniversalSet(functional_set, terminal_set)
        return uniset

    def _fitness_function(self,
                          population: np.ndarray,
                          X,
                          targets: np.ndarray) -> np.ndarray:
        output3d = np.array([net.forward(X)[0]
                            for net in population], dtype=np.float64)
        fitness = categorical_crossentropy3d(targets, output3d)
        return fitness

    def _fit(self,
             X: np.ndarray,
             y: np.ndarray):

        if self._optimizer_weights in optimizer_binary_coded:
            self._train_func = self._train_net_bit
        else:
            self._train_func = self._train_net

        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        n_outputs = len(set(y))
        eye = np.eye(n_outputs)

        X_train, X_test, y_train, y_test = train_test_split_stratified(
            X, y, self._test_sample_ratio)
        proba_test = eye[y_test]
        proba_train = eye[y_train]

        self._optimizer._fitness_function = \
            lambda population: self._fitness_function(
                population, X_test, proba_test)

        self._optimizer._genotype_to_phenotype =\
            lambda trees: self._genotype_to_phenotype(
                X_train, proba_train, trees, n_outputs)

        self._optimizer._uniset = self._define_uniset(X)
        self._optimizer.fit()

        return self

    def _predict(self, X):
        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        fittest = self._optimizer.get_fittest()
        genotype, phenotype, fitness = fittest.get()

        output = phenotype.forward(X)[0]
        y_pred = np.argmax(output, axis=1)
        return y_pred
