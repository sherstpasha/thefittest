from typing import Optional

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
from ..optimizers import OptimizerStringType
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
    def __init__(
        self,
        iters: int,
        pop_size: int,
        input_block_size: int = 1,
        max_hidden_block_size: int = 9,
        offset: bool = True,
        output_activation: str = "softmax",
        test_sample_ratio: float = 0.5,
        no_increase_num: Optional[int] = None,
        show_progress_each: Optional[int] = None,
        keep_history: bool = False,
        optimizer: OptimizerTreeType = SelfCGP,
        optimizer_weights: OptimizerStringType = SHADE,
        optimizer_weights_bounds: tuple = (-10, 10),
        optimizer_weights_eval_num: int = 10000,
        optimizer_weights_n_bit: int = 16,
        cache: bool = True,
    ):
        Model.__init__(self)
        self._input_block_size = input_block_size
        self._max_hidden_block_size = max_hidden_block_size
        self._offset = offset
        self._output_activation = output_activation
        self._test_sample_ratio = test_sample_ratio
        self.optimizer = optimizer(
            fitness_function=donothing,
            uniset=UniversalSet,
            iters=iters,
            pop_size=pop_size,
            optimal_value=None,
            no_increase_num=no_increase_num,
            show_progress_each=show_progress_each,
            keep_history=keep_history,
            minimization=True,
        )
        self._optimizer_weights_type = optimizer_weights
        self.optimizer_weights = self._init_optimizer_weights(
            optimizer_weights, optimizer_weights_eval_num
        )
        self._optimizer_weights_bounds = optimizer_weights_bounds
        self._optimizer_weights_eval_num = optimizer_weights_eval_num
        self._optimizer_weights_n_bit = optimizer_weights_n_bit
        self._cache_condition = cache
        self._cache = {}

        Model.__init__(self)

    def _init_optimizer_weights(
        self, optimizer_weights_type: OptimizerStringType, optimizer_weights_eval_num: int
    ) -> OptimizerStringType:
        iters = int(np.sqrt(optimizer_weights_eval_num))
        pop_size = int(optimizer_weights_eval_num / iters)

        if optimizer_weights_type in optimizer_binary_coded:
            optimizer = optimizer_weights_type(
                fitness_function=donothing,
                iters=iters,
                pop_size=pop_size,
                str_len=1,
                minimization=True,
            )

        else:
            optimizer = optimizer_weights_type(
                fitness_function=donothing,
                iters=iters,
                pop_size=pop_size,
                left=np.empty(shape=(1), dtype=np.float64),
                right=np.empty(shape=(1), dtype=np.float64),
                minimization=True,
            )
        return optimizer

    def _train_net(self, net, X_train, proba_train):
        self.optimizer_weights.clear()

        def fitness_function(population):
            return self._evaluate_nets(population, net, X_train, proba_train)

        left = np.full(
            shape=len(net._weights), fill_value=self._optimizer_weights_bounds[0], dtype=np.float64
        )
        right = np.full(
            shape=len(net._weights), fill_value=self._optimizer_weights_bounds[1], dtype=np.float64
        )

        initial_population = float_population(self.optimizer_weights._pop_size, left, right)
        initial_population[0] = net._weights.copy()

        if self._optimizer_weights_type in optimizer_binary_coded:
            parts = np.full(
                shape=len(net._weights), fill_value=self._optimizer_weights_n_bit, dtype=np.int64
            )

            genotype_to_phenotype = GrayCode(fit_by="parts").fit(left, right, parts)

            self.optimizer_weights._genotype_to_phenotype = genotype_to_phenotype.transform
            self.optimizer_weights._str_len = np.sum(parts)
            self.optimizer_weights._update_pool()

            initial_population = genotype_to_phenotype.inverse_transform(initial_population)
        else:
            self.optimizer_weights._left = left
            self.optimizer_weights._right = right

        self.optimizer_weights._fitness_function = fitness_function
        self.optimizer_weights.set_strategy(initial_population=initial_population)
        self.optimizer_weights.fit()

        fittest = self.optimizer_weights.get_fittest().get()

        return fittest["phenotype"]

    def _genotype_to_phenotype_tree(self, n_variables: int, n_outputs: int, tree: Tree) -> Net:
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
                    activs = dict(zip(hidden_id, [node._value._activ] * len(hidden_id)))
                    n = end
                    unit = Net(hidden_layers=[hidden_id], activs=activs)
                pack.append(unit)
        end = n + n_outputs
        output_id = set(range(n, end))
        activs = dict(zip(output_id, [ACTIV_NAME_INV[self._output_activation]] * len(output_id)))
        to_return = pack[0] > Net(outputs=output_id, activs=activs)
        to_return = to_return._fix(set(range(n_variables)))
        return to_return

    def _evaluate_nets(self, weights: np.ndarray, net, X: np.ndarray, targets: np.ndarray) -> float:
        output3d = net.forward(X, weights)
        error = categorical_crossentropy3d(targets, output3d)
        return error

    def _genotype_to_phenotype(
        self,
        X_train: NDArray[np.float64],
        proba_train: NDArray[np.float64],
        population_g: NDArray,
        n_outputs,
    ) -> NDArray:
        n_variables = X_train.shape[1]

        population_ph = np.empty(shape=len(population_g), dtype=object)
        need_to_train_cond = np.full_like(population_ph, fill_value=False, dtype=bool)

        need_to_train_list = []
        need_to_train_list_str = []

        for i, individ_g in enumerate(population_g):
            str_tree = str(individ_g)
            if str_tree in self._cache.keys():
                population_ph[i] = self._cache[str_tree].copy()
            else:
                net = self._genotype_to_phenotype_tree(n_variables, n_outputs, individ_g)

                need_to_train_list.append(net)
                need_to_train_list_str.append(str_tree)
                need_to_train_cond[i] = True

        trained_weights = list(
            map(lambda net: self._train_net(net, X_train, proba_train), need_to_train_list)
        )

        for net, weight in zip(need_to_train_list, trained_weights):
            net._weights = weight
        population_ph[need_to_train_cond] = need_to_train_list

        new_cache = dict(zip(need_to_train_list_str, need_to_train_list))
        self._cache = dict(list(self._cache.items()) + list(new_cache.items()))

        return population_ph

    def _define_uniset(self, X: NDArray[np.float64]) -> UniversalSet:
        if self._offset:
            n_dimension = X.shape[1] - 1
        else:
            n_dimension = X.shape[1]

        cut_id = np.arange(self._input_block_size, n_dimension, self._input_block_size)
        variables_pool = np.split(np.arange(n_dimension), cut_id)

        functional_set = (FunctionalNode(Add()), FunctionalNode(More()))

        def random_hidden_block():
            return HiddenBlock(self._max_hidden_block_size)

        terminal_set = [
            TerminalNode(set(variables), "in{}".format(i))
            for i, variables in enumerate(variables_pool)
        ]
        if self._offset:
            terminal_set.append(
                TerminalNode(value=set([n_dimension]), name="in{}".format(len(variables_pool)))
            )
        terminal_set.append(EphemeralNode(random_hidden_block))

        uniset = UniversalSet(functional_set, terminal_set)
        return uniset

    def _fitness_function(self, population: np.ndarray, X, targets: np.ndarray) -> np.ndarray:
        output3d = np.array([net.forward(X)[0] for net in population], dtype=np.float64)
        fitness = categorical_crossentropy3d(targets, output3d)
        return fitness

    def _fit(self, X: np.ndarray, y: np.ndarray):
        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        n_outputs = len(set(y))
        eye = np.eye(n_outputs)

        X_train, X_test, y_train, y_test = train_test_split_stratified(
            X, y, self._test_sample_ratio
        )
        proba_test = eye[y_test]
        proba_train = eye[y_train]

        self.optimizer._fitness_function = lambda population: self._fitness_function(
            population, X_test, proba_test
        )

        self.optimizer._genotype_to_phenotype = lambda trees: self._genotype_to_phenotype(
            X_train, proba_train, trees, n_outputs
        )

        self.optimizer._uniset = self._define_uniset(X)
        self.optimizer.fit()

        return self

    def _predict(self, X):
        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        fittest = self.optimizer.get_fittest()
        genotype, phenotype, fitness = fittest.get().values()

        output = phenotype.forward(X)[0]
        y_pred = np.argmax(output, axis=1)
        return y_pred
