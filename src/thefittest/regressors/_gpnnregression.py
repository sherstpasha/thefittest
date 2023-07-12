from ..classifiers import GeneticProgrammingNeuralNetClassifier
from ..base._model import Model
from typing import Optional
from typing import Union
from ..optimizers import SelfCGP
from ..base import FunctionalNode
from ..base import TerminalNode
from ..base import EphemeralNode
from ..base import UniversalSet
from ..base import Tree
from ..tools import donothing
from numpy.typing import NDArray
import numpy as np
from ..tools.operators import Add
from ..tools.operators import More
from ..base._net import HiddenBlock
from ..base._net import Net
from ..tools.metrics import root_mean_square_error2d
from ..optimizers import OptimizerAnyType
from ..optimizers import optimizer_binary_coded
from ..optimizers import OptimizerTreeType
from ..optimizers import SHADE
from ..tools.random import train_test_split
from ..tools.random import float_population
from ..tools.transformations import GrayCode


# на выходе будет просто сигма
class GeneticProgrammingNeuralNetRegressor(
        GeneticProgrammingNeuralNetClassifier):
    def __init__(self,
                 iters: int,
                 pop_size: int,
                 input_block_size: int = 1,
                 max_hidden_block_size: int = 9,
                 offset: bool = True,
                 test_sample_ratio: float = 0.5,
                 no_increase_num: Optional[int] = None,
                 show_progress_each: Optional[int] = None,
                 keep_history: bool = False,
                 optimizer: OptimizerTreeType = SelfCGP,
                 optimizer_weights: OptimizerAnyType = SHADE,
                 optimizer_weights_bounds: tuple = (-2, 2),
                 optimizer_weights_eval_num: int = 10000,
                 optimizer_weights_n_bit: int = 16,
                 cache: bool = True):
        GeneticProgrammingNeuralNetClassifier.__init__(
            self,
            iters=iters,
            pop_size=pop_size,
            input_block_size=input_block_size,
            max_hidden_block_size=max_hidden_block_size,
            offset=offset,
            test_sample_ratio=test_sample_ratio,
            no_increase_num=no_increase_num,
            show_progress_each=show_progress_each,
            keep_history=keep_history,
            optimizer=optimizer,
            optimizer_weights=optimizer_weights,
            optimizer_weights_bounds=optimizer_weights_bounds,
            optimizer_weights_eval_num=optimizer_weights_eval_num,
            optimizer_weights_n_bit=optimizer_weights_n_bit,
            cache=cache)

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
                        zip(hidden_id, [node._value._activ]*len(hidden_id)))
                    n = end
                    unit = Net(hidden_layers=[hidden_id], activs=activs)
                pack.append(unit)
        end = n + n_outputs
        output_id = set(range(n, end))
        activs = dict(
            zip(output_id, [0]*len(output_id)))
        to_return = pack[0] > Net(outputs=output_id, activs=activs)
        to_return = to_return._fix(set(range(n_variables)))
        return to_return

    def _evaluate_nets(self,
                       weights: np.ndarray,
                       net,
                       X: np.ndarray,
                       targets: np.ndarray) -> float:

        output2d = net.forward(X, weights)[:,:,0]
        error = root_mean_square_error2d(targets, output2d)
        return error

    def _fitness_function(self,
                          population: np.ndarray,
                          X,
                          targets: np.ndarray) -> np.ndarray:
        output2d = np.array([net.forward(X)[0]
                            for net in population], dtype=np.float64)[:,:,0]
        fitness = root_mean_square_error2d(targets, output2d)
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

        n_outputs = 1

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, self._test_sample_ratio)

        self._optimizer._fitness_function = \
            lambda population: self._fitness_function(
                population, X_test, y_test)

        self._optimizer._genotype_to_phenotype =\
            lambda trees: self._genotype_to_phenotype(
                X_train, y_train, trees, n_outputs)

        self._optimizer._uniset = self._define_uniset(X)
        self._optimizer.fit()

        return self

    def _predict(self, X):
        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        fittest = self._optimizer.get_fittest()
        genotype, phenotype, fitness = fittest.get()

        output = phenotype.forward(X)[0,:,0]
        y_pred = output
        return y_pred
