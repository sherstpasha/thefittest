from ..base._model import Model
from typing import Optional
from typing import Union
from ..optimizers import SelfCGP
from ..optimizers import GeneticProgramming
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
from ..tools.metrics import categorical_crossentropy3d
from ..optimizers import OptimizerStringType
from ..optimizers import jDE, SHADE
from ..tools.random import train_test_split_stratified


class GeneticProgrammingNeuralNetClassifier(Model):
    def __init__(self,
                 iters: int,
                 pop_size: int,
                 no_increase_num: Optional[int] = None,
                 show_progress_each: Optional[int] = None,
                 keep_history: bool = False,
                 optimizer: Union[SelfCGP, GeneticProgramming] = SelfCGP,
                 optimizer_weights: OptimizerStringType = SHADE,
                 input_block_size: int = 1,
                 max_hidden_block_size: int = 5,
                 test_sample_ratio: float = 0.5):

        self._optimizer = optimizer(fitness_function=donothing,
                                    genotype_to_phenotype=donothing,
                                    uniset=UniversalSet,
                                    iters=iters,
                                    pop_size=pop_size,
                                    optimal_value=None,
                                    no_increase_num=no_increase_num,
                                    show_progress_each=show_progress_each,
                                    keep_history=keep_history,
                                    minimization=True)
        self._optimizer_weights = optimizer_weights
        self._input_block_size = input_block_size
        self._max_hidden_block_size = max_hidden_block_size
        self._test_sample_ratio = test_sample_ratio

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
                    end = n + node._value.size
                    hidden_id = set(range(n, end))
                    activs = dict(
                        zip(hidden_id, [node._value.activ]*len(hidden_id)))
                    n = end
                    unit = Net(hidden_layers=[hidden_id], activs=activs)
                pack.append(unit)
        end = n + n_outputs
        to_return = pack[0] > Net(outputs=set(range(n, end)))
        to_return = to_return._fix(set(range(n_variables)))
        return to_return

    def _evaluate_nets(self,
                       weights: np.ndarray,
                       net,
                       X: np.ndarray,
                       targets: np.ndarray) -> float:

        output3d = net.forward_softmax(X, weights)
        error = categorical_crossentropy3d(targets, output3d)
        return error

# итерации и популяция в параметры алгоритма вывыести. Написать возможность обучать с помошью ГА линейное увеличение как параметр алгоритма
# добавить смешение в набор X 
    def _train_net(self, net, X_train, proba_train):

        def fitness_function(population): return self._evaluate_nets(
            population, net, X_train, proba_train)
        
        left = np.full(shape = len(net._weights), fill_value=-2, dtype=np.float64)
        right = np.full(shape = len(net._weights), fill_value=2, dtype=np.float64)
        optimizer_weights = self._optimizer_weights(
            fitness_function, donothing, 100, 100, left, right, minimization = True)
        optimizer_weights.fit()
        fittest = optimizer_weights.get_fittest()
        genotype, phenotype, fitness = fittest.get()
        return phenotype

    def _genotype_to_phenotype(self,
                               X_train: NDArray[np.float64],
                               proba_train: NDArray[np.float64],
                               population_g: NDArray) -> NDArray:
        n_variables = X_train.shape[1]
        n_outputs = proba_train.shape[1]

        population_ph = []
        for individ_g in population_g:
            net = self._genotype_to_phenotype_tree(
                n_variables, n_outputs, individ_g)
            net._weights = self._train_net(net, X_train, proba_train)
            population_ph.append(net)

        return population_ph

    def _define_uniset(self,
                       X: NDArray[np.float64]) -> UniversalSet:
        n_dimension = X.shape[1]

        cut_id = np.arange(self._input_block_size,
                           n_dimension,
                           self._input_block_size)
        variables_pool = np.split(np.arange(n_dimension), cut_id)

        functional_set = (FunctionalNode(Add()),
                          FunctionalNode(More()))

        def random_hidden_block(): return HiddenBlock()(self._max_hidden_block_size)

        terminal_set = [TerminalNode(set(variables), 'in{}'.format(i))
                        for i, variables in enumerate(variables_pool)]
        terminal_set.append(EphemeralNode(random_hidden_block))

        uniset = UniversalSet(functional_set, terminal_set)
        return uniset

    def _fitness_function(self,
                          population: np.ndarray,
                          X,
                          targets: np.ndarray) -> np.ndarray:
        output3d = np.array([net.forward_softmax(X)[0]
                            for net in population], dtype=np.float64)
        fitness = categorical_crossentropy3d(targets, output3d)
        return fitness

    def _fit(self,
             X: np.ndarray,
             y: np.ndarray):
        n_inputs = X.shape[1]
        n_outputs = len(set(y))
        eye = np.eye(n_outputs)
        # target_probas = eye[y]

        X_train, X_test, y_train, y_test = train_test_split_stratified(
            X, y, self._test_sample_ratio)
        proba_test = eye[y_test]
        proba_train = eye[y_train]

        self._optimizer._fitness_function = \
            lambda population: self._fitness_function(
                population, X_test, proba_test)

        self._optimizer._genotype_to_phenotype =\
            lambda trees: self._genotype_to_phenotype(
                X_train, proba_train, trees)

        self._optimizer._uniset = self._define_uniset(X)
        self._optimizer.fit()

        return self
    
    def _predict(self, X):
        fittest = self._optimizer.get_fittest()
        genotype, phenotype, fitness = fittest.get()

        output = phenotype.forward_softmax(X)[0]
        y_pred = np.argmax(output, axis=1)
        return y_pred


