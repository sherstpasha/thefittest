from typing import Optional
from typing import Union
import numpy as np
from ..tools.transformations import coefficient_determination
from ..optimizers import SelfCGP
from ..optimizers import GeneticProgramming
from ..base import FunctionalNode
from ..base import TerminalNode
from ..base import EphemeralNode
from ..base import UniversalSet
from ..base import Tree
from ..tools.operators import Add
from ..tools.operators import Mul
from ..tools.operators import Neg
from ..tools.operators import Inv
from ..tools.operators import Pow2
from ..tools.operators import Cos
from ..tools.operators import Sin
from ..tools import donothing
from ..base._model import Model


class SymbolicRegressionGP(Model):
    def __init__(self,
                 iters: int,
                 pop_size: int,
                 no_increase_num: Optional[int] = None,
                 show_progress_each: Optional[int] = None,
                 keep_history: bool = False,
                 optimizer: Union[SelfCGP, GeneticProgramming] = SelfCGP) -> None:

        self.optimizer = optimizer(fitness_function=donothing,
                                   genotype_to_phenotype=donothing,
                                   uniset=UniversalSet,
                                   iters=iters,
                                   pop_size=pop_size,
                                   optimal_value=1.,
                                   no_increase_num=no_increase_num,
                                   show_progress_each=show_progress_each,
                                   keep_history=keep_history)
        Model.__init__(self)

    def _evaluate_tree(self,
                       tree: Tree,
                       y: np.ndarray) -> float:
        y_pred = tree()*np.ones(len(y))
        fitness = coefficient_determination(y, y_pred)
        return fitness

    def _fitness_function(self,
                          trees: np.ndarray,
                          y: np.ndarray) -> np.ndarray:
        fitness = [self._evaluate_tree(tree, y) for tree in trees]
        return np.array(fitness)

    def _generator1(self) -> float:
        value = np.round(np.random.uniform(0, 10), 4)
        return value

    def _generator2(self) -> int:
        value = np.random.randint(0, 10)
        return value

    def _define_uniset(self,
                       X: np.ndarray) -> UniversalSet:
        n_dimension = X.shape[1]
        functional_set = [FunctionalNode(Add()),
                          FunctionalNode(Mul()),
                          FunctionalNode(Neg()),
                          FunctionalNode(Inv()),
                          FunctionalNode(Pow2()),
                          FunctionalNode(Cos()),
                          FunctionalNode(Sin())]

        terminal_set = [TerminalNode(X[:, i], f'x{i}')
                        for i in range(n_dimension)]
        terminal_set.extend([EphemeralNode(self._generator1),
                             EphemeralNode(self._generator2)])
        uniset = UniversalSet(functional_set, terminal_set)
        return uniset

    def _fit(self,
             X: np.ndarray,
             y: np.ndarray):
        self.optimizer._fitness_function = lambda trees: self._fitness_function(
            trees, y)
        self.optimizer._uniset = self._define_uniset(X)
        self.optimizer.fit()
        return self

    def _predict(self,
                 X: np.ndarray) -> np.ndarray:
        n_dimension = X.shape[1]
        solution = self.optimizer.get_fittest()
        genotype, *_ = solution.get()
        genotype_for_pred = genotype.set_terminals(
            {f'x{i}': X[:, i]for i in range(n_dimension)})

        y_pred = genotype_for_pred()*np.ones(len(X))
        return y_pred
