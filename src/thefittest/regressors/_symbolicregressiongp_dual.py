from __future__ import annotations

from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from ..base import Tree
import numpy as np
from numpy.typing import NDArray

from ..base import EphemeralNode
from ..base import FunctionalNode
from ..base import TerminalNode
from ..base import UniversalSet
from ..base._model import Model
from ..optimizers import DifferentialEvolution
from ..optimizers import GeneticAlgorithm
from ..optimizers import GeneticProgramming
from ..optimizers import SHADE
from ..optimizers import SHAGA
from ..optimizers import SelfCGA
from ..optimizers import SelfCGP
from ..optimizers import jDE
from ..tools.metrics import coefficient_determination
from ..tools.metrics import root_mean_square_error
from thefittest.tools.operators import (
    Mul,
    Add,
    Div,
    Sin,
    Exp,
    Sub,
    SqrtAbs,
    LogAbs,
    Cos,
)
from ..base._tree import DualNode


def mean_of_y(trees, y):
    results = []
    for tree in trees:
        results.append(tree() * np.ones(len(y)))

    # print(y.shape, np.array(results).shape, np.mean(results, axis=0).shape)
    # return
    return np.mean(results, axis=0)


def fitness_function(trees: NDArray, y: NDArray[np.float32]) -> NDArray[np.float32]:
    fitness = []
    for tree in trees:
        y_pred = mean_of_y(tree, y)
        fitness.append(root_mean_square_error(y, y_pred.astype(np.float32)))
    return np.array(fitness, dtype=np.float32)


def generator1() -> float:
    value = np.round(np.random.uniform(0, 10), 4)
    return value


def generator2() -> int:
    value = np.random.randint(0, 10)
    return value


def split_tree(tree: Tree) -> Tuple[Tree, Tree]:
    new_tree = Tree([])
    remain_tree = tree.copy()
    for i, node in enumerate(reversed(tree._nodes)):
        index = len(tree) - i - 1
        if isinstance(node._value, DualNode):
            begin, end = tree.subtree_id(index)
            new_nodes = tree._nodes[begin:end].copy()
            new_nodes[0] = node._value._bottom_node
            new_tree = Tree(nodes=new_nodes)

            remain_nodes = tree._nodes[:begin].copy() + tree._nodes[end - 1 :].copy()
            remain_nodes[begin] = node._value._top_node
            remain_tree = Tree(nodes=remain_nodes)

            break

    return (remain_tree, new_tree)


def genotype_to_phenotype(x):
    phenotypes = []
    for tree in x:
        trees = []
        remain_tree, new_tree = split_tree(tree)
        if len(new_tree) > 0:
            trees.append(new_tree)
        while True:
            remain_tree, new_tree = split_tree(remain_tree)
            if len(new_tree) > 0:
                trees.append(new_tree)
            else:
                break

        if len(remain_tree) > 0:
            trees.append(remain_tree)
        phenotypes.append(trees)

    return np.array(phenotypes, dtype=object)


class SymbolicRegressionGP_DUAL(Model):
    def __init__(
        self,
        iters: int,
        pop_size: int,
        uniset: Optional[UniversalSet] = None,
        optimizer: Union[Type[SelfCGP], Type[GeneticProgramming]] = SelfCGP,
        optimizer_args: Optional[dict[str, Any]] = None,
    ):
        Model.__init__(self)

        self._iters: int = iters
        self._pop_size: int = pop_size
        self._uniset: Optional[UniversalSet] = uniset
        self._optimizer_args: Optional[dict[str, Any]] = optimizer_args
        self._optimizer_class: Union[Type[SelfCGP], Type[GeneticProgramming]] = optimizer
        self._optimizer: Union[SelfCGP, GeneticProgramming]

    def _get_uniset(self: SymbolicRegressionGP_DUAL, X: NDArray[np.float32]) -> UniversalSet:

        n_dimension: int = X.shape[1]

        # Базовые арифметические функции
        base_functionals = [
            FunctionalNode(Add()),
            FunctionalNode(Sub()),
            FunctionalNode(Mul()),
            FunctionalNode(Div()),
            FunctionalNode(SqrtAbs()),
            FunctionalNode(Exp()),
            FunctionalNode(LogAbs()),
            FunctionalNode(Sin()),
            FunctionalNode(Cos()),
        ]

        # Переменные как терминальные узлы
        terminal_set: List[Union[TerminalNode, EphemeralNode]] = [
            TerminalNode(X[:, i], f"x{i}") for i in range(n_dimension)
        ]

        # Расширение функционального множества DualNode-ами
        functional_set: List[FunctionalNode] = base_functionals.copy()

        for term in terminal_set:
            for base_func in base_functionals:
                dual = DualNode(term, base_func)
                functional_set.append(FunctionalNode(dual))
                # Эфемеральные узлы с константами

        terminal_set.extend(
            [
                EphemeralNode(generator1),
                EphemeralNode(generator2),
            ]
        )

        # Возвращаем универсальное множество
        return UniversalSet(tuple(functional_set), tuple(terminal_set))

    def get_optimizer(
        self: SymbolicRegressionGP_DUAL,
    ) -> Union[
        DifferentialEvolution,
        GeneticAlgorithm,
        GeneticProgramming,
        jDE,
        SelfCGA,
        SelfCGP,
        SHADE,
        SHAGA,
    ]:
        return self._optimizer

    def _fit(
        self: SymbolicRegressionGP_DUAL,
        X: NDArray[np.float32],
        y: NDArray[Union[np.float32, np.int64]],
    ) -> SymbolicRegressionGP_DUAL:
        optimizer_args: dict[str, Any]
        uniset: UniversalSet

        if self._uniset is None:
            uniset = self._get_uniset(X)
        else:
            uniset = self._uniset

        if self._optimizer_args is not None:
            assert (
                "iters" not in self._optimizer_args.keys()
                and "pop_size" not in self._optimizer_args.keys()
                and "uniset" not in self._optimizer_args.keys()
            ), """Do not set the "iters", "pop_size", or "uniset" in the "optimizer_args". Instead,
              use the "SymbolicRegressionGP" arguments"""
            assert (
                "fitness_function" not in self._optimizer_args.keys()
            ), """Do not set the "fitness_function"
              to the "optimizer_args". It is defined automatically"""
            assert (
                "minimization" not in self._optimizer_args.keys()
            ), """Do not set the "minimization"
              to the "optimizer_args". It is defined automatically"""
            optimizer_args = self._optimizer_args.copy()

        else:
            optimizer_args = {}

        optimizer_args["fitness_function"] = fitness_function
        optimizer_args["fitness_function_args"] = {"y": y}
        optimizer_args["minimization"] = True
        optimizer_args["iters"] = self._iters
        optimizer_args["pop_size"] = self._pop_size
        optimizer_args["uniset"] = uniset
        optimizer_args["genotype_to_phenotype"] = genotype_to_phenotype

        self._optimizer = self._optimizer_class(**optimizer_args)
        self._optimizer.fit()

        return self

    def _predict(
        self: SymbolicRegressionGP_DUAL, X: NDArray[np.float32]
    ) -> NDArray[Union[np.float32, np.int64]]:
        n_dimension = X.shape[1]
        solution = self.get_optimizer().get_fittest()

        genotype_for_pred = []

        for tree_i in solution["phenotype"]:
            genotype_for_pred.append(
                tree_i.set_terminals(**{f"x{i}": X[:, i] for i in range(n_dimension)})
            )

        y_pred = mean_of_y(genotype_for_pred, X)

        return y_pred
