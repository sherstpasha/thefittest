from __future__ import annotations

from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

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
from ..tools.operators import Add
from ..tools.operators import Cos
from ..tools.operators import Inv
from ..tools.operators import Mul
from ..tools.operators import Neg
from ..tools.operators import Pow2
from ..tools.operators import Sin


def fitness_function(trees: NDArray, y: NDArray[np.float32]) -> NDArray[np.float32]:
    fitness = []
    for tree in trees:
        y_pred = tree() * np.ones(len(y))
        fitness.append(coefficient_determination(y, y_pred.astype(np.float32)))
    return np.array(fitness, dtype=np.float32)


def generator1() -> float:
    value = np.round(np.random.uniform(0, 10), 4)
    return value


def generator2() -> int:
    value = np.random.randint(0, 10)
    return value


class SymbolicRegressionGP(Model):
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

    def _get_uniset(self: SymbolicRegressionGP, X: NDArray[np.float32]) -> UniversalSet:
        uniset: UniversalSet
        terminal_set: Union[
            List[Union[TerminalNode, EphemeralNode]], Tuple[Union[TerminalNode, EphemeralNode]]
        ]
        n_dimension: int = X.shape[1]

        functional_set = (
            FunctionalNode(Add()),
            FunctionalNode(Mul()),
            FunctionalNode(Neg()),
            FunctionalNode(Inv()),
            FunctionalNode(Pow2()),
            FunctionalNode(Cos()),
            FunctionalNode(Sin()),
        )
        terminal_set = [TerminalNode(X[:, i], f"x{i}") for i in range(n_dimension)]
        terminal_set.extend([EphemeralNode(generator1), EphemeralNode(generator2)])
        uniset = UniversalSet(functional_set, tuple(terminal_set))
        return uniset

    def get_optimizer(
        self: SymbolicRegressionGP,
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
        self: SymbolicRegressionGP, X: NDArray[np.float32], y: NDArray[Union[np.float32, np.int64]]
    ) -> SymbolicRegressionGP:
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
        optimizer_args["iters"] = self._iters
        optimizer_args["pop_size"] = self._pop_size
        optimizer_args["uniset"] = uniset

        self._optimizer = self._optimizer_class(**optimizer_args)
        self._optimizer.fit()

        return self

    def _predict(
        self: SymbolicRegressionGP, X: NDArray[np.float32]
    ) -> NDArray[Union[np.float32, np.int64]]:
        n_dimension = X.shape[1]
        solution = self.get_optimizer().get_fittest()

        genotype_for_pred = solution["phenotype"].set_terminals(
            **{f"x{i}": X[:, i] for i in range(n_dimension)}
        )

        y_pred = genotype_for_pred() * np.ones(len(X))
        return y_pred
