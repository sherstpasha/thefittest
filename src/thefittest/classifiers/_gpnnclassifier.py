from __future__ import annotations

from typing import Any
from typing import List
from typing import Optional
from typing import Type
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
from ..optimizers import DifferentialEvolution
from ..optimizers import GeneticAlgorithm
from ..optimizers import GeneticProgramming
from ..optimizers import SHADE
from ..optimizers import SHAGA
from ..optimizers import SelfCGA
from ..optimizers import SelfCGP
from ..optimizers import jDE
from ..tools.metrics import categorical_crossentropy3d
from ..tools.operators import Add
from ..tools.operators import More
from ..tools.random import float_population
from ..tools.random import train_test_split_stratified
from ..tools.transformations import GrayCode


weights_type_optimizer_alias = Union[
    Type[DifferentialEvolution],
    Type[jDE],
    Type[SHADE],
    Type[GeneticAlgorithm],
    Type[SelfCGA],
    Type[SHAGA],
]
weights_optimizer_alias = Union[DifferentialEvolution, jDE, SHADE, GeneticAlgorithm, SelfCGA, SHAGA]


class GeneticProgrammingNeuralNetClassifier(Model):
    """(Lipinsky L., Semenkin E., Bulletin of the Siberian State Aerospace University., 3(10),
    22-26 (2006). In Russian);"""

    def __init__(
        self,
        iters: int,
        pop_size: int,
        input_block_size: int = 1,
        max_hidden_block_size: int = 9,
        offset: bool = True,
        output_activation: str = "softmax",
        test_sample_ratio: float = 0.5,
        optimizer: Union[Type[SelfCGP], Type[GeneticProgramming]] = SelfCGP,
        optimizer_args: Optional[dict[str, Any]] = None,
        weights_optimizer: weights_type_optimizer_alias = SHADE,
        weights_optimizer_args: Optional[dict[str, Any]] = None,
        cache: bool = True,
    ):
        Model.__init__(self)
        self._iters: int = iters
        self._pop_size: int = pop_size
        self._input_block_size: int = input_block_size
        self._max_hidden_block_size: int = max_hidden_block_size
        self._offset: bool = offset
        self._output_activation: str = output_activation
        self._test_sample_ratio: float = test_sample_ratio
        self._optimizer_class: Union[Type[SelfCGP], Type[GeneticProgramming]] = optimizer
        self._optimizer_args: Optional[dict[str, Any]] = optimizer_args
        self._weights_optimizer_class: weights_type_optimizer_alias = weights_optimizer
        self._weights_optimizer_args: Optional[dict[str, Any]] = weights_optimizer_args
        self._cache_condition: bool = cache

        self._optimizer: Union[SelfCGP, GeneticProgramming]
        self._cache: List[Net] = []

    def _get_uniset(
        self: GeneticProgrammingNeuralNetClassifier, X: NDArray[np.float64]
    ) -> UniversalSet:
        uniset: UniversalSet
        if self._offset:
            n_dimension = X.shape[1] - 1
        else:
            n_dimension = X.shape[1]

        cut_id: NDArray[np.int64] = np.arange(
            self._input_block_size, n_dimension, self._input_block_size, dtype=np.int64
        )
        variables_pool: List = np.split(np.arange(n_dimension), cut_id)

        functional_set = (FunctionalNode(Add()), FunctionalNode(More()))

        def random_hidden_block() -> HiddenBlock:
            return HiddenBlock(self._max_hidden_block_size)

        terminal_set: List[Union[TerminalNode, EphemeralNode]] = [
            TerminalNode(set(variables), "in{}".format(i))
            for i, variables in enumerate(variables_pool)
        ]
        if self._offset:
            terminal_set.append(
                TerminalNode(value={(n_dimension)}, name="in{}".format(len(variables_pool)))
            )
        terminal_set.append(EphemeralNode(random_hidden_block))

        uniset = UniversalSet(functional_set, tuple(terminal_set))
        return uniset

    def _fitness_function(
        self, population: NDArray, X: NDArray[np.float64], targets: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        output3d = np.array([net.forward(X)[0] for net in population], dtype=np.float64)
        fitness = categorical_crossentropy3d(targets, output3d)
        return fitness

    def _genotype_to_phenotype_tree(self, n_variables: int, n_outputs: int, tree: Tree) -> Net:
        pack: Any = []

        n = n_variables
        for node in reversed(tree._nodes):
            args = []
            for _ in range(node._n_args):
                args.append(pack.pop())
            if isinstance(node, FunctionalNode):
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

    def _evaluate_nets(
        self,
        weights: NDArray[np.float64],
        net: Net,
        X: NDArray[np.float64],
        targets: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        output3d = net.forward(X, weights)
        error = categorical_crossentropy3d(targets, output3d)
        return error

    def _train_net(
        self,
        net: Net,
        X_train: NDArray[np.float64],
        proba_train: NDArray[np.float64],
    ) -> Net:
        if self._weights_optimizer_args is not None:
            for arg in (
                "fitness_function",
                "left",
                "right",
                "str_len",
                "genotype_to_phenotype",
                "minimization",
            ):
                assert (
                    arg not in self._weights_optimizer_args.keys()
                ), f"""Do not set the "{arg}"
              to the "weights_optimizer_args". It is defined automatically"""
            weights_optimizer_args = self._weights_optimizer_args.copy()

        else:
            weights_optimizer_args = {"iters": 100, "pop_size": 100}

        left: NDArray[np.float64] = np.full(
            shape=len(net._weights), fill_value=-10, dtype=np.float64
        )
        right: NDArray[np.float64] = np.full(
            shape=len(net._weights), fill_value=10, dtype=np.float64
        )
        initial_population: Union[NDArray[np.float64], NDArray[np.byte]] = float_population(
            weights_optimizer_args["pop_size"], left, right
        )
        initial_population[0] = net._weights.copy()
        weights_optimizer_args["fitness_function"] = lambda population: self._evaluate_nets(
            population, net, X_train, proba_train
        )
        if self._weights_optimizer_class in (SHADE, DifferentialEvolution, jDE):
            weights_optimizer_args["left"] = left
            weights_optimizer_args["right"] = right
        else:
            parts: NDArray[np.int64] = np.full(
                shape=len(net._weights), fill_value=16, dtype=np.int64
            )
            genotype_to_phenotype = GrayCode(fit_by="parts").fit(left, right, parts)
            weights_optimizer_args["str_len"] = np.sum(parts)
            weights_optimizer_args["genotype_to_phenotype"] = genotype_to_phenotype.transform

        weights_optimizer_args["minimization"] = True
        optimizer = self._weights_optimizer_class(**weights_optimizer_args)
        optimizer.fit()

        phenotype = optimizer.get_fittest()["phenotype"]
        net._weights = phenotype
        return net

    def _genotype_to_phenotype(
        self: GeneticProgrammingNeuralNetClassifier,
        X_train: NDArray[np.float64],
        proba_train: NDArray[np.float64],
        population_g: NDArray,
        n_outputs: int,
    ) -> NDArray:
        n_variables: int = X_train.shape[1]

        population_ph = np.empty(shape=len(population_g), dtype=object)

        for i, individ_g in enumerate(population_g):
            net = self._genotype_to_phenotype_tree(n_variables, n_outputs, individ_g)
            trained = False
            if self._cache_condition:
                for net_i in self._cache:
                    if net_i == net:
                        population_ph[i] = net_i.copy()
                        trained = True
                        break
            if not trained:
                population_ph[i] = self._train_net(net, X_train, proba_train)
                trained = True
                self._cache.append(population_ph[i].copy())

        return population_ph

    def get_optimizer(
        self: GeneticProgrammingNeuralNetClassifier,
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

    def get_net(self: GeneticProgrammingNeuralNetClassifier) -> Net:
        optimizer = self.get_optimizer()
        net = optimizer.get_fittest()["phenotype"]
        return net

    def _fit(
        self: GeneticProgrammingNeuralNetClassifier,
        X: NDArray[np.float64],
        y: NDArray[Union[np.float64, np.int64]],
    ) -> GeneticProgrammingNeuralNetClassifier:
        optimizer_args: dict[str, Any]

        indexes = np.arange(len(X))
        np.random.shuffle(indexes)

        X = X.copy()
        X = X[indexes]

        y = y.copy()
        y = y[indexes]

        if self._offset:
            X = np.hstack([X.copy(), np.ones((X.shape[0], 1))])

        n_outputs: int = len(set(y))
        eye: NDArray[np.float64] = np.eye(n_outputs, dtype=np.float64)

        X_train, X_test, y_train, y_test = train_test_split_stratified(
            X, y.astype(np.int64), self._test_sample_ratio
        )

        proba_test: NDArray[np.float64] = eye[y_test]
        proba_train: NDArray[np.float64] = eye[y_train]

        if self._optimizer_args is not None:
            assert (
                "iters" not in self._optimizer_args.keys()
                and "pop_size" not in self._optimizer_args.keys()
            ), """Do not set the "iters" or "pop_size" in the "optimizer_args". Instead,
              use the "SymbolicRegressionGP" arguments"""
            for arg in (
                "fitness_function",
                "uniset",
                "minimization",
            ):
                assert (
                    arg not in self._optimizer_args.keys()
                ), f"""Do not set the "{arg}"
                to the "optimizer_args". It is defined automatically"""
            optimizer_args = self._optimizer_args.copy()

        else:
            optimizer_args = {}

        uniset: UniversalSet = self._get_uniset(X)

        optimizer_args["fitness_function"] = lambda population: self._fitness_function(
            population, X_test, proba_test
        )
        optimizer_args["genotype_to_phenotype"] = lambda trees: self._genotype_to_phenotype(
            X_train, proba_train, trees, n_outputs
        )
        optimizer_args["iters"] = self._iters
        optimizer_args["pop_size"] = self._pop_size
        optimizer_args["uniset"] = uniset
        optimizer_args["minimization"] = False

        self._optimizer = self._optimizer_class(**optimizer_args)
        self._optimizer.fit()

        return self

    def _prepare_output(
        self: GeneticProgrammingNeuralNetClassifier, output: NDArray[np.float64]
    ) -> Union[NDArray[np.float64], NDArray[np.int64]]:
        return np.argmax(output, axis=1)

    def _predict(self, X: NDArray[np.float64]) -> NDArray[Union[np.float64, np.int64]]:
        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        fittest = self._optimizer.get_fittest()

        output = fittest["phenotype"].forward(X)[0]
        return self._prepare_output(output)
