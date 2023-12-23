from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Callable
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
from ..classifiers._mlpeaclassifier import fitness_function as evaluate_nets
from ..classifiers._mlpeaclassifier import weights_type_optimizer_alias
from ..optimizers import DifferentialEvolution
from ..optimizers import GeneticAlgorithm
from ..optimizers import GeneticProgramming
from ..optimizers import SHADE
from ..optimizers import SHAGA
from ..optimizers import SelfCGA
from ..optimizers import SelfCGP
from ..optimizers import jDE
from ..utils.metrics import categorical_crossentropy3d
from ..utils.operators import Add
from ..utils.operators import More
from ..utils.random import float_population
from ..utils.random import train_test_split_stratified
from ..utils.transformations import GrayCode


def fitness_function(
    population: NDArray,
    X: NDArray[np.float64],
    targets: NDArray[np.float64],
    net_size_penalty: float,
) -> NDArray[np.float64]:
    output3d = np.array([net.forward(X)[0] for net in population], dtype=np.float64)
    lens = np.array(list(map(len, population)))
    fitness = categorical_crossentropy3d(targets, output3d) + net_size_penalty * lens
    return fitness


def genotype_to_phenotype_tree(
    tree: Tree, n_variables: int, n_outputs: int, output_activation: str, offset: bool
) -> Net:
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
    activs = dict(zip(output_id, [ACTIV_NAME_INV[output_activation]] * len(output_id)))
    to_return = pack[0] > Net(outputs=output_id, activs=activs)
    to_return = to_return._fix(set(range(n_variables)))
    to_return._offset = offset

    return to_return


def genotype_to_phenotype(
    population_g: NDArray,
    n_outputs: int,
    X_train: NDArray[np.float64],
    proba_train: NDArray[np.float64],
    weights_optimizer_args: Dict,
    weights_optimizer_class: weights_type_optimizer_alias,
    output_activation: str,
    offset: bool,
    evaluate_nets: Callable,
) -> NDArray:
    n_variables: int = X_train.shape[1]

    population_ph = np.array(
        [
            train_net(
                genotype_to_phenotype_tree(
                    individ_g, n_variables, n_outputs, output_activation, offset
                ),
                X_train=X_train,
                proba_train=proba_train,
                weights_optimizer_args=weights_optimizer_args,
                weights_optimizer_class=weights_optimizer_class,
                fitness_function=evaluate_nets,
            )
            for individ_g in population_g
        ],
        dtype=object,
    )

    return population_ph


def train_net(
    net: Net,
    X_train: NDArray[np.float64],
    proba_train: NDArray[np.float64],
    weights_optimizer_args: Dict,
    weights_optimizer_class: weights_type_optimizer_alias,
    fitness_function: Callable,
) -> Net:
    if weights_optimizer_args is not None:
        for arg in (
            "fitness_function",
            "left",
            "right",
            "str_len",
            "genotype_to_phenotype",
            "minimization",
        ):
            assert (
                arg not in weights_optimizer_args.keys()
            ), f"""Do not set the "{arg}"
              to the "weights_optimizer_args". It is defined automatically"""
        weights_optimizer_args = weights_optimizer_args.copy()

    else:
        weights_optimizer_args = {"iters": 100, "pop_size": 100}

    left: NDArray[np.float64] = np.full(shape=len(net._weights), fill_value=-10, dtype=np.float64)
    right: NDArray[np.float64] = np.full(shape=len(net._weights), fill_value=10, dtype=np.float64)
    initial_population: Union[NDArray[np.float64], NDArray[np.byte]] = float_population(
        weights_optimizer_args["pop_size"], left, right
    )
    initial_population[0] = net._weights.copy()
    weights_optimizer_args["fitness_function"] = fitness_function
    weights_optimizer_args["fitness_function_args"] = {
        "net": net,
        "X": X_train,
        "targets": proba_train,
    }

    if weights_optimizer_class in (SHADE, DifferentialEvolution, jDE):
        weights_optimizer_args["left"] = left
        weights_optimizer_args["right"] = right
    else:
        parts: NDArray[np.int64] = np.full(shape=len(net._weights), fill_value=16, dtype=np.int64)
        genotype_to_phenotype = GrayCode(fit_by="parts").fit(left, right, parts)
        weights_optimizer_args["str_len"] = np.sum(parts)
        weights_optimizer_args["genotype_to_phenotype"] = genotype_to_phenotype.transform

    weights_optimizer_args["minimization"] = True
    optimizer = weights_optimizer_class(**weights_optimizer_args)
    optimizer.fit()

    phenotype = optimizer.get_fittest()["phenotype"]
    net._weights = phenotype

    return net.copy()


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
        net_size_penalty: float = 0.0,
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

        self._optimizer: Union[SelfCGP, GeneticProgramming]
        self._net_size_penalty: float = net_size_penalty

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

    def _define_optimizer(
        self: GeneticProgrammingNeuralNetClassifier,
        uniset: UniversalSet,
        n_outputs: int,
        X_train: NDArray[np.float64],
        target_train: NDArray[np.float64],
        X_test: NDArray[np.float64],
        target_test: NDArray[np.float64],
        fitness_function: Callable,
        evaluate_nets: Callable,
    ) -> Union[SelfCGP, GeneticProgramming]:
        optimizer_args: dict[str, Any]

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

        optimizer_args["fitness_function"] = fitness_function
        optimizer_args["fitness_function_args"] = {
            "X": X_test,
            "targets": target_test,
            "net_size_penalty": self._net_size_penalty,
        }

        optimizer_args["genotype_to_phenotype"] = genotype_to_phenotype
        optimizer_args["genotype_to_phenotype_args"] = {
            "n_outputs": n_outputs,
            "X_train": X_train,
            "proba_train": target_train,
            "weights_optimizer_args": self._weights_optimizer_args,
            "weights_optimizer_class": self._weights_optimizer_class,
            "output_activation": self._output_activation,
            "offset": self._offset,
            "evaluate_nets": evaluate_nets,
        }

        optimizer_args["iters"] = self._iters
        optimizer_args["pop_size"] = self._pop_size
        optimizer_args["uniset"] = uniset
        optimizer_args["minimization"] = True

        return self._optimizer_class(**optimizer_args)

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
        if self._offset:
            X = np.hstack([X.copy(), np.ones((X.shape[0], 1))])

        n_outputs: int = len(set(y))
        eye: NDArray[np.float64] = np.eye(n_outputs, dtype=np.float64)

        X_train, X_test, y_train, y_test = train_test_split_stratified(
            X, y.astype(np.int64), self._test_sample_ratio
        )

        proba_test: NDArray[np.float64] = eye[y_test]
        proba_train: NDArray[np.float64] = eye[y_train]

        uniset: UniversalSet = self._get_uniset(X)

        self._optimizer = self._define_optimizer(
            uniset=uniset,
            n_outputs=n_outputs,
            X_train=X_train,
            target_train=proba_train,
            X_test=X_test,
            target_test=proba_test,
            fitness_function=fitness_function,
            evaluate_nets=evaluate_nets,
        )

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
