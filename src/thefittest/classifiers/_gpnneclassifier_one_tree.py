from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
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
from ..base import Tree
from ..base import UniversalSet
from ..base._ea import MultiGenome
from ..base._net import HiddenBlock
from ..base._net import NetEnsemble, Net
from ..base._tree import DualNode
from ..base._tree import EnsembleUniversalSet
from ..classifiers import GeneticProgrammingNeuralNetClassifier
from ..classifiers._gpnnclassifier import genotype_to_phenotype_tree
from ..classifiers._gpnnclassifier import train_net
from ..optimizers import DifferentialEvolution
from ..optimizers import GeneticAlgorithm, GeneticProgramming
from ..optimizers import SHADE
from ..optimizers import SHAGA
from ..optimizers import SelfCGA
from ..optimizers import jDE
from ..tools.metrics import categorical_crossentropy
from ..tools.operators import Add
from ..tools.operators import More

from ..tools import donothing
from ..tools.random import half_and_half
from ..classifiers._mlpeaclassifier import fitness_function as evaluate_nets
from ..tools.random import train_test_split_stratified
from ..base._net import ACTIV_NAME_INV
import torch


weights_type_optimizer_alias = Union[
    Type[DifferentialEvolution],
    Type[jDE],
    Type[SHADE],
    Type[GeneticAlgorithm],
    Type[SelfCGA],
    Type[SHAGA],
]
weights_optimizer_alias = Union[DifferentialEvolution, jDE, SHADE, GeneticAlgorithm, SelfCGA, SHAGA]


def _defitne_net(
    n_inputs: int,
    n_outputs: int,
    hidden_layers,
    activation,
    offset,
    output_activation,
) -> Net:
    start = 0
    end = n_inputs
    inputs_id = set(range(start, end))

    net = Net(inputs=inputs_id)

    for n_layer in hidden_layers:
        start = end
        end = end + n_layer
        inputs_id = {(n_inputs - 1)}
        hidden_id = set(range(start, end))
        activs = dict(zip(hidden_id, [ACTIV_NAME_INV[activation]] * len(hidden_id)))

        if offset:
            layer_net = Net(inputs=inputs_id) > Net(hidden_layers=[hidden_id], activs=activs)
        else:
            layer_net = Net(hidden_layers=[hidden_id], activs=activs)

        net = net > layer_net

    start = end
    end = end + n_outputs
    inputs_id = {(n_inputs - 1)}
    output_id = set(range(start, end))
    activs = dict(zip(output_id, [ACTIV_NAME_INV[output_activation]] * len(output_id)))

    if offset:
        layer_net = Net(inputs=inputs_id) > Net(outputs=output_id, activs=activs)
    else:
        layer_net = Net(outputs=output_id, activs=activs)

    net = net > layer_net
    net._offset = offset
    return net


def fitness_function(
    population: NDArray,
    X: NDArray[np.float32],
    targets: NDArray[np.float32],
    net_size_penalty: float,
) -> NDArray[np.float32]:
    fitness = np.empty(shape=len(population), dtype=np.float32)
    for i, ensemble in enumerate(population):
        output2d = ensemble.meta_output(X)
        fitness[i] = categorical_crossentropy(targets.cpu().numpy(), output2d.cpu().numpy())

    return fitness


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


def genotype_to_phenotype_ensemble(
    tree,
    n_variables: int,
    n_outputs: int,
    output_activation: str,
    offset: bool,
) -> NetEnsemble:
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

    nets = [
        genotype_to_phenotype_tree(
            tree=tree_i,
            n_variables=n_variables,
            n_outputs=n_outputs,
            output_activation=output_activation,
            offset=offset,
        )
        for tree_i in trees
    ]

    ens = NetEnsemble(nets=np.array(nets, dtype=object))
    ens._meta_tree = None
    ens._trees = trees
    return ens


def genotype_to_phenotype(
    population_g: NDArray,
    X_train_ens: NDArray[np.float32],
    proba_train_ens: NDArray[np.float32],
    X_train_meta: NDArray[np.float32],
    proba_train_meta: NDArray[np.float32],
    weights_optimizer_args: Dict,
    weights_optimizer_class: weights_type_optimizer_alias,
    n_outputs: int,
    output_activation: str,
    offset: bool,
    evaluate_nets: Callable,
) -> NDArray:
    n_variables: int = X_train_ens.shape[1]

    population_ph: NDArray = np.empty(shape=len(population_g), dtype=object)

    population_ph = np.array(
        [
            train_ensemble(
                ensemble=genotype_to_phenotype_ensemble(
                    tree=individ_g,
                    n_variables=n_variables,
                    n_outputs=n_outputs,
                    output_activation=output_activation,
                    offset=offset,
                ),
                X_train_ens=X_train_ens,
                proba_train_ens=proba_train_ens,
                X_train_meta=X_train_meta,
                proba_train_meta=proba_train_meta,
                weights_optimizer_args=weights_optimizer_args,
                weights_optimizer_class=weights_optimizer_class,
                fitness_function=evaluate_nets,
                output_activation=output_activation,
                offset=offset,
            )
            for individ_g in population_g
        ],
        dtype=object,
    )

    return population_ph


def train_ensemble(
    ensemble: NetEnsemble,
    X_train_ens: NDArray[np.float32],
    proba_train_ens: NDArray[np.float32],
    X_train_meta: NDArray[np.float32],
    proba_train_meta: NDArray[np.float32],
    weights_optimizer_args: Dict,
    weights_optimizer_class: weights_type_optimizer_alias,
    fitness_function: Callable,
    output_activation: str,
    offset: bool,
) -> NetEnsemble:

    subsamples = create_bootstrap_subsamples(
        X_train_ens, proba_train_ens, n_subsamples=len(ensemble._nets), seed=123
    )

    nets = []
    for subsample, net in zip(subsamples, ensemble._nets):
        net = train_net(
            net=net,
            X_train=subsample[0],
            proba_train=subsample[1],
            weights_optimizer_args=weights_optimizer_args,
            weights_optimizer_class=weights_optimizer_class,
            fitness_function=fitness_function,
        )
        nets.append(net)

    ensemble._nets = nets
    X_meta = ensemble._get_meta_inputs(X_train_meta, offset=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_meta = torch.tensor(X_meta, device=device, dtype=torch.float32)

    y_meta = np.argmax(proba_train_meta.cpu().numpy(), axis=1)

    tree = ensemble._meta_tree

    if tree is not None:
        meta_net = genotype_to_phenotype_tree(
            tree=tree,
            n_variables=X_meta.shape[1],
            n_outputs=len(set(y_meta)),
            output_activation=output_activation,
            offset=offset,
        )

        meta_net._offset = True

        ensemble._meta_algorithm = train_net(
            net=meta_net,
            X_train=X_meta,
            proba_train=proba_train_meta,
            weights_optimizer_args=weights_optimizer_args,
            weights_optimizer_class=weights_optimizer_class,
            fitness_function=fitness_function,
        )

        return ensemble.copy()
    else:

        meta_net = _defitne_net(
            n_inputs=X_meta.shape[1],
            n_outputs=len(set(y_meta)),
            hidden_layers=(0,),
            activation="relu",
            offset=True,
            output_activation="softmax",
        )

        ensemble._meta_algorithm = train_net(
            net=meta_net,
            X_train=X_meta,
            proba_train=proba_train_meta,
            weights_optimizer_args=weights_optimizer_args,
            weights_optimizer_class=weights_optimizer_class,
            fitness_function=fitness_function,
        )

        return ensemble.copy()


def create_bootstrap_subsamples(X, y, n_subsamples, subsample_size=None, seed=42):
    """
    Создает n_subsamples бутстрап-подвыборок из данных X и y.

    Аргументы:
      X (torch.Tensor): Признаковая матрица.
      y (torch.Tensor): Целевые значения.
      n_subsamples (int): Число создаваемых подвыборок.
      subsample_size (int, optional): Размер каждой подвыборки.
                                      Если None, то берется полный размер X.
      seed (int): Зерно для воспроизводимости разбиения.

    Возвращает:
      List[Tuple[torch.Tensor, torch.Tensor]]: Список кортежей (X_sub, y_sub)
    """
    n_samples = X.size(0)
    if subsample_size is None:
        subsample_size = n_samples

    subsamples = []

    # Основной генератор случайных чисел для воспроизводимости
    base_generator = torch.Generator()
    base_generator.manual_seed(seed)

    # Получаем фиксированные стартовые состояния для каждой подвыборки,
    # чтобы разбиение было одинаковым при одинаковом количестве подвыборок
    seeds = [base_generator.seed() for _ in range(n_subsamples)]

    for s in seeds:
        gen = torch.Generator()
        gen.manual_seed(s)
        indices = torch.randint(0, n_samples, (subsample_size,), generator=gen)
        subsamples.append((X[indices], y[indices]))

    return subsamples


class GeneticProgrammingNeuralNetStackingClassifier(GeneticProgrammingNeuralNetClassifier):
    def __init__(
        self,
        iters: int,
        pop_size: int,
        input_block_size: int = 1,
        max_hidden_block_size: int = 9,
        offset: bool = True,
        output_activation: str = "softmax",
        test_sample_ratio: float = 0.5,
        optimizer=GeneticProgramming,
        optimizer_args: Optional[dict[str, Any]] = None,
        weights_optimizer: weights_type_optimizer_alias = SHADE,
        weights_optimizer_args: Optional[dict[str, Any]] = None,
    ):
        GeneticProgrammingNeuralNetClassifier.__init__(
            self,
            iters=iters,
            pop_size=pop_size,
            input_block_size=input_block_size,
            max_hidden_block_size=max_hidden_block_size,
            offset=offset,
            output_activation=output_activation,
            test_sample_ratio=test_sample_ratio,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            weights_optimizer=weights_optimizer,
            weights_optimizer_args=weights_optimizer_args,
        )
        self._cache: List[NetEnsemble] = []

    def _get_uniset_1(
        self: GeneticProgrammingNeuralNetStackingClassifier, X: NDArray[np.float32]
    ) -> EnsembleUniversalSet:
        uniset: EnsembleUniversalSet
        if self._offset:
            n_dimension = X.shape[1] - 1
        else:
            n_dimension = X.shape[1]

        cut_id: NDArray[np.int64] = np.arange(
            self._input_block_size, n_dimension, self._input_block_size, dtype=np.int64
        )
        variables_pool: List = np.split(np.arange(n_dimension), cut_id)

        def random_hidden_block() -> HiddenBlock:
            return HiddenBlock(self._max_hidden_block_size)

        functional_set = [
            FunctionalNode(Add()),
            FunctionalNode(More()),
            FunctionalNode(DualNode(EphemeralNode(random_hidden_block), FunctionalNode(Add()))),
            FunctionalNode(DualNode(EphemeralNode(random_hidden_block), FunctionalNode(More()))),
        ]

        terminal_set: List[Union[TerminalNode, EphemeralNode]] = [
            EphemeralNode(random_hidden_block),
        ]
        for i, variables in enumerate(variables_pool):
            terminal = TerminalNode(set(variables), "in{}".format(i))
            terminal_set.append(terminal)
            functional_set.append(FunctionalNode(DualNode(terminal, FunctionalNode(Add()))))
            functional_set.append(FunctionalNode(DualNode(terminal, FunctionalNode(More()))))

        if self._offset:
            terminal_set.append(
                TerminalNode(value={(n_dimension)}, name="in{}".format(len(variables_pool)))
            )
        terminal_set.append(EphemeralNode(random_hidden_block))

        uniset = EnsembleUniversalSet(tuple(functional_set), tuple(terminal_set))
        return uniset

    def _define_optimizer_ensembles(
        self: GeneticProgrammingNeuralNetStackingClassifier,
        uniset: UniversalSet,
        n_outputs: int,
        X_train_ens: NDArray[np.float32],
        proba_train_ens: NDArray[np.float32],
        X_train_meta: NDArray[np.float32],
        proba_train_meta: NDArray[np.float32],
        X_test: NDArray[np.float32],
        target_test: NDArray[np.float32],
        fitness_function: Callable,
        evaluate_nets: Callable,
    ):
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
            "X_train_ens": X_train_ens,
            "proba_train_ens": proba_train_ens,
            "X_train_meta": X_train_meta,
            "proba_train_meta": proba_train_meta,
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

    def _fit(
        self: GeneticProgrammingNeuralNetStackingClassifier,
        X: NDArray[np.float32],
        y: NDArray[Union[np.float32, np.int64]],
    ) -> GeneticProgrammingNeuralNetStackingClassifier:
        if self._offset:
            X = np.hstack([X.copy(), np.ones((X.shape[0], 1))])

        n_outputs: int = len(set(y))
        eye: NDArray[np.float32] = np.eye(n_outputs, dtype=np.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_train, X_test, y_train, y_test = train_test_split_stratified(
            X, y.astype(np.int64), self._test_sample_ratio
        )

        X_train_ens, X_train_meta, y_train_ens, y_train_meta = train_test_split_stratified(
            X_train, y_train, 0.5
        )

        proba_test: NDArray[np.float32] = eye[y_test]
        proba_train_ens: NDArray[np.float32] = eye[y_train_ens]
        proba_train_meta: NDArray[np.float32] = eye[y_train_meta]

        X_train_ens = torch.tensor(X_train_ens, device=device, dtype=torch.float32)
        X_train_meta = torch.tensor(X_train_meta, device=device, dtype=torch.float32)
        X_test = torch.tensor(X_test, device=device, dtype=torch.float32)
        proba_test = torch.tensor(proba_test, dtype=torch.float32, device=device)
        proba_train_ens = torch.tensor(proba_train_ens, dtype=torch.float32, device=device)
        proba_train_meta = torch.tensor(proba_train_meta, dtype=torch.float32, device=device)

        self._optimizer = self._define_optimizer_ensembles(
            uniset=self._get_uniset_1(X),
            n_outputs=n_outputs,
            X_train_ens=X_train_ens,
            proba_train_ens=proba_train_ens,
            X_train_meta=X_train_meta,
            proba_train_meta=proba_train_meta,
            X_test=X_test,
            target_test=proba_test,
            fitness_function=fitness_function,
            evaluate_nets=evaluate_nets,
        )

        self._optimizer.fit()

        return self

    def _predict(self, X: NDArray[np.float32]) -> NDArray[Union[np.float32, np.int64]]:
        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = torch.tensor(X, device=device, dtype=torch.float32)
        fittest = self._optimizer.get_fittest()
        output = fittest["phenotype"].meta_output(X)

        return np.argmax(output.cpu().numpy(), axis=1)
