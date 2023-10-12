from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..base import EphemeralNode
from ..base._tree import EphemeralConstantNode
from ..base._tree import DualNode
from ..base import FunctionalNode
from ..base import TerminalNode
from ..base import Tree
from ..base._tree import EnsembleUniversalSet
from ..base._model import Model
from ..base._net import ACTIV_NAME_INV
from ..base._net import HiddenBlock
from ..base._net import Net
from ..base._net import NetEnsemble
from ..optimizers import DifferentialEvolution
from ..optimizers import GeneticAlgorithm
from ..optimizers import GeneticProgramming
from ..optimizers import SHADE
from ..optimizers import SHAGA
from ..optimizers import SelfCGA
from ..optimizers import SelfCGP
from ..optimizers import jDE
from ..tools.metrics import categorical_crossentropy
from ..tools.metrics import categorical_crossentropy3d
from ..tools.metrics import f1_score
from ..tools.metrics import f1_score2d
from ..tools.operators import Add
from ..tools.operators import More
from ..tools.random import float_population
from ..tools.random import train_test_split_stratified
from ..tools.transformations import GrayCode
from ..base._tree import Operator
from ..classifiers import GeneticProgrammingNeuralNetClassifier


weights_type_optimizer_alias = Union[
    Type[DifferentialEvolution],
    Type[jDE],
    Type[SHADE],
    Type[GeneticAlgorithm],
    Type[SelfCGA],
    Type[SHAGA],
]
weights_optimizer_alias = Union[DifferentialEvolution, jDE, SHADE, GeneticAlgorithm, SelfCGA, SHAGA]


class GeneticProgrammingNeuralNetEnsemblesClassifier(GeneticProgrammingNeuralNetClassifier):
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
            cache=cache,
        )
        self._cache: List[NetEnsemble] = []

    def _get_uniset(
        self: GeneticProgrammingNeuralNetEnsemblesClassifier, X: NDArray[np.float64]
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

    def _split_tree(
        self: GeneticProgrammingNeuralNetEnsemblesClassifier, tree: Tree
    ) -> Tuple[Tree, Tree]:
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

    def _genotype_to_phenotype_ensemble(
        self, n_variables: int, n_outputs: int, tree: Tree
    ) -> NetEnsemble:
        trees = []
        remain_tree, new_tree = self._split_tree(tree)
        if len(new_tree) > 0:
            trees.append(new_tree)
        while True:
            remain_tree, new_tree = self._split_tree(remain_tree)
            if len(new_tree) > 0:
                trees.append(new_tree)
            else:
                break

        if len(remain_tree) > 0:
            trees.append(remain_tree)

        nets = [
            self._genotype_to_phenotype_tree(n_variables, n_outputs, tree_i) for tree_i in trees
        ]

        return NetEnsemble(nets=np.array(nets, dtype=object))

    def _genotype_to_phenotype(
        self: GeneticProgrammingNeuralNetEnsemblesClassifier,
        X_train: NDArray[np.float64],
        proba_train: NDArray[np.float64],
        population_g: NDArray,
        n_outputs: int,
    ) -> NDArray:
        n_variables: int = X_train.shape[1]

        population_ph: NDArray = np.empty(shape=len(population_g), dtype=object)

        population_ph = np.array(
            [
                self._train_ensemble(
                    self._genotype_to_phenotype_ensemble(n_variables, n_outputs, individ_g),
                    X_train,
                    proba_train,
                )
                for individ_g in population_g
            ],
            dtype=object,
        )

        return population_ph

    def _fitness_function(
        self, population: NDArray, X: NDArray[np.float64], targets: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        fitness = np.empty(shape=self._pop_size, dtype=np.float64)
        for i, ensemble in enumerate(population):
            # output2d = ensemble.average_output(X)[0]
            output3d = ensemble.voting_output_classifier(X)[0]
            # fitness[i] = categorical_crossentropy(targets, output2d)
            y_true = np.argmax(targets, axis=1)
            fitness[i] = f1_score(y_true, output3d)
        return fitness

    def _train_ensemble(
        self,
        ensemble: NetEnsemble,
        X_train: NDArray[np.float64],
        proba_train: NDArray[np.float64],
    ) -> NetEnsemble:
        if self._cache_condition:
            for ensemble_i in self._cache:
                if ensemble_i == ensemble:
                    return ensemble_i.copy()

        n_nets = len(ensemble)
        print(n_nets)

        step = int(np.ceil(len(X_train) / n_nets))
        split_points = np.arange(step, len(X_train), step)

        # X_train_groups = np.split(X_train, split_points)
        # proba_train_groups = np.split(proba_train, split_points)
        # # for i, net_i in enumerate(ensemble._nets):
        # #     ensemble._nets[i] = self._train_net(net_i, X_train, proba_train)

        return self._train_all_ensemble(ensemble, X_train, proba_train)

    def _evaluate_ensemble(
        self,
        all_weights: NDArray[np.float64],
        cut_points: NDArray[np.int64],
        ensemble: NetEnsemble,
        X: NDArray[np.float64],
        targets: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        weights_list = np.split(all_weights, cut_points, axis=-1)
        # output3d = ensemble.average_output(X, weights_list)
        output3d = ensemble.voting_output_classifier(X, weights_list)
        y_true = np.argmax(targets, axis=-1)
        error = f1_score2d(y_true, output3d)
        return error

    def _train_all_ensemble(
        self,
        ensemble: NetEnsemble,
        X_train: NDArray[np.float64],
        proba_train: NDArray[np.float64],
    ) -> NetEnsemble:
        if self._cache_condition:
            for ensemble_i in self._cache:
                if ensemble_i == ensemble:
                    return ensemble_i.copy()

        net_lens = [len(net_i) for net_i in ensemble._nets]
        cut_points = np.add.accumulate(net_lens, dtype=np.int64)[:-1]

        n_variables = sum(net_lens)
        initial_weights = np.concatenate([net_i._weights for net_i in ensemble._nets])

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

        weights_optimizer_args["iters"] = weights_optimizer_args["iters"] * len(net_lens)
        left: NDArray[np.float64] = np.full(shape=n_variables, fill_value=-10, dtype=np.float64)
        right: NDArray[np.float64] = np.full(shape=n_variables, fill_value=10, dtype=np.float64)
        initial_population: Union[NDArray[np.float64], NDArray[np.byte]] = float_population(
            weights_optimizer_args["pop_size"], left, right
        )
        initial_population[0] = initial_weights.copy()
        weights_optimizer_args["fitness_function"] = lambda population: self._evaluate_ensemble(
            all_weights=population,
            ensemble=ensemble,
            cut_points=cut_points,
            X=X_train,
            targets=proba_train,
        )
        if self._weights_optimizer_class in (SHADE, DifferentialEvolution, jDE):
            weights_optimizer_args["left"] = left
            weights_optimizer_args["right"] = right
        else:
            parts: NDArray[np.int64] = np.full(shape=n_variables, fill_value=16, dtype=np.int64)
            genotype_to_phenotype = GrayCode(fit_by="parts").fit(left, right, parts)
            weights_optimizer_args["str_len"] = np.sum(parts)
            weights_optimizer_args["genotype_to_phenotype"] = genotype_to_phenotype.transform

        weights_optimizer_args["minimization"] = False
        optimizer = self._weights_optimizer_class(**weights_optimizer_args)
        optimizer.fit()

        phenotype = optimizer.get_fittest()["phenotype"]

        list_weights = np.split(phenotype, cut_points)
        for i, weights_i in enumerate(list_weights):
            ensemble._nets[i]._weights = weights_i.copy()

        print(optimizer.get_fittest()["fitness"])
        return ensemble

    def _predict(self, X: NDArray[np.float64]) -> NDArray[Union[np.float64, np.int64]]:
        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        fittest = self._optimizer.get_fittest()
        predict = fittest["phenotype"].average_output_classifier(X)[0]
        return predict
