from __future__ import annotations

from typing import Any
from typing import Callable
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
from ..base._ea import MultiGenome
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
from ..tools.random import train_test_split_stratified, train_test_split
from ..tools.transformations import GrayCode
from ..base._tree import Operator
from ..classifiers import GeneticProgrammingNeuralNetClassifier
from ..classifiers import MLPEAClassifier
from ..base._ea import MultiGenome
from ..base import UniversalSet
from ..tools import donothing
from ..tools.random import half_and_half


weights_type_optimizer_alias = Union[
    Type[DifferentialEvolution],
    Type[jDE],
    Type[SHADE],
    Type[GeneticAlgorithm],
    Type[SelfCGA],
    Type[SHAGA],
]
weights_optimizer_alias = Union[DifferentialEvolution, jDE, SHADE, GeneticAlgorithm, SelfCGA, SHAGA]


class TwoTreeGeneticProgramming(GeneticAlgorithm):
    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        uniset_1: UniversalSet,
        uniset_2: UniversalSet,
        iters: int,
        pop_size: int,
        tour_size: int = 2,
        mutation_rate: float = 0.05,
        parents_num: int = 7,
        elitism: bool = True,
        selection: str = "rank",
        crossover: str = "gp_standart",
        mutation: str = "gp_weak_grow",
        max_level: int = 16,
        init_level: int = 5,
        genotype_to_phenotype: Callable[[NDArray[Any]], NDArray[Any]] = donothing,
        optimal_value: Optional[float] = None,
        termination_error_value: float = 0.0,
        no_increase_num: Optional[int] = None,
        minimization: bool = False,
        show_progress_each: Optional[int] = None,
        keep_history: bool = False,
    ):
        GeneticAlgorithm.__init__(
            self,
            fitness_function=fitness_function,
            iters=iters,
            pop_size=pop_size,
            str_len=1,
            tour_size=tour_size,
            mutation_rate=mutation_rate,
            parents_num=parents_num,
            elitism=elitism,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            init_population=None,
            genotype_to_phenotype=genotype_to_phenotype,
            optimal_value=optimal_value,
            termination_error_value=termination_error_value,
            no_increase_num=no_increase_num,
            minimization=minimization,
            show_progress_each=show_progress_each,
            keep_history=keep_history,
        )

        self._uniset_1: UniversalSet = uniset_1
        self._uniset_2: UniversalSet = uniset_2
        self._max_level: int = max_level
        self._init_level: int = init_level

    def _first_generation(self: TwoTreeGeneticProgramming) -> None:
        population_g_i_1 = self._population_g_i = half_and_half(
            pop_size=self._pop_size, uniset=self._uniset_1, max_level=self._init_level
        )
        population_g_i_2 = self._population_g_i = half_and_half(
            pop_size=self._pop_size, uniset=self._uniset_2, max_level=self._init_level
        )

        populations = [population_g_i_1, population_g_i_2]

        population_g_i = np.empty(shape=self._pop_size, dtype=object)
        for i, population_i in enumerate(population_g_i):
            population_g_i[i] = MultiGenome(
                genotypes=tuple(populations_j[i] for populations_j in populations)
            )
        self._population_g_i = population_g_i

    def _get_new_individ_g(
        self: TwoTreeGeneticProgramming,
        specified_selection: str,
        specified_crossover: str,
        specified_mutation: str,
    ) -> MultiGenome:
        selection_func, tour_size = self._selection_pool[specified_selection]
        crossover_func, quantity = self._crossover_pool[specified_crossover]
        mutation_func, proba, is_constant_rate = self._mutation_pool[specified_mutation]

        selected_id = selection_func(
            self._fitness_scale_i, self._fitness_rank_i, np.int64(tour_size), np.int64(quantity)
        )

        selected_g_1 = [selected_g[0] for selected_g in self._population_g_i[selected_id]]
        selected_g_2 = [selected_g[1] for selected_g in self._population_g_i[selected_id]]

        offspring_no_mutated_1 = crossover_func(
            selected_g_1,
            self._fitness_scale_i[selected_id],
            self._fitness_rank_i[selected_id],
            self._max_level,
        )

        if is_constant_rate:
            proba = proba
        else:
            proba = proba / len(offspring_no_mutated_1)

        offspring_1 = mutation_func(offspring_no_mutated_1, self._uniset_1, proba, self._max_level)

        offspring_no_mutated_2 = crossover_func(
            selected_g_2,
            self._fitness_scale_i[selected_id],
            self._fitness_rank_i[selected_id],
            self._max_level,
        )

        if is_constant_rate:
            proba = proba
        else:
            proba = proba / len(offspring_no_mutated_2)

        offspring_2 = mutation_func(offspring_no_mutated_2, self._uniset_2, proba, self._max_level)

        return MultiGenome(genotypes=(offspring_1.copy(), offspring_2.copy()))


class TwoTreeSelfCGP(TwoTreeGeneticProgramming, SelfCGA):
    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        uniset_1: UniversalSet,
        uniset_2: UniversalSet,
        iters: int,
        pop_size: int,
        tour_size: int = 2,
        mutation_rate: float = 0.05,
        parents_num: int = 2,
        elitism: bool = True,
        selections: Tuple[str, ...] = (
            "proportional",
            "rank",
            "tournament_3",
            "tournament_5",
            "tournament_7",
        ),
        crossovers: Tuple[str, ...] = ("gp_standart", "gp_one_point", "gp_uniform_rank_2"),
        mutations: Tuple[str, ...] = (
            "gp_weak_point",
            "gp_average_point",
            "gp_strong_point",
            "gp_weak_grow",
            "gp_average_grow",
            "gp_strong_grow",
        ),
        max_level: int = 16,
        init_level: int = 4,
        init_population: Optional[NDArray] = None,
        K: float = 2,
        threshold: float = 0.05,
        genotype_to_phenotype: Callable[[NDArray], NDArray[Any]] = donothing,
        optimal_value: Optional[float] = None,
        termination_error_value: float = 0.0,
        no_increase_num: Optional[int] = None,
        minimization: bool = False,
        show_progress_each: Optional[int] = None,
        keep_history: bool = False,
    ):
        SelfCGA.__init__(
            self,
            fitness_function=fitness_function,
            iters=iters,
            pop_size=pop_size,
            str_len=1,
            tour_size=tour_size,
            mutation_rate=mutation_rate,
            parents_num=parents_num,
            elitism=elitism,
            selections=selections,
            crossovers=crossovers,
            mutations=mutations,
            init_population=init_population,
            K=K,
            threshold=threshold,
            genotype_to_phenotype=genotype_to_phenotype,
            optimal_value=optimal_value,
            termination_error_value=termination_error_value,
            no_increase_num=no_increase_num,
            minimization=minimization,
            show_progress_each=show_progress_each,
            keep_history=keep_history,
        )

        TwoTreeGeneticProgramming.__init__(
            self,
            fitness_function=fitness_function,
            uniset_1=uniset_1,
            uniset_2=uniset_2,
            iters=iters,
            pop_size=pop_size,
            tour_size=tour_size,
            mutation_rate=mutation_rate,
            parents_num=parents_num,
            elitism=elitism,
            max_level=max_level,
            init_level=init_level,
            genotype_to_phenotype=genotype_to_phenotype,
            optimal_value=optimal_value,
            termination_error_value=termination_error_value,
            no_increase_num=no_increase_num,
            minimization=minimization,
            show_progress_each=show_progress_each,
            keep_history=keep_history,
        )


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

    def _get_uniset_1(
        self: GeneticProgrammingNeuralNetStackingClassifier, X: NDArray[np.float64]
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

    def _get_uniset_2(self: GeneticProgrammingNeuralNetStackingClassifier) -> EnsembleUniversalSet:
        uniset: EnsembleUniversalSet

        def random_hidden_block() -> HiddenBlock:
            return HiddenBlock(self._max_hidden_block_size)

        functional_set = [
            FunctionalNode(Add()),
            FunctionalNode(More()),
        ]

        terminal_set: List[Union[TerminalNode, EphemeralNode]] = [
            EphemeralNode(random_hidden_block),
        ]

        terminal_set.append(EphemeralNode(random_hidden_block))

        uniset = EnsembleUniversalSet(tuple(functional_set), tuple(terminal_set))
        return uniset

    def _split_tree(
        self: GeneticProgrammingNeuralNetStackingClassifier, tree: Tree
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
        self, n_variables: int, n_outputs: int, two_trees: MultiGenome
    ) -> NetEnsemble:
        tree = two_trees[0]
        print(tree)
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

        ens = NetEnsemble(nets=np.array(nets, dtype=object))
        ens._meta_tree = two_trees[1]

        return ens

    def _genotype_to_phenotype(
        self: GeneticProgrammingNeuralNetStackingClassifier,
        X_train_ens: NDArray[np.float64],
        proba_train_ens: NDArray[np.float64],
        X_train_meta: NDArray[np.float64],
        proba_train_meta: NDArray[np.float64],
        population_g: NDArray,
        n_outputs: int,
    ) -> NDArray:
        n_variables: int = X_train_ens.shape[1]

        population_ph: NDArray = np.empty(shape=len(population_g), dtype=object)

        population_ph = np.array(
            [
                self._train_ensemble(
                    ensemble=self._genotype_to_phenotype_ensemble(
                        n_variables, n_outputs, individ_g
                    ),
                    X_train_ens=X_train_ens,
                    proba_train_ens=proba_train_ens,
                    X_train_meta=X_train_meta,
                    proba_train_meta=proba_train_meta,
                )
                for individ_g in population_g
            ],
            dtype=object,
        )

        # func = lambda ens: self._train_ensemble(
        #     ens, X_train_ens, proba_train_ens, X_train_meta, proba_train_meta
        # )

        # population_ph = np.array(list(map(func, population_ph)), dtype=object)

        return population_ph

    def _fitness_function(
        self, population: NDArray, X: NDArray[np.float64], targets: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        fitness = np.empty(shape=self._pop_size, dtype=np.float64)
        for i, ensemble in enumerate(population):
            output2d = ensemble.meta_output(X)
            fitness[i] = categorical_crossentropy(targets, output2d)
            print(fitness[i], len(ensemble._nets))

        return fitness

    def _train_ensemble(
        self,
        ensemble: NetEnsemble,
        X_train_ens: NDArray[np.float64],
        proba_train_ens: NDArray[np.float64],
        X_train_meta: NDArray[np.float64],
        proba_train_meta: NDArray[np.float64],
    ) -> NetEnsemble:
        # if self._cache_condition:
        #     for ensemble_i in self._cache:
        #         if ensemble_i == ensemble:
        #             return ensemble_i.copy()
        n_nets = len(ensemble)
        print(n_nets)

        trained_ensemble = self._train_ensemble_individually(ensemble, X_train_ens, proba_train_ens)

        X_meta = trained_ensemble._get_meta_inputs(X_train_meta, offset=True)
        y_meta = np.argmax(proba_train_meta, axis=1)

        tree = trained_ensemble._meta_tree
        print(tree)
        meta_net = self._genotype_to_phenotype_tree(X_meta.shape[1], len(set(y_meta)), tree)
        meta_net._offset = True

        trained_ensemble._meta_algorithm = self._train_net(meta_net, X_meta, proba_train_meta)

        # meta_model = MLPEAClassifier(
        #     iters=100,
        #     pop_size=100,
        #     hidden_layers=(5,),
        #     weights_optimizer=self._weights_optimizer_class,
        #     offset=True,
        # )

        # meta_model.fit(X_meta, y_meta)
        # optimizer = meta_model.get_optimizer()
        # trained_ensemble._meta_algorithm = meta_model._net.copy()

        # print("meta error optimization", optimizer.get_fittest()["fitness"], X_meta.shape)

        output2d = ensemble.meta_output(X_train_ens)
        error_meta_train = categorical_crossentropy(proba_train_ens, output2d)
        print("meta error ", error_meta_train)

        return trained_ensemble

    def _train_ensemble_individually(
        self, ensemble: NetEnsemble, X_train: NDArray[np.float64], proba_train: NDArray[np.float64]
    ) -> NetEnsemble:
        if self._cache_condition:
            for ensemble_i in self._cache:
                if ensemble_i == ensemble:
                    return ensemble_i.copy()
        for net in ensemble._nets:
            self._train_net(net, X_train, proba_train)

        return ensemble

    def _fit(
        self: GeneticProgrammingNeuralNetStackingClassifier,
        X: NDArray[np.float64],
        y: NDArray[Union[np.float64, np.int64]],
    ) -> GeneticProgrammingNeuralNetStackingClassifier:
        optimizer_args: dict[str, Any]

        if self._offset:
            X = np.hstack([X.copy(), np.ones((X.shape[0], 1))])

        n_outputs: int = len(set(y))
        eye: NDArray[np.float64] = np.eye(n_outputs, dtype=np.float64)

        X_train, X_test, y_train, y_test = train_test_split_stratified(
            X, y.astype(np.int64), self._test_sample_ratio
        )

        X_train_ens, X_train_meta, y_train_ens, y_train_meta = train_test_split_stratified(
            X_train, y_train, 0.5
        )

        proba_test: NDArray[np.float64] = eye[y_test]
        proba_train_ens: NDArray[np.float64] = eye[y_train_ens]
        proba_train_meta: NDArray[np.float64] = eye[y_train_meta]

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

        uniset_1: EnsembleUniversalSet = self._get_uniset_1(X)
        uniset_2: EnsembleUniversalSet = self._get_uniset_2()

        optimizer_args["fitness_function"] = lambda population: self._fitness_function(
            population, X_test, proba_test
        )
        optimizer_args["genotype_to_phenotype"] = lambda trees: self._genotype_to_phenotype(
            X_train_ens=X_train_ens,
            proba_train_ens=proba_train_ens,
            X_train_meta=X_train_meta,
            proba_train_meta=proba_train_meta,
            population_g=trees,
            n_outputs=n_outputs,
        )
        optimizer_args["iters"] = self._iters
        optimizer_args["pop_size"] = self._pop_size
        optimizer_args["uniset_1"] = uniset_1
        optimizer_args["uniset_2"] = uniset_2
        optimizer_args["minimization"] = True

        self._optimizer = self._optimizer_class(**optimizer_args)
        self._optimizer.fit()

        return self

    def _predict(self, X: NDArray[np.float64]) -> NDArray[Union[np.float64, np.int64]]:
        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        fittest = self._optimizer.get_fittest()
        output = fittest["phenotype"].meta_output(X)

        return np.argmax(output, axis=1)
