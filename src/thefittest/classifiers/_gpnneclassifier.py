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
from ..base._net import NetEnsemble
from ..base._tree import DualNode
from ..base._tree import EnsembleUniversalSet
from ..classifiers import GeneticProgrammingNeuralNetClassifier
from ..classifiers._gpnnclassifier import genotype_to_phenotype_tree
from ..classifiers._gpnnclassifier import train_net
from ..optimizers import DifferentialEvolution
from ..optimizers import GeneticAlgorithm
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


weights_type_optimizer_alias = Union[
    Type[DifferentialEvolution],
    Type[jDE],
    Type[SHADE],
    Type[GeneticAlgorithm],
    Type[SelfCGA],
    Type[SHAGA],
]
weights_optimizer_alias = Union[DifferentialEvolution, jDE, SHADE, GeneticAlgorithm, SelfCGA, SHAGA]


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
    two_trees: MultiGenome,
    n_variables: int,
    n_outputs: int,
    output_activation: str,
    offset: bool,
) -> NetEnsemble:
    tree = two_trees[0]
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
    ens._meta_tree = two_trees[1]
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
                    two_trees=individ_g,
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


import torch


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
    # Создаем бутстрап-подвыборки для участников
    subsamples = create_bootstrap_subsamples(
        X_train_ens, proba_train_ens, n_subsamples=len(ensemble._nets), seed=123
    )
    print("Участники:")
    nets = []
    for subsample, net in zip(subsamples, ensemble._nets):
        trained_net = train_net(
            net=net,
            X_train=subsample[0],
            proba_train=subsample[1],
            weights_optimizer_args=weights_optimizer_args,
            weights_optimizer_class=weights_optimizer_class,
            fitness_function=fitness_function,
        )
        nets.append(trained_net)
    ensemble._nets = nets

    # Формируем метапризнаки
    X_meta = ensemble._get_meta_inputs(X_train_meta, offset=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_meta = torch.tensor(X_meta, device=device, dtype=torch.float32)

    # Преобразуем мета-таргеты для вычисления числа классов
    if isinstance(proba_train_meta, torch.Tensor):
        proba_train_meta_np = proba_train_meta.cpu().numpy()
    else:
        proba_train_meta_np = proba_train_meta
    y_meta = np.argmax(proba_train_meta_np, axis=1)

    tree = ensemble._meta_tree
    print("Мета-модель:")
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
        raise ValueError("Meta_net is None")


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


class TwoTreeGeneticProgramming(GeneticAlgorithm):
    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float32]],
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
        n_jobs: int = 1,
        fitness_function_args: Optional[Dict] = None,
        genotype_to_phenotype_args: Optional[Dict] = None,
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
            n_jobs=n_jobs,
            fitness_function_args=fitness_function_args,
            genotype_to_phenotype_args=genotype_to_phenotype_args,
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
        for i in range(len(population_g_i)):
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
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float32]],
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
        n_jobs: int = 1,
        fitness_function_args: Optional[Dict] = None,
        genotype_to_phenotype_args: Optional[Dict] = None,
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
            # threshold=threshold,
            genotype_to_phenotype=genotype_to_phenotype,
            optimal_value=optimal_value,
            termination_error_value=termination_error_value,
            no_increase_num=no_increase_num,
            minimization=minimization,
            show_progress_each=show_progress_each,
            keep_history=keep_history,
            n_jobs=n_jobs,
            fitness_function_args=fitness_function_args,
            genotype_to_phenotype_args=genotype_to_phenotype_args,
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
            n_jobs=n_jobs,
            fitness_function_args=fitness_function_args,
            genotype_to_phenotype_args=genotype_to_phenotype_args,
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
        optimizer: Union[
            Type[TwoTreeSelfCGP], Type[TwoTreeGeneticProgramming]
        ] = TwoTreeGeneticProgramming,
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
                TerminalNode(value={n_dimension}, name="in{}".format(len(variables_pool)))
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

    def _define_optimizer_ensembles(
        self: GeneticProgrammingNeuralNetStackingClassifier,
        uniset_1: UniversalSet,
        uniset_2: UniversalSet,
        n_outputs: int,
        X_train_ens: NDArray[np.float32],
        proba_train_ens: NDArray[np.float32],
        X_train_meta: NDArray[np.float32],
        proba_train_meta: NDArray[np.float32],
        X_test: NDArray[np.float32],
        target_test: NDArray[np.float32],
        fitness_function: Callable,
        evaluate_nets: Callable,
    ) -> Union[TwoTreeSelfCGP, TwoTreeGeneticProgramming]:
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
        optimizer_args["uniset_1"] = uniset_1
        optimizer_args["uniset_2"] = uniset_2
        optimizer_args["minimization"] = True
        return self._optimizer_class(**optimizer_args)

    def _fit(
        self: GeneticProgrammingNeuralNetStackingClassifier,
        X: NDArray[np.float32],
        y: NDArray[Union[np.float32, np.int64]],
    ) -> GeneticProgrammingNeuralNetStackingClassifier:
        # Если используется offset, добавляем столбец единиц
        if self._offset:
            X = np.hstack([X.copy(), np.ones((X.shape[0], 1))])

        n_outputs: int = len(set(y))
        eye: NDArray[np.float32] = np.eye(n_outputs, dtype=np.float32)

        # Определяем устройство (GPU, если доступно)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", device)

        # Делим данные на обучающие и тестовые
        X_train, X_test, y_train, y_test = train_test_split_stratified(
            X, y.astype(np.int64), self._test_sample_ratio
        )
        # Дополнительное разделение для участников ансамбля и метамодели
        X_train_ens, X_train_meta, y_train_ens, y_train_meta = train_test_split_stratified(
            X_train, y_train, 0.5
        )

        proba_test: NDArray[np.float32] = eye[y_test]
        proba_train_ens: NDArray[np.float32] = eye[y_train_ens]
        proba_train_meta: NDArray[np.float32] = eye[y_train_meta]

        # Преобразуем все данные в тензоры на нужном устройстве
        X_train_ens = torch.tensor(X_train_ens, device=device, dtype=torch.float32)
        X_train_meta = torch.tensor(X_train_meta, device=device, dtype=torch.float32)
        X_test = torch.tensor(X_test, device=device, dtype=torch.float32)
        proba_test = torch.tensor(proba_test, device=device, dtype=torch.float32)
        proba_train_ens = torch.tensor(proba_train_ens, device=device, dtype=torch.float32)
        proba_train_meta = torch.tensor(proba_train_meta, device=device, dtype=torch.float32)

        # Здесь используется _get_uniset_1 – этот метод остаётся неизменным,
        # также, если в коде есть _get_uniset_2, он сохраняется.
        self._optimizer = self._define_optimizer_ensembles(
            uniset_1=self._get_uniset_1(X),
            uniset_2=self._get_uniset_2(),
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

    def _predict(
        self: GeneticProgrammingNeuralNetClassifier, X: NDArray[np.float32]
    ) -> NDArray[Union[np.float32, np.int64]]:
        if self._offset:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Преобразуем входные данные в тензор
        X_tensor = torch.tensor(X, device=device, dtype=torch.float32)
        fittest = self._optimizer.get_fittest()
        # Вычисляем выход модели через forward (предполагается, что метод forward возвращает кортеж, где первый элемент – выход)
        output = fittest["phenotype"].forward(X_tensor)[0]
        # Возвращаем предсказанные классы
        return np.argmax(output.cpu().numpy(), axis=1)
