from __future__ import annotations

from collections import defaultdict
from collections import Counter
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union
from typing import List

import numpy as np
from numpy.typing import NDArray

from ._geneticalgorithm import GeneticAlgorithm
from ..tools import donothing
from ..tools.transformations import numpy_group_by

#MyAdaptGA отличается от MyAdaptGAVar2 тем, что в одном случае мутация имеет 3 значения (оператор), а во втором вероятность мутации - вещественная
class MyAdaptGA(GeneticAlgorithm):
    """Semenkin, E.S., Semenkina, M.E. Self-configuring Genetic Algorithm with Modified Uniform
    Crossover Operator. LNCS, 7331, 2012, pp. 414-421. https://doi.org/10.1007/978-3-642-30976-2_50
    """

    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        iters: int,
        pop_size: int,
        str_len: int,
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
        crossovers: Tuple[str, ...] = (
            "empty",
            "one_point",
            "two_point",
            "uniform_2",
            "uniform_7",
            "uniform_prop_2",
            "uniform_prop_7",
            "uniform_rank_2",
            "uniform_rank_7",
            "uniform_tour_3",
            "uniform_tour_7",
        ),
        mutations: Tuple[str, ...] = ("weak", "average", "strong"),
        init_population: Optional[NDArray[np.byte]] = None,
        adaptation_operator: str = "rank",
        adaptation_tour_size: int = 2,
        genotype_to_phenotype: Callable[[NDArray[np.byte]], NDArray[Any]] = donothing,
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
            str_len=str_len,
            tour_size=tour_size,
            mutation_rate=mutation_rate,
            parents_num=parents_num,
            elitism=elitism,
            init_population=init_population,
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

        self._adaptation_operator = self._selection_pool[adaptation_operator]
        self._adaptation_tour_size = adaptation_tour_size

        self._selection_set: Dict[str, Tuple[Callable, Union[float, int]]] = {}
        self._crossover_set: Dict[str, Tuple[Callable, Union[float, int]]] = {}
        self._mutation_set: Dict[str, Tuple[Callable, Union[float, int], bool]] = {}

        for operator_name in selections:
            self._selection_set[operator_name] = self._selection_pool[operator_name]
        self._selection_set = dict(sorted(self._selection_set.items()))

        for operator_name in crossovers:
            self._crossover_set[operator_name] = self._crossover_pool[operator_name]
        self._crossover_set = dict(sorted(self._crossover_set.items()))

        for operator_name in mutations:
            self._mutation_set[operator_name] = self._mutation_pool[operator_name]
        self._mutation_set = dict(sorted(self._mutation_set.items()))

        self._selection_operators: NDArray = self._random_choice_operators(
            list(self._selection_set.keys()), self._pop_size
        )
        self._crossover_operators: NDArray = self._random_choice_operators(
            list(self._crossover_set.keys()), self._pop_size
        )
        self._mutation_operators: NDArray = self._random_choice_operators(
            list(self._mutation_set.keys()), self._pop_size
        )

    def _random_choice_operators(self: MyAdaptGA, operators_set: List[str], size: int) -> NDArray:
        chosen_operator = np.random.choice(operators_set, size)
        return chosen_operator

    def _mutate_operators(self: MyAdaptGA, operators: NDArray, operators_set: List[str]) -> NDArray:
        new_operators = operators.copy()
        roll = np.random.random(size=len(operators)) < 1 / len(operators)
        n_replaces = np.sum(roll, dtype=int)

        if n_replaces > 0:
            new_operators[roll] = self._random_choice_operators(operators_set, size=n_replaces)
        return new_operators

    def _choice_operators_by_selection(
        self: MyAdaptGA,
        operators: NDArray,
        fitness: NDArray[np.float64],
        fitness_rank: NDArray[np.float64],
    ) -> NDArray:
        selection, _ = self._adaptation_operator
        tour_size = self._adaptation_tour_size

        quantity = len(operators)
        selected_id = selection(fitness, fitness_rank, np.int64(tour_size), np.int64(quantity))
        new_operators = operators[selected_id]
        return new_operators

    def _update_data(self: MyAdaptGA) -> None:
        super()._update_data()

        keys, values = np.unique(self._selection_operators, return_counts=True)
        selection_used = dict(zip(keys, values))

        for key in self._selection_set.keys():
            if key not in selection_used:
                selection_used[key] = 0

        keys, values = np.unique(self._crossover_operators, return_counts=True)
        crossover_used = dict(zip(keys, values))

        for key in self._crossover_set.keys():
            if key not in crossover_used:
                crossover_used[key] = 0

        keys, values = np.unique(self._mutation_operators, return_counts=True)
        mutation_used = dict(zip(keys, values))

        for key in self._mutation_set.keys():
            if key not in mutation_used:
                mutation_used[key] = 0

        self._update_stats(
            s_used=selection_used,
            c_used=crossover_used,
            m_used=mutation_used,
        )

    def _adapt(self: MyAdaptGA) -> None:
        self._selection_operators = self._mutate_operators(
            self._choice_operators_by_selection(
                self._selection_operators, self._fitness_scale_i, self._fitness_rank_i
            ),
            list(self._selection_set.keys()),
        )

        self._crossover_operators = self._mutate_operators(
            self._choice_operators_by_selection(
                self._crossover_operators, self._fitness_scale_i, self._fitness_rank_i
            ),
            list(self._crossover_set.keys()),
        )

        self._mutation_operators = self._mutate_operators(
            self._choice_operators_by_selection(
                self._mutation_operators, self._fitness_scale_i, self._fitness_rank_i
            ),
            list(self._mutation_set.keys()),
        )
