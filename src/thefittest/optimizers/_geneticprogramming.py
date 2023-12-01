from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ._geneticalgorithm import GeneticAlgorithm
from ..base import Tree
from ..base import UniversalSet
from ..tools import donothing
from ..tools.random import half_and_half


class GeneticProgramming(GeneticAlgorithm):
    """Koza, John R.. “Genetic programming - on the programming of computers by means
    of natural selection.” Complex Adaptive Systems (1993)"""

    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        uniset: UniversalSet,
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
        init_population: Optional[NDArray] = None,
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

        self._uniset: UniversalSet = uniset
        self._max_level: int = max_level
        self._init_level: int = init_level

    def _first_generation(self: GeneticProgramming) -> None:
        if self._init_population is None:
            self._population_g_i = self._population_g_i = half_and_half(
                pop_size=self._pop_size, uniset=self._uniset, max_level=self._init_level
            )
        else:
            self._population_g_i = self._init_population.copy()

    def _get_new_individ_g(
        self: GeneticProgramming,
        specified_selection: str,
        specified_crossover: str,
        specified_mutation: str,
    ) -> Tree:
        selection_func, tour_size = self._selection_pool[specified_selection]
        crossover_func, quantity = self._crossover_pool[specified_crossover]
        mutation_func, proba, is_constant_rate = self._mutation_pool[specified_mutation]

        selected_id = selection_func(
            self._fitness_scale_i, self._fitness_rank_i, np.int64(tour_size), np.int64(quantity)
        )

        offspring_no_mutated = crossover_func(
            self._population_g_i[selected_id],
            self._fitness_scale_i[selected_id],
            self._fitness_rank_i[selected_id],
            self._max_level,
        )

        if is_constant_rate:
            proba = proba
        else:
            proba = proba / len(offspring_no_mutated)

        offspring = mutation_func(offspring_no_mutated, self._uniset, proba, self._max_level)
        return offspring
