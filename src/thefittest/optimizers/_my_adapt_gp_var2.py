from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import List

import numpy as np
from numpy.typing import NDArray

from ._geneticprogramming import GeneticProgramming
from thefittest.optimizers._my_adapt_ga_var2 import MyAdaptGAVar2
from ..base import UniversalSet
from ..tools import donothing
from ..tools.operators import growing_mutation
from ..tools.operators import point_mutation
from ..tools.operators import shrink_mutation
from ..tools.operators import swap_mutation
from ..base import Tree


class MyAdaptGPVar2(GeneticProgramming, MyAdaptGAVar2):
    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        uniset: UniversalSet,
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
        mutation: str = "point",
        max_level: int = 16,
        init_level: int = 4,
        init_population: Optional[NDArray] = None,
        adaptation_operator: str = "tournament_k",
        adaptation_tour_size: int = 2,
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
        MyAdaptGAVar2.__init__(
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
            init_population=init_population,
            adaptation_operator=adaptation_operator,
            adaptation_tour_size=adaptation_tour_size,
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

        GeneticProgramming.__init__(
            self,
            fitness_function=fitness_function,
            uniset=uniset,
            iters=iters,
            pop_size=pop_size,
            tour_size=tour_size,
            mutation_rate=mutation_rate,
            parents_num=parents_num,
            elitism=elitism,
            max_level=max_level,
            init_level=init_level,
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

        self._adapt_mutation_pool: Dict[str, Callable] = {
            "point": point_mutation,
            "grow": growing_mutation,
            "swap": swap_mutation,
            "shrink": shrink_mutation,
        }
        self._adapt_specified_mutation: str = mutation

    def _get_new_population(self: MyAdaptGAVar2) -> None:
        self._population_g_i = np.array(
            [
                self._get_new_individ_g(
                    self._selection_operators[i],
                    self._crossover_operators[i],
                    self._mutation_probas[i],
                )
                for i in range(self._pop_size)
            ],
            dtype=self._population_g_i.dtype,
        )

    def _get_new_individ_g(
        self: MyAdaptGPVar2,
        specified_selection: str,
        specified_crossover: str,
        specified_mutation_proba: np.float64,
    ) -> Tree:
        selection_func, tour_size = self._selection_pool[specified_selection]
        crossover_func, quantity = self._crossover_pool[specified_crossover]
        selected_id = selection_func(
            self._fitness_scale_i, self._fitness_rank_i, np.int64(tour_size), np.int64(quantity)
        )

        offspring_no_mutated = crossover_func(
            self._population_g_i[selected_id],
            self._fitness_scale_i[selected_id],
            self._fitness_rank_i[selected_id],
            self._max_level,
        )

        mutation = self._adapt_mutation_pool[self._adapt_specified_mutation]

        offspring = mutation(
            offspring_no_mutated,
            self._uniset,
            np.float64(specified_mutation_proba),
            self._max_level,
        )
        return offspring
