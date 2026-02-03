from __future__ import annotations

from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

from ..base import UniversalSet

from ..utils.transformations import minmax_scale

from . import SHAGA
from ._geneticprogramming import GeneticProgramming


class CSHAGP(GeneticProgramming, SHAGA):
    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float32]],
        uniset: UniversalSet,
        iters: int,
        pop_size: int,
        elitism: bool = True,
        max_level: int = 16,
        init_level: int = 4,
        init_population: Optional[NDArray[np.byte]] = None,
        genotype_to_phenotype: Optional[Callable[[NDArray[np.byte]], NDArray[Any]]] = None,
        optimal_value: Optional[float] = None,
        termination_error_value: float = 0.0,
        no_increase_num: Optional[int] = None,
        parents_num: int = 2,
        tour_size: int = 2,
        selection: str = "rank",
        crossover: str = "shagp_standard",
        mutation: str = "shagp_grow",
        minimization: bool = False,
        show_progress_each: Optional[int] = None,
        keep_history: bool = False,
        n_jobs: int = 1,
        fitness_function_args: Optional[Dict] = None,
        genotype_to_phenotype_args: Optional[Dict] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        on_generation: Optional[Callable] = None,
        fitness_update_eps: float = 0.0,
    ):
        GeneticProgramming.__init__(
            self,
            fitness_function=fitness_function,
            uniset=uniset,
            iters=iters,
            pop_size=pop_size,
            tour_size=tour_size,
            mutation_rate=0.05,
            parents_num=parents_num,
            elitism=elitism,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
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
            random_state=random_state,
            on_generation=on_generation,
            fitness_update_eps=fitness_update_eps,
        )

        SHAGA.__init__(
            self,
            fitness_function=fitness_function,
            iters=iters,
            pop_size=pop_size,
            str_len=1,
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
            random_state=random_state,
            on_generation=on_generation,
            fitness_update_eps=fitness_update_eps,
        )

        self._H_MR = np.full(self._H_size, 0.0, dtype=np.float64)
        self._H_CR = np.full(self._H_size, 0.0, dtype=np.float64)

    def _get_init_population(self) -> None:
        self._first_generation()
        self._population_ph_i = self._get_phenotype(self._population_g_i)
        self._fitness_i = self._get_fitness(self._population_ph_i)
        self._fitness_scale_i = minmax_scale(self._fitness_i)
        self._fitness_rank_i = rankdata(self._fitness_i)

    def _get_new_individ_g(
        self,
        specified_selection: str,
        specified_crossover: str,
        specified_mutation: str,
        individ_g: NDArray[np.float64],
        MR: float,
        CR: float,
    ) -> NDArray[np.float64]:
        selection_func, tour_size = self._selection_pool[specified_selection]
        crossover_func, quantity = self._crossover_pool[specified_crossover]
        mutation_func = self._mutation_pool[specified_mutation][0]

        selected_id = selection_func(
            self._fitness_scale_i,
            self._fitness_rank_i,
            np.int64(tour_size),
            np.int64(quantity),
        )

        second_parent = self._population_g_i[selected_id].copy()

        offspring_no_mutated = crossover_func(
            individ_g,
            second_parent,
            self._fitness_scale_i[selected_id],
            self._fitness_rank_i[selected_id],
            self._max_level,
            CR,
        )

        offspring = mutation_func(offspring_no_mutated, self._uniset, MR, self._max_level)
        return offspring

    def _get_new_population(self) -> None:
        get_new_individ_g = partial(
            self._get_new_individ_g,
            self._specified_selection,
            self._specified_crossover,
            self._specified_mutation,
        )

        self._MR, self._CR = self._generate_MR_CR(
            randc_scale=0.1,
            randc_lower=0.0,
            randc_upper=1.0,
            randn_scale=0.1,
            randn_lower=0.0,
            randn_upper=1.0,
        )

        mutant_cr_b_g = np.array(
            [
                get_new_individ_g(
                    individ_g=self._population_g_i[i],
                    MR=self._MR[i],
                    CR=self._CR[i],
                )
                for i in range(self._pop_size)
            ],
            dtype=object,
        )

        mutant_cr_ph = self._get_phenotype(mutant_cr_b_g)
        mutant_cr_fit = self._get_fitness(mutant_cr_ph)

        succeses_MR, succeses_CR, d_fitness = self._replace_population(
            mutant_cr_b_g,
            mutant_cr_ph,
            mutant_cr_fit,
        )

        self._fitness_scale_i = minmax_scale(self._fitness_i)
        self._fitness_rank_i = rankdata(self._fitness_i)

        next_k = (self._k + 1) % self._H_size
        self._H_MR[next_k] = self._update_u(self._H_MR[self._k], succeses_MR, d_fitness)
        self._H_CR[next_k] = self._update_u(self._H_CR[self._k], succeses_CR, d_fitness)
        self._k = next_k
