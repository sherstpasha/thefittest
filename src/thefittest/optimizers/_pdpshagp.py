from __future__ import annotations

from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

from ..base import UniversalSet
from ..utils.transformations import minmax_scale

from ._cshagp import CSHAGP
from ._pdpga import PDPGA


class PDPSHAGP(PDPGA, CSHAGP):
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
        selections: Tuple[str, ...] = (
            "proportional",
            "rank",
            "tournament_3",
            "tournament_5",
            "tournament_7",
        ),
        crossovers: Tuple[str, ...] = (
            "empty",
            "gp_standard",
            "gp_one_point",
            "gp_uniform_2",
            "gp_uniform_7",
            "gp_uniform_rank_2",
            "gp_uniform_rank_7",
            "gp_uniform_tour_3",
            "gp_uniform_tour_7",
        ),
        mutations: Tuple[str, ...] = ("point", "grow", "swap", "shrink"),
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
        PDPGA.__init__(
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
            parents_num=parents_num,
            tour_size=tour_size,
            selections=selections,
            crossovers=crossovers,
            mutations=mutations,
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

        CSHAGP.__init__(
            self,
            fitness_function=fitness_function,
            uniset=uniset,
            iters=iters,
            pop_size=pop_size,
            elitism=elitism,
            max_level=max_level,
            init_level=init_level,
            init_population=init_population,
            genotype_to_phenotype=genotype_to_phenotype,
            optimal_value=optimal_value,
            termination_error_value=termination_error_value,
            no_increase_num=no_increase_num,
            parents_num=parents_num,
            tour_size=tour_size,
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

        self._previous_fitness_i: List = []
        self._success_i: NDArray[np.bool_]

    def _get_new_individ_g(
        self: PDPSHAGP,
        specified_selection: str,
        specified_crossover: str,
        specified_mutation: str,
        individ_g: NDArray[np.float32],
        fitness_g_i: np.float32,
        MR: float,
        CR: float,
    ) -> NDArray[np.float32]:
        selection_func, tour_size = self._selection_pool[specified_selection]
        crossover_func, quantity = self._crossover_pool[specified_crossover]
        mutation_func = self._mutation_pool[specified_mutation]

        selected_id = selection_func(
            self._fitness_scale_i,
            self._fitness_rank_i,
            np.int64(tour_size),
            np.int64(quantity),
        )

        previous_fitness = self._choice_parent(np.append(self._fitness_i[selected_id], fitness_g_i))
        self._previous_fitness_i.append(previous_fitness)

        second_parent = self._population_g_i[selected_id].copy()

        offspring = crossover_func(
            individ_g,
            second_parent,
            self._fitness_scale_i[selected_id],
            self._fitness_rank_i[selected_id],
            self._max_level,
            CR,
        )
        mutant = mutation_func(offspring, self._uniset, MR, self._max_level)
        return mutant

    def _get_new_population(self: PDPSHAGP) -> None:
        get_new_individ_g = partial(
            self._get_new_individ_g,
        )
        self._MR, self._CR = self._generate_MR_CR()

        mutant_cr_b_g = np.array(
            [
                get_new_individ_g(
                    specified_selection=self._selection_operators[i],
                    specified_crossover=self._crossover_operators[i],
                    specified_mutation=self._mutation_operators[i],
                    individ_g=self._population_g_i[i],
                    fitness_g_i=self._fitness_i[i],
                    MR=self._MR[i],
                    CR=self._CR[i],
                )
                for i in range(self._pop_size)
            ],
        )

        mutant_cr_ph = self._get_phenotype(mutant_cr_b_g)
        mutant_cr_fit = self._get_fitness(mutant_cr_ph)
        mask = mutant_cr_fit >= self._fitness_i
        succeses = mutant_cr_fit > self._fitness_i

        succeses_MR = self._MR[succeses]
        succeses_CR = self._CR[succeses]

        will_be_replaced_fit = self._fitness_i[succeses].copy()

        self._population_g_i[mask] = mutant_cr_b_g[mask]
        self._population_ph_i[mask] = mutant_cr_ph[mask]
        self._fitness_i[mask] = mutant_cr_fit[mask]
        self._fitness_scale_i = minmax_scale(self._fitness_i)
        self._fitness_rank_i = rankdata(self._fitness_i)

        d_fitness = np.abs(will_be_replaced_fit - self._fitness_i[succeses])
        d_fitness[d_fitness == np.inf] = 1e4

        if self._k + 1 == self._H_size:
            next_k = 0
        else:
            next_k = self._k + 1

        self._H_MR[next_k] = self._update_u(self._H_MR[self._k], succeses_MR, d_fitness)
        self._H_CR[next_k] = self._update_u(self._H_CR[self._k], succeses_CR, d_fitness)

        if self._k == self._H_size - 1:
            self._k = 0
        else:
            self._k += 1
