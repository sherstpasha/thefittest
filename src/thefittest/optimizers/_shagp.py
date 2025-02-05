from __future__ import annotations

from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from ..base import Tree

from ._geneticprogramming import GeneticProgramming
from ._selfcga import SelfCGA
from ..base import UniversalSet
from ..tools import donothing
from ..base._ea import EvolutionaryAlgorithm
from ..optimizers import SHAGA
from ..tools.random import half_and_half
from ..tools.operators import tournament_selection
from ..tools.operators import uniform_crossoverSHAGP
from ..tools.operators import standart_crossover_shagp
from ..tools.operators import one_point_crossover_SHAGP
from ..tools.operators import point_mutation, growing_mutation, swap_mutation, shrink_mutation
from ..tools.random import cauchy_distribution
from typing import Union
from ..tools.operators import proportional_selection
from ..tools.operators import rank_selection
from ..tools.transformations import rank_data
from ..tools.transformations import scale_data


class SHAGP(SHAGA):
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
        genotype_to_phenotype: Callable[[NDArray[np.byte]], NDArray[Any]] = donothing,
        optimal_value: Optional[float] = None,
        termination_error_value: float = 0.0,
        no_increase_num: Optional[int] = None,
        parents_num: int = 2,
        tour_size: int = 2,
        selection: str = "rank",
        crossover: str = "gp_standart",
        mutation: str = "shrink",
        minimization: bool = False,
        show_progress_each: Optional[int] = None,
        keep_history: bool = False,
        n_jobs: int = 1,
        fitness_function_args: Optional[Dict] = None,
        genotype_to_phenotype_args: Optional[Dict] = None,
    ):
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
        )

        self._uniset: UniversalSet = uniset
        self._max_level: int = max_level
        self._init_level: int = init_level
        self._H_MR = np.full(self._H_size, 0.1, dtype=np.float32)
        self._H_CR = np.full(self._H_size, 0.5, dtype=np.float32)
        self._parents_num: int = parents_num
        self._tour_size: int = tour_size
        self._specified_selection: str = selection
        self._specified_crossover: str = crossover
        self._specified_mutation: str = mutation

        self._selection_pool: Dict[str, Tuple[Callable, Union[float, int]]] = {
            "proportional": (proportional_selection, 0),
            "rank": (rank_selection, 0),
            "tournament_k": (tournament_selection, self._tour_size),
            "tournament_3": (tournament_selection, 3),
            "tournament_5": (tournament_selection, 5),
            "tournament_7": (tournament_selection, 7),
        }

        self._crossover_pool: Dict[str, Tuple[Callable, Union[float, int]]] = {
            # "empty": (empty_crossover_shaga, 1),
            "gp_standart": (standart_crossover_shagp, 1),
            "gp_uniform_1": (uniform_crossoverSHAGP, 1),
            "gp_one_point": (one_point_crossover_SHAGP, 1),
            # "gp_uniform_7": (uniform_crossoverGP, 7),
            # "gp_uniform_k": (uniform_crossoverGP, self._parents_num),
            # "gp_uniform_prop_2": (uniform_prop_crossover_GP, 2),
            # "gp_uniform_prop_7": (uniform_prop_crossover_GP, 7),
            # "gp_uniform_prop_k": (uniform_prop_crossover_GP, self._parents_num),
            # "gp_uniform_rank_2": (uniform_rank_crossover_GP, 2),
            # "gp_uniform_rank_7": (uniform_rank_crossover_GP, 7),
            # "gp_uniform_rank_k": (uniform_rank_crossover_GP, self._parents_num),
            # "gp_uniform_tour_3": (uniform_tour_crossover_GP, 3),
            # "gp_uniform_tour_7": (uniform_tour_crossover_GP, 7),
            # "gp_uniform_tour_k": (uniform_tour_crossover_GP, self._parents_num),
        }

        self._mutation_pool: Dict[str, Callable] = {
            "point": point_mutation,
            "grow": growing_mutation,
            "swap": swap_mutation,
            "shrink": shrink_mutation,
        }

    def _first_generation(self: GeneticProgramming) -> None:
        if self._init_population is None:
            self._population_g_i = self._population_g_i = half_and_half(
                pop_size=self._pop_size, uniset=self._uniset, max_level=self._init_level
            )
        else:
            self._population_g_i = self._init_population.copy()

    def _get_init_population(self: SHAGP) -> None:
        self._first_generation()
        self._population_ph_i = self._get_phenotype(self._population_g_i)
        self._fitness_i = self._get_fitness(self._population_ph_i)
        self._fitness_scale_i = scale_data(self._fitness_i)
        self._fitness_rank_i = rank_data(self._fitness_i)

    def _get_new_individ_g(
        self: SHAGA,
        specified_selection: str,
        specified_crossover: str,
        specified_mutation: str,
        individ_g: NDArray[np.float32],
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

    def _randc(self: SHAGA, u: float, scale: float) -> NDArray[np.float32]:
        value = cauchy_distribution(loc=u, scale=scale, size=1)[0]
        while value <= 0 or value > 1:
            value = cauchy_distribution(loc=u, scale=scale, size=1)[0]
        return value

    def _generate_MR_CR(self: SHAGA) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        MR_i = np.zeros(self._pop_size)
        CR_i = np.zeros(self._pop_size)
        for i in range(self._pop_size):
            r_i = np.random.randint(0, self._H_size)
            u_MR = self._H_MR[r_i]
            u_CR = self._H_CR[r_i]
            MR_i[i] = self._randc(u_MR, 0.1)
            CR_i[i] = self._randn(u_CR, 0.1)
        return MR_i, CR_i

    def _get_new_population(self: SHAGA) -> None:
        get_new_individ_g = partial(
            self._get_new_individ_g,
            self._specified_selection,
            self._specified_crossover,
            self._specified_mutation,
        )
        self._MR, self._CR = self._generate_MR_CR()

        mutant_cr_b_g = np.array(
            [
                get_new_individ_g(individ_g=self._population_g_i[i], MR=self._MR[i], CR=self._CR[i])
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
        self._fitness_scale_i = scale_data(self._fitness_i)
        self._fitness_rank_i = rank_data(self._fitness_i)

        d_fitness = np.abs(will_be_replaced_fit - self._fitness_i[succeses])

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
