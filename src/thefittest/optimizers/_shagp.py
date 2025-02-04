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
from ..tools.operators import uniform_crossoverSHAGP, uniform_crossoverGP
from ..tools.operators import one_point_crossoverGP, standart_crossover
from ..tools.operators import point_mutation, growing_mutation
from ..tools.random import cauchy_distribution


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

    def _first_generation(self: GeneticProgramming) -> None:
        if self._init_population is None:
            self._population_g_i = self._population_g_i = half_and_half(
                pop_size=self._pop_size, uniset=self._uniset, max_level=self._init_level
            )
        else:
            self._population_g_i = self._init_population.copy()

    def _get_new_individ_g(
        self: SHAGA,
        individ_g: NDArray[np.float32],
        MR: float,
        CR: float,
    ) -> NDArray[np.float32]:
        # print(MR)
        # print(CR)
        second_parent_id = tournament_selection(self._fitness_i, self._fitness_i, 2, 1)[0]
        second_parent = self._population_g_i[second_parent_id].copy()

        # offspring = uniform_crossoverSHAGP(individ_g, second_parent, self._max_level, CR)
        offspring = standart_crossover([individ_g, second_parent], [1, 1], [1, 1], self._max_level)

        mutant = growing_mutation(offspring, self._uniset, MR, self._max_level)
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
