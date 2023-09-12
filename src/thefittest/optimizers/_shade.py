from __future__ import annotations

from functools import partial
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from ._jade import JADE
from ..tools import donothing
from ..tools import find_pbest_id
from ..tools.operators import binomial
from ..tools.operators import current_to_pbest_1_archive_p_min
from ..tools.random import float_population
from ..tools.random import randc01
from ..tools.random import randn01
from ..tools.transformations import bounds_control_mean
from ..tools.transformations import lehmer_mean


class SHADE(JADE):
    """Tanabe, Ryoji & Fukunaga, Alex. (2013). Success-history based parameter adaptation
    for Differential Evolution. 2013 IEEE Congress on Evolutionary Computation,
    CEC 2013. 71-78. 10.1109/CEC.2013.6557555."""

    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        iters: int,
        pop_size: int,
        left: NDArray[np.float64],
        right: NDArray[np.float64],
        genotype_to_phenotype: Callable[[NDArray[np.float64]], NDArray[Any]] = donothing,
        optimal_value: Optional[float] = None,
        termination_error_value: float = 0.0,
        no_increase_num: Optional[int] = None,
        minimization: bool = False,
        show_progress_each: Optional[int] = None,
        keep_history: bool = False,
    ):
        JADE.__init__(
            self,
            fitness_function=fitness_function,
            iters=iters,
            pop_size=pop_size,
            left=left,
            right=right,
            genotype_to_phenotype=genotype_to_phenotype,
            optimal_value=optimal_value,
            termination_error_value=termination_error_value,
            no_increase_num=no_increase_num,
            minimization=minimization,
            show_progress_each=show_progress_each,
            keep_history=keep_history,
        )

        self._H_size: int = pop_size

    def _shade_mutation_and_crossover(
        self,
        popuation_g: NDArray[np.float64],
        popuation_g_archive: NDArray[np.float64],
        pbest_id: NDArray[np.int64],
        individ_g: NDArray[np.float64],
        F: float,
        CR: float,
    ) -> NDArray[np.float64]:
        mutant = current_to_pbest_1_archive_p_min(
            individ_g, popuation_g, pbest_id, F, popuation_g_archive
        )

        mutant_cr_g = binomial(individ_g, mutant, CR)
        mutant_cr_g = bounds_control_mean(mutant_cr_g, self._left, self._right)
        return mutant_cr_g

    def _evaluate_and_selection(
        self,
        mutant_cr_g: NDArray[np.float64],
        population_g: NDArray[np.float64],
        population_ph: NDArray,
        fitness: NDArray[np.float64],
    ) -> Tuple:
        offspring_g = population_g.copy()
        offspring_ph = population_ph.copy()
        offspring_fit = fitness.copy()

        mutant_cr_ph = self._get_phenotype(mutant_cr_g)
        mutant_cr_fit = self._get_fitness(mutant_cr_ph)
        mask_more_equal = mutant_cr_fit >= fitness
        offspring_g[mask_more_equal] = mutant_cr_g[mask_more_equal]
        offspring_ph[mask_more_equal] = mutant_cr_ph[mask_more_equal]
        offspring_fit[mask_more_equal] = mutant_cr_fit[mask_more_equal]
        mask_more = mutant_cr_fit > fitness
        return offspring_g, offspring_ph, offspring_fit, mask_more

    def _generate_F_CR(
        self, H_F_i: NDArray[np.float64], H_CR_i: NDArray[np.float64], size: int
    ) -> Tuple:
        F_i = np.zeros(size)
        CR_i = np.zeros(size)
        for i in range(size):
            r_i = np.random.randint(0, len(H_F_i))
            u_F = H_F_i[r_i]
            u_CR = H_CR_i[r_i]
            F_i[i] = randc01(np.float64(u_F))
            CR_i[i] = randn01(np.float64(u_CR))
        return F_i, CR_i

    def _shade_update_u_F(self, u_F: float, S_F: NDArray[np.float64]) -> float:
        if len(S_F):
            return lehmer_mean(S_F)
        return u_F

    def _shade_update_u_CR(
        self, u_CR: float, S_CR: NDArray[np.float64], df: NDArray[np.float64]
    ) -> float:
        if len(S_CR):
            sum_ = np.sum(df)
            if sum_ > 0:
                weight_i = df / sum_
                return np.sum(weight_i * S_CR)
        return u_CR

    def set_strategy(
        self, elitism_param: bool = True, initial_population: Optional[NDArray[np.float64]] = None
    ) -> None:
        self._update_pool()
        self._elitism = elitism_param
        self._initial_population = initial_population

    def fit(self) -> SHADE:
        if self._initial_population is None:
            population_g = float_population(self._pop_size, self._left, self._right)
        else:
            population_g = self._initial_population.copy()

        H_F = np.full(self._H_size, 0.5)
        H_CR = np.full(self._H_size, 0.5)
        k = 0
        next_k = 1

        external_archive = np.zeros(shape=(0, len(self._left)))

        population_ph = self._get_phenotype(population_g)
        fitness = self._get_fitness(population_ph)
        pbest_id = find_pbest_id(fitness, np.float64(self._p))
        self._update_fittest(population_g, population_ph, fitness)
        self._update_stats(
            population_g=population_g, fitness_max=self._thefittest._fitness, H_F=H_F, H_CR=H_CR
        )
        for i in range(self._iters - 1):
            self._show_progress(i)
            if self._termitation_check():
                break
            else:
                F_i, CR_i = self._generate_F_CR(H_F, H_CR, self._pop_size)
                pop_archive = np.vstack([population_g, external_archive])

                mutation_and_crossover = partial(
                    self._shade_mutation_and_crossover, population_g, pop_archive, pbest_id
                )
                mutant_cr_g = np.array(list(map(mutation_and_crossover, population_g, F_i, CR_i)))

                stack = self._evaluate_and_selection(
                    mutant_cr_g, population_g, population_ph, fitness
                )

                succeses = stack[3]
                will_be_replaced_pop = population_g[succeses].copy()
                will_be_replaced_fit = fitness[succeses].copy()
                s_F = F_i[succeses]
                s_CR = CR_i[succeses]

                external_archive = self._append_archive(external_archive, will_be_replaced_pop)

                population_g = stack[0]
                population_ph = stack[1]
                fitness = stack[2]

                df = np.abs(will_be_replaced_fit - fitness[succeses])

                if self._elitism:
                    (
                        population_g[-1],
                        population_ph[-1],
                        fitness[-1],
                    ) = self._thefittest.get().values()
                pbest_id = find_pbest_id(fitness, np.float64(self._p))

                if next_k == self._H_size:
                    next_k = 0

                H_F[next_k] = self._shade_update_u_F(H_F[k], s_F)
                H_CR[next_k] = self._shade_update_u_CR(H_CR[k], s_CR, df)

                if k == self._H_size - 1:
                    k = 0
                    next_k = 1
                else:
                    k += 1
                    next_k += 1

                self._update_fittest(population_g, population_ph, fitness)
                self._update_stats(
                    population_g=population_g,
                    fitness_max=self._thefittest._fitness,
                    H_F=H_F,
                    H_CR=H_CR,
                )

        return self
