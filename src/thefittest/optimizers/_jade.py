from functools import partial
from typing import Callable
from typing import Optional

import numpy as np

from ._differentialevolution import DifferentialEvolution
from ..tools import donothing
from ..tools import find_pbest_id
from ..tools.operators import binomial
from ..tools.operators import current_to_pbest_1_archive
from ..tools.random import cauchy_distribution
from ..tools.random import float_population
from ..tools.transformations import bounds_control_mean
from ..tools.transformations import lehmer_mean


class JADE(DifferentialEvolution):
    '''Zhang, Jingqiao & Sanderson, A.C.. (2009). JADE: Adaptive Differential Evolution
      With Optional External Archive.
     Evolutionary Computation, IEEE Transactions on. 13. 945 - 958. 10.1109/TEVC.2009.2014613. '''

    def __init__(self,
                 fitness_function: Callable,
                 iters: int,
                 pop_size: int,
                 left: np.ndarray,
                 right: np.ndarray,
                 genotype_to_phenotype: Callable = donothing,
                 optimal_value: Optional[float] = None,
                 termination_error_value: float = 0.,
                 no_increase_num: Optional[int] = None,
                 minimization: bool = False,
                 show_progress_each: Optional[int] = None,
                 keep_history: bool = False):
        DifferentialEvolution.__init__(
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
            keep_history=keep_history)

        self._c: float = 0.1
        self._p: float = 0.05

        self.set_strategy()

    def _generate_F(self,
                    u_F: float) -> np.ndarray:
        F_i = cauchy_distribution(loc=u_F, scale=0.1, size=self._pop_size)
        mask = F_i <= 0
        while np.any(mask):
            F_i[mask] = cauchy_distribution(loc=u_F, scale=0.1,
                                            size=len(F_i[mask]))
            mask = F_i <= 0
        F_i[F_i >= 1] = 1
        return F_i

    def _generate_CR(self,
                     u_CR: float) -> np.ndarray:
        CR_i = np.random.normal(u_CR, 0.1, self._pop_size)
        CR_i[CR_i >= 1] = 1
        CR_i[CR_i <= 0] = 0
        return CR_i

    def _update_u_F(self,
                    u_F: float,
                    S_F: np.ndarray) -> float:
        if len(S_F):
            u_F = (1 - self._c) * u_F + self._c * lehmer_mean(S_F)
        return u_F

    def _update_u_CR(self,
                     u_CR: float,
                     S_CR: np.ndarray) -> float:
        if len(S_CR):
            u_CR = (1 - self._c) * u_CR + self._c * np.mean(S_CR)
        return u_CR

    def _append_archive(self,
                        archive: np.ndarray,
                        worse_i: np.ndarray) -> np.ndarray:
        archive = np.append(archive, worse_i, axis=0)
        if len(archive) > self._pop_size:
            np.random.shuffle(archive)
            archive = archive[:self._pop_size]
        return archive

    def _mutation_and_crossover(self,
                                popuation_g: np.ndarray,
                                popuation_g_archive: np.ndarray,
                                pbest_id: np.ndarray,
                                individ_g: np.ndarray,
                                F: float,
                                CR: float) -> np.ndarray:
        mutant = current_to_pbest_1_archive(individ_g, popuation_g,
                                            pbest_id, F, popuation_g_archive)

        mutant_cr_g = binomial(individ_g, mutant, CR)
        mutant_cr_g = bounds_control_mean(mutant_cr_g,
                                          self._left,
                                          self._right)
        return mutant_cr_g

    def set_strategy(self,
                     c_param: float = 0.1,
                     p_param: float = 0.05,
                     elitism_param: bool = True,
                     initial_population: Optional[int] = None) -> None:
        self._update_pool()
        self._c = c_param
        self._p = p_param
        self._elitism = elitism_param
        self._initial_population = initial_population

    def fit(self):

        if self._initial_population is None:
            population_g = float_population(
                self._pop_size, self._left, self._right)
        else:
            population_g = self._initial_population.copy()

        u_F = u_CR = 0.5
        external_archive = np.zeros(shape=(0, len(self._left)))

        population_ph = self._get_phenotype(population_g)
        fitness = self._get_fitness(population_ph)
        pbest_id = find_pbest_id(fitness, np.float64(self._p))
        self._update_fittest(population_g, population_ph, fitness)
        self._update_stats(population_g=population_g,
                           fitness_max=self._thefittest._fitness,
                           u_F=u_F,
                           u_CR=u_CR)

        for i in range(self._iters - 1):

            self._show_progress(i)
            if self._termitation_check():
                break
            else:
                F_i = self._generate_F(u_F)
                CR_i = self._generate_CR(u_CR)
                pop_archive = np.vstack([population_g, external_archive])

                mutation_and_crossover = partial(self._mutation_and_crossover,
                                                 population_g, pop_archive, pbest_id)
                mutant_cr_g = np.array(list(map(mutation_and_crossover,
                                                population_g, F_i, CR_i)))

                stack = self._evaluate_and_selection(mutant_cr_g,
                                                     population_g,
                                                     population_ph,
                                                     fitness)

                succeses = stack[3]
                will_be_replaced = population_g[succeses].copy()
                s_F = F_i[succeses]
                s_CR = CR_i[succeses]

                external_archive = self._append_archive(external_archive,
                                                        will_be_replaced)

                population_g = stack[0]
                population_ph = stack[1]
                fitness = stack[2]

                if self._elitism:
                    population_g[-1], population_ph[-1], fitness[-1] =\
                        self._thefittest.get().values()
                pbest_id = find_pbest_id(fitness, np.float64(self._p))

                u_F = self._update_u_F(u_F, s_F)
                u_CR = self._update_u_CR(u_CR, s_CR)
                self._update_fittest(population_g, population_ph, fitness)
                self._update_stats(population_g=population_g,
                                   fitness_max=self._thefittest._fitness,
                                   u_F=u_F,
                                   u_CR=u_CR)

        return self
