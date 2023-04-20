from functools import partial
from typing import Callable
from typing import Optional
from typing import Tuple
import numpy as np
from ._jade import JADE
from ..tools.generators import cauchy_distribution
from ..tools.transformations import lehmer_mean
from ..tools.generators import float_population


class SHADE(JADE):
    '''Tanabe, Ryoji & Fukunaga, Alex. (2013). Success-history based parameter adaptation
    for Differential Evolution. 2013 IEEE Congress on Evolutionary Computation,
    CEC 2013. 71-78. 10.1109/CEC.2013.6557555. '''

    def __init__(self,
                 fitness_function: Callable,
                 genotype_to_phenotype: Callable,
                 iters: int,
                 pop_size: int,
                 left: np.ndarray,
                 right: np.ndarray,
                 optimal_value: Optional[float] = None,
                 termination_error_value: float = 0.,
                 no_increase_num: Optional[int] = None,
                 minimization: bool = False,
                 show_progress_each: Optional[int] = None,
                 keep_history: bool = False):
        JADE.__init__(
            self,
            fitness_function=fitness_function,
            genotype_to_phenotype=genotype_to_phenotype,
            iters=iters,
            pop_size=pop_size,
            left=left,
            right=right,
            optimal_value=optimal_value,
            termination_error_value=termination_error_value,
            no_increase_num=no_increase_num,
            minimization=minimization,
            show_progress_each=show_progress_each,
            keep_history=keep_history)

        self._H_size: int = pop_size

    def _current_to_pbest_1_archive(self,
                                    current: np.ndarray,
                                    population: np.ndarray,
                                    F: float,
                                    pop_archive: np.ndarray) -> np.ndarray:
        range_pop = range(len(population))
        range_pop_arch = range(len(pop_archive))
        p_min = 2/len(population)
        p_i = np.random.uniform(p_min, 0.2)
        value = int(p_i*len(population))
        pbest = population[-value:]
        p_best_ind = np.random.randint(0, len(pbest))
        best = pbest[p_best_ind]

        r1 = np.random.choice(range_pop, size=1, replace=False)[0]
        r2 = np.random.choice(range_pop_arch, size=1, replace=False)[0]
        offspring = current + F*(best - current) + \
            F*(population[r1] - pop_archive[r2])
        return offspring

    def _evaluate_and_selection(self,
                                mutant_cr_g: np.ndarray,
                                population_g: np.ndarray,
                                population_ph: np.ndarray,
                                fitness: np.ndarray) -> np.ndarray:
        offspring_g = population_g
        offspring_ph = population_ph
        offspring_fit = fitness

        mutant_cr_ph = self._get_phenotype(mutant_cr_g)
        mutant_cr_fit = self._evaluate(mutant_cr_ph)
        mask_more_equal = mutant_cr_fit >= fitness
        offspring_g[mask_more_equal] = mutant_cr_g[mask_more_equal]
        offspring_ph[mask_more_equal] = mutant_cr_ph[mask_more_equal]
        offspring_fit[mask_more_equal] = mutant_cr_fit[mask_more_equal]
        mask_more = mutant_cr_fit > fitness
        return offspring_g, offspring_ph, offspring_fit, mask_more

    def _generate_F_CR(self,
                       H_F_i: np.ndarray,
                       H_CR_i: np.ndarray,
                       size: int) -> Tuple:
        F_i = np.zeros(size)
        CR_i = np.zeros(size)
        for i in range(size):
            r_i = np.random.randint(0, len(H_F_i))
            u_F = H_F_i[r_i]
            u_CR = H_CR_i[r_i]
            F_i[i] = self._randc01(u_F)
            CR_i[i] = self._randn01(u_CR)
        return F_i, CR_i

    def _randc01(self,
                 u: float) -> float:
        value = cauchy_distribution(loc=u, scale=0.1)[0]
        while value <= 0:
            value = cauchy_distribution(loc=u, scale=0.1)[0]
        if value > 1:
            value = 1
        return value

    def _randn01(self,
                 u: float) -> float:
        value = np.random.normal(u, 0.1)
        if value < 0:
            value = 0
        elif value > 1:
            value = 1
        return value

    def _update_u_F(self,
                    u_F: float,
                    S_F: np.ndarray) -> float:
        if len(S_F):
            return lehmer_mean(S_F)
        return u_F

    def _update_u_CR(self,
                     u_CR: float,
                     S_CR: np.ndarray,
                     df: np.ndarray) -> float:
        if len(S_CR):
            sum_ = np.sum(df)
            if sum_ > 0:
                weight_i = df/sum_
                return np.sum(weight_i*S_CR)
        return u_CR

    def set_strategy(self,
                     elitism_param: bool = True,
                     initial_population: Optional[int] = None) -> None:
        self._update_pool()
        self._elitism = elitism_param
        self._initial_population = initial_population
        return self

    def fit(self):

        if self._initial_population is None:
            population_g = float_population(
                self._pop_size, self._left, self._right)
        else:
            population_g = self._initial_population

        H_F = np.full(self._H_size, 0.5)
        H_CR = np.full(self._H_size, 0.5)
        k = 0
        next_k = 1

        external_archive = np.zeros(shape=(0, len(self._left)))

        population_ph = self._get_phenotype(population_g)
        fitness = self._evaluate(population_ph)

        argsort = np.argsort(fitness)
        population_g = population_g[argsort]
        population_ph = population_ph[argsort]
        fitness = fitness[argsort]

        for i in range(self._iters-1):
            self._update_fittest(population_g, population_ph, fitness)
            self._update_stats({'population_g': population_g.copy(),
                                'fitness_max': self._thefittest._fitness,
                                'H_F': H_F.copy(),
                                'H_CR': H_CR.copy()})
            self._show_progress(i)
            if self._termitation_check():
                break
            else:
                F_i, CR_i = self._generate_F_CR(H_F, H_CR, self._pop_size)
                pop_archive = np.vstack([population_g, external_archive])

                mutation_and_crossover = partial(self._mutation_and_crossover,
                                                 population_g, pop_archive)
                mutant_cr_g = np.array(list(map(mutation_and_crossover,
                                                population_g, F_i, CR_i)))

                stack = self._evaluate_and_selection(mutant_cr_g,
                                                     population_g,
                                                     population_ph,
                                                     fitness)

                succeses = stack[3]
                will_be_replaced_pop = population_g[succeses].copy()
                will_be_replaced_fit = fitness[succeses].copy()
                s_F = F_i[succeses]
                s_CR = CR_i[succeses]

                external_archive = self._append_archive(external_archive,
                                                        will_be_replaced_pop)

                population_g = stack[0]
                population_ph = stack[1]
                fitness = stack[2]

                df = np.abs(will_be_replaced_fit - fitness[succeses])

                if self._elitism:
                    population_g[-1], population_ph[-1], fitness[-1] = self._thefittest.get()

                argsort = np.argsort(fitness)
                population_g = population_g[argsort]
                population_ph = population_ph[argsort]
                fitness = fitness[argsort]

                if next_k == self._H_size:
                    next_k = 0

                H_F[next_k] = self._update_u_F(H_F[k], s_F)
                H_CR[next_k] = self._update_u_CR(H_CR[k], s_CR, df)

                if k == self._H_size - 1:
                    k = 0
                    next_k = 1
                else:
                    k += 1
                    next_k += 1

        return self
