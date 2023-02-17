import numpy as np
from typing import Optional
from typing import Callable
from typing import Any
from ._base import TheFittest
from ._base import LastBest
from functools import partial
from ._differentialevolution import DifferentialEvolution
from ._crossovers import binomial


class StatisticsjDE:
    def __init__(self, mode='quick'):
        self.mode = mode
        self.population_g = np.array([])
        self.fitness = np.array([])
        self.F = np.array([], dtype=float)
        self.CR = np.array([], dtype=float)

    def append_arr(self, arr_to, arr_from):
        shape_to = (-1, arr_from.shape[0], arr_from.shape[1])
        shape_from = (1, arr_from.shape[0], arr_from.shape[1])
        result = np.vstack([arr_to.reshape(shape_to),
                            arr_from.copy().reshape(shape_from)])
        return result

    def update(self,
               population_g_i: np.ndarray,
               fitness_i: np.ndarray,
               F_i, CR_i):
        if self.mode == 'quick':
            self.fitness = np.append(self.fitness, np.max(fitness_i))
        elif self.mode == 'full':
            self.fitness = np.append(self.fitness, np.max(fitness_i))
            self.population_g = self.append_arr(self.population_g,
                                                population_g_i)
            if not len(self.F):
                self.F = F_i.copy().reshape(1, -1)
                self.CR = CR_i.copy().reshape(1, -1)
            else:
                self.F = np.append(self.F, F_i.copy().reshape(1, -1), axis=0)
                self.CR = np.append(
                    self.CR, CR_i.copy().reshape(1, -1), axis=0)
        else:
            raise ValueError('the "mode" must be either "quick" or "full"')
        return self


class jDE(DifferentialEvolution):
    '''Brest, Janez & Greiner, Sao & Bošković, Borko & Mernik, Marjan & Zumer, Viljem. (2007).
    Self-Adapting Control Parameters in Differential Evolution: A Comparative Study on Numerical
    Benchmark Problems. Evolutionary Computation, IEEE Transactions on. 10. 646 - 657. 10.1109/TEVC.2006.872133. '''

    def __init__(self,
                 fitness_function: Callable[[np.ndarray[Any]], np.ndarray[float]],
                 genotype_to_phenotype: Callable[[np.ndarray[Any]], np.ndarray[Any]],
                 iters: int,
                 pop_size: int,
                 left: np.ndarray[float],
                 right: np.ndarray[float],
                 optimal_value: Optional[float] = None,
                 termination_error_value: float = 0.,
                 no_increase_num: Optional[int] = None,
                 minimization: bool = False,
                 show_progress_each: Optional[int] = None,
                 keep_history: Optional[str] = None):
        DifferentialEvolution.__init__(
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

        self.thefittest: TheFittest
        self.stats: StatisticsjDE
        self.m_function = self.m_pool['rand_1']
        self.F_left = 0.1
        self.F_right = 0.9
        self.t_f = 0.1
        self.t_cr = 0.1

    def set_strategy(self,
                     F_left_param: Optional[float] = None,
                     F_right_param: Optional[float] = None,
                     t_f_param: Optional[float] = None,
                     t_cr_param: Optional[float] = None):
        if F_left_param is not None:
            self.F_left = F_left_param
        if F_right_param is not None:
            self.F_right = F_right_param
        if t_f_param is not None:
            self.t_f = t_cr_param
        if t_cr_param is not None:
            self.t_cr = t_f_param
        return self

    def mutation_and_crossover(self, popuation_g, individ_g, F_i, CR_i):
        mutant = self.m_function(individ_g, popuation_g, F_i)

        mutant_cr_g = binomial(individ_g, mutant, CR_i)
        mutant_cr_g = self.bounds_control(mutant_cr_g)
        return mutant_cr_g

    def evaluate_and_selection(self, mutant_cr_g, population_g, population_ph, fitness):
        offspring_g = population_g.copy()
        offspring_ph = population_ph.copy()
        offspring_fit = fitness.copy()

        mutant_cr_ph = self.genotype_to_phenotype(mutant_cr_g)
        mutant_cr_fit = self.evaluate(mutant_cr_ph)
        mask = mutant_cr_fit >= fitness
        offspring_g[mask] = mutant_cr_g[mask]
        offspring_ph[mask] = mutant_cr_ph[mask]
        offspring_fit[mask] = mutant_cr_fit[mask]
        return offspring_g, offspring_ph, offspring_fit, mask

    def bounds_control(self, individ_g):
        low_mask = individ_g < self.left
        high_mask = individ_g > self.right

        individ_g[low_mask] = self.left[low_mask]
        individ_g[high_mask] = self.right[high_mask]
        return individ_g

    def regenerate_F(self, F_i):
        mask = np.random.random(size=len(F_i)) < self.t_f
        F_i[mask] = self.F_left + \
            np.random.random(size=np.sum(mask))*self.F_right
        return F_i

    def regenerate_CR(self, CR_i):
        mask = np.random.random(size=len(CR_i)) < self.t_cr
        CR_i[mask] = np.random.random(size=np.sum(mask))
        return CR_i

    def fit(self):
        population_g = self.generate_init_pop()
        population_ph = self.genotype_to_phenotype(population_g)
        fitness = self.evaluate(population_ph)

        argsort = np.argsort(fitness)
        population_g = population_g[argsort]
        population_ph = population_ph[argsort]
        fitness = fitness[argsort]

        F_i = np.full(self.pop_size-1, 0.5)
        CR_i = np.full(self.pop_size-1, 0.9)

        self.thefittest = TheFittest().update(population_g,
                                              population_ph,
                                              fitness)
        lastbest = LastBest().update(self.thefittest.fitness)
        if self.keep_history is not None:
            self.stats = StatisticsjDE(
                mode=self.keep_history).update(population_g,
                                               fitness,
                                               F_i,
                                               CR_i)

        for i in range(self.iters-1):
            self.show_progress(i)
            if self.termitation_check(lastbest.no_increase):
                break
            else:

                F_i_new = self.regenerate_F(F_i.copy())
                CR_i_new = self.regenerate_CR(CR_i.copy())

                partial_mut_and_cross = partial(self.mutation_and_crossover,
                                                population_g)
                mutant_cr_g = np.array(list(map(partial_mut_and_cross,
                                                population_g[:-1],
                                                F_i_new, CR_i_new)))

                stack = self.evaluate_and_selection(mutant_cr_g,
                                                    population_g[:-1],
                                                    population_ph[:-1],
                                                    fitness[:-1])
                population_g[:-1] = stack[0]
                population_ph[:-1] = stack[1]
                fitness[:-1] = stack[2]

                succeses = stack[3]
                F_i[succeses] = F_i_new[succeses]
                CR_i[succeses] = CR_i_new[succeses]

                population_g[-1], population_ph[-1], fitness[-1] = self.thefittest.get()
                argsort = np.argsort(fitness)
                population_g = population_g[argsort]
                population_ph = population_ph[argsort]
                fitness = fitness[argsort]

                self.thefittest.update(population_g, population_ph, fitness)
                lastbest.update(self.thefittest.fitness)
                if self.keep_history is not None:
                    self.stats.update(population_g,
                                      fitness,
                                      F_i,
                                      CR_i)
        return self
