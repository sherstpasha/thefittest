import numpy as np
from typing import Optional
from typing import Callable
from typing import Any
from ._base import TheFittest
from ._base import StaticticSaDE
from ._base import LastBest
from functools import partial
from ._differentialevolution import DifferentialEvolution
from ._crossovers import binomial
from ._mutations import rand_1
from ._mutations import current_to_best_1
from ..tools import numpy_group_by


class jDE(DifferentialEvolution):
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
                 keep_history: bool = False):
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
        self.stats: StaticticSaDE
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
        if self.F_left_param is not None:
            self.F_left = F_left_param
        if self.F_right_param is not None:
            self.F_right = F_right_param
        if self.t_f_param is not None:
            self.t_f = t_cr_param
        if self.t_cr is not None:
            self.t_cr = t_f_param

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
        return offspring_g, offspring_ph, offspring_fit

    def bounds_control(self, individ_g):
        low_mask = individ_g < self.left
        high_mask = individ_g > self.right

        individ_g[low_mask] = self.left[low_mask]
        individ_g[high_mask] = self.right[high_mask]
        return individ_g

    def fit(self):
        population_g = self.generate_init_pop()
        population_ph = self.genotype_to_phenotype(population_g)
        fitness = self.evaluate(population_ph)

        argsort = np.argsort(fitness)
        population_g = population_g[argsort]
        population_ph = population_ph[argsort]
        fitness = fitness[argsort]

        self.thefittest = TheFittest().update(population_g,
                                              population_ph,
                                              fitness)
        lastbest = LastBest().update(self.thefittest.fitness)
        # if self.keep_history:
        #     self.stats = Statistics().update(population_g,
        #                                      population_ph,
        #                                      fitness)

        for i in range(self.iters-1):
            self.show_progress(i)
            if self.termitation_check(lastbest.no_increase):
                break
            else:
                F_i = np.full(self.pop_size-1, self.F)
                CR_i = np.full(self.pop_size-1, self.CR)

                partial_mut_and_cross = partial(self.mutation_and_crossover,
                                                population_g)
                mutant_cr_g = np.array(list(map(partial_mut_and_cross,
                                                population_g[:-1],
                                                F_i, CR_i)))

                stack = self.evaluate_and_selection(mutant_cr_g,
                                                    population_g[:-1],
                                                    population_ph[:-1],
                                                    fitness[:-1])
                population_g[:-1], population_ph[:-1], fitness[:-1] = stack

                population_g[-1], population_ph[-1], fitness[-1] = self.thefittest.get()
                argsort = np.argsort(fitness)
                population_g = population_g[argsort]
                population_ph = population_ph[argsort]
                fitness = fitness[argsort]

                self.thefittest.update(population_g, population_ph, fitness)
                lastbest.update(self.thefittest.fitness)
                if self.keep_history:
                    self.stats.update(population_g,
                                      population_ph,
                                      fitness)
        return self

