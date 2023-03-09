import numpy as np
from typing import Optional
from typing import Callable
from typing import Any
from ._base import TheFittest
from ._base import EvolutionaryAlgorithm
from ._base import LastBest
from functools import partial
from ..tools.operators import binomial
from ..tools.operators import best_1
from ..tools.operators import best_2
from ..tools.operators import rand_to_best1
from ..tools.operators import current_to_best_1
from ..tools.operators import rand_1
from ..tools.operators import current_to_pbest_1
from ..tools.operators import rand_2
from ..tools.generators import float_population


class Statistics:
    def __init__(self, mode='quick'):
        self.mode = mode
        self.population_g = np.array([])
        self.fitness = np.array([])

    def append_arr(self, arr_to, arr_from):
        shape_to = (-1, arr_from.shape[0], arr_from.shape[1])
        shape_from = (1, arr_from.shape[0], arr_from.shape[1])
        result = np.vstack([arr_to.reshape(shape_to),
                            arr_from.copy().reshape(shape_from)])
        return result

    def update(self,
               population_g_i: np.ndarray,
               fitness_i: np.ndarray):
        if self.mode == 'quick':
            self.fitness = np.append(self.fitness, np.max(fitness_i))
        elif self.mode == 'full':
            self.fitness = np.append(self.fitness, np.max(fitness_i))
            self.population_g = self.append_arr(self.population_g,
                                                population_g_i)
        else:
            raise ValueError('the "mode" must be either "quick" or "full"')
        return self


class DifferentialEvolution(EvolutionaryAlgorithm):
    '''Storn, Rainer & Price, Kenneth. (1995). Differential Evolution: A Simple and Efficient
    Adaptive Scheme for Global Optimization Over Continuous Spaces. Journal of Global Optimization. 23'''

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
        EvolutionaryAlgorithm.__init__(
            self,
            fitness_function=fitness_function,
            genotype_to_phenotype=genotype_to_phenotype,
            iters=iters,
            pop_size=pop_size,
            optimal_value=optimal_value,
            termination_error_value=termination_error_value,
            no_increase_num=no_increase_num,
            minimization=minimization,
            show_progress_each=show_progress_each,
            keep_history=keep_history)

        self.left = left
        self.right = right
        self.thefittest: TheFittest
        self.stats: Statistics
        self.initial_population: Optional[np.ndarray]
        self.m_function: Callable
        self.F: float
        self.CR: float
        self.elitism: bool

        self.set_strategy()

    def generate_init_pop(self):
        if self.initial_population is None:
            population_g = float_population(
                self.pop_size, self.left, self.right)
        else:
            population_g = self.initial_population
        return population_g

    def update_pool(self):
        self.m_pool = {'best_1': best_1,
                       'rand_1': rand_1,
                       'current_to_best_1': current_to_best_1,
                       'current_to_pbest_1': current_to_pbest_1,
                       'rand_to_best1': rand_to_best1,
                       'best_2': best_2,
                       'rand_2': rand_2}

    def set_strategy(self,
                     mutation_oper: str = 'rand_1',
                     F_param: float = 0.5,
                     CR_param: float = 0.5,
                     elitism_param: bool = True,
                     initial_population: Optional[np.ndarray] = None):
        
        self.update_pool()
        self.m_function = self.m_pool[mutation_oper]
        self.F = F_param
        self.CR = CR_param
        self.elitism = elitism_param
        self.initial_population = initial_population
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
        return offspring_g, offspring_ph, offspring_fit

    def bounds_control(self, individ_g):
        individ_g = individ_g.copy()
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
        if self.keep_history is not None:
            self.stats = Statistics(
                mode=self.keep_history).update(population_g,
                                               fitness)

        for i in range(self.iters-1):
            self.show_progress(i)
            if self.termitation_check(lastbest.no_increase):
                break
            else:
                F_i = np.full(self.pop_size, self.F)
                CR_i = np.full(self.pop_size, self.CR)

                partial_mut_and_cross = partial(self.mutation_and_crossover,
                                                population_g)
                mutant_cr_g = np.array(list(map(partial_mut_and_cross,
                                                population_g,
                                                F_i, CR_i)))

                stack = self.evaluate_and_selection(mutant_cr_g,
                                                    population_g,
                                                    population_ph,
                                                    fitness)
                population_g, population_ph, fitness = stack

                if self.elitism:
                    population_g[-1], population_ph[-1], fitness[-1] = self.thefittest.get()
                argsort = np.argsort(fitness)
                population_g = population_g[argsort]
                population_ph = population_ph[argsort]
                fitness = fitness[argsort]

                self.thefittest.update(population_g, population_ph, fitness)
                lastbest.update(self.thefittest.fitness)
                if self.keep_history is not None:
                    self.stats.update(population_g,
                                      fitness)
        return self
