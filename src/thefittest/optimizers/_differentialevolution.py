from functools import partial
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from ..base._ea import EvolutionaryAlgorithm
from ..tools import donothing
from ..tools.operators import best_1
from ..tools.operators import best_2
from ..tools.operators import binomial
from ..tools.operators import current_to_best_1
from ..tools.operators import rand_1
from ..tools.operators import rand_2
from ..tools.operators import rand_to_best1
from ..tools.random import float_population
from ..tools.transformations import bounds_control


class DifferentialEvolution(EvolutionaryAlgorithm):
    '''Storn, Rainer & Price, Kenneth. (1995). Differential Evolution: A Simple and Efficient
    Adaptive Scheme for Global Optimization Over Continuous Spaces.
    Journal of Global Optimization. 23'''

    def __init__(self,
                 fitness_function: Callable,
                 iters: int,
                 pop_size: int,
                 left: NDArray[np.float64],
                 right: NDArray[np.float64],
                 genotype_to_phenotype: Callable = donothing,
                 optimal_value: Optional[float] = None,
                 termination_error_value: float = 0.,
                 no_increase_num: Optional[int] = None,
                 minimization: bool = False,
                 show_progress_each: Optional[int] = None,
                 keep_history: bool = False):
        EvolutionaryAlgorithm.__init__(
            self,
            fitness_function=fitness_function,
            iters=iters,
            pop_size=pop_size,
            genotype_to_phenotype=genotype_to_phenotype,
            optimal_value=optimal_value,
            termination_error_value=termination_error_value,
            no_increase_num=no_increase_num,
            minimization=minimization,
            show_progress_each=show_progress_each,
            keep_history=keep_history)

        self._left = left
        self._right = right
        self._initial_population: NDArray[np.float64]
        self._mutation_pool: dict
        self._specified_mutation: Callable
        self._F: float
        self._CR: float
        self._elitism: bool

        self.set_strategy()

    def _update_pool(self):
        self._mutation_pool = {'best_1': best_1,
                               'rand_1': rand_1,
                               'current_to_best_1': current_to_best_1,
                               'rand_to_best1': rand_to_best1,
                               'best_2': best_2,
                               'rand_2': rand_2}

    def _mutation_and_crossover(self,
                                popuation_g: NDArray[np.float64],
                                individ_g: NDArray[np.float64],
                                F: float,
                                CR: float) -> NDArray[np.float64]:
        mutant = self._specified_mutation(individ_g,
                                          self._thefittest._genotype,
                                          popuation_g,
                                          np.float64(F))
        mutant_cr_g = binomial(individ_g, mutant, np.float64(CR))
        mutant_cr_g = bounds_control(mutant_cr_g, self._left, self._right)
        return mutant_cr_g

    def _evaluate_and_selection(self,
                                mutant_cr_g: NDArray[np.float64],
                                population_g: NDArray[np.float64],
                                population_ph: NDArray[Any],
                                fitness: NDArray[np.float64]) -> Tuple:
        offspring_g = population_g.copy()
        offspring_ph = population_ph.copy()
        offspring_fit = fitness.copy()

        mutant_cr_ph = self._get_phenotype(mutant_cr_g)
        mutant_cr_fit = self._get_fitness(mutant_cr_ph)
        mask = mutant_cr_fit >= fitness
        offspring_g[mask] = mutant_cr_g[mask]
        offspring_ph[mask] = mutant_cr_ph[mask]
        offspring_fit[mask] = mutant_cr_fit[mask]
        return offspring_g, offspring_ph, offspring_fit, mask

    def set_strategy(self,
                     mutation_oper: str = 'rand_1',
                     F_param: float = 0.5,
                     CR_param: float = 0.5,
                     elitism_param: bool = True,
                     initial_population: Optional[NDArray[np.float64]] = None) -> None:
        '''
        - mutation oper: must be a Tuple of:
            'best_1', 'rand_1', 'current_to_best_1',
            'rand_to_best1', 'best_2', 'rand_2'
        '''
        self._update_pool()
        self._specified_mutation = self._mutation_pool[mutation_oper]
        self._F = F_param
        self._CR = CR_param
        self._elitism = elitism_param
        self._initial_population = initial_population

    def fit(self):
        if self._initial_population is None:
            population_g = float_population(
                self._pop_size, self._left, self._right)
        else:
            population_g = self._initial_population.copy()

        population_ph = self._get_phenotype(population_g)
        fitness = self._get_fitness(population_ph)
        self._update_fittest(population_g, population_ph, fitness)
        self._update_stats(population_g=population_g,
                           fitness_max=self._thefittest._fitness)

        for i in range(self._iters-1):

            self._show_progress(i)
            if self._termitation_check():
                break
            else:
                F_i = [self._F]*self._pop_size
                CR_i = [self._CR]*self._pop_size

                mutation_and_crossover = partial(self._mutation_and_crossover,
                                                 population_g)
                mutant_cr_g = np.array(list(map(mutation_and_crossover,
                                                population_g, F_i, CR_i)))

                stack = self._evaluate_and_selection(mutant_cr_g,
                                                     population_g,
                                                     population_ph,
                                                     fitness)
                population_g, population_ph, fitness, _ = stack

                if self._elitism:
                    population_g[-1], population_ph[-1], fitness[-1] =\
                        self._thefittest.get().values()
                self._update_fittest(population_g, population_ph, fitness)
                self._update_stats(population_g=population_g,
                                   fitness_max=self._thefittest._fitness)

        return self
