from typing import Callable
from typing import Optional
from typing import Tuple
import numpy as np
from ..base import EvolutionaryAlgorithm
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


class DifferentialEvolution(EvolutionaryAlgorithm):
    '''Storn, Rainer & Price, Kenneth. (1995). Differential Evolution: A Simple and Efficient
    Adaptive Scheme for Global Optimization Over Continuous Spaces. Journal of Global Optimization. 23'''

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

        self._left = left
        self._right = right
        self._initial_population: np.ndarray
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
                               'current_to_pbest_1': current_to_pbest_1,
                               'rand_to_best1': rand_to_best1,
                               'best_2': best_2,
                               'rand_2': rand_2}

    def _mutation_and_crossover(self,
                                popuation_g: np.ndarray,
                                individ_g: np.ndarray,
                                F: float,
                                CR: float) -> np.ndarray:
        mutant = self._specified_mutation(individ_g, popuation_g, F)
        mutant_cr_g = binomial(individ_g, mutant, CR)
        mutant_cr_g = self._bounds_control(mutant_cr_g)
        return mutant_cr_g

    def _evaluate_and_selection(self,
                                mutant_cr_g: np.ndarray,
                                population_g: np.ndarray,
                                population_ph: np.ndarray,
                                fitness: np.ndarray) -> Tuple:
        offspring_g = population_g
        offspring_ph = population_ph
        offspring_fit = fitness

        mutant_cr_ph = self._get_phenotype(mutant_cr_g)
        mutant_cr_fit = self._evaluate(mutant_cr_ph)
        mask = mutant_cr_fit >= fitness
        offspring_g[mask] = mutant_cr_g[mask]
        offspring_ph[mask] = mutant_cr_ph[mask]
        offspring_fit[mask] = mutant_cr_fit[mask]
        return offspring_g, offspring_ph, offspring_fit, mask

    def _bounds_control(self,
                        individ_g: np.ndarray) -> np.ndarray:
        individ_g = individ_g.copy()
        low_mask = individ_g < self._left
        high_mask = individ_g > self._right

        individ_g[low_mask] = self._left[low_mask]
        individ_g[high_mask] = self._right[high_mask]
        return individ_g

    def set_strategy(self,
                     mutation_oper: str = 'rand_1',
                     F_param: float = 0.5,
                     CR_param: float = 0.5,
                     elitism_param: bool = True,
                     initial_population: Optional[np.ndarray] = None) -> None:
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
            population_g = self._initial_population

        population_ph = self._get_phenotype(population_g)
        fitness = self._evaluate(population_ph)

        argsort = np.argsort(fitness)
        population_g = population_g[argsort]
        population_ph = population_ph[argsort]
        fitness = fitness[argsort]

        for i in range(self._iters-1):
            self._update_fittest(population_g, population_ph, fitness)
            self._update_stats({'population_g': population_g.copy(),
                               'fitness_max': self._thefittest._fitness})

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
                    population_g[-1], population_ph[-1], fitness[-1] = self._thefittest.get()

                argsort = np.argsort(fitness)
                population_g = population_g[argsort]
                population_ph = population_ph[argsort]
                fitness = fitness[argsort]

        return self
