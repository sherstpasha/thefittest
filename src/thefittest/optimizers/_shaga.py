from functools import partial
from typing import Callable
from typing import Optional
from typing import Tuple
import numpy as np
from ..base._ea import EvolutionaryAlgorithm
from ..tools.random import cauchy_distribution
from ..tools.random import binary_string_population
from ..tools.operators import tournament_selection
from ..tools.operators import binomialGA
from ..tools.operators import flip_mutation
from ..tools.transformations import lehmer_mean


class SHAGA(EvolutionaryAlgorithm):
    '''Stanovov, Vladimir & Akhmedova, Shakhnaz & Semenkin, Eugene. (2019).
    Genetic Algorithm with Success History based Parameter Adaptation. 180-187.
    10.5220/0008071201800187. '''

    def __init__(self,
                 fitness_function: Callable,
                 genotype_to_phenotype: Callable,
                 iters: int,
                 pop_size: int,
                 str_len: int,
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

        self._str_len = str_len
        self._H_size = pop_size
        self._elitism: bool
        self._initial_population: np.ndarray
        self.set_strategy()

    def _selection_crossover_mutation(self,
                                      population_g: np.ndarray,
                                      fitness: np.ndarray,
                                      current: np.ndarray,
                                      MR,
                                      CR) -> np.ndarray:
        second_parent_id = tournament_selection(fitness, fitness, 2, 1)[0]
        second_parent = population_g[second_parent_id].copy()
        offspring = binomialGA(current, second_parent, CR)
        mutant = flip_mutation(offspring, MR)
        return mutant

    def _evaluate_replace(self,
                          mutant_cr_g: np.ndarray,
                          population_g: np.ndarray,
                          population_ph: np.ndarray,
                          fitness: np.ndarray) -> Tuple:
        offspring_g = population_g.copy()
        offspring_ph = population_ph.copy()
        offspring_fit = fitness.copy()

        mutant_cr_ph = self._get_phenotype(mutant_cr_g)
        mutant_cr_fit = self._evaluate(mutant_cr_ph)
        mask_more_equal = mutant_cr_fit >= fitness
        offspring_g[mask_more_equal] = mutant_cr_g[mask_more_equal]
        offspring_ph[mask_more_equal] = mutant_cr_ph[mask_more_equal]
        offspring_fit[mask_more_equal] = mutant_cr_fit[mask_more_equal]
        mask_more = mutant_cr_fit > fitness
        return offspring_g, offspring_ph, offspring_fit, mask_more

    def _generate_MR_CR(self, H_MR_i, H_CR_i, size):
        MR_i = np.zeros(size)
        CR_i = np.zeros(size)
        for i in range(size):
            r_i = np.random.randint(0, len(H_MR_i))
            u_MR = H_MR_i[r_i]
            u_CR = H_CR_i[r_i]
            MR_i[i] = self._randc(u_MR, 0.1/self._str_len)
            CR_i[i] = self._randn(u_CR, 0.1)
        return MR_i, CR_i

    def _randc(self, u, scale):
        value = cauchy_distribution(loc=u, scale=scale, size=1)[0]
        while value <= 0 or value > 5/self._str_len:
            value = cauchy_distribution(loc=u, scale=scale, size=1)[0]
        return value

    def _randn(self, u, scale):
        value = np.random.normal(u, scale)
        if value < 0:
            value = 0
        elif value > 1:
            value = 1
        return value

    def _update_u(self, u, S, df):
        if len(S):
            sum_ = np.sum(df)
            if sum_ > 0:
                weight_i = df/sum_
                return lehmer_mean(x=S, weight=weight_i)
        return u

    def set_strategy(self,
                     elitism_param: bool = True,
                     initial_population: Optional[np.ndarray] = None) -> None:
        self._elitism = elitism_param
        self._initial_population = initial_population

    def fit(self):
        H_MR = np.full(self._H_size, 1/(self._str_len))
        H_CR = np.full(self._H_size, 0.5)
        k = 0
        next_k = 1

        if self._initial_population is None:
            population_g = binary_string_population(
                self._pop_size, self._str_len)
        else:
            population_g = self._initial_population
        population_ph = self._get_phenotype(population_g)
        fitness = self._evaluate(population_ph)

        for i in range(self._iters-1):
            self._update_fittest(population_g, population_ph, fitness)
            self._update_stats({'population_g': population_g.copy(),
                                'fitness_max': self._thefittest._fitness,
                                'H_MR': H_MR.copy(),
                                'H_CR': H_CR.copy()})
            self._show_progress(i)
            if self._termitation_check():
                break
            else:
                MR_i, CR_i = self._generate_MR_CR(H_MR, H_CR, self._pop_size)

                partial_operators = partial(self._selection_crossover_mutation,
                                            population_g, fitness)

                mutant_cr_g = np.array(list(map(partial_operators,
                                                population_g,
                                                MR_i, CR_i)))

                stack = self._evaluate_replace(mutant_cr_g.copy(),
                                               population_g.copy(),
                                               population_ph.copy(),
                                               fitness.copy())

                succeses = stack[3]
                will_be_replaced_fit = fitness[succeses].copy()
                s_MR = MR_i[succeses]
                s_CR = CR_i[succeses]

                population_g = stack[0]
                population_ph = stack[1]
                fitness = stack[2]

                df = np.abs(will_be_replaced_fit - fitness[succeses])

                if self._elitism:
                    population_g[-1], population_ph[-1], fitness[-1] = self._thefittest.get()

                if next_k == self._H_size:
                    next_k = 0

                H_MR[next_k] = self._update_u(H_MR[k], s_MR, df)
                H_CR[next_k] = self._update_u(H_CR[k], s_CR, df)

                if k == self._H_size - 1:
                    k = 0
                    next_k = 1
                else:
                    k += 1
                    next_k += 1

        return self
