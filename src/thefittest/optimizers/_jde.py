import numpy as np
from functools import partial
from typing import Callable
from typing import Optional
from ._differentialevolution import DifferentialEvolution
from ..tools.random import float_population


class jDE(DifferentialEvolution):
    '''Brest, Janez & Greiner, Sao & Bošković, Borko & Mernik, Marjan & Zumer, Viljem. (2007).
    Self-Adapting Control Parameters in Differential Evolution: A Comparative Study on Numerical
    Benchmark Problems. Evolutionary Computation, IEEE Transactions on. 10. 646 - 657. 10.1109/TEVC.2006.872133. '''

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

        self._F_left: float
        self._F_right: float
        self._t_f: float
        self._t_cr: float

        self.set_strategy()

    def _get_new_F(self,
                   F: np.ndarray) -> np.ndarray:
        F = F.copy()
        mask = np.random.random(size=len(F)) < self._t_f

        random_values = np.random.random(size=np.sum(mask))
        F[mask] = self._F_left + random_values*self._F_right
        return F

    def _get_new_CR(self,
                    CR: np.ndarray) -> np.ndarray:
        CR = CR.copy()
        mask = np.random.random(size=len(CR)) < self._t_cr

        random_values = np.random.random(size=np.sum(mask))
        CR[mask] = random_values
        return CR

    def set_strategy(self,
                     mutation_oper: str = 'rand_1',
                     F_left_param: float = 0.1,
                     F_right_param: float = 0.9,
                     t_f_param: float = 0.1,
                     t_cr_param: float = 0.1,
                     elitism_param: bool = True,
                     initial_population: Optional[np.ndarray] = None) -> None:
        '''
        - mutation oper: must be a Tuple of:
            'best_1', 'rand_1', 'current_to_best_1', 'current_to_pbest_1',
            'rand_to_best1', 'best_2', 'rand_2', 
        '''
        self._update_pool()
        self._specified_mutation = self._mutation_pool[mutation_oper]
        self._F_left = F_left_param
        self._F_right = F_right_param
        self._t_f = t_cr_param
        self._t_cr = t_f_param
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

        F_i = np.full(self._pop_size, 0.5)
        CR_i = np.full(self._pop_size, 0.9)

        for i in range(self._iters-1):
            self._update_fittest(population_g, population_ph, fitness)
            self._update_stats({'population_g': population_g,
                                'fitness_max': self._thefittest._fitness,
                                'F': F_i.copy(),
                                'CR': CR_i.copy()})
            self._show_progress(i)
            if self._termitation_check():
                break
            else:
                F_i_new = self._get_new_F(F_i)
                CR_i_new = self._get_new_CR(CR_i)

                mutation_and_crossover = partial(self._mutation_and_crossover,
                                                 population_g)
                mutant_cr_g = np.array(list(map(mutation_and_crossover,
                                                population_g, F_i_new, CR_i_new)))

                stack = self._evaluate_and_selection(mutant_cr_g,
                                                     population_g,
                                                     population_ph,
                                                     fitness)
                population_g = stack[0]
                population_ph = stack[1]
                fitness = stack[2]

                succeses = stack[3]
                F_i[succeses] = F_i_new[succeses]
                CR_i[succeses] = CR_i_new[succeses]

                if self._elitism:
                    population_g[-1], population_ph[-1], fitness[-1] = self._thefittest.get()

                argsort = np.argsort(fitness)
                population_g = population_g[argsort]
                population_ph = population_ph[argsort]
                fitness = fitness[argsort]

        return self
