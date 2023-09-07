from functools import partial
from typing import Callable
from typing import Optional

import numpy as np

from ..base._ea import EvolutionaryAlgorithm
from ..tools import donothing
from ..tools.operators import empty_crossover
from ..tools.operators import flip_mutation
from ..tools.operators import one_point_crossover
from ..tools.operators import proportional_selection
from ..tools.operators import rank_selection
from ..tools.operators import tournament_selection
from ..tools.operators import two_point_crossover
from ..tools.operators import uniform_crossover
from ..tools.operators import uniform_prop_crossover
from ..tools.operators import uniform_rank_crossover
from ..tools.operators import uniform_tour_crossover
from ..tools.random import binary_string_population
from ..tools.transformations import rank_data
from ..tools.transformations import scale_data


class GeneticAlgorithm(EvolutionaryAlgorithm):
    def __init__(self,
                 fitness_function: Callable,
                 iters: int,
                 pop_size: int,
                 str_len: int,
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
        self._selection_pool: dict
        self._crossover_pool: dict
        self._mutation_pool: dict
        self._initial_population: np.ndarray
        self._tour_size: int
        self._elitism: bool
        self._parents_num: int
        self._mutation_rate: float
        self._specified_selection: tuple
        self._specified_crossover: tuple
        self._specified_mutation: tuple

        self.set_strategy()

    def _update_pool(self):
        self._selection_pool = {'proportional': (proportional_selection, 0),
                                'rank': (rank_selection, 0),
                                'tournament_k': (tournament_selection, self._tour_size),
                                'tournament_3': (tournament_selection, 3),
                                'tournament_5': (tournament_selection, 5),
                                'tournament_7': (tournament_selection, 7)}

        self._crossover_pool = {'empty': (empty_crossover, 1),
                                'one_point': (one_point_crossover, 2),
                                'two_point': (two_point_crossover, 2),
                                'uniform2': (uniform_crossover, 2),
                                'uniform7': (uniform_crossover, 7),
                                'uniformk': (uniform_crossover, self._parents_num),
                                'uniform_prop2': (uniform_prop_crossover, 2),
                                'uniform_prop7': (uniform_prop_crossover, 7),
                                'uniform_propk': (uniform_prop_crossover, self._parents_num),
                                'uniform_rank2': (uniform_rank_crossover, 2),
                                'uniform_rank7': (uniform_rank_crossover, 7),
                                'uniform_rankk': (uniform_rank_crossover, self._parents_num),
                                'uniform_tour3': (uniform_tour_crossover, 3),
                                'uniform_tour7': (uniform_tour_crossover, 7),
                                'uniform_tourk': (uniform_tour_crossover, self._parents_num)}

        self._mutation_pool = {'weak': (flip_mutation, lambda: 1 / (3 * self._str_len)),
                               'average': (flip_mutation, lambda: 1 / (self._str_len)),
                               'strong': (flip_mutation, lambda: min(1, 3 / self._str_len)),
                               'custom_rate': (flip_mutation, self._mutation_rate)}

    def _get_new_individ_g(self,
                           population_g: np.ndarray,
                           fitness_scale: np.ndarray,
                           fitness_rank: np.ndarray,
                           *args) -> np.ndarray:
        selection_func, tour_size = self._specified_selection
        crossover_func, quantity = self._specified_crossover
        mutation_func, proba = self._specified_mutation
        if callable(proba):
            proba = proba()

        selected_id = selection_func(fitness_scale, fitness_rank,
                                     np.int64(tour_size),
                                     np.int64(quantity))

        offspring_no_mutated = crossover_func(population_g[selected_id],
                                              fitness_scale[selected_id],
                                              fitness_rank[selected_id])
        
        offspring = mutation_func(offspring_no_mutated, np.float64(proba))
        return offspring

    def set_strategy(self,
                     selection_oper: str = 'tournament_k',
                     crossover_oper: str = 'uniform2',
                     mutation_oper: str = 'weak',
                     tour_size_param: int = 2,
                     initial_population: Optional[np.ndarray] = None,
                     elitism_param: bool = True,
                     parents_num_param: int = 7,
                     mutation_rate_param: float = 0.05) -> None:
        '''
        - selection_oper: should be one of:
            'proportional', 'rank', 'tournament_k', 'tournament_3', 'tournament_5', 'tournament_7'
        - crossover oper: should be one of:
            'empty', 'one_point', 'two_point', 'uniform2', 'uniform7', 'uniformk', 'uniform_prop2',
            'uniform_prop7', 'uniform_propk', 'uniform_rank2', 'uniform_rank7', 'uniform_rankk',
            'uniform_tour3', 'uniform_tour7', 'uniform_tourk'
        - mutation oper: should be one of:
            'weak', 'average', 'strong', 'custom_rate'
        '''
        self._tour_size = tour_size_param
        self._initial_population = initial_population
        self._elitism = elitism_param
        self._parents_num = parents_num_param
        self._mutation_rate = mutation_rate_param

        self._update_pool()

        self._specified_selection = self._selection_pool[selection_oper]
        self._specified_crossover = self._crossover_pool[crossover_oper]
        self._specified_mutation = self._mutation_pool[mutation_oper]

    def fit(self):
        if self._initial_population is None:
            population_g = binary_string_population(
                self._pop_size, self._str_len)
        else:
            population_g = self._initial_population.copy()

        for i in range(self._iters):
            population_ph = self._get_phenotype(population_g)
            fitness = self._get_fitness(population_ph)

            self._update_fittest(population_g, population_ph, fitness)
            self._update_stats(population_g=population_g,
                               fitness_max=self._thefittest._fitness)

            if self._elitism:
                population_g[-1], population_ph[-1], fitness[-1] = self._thefittest.get().values()

            self._show_progress(i)
            if self._termitation_check():
                break
            else:
                get_new_individ_g = partial(self._get_new_individ_g,
                                            population_g,
                                            scale_data(fitness),
                                            rank_data(fitness))
                map_ = map(get_new_individ_g, range(self._pop_size))
                population_g = np.array(list(map_), dtype=np.byte)
        return self
