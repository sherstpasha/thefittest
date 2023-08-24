from functools import partial
from typing import Callable
from typing import Optional

import numpy as np

from ..base import Tree
from ..base import UniversalSet
from ..base._ea import EvolutionaryAlgorithm
from ..tools import donothing
from ..tools.operators import empty_crossover
from ..tools.operators import growing_mutation
from ..tools.operators import one_point_crossoverGP
from ..tools.operators import point_mutation
from ..tools.operators import proportional_selection
from ..tools.operators import rank_selection
from ..tools.operators import shrink_mutation
from ..tools.operators import standart_crossover
from ..tools.operators import swap_mutation
from ..tools.operators import tournament_selection
from ..tools.operators import uniform_crossoverGP
from ..tools.operators import uniform_crossoverGP_prop
from ..tools.operators import uniform_crossoverGP_rank
from ..tools.operators import uniform_crossoverGP_tour
from ..tools.random import half_and_half
from ..tools.transformations import rank_data
from ..tools.transformations import scale_data


class GeneticProgramming(EvolutionaryAlgorithm):
    '''Koza, John R.. “Genetic programming - on the programming of computers by means
    of natural selection.” Complex Adaptive Systems (1993)'''

    def __init__(self,
                 fitness_function: Callable,
                 uniset: UniversalSet,
                 iters: int,
                 pop_size: int,
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

        self._uniset = uniset
        self._selection_pool: dict
        self._crossover_pool: dict
        self._mutation_pool: dict
        self._tour_size: int
        self._max_level: int
        self._init_level: int
        self._initial_population: np.ndarray
        self._specified_selection: tuple
        self._specified_crossover: tuple
        self._specified_mutation: tuple

        self.set_strategy()

    def _update_pool(self):
        self._selection_pool = {
            'proportional': (proportional_selection, 0),
            'rank': (rank_selection, 0),
            'tournament_k': (tournament_selection, self._tour_size),
            'tournament_3': (tournament_selection, 3),
            'tournament_5': (tournament_selection, 5),
            'tournament_7': (tournament_selection, 7)}

        self._crossover_pool = {
            'empty': (empty_crossover, 1),
            'standart': (standart_crossover, 2),
            'one_point': (one_point_crossoverGP, 2),
            'uniform2': (uniform_crossoverGP, 2),
            'uniform7': (uniform_crossoverGP, 7),
            'uniformk': (uniform_crossoverGP, self._parents_num),
            'uniform_prop2': (uniform_crossoverGP_prop, 2),
            'uniform_prop7': (uniform_crossoverGP_prop, 7),
            'uniform_propk': (uniform_crossoverGP_prop, self._parents_num),
            'uniform_rank2': (uniform_crossoverGP_rank, 2),
            'uniform_rank7': (uniform_crossoverGP_rank, 7),
            'uniform_rankk': (uniform_crossoverGP_rank, self._parents_num),
            'uniform_tour3': (uniform_crossoverGP_tour, 3),
            'uniform_tour7': (uniform_crossoverGP_tour, 7),
            'uniform_tourk': (uniform_crossoverGP_tour, self._parents_num)}

        self._mutation_pool = {
            'weak_point': (point_mutation, 0.25, True),
            'average_point': (point_mutation, 1, True),
            'strong_point': (point_mutation, 4, True),
            'custom_rate_point': (point_mutation, self._mutation_rate, False),
            'weak_grow': (growing_mutation, 0.25, True),
            'average_grow': (growing_mutation, 1, True),
            'strong_grow': (growing_mutation, 4, True),
            'custom_rate_grow': (growing_mutation, self._mutation_rate, False),
            'weak_swap': (swap_mutation, 0.25, True),
            'average_swap': (swap_mutation, 1, True),
            'strong_swap': (swap_mutation, 4, True),
            'custom_rate_swap': (swap_mutation, self._mutation_rate, False),
            'weak_shrink': (shrink_mutation, 0.25, True),
            'average_shrink': (shrink_mutation, 1, True),
            'strong_shrink': (shrink_mutation, 4, True),
            'custom_rate_shrink': (shrink_mutation, self._mutation_rate, False)}

    def _get_new_individ_g(self,
                           population_g: np.ndarray,
                           fitness_scale: np.ndarray,
                           fitness_rank: np.ndarray,
                           *args) -> Tree:
        selection_func, tour_size = self._specified_selection
        crossover_func, quantity = self._specified_crossover
        mutation_func, proba_up, scale = self._specified_mutation

        selected_id = selection_func(fitness_scale,
                                     fitness_rank,
                                     np.int64(tour_size),
                                     np.int64(quantity))

        offspring_no_mutated = crossover_func(population_g[selected_id],
                                              fitness_scale[selected_id],
                                              fitness_rank[selected_id],
                                              self._max_level)

        proba = proba_up / len(offspring_no_mutated) if scale else proba_up

        offspring = mutation_func(offspring_no_mutated, self._uniset,
                                  proba, self._max_level)
        return offspring

    def set_strategy(self,
                     selection_oper: str = 'rank',
                     crossover_oper: str = 'standart',
                     mutation_oper: str = 'weak_grow',
                     tour_size_param: int = 2,
                     initial_population: Optional[np.ndarray] = None,
                     max_level_param: int = 16,
                     init_level_param: int = 5,
                     elitism_param: bool = True,
                     parents_num_param: int = 7,
                     mutation_rate_param: float = 0.05) -> None:
        '''
        - selection_oper: should be one of:
            'proportional', 'rank', 'tournament_k', 'tournament_3', 'tournament_5', 'tournament_7'
        - crossover oper: should be one of:
            'empty', 'standart', 'one_point', 'uniform2', 'uniform7', 'uniformk', 'uniform_prop2',
            'uniform_prop7', 'uniform_propk', 'uniform_rank2', 'uniform_rank7', 'uniform_rankk',
            'uniform_tour3', 'uniform_tour7', 'uniform_tourk'
        - mutation oper: should be one of:
            'weak_point', 'average_point', 'strong_point', 'custom_rate_point',
            'weak_grow', 'average_grow', 'strong_grow', 'custom_rate_grow',
            'weak_swap', 'average_swap', 'strong_swap', 'custom_rate_swap',
            'weak_shrink', 'average_shrink', 'strong_shrink', 'custom_rate_shrink',
        '''

        self._tour_size = tour_size_param
        self._initial_population = initial_population
        self._max_level = max_level_param
        self._init_level = init_level_param
        self._elitism = elitism_param
        self._parents_num = parents_num_param
        self._mutation_rate = mutation_rate_param

        self._update_pool()

        self._specified_selection = self._selection_pool[selection_oper]
        self._specified_crossover = self._crossover_pool[crossover_oper]
        self._specified_mutation = self._mutation_pool[mutation_oper]

    def fit(self):
        if self._initial_population is None:
            population_g = half_and_half(
                self._pop_size, self._uniset, self._init_level)
        else:
            population_g = self._initial_population

        for i in range(self._iters - 1):
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
                population_g = np.array(list(map_), dtype=object)

        return self
