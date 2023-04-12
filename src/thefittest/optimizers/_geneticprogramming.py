import numpy as np
import numpy as np
from ..base import TheFittest
from ..base import EvolutionaryAlgorithm
from ..base import LastBest
from ..base import Statistics
from ..tools.operators import proportional_selection
from ..tools.operators import rank_selection
from ..tools.operators import tournament_selection
from ..tools.operators import one_point_crossoverGP
from ..tools.operators import standart_crossover
from ..tools.operators import uniform_crossoverGP
from ..tools.operators import uniform_crossoverGP_prop
from ..tools.operators import uniform_crossoverGP_rank
from ..tools.operators import uniform_crossoverGP_tour
from ..tools.operators import empty_crossover
from ..tools.operators import point_mutation
from ..tools.operators import growing_mutation
from ..tools.operators import swap_mutation
from ..tools.operators import shrink_mutation
from ..tools.generators import half_and_half
from ..tools.transformations import scale_data
from ..tools.transformations import rank_data
from functools import partial


class GeneticProgramming(EvolutionaryAlgorithm):
    '''Koza, John R.. “Genetic programming - on the programming of computers by means
    of natural selection.” Complex Adaptive Systems (1993)'''

    def __init__(self,
                 fitness_function,
                 genotype_to_phenotype,
                 uniset,
                 iters,
                 pop_size,
                 optimal_value=None,
                 termination_error_value=0.,
                 no_increase_num=None,
                 minimization=False,
                 show_progress_each=None,
                 keep_history=None):

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

        self.uniset = uniset
        self.thefittest: TheFittest
        self.stats: Statistics
        self.s_pool: dict
        self.c_pool: dict
        self.m_pool: dict
        self.tour_size: int
        self.max_level: int
        self.init_level: int
        self.initial_population: np.ndarray
        self.s_set: tuple
        self.c_set: tuple
        self.m_set: tuple

        self.set_strategy()

    def update_pool(self):
        self.s_pool = {'proportional': (proportional_selection, None),
                       'rank': (rank_selection, None),
                       'tournament_k': (tournament_selection, self.tour_size),
                       'tournament_3': (tournament_selection, 3),
                       'tournament_5': (tournament_selection, 5),
                       'tournament_7': (tournament_selection, 7)}

        self.c_pool = {'empty': (empty_crossover, 1),
                       'standart': (standart_crossover, 2),
                       'one_point': (one_point_crossoverGP, 2),
                       'uniform2': (uniform_crossoverGP, 2),
                       'uniform7': (uniform_crossoverGP, 7),
                       'uniformk': (uniform_crossoverGP, self.parents_num),
                       'uniform_prop2': (uniform_crossoverGP_prop, 2),
                       'uniform_prop7': (uniform_crossoverGP_prop, 7),
                       'uniform_propk': (uniform_crossoverGP_prop, self.parents_num),
                       'uniform_rank2': (uniform_crossoverGP_rank, 2),
                       'uniform_rank7': (uniform_crossoverGP_rank, 7),
                       'uniform_rankk': (uniform_crossoverGP_rank, self.parents_num),
                       'uniform_tour3': (uniform_crossoverGP_tour, 3),
                       'uniform_tour7': (uniform_crossoverGP_tour, 7),
                       'uniform_tourk': (uniform_crossoverGP_tour, self.parents_num)}

        self.m_pool = {'weak_point': (point_mutation, 0.25, False),
                       'average_point': (point_mutation, 1, False),
                       'strong_point': (point_mutation, 4, False),
                       'custom_rate_point': (point_mutation, self.mutation_rate, True),
                       'weak_grow': (growing_mutation, 0.25, False),
                       'average_grow': (growing_mutation, 1, False),
                       'strong_grow': (growing_mutation, 4, False),
                       'custom_rate_grow': (growing_mutation, self.mutation_rate, True),
                       'weak_swap': (swap_mutation, 0.25, False),
                       'average_swap': (swap_mutation, 1, False),
                       'strong_swap': (swap_mutation, 4, False),
                       'custom_rate_swap': (swap_mutation, self.mutation_rate, True),
                       'weak_shrink': (shrink_mutation, 0.25, False),
                       'average_shrink': (shrink_mutation, 1, False),
                       'strong_shrink': (shrink_mutation, 4, False),
                       'custom_rate_shrink': (shrink_mutation, self.mutation_rate, True)}

    def set_strategy(self,
                     selection_oper='rank',
                     crossover_oper='standart',
                     mutation_oper='weak_grow',
                     tour_size_param=2,
                     initial_population=None,
                     max_level_param=16,
                     init_level_param=5,
                     elitism_param=True,
                     parents_num_param=7,
                     mutation_rate_param=0.05):
        self.tour_size = tour_size_param
        self.initial_population = initial_population
        self.max_level = max_level_param
        self.init_level = init_level_param
        self.elitism = elitism_param
        self.parents_num = parents_num_param
        self.mutation_rate = mutation_rate_param

        self.update_pool()

        self.s_set = self.s_pool[selection_oper]
        self.c_set = self.c_pool[crossover_oper]
        self.m_set = self.m_pool[mutation_oper]

        return self

    def create_offspring(self, population_g, fitness_scale, fitness_rank, _):
        crossover_func, quantity = self.c_set
        selection_func, tour_size = self.s_set
        mutation_func, proba_up, not_scale = self.m_set

        indexes = selection_func(fitness_scale,
                                 fitness_rank,
                                 tour_size,
                                 quantity)

        parents = population_g[indexes]
        fitness_scale_p = fitness_scale[indexes]
        fitness_rank_p = fitness_rank[indexes]

        offspring_no_mutated = crossover_func(parents,
                                              fitness_scale_p,
                                              fitness_rank_p,
                                              self.max_level)

        if not_scale:
            proba = proba_up
        else:
            proba = proba_up/len(offspring_no_mutated)

        mutant = mutation_func(offspring_no_mutated,
                               self.uniset, proba, self.max_level)
        return mutant

    def fit(self):
        population_g = half_and_half(
            self.pop_size, self.uniset, self.init_level)
        population_ph = self.genotype_to_phenotype(population_g)
        fitness = self.evaluate(population_ph)
        fitness_scale = scale_data(fitness)
        fitness_rank = rank_data(fitness)

        self.thefittest = TheFittest().update(population_g,
                                              population_ph,
                                              fitness)
        lastbest = LastBest().update(self.thefittest.fitness)
        if self.keep_history:
            self.stats = Statistics(
                mode=self.keep_history).update(
                {'individ_max': self.thefittest.genotype.copy(),
                 'fitness_max': self.thefittest.fitness})
        for i in range(self.iters-1):
            self.show_progress(i)
            if self.termitation_check(lastbest.no_increase):
                break
            else:
                partial_create_offspring = partial(self.create_offspring,
                                                   population_g,
                                                   fitness_scale,
                                                   fitness_rank)
                map_ = map(partial_create_offspring, range(self.pop_size))
                population_g = np.array(list(map_), dtype=object)
                population_ph = self.genotype_to_phenotype(population_g)
                fitness = self.evaluate(population_ph)

                if self.elitism:
                    population_g[-1], population_ph[-1], fitness[-1] = self.thefittest.get()
                fitness_scale = scale_data(fitness)
                fitness_rank = rank_data(fitness)
                self.thefittest.update(population_g, population_ph, fitness)
                lastbest.update(self.thefittest.fitness)
                if self.keep_history:
                    self.stats.update(
                        {'individ_max': self.thefittest.genotype.copy(),
                         'fitness_max': self.thefittest.fitness})
        return self
