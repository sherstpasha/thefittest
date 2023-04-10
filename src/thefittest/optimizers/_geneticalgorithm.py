import numpy as np
from functools import partial
from ..base import TheFittest
from ..base import EvolutionaryAlgorithm
from ..base import LastBest
from ..tools.operators import proportional_selection
from ..tools.operators import rank_selection
from ..tools.operators import tournament_selection
from ..tools.operators import empty_crossover
from ..tools.operators import one_point_crossover
from ..tools.operators import two_point_crossover
from ..tools.operators import uniform_crossover
from ..tools.operators import uniform_prop_crossover
from ..tools.operators import uniform_rank_crossover
from ..tools.operators import uniform_tour_crossover
from ..tools.operators import flip_mutation
from ..tools.generators import binary_string_population
from ..tools.transformations import scale_data
from ..tools.transformations import rank_data


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
               population_g_i,
               fitness_i):
        if self.mode == 'quick':
            self.fitness = np.append(self.fitness, np.max(fitness_i))
        elif self.mode == 'full':
            self.fitness = np.append(self.fitness, np.max(fitness_i))
            self.population_g = self.append_arr(self.population_g,
                                                population_g_i)
        else:
            raise ValueError('the "mode" must be either "quick" or "full"')
        return self


class GeneticAlgorithm(EvolutionaryAlgorithm):
    def __init__(self,
                 fitness_function,
                 genotype_to_phenotype,
                 iters,
                 pop_size,
                 str_len,
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

        self.str_len = str_len
        self.thefittest: TheFittest
        self.stats: Statistics
        self.s_pool: dict
        self.c_pool: dict
        self.m_pool: dict
        self.initial_population: np.ndarray
        self.tour_size: int
        self.elitism: bool
        self.parents_num: int
        self.mutation_rate: float
        self.s_set: tuple
        self.c_set: tuple
        self.m_set: tuple

        self.set_strategy()

    def generate_init_pop(self):
        if self.initial_population is None:
            population_g = binary_string_population(
                self.pop_size, self.str_len)
        else:
            population_g = self.initial_population
        return population_g

    def update_pool(self):
        self.s_pool = {'proportional': (proportional_selection, None),
                       'rank': (rank_selection, None),
                       'tournament_k': (tournament_selection, self.tour_size),
                       'tournament_3': (tournament_selection, 3),
                       'tournament_5': (tournament_selection, 5),
                       'tournament_7': (tournament_selection, 7)}

        self.c_pool = {'empty': (empty_crossover, 1),
                       'one_point': (one_point_crossover, 2),
                       'two_point': (two_point_crossover, 2),
                       'uniform2': (uniform_crossover, 2),
                       'uniform7': (uniform_crossover, 7),
                       'uniformk': (uniform_crossover, self.parents_num),
                       'uniform_prop2': (uniform_prop_crossover, 2),
                       'uniform_prop7': (uniform_prop_crossover, 7),
                       'uniform_propk': (uniform_prop_crossover, self.parents_num),
                       'uniform_rank2': (uniform_rank_crossover, 2),
                       'uniform_rank7': (uniform_rank_crossover, 7),
                       'uniform_rankk': (uniform_rank_crossover, self.parents_num),
                       'uniform_tour3': (uniform_tour_crossover, 3),
                       'uniform_tour7': (uniform_tour_crossover, 7),
                       'uniform_tourk': (uniform_tour_crossover, self.parents_num)}

        self.m_pool = {'weak':  (flip_mutation, 1/(3*self.str_len)),
                       'average':  (flip_mutation, 1/(self.str_len)),
                       'strong': (flip_mutation, min(1, 3/self.str_len)),
                       'custom_rate': (flip_mutation, self.mutation_rate)}

    def set_strategy(self,
                     selection_oper='tournament_k',
                     crossover_oper='uniform2',
                     mutation_oper='weak',
                     tour_size_param=2,
                     initial_population=None,
                     elitism_param=True,
                     parents_num_param=7,
                     mutation_rate_param=0.05):

        self.tour_size = tour_size_param
        self.initial_population = initial_population
        self.elitism = elitism_param
        self.parents_num = parents_num_param
        self.mutation_rate = mutation_rate_param

        self.update_pool()

        self.s_set = self.s_pool[selection_oper]
        self.c_set = self.c_pool[crossover_oper]
        self.m_set = self.m_pool[mutation_oper]

        return self

    def create_offspring(self, population_g, fitness_scale, fitness_rank):
        crossover_func, quantity = self.c_set
        selection_func, tour_size = self.s_set
        mutation_func, proba = self.m_set

        indexes = selection_func(fitness_scale,
                                 fitness_rank,
                                 tour_size,
                                 quantity)

        parents = population_g[indexes].copy()
        fitness_scale_p = fitness_scale[indexes].copy()
        fitness_rank_p = fitness_rank[indexes].copy()

        offspring_no_mutated = crossover_func(parents,
                                              fitness_scale_p,
                                              fitness_rank_p)

        mutant = mutation_func(offspring_no_mutated, proba)
        return mutant

    def fit(self):
        population_g = self.generate_init_pop()
        population_ph = self.genotype_to_phenotype(population_g)
        fitness = self.evaluate(population_ph)
        fitness_scale = scale_data(fitness)
        fitness_rank = rank_data(fitness)

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
                partial_create_offspring = partial(self.create_offspring,
                                                   population_g,
                                                   fitness_scale)
                map_ = map(partial_create_offspring,
                           fitness_scale, fitness_rank)
                population_g = np.array(list(map_), dtype=np.byte)

                population_ph = self.genotype_to_phenotype(population_g)
                fitness = self.evaluate(population_ph)

                if self.elitism:
                    population_g[-1], population_ph[-1], fitness[-1] = self.thefittest.get()
                fitness_scale = scale_data(fitness)
                fitness_rank = rank_data(fitness)

                self.thefittest.update(population_g, population_ph, fitness)
                lastbest.update(self.thefittest.fitness)
                if self.keep_history is not None:
                    self.stats.update(population_g,
                                      fitness)
        return self
