import numpy as np
from typing import Optional
from typing import Callable
from typing import Any
from ._base import TheFittest
from ._base import Statistics
from ._base import EvolutionaryAlgorithm
from ._base import LastBest
from ._selections import proportional_selection
from ._selections import rank_selection
from ._selections import tournament_selection
from ._crossovers import empty_crossover
from ._crossovers import one_point_crossover
from ._crossovers import two_point_crossover
from ._crossovers import uniform_crossover
from ._crossovers import uniform_prop_crossover
from ._crossovers import uniform_rank_crossover
from ._crossovers import uniform_tour_crossover
from ._initializations import binary_string_population
from ._mutations import flip_mutation
from ..tools import scale_data
from ..tools import rank_data
from functools import partial


class GeneticAlgorithm(EvolutionaryAlgorithm):
    def __init__(self,
                 fitness_function: Callable[[np.ndarray[Any]], np.ndarray[float]],
                 genotype_to_phenotype: Callable[[np.ndarray[Any]], np.ndarray[Any]],
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

        self.str_len = str_len
        self.tour_size = 2
        self.initial_population = None
        self.thefittest: TheFittest
        self.stats: Statistics

        self.s_pool = {'proportional': (proportional_selection, None),
                       'rank': (rank_selection, None),
                       'tournament': (tournament_selection, self.tour_size)}

        self.c_pool = {'empty': (empty_crossover, 1),
                       'one_point': (one_point_crossover, 2),
                       'two_point': (two_point_crossover, 2),
                       'uniform2': (uniform_crossover, 2),
                       'uniform7': (uniform_crossover, 7),
                       'uniform_prop2': (uniform_prop_crossover, 2),
                       'uniform_prop7': (uniform_prop_crossover, 7),
                       'uniform_rank2': (uniform_rank_crossover, 2),
                       'uniform_rank7': (uniform_rank_crossover, 7),
                       'uniform_tour3': (uniform_tour_crossover, 3),
                       'uniform_tour7': (uniform_tour_crossover, 7)}

        self.m_pool = {'no': (flip_mutation, 0),
                       'weak':  (flip_mutation, 1/(3*self.str_len)),
                       'average':  (flip_mutation, 1/(self.str_len)),
                       'strong': (flip_mutation, min(1, 3/self.str_len))}

        self.s_set = self.s_pool['tournament']
        self.m_set = self.m_pool['weak']
        self.c_set = self.c_pool['uniform2']

    def generate_init_pop(self):
        if self.initial_population is None:
            population_g = binary_string_population(self.pop_size, self.str_len)
        else:
            population_g = self.initial_population
        return population_g

    def set_strategy(self,
                     selection_oper: Optional[str] = None,
                     crossover_oper: Optional[str] = None,
                     mutation_oper: Optional[str] = None,
                     tour_size_param: Optional[int] = None,
                     initial_population: Optional[np.ndarray] = None):
        if selection_oper is not None:
            self.s_set = self.s_pool[selection_oper]
        if crossover_oper is not None:
            self.c_set = self.c_pool[crossover_oper]
        if mutation_oper is not None:
            self.m_set = self.m_pool[mutation_oper]
        if tour_size_param is not None:
            self.tour_size = tour_size_param
        self.initial_population = initial_population
        return self

    def create_offspring(self, population_g, fitness_scale, fitness_rank, _):
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
        if self.keep_history:
            self.stats = Statistics().update(population_g,
                                             population_ph,
                                             fitness)

        for i in range(self.iters-1):
            self.show_progress(i)
            if self.termitation_check(lastbest.no_increase):
                break
            else:
                partial_create_offspring = partial(self.create_offspring,
                                                   population_g,
                                                   fitness_scale,
                                                   fitness_rank)
                map_ = map(partial_create_offspring, range(self.pop_size-1))
                population_g[:-1] = np.array(list(map_), dtype=np.byte)

                population_ph[:-1] = self.genotype_to_phenotype(
                    population_g[:-1])
                fitness[:-1] = self.evaluate(population_ph[:-1])

                population_g[-1], population_ph[-1], fitness[-1] = self.thefittest.get()
                fitness_scale = scale_data(fitness)
                fitness_rank = rank_data(fitness)

                self.thefittest.update(population_g, population_ph, fitness)
                lastbest.update(self.thefittest.fitness)
                if self.keep_history:
                    self.stats.update(population_g,
                                      population_ph,
                                      fitness)
        return self