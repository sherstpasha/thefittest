import numpy as np
import numpy as np
from typing import Optional
from typing import Callable
from typing import Any
from ._base import TheFittest
from ._base import EvolutionaryAlgorithm
from ._base import LastBest
from ._base import UniversalSet
from ._selections import proportional_selection
from ._selections import rank_selection
from ._selections import tournament_selection
from ._crossovers import one_point_crossoverGP
from ._crossovers import standart_crossover
from ._crossovers import uniform_crossoverGP
from ._crossovers import empty_crossover
from ._mutations import point_mutation
from ._mutations import growing_mutation
from ._mutations import simplify_mutations
from ._initializations import half_and_half
from ..tools import scale_data
from ..tools import rank_data
from functools import partial


class StatisticsGP:
    def __init__(self, mode='quick'):
        self.mode = mode
        self.fittest = np.array([])
        self.fitness = np.array([])

    def update(self,
               fittest_i: np.ndarray,
               fitness_i: np.ndarray):
        if self.mode == 'quick':
            self.fitness = np.append(self.fitness, np.max(fitness_i))
        elif self.mode == 'full':
            self.fittest = np.append(self.fittest, fittest_i.copy())
            self.fitness = np.append(self.fitness, np.max(fitness_i))
        else:
            raise ValueError('the "mode" must be either "quick" or "full"')
        return self


class GeneticProgramming(EvolutionaryAlgorithm):
    '''Koza, John R.. “Genetic programming - on the programming of computers by means
    of natural selection.” Complex Adaptive Systems (1993)'''

    def __init__(self,
                 fitness_function: Callable[[np.ndarray[Any]], np.ndarray[float]],
                 genotype_to_phenotype: Callable[[np.ndarray[Any]], np.ndarray[Any]],
                 uniset: UniversalSet,
                 iters: int,
                 pop_size: int,
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

        self.uniset = uniset
        self.tour_size = 2
        self.max_level = 10
        self.initial_population = None
        self.thefittest: TheFittest
        self.stats: StatisticsGP

        self.s_pool = {'proportional': (proportional_selection, None),
                       'rank': (rank_selection, None),
                       'tournament': (tournament_selection, self.tour_size)}

        self.c_pool = {'empty': (empty_crossover, 1),
                       'uniform': (uniform_crossoverGP, 2),
                       'standart': (standart_crossover, 2),
                       'one_point': (one_point_crossoverGP, 2)
                       }

        self.m_pool = {'weak_point': (point_mutation, 0.25),
                       'average_point': (point_mutation, 1),
                       'strong_point': (point_mutation, 4),
                       'weak_grow': (growing_mutation, 0),
                       'average_grow': (growing_mutation, 1),
                       'strong_grow': (growing_mutation, 4),
                       'weak_simplify': (simplify_mutations, 0.25),
                       'average_simplify': (simplify_mutations, 1),
                       'strong_simplify': (simplify_mutations, 4)}

        self.s_set = self.s_pool['tournament']
        self.c_set = self.c_pool['standart']
        self.m_set = self.m_pool['weak_grow']

# добавить сюда max_level
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

        parents = population_g[indexes]
        fitness_scale_p = fitness_scale[indexes]
        fitness_rank_p = fitness_rank[indexes]

        offspring_no_mutated = crossover_func(parents,
                                              fitness_scale_p,
                                              fitness_rank_p,
                                              self.max_level)
        mutant = mutation_func(offspring_no_mutated,
                               self.uniset, proba, self.max_level)
        return mutant

    def fit(self):
        population_g = half_and_half(
            self.pop_size, self.uniset, self.max_level)
        population_ph = self.genotype_to_phenotype(population_g)
        fitness = self.evaluate(population_ph)
        fitness_scale = scale_data(fitness)
        fitness_rank = rank_data(fitness)

        self.thefittest = TheFittest().update(population_g,
                                              population_ph,
                                              fitness)
        lastbest = LastBest().update(self.thefittest.fitness)
        if self.keep_history is not None:
            self.stats = StatisticsGP(
                mode=self.keep_history).update(self.thefittest.genotype,
                                               fitness)
        for i in range(self.iters-1):
            self.show_progress(i)
            levels = [tree.get_max_level() for tree in population_g]
            print('levels', np.max(levels), np.mean(levels))
            print('fitness', np.max(fitness), np.mean(fitness))
            if self.termitation_check(lastbest.no_increase):
                break
            else:
                partial_create_offspring = partial(self.create_offspring,
                                                   population_g,
                                                   fitness_scale,
                                                   fitness_rank)
                map_ = map(partial_create_offspring, range(self.pop_size-1))
                population_g[:-1] = np.array(list(map_), dtype=object)
                population_ph[:-1] = self.genotype_to_phenotype(
                    population_g[:-1])
                fitness[:-1] = self.evaluate(population_ph[:-1])

                population_g[-1], population_ph[-1], fitness[-1] = self.thefittest.get()
                fitness_scale = scale_data(fitness)
                fitness_rank = rank_data(fitness)

                self.thefittest.update(population_g, population_ph, fitness)
                lastbest.update(self.thefittest.fitness)
                if self.keep_history is not None:
                    self.stats.update(self.thefittest.genotype,
                                      fitness)
        return self
