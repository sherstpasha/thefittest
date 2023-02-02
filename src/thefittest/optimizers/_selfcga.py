import numpy as np
from typing import Optional
from typing import Callable
from typing import Any
from ._geneticalgorithm import GeneticAlgorithm
from ..tools import scale_data
from ..tools import rank_data
from functools import partial
from ._base import TheFittest
from ._base import StaticticSelfCGA
from ._base import LastBest


class SelfCGA(GeneticAlgorithm):
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
        GeneticAlgorithm.__init__(self,
                                  fitness_function=fitness_function,
                                  genotype_to_phenotype=genotype_to_phenotype,
                                  iters=iters,
                                  pop_size=pop_size,
                                  str_len=str_len,
                                  optimal_value=optimal_value,
                                  termination_error_value=termination_error_value,
                                  no_increase_num=no_increase_num,
                                  minimization=minimization,
                                  show_progress_each=show_progress_each,
                                  keep_history=keep_history)

        self.K = 2
        self.threshold = 0.05
        self.set_strategy(select_opers=['proportional',
                                        'rank',
                                        'tournament'],
                          crossover_opers=['one_point',
                                           'two_point',
                                           'uniform2'],
                          mutation_opers=['weak',
                                          'average',
                                          'strong'])
        self.stats: StaticticSelfCGA

    def set_strategy(self, select_opers: Optional[list[str]] = None,
                     crossover_opers: Optional[list[str]] = None,
                     mutation_opers: Optional[list[str]] = None,
                     tour_size_param: Optional[int] = None,
                     initial_population: Optional[np.ndarray] = None):

        if select_opers is not None:
            s_sets = {}
            for operator_name in select_opers:
                value = self.s_pool[operator_name]
                s_sets[operator_name] = value
            self.s_sets = dict(sorted(s_sets.items()))

        if crossover_opers is not None:
            c_sets = {}
            for operator_name in crossover_opers:
                value = self.c_pool[operator_name]
                c_sets[operator_name] = value
            self.c_sets = dict(sorted(c_sets.items()))

        if mutation_opers is not None:
            m_sets = {}
            for operator_name in mutation_opers:
                value = self.m_pool[operator_name]
                m_sets[operator_name] = value
            self.m_sets = dict(sorted(m_sets.items()))
        if tour_size_param is not None:
            self.tour_size = tour_size_param
        self.initial_population = initial_population
        return self

    def create_offs(self, popuation, fitness, ranks,
                    selection, crossover, mutation):
        crossover_func, quantity = self.c_sets[crossover]
        selection_func, tour_size = self.s_sets[selection]
        mutation_func, proba = self.m_sets[mutation]
        indexes = selection_func(fitness, ranks,
                                 tour_size, quantity)
        parents = popuation[indexes].copy()
        fitness_p = fitness[indexes].copy()
        ranks_p = ranks[indexes].copy()
        offspring_no_mutated = crossover_func(parents, fitness_p, ranks_p)
        return mutation_func(offspring_no_mutated, proba)

    def choice_operators(self, operators, proba):
        return np.random.choice(list(operators), self.pop_size - 1, p=proba)

    def update_proba(self, proba, z, index):
        proba[index] += self.K/self.iters
        proba -= self.K/(z*self.iters)
        proba = proba.clip(self.threshold, 1)
        return proba/proba.sum()

    def find_fittest_operator(self, operators, fitness):
        operators_fitness = np.vstack([operators, fitness]).T
        argsort = np.argsort(operators_fitness[:, 0])
        operators_fitness = operators_fitness[argsort]

        keys, cut_index = np.unique(operators_fitness[:, 0], return_index=True)
        groups = np.split(operators_fitness[:, 1].astype(float), cut_index)[1:]
        mean_fit = np.array(list(map(np.mean, groups)))

        return keys[np.argmax(mean_fit)]

    def fit(self):

        z_s, z_c, z_m = list(map(len, (self.s_sets, self.c_sets, self.m_sets)))
        s_proba = np.full(z_s, 1/z_s)
        c_proba = np.full(z_c, 1/z_c)
        m_proba = np.full(z_m, 1/z_m)

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
            self.stats = StaticticSelfCGA().update(population_g,
                                                   population_ph,
                                                   fitness,
                                                   s_proba,
                                                   c_proba,
                                                   m_proba)

        for i in range(self.iters-1):
            self.show_progress(i)
            if self.termitation_check(lastbest.no_increase):
                break
            else:
                # print(s_proba, self.s_sets.keys())
                # print(c_proba, self.c_sets.keys())
                # print(m_proba, self.m_sets.keys())
                s_operators = self.choice_operators(
                    self.s_sets.keys(), s_proba)
                c_operators = self.choice_operators(
                    self.c_sets.keys(), c_proba)
                m_operators = self.choice_operators(
                    self.m_sets.keys(), m_proba)

                create_offs = partial(
                    self.create_offs, population_g.copy(),
                    fitness_scale.copy(), fitness_rank.copy())

                population_g[:-1] = np.array(list(map(create_offs, s_operators,
                                                  c_operators, m_operators)))
                population_ph[:-1] = self.genotype_to_phenotype(
                    population_g[:-1])
                fitness[:-1] = self.evaluate(population_ph[:-1])

                s_fittest_operator = self.find_fittest_operator(
                    s_operators, fitness[:-1])
                s_index = list(self.s_sets.keys()).index(s_fittest_operator)
                s_proba = self.update_proba(s_proba, z_s, s_index)

                c_fittest_operator = self.find_fittest_operator(
                    c_operators, fitness[:-1])
                c_index = list(self.c_sets.keys()).index(c_fittest_operator)
                c_proba = self.update_proba(c_proba, z_c, c_index)

                m_fittest_operator = self.find_fittest_operator(
                    m_operators, fitness[:-1])
                m_index = list(self.m_sets.keys()).index(m_fittest_operator)
                m_proba = self.update_proba(m_proba, z_m, m_index)

                population_g[-1], population_ph[-1], fitness[-1] = self.thefittest.get()
                fitness_scale = scale_data(fitness)
                fitness_rank = rank_data(fitness)

                self.thefittest.update(population_g, population_ph, fitness)
                lastbest.update(self.thefittest.fitness)
                if self.keep_history:
                    self.stats.update(population_g,
                                      population_ph,
                                      fitness,
                                      s_proba,
                                      c_proba,
                                      m_proba)
        return self
