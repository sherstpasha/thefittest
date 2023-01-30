import numpy as np
from functools import partial
from ..tools import scale_data
from ..tools import rank_data
from ._units import TheFittest
from ._units import Statictic
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
from ._mutations import flip_mutation
import copy


class SelfCGA:

    def __init__(self, fitness_function, genotype_to_phenotype,
                 iters, pop_size, str_len, tour_size=2, K=2.,
                 threshold=0.05,
                 optimal_value=None,
                 termination_error_value=0.,
                 no_increase_num=None,
                 minimization=False,
                 show_progress_each=None,
                 keep_history=False):
        self.fitness_function = fitness_function
        self.genotype_to_phenotype = genotype_to_phenotype
        self.iters = iters
        self.pop_size = pop_size
        self.str_len = str_len
        self.K = K
        self.threshold = threshold
        self.optimal_value = optimal_value
        self.termination_error_value = termination_error_value
        self.no_increase_num = no_increase_num
        self.minimization = minimization
        self.show_progress_each = show_progress_each
        self.keep_history = keep_history
        self.remains: int = 0
        self.calls: int = 0

        self.s_pool = {'proportional': (proportional_selection, 0),
                       'rank': (rank_selection, 0),
                       'tournament': (tournament_selection, tour_size)}

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

        self.thefittest: TheFittest
        self.stats: Statictic
        self.s_sets: dict
        self.m_sets = dict
        self.c_sets = dict

        self.operators_selector(select_opers=[
            'proportional',
            'rank',
            'tournament'],
            crossover_opers=['one_point',
                             'two_point',
                             'uniform2'],
            mutation_opers=['weak',
                            'average',
                            'strong'])

    def evaluate(self, population_ph):
        if self.minimization:
            return -self.fitness_function(population_ph)
        else:
            return self.fitness_function(population_ph)

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

    def find_fittest_operator(self, operators, fitness):
        operators_fitness = np.vstack([operators, fitness]).T
        argsort = np.argsort(operators_fitness[:, 0])
        operators_fitness = operators_fitness[argsort]

        keys, cut_index = np.unique(operators_fitness[:, 0], return_index=True)
        groups = np.split(operators_fitness[:, 1].astype(float), cut_index)[1:]
        mean_fit = np.array(list(map(np.mean, groups)))

        return keys[np.argmax(mean_fit)]

    def update_proba(self, proba, z, index):
        proba[index] += self.K/self.iters
        proba -= self.K/(z*self.iters)
        proba = proba.clip(self.threshold, 1)
        return proba/proba.sum()

    def operators_selector(self, select_opers=None,
                           crossover_opers=None,
                           mutation_opers=None):

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

    def fit(self, initial_population=None):
        calls = 0
        no_increase = 0

        z_s, z_c, z_m = list(map(len, (self.s_sets, self.c_sets, self.m_sets)))

        s_proba = np.full(z_s, 1/z_s)
        c_proba = np.full(z_c, 1/z_c)
        m_proba = np.full(z_m, 1/z_m)

        if initial_population is None:
            population_g = np.random.randint(low=2, size=(self.pop_size,
                                                          self.str_len),
                                             dtype=np.byte)
        else:
            population_g = initial_population
        population_ph = self.genotype_to_phenotype(population_g)
        fitness = self.evaluate(population_ph)
        calls += len(population_ph)
        fitness_scale = scale_data(fitness)
        fitness_rank = rank_data(fitness)

        self.thefittest = TheFittest(
            genotype=population_g[np.argmax(fitness)].copy(),
            phenotype=population_ph[np.argmax(fitness)].copy(),
            fitness=fitness[np.argmax(fitness)].copy())
        last_best = fitness[np.argmax(fitness)].copy()
        if self.keep_history:
            self.stats = Statictic().update(population_g, fitness,
                                            s_proba, c_proba, m_proba)
        else:
            self.stats = None

        for i in range(self.iters-1):

            if self.show_progress_each is not None:
                if i % self.show_progress_each == 0:
                    print(f'{i} iterstion with fitness = {self.thefittest.fitness}')
            if self.optimal_value is not None:
                if self.minimization:
                    aim = -(self.optimal_value) - self.termination_error_value
                else:
                    aim = self.optimal_value - self.termination_error_value

                find_opt = self.thefittest.fitness >= aim

            else:
                find_opt = False

            no_increase_cond = no_increase == self.no_increase_num

            if find_opt or no_increase_cond:
                break

            s_operators = self.choice_operators(self.s_sets.keys(), s_proba)
            c_operators = self.choice_operators(self.c_sets.keys(), c_proba)
            m_operators = self.choice_operators(self.m_sets.keys(), m_proba)

            create_offs = partial(
                self.create_offs, population_g.copy(),
                fitness_scale.copy(), fitness_rank.copy())

            population_g[:-1] = np.array(list(map(create_offs, s_operators,
                                                  c_operators, m_operators)))
            population_ph[:-1] = self.genotype_to_phenotype(population_g[:-1])

            fitness[:-1] = self.evaluate(population_ph[:-1])
            calls += len(population_ph[:-1])

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

            temp_best = self.thefittest.fitness.copy()
            if last_best == temp_best:
                no_increase += 1
            else:
                last_best = temp_best

            if self.keep_history:
                self.stats = self.stats.update(population_g, fitness,
                                               s_proba, c_proba, m_proba)

        self.remains = (self.pop_size + (self.iters-1)
                        * (self.pop_size-1)) - calls
        self.calls = calls
        if self.minimization:
            self.thefittest.fitness = -self.thefittest.fitness
        return self
