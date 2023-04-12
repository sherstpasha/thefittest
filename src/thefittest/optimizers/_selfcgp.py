import numpy as np
from ._geneticprogramming import GeneticProgramming
from ..tools.transformations import scale_data
from ..tools.transformations import rank_data
from functools import partial
from ..base import TheFittest
from ..base import LastBest
from ..base import Statistics
from ..tools.transformations import numpy_group_by
from ..tools.generators import half_and_half


class SelfCGP(GeneticProgramming):
    '''Semenkin, Eugene & Semenkina, Maria. (2012). Self-configuring genetic programming algorithm
    with modified uniform crossover. 1-6. 10.1109/CEC.2012.6256587. '''

    def __init__(self,
                 fitness_function,
                 genotype_to_phenotype,
                 uniset,
                 iters,
                 pop_size,
                 optimal_value=None,
                 termination_error_value=0,
                 no_increase_num=None,
                 minimization=False,
                 show_progress_each=None,
                 keep_history=False):
        super().__init__(fitness_function,
                         genotype_to_phenotype,
                         uniset, iters,
                         pop_size,
                         optimal_value,
                         termination_error_value,
                         no_increase_num,
                         minimization,
                         show_progress_each,
                         keep_history)

        self.thefittest: TheFittest
        self.stats: Statistics
        self.s_sets: dict
        self.c_sets: dict
        self.m_sets: dict
        self.K: int
        self.threshold: float
        self.s_set: dict
        self.c_set: dict
        self.m_set: dict

        self.set_strategy()

    def set_strategy(self, select_opers=['proportional',
                                         'rank',
                                         'tournament_3',
                                         'tournament_5',
                                         'tournament_7'],
                     crossover_opers=['standart',
                                      'one_point',
                                      'uniform_rank2'],
                     mutation_opers=['weak_point',
                                     'average_point',
                                     'strong_point',
                                     'weak_grow',
                                     'average_grow',
                                     'strong_grow'],
                     tour_size_param=2,
                     initial_population=None,
                     elitism_param=True,
                     parents_num_param=3,
                     mutation_rate_param=0.05,
                     K_param=2,
                     threshold_param=0.1,
                     max_level_param=16,
                     init_level_param=5):
        self.tour_size = tour_size_param
        self.initial_population = initial_population
        self.elitism = elitism_param
        self.parents_num = parents_num_param
        self.mutation_rate = mutation_rate_param
        self.K = K_param
        self.threshold = threshold_param
        self.max_level = max_level_param
        self.init_level = init_level_param

        self.update_pool()

        s_sets = {}
        for operator_name in select_opers:
            value = self.s_pool[operator_name]
            s_sets[operator_name] = value
        self.s_sets = dict(sorted(s_sets.items()))

        c_sets = {}
        for operator_name in crossover_opers:
            value = self.c_pool[operator_name]
            c_sets[operator_name] = value
        self.c_sets = dict(sorted(c_sets.items()))

        m_sets = {}
        for operator_name in mutation_opers:
            value = self.m_pool[operator_name]
            m_sets[operator_name] = value
        self.m_sets = dict(sorted(m_sets.items()))

        return self

    def create_offspring(self, population_g, fitness_scale, fitness_rank,
                         selection, crossover, mutation):
        crossover_func, quantity = self.c_sets[crossover]
        selection_func, tour_size = self.s_sets[selection]
        mutation_func, proba_up, not_scale = self.m_sets[mutation]

        indexes = selection_func(fitness_scale,
                                 fitness_rank,
                                 tour_size,
                                 quantity)

        parents = population_g[indexes]
        fitness_scale_p = fitness_scale[indexes].copy()
        fitness_rank_p = fitness_rank[indexes].copy()

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

    def choice_operators(self, proba_dict):
        operators = list(proba_dict.keys())
        proba = list(proba_dict.values())
        return np.random.choice(list(operators), self.pop_size, p=proba)

    def update_proba(self, proba, operator):
        proba[operator] += self.K/self.iters
        proba_value = np.array(list(proba.values()))
        proba_value -= self.K/(len(proba)*self.iters)
        proba_value = proba_value.clip(self.threshold, 1)
        proba_value = proba_value/proba_value.sum()
        return dict(zip(proba.keys(), proba_value))

    def find_fittest_operator(self, operators, fitness):
        keys, groups = numpy_group_by(group=fitness, by=operators)
        mean_fit = np.array(list(map(np.mean, groups)))
        return keys[np.argmax(mean_fit)]

    def fit(self):
        z_s, z_c, z_m = list(map(len, (self.s_sets, self.c_sets, self.m_sets)))
        s_proba = dict(zip(list(self.s_sets.keys()), np.full(z_s, 1/z_s)))
        m_proba = dict(zip(list(self.m_sets.keys()), np.full(z_m, 1/z_m)))
        if 'empty' in self.c_sets.keys():
            c_proba = dict(zip(list(self.c_sets.keys()),
                           np.full(z_c, 0.9/(z_c-1))))
            c_proba['empty'] = 0.1
        else:
            c_proba = dict(zip(list(self.c_sets.keys()), np.full(z_c, 1/z_c)))

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
                mode=self.keep_history).update({'individ_max': self.thefittest.genotype.copy(),
                                                'fitness_max': self.thefittest.fitness,
                                                's_proba': s_proba.copy(),
                                                'c_proba': c_proba.copy(),
                                                'm_proba': m_proba.copy()})
        for i in range(self.iters-1):
            self.show_progress(i)
            if self.termitation_check(lastbest.no_increase):
                break
            else:
                s_operators = self.choice_operators(s_proba)
                c_operators = self.choice_operators(c_proba)
                m_operators = self.choice_operators(m_proba)

                partial_create_offspring = partial(self.create_offspring,
                                                   population_g,
                                                   fitness_scale,
                                                   fitness_rank)
                map_ = map(partial_create_offspring,
                           s_operators,
                           c_operators,
                           m_operators)
                population_g = np.array(list(map_), dtype=object)
                population_ph = self.genotype_to_phenotype(
                    population_g)
                fitness = self.evaluate(population_ph)

                if self.elitism:
                    population_g[-1], population_ph[-1], fitness[-1] = self.thefittest.get()
                fitness_scale = scale_data(fitness)
                fitness_rank = rank_data(fitness)

                s_fittest_oper = self.find_fittest_operator(
                    s_operators, fitness_scale)
                s_proba = self.update_proba(s_proba, s_fittest_oper)

                c_fittest_oper = self.find_fittest_operator(
                    c_operators, fitness_scale)
                c_proba = self.update_proba(c_proba, c_fittest_oper)

                m_fittest_oper = self.find_fittest_operator(
                    m_operators, fitness_scale)
                m_proba = self.update_proba(m_proba, m_fittest_oper)

                self.thefittest.update(population_g, population_ph, fitness)
                lastbest.update(self.thefittest.fitness)

                if self.keep_history:
                    self.stats.update({'individ_max': self.thefittest.genotype.copy(),
                                       'fitness_max': self.thefittest.fitness,
                                       's_proba': s_proba.copy(),
                                       'c_proba': c_proba.copy(),
                                       'm_proba': m_proba.copy()})
        return self
