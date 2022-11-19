import numpy as np
from functools import partial


class TheFittest:
    def __init__(self, genotype, phenotype, fitness):
        self.genotype = genotype
        self.phenotype = phenotype
        self.fitness = fitness

    def update(self, population_g, population_ph, fitness):
        temp_best_id = np.argmax(fitness)
        temp_best_fitness = fitness[temp_best_id].copy()
        if temp_best_fitness > self.fitness:
            self.genotype = population_g[temp_best_id].copy()
            self.phenotype = population_ph[temp_best_id].copy()
            self.fitness = temp_best_fitness

        return self

    def __str__(self):
        return 'genotype: ' + str(self.genotype) + '\n' + \
            'phenotype: ' + str(self.phenotype) + '\n' + \
            'fitness: ' + str(self.fitness)

    def get(self):
        return self.genotype.copy(), self.phenotype.copy(), self.fitness.copy()


class Statictic:
    def __init__(self):
        self.fitness = np.array([], dtype=float)
        self.s_proba = np.array([], dtype=float)
        self.c_proba = np.array([], dtype=float)
        self.m_proba = np.array([], dtype=float)

    def update(self, fitness_i, s_proba_i, c_proba_i, m_proba_i):
        self.fitness = np.append(self.fitness, np.max(fitness_i))
        self.s_proba = np.vstack([self.s_proba.reshape(-1, len(s_proba_i)),
                                  s_proba_i.copy()])
        self.c_proba = np.vstack([self.c_proba.reshape(-1, len(c_proba_i)),
                                  c_proba_i.copy()])
        self.m_proba = np.vstack([self.m_proba.reshape(-1, len(m_proba_i)),
                                  m_proba_i.copy()])
        return self


def rank_scale_data(arr):
    arr = arr.copy()
    raw_ranks = np.zeros(shape=(arr.shape[0]))
    argsort = np.argsort(arr)

    s = (arr[:, np.newaxis] == arr).astype(int)
    raw_ranks[argsort] = np.arange(arr.shape[0]) + 1
    ranks = np.sum(raw_ranks*s, axis=1)/s.sum(axis=0)
    return ranks/ranks.sum()


def scale_data(arr):
    arr = arr.copy()
    max_ = arr.max()
    min_ = arr.min()
    if max_ == min_:
        arr_n = np.ones_like(arr)
    else:
        arr_n = (arr - min_)/(max_ - min_)
    return arr_n


class SamplingGrid:

    def __init__(self, left, right, parts):
        self.left = left
        self.right = right
        self.parts = parts
        self.h = np.abs(left - right)/(2.0**parts - 1)

    def decoder(self, population_parts, left_i, h_i):
        ipp = population_parts.astype(int)
        int_convert = np.sum(ipp*(2**np.arange(ipp.shape[1],
                                               dtype=int)),
                             axis=1)
        return left_i + h_i*int_convert

    def transform(self, population):
        splits = np.add.accumulate(self.parts)
        p_parts = np.split(population, splits[:-1], axis=1)
        fpp = [self.decoder(p_parts_i, left_i, h_i)
               for p_parts_i, left_i, h_i in zip(p_parts,
                                                 self.left,
                                                 self.h)]
        return np.vstack(fpp).T


class SelfCGeneticAlgorithm:

    def __init__(self, fitness_function, genotype_to_phenotype, iters,
                 pop_size, str_len, K = 2.,
                 threshold = 0.05,
                 optimal_value = None,
                 no_increase_num = None):
        self.fitness_function = fitness_function
        self.genotype_to_phenotype = genotype_to_phenotype
        self.iters = iters
        self.pop_size = pop_size
        self.str_len = str_len
        self.K = K
        self.threshold = threshold
        self.optimal_value = optimal_value
        self.no_increase_num = no_increase_num
        self.remains:int = 0

        self.s_sets = {'proportional': (self.proportional_selection, 0),
                       'rank': (self.rank_selection, 0),
                       'tournament3': (self.tournament_selection, 3),
                       'tournament5': (self.tournament_selection, 5),
                       'tournament7': (self.tournament_selection, 7)}
        self.s_sets = dict(sorted(self.s_sets.items()))

        self.m_sets = {'weak': 1/(3*self.str_len),
                       'average':  1/(self.str_len),
                       'strong': min(1, 3/self.str_len)}
        self.m_sets = dict(sorted(self.m_sets.items()))

        self.c_sets = {'one_point': (self.one_point_crossover, 2),
                       'two_point': (self.two_point_crossover, 2),
                       'uniform2': (self.uniform_crossover, 2),
                       'uniform7': (self.uniform_crossover, 7),
                       'uniform_prop2': (self.uniform_prop_crossover, 2),
                       'uniform_prop7': (self.uniform_prop_crossover, 7),
                       'uniform_rank2': (self.uniform_rank_crossover, 2),
                       'uniform_rank7': (self.uniform_rank_crossover, 7),
                       'uniform_tour3': (self.uniform_tour_crossover, 3),
                       'uniform_tour7': (self.uniform_tour_crossover, 7)}
        self.c_sets = dict(sorted(self.c_sets.items()))

        self.thefittest:TheFittest
        self.stats:Statictic

    @staticmethod
    def mutation(individ, proba):
        mask = np.random.random(size=individ.shape) < proba
        individ[mask] = 1 - individ[mask]
        return individ

    @staticmethod
    def one_point_crossover(individs, fitness):
        cross_point = np.random.randint(0, len(individs[0]))
        if np.random.random() > 0.5:
            offspring = individs[0]
            offspring[:cross_point] = individs[1][:cross_point]
        else:
            offspring = individs[1]
            offspring[:cross_point] = individs[0][:cross_point]
        return offspring

    @staticmethod
    def two_point_crossover(individs, fitness):
        c_point_1, c_point_2 = np.sort(np.random.choice(range(len(individs[0])),
                                                        size=2,
                                                        replace=False))
        if np.random.random() > 0.5:
            offspring = individs[0]
            offspring[c_point_1:c_point_2] = individs[1][c_point_1:c_point_2]
        else:
            offspring = individs[1]
            offspring[c_point_1:c_point_2] = individs[0][c_point_1:c_point_2]

        return offspring

    @staticmethod
    def uniform_crossover(individs, fitness):
        choosen = np.random.choice(range(len(individs)),
                                   size=len(individs[0]))
        diag = range(len(individs[0]))
        return individs[choosen, diag]

    @staticmethod
    def uniform_prop_crossover(individs, fitness):
        probability = fitness/fitness.sum()
        choosen = np.random.choice(range(len(individs)),
                                   size=len(individs[0]), p=probability)

        diag = range(len(individs[0]))
        return individs[choosen, diag]

    @staticmethod
    def uniform_rank_crossover(individs, fitness):
        probability = rank_scale_data(fitness)
        choosen = np.random.choice(range(len(individs)),
                                   size=len(individs[0]), p=probability)

        diag = range(len(individs[0]))
        return individs[choosen, diag]

    @staticmethod
    def uniform_tour_crossover(individs, fitness):
        tournament = np.random.choice(range(len(individs)), 2*len(individs[0]))
        tournament = tournament.reshape(-1, 2)

        choosen = np.argmax(fitness[tournament], axis=1)
        diag = range(len(individs[0]))
        return individs[choosen, diag]

    @staticmethod
    def proportional_selection(population, fitness, tour_size, quantity):
        probability = fitness/fitness.sum()
        choosen = np.random.choice(range(len(population)),
                                   size=quantity, p=probability)
        return choosen

    @staticmethod
    def rank_selection(population, fitness, tour_size, quantity):
        probability = rank_scale_data(fitness)
        choosen = np.random.choice(range(len(population)),
                                   size=quantity, p=probability)
        return choosen

    @staticmethod
    def tournament_selection(population, fitness, tour_size, quantity):
        tournament = np.random.choice(
            range(len(population)), tour_size*quantity)
        tournament = tournament.reshape(-1, tour_size)
        max_fit_id = np.argmax(fitness[tournament], axis=1)
        choosen = np.diag(tournament[:, max_fit_id])
        return choosen

    def create_offs(self, popuation, fitness,
                    selection, crossover, mutation):
        crossover_func, quantity = self.c_sets[crossover]
        selection_func, tour_size = self.s_sets[selection]
        indexes = selection_func(popuation, fitness, tour_size, quantity)
        parents = popuation[indexes].copy()
        fitness_p = fitness[indexes].copy()
        offspring_no_mutated = crossover_func(parents, fitness_p)
        return self.mutation(offspring_no_mutated, self.m_sets[mutation])

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

    def operators_selector(self, select_opers,
                           crossover_opers,
                           mutation_opers):

        s_sets = {}
        c_sets = {}
        m_sets = {}
        for operator_name in select_opers:
            value = self.s_sets[operator_name]
            s_sets[operator_name] = value

        for operator_name in crossover_opers:
            value = self.c_sets[operator_name]
            c_sets[operator_name] = value

        for operator_name in mutation_opers:
            value = self.m_sets[operator_name]
            m_sets[operator_name] = value

        self.s_sets = s_sets
        self.c_sets = c_sets
        self.m_sets = m_sets

    def fit(self):
        calls = 0
        no_increase = 0

        z_s, z_c, z_m = list(map(len, (self.s_sets, self.c_sets, self.m_sets)))

        s_proba = np.full(z_s, 1/z_s)
        c_proba = np.full(z_c, 1/z_c)
        m_proba = np.full(z_m, 1/z_m)

        population_g = np.random.randint(low=2, size=(self.pop_size,
                                                      self.str_len),
                                         dtype=np.byte)
        population_ph = self.genotype_to_phenotype(population_g)
        fitness = self.fitness_function(population_ph)
        calls += len(population_ph)
        fitness_scale = scale_data(fitness)

        self.thefittest = TheFittest(
            genotype=population_g[np.argmax(fitness)].copy(),
            phenotype=population_ph[np.argmax(fitness)].copy(),
            fitness=fitness[np.argmax(fitness)].copy())
        last_best = fitness[np.argmax(fitness)].copy()
        self.stats = Statictic().update(fitness, s_proba, c_proba, m_proba)

        for i in range(self.iters-1):

            find_opt = self.thefittest.fitness == self.optimal_value
            no_increase_cond = no_increase == self.no_increase_num

            if find_opt or no_increase_cond:
                break

            s_operators = self.choice_operators(self.s_sets.keys(), s_proba)
            c_operators = self.choice_operators(self.c_sets.keys(), c_proba)
            m_operators = self.choice_operators(self.m_sets.keys(), m_proba)

            create_offs = partial(
                self.create_offs, population_g, fitness_scale)
            population_g[:-1] = np.array(list(map(create_offs, s_operators,
                                                  c_operators, m_operators)))
            population_ph[:-1] = self.genotype_to_phenotype(population_g[:-1])
            fitness[:-1] = self.fitness_function(population_ph[:-1])
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

            self.thefittest.update(population_g, population_ph, fitness)

            temp_best = self.thefittest.fitness.copy()
            if last_best == temp_best:
                no_increase += 1
            else:
                last_best = temp_best

            self.stats = self.stats.update(fitness, s_proba, c_proba, m_proba)

        self.remains = (self.pop_size + (self.iters-1)
                        * (self.pop_size-1)) - calls
        return (self.thefittest, self.stats)
