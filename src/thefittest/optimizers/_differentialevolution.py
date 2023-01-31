import numpy as np
from functools import partial
from ._units import TheFittest
from ._units import StaticticDifferentialEvolution
from ._selections import selection_de
from ._crossovers import binomial
from ._mutations import best_1
from ._mutations import best_2
from ._mutations import rand_to_best1
from ._mutations import current_to_best_1
from ._mutations import rand_1
from ._mutations import current_to_pbest_1
from ._mutations import rand_2


class DifferentialEvolution:
    def __init__(self,
                 fitness_function,
                 genotype_to_phenotype,
                 left,
                 right,
                 iters,
                 pop_size,
                 optimal_value=None,
                 termination_error_value=0.,
                 no_increase_num=None,
                 minimization=False,
                 show_progress_each=None,
                 keep_history=False):

        self.fitness_function = fitness_function
        self.genotype_to_phenotype = genotype_to_phenotype
        self.left = left
        self.right = right
        self.iters = iters
        self.pop_size = pop_size

        self.optimal_value = optimal_value
        self.termination_error_value = termination_error_value
        self.no_increase_num = no_increase_num
        self.minimization = minimization
        self.show_progress_each = show_progress_each
        self.keep_history = keep_history

        self.thefittest = None

        self.m_pool = {'best_1': best_1,
                       'rand_1': rand_1,
                       'current_to_best_1': current_to_best_1,
                       'current_to_pbest_1': current_to_pbest_1,
                       'rand_to_best1': rand_to_best1,
                       'best_2': best_2,
                       'rand_2': rand_2}

        self.thefittest: TheFittest
        self.stats: StaticticDifferentialEvolution
        self.m_function = None
        self.F = None
        self.CR = None

        self.set_strategy(mutation_oper='best_1',
                          F_param=0.9, CR_param=0.9)

    def evaluate(self, population_ph):
        if self.minimization:
            return -self.fitness_function(population_ph)
        else:
            return self.fitness_function(population_ph)

    def create_offs(self, popuation_g, fitness, individ_g, individ_ph, fitness_i, F_i, CR_i):
        mutant = self.m_function(individ_g, popuation_g, F_i)

        mutant_cr_g = binomial(individ_g, mutant, CR_i)
        mutant_cr_g = self.bounds_control(mutant_cr_g)
        mutant_cr_ph = self.genotype_to_phenotype(individ_ph.reshape(1, -1))[0]

        fitness_cr = self.evaluate(mutant_cr_g.reshape(1, -1))[0]

        return selection_de(individ_g, individ_ph, mutant_cr_g, mutant_cr_ph, fitness_i, fitness_cr)

    def bounds_control(self, individ_g):
        low_mask = individ_g < self.left
        high_mask = individ_g > self.right

        individ_g[low_mask] = 2*self.left[low_mask] - individ_g[low_mask]
        individ_g[high_mask] = 2*self.right[high_mask] - individ_g[high_mask]
        return individ_g

    def set_strategy(self,
                     mutation_oper=None,
                     F_param=None,
                     CR_param=None):
        if mutation_oper is not None:
            self.m_function = self.m_pool[mutation_oper]
        if F_param is not None:
            self.F = F_param
        if CR_param is not None:
            self.CR = CR_param

    def fit(self, initial_population=None):
        calls = 0
        no_increase = 0

        if initial_population is None:
            population_g = np.array([np.random.uniform(left_i, right_i, self.pop_size)
                                     for left_i, right_i in zip(self.left, self.right)]).T
        else:
            population_g = initial_population
        population_ph = self.genotype_to_phenotype(population_g)
        fitness = self.evaluate(population_ph)

        argsort = np.argsort(fitness)
        population_g = population_g[argsort]
        population_ph = population_ph[argsort]
        fitness = fitness[argsort]

        calls += len(population_ph)

        argmax = np.argmax(fitness)
        self.thefittest = TheFittest(
            genotype=population_g[argmax].copy(),
            phenotype=population_ph[argmax].copy(),
            fitness=fitness[argmax].copy())

        last_best = fitness[np.argmax(fitness)].copy()
        if self.keep_history:
            self.stats = StaticticDifferentialEvolution().update(population_g, fitness)
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

            create_offs = partial(self.create_offs, population_g, fitness)

            F_i = np.full(self.pop_size-1, self.F)
            CR_i = np.full(self.pop_size-1, self.F)

            temp_map = map(create_offs, population_g,
                           population_ph, fitness, F_i, CR_i)
            next_pop_g, next_pop_ph, next_fit = list(zip(*list(temp_map)))
            population_g[:-1] = np.array(next_pop_g)
            population_ph[:-1] = np.array(next_pop_ph)
            fitness[:-1] = np.array(next_fit)

            population_g[-1], population_ph[-1], fitness[-1] = self.thefittest.get()
            argsort = np.argsort(fitness)
            population_g = population_g[argsort]
            population_ph = population_ph[argsort]
            fitness = fitness[argsort]

            self.thefittest.update(population_g, population_ph, fitness)

            temp_best = self.thefittest.fitness.copy()
            if last_best == temp_best:
                no_increase += 1
            else:
                last_best = temp_best

            if self.keep_history:
                self.stats = self.stats.update(population_g, fitness)

        self.remains = (self.pop_size + (self.iters-1)
                        * (self.pop_size-1)) - calls
        self.calls = calls
        if self.minimization:
            self.thefittest.fitness = -self.thefittest.fitness
        return self