import numpy as np
from ..base import TheFittest
from ..base import EvolutionaryAlgorithm
from ..base import LastBest
from ..base import Statistics
from ..tools.generators import cauchy_distribution
from ..tools.generators import binary_string_population
from ..tools.operators import tournament_selection
from ..tools.operators import binomial
from ..tools.operators import flip_mutation
from functools import partial


class SHAGA(EvolutionaryAlgorithm):
    '''Stanovov, Vladimir & Akhmedova, Shakhnaz & Semenkin, Eugene. (2019).
    Genetic Algorithm with Success History based Parameter Adaptation. 180-187.
    10.5220/0008071201800187. '''

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
                 keep_history=False):
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

        self.H_size = pop_size
        self.s_set = tournament_selection
        self.m_set = flip_mutation
        self.c_set = binomial
        self.tour_size = 2
        self.elitism: bool
        self.initial_population: np.ndarray
        self.set_strategy()

    def set_strategy(self,
                     elitism_param=True,
                     initial_population=None):
        self.elitism = elitism_param
        self.initial_population = initial_population
        return self

    def generate_init_pop(self):
        if self.initial_population is None:
            population_g = binary_string_population(
                self.pop_size, self.str_len)
        else:
            population_g = self.initial_population
        return population_g

    def selection_crossover_mutation(self, population_g, fitness, current, MR_i, CR_i):
        second_parent_id = self.s_set(fitness, fitness, self.tour_size, 1)[0]
        second_parent = population_g[second_parent_id].copy()
        offspring = self.c_set(current, second_parent, CR_i)
        mutant = self.m_set(offspring, MR_i)
        return mutant

    def evaluate_replace(self, mutant_cr_g, population_g, population_ph, fitness):
        offspring_g = population_g.copy()
        offspring_ph = population_ph.copy()
        offspring_fit = fitness.copy()

        mutant_cr_ph = self.genotype_to_phenotype(mutant_cr_g)
        mutant_cr_fit = self.evaluate(mutant_cr_ph)
        mask_more_equal = mutant_cr_fit >= fitness
        offspring_g[mask_more_equal] = mutant_cr_g[mask_more_equal]
        offspring_ph[mask_more_equal] = mutant_cr_ph[mask_more_equal]
        offspring_fit[mask_more_equal] = mutant_cr_fit[mask_more_equal]
        mask_more = mutant_cr_fit > fitness
        return offspring_g, offspring_ph, offspring_fit, mask_more

    def generate_MR_CR(self, H_MR_i, H_CR_i, size):
        MR_i = np.zeros(size)
        CR_i = np.zeros(size)
        for i in range(size):
            r_i = np.random.randint(0, len(H_MR_i))
            u_MR = H_MR_i[r_i]
            u_CR = H_CR_i[r_i]
            MR_i[i] = self.randc(u_MR, 0.1/self.str_len)
            CR_i[i] = self.randn(u_CR, 0.1)
        return MR_i, CR_i

    def randc(self, u, scale):
        value = cauchy_distribution(loc=u, scale=scale)[0]
        while value <= 0 or value > 5/self.str_len:
            value = cauchy_distribution(loc=u, scale=scale)[0]
        return value

    def randn(self, u, scale):
        value = np.random.normal(u, scale)
        if value < 0:
            value = 0
        elif value > 1:
            value = 1
        return value

    def weighted_lehmer2(self, w, x):
        up = np.sum(w*x*x)
        down = np.sum(w*x)
        return up/down

    def update_u(self, u, S, df):
        if len(S):
            sum_ = np.sum(df)
            if sum_ > 0:
                weight_i = df/sum_
                return self.weighted_lehmer2(weight_i, S)
        return u

    def fit(self):
        H_MR = np.full(self.H_size, 1/(self.str_len))
        H_CR = np.full(self.H_size, 0.5)
        k = 0
        next_k = 1

        population_g = self.generate_init_pop()
        population_ph = self.genotype_to_phenotype(population_g)
        fitness = self.evaluate(population_ph)

        self.thefittest = TheFittest().update(population_g,
                                              population_ph,
                                              fitness)
        lastbest = LastBest().update(self.thefittest.fitness)
        if self.keep_history:
            self.stats = Statistics(
                mode=self.keep_history).update({'population_g': population_g.copy(),
                                                'fitness_max': self.thefittest.fitness,
                                                'H_MR': H_MR.copy(),
                                                'H_CR': H_CR.copy()})

        for i in range(self.iters-1):
            self.show_progress(i)
            if self.termitation_check(lastbest.no_increase_counter):
                break
            else:
                MR_i, CR_i = self.generate_MR_CR(H_MR, H_CR, self.pop_size)

                partial_operators = partial(self.selection_crossover_mutation,
                                            population_g.copy(), fitness.copy())

                mutant_cr_g = np.array(list(map(partial_operators,
                                                population_g.copy(),
                                                MR_i.copy(), CR_i.copy())))

                stack = self.evaluate_replace(mutant_cr_g.copy(),
                                              population_g.copy(),
                                              population_ph.copy(),
                                              fitness.copy())

                succeses = stack[3]
                will_be_replaced_fit = fitness[succeses].copy()
                s_MR = MR_i[succeses].copy()
                s_CR = CR_i[succeses].copy()

                population_g = stack[0]
                population_ph = stack[1]
                fitness = stack[2]

                df = np.abs(will_be_replaced_fit - fitness[succeses])

                if self.elitism:
                    population_g[-1], population_ph[-1], fitness[-1] = self.thefittest.get()

                if next_k == self.H_size:
                    next_k = 0

                H_MR[next_k] = self.update_u(H_MR[k], s_MR, df)
                H_CR[next_k] = self.update_u(H_CR[k], s_CR, df)

                self.thefittest.update(population_g, population_ph, fitness)
                lastbest.update(self.thefittest.fitness)

                if self.keep_history:
                    self.stats.update({'population_g': population_g.copy(),
                                       'fitness_max': self.thefittest.fitness,
                                       'H_MR': H_MR.copy(),
                                       'H_CR': H_CR.copy()})

                if k == self.H_size - 1:
                    k = 0
                    next_k = 1
                else:
                    k += 1
                    next_k += 1

        return self
