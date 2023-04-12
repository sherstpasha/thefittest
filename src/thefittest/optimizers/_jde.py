import numpy as np
from ..base import TheFittest
from ..base import LastBest
from ..base import Statistics
from functools import partial
from ._differentialevolution import DifferentialEvolution
from ..tools.operators import binomial


class jDE(DifferentialEvolution):
    '''Brest, Janez & Greiner, Sao & Bošković, Borko & Mernik, Marjan & Zumer, Viljem. (2007).
    Self-Adapting Control Parameters in Differential Evolution: A Comparative Study on Numerical
    Benchmark Problems. Evolutionary Computation, IEEE Transactions on. 10. 646 - 657. 10.1109/TEVC.2006.872133. '''

    def __init__(self,
                 fitness_function,
                 genotype_to_phenotype,
                 iters,
                 pop_size,
                 left,
                 right,
                 optimal_value=None,
                 termination_error_value=0.,
                 no_increase_num=None,
                 minimization=False,
                 show_progress_each=None,
                 keep_history=False):
        DifferentialEvolution.__init__(
            self,
            fitness_function=fitness_function,
            genotype_to_phenotype=genotype_to_phenotype,
            iters=iters,
            pop_size=pop_size,
            left=left,
            right=right,
            optimal_value=optimal_value,
            termination_error_value=termination_error_value,
            no_increase_num=no_increase_num,
            minimization=minimization,
            show_progress_each=show_progress_each,
            keep_history=keep_history)

        self.thefittest: TheFittest
        self.stats: Statistics
        self.F_left: float
        self.F_right: float
        self.t_f: float
        self.t_cr: float

    def set_strategy(self,
                     mutation_oper='rand_1',
                     F_left_param=0.1,
                     F_right_param=0.9,
                     t_f_param=0.1,
                     t_cr_param=0.1,
                     elitism_param=True,
                     initial_population=None):
        self.update_pool()
        self.m_function = self.m_pool[mutation_oper]
        self.F_left = F_left_param
        self.F_right = F_right_param
        self.t_f = t_cr_param
        self.t_cr = t_f_param
        self.elitism = elitism_param
        self.initial_population = initial_population
        return self

    def mutation_and_crossover(self, popuation_g, individ_g, F_i, CR_i):
        mutant = self.m_function(individ_g, popuation_g, F_i)

        mutant_cr_g = binomial(individ_g, mutant, CR_i)
        mutant_cr_g = self.bounds_control(mutant_cr_g)
        return mutant_cr_g

    def evaluate_and_selection(self, mutant_cr_g, population_g, population_ph, fitness):
        offspring_g = population_g.copy()
        offspring_ph = population_ph.copy()
        offspring_fit = fitness.copy()

        mutant_cr_ph = self.genotype_to_phenotype(mutant_cr_g)
        mutant_cr_fit = self.evaluate(mutant_cr_ph)
        mask = mutant_cr_fit >= fitness
        offspring_g[mask] = mutant_cr_g[mask]
        offspring_ph[mask] = mutant_cr_ph[mask]
        offspring_fit[mask] = mutant_cr_fit[mask]
        return offspring_g, offspring_ph, offspring_fit, mask

    def bounds_control(self, individ_g):
        low_mask = individ_g < self.left
        high_mask = individ_g > self.right

        individ_g[low_mask] = self.left[low_mask]
        individ_g[high_mask] = self.right[high_mask]
        return individ_g

    def regenerate_F(self, F_i):
        mask = np.random.random(size=len(F_i)) < self.t_f
        F_i[mask] = self.F_left + \
            np.random.random(size=np.sum(mask))*self.F_right
        return F_i

    def regenerate_CR(self, CR_i):
        mask = np.random.random(size=len(CR_i)) < self.t_cr
        CR_i[mask] = np.random.random(size=np.sum(mask))
        return CR_i

    def fit(self):
        population_g = self.generate_init_pop()
        population_ph = self.genotype_to_phenotype(population_g)
        fitness = self.evaluate(population_ph)

        argsort = np.argsort(fitness)
        population_g = population_g[argsort]
        population_ph = population_ph[argsort]
        fitness = fitness[argsort]

        F_i = np.full(self.pop_size, 0.5)
        CR_i = np.full(self.pop_size, 0.9)

        self.thefittest = TheFittest().update(population_g,
                                              population_ph,
                                              fitness)
        lastbest = LastBest().update(self.thefittest.fitness)
        if self.keep_history:
            self.stats = Statistics(
                mode=self.keep_history).update({'population_g': population_g,
                                                'fitness_max': self.thefittest.fitness,
                                                'F': F_i.copy(),
                                                'CR': CR_i.copy()})

        for i in range(self.iters-1):
            self.show_progress(i)
            if self.termitation_check(lastbest.no_increase):
                break
            else:

                F_i_new = self.regenerate_F(F_i.copy())
                CR_i_new = self.regenerate_CR(CR_i.copy())

                partial_mut_and_cross = partial(self.mutation_and_crossover,
                                                population_g)
                mutant_cr_g = np.array(list(map(partial_mut_and_cross,
                                                population_g,
                                                F_i_new, CR_i_new)))

                stack = self.evaluate_and_selection(mutant_cr_g,
                                                    population_g,
                                                    population_ph,
                                                    fitness)
                population_g = stack[0]
                population_ph = stack[1]
                fitness = stack[2]

                succeses = stack[3]
                F_i[succeses] = F_i_new[succeses]
                CR_i[succeses] = CR_i_new[succeses]

                if self.elitism:
                    population_g[-1], population_ph[-1], fitness[-1] = self.thefittest.get()
                argsort = np.argsort(fitness)
                population_g = population_g[argsort]
                population_ph = population_ph[argsort]
                fitness = fitness[argsort]

                self.thefittest.update(population_g, population_ph, fitness)
                lastbest.update(self.thefittest.fitness)
                if self.keep_history:
                    self.stats.update({'population_g': population_g,
                                       'fitness_max': self.thefittest.fitness,
                                       'F': F_i.copy(),
                                       'CR': CR_i.copy()})
        return self
