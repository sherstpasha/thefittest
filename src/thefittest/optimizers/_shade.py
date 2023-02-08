import numpy as np
from typing import Optional
from typing import Callable
from typing import Any
from ._base import TheFittest
from ._base import Statistics
from ._base import LastBest
from functools import partial
from ._differentialevolution import DifferentialEvolution
from ._crossovers import binomial
from ..tools import cauchy_distribution
from ..tools import lehmer_mean


class StatisticsSHADE(Statistics):
    def __init__(self):
        Statistics.__init__(self)
        self.H_F = np.array([], dtype=float)
        self.H_CR = np.array([], dtype=float)

    def update(self,
               population_g_i: np.ndarray,
               population_ph_i: np.ndarray,
               fitness_i: np.ndarray,
               H_F_i: np.ndarray,
               H_CR_i: np.ndarray):
        super().update(population_g_i, population_ph_i, fitness_i)
        if not len(self.H_F):
            self.H_F = H_F_i.copy().reshape(1, -1)
            self.H_CR = H_CR_i.copy().reshape(1, -1)
        else:
            self.H_F = np.append(self.H_F, H_F_i.copy().reshape(1, -1), axis=0)
            self.H_CR = np.append(
                self.H_CR, H_CR_i.copy().reshape(1, -1), axis=0)
        return self


class SHADE(DifferentialEvolution):
    '''Tanabe, Ryoji & Fukunaga, Alex. (2013). Success-history based parameter adaptation
    for Differential Evolution. 2013 IEEE Congress on Evolutionary Computation,
    CEC 2013. 71-78. 10.1109/CEC.2013.6557555. '''

    def __init__(self,
                 fitness_function: Callable[[np.ndarray[Any]], np.ndarray[float]],
                 genotype_to_phenotype: Callable[[np.ndarray[Any]], np.ndarray[Any]],
                 iters: int,
                 pop_size: int,
                 left: np.ndarray[float],
                 right: np.ndarray[float],
                 optimal_value: Optional[float] = None,
                 termination_error_value: float = 0.,
                 no_increase_num: Optional[int] = None,
                 minimization: bool = False,
                 show_progress_each: Optional[int] = None,
                 keep_history: bool = False):
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
        self.stats: StatisticsSHADE
        self.H_size = pop_size

    def append_archive(self, archive, worse_i):
        archive = np.append(archive, worse_i, axis=0)
        if len(archive) > self.pop_size:
            np.random.shuffle(archive)
            archive = archive[:self.pop_size]
        return archive

    def current_to_pbest_1_archive(self, current, population, F_value, pop_archive):
        p_min = 2/len(population)
        p_i = np.random.uniform(p_min, 0.2)
        value = int(p_i*len(population))
        pbest = population[-value:]
        p_best_ind = np.random.randint(0, len(pbest))
        best = pbest[p_best_ind]

        r1 = np.random.choice(range(len(population)), size=1, replace=False)[0]
        r2 = np.random.choice(range(len(pop_archive)),
                              size=1, replace=False)[0]
        return current + F_value*(best - current) + F_value*(population[r1] - pop_archive[r2])

    def set_strategy(self):
        return self

    def mutation_and_crossover(self, popuation_g, popuation_g_archive, individ_g, F_i, CR_i):
        mutant = self.current_to_pbest_1_archive(individ_g, popuation_g, F_i,
                                                 popuation_g_archive)

        mutant_cr_g = binomial(individ_g, mutant, CR_i)
        mutant_cr_g = self.bounds_control(mutant_cr_g)
        return mutant_cr_g

    def evaluate_and_selection(self, mutant_cr_g, population_g, population_ph, fitness):
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

    def bounds_control(self, individ_g):
        low_mask = individ_g < self.left
        high_mask = individ_g > self.right

        individ_g[low_mask] = (self.left[low_mask] + individ_g[low_mask])/2
        individ_g[high_mask] = (self.right[high_mask] + individ_g[high_mask])/2
        return individ_g

    def generate_F_CR(self, H_F_i, H_CR_i, size):
        F_i = np.zeros(size)
        CR_i = np.zeros(size)
        for i in range(size):
            r_i = np.random.randint(0, len(H_F_i))
            u_F = H_F_i[r_i]
            u_CR = H_CR_i[r_i]
            F_i[i] = self.randc01(u_F)
            CR_i[i] = self.randn01(u_CR)
        return F_i, CR_i

    def randc01(self, u):
        value = cauchy_distribution(loc=u, scale=0.1)[0]
        while value <= 0:
            value = cauchy_distribution(loc=u, scale=0.1)[0]
        if value > 1:
            value = 1
        return value

    def randn01(self, u):
        value = np.random.normal(u, 0.1)
        if value < 0:
            value = 0
        elif value > 1:
            value = 1
        return value

    def update_u_F(self, u_F, S_F):
        if len(S_F):
            return lehmer_mean(S_F)
        return u_F

    def update_u_CR(self, u_CR, S_CR, df):
        if len(S_CR):
            sum_ = np.sum(df)
            if sum_ > 0:
                weight_i = df/sum_
                return np.sum(weight_i*S_CR)
        return u_CR

    def fit(self):
        H_F = np.full(self.H_size, 0.5)
        H_CR = np.full(self.H_size, 0.5)
        k = 0
        next_k = 1

        external_archive = np.zeros(shape=(0, len(self.left)))

        population_g = self.generate_init_pop()
        population_ph = self.genotype_to_phenotype(population_g)
        fitness = self.evaluate(population_ph)

        argsort = np.argsort(fitness)
        population_g = population_g[argsort]
        population_ph = population_ph[argsort]
        fitness = fitness[argsort]

        self.thefittest = TheFittest().update(population_g,
                                              population_ph,
                                              fitness)
        lastbest = LastBest().update(self.thefittest.fitness)
        if self.keep_history:
            self.stats = StatisticsSHADE().update(population_g,
                                                  population_ph,
                                                  fitness,
                                                  H_F,
                                                  H_CR)
        for i in range(self.iters-1):
            self.show_progress(i)
            if self.termitation_check(lastbest.no_increase):
                break
            else:
                F_i, CR_i = self.generate_F_CR(H_F, H_CR, self.pop_size - 1)
                pop_archive = np.vstack(
                    [population_g.copy(), external_archive.copy()])

                partial_mut_and_cross = partial(self.mutation_and_crossover,
                                                population_g.copy(), pop_archive.copy())
                mutant_cr_g = np.array(list(map(partial_mut_and_cross,
                                                population_g[:-1].copy(),
                                                F_i.copy(), CR_i.copy())))

                stack = self.evaluate_and_selection(mutant_cr_g.copy(),
                                                    population_g[:-1].copy(),
                                                    population_ph[:-1].copy(),
                                                    fitness[:-1].copy())

                succeses = stack[3]
                will_be_replaced_pop = population_g[:-1][succeses].copy()
                will_be_replaced_fit = fitness[:-1][succeses].copy()
                s_F = F_i[succeses].copy()
                s_CR = CR_i[succeses].copy()

                external_archive = self.append_archive(
                    external_archive, will_be_replaced_pop)

                population_g[:-1] = stack[0]
                population_ph[:-1] = stack[1]
                fitness[:-1] = stack[2]

                df = np.abs(will_be_replaced_fit - fitness[:-1][succeses])

                population_g[-1], population_ph[-1], fitness[-1] = self.thefittest.get()
                argsort = np.argsort(fitness)
                population_g = population_g[argsort]
                population_ph = population_ph[argsort]
                fitness = fitness[argsort]

                if next_k == self.H_size:
                    next_k = 0

                H_F[next_k] = self.update_u_F(H_F[k], s_F)
                H_CR[next_k] = self.update_u_CR(H_CR[k], s_CR, df)

                self.thefittest.update(population_g, population_ph, fitness)
                lastbest.update(self.thefittest.fitness)

                if self.keep_history:
                    self.stats.update(population_g,
                                      population_ph,
                                      fitness,
                                      H_F,
                                      H_CR)

                if k == self.H_size - 1:
                    k = 0
                    next_k = 1
                else:
                    k += 1
                    next_k += 1

        return self
