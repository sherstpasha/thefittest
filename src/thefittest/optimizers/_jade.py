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


class StatisticsJADE(Statistics):
    def __init__(self):
        Statistics.__init__(self)
        self.u_F = np.array([], dtype=float)
        self.u_CR = np.array([], dtype=float)

    def update(self,
               population_g_i: np.ndarray,
               population_ph_i: np.ndarray,
               fitness_i: np.ndarray,
               u_F_i: np.ndarray,
               U_CR_i: np.ndarray):
        super().update(population_g_i, population_ph_i, fitness_i)
        self.u_F = np.append(self.u_F, u_F_i)
        self.u_CR = np.append(self.u_CR, U_CR_i)
        return self


class JADE(DifferentialEvolution):
    '''Zhang, Jingqiao & Sanderson, A.C.. (2009). JADE: Adaptive Differential Evolution With Optional External Archive.
     Evolutionary Computation, IEEE Transactions on. 13. 945 - 958. 10.1109/TEVC.2009.2014613. '''

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
        self.stats: StatisticsJADE

        self.c = 0.1
        self.p = 0.05

    def current_to_pbest_1_archive(self, current, population, F_value, pop_archive):
        p_i = self.p
        value = int(p_i*len(population))
        pbest = population[-value:]
        p_best_ind = np.random.randint(0, len(pbest))
        best = pbest[p_best_ind]

        r1 = np.random.choice(range(len(population)), size=1, replace=False)[0]
        r2 = np.random.choice(range(len(pop_archive)),
                              size=1, replace=False)[0]
        return current + F_value*(best - current) + F_value*(population[r1] - pop_archive[r2])

    def set_strategy(self, c_param: Optional[float] = None,
                     p_param: Optional[float] = None):
        if c_param is not None:
            self.c = c_param
        if p_param is not None:
            self.p = p_param
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
        mask = mutant_cr_fit >= fitness
        offspring_g[mask] = mutant_cr_g[mask]
        offspring_ph[mask] = mutant_cr_ph[mask]
        offspring_fit[mask] = mutant_cr_fit[mask]
        return offspring_g, offspring_ph, offspring_fit, mask

    def bounds_control(self, individ_g):
        low_mask = individ_g < self.left
        high_mask = individ_g > self.right

        individ_g[low_mask] = (self.left[low_mask] + individ_g[low_mask])/2
        individ_g[high_mask] = (self.right[high_mask] + individ_g[high_mask])/2
        return individ_g

    def generate_F(self, u_F):
        F_i = cauchy_distribution(loc=u_F, scale=0.1, size=self.pop_size-1)
        mask = F_i <= 0
        while np.any(mask):
            F_i[mask] = cauchy_distribution(
                loc=u_F, scale=0.1, size=len(F_i[mask]))
            mask = F_i <= 0
        F_i[F_i >= 1] = 1
        return F_i

    def generate_CR(self, u_CR):
        CR_i = np.random.normal(u_CR, 0.1, self.pop_size-1)
        CR_i[CR_i >= 1] = 1
        CR_i[CR_i <= 0] = 0
        return CR_i

    def update_u_F(self, u_F, S_F):
        u_F = (1 - self.c)*u_F + self.c*lehmer_mean(S_F)
        return u_F

    def update_u_CR(self, u_CR, S_CR):
        u_CR = (1 - self.c)*u_CR + self.c*np.mean(S_CR)
        return u_CR

    def append_archive(self, archive, worse_i):
        archive = np.append(archive, worse_i, axis=0)
        if len(archive) > self.pop_size:
            np.random.shuffle(archive)
            archive = archive[:self.pop_size]
        return archive

    def fit(self):
        u_F = u_CR = 0.5
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
            self.stats = StatisticsJADE().update(population_g,
                                                 population_ph,
                                                 fitness,
                                                 u_F,
                                                 u_CR)

        for i in range(self.iters-1):
            self.show_progress(i)
            if self.termitation_check(lastbest.no_increase):
                break
            else:
                F_i = self.generate_F(u_F)
                CR_i = self.generate_CR(u_CR)
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
                will_be_replaced = population_g[:-1][succeses].copy()
                s_F = F_i[succeses].copy()
                s_CR = CR_i[succeses].copy()

                external_archive = self.append_archive(
                    external_archive, will_be_replaced)

                population_g[:-1] = stack[0]
                population_ph[:-1] = stack[1]
                fitness[:-1] = stack[2]

                population_g[-1], population_ph[-1], fitness[-1] = self.thefittest.get()
                argsort = np.argsort(fitness)
                population_g = population_g[argsort]
                population_ph = population_ph[argsort]
                fitness = fitness[argsort]

                u_F = self.update_u_F(u_F, s_F)
                u_CR = self.update_u_CR(u_CR, s_CR)

                self.thefittest.update(population_g, population_ph, fitness)
                lastbest.update(self.thefittest.fitness)

                if self.keep_history:
                    self.stats.update(population_g,
                                      population_ph,
                                      fitness,
                                      u_F,
                                      u_CR)

        return self
