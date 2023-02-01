import numpy as np
from functools import partial
from ._units import TheFittest
from ._differentialevolution import DifferentialEvolution
from ._crossovers import binomial
from ._mutations import current_to_pbest_1_archive
from ..tools._transformations import lehmer_mean
from scipy import stats

class JADE(DifferentialEvolution):
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
        DifferentialEvolution.__init__(self,
                                       fitness_function, genotype_to_phenotype,
                                       left,
                                       right,
                                       iters,
                                       pop_size,
                                       optimal_value,
                                       termination_error_value,
                                       no_increase_num,
                                       minimization,
                                       show_progress_each,
                                       keep_history)

        self.m_pool = {
            'current_to_pbest_1_archive': current_to_pbest_1_archive}
        self.m_function = self.m_pool['current_to_pbest_1_archive']
        self.c = 0.1

    def selection(self, mutant_cr_g, popuation_g, popuation_ph, fitness, F_i, CR_i):
        offspring_g = popuation_g.copy()
        offspring_ph = popuation_ph.copy()
        offspring_fit = fitness.copy()

        mutant_cr_ph = self.genotype_to_phenotype(mutant_cr_g)
        mutant_cr_fit = self.evaluate(mutant_cr_ph)
        mask = mutant_cr_fit >= fitness
        worse = popuation_g[mask].copy()

        offspring_g[mask] = mutant_cr_g[mask]
        offspring_ph[mask] = mutant_cr_ph[mask]
        offspring_fit[mask] = mutant_cr_fit[mask]

        return offspring_g, offspring_ph, offspring_fit, worse, F_i[mask], CR_i[mask]

    def append_archive(self, archive, worse_i):
        if len(archive) + len(worse_i) > self.pop_size:
            np.random.shuffle(archive)
            archive = archive[:self.pop_size - len(worse_i)]
        archive = np.append(archive, worse_i, axis=0)
        return archive

    def create_offs(self, popuation_g, popuation_g_archive, individ_g, F_i, CR_i):
        mutant = self.m_function(
            individ_g, popuation_g, F_i, popuation_g_archive)

        mutant_cr_g = binomial(individ_g, mutant, CR_i)
        mutant_cr_g = self.bounds_control(mutant_cr_g)
        return mutant_cr_g

    def generate_params(self, u_F, u_CR):
        F_i = self.cauchy(u_F, 0.1, self.pop_size-1)
        # F_i = stats.cauchy.rvs(loc=u_F, scale=0.1, size=self.pop_size-1)
        # F_i = np.random.normal(u_F, 0.1, self.pop_size-1)

        mask = F_i <= 0
        while np.any(mask):
            F_i[mask] = self.cauchy(u_F, 0.1, len(F_i[mask]))
            # F_i[mask] = stats.cauchy.rvs(loc=u_F, scale=0.1, size = len(F_i[mask]))
            # F_i[mask] = np.random.normal(u_F, 0.1, len(F_i[mask]))

            mask = F_i <= 0 

        F_i = np.clip(F_i, -1, 1)

        CR_i = self.cauchy(u_CR, 0.1, self.pop_size-1)
        # F_i = stats.cauchy.rvs(loc=u_F, scale=0.1, size=self.pop_size-1)
        # F_i = np.random.normal(u_F, 0.1, self.pop_size-1)

        mask = CR_i <= 0
        while np.any(mask):
            CR_i[mask] = self.cauchy(u_CR, 0.1, len(CR_i[mask]))
            # F_i[mask] = stats.cauchy.rvs(loc=u_F, scale=0.1, size = len(F_i[mask]))
            # F_i[mask] = np.random.normal(u_F, 0.1, len(F_i[mask]))

            mask = CR_i <= 0 

        CR_i = np.clip(CR_i, -1, 1)


        # CR_i = np.random.normal(u_CR, 0.1, self.pop_size-1)
        # CR_i = np.clip(CR_i, 1e-6, 1)
        return F_i, CR_i

    def update_u(self, u_F, u_CR, S_F, S_CR):
        if len(S_F) > 0:
 
            u_F = (1 - self.c)*u_F + self.c*lehmer_mean(S_F)
            u_CR = (1 - self.c)*u_CR + self.c*np.mean(S_CR)

        return u_F, u_CR

    def cauchy(self, loc, scale, size):
        x_ = np.random.standard_cauchy(size = size)
        return loc + scale*x_

    def fit(self, initial_population=None):

        calls = 0
        no_increase = 0
        u_F = u_CR = 0.5
        external_archive = np.zeros(shape=(0, len(self.left)))
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

        # calls += len(population_ph)

        argmax = np.argmax(fitness)
        self.thefittest = TheFittest(
            genotype=population_g[argmax].copy(),
            phenotype=population_ph[argmax].copy(),
            fitness=fitness[argmax].copy())

        # # last_best = fitness[np.argmax(fitness)].copy()
        # # if self.keep_history:
        # #     self.stats = StaticticDifferentialEvolution().update(population_g, fitness)
        # # else:
        # #     self.stats = None

        for i in range(self.iters-1):

            if self.show_progress_each is not None:
                if i % self.show_progress_each == 0:
                    print(f'{i} iterstion with fitness = {self.thefittest.fitness}')
            # if self.optimal_value is not None:
            #     if self.minimization:
            #         aim = -(self.optimal_value) - self.termination_error_value
            #     else:
            #         aim = self.optimal_value - self.termination_error_value

            #     find_opt = self.thefittest.fitness >= aim

            # else:
            #     find_opt = False

            # no_increase_cond = no_increase == self.no_increase_num

            # if find_opt or no_increase_cond:
            #     break
            pop_archive = np.vstack([population_g, external_archive])
            create_offs = partial(self.create_offs, population_g, pop_archive)

            F_i, CR_i = self.generate_params(u_F, u_CR)
            # self.F = 0.5
            # self.CR = 0.5
            # F_i = np.full(self.pop_size-1, self.F)
            # CR_i = np.full(self.pop_size-1, self.CR)


            temp_map = map(create_offs, population_g, F_i, CR_i)
            mutant_cr_g = np.array(list(temp_map))

            population_g[:-1], population_ph[:-1], fitness[:-1], worse, S_F, S_CR = self.selection(mutant_cr_g,
                                                                                                   population_g[:-1],
                                                                                                   population_ph[:-1],
                                                                                                   fitness[:-1],
                                                                                                   F_i,
                                                                                                   CR_i)
            population_g[-1], population_ph[-1], fitness[-1] = self.thefittest.get()
            argsort = np.argsort(fitness)
            population_g = population_g[argsort]
            population_ph = population_ph[argsort]
            fitness = fitness[argsort]
            external_archive = self.append_archive(external_archive, worse)

            u_F, u_CR = self.update_u(u_F, u_CR, S_F, S_CR)
            # print(u_F, u_CR)
            self.thefittest.update(population_g, population_ph, fitness)
            # temp_best = self.thefittest.fitness.copy()
            # if last_best == temp_best:
            #     no_increase += 1
            # else:
            #     last_best = temp_best

            # if self.keep_history:
            #     self.stats = self.stats.update(population_g, fitness)

        # self.remains = (self.pop_size + (self.iters-1)
        #                 * (self.pop_size-1)) - calls
        # self.calls = calls

        if self.minimization:
            self.thefittest.fitness = -self.thefittest.fitness
        return self
