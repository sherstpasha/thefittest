import numpy as np
from typing import Optional
from typing import Callable
from typing import Any
from ._base import TheFittest
from ._base import StaticticSaDE
from ._base import LastBest
from functools import partial
from ._differentialevolution import DifferentialEvolution
from ._crossovers import binomial
from ._mutations import rand_1
from ._mutations import current_to_best_1
from ._mutations import rand_2
from ._mutations import current_to_rand_1


class SaDE(DifferentialEvolution):
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

        self.m_sets = {'rand_1': rand_1,
                       'current_to_best_1': current_to_best_1,
                       'rand_2': rand_2,
                       'current_to_rand_1': current_to_rand_1}
        self.m_sets = dict(sorted(self.m_sets.items()))

        self.thefittest: TheFittest
        self.stats: StaticticSaDE
        self.m_learning_period = 20
        self.CR_update_timer = 5
        self.CR_m_learning_period = 20

    def set_strategy(self, m_learning_period_param: Optional[int] = None,
                     CR_update_timer_param: Optional[int] = None,
                     CR_m_learning_period_param: Optional[int] = None):
        if m_learning_period_param is not None:
            self.m_learning_period = m_learning_period_param
        if CR_update_timer_param is not None:
            self.CR_update_timer = CR_update_timer_param
        if CR_m_learning_period_param is not None:
            self.CR_m_learning_period = CR_m_learning_period_param
        return self

    def mutation_and_crossover(self, popuation_g, individ_g,
                               mutation_type, F_i, CR_i):
        mutant = self.m_sets[mutation_type](individ_g, popuation_g, F_i)

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

    def choice_operators(self, operators, proba):
        return np.random.choice(list(operators), self.pop_size - 1, p=proba)

    def sum_and_diff(self, group_i):
        sum_ = np.sum(group_i)
        return sum_, len(group_i) - sum_

    def groupy_by_ns_nf(self, operators, succeses):
        operators_succeses = np.vstack([operators, succeses.astype(int)]).T
        argsort = np.argsort(operators_succeses[:, 0])
        operators_succeses = operators_succeses[argsort]

        keys, cut_index = np.unique(
            operators_succeses[:, 0], return_index=True)
        groups = np.split(
            operators_succeses[:, 1].astype(float), cut_index)[1:]
        stack = list(map(self.sum_and_diff, groups))
        ns, nf = list(zip(*stack))
        
        ind = [list(self.m_sets.keys()).index(key) for key in keys]

        return np.array(ns), np.array(nf), ind

    def update_proba(self, ns_i, nf_i):
        down = (ns_i + nf_i)
        down[down==0]=1
        percentage = ns_i/down

        down = np.sum(percentage)
        if down == 0:
            proba = 1/len(self.m_sets)
        else:
            proba = percentage/down
        return proba

    def fit(self):

        z_m = len(self.m_sets)
        m_proba = np.full(z_m, 1/z_m)

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
            self.stats = StaticticSaDE().update(population_g,
                                                population_ph,
                                                fitness)
        ns = np.zeros(z_m)
        nf = np.zeros(z_m)
        ns_nf_counter = 0
        for i in range(self.iters-1):

            self.show_progress(i)
            if self.termitation_check(lastbest.no_increase):
                break
            else:

                m_operators = self.choice_operators(
                    self.m_sets.keys(), m_proba)
                F_i = np.full(self.pop_size-1, self.F)
                CR_i = np.full(self.pop_size-1, self.CR)

                partial_mut_and_cross = partial(self.mutation_and_crossover,
                                                population_g)
                mutant_cr_g = np.array(list(map(partial_mut_and_cross,
                                                population_g[:-1],
                                                m_operators,
                                                F_i, CR_i)))

                stack = self.evaluate_and_selection(mutant_cr_g,
                                                    population_g[:-1],
                                                    population_ph[:-1],
                                                    fitness[:-1])
                population_g[:-1], population_ph[:-
                                                 1], fitness[:-1] = stack[:-1]
                successes = stack[-1]

                ns_i, nf_i, ind = self.groupy_by_ns_nf(m_operators, successes)
                ns[ind] += ns_i
                nf[ind] += nf_i
                ns_nf_counter += 1

                population_g[-1], population_ph[-1], fitness[-1] = self.thefittest.get()
                argsort = np.argsort(fitness)
                population_g = population_g[argsort]
                population_ph = population_ph[argsort]
                fitness = fitness[argsort]

                self.thefittest.update(population_g, population_ph, fitness)
                lastbest.update(self.thefittest.fitness)
                if self.keep_history:
                    self.stats.update(population_g,
                                      population_ph,
                                      fitness)

                m_proba = self.update_proba(ns, nf)
                print(ns_nf_counter)
                if ns_nf_counter == 20:
                    ns_nf_counter = 0
                    ns = np.zeros(z_m)
                    nf = np.zeros(z_m)

        return self
