import numpy as np
from typing import Optional
from typing import Callable
from typing import Any
from ._base import TheFittest
from ._base import LastBest
from functools import partial
from ._differentialevolution import DifferentialEvolution
from ..tools.operators import binomial
from ..tools.operators import rand_1
from ..tools.operators import current_to_best_1
from ..tools.transformations import numpy_group_by


class StaticticSaDE:
    def __init__(self, mode='quick'):
        self.mode = mode
        self.population_g = np.array([])
        self.fitness = np.array([])
        self.m_proba = dict()
        self.CRm = np.array([], dtype=float)

    def append_arr(self, arr_to, arr_from):
        shape_to = (-1, arr_from.shape[0], arr_from.shape[1])
        shape_from = (1, arr_from.shape[0], arr_from.shape[1])
        result = np.vstack([arr_to.reshape(shape_to),
                            arr_from.copy().reshape(shape_from)])
        return result

    def update(self,
               population_g_i: np.ndarray,
               fitness_i: np.ndarray,
               m_proba_i, CRm_i):
        if self.mode == 'quick':
            self.fitness = np.append(self.fitness, np.max(fitness_i))
            if not len(self.m_proba):
                for key, value in m_proba_i.items():
                    self.m_proba[key] = np.array(value)
            else:
                for key, value in m_proba_i.items():
                    self.m_proba[key] = np.append(
                        self.m_proba[key], np.array(value))
            self.CRm = np.append(self.CRm, CRm_i)
        elif self.mode == 'full':
            self.fitness = np.append(self.fitness, np.max(fitness_i))
            self.population_g = self.append_arr(self.population_g,
                                                population_g_i)
            if not len(self.m_proba):
                for key, value in m_proba_i.items():
                    self.m_proba[key] = np.array(value)
            else:
                for key, value in m_proba_i.items():
                    self.m_proba[key] = np.append(
                        self.m_proba[key], np.array(value))
            self.CRm = np.append(self.CRm, CRm_i)
        else:
            raise ValueError('the "mode" must be either "quick" or "full"')
        return self


class SaDE2005(DifferentialEvolution):
    '''Qin, Kai & Suganthan, Ponnuthurai. (2005). Self-adaptive differential evolution
    algorithm for numerical optimization. 2005 IEEE Congress on Evolutionary Computation,
    IEEE CEC 2005. Proceedings. 2. 1785-1791. 10.1109/CEC.2005.1554904'''

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
                 keep_history: Optional[str] = None):
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
                       'current_to_best_1': current_to_best_1}
        self.m_sets = dict(sorted(self.m_sets.items()))
        self.m_sets_keys = list(self.m_sets.keys())

        self.thefittest: TheFittest
        self.stats: StaticticSaDE
        self.m_learning_period = 50
        self.CR_update_timer = 5
        self.CR_m_learning_period = 25
        self.threshold = 0.1

        self.Fm = 0.5
        self.F_sigma = 0.3
        self.CR_sigma = 0.1

    def set_strategy(self, m_learning_period_param: Optional[int] = None,
                     CR_update_timer_param: Optional[int] = None,
                     CR_m_learning_period_param: Optional[int] = None,
                     threshold_params: Optional[float] = None):

        if m_learning_period_param is not None:
            self.m_learning_period = m_learning_period_param

        if CR_update_timer_param is not None:
            self.CR_update_timer = CR_update_timer_param

        if CR_m_learning_period_param is not None:
            self.CR_m_learning_period = CR_m_learning_period_param

        if threshold_params is not None:
            self.threshold = threshold_params

        return self

    def mutation_and_crossover(self, popuation_g, individ_g,
                               mutation_type, CR_i):
        F_i = np.random.normal(self.Fm, self.F_sigma)
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
        return (offspring_g, offspring_ph, offspring_fit, mask)

    def choice_operators(self, proba_dict):
        operators = list(proba_dict.keys())
        proba = list(proba_dict.values())
        return np.random.choice(list(operators), self.pop_size - 1, p=proba)

    def succeses_to_ns(self, succeses):
        return np.sum(succeses)

    def succeses_to_nf(self, succeses):
        return np.sum(~succeses)

    def update_ns_nf(self, operators, succeses, ns_i, nf_i):
        grouped = dict(zip(*numpy_group_by(group=succeses, by=operators)))

        for key in self.m_sets.keys():
            if key in grouped.keys():
                ns_i[key] += self.succeses_to_ns(grouped[key])
                nf_i[key] += self.succeses_to_nf(grouped[key])
        return ns_i, nf_i

    def update_proba(self, ns_i, nf_i):
        up = ns_i['rand_1']*(ns_i['current_to_best_1'] +
                             nf_i['current_to_best_1'])
        down = ns_i['current_to_best_1']*(ns_i['rand_1'] + nf_i['rand_1']) + up

        p1 = up/down
        return {'rand_1': p1, 'current_to_best_1': 1 - p1}

    def generate_CR(self, size, CRm):
        value = np.random.normal(CRm, self.CR_sigma, size)
        value = np.clip(value, 1e-6, 1)
        return value

    def if_period_ended(self, i, period):
        return i % period == 0 and i != 0

    def fit(self):
        z_m = len(self.m_sets)
        m_proba = dict(zip(self.m_sets_keys, np.full(z_m, 1/z_m)))
        CRm = 0.5
        CR_i = self.generate_CR(self.pop_size-1, CRm)

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
        if self.keep_history is not None:
            self.stats = StaticticSaDE(
                mode=self.keep_history).update(population_g,
                                               fitness,
                                               m_proba,
                                               CRm)

        ns = dict(zip(self.m_sets_keys, np.zeros(z_m, dtype=int)))
        nf = dict(zip(self.m_sets_keys, np.zeros(z_m, dtype=int)))
        CR_s_pool = np.array([], dtype=float)
        for i in range(self.iters-1):
            self.show_progress(i)
            if self.termitation_check(lastbest.no_increase):
                break
            else:
                m_operators = self.choice_operators(m_proba)

                partial_mut_and_cross = partial(self.mutation_and_crossover,
                                                population_g)
                mutant_cr_g = np.array(list(map(partial_mut_and_cross,
                                                population_g[:-1],
                                                m_operators, CR_i)))

                stack = self.evaluate_and_selection(mutant_cr_g,
                                                    population_g[:-1],
                                                    population_ph[:-1],
                                                    fitness[:-1])

                population_g[:-1] = stack[0]
                population_ph[:-1] = stack[1]
                fitness[:-1] = stack[2]

                population_g[-1], population_ph[-1], fitness[-1] = self.thefittest.get()
                argsort = np.argsort(fitness)
                population_g = population_g[argsort]
                population_ph = population_ph[argsort]
                fitness = fitness[argsort]

                self.thefittest.update(population_g, population_ph, fitness)
                lastbest.update(self.thefittest.fitness)
                if self.keep_history is not None:
                    self.stats.update(population_g,
                                      fitness,
                                      m_proba,
                                      CRm)
                successes = stack[-1]
                ns, nf = self.update_ns_nf(m_operators, successes, ns, nf)
                CR_s_pool = np.append(CR_s_pool, CR_i[successes])

                if self.if_period_ended(i, self.m_learning_period):
                    m_proba = self.update_proba(ns, nf)
                    ns = dict(zip(self.m_sets_keys, np.zeros(z_m, dtype=int)))
                    nf = dict(zip(self.m_sets_keys, np.zeros(z_m, dtype=int)))

                if self.if_period_ended(i, self.CR_update_timer):
                    CR_i = self.generate_CR(self.pop_size-1, CRm)
                    if self.if_period_ended(i, self.CR_m_learning_period):
                        if len(CR_s_pool):
                            CRm = np.mean(CR_s_pool)
                        CR_s_pool = np.array([], dtype=float)
        return self
