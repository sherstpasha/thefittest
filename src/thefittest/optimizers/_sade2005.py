from functools import partial
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Dict
import numpy as np
from ._differentialevolution import DifferentialEvolution
from ..tools.operators import binomial
from ..tools.operators import rand_1
from ..tools.operators import current_to_best_1
from ..tools.transformations import numpy_group_by
from ..tools.generators import float_population


class SaDE2005(DifferentialEvolution):
    '''Qin, Kai & Suganthan, Ponnuthurai. (2005). Self-adaptive differential evolution
    algorithm for numerical optimization. 2005 IEEE Congress on Evolutionary Computation,
    IEEE CEC 2005. Proceedings. 2. 1785-1791. 10.1109/CEC.2005.1554904'''

    def __init__(self,
                 fitness_function: Callable,
                 genotype_to_phenotype: Callable,
                 iters: int,
                 pop_size: int,
                 left: np.ndarray,
                 right: np.ndarray,
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

        # self.m_sets_keys = list(self.m_sets.keys())
        self._m_learning_period: int
        self._CR_update_timer: int
        self._CR_m_learning_period: int
        self._threshold: int

        self._Fm = 0.5
        self._F_sigma = 0.3
        self._CR_sigma = 0.1

        self.set_strategy()

    def _update_pool(self):
        self._mutation_pool = {'rand_1': rand_1,
                               'current_to_best_1': current_to_best_1}

    def _mutation_and_crossover(self,
                                popuation_g: np.ndarray,
                                individ_g: np.ndarray,
                                mutation: str,
                                CR: float) -> np.ndarray:
        F = np.random.normal(self._Fm, self._F_sigma)
        mutant = self._mutation_pool[mutation](individ_g, popuation_g, F)

        mutant_cr_g = binomial(individ_g, mutant, CR)
        mutant_cr_g = self._bounds_control(mutant_cr_g)
        return mutant_cr_g

    def _choice_operators(self,
                          proba_dict: Dict) -> np.ndarray:
        operators = list(proba_dict.keys())
        proba = list(proba_dict.values())
        chosen_operator = np.random.choice(operators, self._pop_size, p=proba)
        return chosen_operator

    def _succeses_to_ns(self,
                        succeses: np.ndarray) -> int:
        return np.sum(succeses)

    def _succeses_to_nf(self,
                        succeses: np.ndarray) -> int:
        return np.sum(~succeses)

    def _update_ns_nf(self,
                      operators: str,
                      succeses: np.ndarray,
                      ns_i: Dict,
                      nf_i: Dict) -> Tuple:
        grouped = dict(zip(*numpy_group_by(group=succeses, by=operators)))

        for key in self._mutation_pool.keys():
            if key in grouped.keys():
                ns_i[key] += self._succeses_to_ns(grouped[key])
                nf_i[key] += self._succeses_to_nf(grouped[key])
        return ns_i, nf_i

    def _update_proba(self,
                      ns_i: Dict,
                      nf_i:  Dict) -> Dict:
        up = ns_i['rand_1']*(ns_i['current_to_best_1'] +
                             nf_i['current_to_best_1'])
        down = ns_i['current_to_best_1']*(ns_i['rand_1'] + nf_i['rand_1']) + up

        p1 = up/down
        new_m_proba = {'rand_1': p1, 'current_to_best_1': 1 - p1}
        return new_m_proba

    def _generate_CR(self,
                     CRm: float) -> np.ndarray:
        value = np.random.normal(CRm, self._CR_sigma, self._pop_size)
        value = np.clip(value, 1e-6, 1)
        return value

    def _if_period_ended(self,
                         i: int,
                         period: int) -> bool:
        return i % period == 0 and i != 0

    def set_strategy(self,
                     m_learning_period_param: int = 50,
                     CR_update_timer_param: int = 5,
                     CR_m_learning_period_param: int = 25,
                     threshold_params: float = 0.1,
                     Fm_param: float = 0.5,
                     F_sigma_param: float = 0.3,
                     CR_sigma_param: float = 0.1,
                     elitism_param: bool = True,
                     initial_population: Optional[np.ndarray] = None) -> None:
        self._update_pool()
        self._m_learning_period = m_learning_period_param
        self._CR_update_timer = CR_update_timer_param
        self._CR_m_learning_period = CR_m_learning_period_param
        self._threshold = threshold_params
        self._Fm = Fm_param
        self._F_sigma = F_sigma_param
        self._CR_sigma = CR_sigma_param
        self._elitism = elitism_param
        self._initial_population = initial_population
        return self

    def fit(self):
        if self._initial_population is None:
            population_g = float_population(
                self._pop_size, self._left, self._right)
        else:
            population_g = self._initial_population

        z_mutation = len(self._mutation_pool)
        m_proba = dict(zip(self._mutation_pool.keys(),
                           np.full(z_mutation, 1/z_mutation)))
        CRm = 0.5
        CR_i = np.full(self._pop_size, CRm)

        ns = dict(zip(self._mutation_pool.keys(),
                  np.zeros(z_mutation, dtype=int)))
        nf = dict(zip(self._mutation_pool.keys(),
                  np.zeros(z_mutation, dtype=int)))
        CR_s_pool = np.array([], dtype=float)

        population_ph = self._get_phenotype(population_g)
        fitness = self._evaluate(population_ph)
        argsort = np.argsort(fitness)
        population_g = population_g[argsort]
        population_ph = population_ph[argsort]
        fitness = fitness[argsort]

        for i in range(self._iters):
            self._update_fittest(population_g, population_ph, fitness)
            self._update_stats({'population_g': population_g,
                                'fitness_max': self._thefittest._fitness,
                                'm_proba': m_proba.copy(),
                                'CRm': CRm})
            if self._elitism:
                population_g[-1], population_ph[-1], fitness[-1] = self._thefittest.get()

            if self._if_period_ended(i, self._m_learning_period):
                m_proba = self._update_proba(ns, nf)
                ns = dict(zip(self._mutation_pool.keys(),
                              np.zeros(z_mutation, dtype=int)))
                nf = dict(zip(self._mutation_pool.keys(),
                              np.zeros(z_mutation, dtype=int)))

            if self._if_period_ended(i, self._CR_update_timer):
                CR_i = self._generate_CR(CRm)
                if self._if_period_ended(i, self._CR_m_learning_period):
                    if len(CR_s_pool):
                        CRm = np.mean(CR_s_pool)
                    CR_s_pool = np.array([], dtype=float)

            self._show_progress(i)
            if self._termitation_check():
                break
            else:
                m_operators = self._choice_operators(m_proba)

                mutation_and_crossover = partial(self._mutation_and_crossover,
                                                 population_g)
                mutant_cr_g = np.array(list(map(mutation_and_crossover,
                                                population_g,  m_operators, CR_i)))

                stack = self._evaluate_and_selection(mutant_cr_g,
                                                     population_g,
                                                     population_ph,
                                                     fitness)
                population_g, population_ph, fitness, succeses = stack

                ns, nf = self._update_ns_nf(m_operators, succeses, ns, nf)
                CR_s_pool = np.append(CR_s_pool, CR_i[succeses])

        return self

    def fit(self):
        if self._initial_population is None:
            population_g = float_population(
                self._pop_size, self._left, self._right)
        else:
            population_g = self._initial_population

        z_mutation = len(self._mutation_pool)
        m_proba = dict(zip(self._mutation_pool.keys(),
                           np.full(z_mutation, 1/z_mutation)))
        CRm = 0.5
        CR_i = self._generate_CR( CRm)

        population_ph = self._get_phenotype(population_g)
        fitness = self._evaluate(population_ph)

        argsort = np.argsort(fitness)
        population_g = population_g[argsort]
        population_ph = population_ph[argsort]
        fitness = fitness[argsort]

        ns = dict(zip(self._mutation_pool.keys(),
                  np.zeros(z_mutation, dtype=int)))
        nf = dict(zip(self._mutation_pool.keys(),
                  np.zeros(z_mutation, dtype=int)))
        CR_s_pool = np.array([], dtype=float)
        for i in range(self._iters-1):
            self._update_fittest(population_g, population_ph, fitness)
            self._update_stats({'population_g': population_g,
                                'fitness_max': self._thefittest._fitness,
                                'm_proba': m_proba.copy(),
                                'CRm': CRm})
            self._show_progress(i)
            if self._termitation_check():
                break
            else:
                m_operators = self._choice_operators(m_proba)

                mutation_and_crossover = partial(self._mutation_and_crossover,
                                                 population_g)
                mutant_cr_g = np.array(list(map(mutation_and_crossover,
                                                population_g,  m_operators, CR_i)))

                stack = self._evaluate_and_selection(mutant_cr_g,
                                                    population_g,
                                                    population_ph,
                                                    fitness)

                population_g, population_ph, fitness, succeses = stack

                if self._elitism:
                    population_g[-1], population_ph[-1], fitness[-1] = self._thefittest.get()

                argsort = np.argsort(fitness)
                population_g = population_g[argsort]
                population_ph = population_ph[argsort]
                fitness = fitness[argsort]

                ns, nf = self._update_ns_nf(m_operators, succeses, ns, nf)
                CR_s_pool = np.append(CR_s_pool, CR_i[succeses])

                if self._if_period_ended(i, self._m_learning_period):
                    m_proba = self._update_proba(ns, nf)
                    ns = dict(zip(self._mutation_pool.keys(),
                                np.zeros(z_mutation, dtype=int)))
                    nf = dict(zip(self._mutation_pool.keys(),
                                np.zeros(z_mutation, dtype=int)))

                if self._if_period_ended(i, self._CR_update_timer):
                    CR_i = self._generate_CR(CRm)
                    if self._if_period_ended(i, self._CR_m_learning_period):
                        if len(CR_s_pool):
                            CRm = np.mean(CR_s_pool)
                        CR_s_pool = np.array([], dtype=float)
        return self
