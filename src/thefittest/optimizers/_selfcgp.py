from functools import partial
from typing import Callable
from typing import Optional
from typing import List
from typing import Dict
import numpy as np
from ..tools.transformations import scale_data
from ..tools.transformations import rank_data
from ..base import Tree
from ..base import UniversalSet
from ..base import TheFittest
from ..base import LastBest
from ..base import Statistics
from ..tools.transformations import numpy_group_by
from ._geneticprogramming import GeneticProgramming
from ..tools.generators import half_and_half


class SelfCGP(GeneticProgramming):
    '''Semenkin, Eugene & Semenkina, Maria. (2012). Self-configuring genetic programming algorithm
    with modified uniform crossover. 1-6. 10.1109/CEC.2012.6256587. '''

    def __init__(self,
                 fitness_function: Callable,
                 genotype_to_phenotype: Callable,
                 uniset: UniversalSet,
                 iters: int,
                 pop_size: int,
                 optimal_value: Optional[float] = None,
                 termination_error_value: float = 0.,
                 no_increase_num: Optional[int] = None,
                 minimization: bool = False,
                 show_progress_each: Optional[int] = None,
                 keep_history: bool = False):
        GeneticProgramming.__init__(
            self,
            fitness_function=fitness_function,
            genotype_to_phenotype=genotype_to_phenotype,
            uniset=uniset,
            iters=iters,
            pop_size=pop_size,
            optimal_value=optimal_value,
            termination_error_value=termination_error_value,
            no_increase_num=no_increase_num,
            minimization=minimization,
            show_progress_each=show_progress_each,
            keep_history=keep_history)

        self._selection_set: dict
        self._crossover_set: dict
        self._mutation_set: dict
        self._K: int
        self._threshold: float

        self.set_strategy()

    def _choice_operators(self,
                          proba_dict: Dict) -> np.ndarray:
        operators = list(proba_dict.keys())
        proba = list(proba_dict.values())
        chosen_operator = np.random.choice(operators, self._pop_size, p=proba)
        return chosen_operator

    def _update_proba(self,
                      proba_dict: Dict,
                      operator: str) -> Dict:
        proba_dict[operator] += self._K/self._iters
        proba_value = np.array(list(proba_dict.values()))
        proba_value -= self._K/(len(proba_dict)*self._iters)
        proba_value = proba_value.clip(self._threshold, 1)
        proba_value = proba_value/proba_value.sum()
        new_proba_dict = dict(zip(proba_dict.keys(), proba_value))
        return new_proba_dict

    def _find_fittest_operator(self,
                               operators: np.ndarray,
                               fitness: np.ndarray) -> str:
        keys, groups = numpy_group_by(group=fitness, by=operators)
        mean_fit = np.array(list(map(np.mean, groups)))
        fittest_operator = keys[np.argmax(mean_fit)]
        return fittest_operator

    def _get_new_individ_g(self,
                           population_g: np.ndarray,
                           fitness_scale: np.ndarray,
                           fitness_rank: np.ndarray,
                           selection: str,
                           crossover: str,
                           mutation: str) -> Tree:

        selection_func, tour_size = self._selection_set[selection]
        crossover_func, quantity = self._crossover_set[crossover]
        mutation_func, proba_up, scale = self._mutation_set[mutation]

        selected_id = selection_func(fitness_scale, fitness_rank,
                                     tour_size, quantity)

        offspring_no_mutated = crossover_func(population_g[selected_id],
                                              fitness_scale[selected_id],
                                              fitness_rank[selected_id],
                                              self._max_level)

        proba = proba_up/len(offspring_no_mutated) if scale else proba_up

        offspring = mutation_func(offspring_no_mutated, self._uniset,
                                  proba, self._max_level)
        return offspring

    def set_strategy(self,
                     select_opers: List = ['proportional',
                                           'rank',
                                           'tournament_3',
                                           'tournament_5',
                                           'tournament_7'],
                     crossover_opers: List = ['standart',
                                              'one_point',
                                              'uniform_rank2'],
                     mutation_opers: List = ['weak_point',
                                             'average_point',
                                             'strong_point',
                                             'weak_grow',
                                             'average_grow',
                                             'strong_grow'],
                     tour_size_param:  int = 2,
                     initial_population: Optional[np.ndarray] = None,
                     elitism_param: bool = True,
                     parents_num_param: int = 7,
                     mutation_rate_param: float = 0.05,
                     K_param: float = 2,
                     threshold_param: float = 0.05,
                     max_level_param: int = 16,
                     init_level_param: int = 4) -> None:
        self._tour_size = tour_size_param
        self._initial_population = initial_population
        self._elitism = elitism_param
        self._parents_num = parents_num_param
        self._mutation_rate = mutation_rate_param
        self._K = K_param
        self._threshold = threshold_param
        self._max_level = max_level_param
        self._init_level = init_level_param

        self._update_pool()

        selection_set = {}
        for operator_name in select_opers:
            value = self._selection_pool[operator_name]
            selection_set[operator_name] = value
        self._selection_set = dict(sorted(selection_set.items()))

        crossover_set = {}
        for operator_name in crossover_opers:
            value = self._crossover_pool[operator_name]
            crossover_set[operator_name] = value
        self._crossover_set = dict(sorted(crossover_set.items()))

        mutation_set = {}
        for operator_name in mutation_opers:
            value = self._mutation_pool[operator_name]
            mutation_set[operator_name] = value
        self._mutation_set = dict(sorted(mutation_set.items()))

    def fit(self):

        z_selection = len(self._selection_set)
        z_crossover = len(self._crossover_set)
        z_mutation = len(self._mutation_set)

        s_proba = dict(zip(list(self._selection_set.keys()),
                           np.full(z_selection, 1/z_selection)))
        if 'empty' in self._crossover_set.keys():
            c_proba = dict(zip(list(self._crossover_set.keys()),
                           np.full(z_crossover, 0.9/(z_crossover-1))))
            c_proba['empty'] = 0.1
        else:
            c_proba = dict(zip(list(self._crossover_set.keys()),
                               np.full(z_crossover, 1/z_crossover)))
        m_proba = dict(zip(list(self._mutation_set.keys()),
                       np.full(z_mutation, 1/z_mutation)))

        if self._initial_population is None:
            population_g = half_and_half(
                self._pop_size, self._uniset, self._init_level)
        else:
            population_g = self._initial_population

        for i in range(self._iters):
            population_ph = self._get_phenotype(population_g)
            fitness = self._evaluate(population_ph)

            self._update_fittest(population_g, population_ph, fitness)
            self._update_stats({'individ_max': self._thefittest._genotype.copy(),
                                'fitness_max': self._thefittest._fitness,
                                's_proba': s_proba.copy(),
                                'c_proba': c_proba.copy(),
                                'm_proba': m_proba.copy()})
            if self._elitism:
                population_g[-1], population_ph[-1], fitness[-1] = self._thefittest.get()
            fitness_scale = scale_data(fitness)

            if i > 0:
                s_fittest_oper = self._find_fittest_operator(
                    s_operators, fitness_scale)
                s_proba = self._update_proba(s_proba, s_fittest_oper)

                c_fittest_oper = self._find_fittest_operator(
                    c_operators, fitness_scale)
                c_proba = self._update_proba(c_proba, c_fittest_oper)

                m_fittest_oper = self._find_fittest_operator(
                    m_operators, fitness_scale)
                m_proba = self._update_proba(m_proba, m_fittest_oper)

            self._show_progress(i)
            if self._termitation_check():
                break
            else:
                s_operators = self._choice_operators(s_proba)
                c_operators = self._choice_operators(c_proba)
                m_operators = self._choice_operators(m_proba)

                get_new_individ_g = partial(self._get_new_individ_g,
                                            population_g,
                                            fitness_scale,
                                            rank_data(fitness))
                map_ = map(get_new_individ_g,
                           s_operators, c_operators, m_operators)
                population_g = np.array(list(map_), dtype=object)

        return self
