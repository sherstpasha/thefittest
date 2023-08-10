from functools import partial
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Dict
from typing import Union
import random
import numpy as np
from numpy.typing import NDArray
from ._geneticalgorithm import GeneticAlgorithm
from ..tools.transformations import scale_data
from ..tools.transformations import rank_data
from ..tools.transformations import numpy_group_by
from ..tools.random import binary_string_population
from ..tools.operators import flip_mutation
from ..tools.operators import proportional_selection
from ..tools.operators import rank_selection
from ..tools.operators import tournament_selection


class SelfAGA(GeneticAlgorithm):
    def __init__(self,
                 fitness_function: Callable,
                 genotype_to_phenotype: Callable,
                 iters: int,
                 pop_size: int,
                 str_len: int,
                 optimal_value: Optional[float] = None,
                 termination_error_value: float = 0.,
                 no_increase_num: Optional[int] = None,
                 minimization: bool = False,
                 show_progress_each: Optional[int] = None,
                 keep_history: bool = False):
        GeneticAlgorithm.__init__(
            self,
            fitness_function=fitness_function,
            genotype_to_phenotype=genotype_to_phenotype,
            iters=iters,
            pop_size=pop_size,
            str_len=str_len,
            optimal_value=optimal_value,
            termination_error_value=termination_error_value,
            no_increase_num=no_increase_num,
            minimization=minimization,
            show_progress_each=show_progress_each,
            keep_history=keep_history)

        self._selection_set: dict
        self._crossover_set: dict
        self._adapting_selection_pool: dict
        self._specified_adapting_selection: tuple
        self._p_operator_param: float
        self._p_mutate_param: float

        self._min_mutation_rate = 1/self._str_len
        self._max_mutation_rate = min(1, 3/self._str_len)

    def _get_new_individ_g(self,
                           population_g: np.ndarray,
                           fitness_scale: np.ndarray,
                           fitness_rank: np.ndarray,
                           selection: str,
                           crossover: str,
                           proba: float) -> np.ndarray:
        selection_func, tour_size = self._selection_set[selection]
        crossover_func, quantity = self._crossover_set[crossover]

        selected_id = selection_func(fitness_scale, fitness_rank,
                                     np.int64(tour_size), np.int64(quantity))
        offspring_no_mutated = crossover_func(population_g[selected_id],
                                              fitness_scale[selected_id],
                                              fitness_rank[selected_id])
        offspring = flip_mutation(offspring_no_mutated, np.float64(proba))
        return offspring

    def _init_operators_uniform(self,
                                oper_sets: Dict):
        to_return = []
        operators_count = len(oper_sets.keys())
        quantity_operators = int(self._pop_size/operators_count)

        for operator in oper_sets.keys():
            to_return.extend([operator]*quantity_operators)

        quantity_remain = self._pop_size - len(to_return)
        if quantity_remain > 0:
            to_return.extend(random.choices(list(oper_sets.keys()), k = quantity_remain))

        random.shuffle(to_return)
        
        return np.array(to_return)

    def _init_parameter_uniform_float(self, low: np.float64, high: np.float) -> NDArray[np.float64]:
        return np.linspace(low, high, self._pop_size)
        # return np.random.uniform(low=low, high=high, size=self._pop_size)

    def _update_pool(self):
        GeneticAlgorithm._update_pool(self)
        self.adapting_selection_pool = {'proportional': (proportional_selection, 0),
                                        'rank': (rank_selection, 0),
                                        'tournament_k': (tournament_selection, self._tour_size),
                                        'tournament_3': (tournament_selection, 3),
                                        'tournament_5': (tournament_selection, 5),
                                        'tournament_7': (tournament_selection, 7)}

    def set_strategy(self,
                     selection_opers: Tuple = ('proportional',
                                               'rank',
                                               'tournament_3',
                                               'tournament_5',
                                               'tournament_7'),
                     crossover_opers: Tuple = ('empty',
                                               'one_point',
                                               'two_point',
                                               'uniform2',
                                               'uniform7',
                                               'uniform_prop2',
                                               'uniform_prop7',
                                               'uniform_rank2',
                                               'uniform_rank7',
                                               'uniform_tour3',
                                               'uniform_tour7'),
                     tour_size_param: int = 2,
                     initial_population: Optional[np.ndarray] = None,
                     elitism_param: bool = True,
                     parents_num_param: int = 7,
                     p_operator_param: Optional[float] = None,
                     p_mutate_param: Optional[float] = None,
                     adapting_selection_operator: str = 'rank') -> None:
        '''
        - selection_oper: must be a Tuple of:
            'proportional', 'rank', 'tournament_k', 'tournament_3', 'tournament_5', 'tournament_7'
        - crossover oper: must be a Tuple of:
            'empty', 'one_point', 'two_point', 'uniform2', 'uniform7', 'uniformk', 'uniform_prop2',
            'uniform_prop7', 'uniform_propk', 'uniform_rank2', 'uniform_rank7', 'uniform_rankk',
            'uniform_tour3', 'uniform_tour7', 'uniform_tourk'
        - adapting_selection_operator: must be a Str:
            'proportional', 'rank', 'tournament_3', 'tournament_5', 'tournament_7'
        '''
        self._mutation_rate = 0.
        self._tour_size = tour_size_param
        self._initial_population = initial_population
        self._elitism = elitism_param
        self._parents_num = parents_num_param

        if p_operator_param is None:
           self._p_operator_param = 1/self._pop_size
        else:
            self._p_operator_param = p_operator_param
        
        if p_mutate_param is None:
           self._p_mutate_param = 1/self._pop_size
        else:
            self._p_mutate_param = p_mutate_param

        self._update_pool()

        selection_set = {}
        for operator_name in selection_opers:
            value = self._selection_pool[operator_name]
            selection_set[operator_name] = value
        self._selection_set = dict(sorted(selection_set.items()))

        crossover_set = {}
        for operator_name in crossover_opers:
            value = self._crossover_pool[operator_name]
            crossover_set[operator_name] = value
        self._crossover_set = dict(sorted(crossover_set.items()))

        self._specified_adapting_selection =\
            self.adapting_selection_pool[adapting_selection_operator]

    def _update_mutation_rate(self, mutation_rate: NDArray[np.float64]):
        mask = np.random.random(size=len(mutation_rate)) < self._p_mutate_param
        mutation_rate[mask] = np.random.uniform(low=self._min_mutation_rate,
                                                high=self._max_mutation_rate,
                                                size=np.sum(mask))
        return mutation_rate

    def _update_operators(self,
                          operators: NDArray,
                          oper_set: Dict) -> NDArray:
        mask = np.random.random(size=len(operators)) < self._p_operator_param
        operators[mask] = np.random.choice(
            list(oper_set.keys()), np.sum(mask))
        return operators

    def _choice_operators_params(self,
                                 individs: NDArray,
                                 fitness_scale: NDArray,
                                 fitness_rank: NDArray) -> NDArray:

        selection_func, tour_size = self._specified_adapting_selection
        selected_id = selection_func(fitness_scale, fitness_rank,
                                     np.int64(tour_size), self._pop_size)

        new_individs = individs[selected_id]
        return new_individs

    def fit(self):
        s_operators = self._init_operators_uniform(self._selection_set)
        c_operators = self._init_operators_uniform(self._crossover_set)
        mutation_rate = self._init_parameter_uniform_float(
            self._min_mutation_rate, self._max_mutation_rate)

        if self._initial_population is None:
            population_g = binary_string_population(
                self._pop_size, self._str_len)
        else:
            population_g = self._initial_population

        for i in range(self._iters):
            population_ph = self._get_phenotype(population_g)
            fitness = self._evaluate(population_ph)

            self._update_fittest(population_g, population_ph, fitness)
            self._update_stats({'population_g': population_g.copy(),
                                'fitness_max': self._thefittest._fitness,
                                's_opers': s_operators.copy(),
                                'c_opers': c_operators.copy(),
                                'm_proba': mutation_rate.copy()})
            if self._elitism:
                population_g[-1], population_ph[-1], fitness[-1] = self._thefittest.get()
            fitness_scale = scale_data(fitness)
            fitness_rank = rank_data(fitness)
            if i > 0:
                pass
                s_operators = self._choice_operators_params(
                    s_operators, fitness_scale, fitness_rank)
                c_operators = self._choice_operators_params(
                    c_operators, fitness_scale, fitness_rank)
                mutation_rate = self._choice_operators_params(
                    mutation_rate, fitness_scale, fitness_rank)

            self._show_progress(i)
            if self._termitation_check():
                break
            else:
                s_operators = self._update_operators(s_operators,
                                                     self._selection_set)
                c_operators = self._update_operators(c_operators,
                                                     self._crossover_set)
                mutation_rate = self._update_mutation_rate(mutation_rate)

                get_new_individ_g = partial(self._get_new_individ_g,
                                            population_g,
                                            fitness_scale,
                                            fitness_rank)
                map_ = map(get_new_individ_g,
                           s_operators, c_operators, mutation_rate)
                population_g = np.array(list(map_), dtype=np.byte)
