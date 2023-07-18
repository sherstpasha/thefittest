from functools import partial
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Dict
from typing import Union
import numpy as np
from numpy.typing import NDArray
from ._geneticalgorithm import GeneticAlgorithm
from ..tools.transformations import scale_data
from ..tools.transformations import rank_data
from ..tools.transformations import numpy_group_by
from ..tools.random import binary_string_population
from ..tools.operators import flip_mutation


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

        self.set_strategy()

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

    def init_operators_uniform(self,
                               oper_sets: Dict):
        return np.random.choice(list(oper_sets.keys()), self.pop_size)

    def init_parameter_uniform_float(self, low: np.float64, high: np.float) -> NDArray[np.float64]:
        return np.random.uniform(low=low, high=high, size=self._pop_size)

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
                     mutation_rate_param: float = 0.05,
                     K_param: float = 2,
                     threshold_param: float = 0.05) -> None:
        pass

    def fit(self):
        s_operators = self.init_operators_uniform(self._selection_set)
        c_operators = self.init_operators_uniform(self._crossover_set)
        mutation_rate_param = self.init_parameter_uniform_float(
            1/(3*self.str_len), min(1, 3/self.str_len))

        if self._initial_population is None:
            population_g = binary_string_population(
                self._pop_size, self._str_len)
        else:
            population_g = self._initial_population
