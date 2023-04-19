from typing import Any
from typing import Tuple
from typing import Dict
from typing import Callable
from typing import Optional
import numpy as np


class LastBest:
    def __init__(self) -> None:
        self._value = np.nan
        self._no_increase_counter = 0

    def _update(self,
                current_value: float):
        if self._value == current_value:
            self._no_increase_counter += 1
        else:
            self._no_increase_counter = 0
            self._value = current_value
        return self


class TheFittest:
    def __init__(self):
        self._genotype: Any
        self._phenotype: Any
        self._fitness = -np.inf

    def _update(self,
                population_g: np.ndarray,
                population_ph: np.ndarray,
                fitness: np.ndarray):
        temp_best_id = np.argmax(fitness)
        temp_best_fitness = fitness[temp_best_id].copy()
        if temp_best_fitness > self._fitness:
            self._genotype = population_g[temp_best_id].copy()
            self._phenotype = population_ph[temp_best_id].copy()
            self._fitness = temp_best_fitness.copy()
        return self

    def get(self) -> Tuple:
        to_return = (self._genotype.copy(),
                     self._phenotype.copy(),
                     self._fitness.copy())
        return to_return


class Statistics(dict):
    def _update(self,
                arg: Dict):
        for key, value in arg.items():
            if key not in self.keys():
                self[key] = [value]
            else:
                self[key].append(value)
        return self


class EvolutionaryAlgorithm:
    def __init__(self,
                 fitness_function: Callable,
                 genotype_to_phenotype: Callable,
                 iters: int,
                 pop_size: int,
                 optimal_value: Optional[float] = None,
                 termination_error_value: float = 0.,
                 no_increase_num: Optional[int] = None,
                 minimization: bool = False,
                 show_progress_each: Optional[int] = None,
                 keep_history: bool = False):
        self._fitness_function = fitness_function
        self._genotype_to_phenotype = genotype_to_phenotype
        self._iters = iters
        self._pop_size = pop_size
        self._no_increase_num = no_increase_num
        self._show_progress_each = show_progress_each
        self._keep_history = keep_history

        self._sign = -1 if minimization else 1

        if optimal_value is not None:
            self._aim = self._sign*optimal_value - termination_error_value
        else:
            self._aim = np.inf

        self._calls = 0

        self._thefittest: Optional[TheFittest] = None
        self._lastbest: Optional[LastBest] = None
        self._stats: Optional[Statistics] = None

    def _evaluate(self,
                  population_ph: np.ndarray) -> np.ndarray:
        self._calls += len(population_ph)
        return self._sign*self._fitness_function(population_ph)

    def _show_progress(self, iter_number: int) -> None:
        cond_show_switch = self._show_progress_each is not None
        cond_show_now = iter_number % self._show_progress_each == 0
        if cond_show_switch and cond_show_now:
            print(
                f'{iter_number} iteration with fitness = {self._thefittest._fitness}')

    def _termitation_check(self):
        cond_aim = self._thefittest._fitness >= self._aim
        cond_no_increase = self._lastbest._no_increase_counter == self._no_increase_num
        return cond_aim or cond_no_increase

    def _update_fittest(self,
                       population_g: np.ndarray,
                       population_ph: np.ndarray,
                       fitness: np.ndarray) -> None:
        if self._thefittest is None:
            self._thefittest = TheFittest()._update(population_g,
                                                    population_ph,
                                                    fitness)
            self._lastbest = LastBest()._update(self._thefittest._fitness)

        else:
            self._thefittest._update(population_g, population_ph, fitness)
            self._lastbest._update(self._thefittest._fitness)

    def _update_stats(self,
                     statistic_args: Dict) -> None:
        if self._keep_history:
            if self._stats is None:
                self._stats = Statistics()._update(statistic_args)
            else:
                self._stats._update(statistic_args)

    def _get_phenotype(self, popultion_g: np.ndarray) -> np.ndarray:
        return self._genotype_to_phenotype(popultion_g)

    def get_remains_calls(self):
        return (self._pop_size + (self._iters-1)*(self._pop_size-1)) - self._calls

    def get_fittest(self) -> TheFittest:
        return self._thefittest

    def get_stats(self) -> Statistics:
        return self._stats
