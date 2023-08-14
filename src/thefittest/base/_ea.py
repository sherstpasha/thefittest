from typing import Any
from typing import Dict
from typing import Callable
from typing import Optional
from typing import Union
from numpy.typing import NDArray
import numpy as np
from ..tools import donothing


class TheFittest:
    def __init__(self):
        self._genotype: Any
        self._phenotype: Any
        self._fitness: Union[float, np.float64] = -np.inf
        self._no_update_counter: int = 0

    def _update(self,
                population_g: NDArray[Any],
                population_ph: NDArray[Any],
                fitness: NDArray[np.float64]) -> None:
        temp_best_id = np.argmax(fitness)
        temp_best_fitness = fitness[temp_best_id]
        if temp_best_fitness > self._fitness:
            self._replace(new_genotype=population_g[temp_best_id],
                          new_phenotype=population_ph[temp_best_id],
                          new_fitness=temp_best_fitness)
            self._no_update_counter = 0
        else:
            self._no_update_counter += 1

    def _replace(self,
                 new_genotype: Any,
                 new_phenotype: Any,
                 new_fitness: np.float64) -> None:
        self._genotype = new_genotype.copy()
        self._phenotype = new_phenotype.copy()
        self._fitness = new_fitness

    def get(self) -> Dict:
        to_return = {'genotype': self._genotype.copy(),
                     'phenotype': self._phenotype.copy(),
                     'fitness': self._fitness}
        return to_return


class Statistics(dict):
    def _update(self,
                arg: Dict):
        for key, value in arg.items():
            if key not in self.keys():
                self[key] = [value.copy()]
            else:
                self[key].append(value.copy())


class EvolutionaryAlgorithm:
    def __init__(self,
                 fitness_function: Callable,
                 iters: int,
                 pop_size: int,
                 genotype_to_phenotype: Callable = donothing,
                 optimal_value: Optional[float] = None,
                 termination_error_value: float = 0.,
                 no_increase_num: Optional[int] = None,
                 minimization: bool = False,
                 show_progress_each: Optional[int] = None,
                 keep_history: bool = False):
        self._fitness_function = fitness_function
        self._iters = iters
        self._pop_size = pop_size
        self._genotype_to_phenotype = genotype_to_phenotype
        self._no_increase_num = no_increase_num
        self._show_progress_each = show_progress_each
        self._keep_history = keep_history

        self._sign = -1 if minimization else 1
        self._get_aim(optimal_value, termination_error_value)
        self._calls = 0

        self._thefittest: Optional[TheFittest] = None
        self._stats: Optional[Statistics] = None

    def _get_aim(self,
                 optimal_value: Optional[float],
                 termination_error_value: float) -> None:
        if optimal_value is not None:
            self._aim = self._sign*optimal_value - termination_error_value
        else:
            self._aim = np.inf

    def _get_fitness(self,
                     population_ph: NDArray[Any]) -> NDArray[Any]:
        self._calls += len(population_ph)
        return self._sign*self._fitness_function(population_ph)

    def _show_progress(self,
                       iter_number: int) -> None:
        cond_show_switch = self._show_progress_each is not None
        if cond_show_switch:
            cond_show_now = iter_number % self._show_progress_each == 0
            if cond_show_now:
                current_best = self._sign*self._thefittest._fitness
                print(f'{iter_number} iteration with fitness = {current_best}', self._thefittest._no_update_counter)

    def _termitation_check(self):
        cond_aim = self._thefittest._fitness >= self._aim
        cond_no_increase =\
            self._thefittest._no_update_counter == self._no_increase_num
        return cond_aim or cond_no_increase

    def _update_fittest(self,
                        population_g: NDArray[Any],
                        population_ph: NDArray[Any],
                        fitness: NDArray[np.float64]) -> None:
        if self._thefittest is None:
            self._thefittest = TheFittest()

        self._thefittest._update(population_g=population_g,
                                 population_ph=population_ph,
                                 fitness=fitness)

    def _update_stats(self,
                      **kwargs) -> None:
        if self._keep_history:
            if self._stats is None:
                self._stats = Statistics()
            self._stats._update(kwargs)

    def _get_phenotype(self,
                       popultion_g: NDArray[Any]) -> NDArray[Any]:
        return self._genotype_to_phenotype(popultion_g)

    def get_remains_calls(self):
        return (self._pop_size*self._iters) - self._calls

    def get_fittest(self) -> TheFittest:
        return self._thefittest

    def get_stats(self) -> Statistics:
        return self._stats
