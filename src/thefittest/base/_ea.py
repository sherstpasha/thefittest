from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..tools import donothing


class TheFittest:
    def __init__(self) -> None:
        self._genotype: Any
        self._phenotype: Any
        self._fitness: float = -np.inf
        self._no_update_counter: int = 0

    def _update(
        self,
        population_g: NDArray[Any],
        population_ph: NDArray[Any],
        fitness: NDArray[np.float64],
    ) -> None:
        temp_best_id = np.argmax(fitness)
        temp_best_fitness = fitness[temp_best_id]
        if temp_best_fitness > self._fitness:
            self._replace(
                new_genotype=population_g[temp_best_id],
                new_phenotype=population_ph[temp_best_id],
                new_fitness=temp_best_fitness,
            )
            self._no_update_counter = 0
        else:
            self._no_update_counter += 1

    def _replace(self, new_genotype: Any, new_phenotype: Any, new_fitness: float) -> None:
        self._genotype = new_genotype.copy()
        self._phenotype = new_phenotype.copy()
        self._fitness = new_fitness

    def get(self) -> Dict:
        return {
            "genotype": self._genotype.copy(),
            "phenotype": self._phenotype.copy(),
            "fitness": self._fitness,
        }


class Statistics(dict):
    def _update(self, arg: Dict[str, Any]) -> None:
        for key, value in arg.items():
            try:
                value_to_append = value.copy()
            except AttributeError:
                value_to_append = value
            if key not in self.keys():
                self[key] = [value_to_append]
            else:
                self[key].append(value_to_append)


class EvolutionaryAlgorithm:
    def __init__(
        self,
        fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]],
        iters: int,
        pop_size: int,
        genotype_to_phenotype: Callable[[NDArray[Any]], NDArray[Any]] = donothing,
        optimal_value: Optional[Union[float, int]] = None,
        termination_error_value: Union[float, int] = 0.0,
        no_increase_num: Optional[int] = None,
        minimization: bool = False,
        show_progress_each: Optional[int] = None,
        keep_history: bool = False,
    ):
        self._fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]] = fitness_function
        self._iters: int = iters
        self._pop_size: int = pop_size
        self._genotype_to_phenotype: Callable = genotype_to_phenotype
        self._no_increase_num: Optional[int] = no_increase_num
        self._show_progress_each: Optional[int] = show_progress_each
        self._keep_history: bool = keep_history

        self._sign: int = -1 if minimization else 1
        self._aim: Union[float, int] = self._get_aim(optimal_value, termination_error_value)
        self._calls: int = 0

        self._thefittest: TheFittest = TheFittest()
        self._stats: Statistics = Statistics()

    def _get_aim(
        self, optimal_value: Optional[Union[float, int]], termination_error_value: Union[float, int]
    ) -> Union[float, int]:
        if optimal_value is not None:
            return self._sign * optimal_value - termination_error_value
        else:
            return np.inf

    def _get_fitness(self, population_ph: NDArray[Any]) -> NDArray[np.float64]:
        self._calls += len(population_ph)
        return self._sign * self._fitness_function(population_ph)

    def _show_progress(self, current_iter: int) -> None:
        if self._show_progress_each is not None:
            cond_show_now = current_iter % self._show_progress_each == 0
            if cond_show_now:
                current_best = self._sign * self._thefittest._fitness
                print(f"{current_iter} iteration with fitness = {current_best}")

    def _termitation_check(self) -> bool:
        cond_aim = self._thefittest._fitness >= self._aim
        cond_no_increase = self._thefittest._no_update_counter == self._no_increase_num
        return bool(cond_aim or cond_no_increase)

    def _update_fittest(
        self,
        population_g: NDArray[Any],
        population_ph: NDArray[Any],
        fitness: NDArray[np.float64],
    ) -> None:
        self._thefittest._update(
            population_g=population_g, population_ph=population_ph, fitness=fitness
        )

    def _update_stats(self, **kwargs: Any) -> None:
        if self._keep_history:
            self._stats._update(kwargs)

    def _get_phenotype(self, popultion_g: NDArray[Any]) -> NDArray[Any]:
        return self._genotype_to_phenotype(popultion_g)

    def get_remains_calls(self) -> int:
        return (self._pop_size * self._iters) - self._calls

    def get_fittest(self) -> TheFittest:
        return self._thefittest

    def get_stats(self) -> Statistics:
        return self._stats

    def clear(self) -> None:
        self._calls = 0
        self._thefittest = TheFittest()
        self._stats = Statistics()
