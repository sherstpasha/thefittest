from __future__ import annotations

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
        elitism: bool = True,
        init_population: Optional[
            Union[NDArray[Any], NDArray[np.byte], NDArray[np.float64]]
        ] = None,
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
        self._elitism: bool = elitism
        self._init_population: Optional[
            Union[NDArray[Any], NDArray[np.byte], NDArray[np.float64]]
        ] = init_population
        self._genotype_to_phenotype: Callable = genotype_to_phenotype
        self._no_increase_num: Optional[int] = no_increase_num
        self._show_progress_each: Optional[int] = show_progress_each
        self._keep_history: bool = keep_history

        self._sign: int = -1 if minimization else 1
        self._aim: Union[float, int] = self._get_aim(optimal_value, termination_error_value)
        self._calls: int = 0

        self._thefittest: TheFittest = TheFittest()
        self._stats: Statistics = Statistics()

        self._population_g_i: Union[NDArray[Any], NDArray[np.byte], NDArray[np.float64]]
        self._population_ph_i: NDArray
        self._fitness_i: NDArray[np.float64]

    def _get_init_population(self: EvolutionaryAlgorithm) -> None:
        return None

    def _get_aim(
        self: EvolutionaryAlgorithm,
        optimal_value: Optional[Union[float, int]],
        termination_error_value: Union[float, int],
    ) -> Union[float, int]:
        if optimal_value is not None:
            return self._sign * optimal_value - termination_error_value
        else:
            return np.inf

    def _get_fitness(
        self: EvolutionaryAlgorithm, population_ph: NDArray[Any]
    ) -> NDArray[np.float64]:
        self._calls += len(population_ph)
        return self._sign * self._fitness_function(population_ph)

    def _show_progress(self: EvolutionaryAlgorithm, current_iter: int) -> None:
        if self._show_progress_each is not None:
            cond_show_now = current_iter % self._show_progress_each == 0
            if cond_show_now:
                current_best = self._sign * self._thefittest._fitness
                print(f"{current_iter} iteration with fitness = {current_best}")

    def _termitation_check(self: EvolutionaryAlgorithm) -> bool:
        cond_aim = self._thefittest._fitness >= self._aim
        cond_no_increase = self._thefittest._no_update_counter == self._no_increase_num
        return bool(cond_aim or cond_no_increase)

    def _update_fittest(
        self: EvolutionaryAlgorithm,
        population_g: NDArray[Any],
        population_ph: NDArray[Any],
        fitness: NDArray[np.float64],
    ) -> None:
        self._thefittest._update(
            population_g=population_g, population_ph=population_ph, fitness=fitness
        )

    def _update_stats(self: EvolutionaryAlgorithm, **kwargs: Any) -> None:
        if self._keep_history:
            self._stats._update(kwargs)

    def _get_phenotype(self, popultion_g: NDArray[Any]) -> NDArray[Any]:
        return self._genotype_to_phenotype(popultion_g)

    def get_remains_calls(self: EvolutionaryAlgorithm) -> int:
        return (self._pop_size * self._iters) - self._calls

    def get_fittest(self: EvolutionaryAlgorithm) -> Dict:
        return self._thefittest.get()

    def get_stats(self: EvolutionaryAlgorithm) -> Statistics:
        return self._stats

    def _update_data(self: EvolutionaryAlgorithm) -> None:
        self._update_fittest(self._population_g_i, self._population_ph_i, self._fitness_i)
        self._update_stats(population_g=self._population_g_i, fitness_max=self._thefittest._fitness)

    def _adapt(self: EvolutionaryAlgorithm) -> None:
        return None

    def _from_population_g_to_fitness(self: EvolutionaryAlgorithm) -> None:
        self._population_ph_i = self._get_phenotype(self._population_g_i)
        self._fitness_i = self._get_fitness(self._population_ph_i)

        self._update_data()

        if self._elitism:
            (
                self._population_g_i[-1],
                self._population_ph_i[-1],
                self._fitness_i[-1],
            ) = self._thefittest.get().values()

    def _get_new_population(self: EvolutionaryAlgorithm) -> None:
        return None

    def fit(self: EvolutionaryAlgorithm) -> EvolutionaryAlgorithm:
        self._get_init_population()
        self._from_population_g_to_fitness()

        for i in range(self._iters - 1):
            self._show_progress(i)
            if self._termitation_check():
                break
            else:
                self._get_new_population()
                self._from_population_g_to_fitness()

        return self
