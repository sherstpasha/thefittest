from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from joblib import Parallel
from joblib import cpu_count
from joblib import delayed

import numpy as np
from numpy.typing import NDArray

from ..utils.random import check_random_state


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
        genotype_to_phenotype: Optional[Callable[[NDArray[Any]], NDArray[Any]]] = None,
        optimal_value: Optional[Union[float, int]] = None,
        termination_error_value: Union[float, int] = 0.0,
        no_increase_num: Optional[int] = None,
        minimization: bool = False,
        show_progress_each: Optional[int] = None,
        keep_history: bool = False,
        n_jobs: int = 1,
        fitness_function_args: Optional[Dict] = None,
        genotype_to_phenotype_args: Optional[Dict] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        on_generation: Optional[Callable] = None,
    ):
        self._fitness_function: Callable[[NDArray[Any]], NDArray[np.float64]] = fitness_function
        self._iters: int = iters
        self._pop_size: int = pop_size
        self._elitism: bool = elitism
        self._init_population: Optional[
            Union[NDArray[Any], NDArray[np.byte], NDArray[np.float64]]
        ] = init_population
        self._genotype_to_phenotype: Optional[Callable] = genotype_to_phenotype
        self._no_increase_num: Optional[int] = no_increase_num
        self._show_progress_each: Optional[int] = show_progress_each
        self._keep_history: bool = keep_history
        self._n_jobs: int = self._get_n_jobs(n_jobs)

        if fitness_function_args is not None:
            self._fitness_function_args = fitness_function_args
        else:
            self._fitness_function_args = {}

        if genotype_to_phenotype_args is not None:
            self._genotype_to_phenotype_args = genotype_to_phenotype_args
        else:
            self._genotype_to_phenotype_args = {}

        self._sign: int = -1 if minimization else 1
        self._aim: Union[float, int] = self._get_aim(optimal_value, termination_error_value)
        self._calls: int = 0

        self._thefittest: TheFittest = TheFittest()
        self._stats: Statistics = Statistics()

        self._population_g_i: Union[NDArray[Any], NDArray[np.byte], NDArray[np.float64]]
        self._population_ph_i: NDArray
        self._fitness_i: NDArray[np.float64]

        if self._n_jobs > 1:
            self._parallel = Parallel(self._n_jobs)

        self._random_state = random_state
        self._on_generation = on_generation

    def _first_generation(self: EvolutionaryAlgorithm) -> None:
        return None

    def _get_init_population(self: EvolutionaryAlgorithm) -> None:
        self._first_generation()

    def _get_aim(
        self: EvolutionaryAlgorithm,
        optimal_value: Optional[Union[float, int]],
        termination_error_value: Union[float, int],
    ) -> Union[float, int]:
        if optimal_value is not None:
            return self._sign * optimal_value - termination_error_value
        else:
            return np.inf

    def _show_progress(self: EvolutionaryAlgorithm, current_iter: int) -> None:
        if self._show_progress_each is not None:
            cond_show_now = current_iter % self._show_progress_each == 0
            if cond_show_now:
                current_best = self._sign * self._thefittest._fitness
                print(f"{current_iter}-th iteration with the best fitness = {current_best}")

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

    def _get_phenotype(self, population_g: NDArray[Any]) -> NDArray[Any]:
        if self._genotype_to_phenotype is not None:
            if self._n_jobs > 1:
                populations_g = self._split_population(population_g)
                populations_ph = self._parallel(
                    delayed(self._genotype_to_phenotype)(
                        populations_g_i, **self._genotype_to_phenotype_args
                    )
                    for populations_g_i in populations_g
                )
                population_ph = np.concatenate(populations_ph, axis=0)
            else:
                population_ph = self._genotype_to_phenotype(
                    population_g, **self._genotype_to_phenotype_args
                )
        else:
            population_ph = population_g
        return population_ph

    def _get_fitness(
        self: EvolutionaryAlgorithm, population_ph: NDArray[Any]
    ) -> NDArray[np.float64]:
        if self._n_jobs > 1:
            populations_ph = self._split_population(population_ph)
            values = self._parallel(
                delayed(self._fitness_function)(populations_ph_i, **self._fitness_function_args)
                for populations_ph_i in populations_ph
            )
            value = np.concatenate(values, axis=0)
        else:
            value = self._fitness_function(population_ph, **self._fitness_function_args)

        self._calls += len(value)
        fitness = self._sign * value
        return fitness

    def get_remains_calls(self: EvolutionaryAlgorithm) -> int:
        return (self._pop_size * self._iters) - self._calls

    def get_fittest(self: EvolutionaryAlgorithm) -> Dict:
        return self._thefittest.get()

    def get_stats(self: EvolutionaryAlgorithm) -> Statistics:
        return self._stats

    def _update_data(self: EvolutionaryAlgorithm) -> None:
        max_fitness_id = np.argmax(self._fitness_i)
        self._update_fittest(self._population_g_i, self._population_ph_i, self._fitness_i)
        self._update_stats(
            fitness=self._fitness_i,
            population_g=self._population_g_i,
            population_ph=self._population_ph_i,
            max_fitness=self._fitness_i[max_fitness_id],
            max_g=self._population_g_i[max_fitness_id],
            max_ph=self._population_ph_i[max_fitness_id],
        )

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

    def _get_n_jobs(self: EvolutionaryAlgorithm, n_jobs: int) -> int:
        if n_jobs < 0:
            return max(cpu_count() + 1 + n_jobs, 1)
        elif n_jobs == 0:
            raise ValueError("Parameter n_jobs == 0 has no meaning.")
        elif n_jobs > self._pop_size:
            return self._pop_size
        else:
            return n_jobs

    def _split_population(self: EvolutionaryAlgorithm, population: NDArray) -> List:
        indexes = np.linspace(start=0, stop=self._pop_size, num=self._n_jobs + 1, dtype=np.int64)
        indexes = indexes[1:-1]

        population_split = np.split(population, indexes)
        return population_split

    def fit(self: EvolutionaryAlgorithm) -> EvolutionaryAlgorithm:

        check_random_state(self._random_state)
        self._get_init_population()
        self._from_population_g_to_fitness()

        for i in range(self._iters - 1):
            self._show_progress(i)
            if self._termitation_check():
                break
            else:
                self._get_new_population()
                self._from_population_g_to_fitness()
                if self._on_generation is not None:
                    self._on_generation(self)

        return self