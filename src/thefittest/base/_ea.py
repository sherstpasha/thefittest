import numpy as np
from typing import Any
from typing import Tuple
from typing import Dict
from typing import Callable
from typing import Optional


class LastBest:
    def __init__(self) -> None:
        self.value = np.nan
        self.no_increase_counter = 0

    def update(self,
               current_value: float):
        if self.value == current_value:
            self.no_increase_counter += 1
        else:
            self.no_increase_counter = 0
            self.value = current_value
        return self


class TheFittest:
    def __init__(self):
        self.genotype: Any
        self.phenotype: Any
        self.fitness = -np.inf

    def update(self,
               population_g: np.ndarray,
               population_ph: np.ndarray,
               fitness: np.ndarray):
        temp_best_id = np.argmax(fitness)
        temp_best_fitness = fitness[temp_best_id].copy()
        if temp_best_fitness > self.fitness:
            self.genotype = population_g[temp_best_id].copy()
            self.phenotype = population_ph[temp_best_id].copy()
            self.fitness = temp_best_fitness.copy()
        return self

    def get(self) -> Tuple:
        to_return = (self.genotype.copy(),
                     self.phenotype.copy(),
                     self.fitness.copy())
        return to_return


class Statistics(dict):
    def update(self,
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
        self.fitness_function = fitness_function
        self.genotype_to_phenotype = genotype_to_phenotype
        self.iters = iters
        self.pop_size = pop_size
        self.no_increase_num = no_increase_num
        self.show_progress_each = show_progress_each
        self.keep_history = keep_history

        self.sign = -1 if minimization else 1

        if optimal_value is not None:
            self.aim = self.sign*optimal_value - termination_error_value
        else:
            self.aim = np.inf

        self.calls = 0

    def evaluate(self,
                 population_ph: np.ndarray) -> np.ndarray:
        self.calls += len(population_ph)
        return self.sign*self.fitness_function(population_ph)

    def show_progress(self, iteration_number: int) -> None:
        cond_show_progress_switch = self.show_progress_each is not None
        cond_show_progress_now = iteration_number % self.show_progress_each == 0
        if cond_show_progress_switch and cond_show_progress_now:
            print(f'{iteration_number} iteration with fitness = {self.thefittest.fitness}')

    def termitation_check(self, no_increase_counter):
        cond_aim_achieved = self.thefittest.fitness >= self.aim
        cond_no_increase_achieved = no_increase_counter == self.no_increase_num
        return cond_aim_achieved or cond_no_increase_achieved

    def get_remains_calls(self):
        return (self.pop_size + (self.iters-1)*(self.pop_size-1)) - self.calls
