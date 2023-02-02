import numpy as np
from typing import Optional
from typing import Callable
from typing import Any


class LastBest:
    def __init__(self):
        self.value = np.nan
        self.no_increase = 0

    def update(self, current_value: float):
        if self.value == current_value:
            self.no_increase += 1
        else:
            self.no_increase = 0
            self.value = current_value.copy()
        return self


class TheFittest:
    def __init__(self):
        self.genotype: Any
        self.phenotype: Any
        self.fitness = -np.inf

    def update(self, population_g: np.ndarray, population_ph: np.ndarray, fitness: np.ndarray[float]):
        temp_best_id = np.argmax(fitness)
        temp_best_fitness = fitness[temp_best_id].copy()
        if temp_best_fitness > self.fitness:
            self.genotype = population_g[temp_best_id].copy()
            self.phenotype = population_ph[temp_best_id].copy()
            self.fitness = temp_best_fitness.copy()

        return self

    def get(self):
        return self.genotype.copy(), self.phenotype.copy(), self.fitness.copy()


class Statistics:
    def __init__(self):
        self.population_g = np.array([])
        self.population_ph = np.array([])
        self.fitness = np.array([])

    def append_arr(self, arr_to, arr_from):
        shape_to = (-1, arr_from.shape[0], arr_from.shape[1])
        shape_from = (1, arr_from.shape[0], arr_from.shape[1])
        result = np.vstack([arr_to.reshape(shape_to),
                            arr_from.copy().reshape(shape_from)])
        return result

    def update(self,
               population_g_i: np.ndarray,
               population_ph_i: np.ndarray,
               fitness_i: np.ndarray):

        self.population_g = self.append_arr(self.population_g,
                                            population_g_i)
        self.population_ph = self.append_arr(self.population_ph,
                                             population_ph_i)
        self.fitness = np.append(self.fitness, np.max(fitness_i))
        return self


class StaticticSelfCGA(Statistics):
    def __init__(self):
        Statistics.__init__(self)
        self.s_proba = np.array([], dtype=float)
        self.c_proba = np.array([], dtype=float)
        self.m_proba = np.array([], dtype=float)

    def update(self, population_g_i: np.ndarray, population_ph_i: np.ndarray, fitness_i: np.ndarray,
               s_proba_i, c_proba_i, m_proba_i):
        super().update(population_g_i, population_ph_i, fitness_i)
        self.s_proba = np.vstack([self.s_proba.reshape(-1, len(s_proba_i)),
                                  s_proba_i.copy()])
        self.c_proba = np.vstack([self.c_proba.reshape(-1, len(c_proba_i)),
                                  c_proba_i.copy()])
        self.m_proba = np.vstack([self.m_proba.reshape(-1, len(m_proba_i)),
                                  m_proba_i.copy()])
        return self


class EvolutionaryAlgorithm:
    def __init__(self,
                 fitness_function: Callable[[np.ndarray[Any]], np.ndarray[float]],
                 genotype_to_phenotype: Callable[[np.ndarray[Any]], np.ndarray[Any]],
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

    def evaluate(self, population_ph: np.ndarray[Any]):
        self.calls += len(population_ph)
        return self.sign*self.fitness_function(population_ph)

    def show_progress(self, i: int):
        if (self.show_progress_each is not None) and (i % self.show_progress_each == 0):
            print(f'{i} iteration with fitness = {self.thefittest.fitness}')

    def termitation_check(self, no_increase: int):
        return (self.thefittest.fitness >= self.aim) or (no_increase == self.no_increase_num)

    def get_remains_calls(self):
        return (self.pop_size + (self.iters-1)*(self.pop_size-1)) - self.calls
