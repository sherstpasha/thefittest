import numpy as np
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
            self.value = current_value
        return self


class TheFittest:
    def __init__(self):
        self.genotype: Any
        self.phenotype: Any
        self.fitness = -np.inf

    def update(self, population_g, population_ph, fitness):
        temp_best_id = np.argmax(fitness)
        temp_best_fitness = fitness[temp_best_id].copy()
        if temp_best_fitness > self.fitness:
            self.genotype = population_g[temp_best_id].copy()
            self.phenotype = population_ph[temp_best_id].copy()
            self.fitness = temp_best_fitness.copy()

        return self

    def get(self):
        return self.genotype.copy(), self.phenotype.copy(), self.fitness.copy()


class EvolutionaryAlgorithm:
    def __init__(self,
                 fitness_function,
                 genotype_to_phenotype,
                 iters,
                 pop_size,
                 optimal_value=None,
                 termination_error_value=0.,
                 no_increase_num=None,
                 minimization=False,
                 show_progress_each=None,
                 keep_history=None):
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

    def evaluate(self, population_ph):
        self.calls += len(population_ph)
        return self.sign*self.fitness_function(population_ph)

    def show_progress(self, i):
        if (self.show_progress_each is not None) and (i % self.show_progress_each == 0):
            print(f'{i} iteration with fitness = {self.thefittest.fitness}')

    def termitation_check(self, no_increase):
        return (self.thefittest.fitness >= self.aim) or (no_increase == self.no_increase_num)

    def get_remains_calls(self):
        return (self.pop_size + (self.iters-1)*(self.pop_size-1)) - self.calls