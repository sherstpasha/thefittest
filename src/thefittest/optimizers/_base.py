import numpy as np
from typing import Optional
from typing import Callable
from typing import Any
import random
from inspect import signature


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


class Tree:
    def __init__(self, nodes, levels):
        self.nodes = nodes
        self.levels = levels

    def subtree(self, index: int):
        n_index = index + 1
        possible_steps = self.nodes[index].n_args
        while possible_steps:
            possible_steps += self.nodes[n_index].n_args - 1
            n_index += 1
        return index, n_index

    def compile(self):
        # может считать с конца просто?
        reverse_nodes = self.nodes[::-1].copy()
        pack = []
        for node in reverse_nodes:
            args = []
            for _ in range(node.n_args):
                args.append(pack.pop())
            if type(node) != FunctionalNode:
                pack.append(node.value)
            else:
                pack.append(node.value(*args))
        return pack[0]

    def __str__(self):
        reverse_nodes = self.nodes[::-1].copy()
        pack = []
        for node in reverse_nodes:
            args = []
            for _ in range(node.n_args):
                args.append(pack.pop())
            if type(node) != FunctionalNode:
                pack.append(node.name)
            else:
                pack.append(node.value.write(*args))
        return pack[0]

    def get_levels(self, origin=0):
        d_i = origin-1
        s = [1]
        d = [origin-1]
        result_list = []
        for node in self.nodes:
            s[-1] = s[-1] - 1
            if s[-1] == 0:
                s.pop()
                d_i = d.pop() + 1
            else:
                d_i = d[-1] + 1
            result_list.append(d_i)
            if node.n_args > 0:
                s.append(node.n_args)
                d.append(d_i)
        return result_list

    def concat(self, index, some_tree):
        left, right = self.subtree(index)
        levels = some_tree.get_levels(origin=self.levels[left])

        new_nodes = self.nodes[:left].copy()
        new_levels = self.levels[:left].copy()

        new_nodes += some_tree.nodes.copy()
        new_levels += levels

        new_nodes += self.nodes[right:].copy()
        new_levels += self.levels[right:].copy()
        to_return = Tree(new_nodes, new_levels)
        return to_return
    
    def copy(self):
        return Tree(self.nodes.copy(), self.levels.copy())


class FunctionalNode:
    def __init__(self, value):
        self.value = value
        self.args = None
        self.name = value.__name__
        self.sign = value.sign
        self.n_args = len(signature(value).parameters)


class TerminalNode:
    def __init__(self, value, name):
        self.value = value
        self.name = name
        self.sign = name
        self.n_args = 0


class TerminalConstantNode:
    def __init__(self, value):
        self.value = value
        self.name = str(value)
        self.sign = str(value)
        self.n_args = 0


class UniversalSet:
    def __init__(self, functional_set, terminal_set, constant_set=None):
        functional_set = list(map(FunctionalNode, functional_set))
        terminal_set = list(TerminalNode(value, key)
                            for key, value in terminal_set.items())
        if constant_set is not None:
            constant_set = list(map(TerminalConstantNode, constant_set))
        self.functional_set = {}
        for unit in functional_set:
            n_args = unit.n_args
            if n_args not in self.functional_set:
                self.functional_set[n_args] = [unit]
            else:
                self.functional_set[n_args].append(unit)
        self.functional_set['any'] = functional_set
        self.terminal_set = terminal_set
        self.constant_set = constant_set

    def choice_terminal(self):
        if self.constant_set is not None:
            if np.random.random() < 0.5:
                return random.choice(self.terminal_set)
            else:
                return random.choice(self.constant_set)
        else:
            return random.choice(self.terminal_set)

    def choice_functional(self, n_args='any'):
        return random.choice(list(self.functional_set[n_args]))

    def mutate_terminal(self, terminal):
        if len(self.terminal_set) > 1:
            remains = list(filter(lambda x: x != terminal,
                                  self.terminal_set))
            to_return = random.choice(remains)
        else:
            to_return = list(self.terminal_set)[0]
        return to_return

    def mutate_constant(self, constant):
        if len(self.constant_set) > 1:
            remains = list(filter(lambda x: x != constant,
                                  self.constant_set))
            to_return = random.choice(remains)
        else:
            to_return = list(self.constant_set)[0]
        return to_return

    def mutate_functional(self, functional):
        n_args = functional.n_args
        if len(self.functional_set[n_args]) > 1:
            remains = list(filter(lambda x: x != functional,
                                  self.functional_set[n_args]))
            to_return = random.choice(remains)
        else:
            to_return = list(self.functional_set[n_args])[0]
        return to_return
