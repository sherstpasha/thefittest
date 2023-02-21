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
            self.value = current_value
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
                 keep_history: Optional[str] = None):
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

    def subtree(self, index: int, return_class=False):
        n_index = index + 1
        possible_steps = self.nodes[index].n_args
        while possible_steps:
            possible_steps += self.nodes[n_index].n_args - 1
            n_index += 1
        if return_class:
            new_tree = Tree(self.nodes[index:n_index].copy(), None)
            new_tree.levels = new_tree.get_levels()
            return new_tree
        return index, n_index

    def compile(self):
        pack = []
        for node in reversed(self.nodes):
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

        new_nodes = self.nodes[:left]
        new_levels = self.levels[:left]

        new_nodes += some_tree.nodes
        new_levels += levels

        new_nodes += self.nodes[right:]
        new_levels += self.levels[right:]
        to_return = Tree(new_nodes.copy(), new_levels.copy())
        return to_return

    def copy(self):
        return Tree(self.nodes.copy(), self.levels.copy())

    def get_max_level(self):
        return np.max(self.levels)

    def __eq__(self, other):
        if len(self.nodes) != len(other.nodes):
            return False
        else:
            for node_1, node_2 in zip(self.nodes, other.nodes):
                if np.all(node_1.value != node_2.value):
                    return False
        return True

    def is_functional(self, node):
        return type(node) is FunctionalNode

    def is_terminal(self, node):
        return type(node) is TerminalNode

    # def is_contains_terminals(self, i=0):
    #     pool = list(map(self.is_terminal, self.nodes))
    #     print(pool)
    #     return np.any(pool)

    def simplify_by_index(self, index=0):
        if type(self.nodes[index]) is TerminalNode:
            return self, False
        n_index = index + 1
        possible_steps = self.nodes[index].n_args
        while possible_steps:
            if type(self.nodes[n_index]) is TerminalNode:
                return self, False
            possible_steps += self.nodes[n_index].n_args - 1
            n_index += 1

        pack = []
        for node in reversed(self.nodes[index:n_index]):
            args = []
            for _ in range(node.n_args):
                args.append(pack.pop())
            if type(node) != FunctionalNode:
                print(node.value)
                pack.append(node.value)
            else:
                pack.append(node.value(*args))

        
        value = pack[0]
        print(value)
        node_ = EphemeralConstant()
        node_.value = value
        node_.name = str(value)
        node_.sign = str(value)
        return self.concat(index, Tree([node_], [0])), True

    def simplify(self):
        for i in range(len(self.nodes)-1, -1, -1):
            if type(self.nodes[i]) is FunctionalNode:
                self, cond = self.simplify_by_index(index=i)
        return self

    def change_terminals(self, change_list):
        tree_copy = self.copy()
        for i, node in enumerate(tree_copy.nodes):
            if type(node) is TerminalNode:
                for key, value in change_list.items():
                    if node.name == key:
                        tree_copy.nodes[i] = TerminalNode(value=value,
                                                          name=node.name + '_')
        return tree_copy

    def get_args_id(self, index):
        n_args = self.nodes[index].n_args
        args_id = []
        root_level = self.levels[index]
        next_level = root_level + 1
        k = index + 1
        while n_args:
            if self.levels[k] == next_level:
                args_id.append(k)
                n_args = n_args - 1
            k = k + 1
        return args_id


class FunctionalNode:
    def __init__(self, value: Callable):
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


class EphemeralConstant:
    def __init__(self, generator: Callable = np.random.random):
        self.value = generator()
        self.name = str(self.value)
        self.sign = str(self.value)
        self.n_args = 0


class UniversalSet:
    def __init__(self, functional_set, terminal_set, constant_set=dict([])):
        functional_set = list(map(FunctionalNode, functional_set))
        terminal_set = list(TerminalNode(value, key)
                            for key, value in terminal_set.items())
        constant_set = list((key, value)
                            for key, value in constant_set.items())
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
        self.union_terminal = terminal_set + constant_set

    def choice_terminal(self):
        choosen = random.choice(self.union_terminal)
        if type(choosen) is not TerminalNode:
            to_return = EphemeralConstant(choosen[1])
        else:
            to_return = choosen
        return to_return

    def choice_functional(self, n_args='any'):
        return random.choice(self.functional_set[n_args])

    def mutate_terminal(self):
        if len(self.union_terminal) > 1:
            choosen = random.choice(self.union_terminal)
            if type(choosen) is not TerminalNode:
                to_return = EphemeralConstant(choosen[1])
            else:
                to_return = choosen
        else:
            to_return = self.union_terminal[0]
        return to_return

    def mutate_functional(self, n_args):
        if len(self.functional_set[n_args]) > 1:
            to_return = random.choice(self.functional_set[n_args])
        else:
            to_return = self.functional_set[n_args][0]
        return to_return
