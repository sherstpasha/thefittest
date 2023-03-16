import numpy as np
from typing import Any
import random
from inspect import signature
from ..tools.numba_funcs import find_end_subtree_from_i
from ..tools.numba_funcs import find_id_args_from_i
from ..tools.numba_funcs import get_levels_tree_from_i


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


class Tree:
    def __init__(self, nodes, n_args):
        self.nodes = nodes
        self.n_args = n_args

    def __len__(self):
        return len(self.nodes)

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

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        else:
            for node_1, node_2 in zip(self.nodes, other.nodes):
                if np.all(node_1.value != node_2.value):
                    return False
        return True

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

    def copy(self):
        return Tree(self.nodes.copy(), self.n_args.copy())

    def change_terminals(self, change_list):
        tree_copy = self.copy()
        for i, node in enumerate(tree_copy.nodes):
            if type(node) is TerminalNode:
                for key, value in change_list.items():
                    if node.name == key:
                        tree_copy.nodes[i] = TerminalNode(value=value,
                                                          name=node.name + '_')
        return tree_copy

    def subtree(self, index: np.int32, return_class=False):
        n_index = find_end_subtree_from_i(index, self.n_args)
        if return_class:
            new_tree = Tree(self.nodes[index:n_index].copy(),
                            self.n_args[index:n_index].copy())
            return new_tree
        return index, n_index

    def concat(self, index, other_tree):
        to_return = self.copy()
        left, right = self.subtree(index)
        to_return.nodes[left:right] = other_tree.nodes.copy()
        to_return.n_args = np.r_[to_return.n_args[:left],
                                 other_tree.n_args.copy(),
                                 to_return.n_args[right:]]
        return to_return

    def get_args_id(self, index):
        args_id = find_id_args_from_i(np.int32(index), self.n_args)
        return args_id

    def get_levels(self, index):
        return get_levels_tree_from_i(np.int32(index), self.n_args)

    def get_max_level(self):
        return max(self.get_levels(0))

    def get_graph(self, keep_id=False):
        pack = []
        edges = []
        nodes = []
        labels = {}
        for i, node in enumerate(reversed(self.nodes)):
            index = len(self) - i - 1
            if keep_id:
                labels[index] = str(index) + '. ' + node.sign[:6]
            else:
                labels[index] = node.sign[:6]

            nodes.append(index)

            for _ in range(node.n_args):
                edges.append((index, len(self) - pack.pop() - 1))
            pack.append(i)

        edges.reverse()
        nodes.reverse()

        levels = self.get_levels(0)
        colors = np.zeros(shape=(len(nodes), 4))
        pos = np.zeros(shape=(len(self), 2))
        for i, lvl_i in enumerate(levels):
            total = 0
            cond = lvl_i == np.array(levels)
            h = 1/(1 + np.sum(cond))
            arange = np.arange(len(pos))[cond]

            for j, a_j in enumerate(arange):
                total += h
                pos[a_j][0] = total

            pos[i][1] = -lvl_i

            if type(self.nodes[i]) is FunctionalNode:
                colors[i] = (1, 0.72, 0.43, 1)
            else:
                colors[i] = (0.21, 0.76, 0.56, 1)

        to_return = {'edges': edges,
                     'labels': labels,
                     'nodes': nodes,
                     'pos': pos,
                     'colors': colors}
        return to_return

    def subtree_(self, index, return_class=False):
        n_index = index + 1
        possible_steps = self.nodes[index].n_args
        while possible_steps:
            possible_steps += self.nodes[n_index].n_args - 1
            n_index += 1
        if return_class:
            new_tree = Tree(self.nodes[index:n_index].copy())
            return new_tree
        return index, n_index

    # def get_args_id_(self, index=0):
    #     levels = self.get_levels()
    #     n_args = self.nodes[index].n_args
    #     args_id = []
    #     root_level = levels[self.get_levels()[index]]
    #     next_level = root_level + 1
    #     k = index + 1
    #     while n_args:
    #         if levels[k] == next_level:
    #             args_id.append(k)
    #             n_args = n_args - 1
    #         k = k + 1
    #     return args_id

    # def concat_(self, index, some_tree):
    #     left, right = self.subtree_(index)

    #     new_nodes = self.nodes[:left] + some_tree.nodes + self.nodes[right:]
    #     new_n_args = np.r_[self.n_args[:left],
    #                        some_tree.n_args,
    #                        self.n_args[right:]]
    #     to_return = Tree(new_nodes.copy(), new_n_args.copy())
    #     return to_return

    # def get_levels_(self, origin=0):
    #     d_i = origin-1
    #     s = [1]
    #     d = [origin-1]
    #     result_list = []
    #     for node in self.nodes[origin:]:
    #         s[-1] = s[-1] - 1
    #         if s[-1] == 0:
    #             s.pop()
    #             d_i = d.pop() + 1
    #         else:
    #             d_i = d[-1] + 1
    #         result_list.append(d_i)
    #         if node.n_args > 0:
    #             s.append(node.n_args)
    #             d.append(d_i)
    #         if len(s) == 0:
    #             break

    #     return np.array(result_list)


class Node:
    def __init__(self,
                 value,
                 name,
                 sign,
                 n_args):
        self.value = value
        self.name = name
        self.sign = sign
        self.n_args = n_args

    def __str__(self):
        return str(self.sign)

    def __eq__(self, other):
        return self.name == other.name


class FunctionalNode(Node):
    def __init__(self, value, sign=None):
        Node.__init__(self,
                      value=value,
                      name=value.__name__,
                      sign=sign or value.sign,
                      n_args=len(signature(value).parameters))


class TerminalNode(Node):
    def __init__(self, value, name):
        Node.__init__(self,
                      value=value,
                      name=name,
                      sign=name,
                      n_args=0)


class EphemeralNode():
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value()


class EphemeralConstantNode(Node):
    def __init__(self, generator):
        value = generator()
        Node.__init__(self,
                      value=value,
                      name=str(value),
                      sign=str(value),
                      n_args=0)


'''functional_set = (FunctionalNode, ..., FunctionalNode)
terminal_set = (TerminalNode, ..., TerminalNode)
ephemeral_set = (EphemeralNode, ..., EphemeralNode)'''


class UniversalSet:
    def __init__(self, functional_set, terminal_set, ephemeral_set=[]):
        self.functional_set = {'any': functional_set}
        for unit in functional_set:
            n_args = unit.n_args
            if n_args not in self.functional_set:
                self.functional_set[n_args] = [unit]
            else:
                self.functional_set[n_args].append(unit)
        self.union_terminal = list(terminal_set) + list(ephemeral_set)
        self.terminal_set = terminal_set
        self.ephemeral_set = ephemeral_set

    def random_terminal(self):
        choosen = random.choice(self.terminal_set)
        return choosen

    def random_ephemeral(self):
        choosen = random.choice(self.ephemeral_set)
        return EphemeralConstantNode(choosen)

    def random_terminal_or_ephemeral(self):
        choosen = random.choice(self.union_terminal)
        if type(choosen) is EphemeralNode:
            to_return = EphemeralConstantNode(choosen)
        else:
            to_return = choosen
        return to_return

    def random_functional(self, n_args='any'):
        choosen = random.choice(self.functional_set[n_args])
        return choosen
