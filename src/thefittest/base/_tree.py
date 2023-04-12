import numpy as np
from inspect import signature
from ..tools.numba_funcs import find_end_subtree_from_i
from ..tools.numba_funcs import find_id_args_from_i
from ..tools.numba_funcs import get_levels_tree_from_i
import random


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
            new_tree = Tree(self.nodes[index:n_index].copy(
            ), self.n_args[index:n_index].copy())
            return new_tree
        return index, n_index

    def concat_(self, index, some_tree):
        left, right = self.subtree_(index)

        new_nodes = self.nodes[:left] + some_tree.nodes + self.nodes[right:]
        new_n_args = np.r_[self.n_args[:left],
                           some_tree.n_args,
                           self.n_args[right:]]
        to_return = Tree(new_nodes.copy(), new_n_args.copy())
        return to_return


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


class UniversalSet:
    def __init__(self, functional_set, terminal_set):
        self.functional_set = {'any': functional_set}
        for unit in functional_set:
            n_args = unit.n_args
            if n_args not in self.functional_set:
                self.functional_set[n_args] = [unit]
            else:
                self.functional_set[n_args].append(unit)
        self.terminal_set = list(terminal_set)

    def random_terminal_or_ephemeral(self):
        choosen = random.choice(self.terminal_set)
        if type(choosen) is EphemeralNode:
            to_return = EphemeralConstantNode(choosen)
        else:
            to_return = choosen
        return to_return

    def random_functional(self, n_args='any'):
        choosen = random.choice(self.functional_set[n_args])
        return choosen
