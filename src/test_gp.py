import numpy as np
from inspect import signature
import random
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


class Operator:
    def write(self, *args):
        return self.formula.format(*args)


class Add(Operator):
    def __init__(self):
        self.formula = '({} + {})'
        # нужен ли этот парамтер его нет по умолчанию? можно создать по умолчанию в родителе init?
        self.__name__ = 'add'
        self.sign = '+'

    def __call__(self, x, y):
        return x + y


class Add3(Operator):
    def __init__(self):
        self.formula = '({} + {} + {})'
        self.__name__ = 'add3'
        self.sign = '+'

    def __call__(self, x, y, z):
        return x + y + z


class Dif4(Operator):
    def __init__(self):
        self.formula = '({} - {} - {} - {})'
        self.__name__ = 'dif4'
        self.sign = '-'

    def __call__(self, x, y, z, t):
        return x - y - z - t


class Neg(Operator):
    def __init__(self):
        self.formula = '-{}'
        self.__name__ = 'neg'
        self.sign = '-'

    def __call__(self, x):
        return -x


class Mul(Operator):
    def __init__(self):
        self.formula = '({} * {})'
        self.__name__ = 'mul'
        self.sign = '*'

    def __call__(self, x, y):
        return x * y


class Cos(Operator):
    def __init__(self):
        self.formula = 'cos({})'
        self.__name__ = 'cos'
        self.sign = 'cos'

    def __call__(self, x):
        return np.cos(x)


class Sin(Operator):
    def __init__(self):
        self.formula = 'sin({})'
        self.__name__ = 'sin'
        self.sign = 'sin'

    def __call__(self, x):
        return np.sin(x)


class Pow(Operator):
    def __init__(self):
        self.formula = '({}**{})'
        self.__name__ = 'pow'
        self.sign = '**'

    def __call__(self, x, y):
        return x**y


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


class UniversalSet:
    def __init__(self, functional_set, terminal_set):
        self.functional_set = {}
        for unit in functional_set:
            if unit.n_args not in self.functional_set:
                self.functional_set[unit.n_args] = {unit}
            else:
                self.functional_set[unit.n_args] =\
                    self.functional_set[unit.n_args].union({unit})
        self.functional_set['any'] = functional_set
        self.terminal_set = set(terminal_set)
        self.union = functional_set + terminal_set

    def choice_terminal(self):
        return random.choice(list(self.terminal_set))

    def choice_functional(self, n_args='any'):
        return random.choice(list(self.functional_set[n_args]))

    def choice_universal(self):
        return random.choice(list(self.union))

    def mutate_terminal(self, terminal):
        if len(self.terminal_set) > 1:
            remains = self.terminal_set - {terminal}
            to_return = random.choice(list(remains))
        else:
            to_return = list(self.terminal_set)[0]
        return to_return
    
    def mutate_functional(self, functional):
        n_args = functional.n_args
        if len(self.functional_set[n_args]) > 1:
            remains = self.functional_set[n_args] - {functional}
            to_return = random.choice(list(remains))
        else:
            to_return = list(self.functional_set[n_args])[0]
        return to_return


class Tree:
    def __init__(self, nodes):
        self.nodes = np.array(nodes, dtype=object)
        self.levels = []

    def subtree(self, index: int):
        n_index = index + 1
        possible_steps = self.nodes[index].n_args
        while possible_steps:
            possible_steps += self.nodes[n_index].n_args - 1
            n_index += 1
        return n_index

    def compile(self):
        reverse_nodes = self.nodes[::-1]
        pack = []
        for node in reverse_nodes:
            args = []
            for _ in range(node.n_args):
                args.append(pack.pop())
            if type(node) == TerminalNode:
                pack.append(node.value)
            else:
                pack.append(node.value(*args))
        return pack[0]

    def __str__(self):
        reverse_nodes = self.nodes[::-1]
        pack = []
        for node in reverse_nodes:
            args = []
            for _ in range(node.n_args):
                args.append(pack.pop())
            if type(node) == TerminalNode:
                pack.append(node.name)
            else:
                pack.append(node.value.write(*args))
        return pack[0]

    def get_levels(self):
        d_i = -1
        s = [1]
        d = [-1]
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
        return np.array(result_list)

    def print_(self):
        t = [node.name for node in self.nodes]
        t2 = [node.n_args for node in self.nodes]
        print(t)
        print(t2)


def full_growing_method(uniset, level_max):
    nodes = []
    levels = []
    possible_steps = [1]
    previous_levels = [-1]
    level_i = -1
    while len(possible_steps):
        possible_steps[-1] = possible_steps[-1] - 1
        if possible_steps[-1] == 0:
            possible_steps.pop()
            level_i = previous_levels.pop() + 1
        else:
            level_i = previous_levels[-1] + 1
        levels.append(level_i)
        if level_i == level_max:
            nodes.append(uniset.choice_terminal())
        else:
            nodes.append(uniset.choice_functional())
            n_i = nodes[-1].n_args

            possible_steps.append(n_i)
            previous_levels.append(level_i)
    to_return = Tree(nodes)
    to_return.levels = np.array(levels)
    return to_return


def growing_method(uniset, level_max):
    nodes = []
    levels = []
    possible_steps = [1]
    previous_levels = [-1]
    level_i = -1
    while len(possible_steps):
        possible_steps[-1] = possible_steps[-1] - 1
        if possible_steps[-1] == 0:
            possible_steps.pop()
            level_i = previous_levels.pop() + 1
        else:
            level_i = previous_levels[-1] + 1
        levels.append(level_i)
        
        if level_i == 0:
            nodes.append(uniset.choice_functional())
            n_i = nodes[-1].n_args

            possible_steps.append(n_i)
            previous_levels.append(level_i)
        elif level_i == level_max:
            nodes.append(uniset.choice_terminal())
        else:
            nodes.append(uniset.choice_universal())
            n_i = nodes[-1].n_args

            if n_i > 0:
                possible_steps.append(n_i)
                previous_levels.append(level_i)
    to_return = Tree(nodes)
    to_return.levels = np.array(levels)
    return to_return


def graph(some_tree):
    reverse_nodes = some_tree.nodes[::-1]
    pack = []
    edges = []
    nodes = []
    labels = {}

    for i, node in enumerate(reverse_nodes):
        labels[len(reverse_nodes) - i - 1] = node.sign
        nodes.append(len(reverse_nodes) - i - 1)

        for _ in range(node.n_args):
            edges.append((len(reverse_nodes) - i - 1,
                         len(reverse_nodes) - pack.pop() - 1))
        pack.append(i)

    edges.reverse()
    nodes.reverse()

    return edges, labels, nodes

def point_mutation(some_tree, uniset, proba_down):
    proba = proba_down/len(some_tree.nodes)
    for i, node in enumerate(some_tree.nodes):
        if type(node) == TerminalNode:
            print(node.name, 'old')
            new_node = uniset.mutate_terminal(node)
            print(new_node.name, 'new')
        # if np.random.random(proba):





def print_tree(some_tree):
    edges, labels, nodes = graph(some_tree)
    levels = some_tree.levels

    colors = np.zeros(shape=(len(nodes), 4))
    pos = np.zeros(shape=(len(nodes), 2))
    for i, lvl_i in enumerate(levels):
        total = 0
        cond = lvl_i == levels
        h = 1/(1 + np.sum(cond))
        arange = np.arange(len(pos))[cond]

        for j, a_j in enumerate(arange):
            total += h
            pos[a_j][0] = total

        pos[i][1] = -lvl_i

    for i, node in enumerate(some_tree.nodes):
        if type(node) is FunctionalNode:
            colors[i] = (1, 0.72, 0.43, 1)
        else:
            colors[i] = (0.21, 0.76, 0.56, 1)

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    nx.draw_networkx_nodes(g, pos, node_color=colors,
                           edgecolors='black', linewidths=0.5)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels, font_size=10)

    plt.savefig('fig.png')


functional_set_ = (FunctionalNode(Mul()),
                  FunctionalNode(Add()),
                  FunctionalNode(Pow()),
                  FunctionalNode(Cos()),
                  FunctionalNode(Sin()))

terminal_set_ = (TerminalNode(1, 'x'),
                # TerminalNode(2, 'y'),
                # TerminalNode(3, 'z')
                )

uniset_ = UniversalSet(functional_set=functional_set_,
                      terminal_set=terminal_set_)

# print(uniset.functional_set)
# print(uniset.terminal_set)
# print(uniset.union)
# # functional_set_ = {2: (FunctionalNode(Mul()),
# #                        FunctionalNode(Add()),
# #                        FunctionalNode(Pow())),
# #                    1: (FunctionalNode(Cos()),
# #                        FunctionalNode(Sin()))}


tree = growing_method(uniset_, level_max=5)
print(tree)
# print(tree.levels)
# tree.print_()

print_tree(tree)

point_mutation(tree, uniset_, 0.25)
# # print(tree.get_levels())
