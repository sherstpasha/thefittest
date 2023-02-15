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
        functional_set = set(map(FunctionalNode, functional_set))
        terminal_set = set(TerminalNode(value, key)
                           for key, value in terminal_set.items())
        self.functional_set = {}
        for unit in functional_set:
            n_args = unit.n_args
            if n_args not in self.functional_set:
                self.functional_set[n_args] = {unit}
            else:
                self.functional_set[n_args] =\
                    self.functional_set[n_args].union({unit})
        self.functional_set['any'] = functional_set
        self.terminal_set = terminal_set
        self.union = self.functional_set['any'].union(terminal_set)

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
            if type(node) == TerminalNode:
                pack.append(node.value)
            else:
                pack.append(node.value(*args))
        return pack[0]

    def __str__(self):
        # может считать с конца просто?
        reverse_nodes = self.nodes[::-1].copy()
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

    def get_downlevels(self):
        levels_numpy = np.array(self.levels)
        cond = (levels_numpy[1:] - levels_numpy[:-1]) > 0
        print(cond)

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
    to_return = Tree(nodes, levels)
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

        if level_i == level_max:
            nodes.append(uniset.choice_terminal())
        else:
            if np.random.random() < 0.5:
                nodes.append(uniset.choice_terminal())
            else:
                nodes.append(uniset.choice_functional())
            n_i = nodes[-1].n_args

            if n_i > 0:
                possible_steps.append(n_i)
                previous_levels.append(level_i)
    to_return = Tree(nodes, levels)
    return to_return


def graph(some_tree):
    reverse_nodes = some_tree.nodes[::-1].copy()
    pack = []
    edges = []
    nodes = []
    labels = {}

    for i, node in enumerate(reverse_nodes):
        labels[len(reverse_nodes) - i -
               1] = str(len(reverse_nodes) - i - 1) + '. ' + node.sign # один раз развернуть или вообще не разворачивать а сразу считать так
        nodes.append(len(reverse_nodes) - i - 1)

        for _ in range(node.n_args):
            edges.append((len(reverse_nodes) - i - 1,
                         len(reverse_nodes) - pack.pop() - 1))
        pack.append(i)

    edges.reverse()
    nodes.reverse()

    return edges, labels, nodes


def point_mutation(some_tree, uniset, proba_down):
    nodes = some_tree.nodes.copy()
    levels = some_tree.levels.copy()

    proba = proba_down/len(nodes)
    for i, node in enumerate(nodes):
        if np.random.random() < proba:
            if type(node) == TerminalNode:
                new_node = uniset.mutate_terminal(node)
            else:
                new_node = uniset.mutate_functional(node)
            nodes[i] = new_node

    new_tree = Tree(nodes, levels)
    return new_tree


def growing_mutation(some_tree, uniset, proba_down):
    proba = proba_down/len(some_tree.nodes)
    for i in range(1, len(some_tree.nodes)):
        if np.random.random() < proba:
            # второй раз выполняется может можно как-то один раз оставить?
            left, right = some_tree.subtree(i)
            max_level = some_tree.levels[left:right][-1] - \
                some_tree.levels[left:right][0]
            new_tree = growing_method(uniset, max_level)
            mutated = some_tree.concat(i, new_tree)
            return mutated
    return some_tree


def standart_crossover(individs):
    individ_1 = individs[0]
    individ_2 = individs[1]
    firts_point = np.random.randint(1,  len(individ_1.nodes))
    second_point = np.random.randint(1,  len(individ_2.nodes))

    if np.random.random() < 0.5:
        print(1, firts_point, second_point)
        left, right = individ_1.subtree(firts_point)
        first_subtree = Tree(individ_1.nodes[left:right],
                            individ_1.levels[left:right])
        offspring = individ_2.concat(second_point, first_subtree)
    else:  
        print(2, firts_point, second_point)
        left, right = individ_2.subtree(second_point)
        second_subtree = Tree(individ_2.nodes[left:right],
                            individ_2.levels[left:right])
        offspring = individ_1.concat(firts_point, second_subtree)
    return offspring

def print_tree(some_tree, fig_name):
    edges, labels, nodes = graph(some_tree)
    levels = some_tree.levels

    colors = np.zeros(shape=(len(nodes), 4))
    pos = np.zeros(shape=(len(nodes), 2))
    for i, lvl_i in enumerate(levels):
        total = 0
        cond = lvl_i == np.array(levels)
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

    plt.savefig(fig_name)
    plt.close()


X = np.ones((100, 2), dtype=int)

functional_set_ = (Mul(),
                   Add(),
                   Pow(),
                   Cos(),
                   Sin())
terminal_set_ = {'x1': X[:, 0],
                 'x2': X[:, 1],
                 '0': 0, '1': 1,
                 '2': 2, '3': 3,
                 '4': 4, '5': 5,
                 '6': 6, '7': 7,
                 '8': 8, '9': 9}


uniset_ = UniversalSet(functional_set=functional_set_,
                       terminal_set=terminal_set_)



F21 = FunctionalNode(Add())
F1 = FunctionalNode(Cos())
T1 = TerminalNode(1, '1')
F22 = FunctionalNode(Mul())
T2 = TerminalNode(2, '2')
T3 = TerminalNode(3, '3')
tree1 = Tree([F21, F1, T1, F22, T2, T3], [0, 1, 2, 1, 2, 2])

tree1.get_downlevels()
# tree1 = full_growing_method(uniset_, level_max=4)
# tree2 = full_growing_method(uniset_, level_max=4)
# tree3 = standart_crossover([tree1, tree2])
# # for i in range(10000):
# #     tree1 = full_growing_method(uniset_, level_max=4)
# #     tree2 = growing_mutation(tree1, uniset_, 4)
# # # tree2 = full_growing_method(uniset_, level_max=4)
# # # tree3 = full_growing_method(uniset_, level_max=4)

# # # ind = np.random.randint(1, len(tree1.nodes))
# # # print(ind)
# # # tree1_tree2 = tree1.concat(ind, tree2)

print_tree(tree1, 'tree1.png')
# print_tree(tree2, 'tree2.png')
# print_tree(tree3, 'tree3.png')
# print_tree(tree1_tree2, 'tree1_tree2.png')
