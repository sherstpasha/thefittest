import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
from thefittest.optimizers._base import Tree
from thefittest.optimizers._base import UniversalSet
from thefittest.optimizers._base import FunctionalNode
from thefittest.optimizers._operators import Mul
from thefittest.optimizers._operators import Add3
from thefittest.optimizers._operators import Pow
from thefittest.optimizers._operators import Cos
from thefittest.optimizers._operators import Sin
from thefittest.optimizers._crossovers import standart_crossover


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
               1] = str(len(reverse_nodes) - i - 1) + '. ' + node.sign  # один раз развернуть или вообще не разворачивать а сразу считать так
        nodes.append(len(reverse_nodes) - i - 1)

        for _ in range(node.n_args):
            edges.append((len(reverse_nodes) - i - 1,
                         len(reverse_nodes) - pack.pop() - 1))
        pack.append(i)

    edges.reverse()
    nodes.reverse()

    return edges, labels, nodes


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

def common_region(trees):

    # trees_nodes = []
    for i, node in enumerate(trees[0].nodes):
        for tree in trees:
            if node.n_args != tree.nodes[i].nargs:
                print(i)



X = np.ones((100, 2), dtype=int)

functional_set_ = (Mul(),
                   Add3(),
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


# F21 = FunctionalNode(Add())
# F1 = FunctionalNode(Cos())
# T1 = TerminalNode(1, '1')
# F22 = FunctionalNode(Mul())
# T2 = TerminalNode(2, '2')
# T3 = TerminalNode(3, '3')
# tree1 = Tree([F21, T1, T2], [0, 1, 1])

# tree1.get_stoppoints()
tree1 = full_growing_method(uniset_, level_max=4)
tree2 = full_growing_method(uniset_, level_max=4)
tree3 = standart_crossover([tree1, tree2])
# # for i in range(10000):
# #     tree1 = full_growing_method(uniset_, level_max=4)
# #     tree2 = growing_mutation(tree1, uniset_, 4)
# # # tree2 = full_growing_method(uniset_, level_max=4)
# # # tree3 = full_growing_method(uniset_, level_max=4)

# # # ind = np.random.randint(1, len(tree1.nodes))
# # # print(ind)
# # # tree1_tree2 = tree1.concat(ind, tree2)

print_tree(tree1, 'tree1.png')
print_tree(tree2, 'tree2.png')
print_tree(tree3, 'tree3.png')
# print_tree(tree1_tree2, 'tree1_tree2.png')
