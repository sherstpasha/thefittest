from thefittest.optimizers._base import Tree
from thefittest.optimizers._base import UniversalSet
from thefittest.optimizers._base import FunctionalNode
from thefittest.optimizers._base import TerminalNode
from thefittest.optimizers._base import EphemeralNode

from thefittest.tools.operators import Add
from thefittest.tools.operators import Sub
from thefittest.tools.operators import Mul
from thefittest.tools.operators import Div
from thefittest.tools.operators import Cos
from thefittest.tools.operators import Sin
from thefittest.tools.operators import Mul3
from thefittest.tools.operators import swap_mutation, shrink_mutation
from thefittest.tools.generators import growing_method, full_growing_method
from thefittest.tools.transformations import common_region
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def graph(some_tree):
    reverse_nodes = some_tree.nodes[::-1].copy()
    pack = []
    edges = []
    nodes = []
    labels = {}

    for i, node in enumerate(reverse_nodes):
        index = len(reverse_nodes) - i - 1
        labels[index] = labels[len(reverse_nodes) - i -
                               1] = str(len(reverse_nodes) - i - 1) + '. ' + node.sign[:7]  # один раз развернуть или вообще не разворачивать а сразу считать так
        #    1] = node.sign[:7]

        nodes.append(index)

        for _ in range(node.n_args):
            edges.append((index, len(reverse_nodes) - pack.pop() - 1))
        pack.append(i)

    edges.reverse()
    nodes.reverse()

    return edges, labels, nodes


def print_tree(some_tree, fig_name, underline_nodes=[]):
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
        if i in underline_nodes:
            colors[i] = (0.5, 0.1, 0.1, 1)
        else:
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


def generator():
    return np.round(np.random.uniform(0, 3), 4)


functional_set = [FunctionalNode(Add()),
                  FunctionalNode(Sub()),
                  FunctionalNode(Mul()),
                  FunctionalNode(Div()),
                  FunctionalNode(Cos()),
                  FunctionalNode(Sin()),
                  FunctionalNode(Mul3())]

terminal_set = [TerminalNode(np.array([1, 2, 3]), 'x0'),
                TerminalNode(np.array([3, 2, 1]), 'x1')]

constant_set = [EphemeralNode(generator)]

uniset = UniversalSet(functional_set, terminal_set)

print(uniset.random_terminal())

# tree_1 = full_growing_method(uniset, 4)
# tree_2 = growing_method(uniset, 5)
# tree_3 = full_growing_method(uniset, 5)
# tree_4 = full_growing_method(uniset, 3)

# # # tree_2 = tree_1

# print(tree_1)
# print(tree_2)
# print(tree_3)
# print(tree_4)
# test, _ = common_region([tree_1, tree_2, tree_3, tree_4])
# tree_5 = uniform_crossoverGP2([tree_1, tree_2, tree_3, tree_4], uniset, 1000, 1000)
# print(tree_5)
# print_tree(tree_1, 'tree_1.png')
# print_tree(tree_2, 'tree_2.png', test[1])
# print_tree(tree_3, 'tree_3.png', test[2])
# print_tree(tree_4, 'tree_4.png', test[3])
# print_tree(tree_5, 'tree_5.png')
