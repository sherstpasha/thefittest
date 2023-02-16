import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
from thefittest.optimizers._base import Tree
from thefittest.optimizers._base import UniversalSet
from thefittest.optimizers._base import FunctionalNode
from thefittest.optimizers._base import TerminalNode
from thefittest.optimizers._operators import Mul
from thefittest.optimizers._operators import Add3
from thefittest.optimizers._operators import Add
from thefittest.optimizers._operators import Pow
from thefittest.optimizers._operators import Cos
from thefittest.optimizers._operators import Sin
from thefittest.optimizers._crossovers import standart_crossover
from thefittest.optimizers._mutations import point_mutation
from thefittest.optimizers._mutations import growing_mutation
from thefittest.optimizers._initializations import full_growing_method
from thefittest.optimizers._initializations import growing_method


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


def common_region(trees):
    terminate = False
    indexes = []
    common_indexes = []
    for tree in trees:
        indexes.append(list(range(len(tree.nodes))))
        common_indexes.append([])

    while not terminate:
        inner_break = False
        iters = np.min(list(map(len, indexes)))
        for i in range(iters):
            first_n_args = trees[0].nodes[indexes[0][i]].n_args
            common_indexes[0].append(indexes[0][i])
            for j in range(1, len(indexes)):
                common_indexes[j].append(indexes[j][i])
                if first_n_args != trees[j].nodes[indexes[j][i]].n_args:
                    inner_break = True

            if inner_break:
                break

        for j in range(len(indexes)):
            _, right = trees[j].subtree(common_indexes[j][-1])
            delete_to = indexes[j].index(right-1) + 1
            indexes[j] = indexes[j][delete_to:]

            if len(indexes[j]) < 1:
                terminate = True
                break

    return common_indexes


def one_point_crossover(individs, fitness, rank):
    individ_1 = individs[0]
    individ_2 = individs[1]
    common_indexes = common_region([individ_1, individ_2])
    point = np.random.randint(0,  len(common_indexes[0]))
    first_point = common_indexes[0][point]
    second_point = common_indexes[1][point]
    # print(point, first_point, second_point)
    if np.random.random() < 0.5:
        print(1, point, first_point, second_point)
        left, right = individ_1.subtree(first_point)
        first_subtree = Tree(individ_1.nodes[left:right],
                             individ_1.levels[left:right])
        offspring = individ_2.concat(second_point, first_subtree)
    else:
        print(2, point, first_point, second_point)
        left, right = individ_2.subtree(second_point)
        second_subtree = Tree(individ_2.nodes[left:right],
                              individ_2.levels[left:right])
        offspring = individ_1.concat(first_point, second_subtree)
    return offspring


X = np.arange(0, 5, 0.01).reshape(-1, 1)
print(X.shape)
target = np.sin(X[:, 0]) + 0.1*X[:,0]**2
target = target + np.random.normal(loc=0, scale=0.1, size=len(target))
functional_set_ = (Mul(),
                   Add(),
                #    Add3(),
                   #    Pow(),
                   Sin()
                   )
terminal_set_ = {'x1': X[:, 0]}

constant_set_ = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


uniset_ = UniversalSet(functional_set=functional_set_,
                       terminal_set=terminal_set_,
                       constant_set=constant_set_)


def FF(some_tree):
    y_ = some_tree.compile()
    y_ = np.ones(len(X))*y_
    return np.mean((y_ - target)**2)

trees = []
fitness = []
for i in range(100000):
    # print(i)
    tree1 = growing_method(uniset_, level_max=10)
    trees.append(tree1.copy())
    fitness.append(FF(tree1))


fitness = np.array(fitness)
print(fitness)
argmin = np.argmin(fitness)

y_ = trees[argmin].compile()
y_ = np.ones(len(X))*y_
plt.scatter(X[:, 0], target, label='y_true')
plt.plot(X[:, 0], y_, label='y_tree', color='red')
plt.legend()
plt.savefig('line1.png')
plt.close()
print_tree(trees[argmin], fig_name='tree1.png')
print(trees[argmin])
