from thefittest.optimizers._base import Tree
from thefittest.optimizers._base import UniversalSet
from thefittest.optimizers._base import FunctionalNode
from thefittest.optimizers._base import TerminalNode
from thefittest.optimizers._base import EphemeralNode
from thefittest.optimizers._base import EphemeralConstantNode

from thefittest.tools.operators import Add
from thefittest.tools.operators import Sub
from thefittest.tools.operators import Mul
from thefittest.tools.operators import Div
from thefittest.tools.operators import Cos
from thefittest.tools.operators import Sin

from thefittest.tools.generators import growing_method, full_growing_method
from thefittest.tools.transformations import common_region
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import random


def print_tree(some_tree, file_name):
    graph = some_tree.get_graph(True)
    g = nx.Graph()
    g.add_nodes_from(graph['nodes'])
    g.add_edges_from(graph['edges'])

    nx.draw_networkx_nodes(g, graph['pos'], node_color=graph['colors'],
                           edgecolors='black', linewidths=0.5)
    nx.draw_networkx_edges(g, graph['pos'])
    nx.draw_networkx_labels(g, graph['pos'], graph['labels'], font_size=10)
    
    plt.savefig(file_name)
    plt.close()


def generator():
    return np.round(np.random.uniform(0, 3), 4)


functional_set = [FunctionalNode(Add()),
                  FunctionalNode(Sub()),
                  FunctionalNode(Mul()),
                  FunctionalNode(Div()),
                  FunctionalNode(Cos()),
                #   FunctionalNode(Sin())
                  ]

terminal_set = [TerminalNode(np.array([1, 2, 3]), 'x0'),
                TerminalNode(np.array([3, 2, 1]), 'x1')]

constant_set = [EphemeralNode(generator)]

uniset = UniversalSet(functional_set, terminal_set, constant_set)


def border_points(trees):
    len_trees = len(trees)
    terminate = False
    indexes = []
    common_indexes = []
    border_indexes = []
    for tree in trees:
        indexes.append(list(range(len(tree.nodes))))
        common_indexes.append([])
        border_indexes.append([])

    
    while True:
        inner_break = False
        iters = min(map(len, indexes))

        for i in range(iters):
            first_n_args = trees[0].nodes[indexes[0][i]].n_args
            for j in range(1, len_trees):
                if not inner_break:
                    other_n_args = trees[j].nodes[indexes[j][i]].n_args
                    if first_n_args != other_n_args:
                        inner_break = True
                        border_indexes[j].append(indexes[j][i])
                else:
                    border_indexes[j].append(indexes[j][i])

            if inner_break:
                border_indexes[0].append(indexes[0][i])
                break

        if inner_break:
            for j in range(len(indexes)):
                to_ = border_indexes[j][-1]
                left, right = trees[j].subtree(to_)

                delete_from = indexes[j].index(left) + 1
                delete_to = indexes[j].index(right - 1) + 1
                common_indexes[j].extend(indexes[j][:delete_from])
                indexes[j] = indexes[j][delete_to:]
                if len(indexes[j]) < 1:
                    terminate = True
        else:
            terminate = True
    
        if terminate:
            for j in range(len(indexes)):
                common_indexes[j].extend(indexes[j])
            break
           
    return common_indexes, border_indexes

def common_region1(trees):
    terminate = False
    indexes = []
    common_indexes = []
    border_indexes = []
    for tree in trees:
        indexes.append(list(range(len(tree.nodes))))
        common_indexes.append([])
        border_indexes.append([])

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
                for j in range(0, len(indexes)):
                    border_indexes[j].append(indexes[j][i])
                break
        for j in range(len(indexes)):
            _, right = trees[j].subtree(common_indexes[j][-1])
            delete_to = indexes[j].index(right-1) + 1
            indexes[j] = indexes[j][delete_to:]

            if len(indexes[j]) < 1:
                terminate = True
                break

    return common_indexes, border_indexes

tree_1 = full_growing_method(uniset, level_max=16)

tree_2 = full_growing_method(uniset, level_max=16)
n = 1000
begin = time.time()
for i in range(n):
    # res = one_point_crossoverGP([tree_1, tree_2], 1, 1, 16)
    res = border_points([tree_1, tree_2])
print(time.time() - begin)

begin = time.time()
for i in range(n):
    res = common_region1([tree_1, tree_2])
    # res = one_point_crossoverGP2([tree_1, tree_2], 1, 1, 16)
print(time.time() - begin)

# app = []
# for i in range(100):
#     tree_1 = full_growing_method(uniset, level_max=4)

#     tree_2 = full_growing_method(uniset, level_max=4)
#     _, border_indexes = border_points([tree_1, tree_2])
#     _2, border_indexes2 = common_region1([tree_1, tree_2])
#     app.append(_ == _2)
#     # print_tree(tree_1, 'tree_1.png')
#     # print_tree(tree_2, 'tree_2.png')
#     print(_, border_indexes, 1)
#     print(_2, border_indexes2, 2)

# # print(app)