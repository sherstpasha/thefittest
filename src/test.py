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
from line_profiler import LineProfiler
from functools import partial

def sattolo_shuffle(items):
    i = len(items)
    while i > 1:
        i = i - 1
        j = random.randrange(i)
        items[j], items[i] = items[i], items[j]
    return

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
                  FunctionalNode(Sin())]

terminal_set = [TerminalNode(np.array([1, 2, 3]), 'x0'),
                TerminalNode(np.array([3, 2, 1]), 'x1')]

constant_set = [EphemeralNode(generator)]

uniset = UniversalSet(functional_set, terminal_set, constant_set)


def one_point_crossoverGP(individs, fitness, rank, max_level):
    individ_1 = individs[0]
    individ_2 = individs[1]
    common_indexes, _ = common_region([individ_1, individ_2])

    point = random.randrange(len(common_indexes[0]))
    first_point = common_indexes[0][point]
    second_point = common_indexes[1][point]
    if random.random() < 0.5:
        first_subtree = individ_1.subtree(first_point, return_class=True)
        offspring = individ_2.concat(second_point, first_subtree)
    else:
        second_subtree = individ_2.subtree(second_point, return_class=True)
        offspring = individ_1.concat(first_point, second_subtree)
    return offspring

def one_point_crossoverGP2(individs, fitness, rank, max_level):
    individ_1 = individs[0]
    individ_2 = individs[1]
    common_indexes, _ = common_region([individ_1, individ_2])

    point = random.randrange(len(common_indexes[0]))
    first_point = common_indexes[0][point]
    second_point = common_indexes[1][point]
    if random.random() < 0.5:
        first_subtree = individ_1.subtree(first_point, return_class=True)
        offspring = individ_2.concat2(second_point, first_subtree)
    else:
        second_subtree = individ_2.subtree(second_point, return_class=True)
        offspring = individ_1.concat2(first_point, second_subtree)
    return offspring


tree_1 = full_growing_method(uniset, 16)
tree_2 = full_growing_method(uniset, 16)

# print_tree(tree_1, 'tree_1.png')
# print_tree(tree_2, 'tree_2.png')
# slice_ = slice(*tree_1.subtree(4))
# tree_3 = tree_1.replace(slice_, tree_2)

# print_tree(tree_3, 'tree_3.png')

n = 1000
begin = time.time()
for i in range(n):
    # res = one_point_crossoverGP([tree_1, tree_2], 1, 1, 16)
    res = one_point_crossoverGP([tree_1, tree_2], 1, 1, 16)
print(time.time() - begin)

begin = time.time()
for i in range(n):
    res = one_point_crossoverGP2([tree_1, tree_2], 1, 1, 16)
    # res = one_point_crossoverGP2([tree_1, tree_2], 1, 1, 16)
print(time.time() - begin)

