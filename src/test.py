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
from thefittest.tools.operators import Operator
from thefittest.tools.operators import point_mutation
from thefittest.tools.operators import growing_mutation
from thefittest.tools.operators import swap_mutation
from thefittest.tools.operators import shrink_mutation
from thefittest.tools.operators import standart_crossover, tournament_selection, tournament_selection_
from thefittest.tools.transformations import protect_norm, common_region, common_region_
from thefittest.tools.generators import growing_method, full_growing_method
# from thefittest.tools.numba_funcs import tournament_selection
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import random
from line_profiler import LineProfiler


class Mul3(Operator):
    def __init__(self):
        self.formula = '({} * {} * {})'
        self.__name__ = 'mul3'
        self.sign = '*'

    def __call__(self, x, y, z):
        return x * y * z


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
                  #   FunctionalNode(Mul3()),
                  FunctionalNode(Sin())]

terminal_set = [TerminalNode(np.array([1, 2, 3]), 'x0'),
                TerminalNode(np.array([3, 2, 1]), 'x1')]

constant_set = [EphemeralNode(generator)]

uniset = UniversalSet(functional_set, terminal_set, constant_set)


tree_1 = full_growing_method(uniset, 3)
tree_2 = full_growing_method(uniset, 3)

fitness = np.random.random(size = 2)
# rank = np.array([1, 2])


def uniform_crossoverGP_prop(individs, fitness, rank, max_level):
    to_return = Tree([], [])
    new_n_args = []
    common, border = common_region(individs)
    for i, common_0_i in enumerate(common[0]):
        j = random.randrange(len(individs))
        id_ = common[j][i]
        print(id_)
        if common_0_i in border[0]:
            subtree = individs[j].subtree(id_, return_class=True)
            print(subtree)
            to_return.nodes.extend(subtree.nodes)
            new_n_args.extend(subtree.n_args)
        else:
            to_return.nodes.append(individs[j].nodes[id_])
            new_n_args.append(individs[j].n_args[id_])

    to_return = to_return.copy()
    to_return.n_args = np.array(new_n_args.copy(), dtype=np.int32)

    return to_return

tree_3 = uniform_crossoverGP_prop([tree_1, tree_2], fitness, fitness, 16)

print(tree_3)
print(len(tree_3.nodes), len(tree_3.n_args))

print_tree(tree_1, 'tree_1.png')
print_tree(tree_2, 'tree_2.png')
print_tree(tree_3, 'tree_3.png')


# n = 1000
# total = 0
# for i in range(n):
#     begin = time.time()
#     tree_3 = tournament_selection_(fitness, fitness, 7, 50)
#     total += time.time() - begin
# print(total)

# total = 0
# for i in range(n):
#     begin = time.time()
#     tree_3 = tournament_selection(fitness, fitness, 7, 50)
#     total += time.time() - begin
# print(total)

# lp = LineProfiler()
# lp_wrapper = lp(uniform_crossoverGP_tour)
# lp_wrapper([tree_1, tree_2], fitness, rank, 16)
# lp.print_stats()