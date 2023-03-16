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
from thefittest.tools.operators import standart_crossover, tournament_selection
from thefittest.tools.transformations import protect_norm, common_region, common_region_
from thefittest.tools.generators import growing_method, full_growing_method
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

fitness = np.array([1, 2])
rank = np.array([1, 2])

def uniform_crossoverGP_tour(individs, fitness, rank, max_level):
    to_return = Tree([], [])
    new_n_args = []
    common, border = common_region(individs)
    for i, common_0_i in enumerate(common[0]):

        j = tournament_selection(fitness, rank, 2, 1)[0]
        id_ = common[j][i]
        to_return.nodes.append(individs[j].nodes[id_])
        new_n_args.append(individs[j].n_args[id_])
        if common_0_i in border[0]:
            left_subtrees = []
            right_subtrees = []
            left_fitness = []
            right_fitness = []

            for k, tree_k in enumerate(individs):
                inner_id = common[k][i]
                args_id = tree_k.get_args_id(inner_id)
                n_args = tree_k.nodes[inner_id].n_args
                if n_args == 1:
                    subtree = tree_k.subtree(args_id[0], return_class=True)
                    left_subtrees.append(subtree)
                    right_subtrees.append(subtree)
                    left_fitness.append(rank[k])
                    right_fitness.append(rank[k])
                elif n_args == 2:
                    subtree_l = tree_k.subtree(args_id[0], return_class=True)
                    subtree_r = tree_k.subtree(args_id[1], return_class=True)
                    left_subtrees.append(subtree_l)
                    right_subtrees.append(subtree_r)
                    left_fitness.append(rank[k])
                    right_fitness.append(rank[k])

            n_args = individs[j].nodes[id_].n_args
            if n_args == 1:
                fitness_i = np.array(left_fitness + right_fitness)
                j = tournament_selection(fitness_i, fitness_i, 2, 1)[0]
                choosen = (left_subtrees + right_subtrees)[j]
                to_return.nodes.extend(choosen.nodes)
                new_n_args.extend(choosen.n_args)
            elif n_args == 2:
                fitness_l = np.array(left_fitness)
                fitness_r = np.array(right_fitness)


                j = tournament_selection(fitness_l, fitness_l, 2, 1)[0]
                choosen_l = left_subtrees[j]
                to_return.nodes.extend(choosen_l.nodes)
                new_n_args.extend(choosen_l.n_args)
                j = tournament_selection(fitness_r, fitness_r, 2, 1)[0]
                choosen_r = right_subtrees[j]
                to_return.nodes.extend(choosen_r.nodes)
                new_n_args.extend(choosen_r.n_args)

    to_return = to_return.copy()
    to_return.n_args = np.array(new_n_args.copy(), dtype=np.int32)
    return to_return


tree_3 = uniform_crossoverGP_tour([tree_1, tree_2], fitness, rank, 16)

# print_tree(tree_1, 'tree_1.png')
# print_tree(tree_2, 'tree_2.png')
# print_tree(tree_3, 'tree_3.png')


# n = 1000
# total = 0
# for i in range(n):
#     begin = time.time()
#     tree_3 = uniform_crossoverGP_tour([tree_1, tree_2], fitness, rank, 16)
#     total += time.time() - begin
# print(total)

lp = LineProfiler()
lp_wrapper = lp(uniform_crossoverGP_tour)
lp_wrapper([tree_1, tree_2], fitness, rank, 16)
lp.print_stats()