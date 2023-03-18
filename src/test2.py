import time
import numpy as np
from thefittest.optimizers._base import UniversalSet
from thefittest.optimizers._base import FunctionalNode
from thefittest.optimizers._base import TerminalNode
from thefittest.optimizers._base import EphemeralNode
from thefittest.tools.operators import Mul
from thefittest.tools.operators import Add
from thefittest.tools.operators import Cos
from thefittest.tools.operators import Sin
from thefittest.tools.operators import Neg
from thefittest.tools.operators import Inv
from thefittest.tools.operators import Pow2
from thefittest.tools.generators import full_growing_method, growing_method
from thefittest.optimizers import SelfCGP
from thefittest.tools.transformations import donothing
from thefittest.tools.transformations import coefficient_determination
from thefittest.benchmarks.symbolicregression17 import problems_dict
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from thefittest.tools.numba_funcs import find_first_difference_between_two
from thefittest.tools.numba_funcs import find_end_subtree_from_i
from thefittest.tools.transformations import common_region_, common_region
from line_profiler import LineProfiler


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


def generator1():
    return np.round(np.random.uniform(0, 10), 4)


def generator2():
    return np.random.randint(0, 10)


F = 'F3'
function = problems_dict[F]['function']
left = problems_dict[F]['bounds'][0]
right = problems_dict[F]['bounds'][1]
size = 300
n_vars = problems_dict[F]['n_vars']
X = np.array([np.linspace(left, right, size) for _ in range(n_vars)]).T

y = function(X)

functional_set = [FunctionalNode(Add()),
                  FunctionalNode(Mul()),
                  FunctionalNode(Pow2()),
                  FunctionalNode(Inv()),
                  FunctionalNode(Neg()),
                  FunctionalNode(Cos()),
                  FunctionalNode(Sin())]
terminal_set = [TerminalNode(X[:, i], f'x{i}') for i in range(n_vars)]
ephemeral_set = [EphemeralNode(generator1), EphemeralNode(generator2)]

uniset = UniversalSet(functional_set, terminal_set, ephemeral_set)







# print()
for i in range(1):
    tree_1 = full_growing_method(uniset, 7)
    tree_2 = full_growing_method(uniset, 7)
    common_region([tree_1, tree_2])
    # print(common_region_([tree_1, tree_2]))
    # print(common_region([tree_1, tree_2]))


# print_tree(tree_1, 'tree_1.png')
# print_tree(tree_2, 'tree_2.png')


total = 0
n = 5000

for i in range(n):
    lvl = np.random.randint(3, 25)
    tree_1 = growing_method(uniset, lvl)
    tree_2 = growing_method(uniset, lvl)
    begin = time.time()
    common_region_([tree_1, tree_2])
    total += time.time() - begin
print(total)

total = 0
for i in range(n):
    lvl = np.random.randint(3, 25)
    tree_1 = growing_method(uniset, lvl)
    tree_2 = growing_method(uniset, lvl)
    begin = time.time()
    common_region([tree_1, tree_2])
    total += time.time() - begin
print(total)


# lp = LineProfiler()
# lp_wrapper = lp(common_region_)
# lp_wrapper([tree_1, tree_2])
# lp.print_stats()


# lp = LineProfiler()
# lp_wrapper = lp(common_region2)
# lp_wrapper(tree_1.n_args, tree_2.n_args)
# lp.print_stats()
