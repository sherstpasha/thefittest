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
from thefittest.tools.operators import standart_crossover
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



