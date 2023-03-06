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
from thefittest.tools.operators import point_mutation
from thefittest.tools.operators import ephemeral_mutation
from thefittest.tools.operators import ephemeral_gauss_mutation
from thefittest.tools.operators import terminal_mutation
from thefittest.tools.operators import growing_mutation
from thefittest.tools.operators import swap_mutation
from thefittest.tools.operators import shrink_mutation
from thefittest.tools.operators import uniform_crossoverGP_prop
from thefittest.tools.generators import growing_method, full_growing_method
from thefittest.tools.transformations import common_region
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


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
                  FunctionalNode(Sin()),
                  FunctionalNode(Mul3())]

terminal_set = [TerminalNode(np.array([1, 2, 3]), 'x0'),
                TerminalNode(np.array([3, 2, 1]), 'x1')]

constant_set = [EphemeralNode(generator)]

uniset = UniversalSet(functional_set, terminal_set, constant_set)

tree_1 = full_growing_method(uniset, 4)
tree_2 = full_growing_method(uniset, 4)
tree_3 = full_growing_method(uniset, 4)

tree_4 = uniform_crossoverGP_prop([tree_1, tree_2, tree_3],
                                   np.array([1, 2, 3]),
                                     np.array([1, 2, 3]), 16)



# tree_2 = growing_mutation(tree_1, uniset, 100, 16)

print(tree_1)
print(tree_2)
print(tree_3)
print_tree(tree_1, 'tree_1.png')
print_tree(tree_2, 'tree_2.png')
print_tree(tree_3, 'tree_3.png')
print_tree(tree_4, 'tree_4.png')