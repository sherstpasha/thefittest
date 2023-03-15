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
from thefittest.tools.transformations import protect_norm
from thefittest.tools.generators import growing_method, full_growing_method
from thefittest.tools.transformations import common_region
from thefittest.tools.transformations import common_region1
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import random
from line_profiler import LineProfiler
from functools import partial
from thefittest.tools.numba_funcs import find_end_subtree_from_i


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

def subtree(self, index, return_class=False):
    n_index = index + 1
    possible_steps = self.nodes[index].n_args
    while possible_steps:
        possible_steps += self.nodes[n_index].n_args - 1
        n_index += 1
    if return_class:
        new_tree = Tree(self.nodes[index:n_index].copy())
        return new_tree
    return index, n_index

def uniform_crossoverGP_rank(individs, fitness, rank, max_level):
    range_ = range(len(individs))
    probability = protect_norm(rank)
    to_return = Tree([])
    common, border = common_region(individs)
    for i, common_0_i in enumerate(common[0]):
        j = random.choices(range_, weights=probability)[0]
        id_ = common[j][i]
        to_return.nodes.append(individs[j].nodes[id_])
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
                proba = protect_norm(fitness_i)
                choosen = random.choices(
                    left_subtrees + right_subtrees, weights=proba, k=1)[0]
                to_return.nodes.extend(choosen.nodes)
            elif n_args == 2:
                fitness_l = np.array(left_fitness)
                fitness_r = np.array(right_fitness)
                proba_l = protect_norm(fitness_l)
                proba_r = protect_norm(fitness_r)

                choosen_l = random.choices(
                    left_subtrees, weights=proba_l, k=1)[0]
                to_return.nodes.extend(choosen_l.nodes)

                choosen_r = random.choices(
                    right_subtrees, weights=proba_r, k=1)[0]
                to_return.nodes.extend(choosen_r.nodes)

    to_return = to_return.copy()
    to_return.levels = to_return.get_levels()
    return to_return

def uniform_crossoverGP_rank2(individs, fitness, rank, max_level):
    range_ = range(len(individs))
    probability = protect_norm(rank)
    to_return = Tree([])
    common, border = common_region(individs)
    for i, common_0_i in enumerate(common[0]):
        j = random.choices(range_, weights=probability)[0]
        id_ = common[j][i]
        to_return.nodes.append(individs[j].nodes[id_])
        if common_0_i in border[0]:
            left_subtrees = []
            right_subtrees = []
            left_fitness = []
            right_fitness = []

            for k, tree_k in enumerate(individs):
                inner_id = common[k][i]
                args_id = np.array(tree_k.get_args_id(inner_id), np.int32)
                n_args = tree_k.nodes[inner_id].n_args
                if n_args == 1:
                    n_index = find_end_subtree_from_i(args_id[0], tree_k.n_args)
                    subtree = Tree(tree_k.nodes[args_id[0]:n_index].copy())
    
                    left_subtrees.append(subtree)
                    right_subtrees.append(subtree)
                    left_fitness.append(rank[k])
                    right_fitness.append(rank[k])
                elif n_args == 2:
                    n_index = find_end_subtree_from_i(args_id[0], tree_k.n_args)
                    subtree_l = Tree(tree_k.nodes[args_id[0]:n_index].copy())

                    n_index = find_end_subtree_from_i(args_id[1], tree_k.n_args)
                    subtree_r = Tree(tree_k.nodes[args_id[1]:n_index].copy())
                    left_subtrees.append(subtree_l)
                    right_subtrees.append(subtree_r)
                    left_fitness.append(rank[k])
                    right_fitness.append(rank[k])

            n_args = individs[j].nodes[id_].n_args
            if n_args == 1:
                fitness_i = np.array(left_fitness + right_fitness)
                proba = protect_norm(fitness_i)
                choosen = random.choices(
                    left_subtrees + right_subtrees, weights=proba, k=1)[0]
                to_return.nodes.extend(choosen.nodes)
            elif n_args == 2:
                fitness_l = np.array(left_fitness)
                fitness_r = np.array(right_fitness)
                proba_l = protect_norm(fitness_l)
                proba_r = protect_norm(fitness_r)

                choosen_l = random.choices(
                    left_subtrees, weights=proba_l, k=1)[0]
                to_return.nodes.extend(choosen_l.nodes)

                choosen_r = random.choices(
                    right_subtrees, weights=proba_r, k=1)[0]
                to_return.nodes.extend(choosen_r.nodes)

    to_return = to_return.copy()
    to_return.levels = to_return.get_levels()
    return to_return

tree_1 = full_growing_method(uniset, 5)


print(tree_1)


