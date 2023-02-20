import numpy as np
from ._base import Tree
from ._base import FunctionalNode
from ._initializations import growing_method


def flip_mutation(individ, proba):
    mask = np.random.random(size=individ.shape) < proba
    individ[mask] = 1 - individ[mask]
    return individ


def best_1(current, population, F_value):
    best = population[-1]
    r1, r2 = np.random.choice(range(len(population)), size=2, replace=False)
    return best + F_value*(population[r1] - population[r2])


def rand_1(current, population, F_value):
    r1, r2, r3 = np.random.choice(
        range(len(population)), size=3, replace=False)
    return population[r3] + F_value*(population[r1] - population[r2])


def rand_to_best1(current, population, F_value):
    best = population[-1]
    r1, r2, r3 = np.random.choice(
        range(len(population)), size=3, replace=False)
    return population[r1] + F_value*(
        best - population[r1]) + F_value*(population[r2] - population[r3])


def current_to_best_1(current, population, F_value):
    best = population[-1]
    r1, r2 = np.random.choice(range(len(population)), size=2, replace=False)
    return current + F_value*(best - current) + F_value*(population[r1] - population[r2])


def best_2(current, population, F_value):
    best = population[-1]
    r1, r2, r3, r4 = np.random.choice(
        range(len(population)), size=4, replace=False)
    return best + F_value*(population[r1] - population[r2]) + F_value*(population[r3] - population[r4])


def rand_2(current, population, F_value):
    r1, r2, r3, r4, r5 = np.random.choice(
        range(len(population)), size=5, replace=False)
    return population[r5] + F_value*(population[r1] - population[r2]) + F_value*(population[r3] - population[r4])


def current_to_pbest_1(current, population, F_value):
    p_min = 2/len(population)
    p_i = np.random.uniform(p_min, 0.2)

    value = int(p_i*len(population))
    pbest = population[-value:]
    p_best_ind = np.random.randint(0, len(pbest))

    best = pbest[p_best_ind]

    r1, r2 = np.random.choice(range(len(population)), size=2, replace=False)
    return current + F_value*(best - current) + F_value*(population[r1] - population[r2])


def current_to_rand_1(current, population, F_value):
    r1, r2, r3 = np.random.choice(
        range(len(population)), size=3, replace=False)
    return population[r1] + F_value*(population[r3] - current) + F_value*(population[r1] - population[r2])


def point_mutation(some_tree, uniset,
                   proba_down, max_level):
    some_tree = some_tree.copy()
    nodes = some_tree.nodes.copy()
    levels = some_tree.levels.copy()

    proba = proba_down/len(nodes)
    for i, node in enumerate(nodes):
        if np.random.random() < proba:
            if type(node) != FunctionalNode:
                new_node = uniset.mutate_terminal()
            else:
                new_node = uniset.mutate_functional(node.n_args)
            nodes[i] = new_node

    to_return = Tree(nodes, levels)
    return to_return


def growing_mutation(some_tree, uniset,
                     proba_down, max_level):
    some_tree = some_tree.copy()
    proba = proba_down/len(some_tree.nodes)
    if np.random.random() < proba:
        i = np.random.randint(0, len(some_tree.nodes))
        left, right = some_tree.subtree(i)
        max_level_i = max_level - some_tree.levels[left:right][0]

        new_tree = growing_method(uniset, max_level_i)
        to_return = some_tree.concat(i, new_tree)
    else:
        to_return = some_tree
    return to_return


def simplify_mutations(some_tree, uniset,
                       proba_down, max_level):
    some_tree = some_tree.copy()
    proba = proba_down/len(some_tree.nodes)
    if np.random.random() < proba:
        i = np.random.randint(0, len(some_tree.nodes))
        to_return = some_tree.simplify_by_index(i)[0]
    else:
        to_return = some_tree
    return to_return
