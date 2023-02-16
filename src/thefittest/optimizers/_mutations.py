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
                   proba_down, growing_method=None):
    nodes = some_tree.nodes.copy()
    levels = some_tree.levels.copy()

    proba = proba_down/len(nodes)
    for i, node in enumerate(nodes):
        
        if np.random.random() < proba:
            if type(node) != FunctionalNode:
                if np.random.random() < 0.5:
                    new_node = uniset.mutate_terminal(node)
                else:
                    new_node = uniset.mutate_constant(node)
            else:
                new_node = uniset.mutate_functional(node)
            nodes[i] = new_node

    new_tree = Tree(nodes, levels)
    return new_tree


def growing_mutation(some_tree, uniset,
                     proba_down, growing_method=growing_method):
    proba = proba_down/len(some_tree.nodes)
    for i in range(1, len(some_tree.nodes)):
        if np.random.random() < proba:
            # второй раз выполняется может можно как-то один раз оставить?
            left, right = some_tree.subtree(i)
            max_level = some_tree.levels[left:right][-1] - \
                some_tree.levels[left:right][0]
            new_tree = growing_method(uniset, max_level)
            mutated = some_tree.concat(i, new_tree)
            return mutated
    return some_tree
