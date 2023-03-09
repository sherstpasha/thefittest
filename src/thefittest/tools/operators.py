import numpy as np
from ..optimizers._base import Tree
from ..optimizers._base import FunctionalNode
from ..optimizers._base import TerminalNode
from ..optimizers._base import EphemeralConstantNode
from .generators import growing_method
from .generators import sattolo_shuffle
from .transformations import protect_norm
from .transformations import common_region
import random


min_value = np.finfo(np.float64).min
max_value = np.finfo(np.float64).max


################################## MUTATUIONS ##################################
# genetic algorithm
def flip_mutation(individ, proba):
    mask = np.random.random(size=individ.shape) < proba
    individ[mask] = 1 - individ[mask]
    return individ


# differential evolution
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
    return current + F_value*(best - current) +\
        F_value*(population[r1] - population[r2])


def best_2(current, population, F_value):
    best = population[-1]
    r1, r2, r3, r4 = np.random.choice(
        range(len(population)), size=4, replace=False)
    return best + F_value*(population[r1] - population[r2]) +\
        F_value*(population[r3] - population[r4])


def rand_2(current, population, F_value):
    r1, r2, r3, r4, r5 = np.random.choice(
        range(len(population)), size=5, replace=False)
    return population[r5] + F_value*(population[r1] - population[r2]) +\
        F_value*(population[r3] - population[r4])


def current_to_pbest_1(current, population, F_value):
    p_min = 2/len(population)
    p_i = np.random.uniform(p_min, 0.2)

    value = int(p_i*len(population))
    pbest = population[-value:]
    p_best_ind = np.random.randint(0, len(pbest))

    best = pbest[p_best_ind]

    r1, r2 = np.random.choice(range(len(population)), size=2, replace=False)
    return current + F_value*(best - current) +\
        F_value*(population[r1] - population[r2])


def current_to_rand_1(current, population, F_value):
    r1, r2, r3 = np.random.choice(
        range(len(population)), size=3, replace=False)
    return population[r1] + F_value*(population[r3] - current) +\
        F_value*(population[r1] - population[r2])


# genetic propramming
def point_mutation(some_tree, uniset,
                   proba, max_level):
    to_return = some_tree.copy()
    if np.random.random() < proba:
        i = np.random.randint(0, len(to_return))
        if type(to_return.nodes[i]) != FunctionalNode:
            new_node = uniset.random_terminal_or_ephemeral()
        else:
            n_args = to_return.nodes[i].n_args
            new_node = uniset.random_functional(n_args)
        to_return.nodes[i] = new_node

    return to_return


def ephemeral_mutation(some_tree, uniset,
                       proba, max_level):
    to_return = some_tree.copy()
    if np.random.random() < proba:
        indexes = [i for i, nodes in enumerate(to_return.nodes)
                   if type(nodes) == EphemeralConstantNode]
        if len(indexes) > 0:
            i = random.choice(indexes)
            new_node = uniset.random_ephemeral()
            to_return.nodes[i] = new_node

    return to_return


def ephemeral_gauss_mutation(some_tree, uniset,
                             proba, max_level):
    to_return = some_tree.copy()
    if np.random.random() < proba:
        indexes = [i for i, nodes in enumerate(to_return.nodes)
                   if type(nodes) == EphemeralConstantNode]
        if len(indexes) > 0:
            i = random.choice(indexes)
            value = to_return.nodes[i].value
            gauss = np.random.normal(1, 0.015)
            new_node = uniset.random_ephemeral()
            new_value = value*gauss

            new_node.value = new_value
            new_node.name = str(new_value)
            new_node.sign = str(new_value)

            to_return.nodes[i] = new_node

    return to_return


def terminal_mutation(some_tree, uniset,
                      proba, max_level):
    to_return = some_tree.copy()
    if np.random.random() < proba:
        indexes = [i for i, nodes in enumerate(to_return.nodes)
                   if type(nodes) == TerminalNode]
        if len(indexes) > 0:
            i = random.choice(indexes)
            new_node = uniset.random_terminal()
            to_return.nodes[i] = new_node

    return to_return


def growing_mutation(some_tree, uniset,
                     proba, max_level):
    to_return = some_tree.copy()
    if np.random.random() < proba:
        i = np.random.randint(0, len(to_return))
        left, right = to_return.subtree(i)
        max_level_i = max_level - to_return.levels[left:right][0]
        new_tree = growing_method(uniset, max_level_i)
        to_return = to_return.concat(i, new_tree)

    return to_return


def swap_mutation(some_tree, uniset,
                  proba, max_level):
    to_return = some_tree.copy()
    if np.random.random() < proba:
        indexes = [i for i, nodes in enumerate(
            to_return.nodes) if nodes.n_args > 1]
        if len(indexes) > 0:
            i = random.choice(indexes)
            args_id = to_return.get_args_id(i)
            new_arg_id = args_id.copy()
            sattolo_shuffle(new_arg_id)

            for old_j, new_j in zip(args_id, new_arg_id):
                subtree = some_tree.subtree(old_j, return_class=True)
                to_return = to_return.concat(new_j, subtree)

            to_return.levels = to_return.get_levels()

    return to_return


def shrink_mutation(some_tree, uniset,
                    proba, max_level):
    to_return = some_tree.copy()
    if len(to_return) > 2:
        if np.random.random() < proba:
            indexes = [i for i, nodes in enumerate(to_return.nodes)
                       if nodes.n_args > 0]
            if len(indexes) > 0:
                i = random.choice(indexes)
                args_id = to_return.get_args_id(i)
                choosen = random.choice(args_id)
                to_return = to_return.concat(i, some_tree.subtree(
                    choosen, return_class=True))

    return to_return


################################## CROSSOVERS ##################################
# genetic algorithm
def empty_crossover(individs, *args):
    return individs[0]


def one_point_crossover(individs, fitness, rank):
    cross_point = np.random.randint(0, len(individs[0]))
    if np.random.random() > 0.5:
        offspring = individs[0].copy()
        offspring[:cross_point] = individs[1][:cross_point].copy()
    else:
        offspring = individs[1].copy()
        offspring[:cross_point] = individs[0][:cross_point].copy()
    return offspring


def two_point_crossover(individs, fitness, rank):
    c_point_1, c_point_2 = np.sort(np.random.choice(range(len(individs[0])),
                                                    size=2,
                                                    replace=False))
    if np.random.random() > 0.5:
        offspring = individs[0].copy()
        offspring[c_point_1:c_point_2] = \
            individs[1][c_point_1:c_point_2].copy()
    else:
        offspring = individs[1].copy()
        offspring[c_point_1:c_point_2] = \
            individs[0][c_point_1:c_point_2].copy()

    return offspring


def uniform_crossover(individs, fitness, rank):
    choosen = np.random.choice(range(len(individs)),
                               size=len(individs[0]))
    diag = range(len(individs[0]))
    return individs[choosen, diag]


def uniform_prop_crossover(individs, fitness, rank):
    probability = protect_norm(fitness)
    choosen = np.random.choice(range(len(individs)),
                               size=len(individs[0]), p=probability)

    diag = range(len(individs[0]))
    return individs[choosen, diag]


def uniform_rank_crossover(individs, fitness, rank):
    probability = protect_norm(rank)
    choosen = np.random.choice(range(len(individs)),
                               size=len(individs[0]), p=probability)

    diag = range(len(individs[0]))
    return individs[choosen, diag]


def uniform_tour_crossover(individs, fitness, rank):
    tournament = np.random.choice(range(len(individs)), 2*len(individs[0]))
    tournament = tournament.reshape(-1, 2)

    choosen = np.argmin(fitness[tournament], axis=1)
    diag = range(len(individs[0]))
    return individs[choosen, diag]


# differential evolution
def binomial(individ, mutant, CR):
    individ = individ.copy()
    j = np.random.choice(range(len(individ)), size=1)[0]
    mask_random = np.random.random(len(individ)) <= CR
    mask_j = np.arange(len(individ)) == j
    mask_union = mask_random | mask_j
    individ[mask_union] = mutant[mask_union].copy()
    return individ


def standart_crossover(individs, fitness, rank, max_level):
    individ_1 = individs[0].copy()
    individ_2 = individs[1].copy()
    first_point = np.random.randint(0,  len(individ_1))
    second_point = np.random.randint(0,  len(individ_2))

    if np.random.random() < 0.5:
        first_subtree = individ_1.subtree(first_point, return_class=True)
        offspring = individ_2.concat(second_point, first_subtree)
        if offspring.get_max_level() > max_level:
            offspring = individ_2
    else:
        second_subtree = individ_2.subtree(second_point, return_class=True)
        offspring = individ_1.concat(first_point, second_subtree)
        if offspring.get_max_level() > max_level:
            offspring = individ_1
    return offspring


# genetic propramming
def one_point_crossoverGP(individs, fitness, rank, max_level):
    individ_1 = individs[0]
    individ_2 = individs[1]
    common_indexes, _ = common_region([individ_1, individ_2])

    point = np.random.randint(0,  len(common_indexes[0]))
    first_point = common_indexes[0][point]
    second_point = common_indexes[1][point]
    if np.random.random() < 0.5:
        first_subtree = individ_1.subtree(first_point, return_class=True)
        offspring = individ_2.concat(second_point, first_subtree)
    else:
        second_subtree = individ_2.subtree(second_point, return_class=True)
        offspring = individ_1.concat(first_point, second_subtree)
    return offspring

def uniform_crossoverGPc(individs, fitness, rank, max_level):
    '''Poli, Riccardo & Langdon, W.. (2001). On the Search
    Properties of Different Crossover Operators in Genetic Programming. '''
    new_nodes = []
    individ_1 = individs[0]
    individ_2 = individs[1]
    common_indexes, border = common_region([individ_1, individ_2])

    for i in range(len(common_indexes[0])):
        if common_indexes[0][i] in border[0]:
            if np.random.random() < 0.5:
                id_ = common_indexes[0][i]
                left, right = individ_1.subtree(index=id_)
                new_nodes.extend(individ_1.nodes[left:right])
            else:
                id_ = common_indexes[1][i]
                left, right = individ_2.subtree(index=id_)
                new_nodes.extend(individ_2.nodes[left:right])
        else:
            if np.random.random() < 0.5:
                id_ = common_indexes[0][i]
                new_nodes.append(individ_1.nodes[id_])
            else:
                id_ = common_indexes[1][i]
                new_nodes.append(individ_2.nodes[id_])
    to_return = Tree(new_nodes.copy())
    to_return.levels = to_return.get_levels()
    return to_return


def uniform_crossoverGP(individs, fitness, rank, max_level):
    range_ = range(len(individs))
    to_return = Tree([])
    common, border = common_region(individs)
    for i, common_0_i in enumerate(common[0]):
        j = np.random.choice(range_, 1)[0]
        id_ = common[j][i]
        to_return.nodes.append(individs[j].nodes[id_])
        if common_0_i in border[0]:
            left_subtrees = []
            right_subtrees = []

            for k, tree_k in enumerate(individs):
                inner_id = common[k][i]
                args_id = tree_k.get_args_id(inner_id)

                n_args = tree_k.nodes[inner_id].n_args
                if n_args == 1:
                    subtree = tree_k.subtree(args_id[0], return_class=True)
                    left_subtrees.append(subtree)
                    right_subtrees.append(subtree)
                elif n_args == 2:
                    subtree_l = tree_k.subtree(args_id[0], return_class=True)
                    subtree_r = tree_k.subtree(args_id[1], return_class=True)
                    left_subtrees.append(subtree_l)
                    right_subtrees.append(subtree_r)

            n_args = individs[j].nodes[id_].n_args
            if n_args == 1:
                choosen = random.choices(
                    left_subtrees + right_subtrees, k=1)[0]
                to_return.nodes.extend(choosen.nodes)
            elif n_args == 2:
                choosen_l = random.choices(left_subtrees, k=1)[0]
                to_return.nodes.extend(choosen_l.nodes)

                choosen_r = random.choices(right_subtrees, k=1)[0]
                to_return.nodes.extend(choosen_r.nodes)

    to_return = to_return.copy()
    to_return.levels = to_return.get_levels()
    return to_return


def uniform_crossoverGP_prop(individs, fitness, rank, max_level):
    range_ = range(len(individs))
    probability = protect_norm(fitness)
    to_return = Tree([])
    common, border = common_region(individs)
    for i, common_0_i in enumerate(common[0]):
        j = np.random.choice(range_, 1, p=probability)[0]
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
                    left_fitness.append(fitness[k])
                    right_fitness.append(fitness[k])
                elif n_args == 2:
                    subtree_l = tree_k.subtree(args_id[0], return_class=True)
                    subtree_r = tree_k.subtree(args_id[1], return_class=True)
                    left_subtrees.append(subtree_l)
                    right_subtrees.append(subtree_r)
                    left_fitness.append(fitness[k])
                    right_fitness.append(fitness[k])

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


def uniform_crossoverGP_rank(individs, fitness, rank, max_level):
    range_ = range(len(individs))
    probability = protect_norm(rank)
    to_return = Tree([])
    common, border = common_region(individs)
    for i, common_0_i in enumerate(common[0]):
        j = np.random.choice(range_, 1, p=probability)[0]
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


def uniform_crossoverGP_tour(individs, fitness, rank, max_level):
    range_ = range(len(individs))
    to_return = Tree([])
    common, border = common_region(individs)
    for i, common_0_i in enumerate(common[0]):

        j = tournament_selection(fitness, rank, 2, 1)[0]
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


################################## SELECTIONS ##################################
# genetic algorithm
def proportional_selection(fitness, ranks,
                           tour_size, quantity):
    probability = fitness/fitness.sum()
    choosen = np.random.choice(range(len(fitness)),
                               size=quantity, p=probability)
    return choosen


def rank_selection(fitness, ranks,
                   tour_size, quantity):
    probability = ranks/np.sum(ranks)
    choosen = np.random.choice(range(len(fitness)),
                               size=quantity, p=probability)
    return choosen


def tournament_selection(fitness, ranks,
                         tour_size, quantity):
    tournament = np.random.choice(
        range(len(fitness)), tour_size*quantity)
    tournament = tournament.reshape(-1, tour_size)
    max_fit_id = np.argmax(fitness[tournament], axis=1)
    choosen = np.diag(tournament[:, max_fit_id])
    return choosen


##################################### MATH #####################################
class Operator:
    def write(self, *args):
        return self.formula.format(*args)


class Cos(Operator):
    def __init__(self):
        self.formula = 'cos({})'
        self.__name__ = 'cos'
        self.sign = 'cos'

    def __call__(self, x):
        return np.cos(x)


class Sin(Operator):
    def __init__(self):
        self.formula = 'sin({})'
        self.__name__ = 'sin'
        self.sign = 'sin'

    def __call__(self, x):
        return np.sin(x)


class Add(Operator):
    def __init__(self):
        self.formula = '({} + {})'
        self.__name__ = 'add'
        self.sign = '+'

    def __call__(self, x, y):
        return np.clip(x + y, min_value, max_value)


class Sub(Operator):
    def __init__(self):
        self.formula = '({} - {})'
        self.__name__ = 'sub'
        self.sign = '-'

    def __call__(self, x, y):
        return np.clip(x - y, min_value, max_value)


class Neg(Operator):
    def __init__(self):
        self.formula = '-{}'
        self.__name__ = 'neg'
        self.sign = '-'

    def __call__(self, x):
        return np.clip(-x, min_value, max_value)


class Mul(Operator):
    def __init__(self):
        self.formula = '({} * {})'
        self.__name__ = 'mul'
        self.sign = '*'

    def __call__(self, x, y):
        return np.clip(x * y, min_value, max_value)


class Pow(Operator):
    def __init__(self):
        self.formula = '({}**{})'
        self.__name__ = 'pow'
        self.sign = '**'

    def __call__(self, x, y):
        if type(x) == np.ndarray:
            res = np.power(x, y, out=np.ones_like(
                x, dtype=np.float64), where=x > 0)
        else:
            if x < 0:
                res = 0
            else:
                res = x**y

        return np.clip(res, min_value, max_value)


class Pow2(Operator):
    def __init__(self):
        self.formula = '({}**2)'
        self.__name__ = 'pow2'
        self.sign = '**2'

    def __call__(self, x):
        res = x**2
        return np.clip(res, min_value, max_value)


# class Pow3(Operator):
#     def __init__(self):
#         self.formula = '({}**3)'
#         self.__name__ = 'pow3'
#         self.sign = '**3'

#     def __call__(self, x):
#         res = x**3
#         return np.clip(res, min_value, max_value)

class Div(Operator):
    def __init__(self):
        self.formula = '({}/{})'
        self.__name__ = 'div'
        self.sign = '/'

    def __call__(self, x, y):
        if type(y) == np.ndarray:
            res = np.divide(x, y, out=np.ones_like(
                y, dtype=np.float64), where=y != 0)
        else:
            if y == 0:
                res = 0
            else:
                res = x/y
        return np.clip(res, min_value, max_value)

# class FloorDiv(Operator):
#     def __init__(self):
#         self.formula = '({}//{})'
#         self.__name__ = 'floor_div'
#         self.sign = '//'

#     def __call__(self, x, y):
#         if type(y) == np.ndarray:
#             res = np.floor_divide(x, y, out=np.ones_like(
#                 y, dtype=np.float64), where=y != 0)
#         else:
#             if y == 0:
#                 res = 0
#             else:
#                 res = x//y
#         return np.clip(res, min_value, max_value)`


class Inv(Operator):
    def __init__(self):
        self.formula = '(1/{})'
        self.__name__ = 'Ind'
        self.sign = '1/'

    def __call__(self, y):
        if type(y) == np.ndarray:
            res = np.divide(1, y, out=np.ones_like(
                y, dtype=np.float64), where=y != 0)
        else:
            if y == 0:
                res = 1
            else:
                res = 1/y
        return np.clip(res, min_value, max_value)


class LogAbs(Operator):
    def __init__(self):
        self.formula = 'log(abs({}))'
        self.__name__ = 'log(abs)'
        self.sign = 'log(abs)'

    def __call__(self, y):
        y_ = np.abs(y)
        if type(y_) == np.ndarray:
            res = np.log(y_, out=np.ones_like(
                y_, dtype=np.float64), where=y_ != 0)
        else:
            if y_ == 0:
                res = 1
            else:
                res = np.log(y_)
        return np.clip(res, min_value, max_value)


class Exp(Operator):
    def __init__(self):
        self.formula = 'exp({})'
        self.__name__ = 'exp'
        self.sign = 'exp'

    def __call__(self, x):
        return np.clip(np.exp(x), min_value, max_value)


class Mul3(Operator):
    def __init__(self):
        self.formula = '({} * {} * {})'
        self.__name__ = 'mul3'
        self.sign = '*'

    def __call__(self, x, y, z):
        return np.clip(x * y * z, min_value, max_value)


class SqrtAbs(Operator):
    def __init__(self):
        self.formula = 'sqrt(abs({}))'
        self.__name__ = 'sqrt(abs)'
        self.sign = 'sqrt(abs)'

    def __call__(self, x):
        return np.clip(np.sqrt(np.abs(x)), min_value, max_value)


# class Abs(Operator):
#     def __init__(self):
#             self.formula = 'abs({})'
#             self.__name__ = 'abs()'
#             self.sign = 'abs()'

#     def __call__(self, x):
#         return np.clip(np.abs(x), min_value, max_value)
