import numpy as np
from ..optimizers._base import Tree
from ..optimizers._base import FunctionalNode
from .generators import growing_method
from .transformations import protect_norm
from .transformations import common_region
import random


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
                   proba_down, max_level):
    to_return = some_tree.copy()

    proba = proba_down/len(to_return.nodes)
    for i, node in enumerate(to_return.nodes):
        if np.random.random() < proba:
            if type(node) != FunctionalNode:
                new_node = uniset.mutate_terminal()
            else:
                new_node = uniset.mutate_functional(node.n_args)
            to_return.nodes[i] = new_node

    return to_return


def growing_mutation(some_tree, uniset,
                     proba_down, max_level):
    to_return = some_tree.copy()
    proba = proba_down/len(to_return.nodes)
    if np.random.random() < proba:
        i = np.random.randint(0, len(to_return.nodes))
        left, right = to_return.subtree(i)
        max_level_i = max_level - to_return.levels[left:right][0]
        new_tree = growing_method(uniset, max_level_i)
        to_return = to_return.concat(i, new_tree)

    return to_return


def simplify_mutation(some_tree, uniset,
                      proba_down, max_level):
    some_tree = some_tree.copy()
    proba = proba_down/len(some_tree.nodes)
    if np.random.random() < proba:
        i = np.random.randint(0, len(some_tree.nodes))
        to_return = some_tree.simplify_by_index(i)[0]
    else:
        to_return = some_tree
    return to_return


def swap_mutation(some_tree, uniset,
                  proba_down, max_level):
    to_return = some_tree.copy()
    proba = proba_down/len(to_return.nodes)
    if np.random.random() < proba:
        indexes = [i for i, nodes in enumerate(
            to_return.nodes) if nodes.n_args > 1]
        if len(indexes) > 0:
            i = random.choice(indexes)
            args_id = to_return.get_args_id(i)
            new_arg_id = args_id.copy()
            np.random.shuffle(new_arg_id)

            for old_j, new_j in zip(args_id, new_arg_id):
                subtree = some_tree.subtree(old_j, return_class=True)
                to_return = to_return.concat(new_j, subtree)

            to_return.levels = to_return.get_levels()

    return to_return


# def shrink_mutation(some_tree, uniset,
#                     proba_down, max_level):

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
    first_point = np.random.randint(0,  len(individ_1.nodes))
    second_point = np.random.randint(0,  len(individ_2.nodes))

    if np.random.random() < 0.5:
        left, right = individ_1.subtree(first_point)
        first_subtree = Tree(individ_1.nodes[left:right],
                             individ_1.levels[left:right])
        offspring = individ_2.concat(second_point, first_subtree)
        if offspring.get_max_level() > max_level:
            offspring = individ_2
    else:
        left, right = individ_2.subtree(second_point)
        second_subtree = Tree(individ_2.nodes[left:right],
                              individ_2.levels[left:right])
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
        left, right = individ_1.subtree(first_point)
        first_subtree = Tree(individ_1.nodes[left:right],
                             individ_1.levels[left:right])
        offspring = individ_2.concat(second_point, first_subtree)
    else:
        left, right = individ_2.subtree(second_point)
        second_subtree = Tree(individ_2.nodes[left:right],
                              individ_2.levels[left:right])
        offspring = individ_1.concat(first_point, second_subtree)
    return offspring


def uniform_crossoverGP(individs, fitness, rank, max_level):
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
    to_return = Tree(new_nodes.copy(), None)
    to_return.levels = to_return.get_levels()
    return to_return


# def uniform_crossoverGP2(individs, fitness, rank, max_level):


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


class Add(Operator):
    def __init__(self):
        self.formula = '({} + {})'
        self.__name__ = 'add'
        self.sign = '+'

    def __call__(self, x, y):
        x = np.clip(x, -1e3, 1e3)
        y = np.clip(y, -1e3, 1e3)
        return x + y


class Sub(Operator):
    def __init__(self):
        self.formula = '({} - {})'
        self.__name__ = 'sub'
        self.sign = '-'

    def __call__(self, x, y):
        x = np.clip(x, -1e3, 1e3)
        y = np.clip(y, -1e3, 1e3)
        return x - y


class Neg(Operator):
    def __init__(self):
        self.formula = '-{}'
        self.__name__ = 'neg'
        self.sign = '-'

    def __call__(self, x):
        return -x


class Mul(Operator):
    def __init__(self):
        self.formula = '({} * {})'
        self.__name__ = 'mul'
        self.sign = '*'

    def __call__(self, x, y):
        x = np.clip(x, -1e3, 1e3)
        y = np.clip(y, -1e3, 1e3)
        return x * y


class Mul3(Operator):
    def __init__(self):
        self.formula = '({} * {} * {})'
        self.__name__ = 'mul3'
        self.sign = '*'

    def __call__(self, x, y, z):
        x = np.clip(x, -1e3, 1e3)
        y = np.clip(y, -1e3, 1e3)
        z = np.clip(z, -1e3, 1e3)
        return x * y * z


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


class Pow(Operator):
    def __init__(self):
        self.formula = '({}**{})'
        self.__name__ = 'pow'
        self.sign = '**'

    def __call__(self, x, y):
        # проверка если степень от 0 до 1, то под степенью должно быть положительное иначе
        x = np.clip(x, -1e3, 1e3)
        y = np.clip(y, -1e1, 1e1)

        return np.abs(x)**np.abs(y)


class Sqrt(Operator):
    def __init__(self):
        self.formula = 'sqrt({})'
        self.__name__ = 'sqrt'
        self.sign = 'sqrt'

    def __call__(self, x):
        return np.sqrt(np.abs(x))


class Exp(Operator):
    def __init__(self):
        self.formula = 'np.exp({})'
        self.__name__ = 'exp'
        self.sign = 'exp'

    def __call__(self, x):
        x = np.clip(x, -np.inf, 1.7976931348623157e+308)
        return np.exp(x)


class Div(Operator):
    def __init__(self):
        self.formula = '({}/{})'
        self.__name__ = 'div'
        self.sign = '/'

    def __call__(self, x, y):
        x = np.clip(x, -1e3, 1e3)
        y = np.clip(y, -1e3, 1e3)
        if type(y) == np.ndarray:
            res = np.divide(x, y, out=np.zeros_like(
                y, dtype=np.float64), where=y != 0)
        else:
            if y == 0:
                res = 1
            else:
                res = x/y
        return res
