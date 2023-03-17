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
from .numba_funcs import select_quantity_id_by_tournament


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
    r1, r2 = np.random.choice(
        np.arange(len(population)), size=2, replace=False)
    return best + F_value*(population[r1] - population[r2])


def rand_1(current, population, F_value):
    r1, r2, r3 = np.random.choice(
        np.arange(len(population)), size=3, replace=False)
    return population[r3] + F_value*(population[r1] - population[r2])


def rand_to_best1(current, population, F_value):
    best = population[-1]
    r1, r2, r3 = np.random.choice(
        np.arange(len(population)), size=3, replace=False)
    return population[r1] + F_value*(
        best - population[r1]) + F_value*(population[r2] - population[r3])


def current_to_best_1(current, population, F_value):
    best = population[-1]
    r1, r2 = np.random.choice(
        np.arange(len(population)), size=2, replace=False)
    return current + F_value*(best - current) +\
        F_value*(population[r1] - population[r2])


def best_2(current, population, F_value):
    best = population[-1]
    r1, r2, r3, r4 = np.random.choice(
        np.arange(len(population)), size=4, replace=False)
    return best + F_value*(population[r1] - population[r2]) +\
        F_value*(population[r3] - population[r4])


def rand_2(current, population, F_value):
    r1, r2, r3, r4, r5 = np.random.choice(
        np.arange(len(population)), size=5, replace=False)
    return population[r5] + F_value*(population[r1] - population[r2]) +\
        F_value*(population[r3] - population[r4])


def current_to_pbest_1(current, population, F_value):
    p_min = 2/len(population)
    p_i = np.random.uniform(p_min, 0.2)

    value = int(p_i*len(population))
    pbest = population[-value:]
    p_best_ind = np.random.randint(0, len(pbest))

    best = pbest[p_best_ind]

    r1, r2 = np.random.choice(
        np.arange(len(population)), size=2, replace=False)
    return current + F_value*(best - current) +\
        F_value*(population[r1] - population[r2])


def current_to_rand_1(current, population, F_value):
    r1, r2, r3 = np.random.choice(
        np.arange(len(population)), size=3, replace=False)
    return population[r1] + F_value*(population[r3] - current) +\
        F_value*(population[r1] - population[r2])


# genetic propramming
def point_mutation(some_tree, uniset,
                   proba, max_level):
    to_return = some_tree.copy()
    if random.random() < proba:
        i = random.randrange(len(to_return))
        if type(to_return.nodes[i]) != FunctionalNode:
            new_node = uniset.random_terminal_or_ephemeral()
        else:
            n_args = to_return.nodes[i].n_args
            new_node = uniset.random_functional(n_args)
        to_return.nodes[i] = new_node

    return to_return


def growing_mutation(some_tree, uniset,
                     proba, max_level):
    to_return = some_tree.copy()
    if random.random() < proba:

        i = random.randrange(len(to_return))
        max_level - max(some_tree.get_levels(i))
        new_tree = growing_method(uniset,  max(to_return.get_levels(i)))
        to_return = to_return.concat(i, new_tree)

    return to_return


def swap_mutation(some_tree, uniset,
                  proba, max_level):
    to_return = some_tree.copy()
    if random.random() < proba:
        indexes = np.arange(len(some_tree))[to_return.n_args > 1]
        if len(indexes) > 0:
            i = random.choice(indexes)
            args_id = to_return.get_args_id(i)
            new_arg_id = args_id.copy()
            sattolo_shuffle(new_arg_id)
            for old_j, new_j in zip(args_id, new_arg_id):
                subtree = some_tree.subtree(old_j, return_class=True)
                to_return = to_return.concat(new_j, subtree)

    return to_return


def shrink_mutation(some_tree, uniset,
                    proba, max_level):
    to_return = some_tree.copy()
    if len(to_return) > 2:
        if random.random() < proba:
            indexes = np.arange(len(some_tree))[to_return.n_args > 0]
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
    cross_point = random.randrange(len(individs[0]))
    slice_ = slice(0, cross_point)

    if random.random() > 0.5:
        offspring = individs[0].copy()
        offspring[slice_] = individs[1][slice_].copy()
    else:
        offspring = individs[1].copy()
        offspring[slice_] = individs[0][slice_].copy()
    return offspring


def two_point_crossover(individs, fitness, rank):
    c_points = sorted(random.sample(range(len(individs[0])), k=2))
    slice_ = slice(c_points[0], c_points[1])

    if random.random() > 0.5:
        offspring = individs[0].copy()
        offspring[slice_] = individs[1][slice_].copy()
    else:
        offspring = individs[1].copy()
        offspring[slice_] = individs[0][slice_].copy()
    return offspring


def uniform_crossover(individs, fitness, rank):
    choosen = np.random.choice(np.arange(len(individs)), size=len(individs[0]))
    return individs[choosen, np.arange(len(individs[0]))]


def uniform_prop_crossover(individs, fitness, rank):
    probability = protect_norm(fitness)
    choosen = np.random.choice(np.arange(len(individs)),
                               size=len(individs[0]),
                               p=probability)
    return individs[choosen, np.arange(len(individs[0]))]


def uniform_rank_crossover(individs, fitness, rank):
    probability = protect_norm(rank)
    choosen = np.random.choice(np.arange(len(individs)),
                               size=len(individs[0]),
                               p=probability)
    return individs[choosen, np.arange(len(individs[0]))]


def uniform_tour_crossover(individs, fitness, rank):
    tournament = np.random.choice(np.arange(len(individs)), 2*len(individs[0]))
    tournament = tournament.reshape(-1, 2)

    choosen = np.argmin(fitness[tournament], axis=1)
    return individs[choosen, np.arange(len(individs[0]))]


# differential evolution
def binomial(individ, mutant, CR):
    individ = individ.copy()
    j = random.randrange(len(individ))
    mask_random = np.random.random(len(individ)) <= CR
    mask_j = np.arange(len(individ)) == j
    mask_union = mask_random | mask_j
    individ[mask_union] = mutant[mask_union].copy()
    return individ


# genetic propramming
def standart_crossover(individs, fitness, rank, max_level):
    individ_1 = individs[0].copy()
    individ_2 = individs[1].copy()
    first_point = random.randrange(len(individ_1))
    second_point = random.randrange(len(individ_2))

    if random.random() < 0.5:
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


def one_point_crossoverGP(individs, fitness, rank, max_level):
    individ_1 = individs[0]
    individ_2 = individs[1]
    common_indexes, _ = common_region([individ_1, individ_2])

    point = random.randrange(len(common_indexes[0]))
    first_point = common_indexes[0][point]
    second_point = common_indexes[1][point]
    if random.random() < 0.5:
        first_subtree = individ_1.subtree(first_point, return_class=True)
        offspring = individ_2.concat(second_point, first_subtree)
    else:
        second_subtree = individ_2.subtree(second_point, return_class=True)
        offspring = individ_1.concat(first_point, second_subtree)
    return offspring


def uniform_crossoverGP(individs, fitness, rank, max_level):
    '''Poli, Riccardo & Langdon, W.. (2001). On the Search
    Properties of Different Crossover Operators in Genetic Programming. '''
    new_nodes = []
    new_n_args = []
    individ_1 = individs[0]
    individ_2 = individs[1]
    common_indexes, border = common_region([individ_1, individ_2])

    for i in range(len(common_indexes[0])):
        if common_indexes[0][i] in border[0]:
            if random.random() < 0.5:
                id_ = common_indexes[0][i]
                left, right = individ_1.subtree(index=id_)
                new_nodes.extend(individ_1.nodes[left:right])
                new_n_args.extend(individ_1.n_args[left:right])
            else:
                id_ = common_indexes[1][i]
                left, right = individ_2.subtree(index=id_)
                new_nodes.extend(individ_2.nodes[left:right])
                new_n_args.extend(individ_2.n_args[left:right])
        else:
            if random.random() < 0.5:
                id_ = common_indexes[0][i]
                new_nodes.append(individ_1.nodes[id_])
                new_n_args.append(individ_1.n_args[id_])
            else:
                id_ = common_indexes[1][i]
                new_nodes.append(individ_2.nodes[id_])
                new_n_args.append(individ_2.n_args[id_])
    to_return = Tree(new_nodes.copy(),
                     np.array(new_n_args.copy(), dtype=np.int32))
    return to_return


def uniform_crossoverGP_prop(individs, fitness, rank, max_level):
    to_return = Tree([], [])
    new_n_args = []
    common, border = common_region(individs)
    range_ = range(len(individs))
    probability = protect_norm(fitness)
    for i, common_0_i in enumerate(common[0]):
        j = random.choices(range_, weights=probability)[0]
        id_ = common[j][i]
        if common_0_i in border[0]:
            subtree = individs[j].subtree(id_, return_class=True)
            to_return.nodes.extend(subtree.nodes)
            new_n_args.extend(subtree.n_args)
        else:
            to_return.nodes.append(individs[j].nodes[id_])
            new_n_args.append(individs[j].n_args[id_])

    to_return = to_return.copy()
    to_return.n_args = np.array(new_n_args.copy(), dtype=np.int32)

    return to_return


def uniform_crossoverGP_rank(individs, fitness, rank, max_level):
    to_return = Tree([], [])
    new_n_args = []
    common, border = common_region(individs)
    range_ = range(len(individs))
    probability = protect_norm(rank)
    for i, common_0_i in enumerate(common[0]):
        j = random.choices(range_, weights=probability)[0]
        id_ = common[j][i]
        if common_0_i in border[0]:
            subtree = individs[j].subtree(id_, return_class=True)
            to_return.nodes.extend(subtree.nodes)
            new_n_args.extend(subtree.n_args)
        else:
            to_return.nodes.append(individs[j].nodes[id_])
            new_n_args.append(individs[j].n_args[id_])

    to_return = to_return.copy()
    to_return.n_args = np.array(new_n_args.copy(), dtype=np.int32)

    return to_return


def uniform_crossoverGP_tour(individs, fitness, rank, max_level):
    to_return = Tree([], [])
    new_n_args = []
    common, border = common_region(individs)
    for i, common_0_i in enumerate(common[0]):
        j = tournament_selection(fitness, rank, 2, 1)[0]
        id_ = common[j][i]
        if common_0_i in border[0]:
            subtree = individs[j].subtree(id_, return_class=True)
            to_return.nodes.extend(subtree.nodes)
            new_n_args.extend(subtree.n_args)
        else:
            to_return.nodes.append(individs[j].nodes[id_])
            new_n_args.append(individs[j].n_args[id_])

    to_return = to_return.copy()
    to_return.n_args = np.array(new_n_args.copy(), dtype=np.int32)

    return to_return


################################## SELECTIONS ##################################
# genetic algorithm
def proportional_selection(proba_fitness, proba_rank, tour_size, quantity):
    choosen = np.random.choice(np.arange(len(proba_fitness)),
                               size=quantity, p=proba_fitness)
    return choosen


def rank_selection(proba_fitness, proba_rank, tour_size, quantity):
    choosen = np.random.choice(np.arange(len(proba_rank)),
                               size=quantity, p=proba_rank)
    return choosen


def tournament_selection(fitness, ranks,
                         tour_size, quantity):
    tournament = np.random.randint(len(fitness),
                                   size=tour_size*quantity)
    tournament = tournament.reshape(-1, tour_size)
    max_fit_id = np.argmax(fitness[tournament], axis=1)
    return tournament[np.arange(quantity), max_fit_id]


def tournament_selection_(fitness, ranks, tour_size, quantity):
    return select_quantity_id_by_tournament(fitness.astype(np.float32),
                                            np.int32(tour_size),
                                            np.int32(quantity))


##################################### GP MATH #####################################
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
        return x + y


class Sub(Operator):
    def __init__(self):
        self.formula = '({} - {})'
        self.__name__ = 'sub'
        self.sign = '-'

    def __call__(self, x, y):
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
        return x * y


class Pow2(Operator):
    def __init__(self):
        self.formula = '({}**2)'
        self.__name__ = 'pow2'
        self.sign = '**2'

    def __call__(self, x):
        res = x**2
        return np.clip(res, min_value, max_value)


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
        return res


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
        return res


class Exp(Operator):
    def __init__(self):
        self.formula = 'exp({})'
        self.__name__ = 'exp'
        self.sign = 'exp'

    def __call__(self, x):
        to_return = np.exp(x)
        return np.clip(to_return, min_value, max_value)


class SqrtAbs(Operator):
    def __init__(self):
        self.formula = 'sqrt(abs({}))'
        self.__name__ = 'sqrt(abs)'
        self.sign = 'sqrt(abs)'

    def __call__(self, x):
        return np.sqrt(np.abs(x))


class Abs(Operator):
    def __init__(self):
        self.formula = 'abs({})'
        self.__name__ = 'abs()'
        self.sign = 'abs()'

    def __call__(self, x):
        return np.abs(x)


class Sigma(Operator):
    def __init__(self):
        self.formula = 'sigma({})'
        self.__name__ = 'sigma()'
        self.sign = 'sigma()'

    def __call__(self, x):
        return 1/(1 + np.exp(-x))
