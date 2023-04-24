import numpy as np
from ..base import Tree
from ..base import UniversalSet
from ..base import FunctionalNode
from .generators import growing_method
from .generators import sattolo_shuffle
from .transformations import protect_norm
from .transformations import common_region
import random
from .numba_funcs import select_quantity_id_by_tournament
from typing import Union


min_value = np.finfo(np.float64).min
max_value = np.finfo(np.float64).max


################################## MUTATUIONS ##################################
# genetic algorithm
def flip_mutation(individ: np.ndarray,
                  proba: float) -> np.ndarray:
    mask = np.random.random(size=individ.shape) < proba
    individ[mask] = 1 - individ[mask]
    return individ


# differential evolution
def best_1(current: np.ndarray,
           population: np.ndarray,
           F: float) -> np.ndarray:
    best = population[-1]
    range_ = np.arange(len(population))
    r1, r2 = np.random.choice(range_, size=2, replace=False)

    offspring = best + F*(population[r1] - population[r2])
    return offspring


def rand_1(current: np.ndarray,
           population: np.ndarray,
           F: float) -> np.ndarray:
    range_ = np.arange(len(population))
    r1, r2, r3 = np.random.choice(range_, size=3, replace=False)

    offspring = population[r3] + F*(population[r1] - population[r2])
    return offspring


def rand_to_best1(current: np.ndarray,
                  population: np.ndarray,
                  F: float) -> np.ndarray:
    best = population[-1]
    range_ = np.arange(len(population))
    r1, r2, r3 = np.random.choice(range_, size=3, replace=False)

    offspring = population[r1] +\
        F*(best - population[r1]) + F*(population[r2] - population[r3])
    return offspring


def current_to_best_1(current: np.ndarray,
                      population: np.ndarray,
                      F: float) -> np.ndarray:
    best = population[-1]
    range_ = np.arange(len(population))
    r1, r2 = np.random.choice(range_, size=2, replace=False)

    offspring = current + F*(best - current) +\
        F*(population[r1] - population[r2])
    return offspring


def best_2(current: np.ndarray,
           population: np.ndarray,
           F: float) -> np.ndarray:
    best = population[-1]
    range_ = np.arange(len(population))
    r1, r2, r3, r4 = np.random.choice(range_, size=4, replace=False)

    offspring = best + F*(population[r1] - population[r2]) +\
        F*(population[r3] - population[r4])
    return offspring


def rand_2(current: np.ndarray,
           population: np.ndarray,
           F: float) -> np.ndarray:
    range_ = np.arange(len(population))
    r1, r2, r3, r4, r5 = np.random.choice(range_, size=5, replace=False)

    offspring = population[r5] + F*(population[r1] - population[r2]) +\
        F*(population[r3] - population[r4])
    return offspring


def current_to_pbest_1(current: np.ndarray,
                       population: np.ndarray,
                       F: float) -> np.ndarray:
    range_ = np.arange(len(population))
    p_min = 2/len(population)
    p_i = np.random.uniform(p_min, 0.2)

    value = int(p_i*len(population))
    pbest = population[-value:]
    p_best_ind = np.random.randint(0, len(pbest))

    best = pbest[p_best_ind]

    r1, r2 = np.random.choice(range_, size=2, replace=False)

    offspring = current + F*(best - current) +\
        F*(population[r1] - population[r2])
    return offspring


def current_to_rand_1(current: np.ndarray,
                      population: np.ndarray,
                      F: float) -> np.ndarray:
    range_ = np.arange(len(population))
    r1, r2, r3 = np.random.choice(range_, size=3, replace=False)

    offspring = population[r1] + F*(population[r3] - current) +\
        F*(population[r1] - population[r2])
    return offspring


# genetic propramming
def point_mutation(tree: Tree,
                   uniset: UniversalSet,
                   proba: float,
                   max_level) -> Tree:
    to_return = tree.copy()
    if random.random() < proba:
        i = random.randrange(len(to_return))
        if to_return._nodes[i].is_functional():
            n_args = to_return._nodes[i]._n_args
            new_node = uniset._random_functional(n_args)
        else:
            new_node = uniset._random_terminal_or_ephemeral()
        to_return._nodes[i] = new_node
    return to_return


def growing_mutation(tree: Tree,
                     uniset: UniversalSet,
                     proba: float,
                     max_level) -> Tree:
    to_return = tree.copy()
    if random.random() < proba:
        i = random.randrange(len(to_return))
        # print(to_return._nodes)
        # print(to_return._n_args)
        grown_tree = growing_method(uniset,  max(to_return.get_levels(i)))
        to_return = to_return.concat(i, grown_tree)
    return to_return


def swap_mutation(tree: Tree,
                  uniset: UniversalSet,
                  proba: float,
                  max_level) -> Tree:
    to_return = tree.copy()
    if random.random() < proba:
        more_one_args_cond = to_return._n_args > 1
        indexes = np.arange(len(tree), dtype=int)[more_one_args_cond]
        if len(indexes) > 0:
            i = random.choice(indexes)
            args_id = to_return.get_args_id(i)
            new_arg_id = args_id.copy()
            sattolo_shuffle(new_arg_id)
            for old_j, new_j in zip(args_id, new_arg_id):
                subtree = tree.subtree(old_j, return_class=True)
                to_return = to_return.concat(new_j, subtree)
    return to_return


def shrink_mutation(tree: Tree,
                    uniset: UniversalSet,
                    proba: float,
                    max_level) -> Tree:
    to_return = tree.copy()
    if len(to_return) > 2:
        if random.random() < proba:
            no_terminal_cond = to_return._n_args > 0
            indexes = np.arange(len(tree), dtype=int)[no_terminal_cond]
            if len(indexes) > 0:
                i = random.choice(indexes)
                args_id = to_return.get_args_id(i)
                choosen_id = random.choice(args_id)
                to_return = to_return.concat(i, tree.subtree(
                    choosen_id, return_class=True))
    return to_return


################################## CROSSOVERS ##################################
# genetic algorithm
def empty_crossover(individs: np.ndarray,
                    fitness: np.ndarray,
                    rank: np.ndarray,
                    args) -> np.ndarray:
    offspring = individs[0].copy()
    return offspring


def one_point_crossover(individs: np.ndarray,
                        fitness: np.ndarray,
                        rank: np.ndarray) -> np.ndarray:
    cross_point = random.randrange(len(individs[0]))
    slice_ = slice(0, cross_point)

    if random.random() > 0.5:
        offspring = individs[0].copy()
        offspring[slice_] = individs[1][slice_].copy()
    else:
        offspring = individs[1].copy()
        offspring[slice_] = individs[0][slice_].copy()
    return offspring


def two_point_crossover(individs: np.ndarray,
                        fitness: np.ndarray,
                        rank: np.ndarray) -> np.ndarray:
    c_points = sorted(random.sample(range(len(individs[0])), k=2))
    slice_ = slice(c_points[0], c_points[1])

    if random.random() > 0.5:
        offspring = individs[0].copy()
        offspring[slice_] = individs[1][slice_].copy()
    else:
        offspring = individs[1].copy()
        offspring[slice_] = individs[0][slice_].copy()
    return offspring


def uniform_crossover(individs: np.ndarray,
                      fitness: np.ndarray,
                      rank: np.ndarray) -> np.ndarray:
    range_ = np.arange(len(individs))
    diag = np.arange(len(individs[0]))

    choosen = np.random.choice(range_, size=len(individs[0]))
    offspring = individs[choosen, diag].copy()
    return offspring


def uniform_prop_crossover(individs: np.ndarray,
                           fitness: np.ndarray,
                           rank: np.ndarray) -> np.ndarray:
    range_ = np.arange(len(individs))
    diag = np.arange(len(individs[0]))

    probability = protect_norm(fitness)
    choosen = np.random.choice(range_, size=len(individs[0]), p=probability)
    offspring = individs[choosen, diag].copy()
    return offspring


def uniform_rank_crossover(individs: np.ndarray,
                           fitness: np.ndarray,
                           rank: np.ndarray) -> np.ndarray:
    range_ = np.arange(len(individs))
    diag = np.arange(len(individs[0]))

    probability = protect_norm(rank)
    choosen = np.random.choice(range_, size=len(individs[0]), p=probability)
    offspring = individs[choosen, diag].copy()
    return offspring


def uniform_tour_crossover(individs: np.ndarray,
                           fitness: np.ndarray,
                           rank: np.ndarray) -> np.ndarray:
    range_ = np.arange(len(individs))
    diag = np.arange(len(individs[0]))

    tournament = np.random.choice(range_, 2*len(individs[0]))
    tournament = tournament.reshape(-1, 2)
    choosen = np.argmin(fitness[tournament], axis=1)
    offspring = individs[choosen, diag].copy()
    return offspring


# differential evolution
def binomial(individ: np.ndarray,
             mutant: np.ndarray,
             CR: float) -> np.ndarray:
    individ = individ.copy()

    j = random.randrange(len(individ))
    mask_random = np.random.random(len(individ)) <= CR
    mask_j = np.arange(len(individ)) == j
    mask_union = mask_random | mask_j
    individ[mask_union] = mutant[mask_union].copy()
    offspring = individ.copy()
    return offspring


# genetic propramming
def standart_crossover(individs: np.ndarray,
                       fitness: np.ndarray,
                       rank: np.ndarray,
                       max_level: int) -> Tree:
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


def one_point_crossoverGP(individs: np.ndarray,
                          fitness: np.ndarray,
                          rank: np.ndarray,
                          max_level: int) -> Tree:
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


def uniform_crossoverGP(individs: np.ndarray,
                        fitness: np.ndarray,
                        rank: np.ndarray,
                        max_level: int) -> Tree:
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
                new_nodes.extend(individ_1._nodes[left:right])
                new_n_args.extend(individ_1._n_args[left:right])
            else:
                id_ = common_indexes[1][i]
                left, right = individ_2.subtree(index=id_)
                new_nodes.extend(individ_2._nodes[left:right])
                new_n_args.extend(individ_2._n_args[left:right])
        else:
            if random.random() < 0.5:
                id_ = common_indexes[0][i]
                new_nodes.append(individ_1._nodes[id_])
                new_n_args.append(individ_1._n_args[id_])
            else:
                id_ = common_indexes[1][i]
                new_nodes.append(individ_2._nodes[id_])
                new_n_args.append(individ_2._n_args[id_])
    to_return = Tree(new_nodes.copy(),
                     new_n_args.copy())
    return to_return


def uniform_crossoverGP_prop(individs: np.ndarray,
                             fitness: np.ndarray,
                             rank: np.ndarray,
                             max_level: int) -> Tree:
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
            to_return._nodes.extend(subtree._nodes)
            new_n_args.extend(subtree._n_args)
        else:
            to_return._nodes.append(individs[j]._nodes[id_])
            new_n_args.append(individs[j]._n_args[id_])

    to_return = to_return.copy()
    to_return._n_args = new_n_args.copy()
    return to_return


def uniform_crossoverGP_rank(individs: np.ndarray,
                             fitness: np.ndarray,
                             rank: np.ndarray,
                             max_level: int) -> Tree:
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
            to_return._nodes.extend(subtree._nodes)
            new_n_args.extend(subtree._n_args)
        else:
            to_return._nodes.append(individs[j]._nodes[id_])
            new_n_args.append(individs[j]._n_args[id_])

    to_return = to_return.copy()
    to_return._n_args = new_n_args.copy()
    return to_return


def uniform_crossoverGP_tour(individs: np.ndarray,
                             fitness: np.ndarray,
                             rank: np.ndarray,
                             max_level: int) -> Tree:
    to_return = Tree([], [])
    new_n_args = []
    common, border = common_region(individs)

    for i, common_0_i in enumerate(common[0]):
        j = tournament_selection(fitness, rank, 2, 1)[0]
        id_ = common[j][i]
        if common_0_i in border[0]:
            subtree = individs[j].subtree(id_, return_class=True)
            to_return._nodes.extend(subtree._nodes)
            new_n_args.extend(subtree._n_args)
        else:
            to_return._nodes.append(individs[j]._nodes[id_])
            new_n_args.append(individs[j]._n_args[id_])

    to_return = to_return.copy()
    to_return._n_args = new_n_args.copy()
    return to_return


################################## SELECTIONS ##################################
# genetic algorithm
def proportional_selection(fitness: np.ndarray,
                           rank: np.ndarray,
                           tour_size: int,
                           quantity: int) -> np.ndarray:
    proba_fitness = protect_norm(fitness)
    range_ = np.arange(len(proba_fitness))

    choosen = np.random.choice(range_, size=quantity, p=proba_fitness)
    return choosen


def rank_selection(fitness: np.ndarray,
                   rank: np.ndarray,
                   tour_size: int,
                   quantity: int) -> np.ndarray:
    proba_rank = protect_norm(rank)
    range_ = np.arange(len(proba_rank))

    choosen = np.random.choice(range_, size=quantity, p=proba_rank)
    return choosen


def tournament_selection(fitness: np.ndarray,
                         rank: np.ndarray,
                         tour_size: int,
                         quantity: int) -> np.ndarray:
    tournament = np.random.randint(len(fitness),
                                   size=tour_size*quantity)
    tournament = tournament.reshape(-1, tour_size)
    max_fit_id = np.argmax(fitness[tournament], axis=1)
    choosen = tournament[np.arange(quantity), max_fit_id]
    return choosen


def tournament_selection_(fitness: np.ndarray,
                          rank: np.ndarray,
                          tour_size: int,
                          quantity: int) -> np.ndarray:
    return select_quantity_id_by_tournament(fitness.astype(np.float32),
                                            np.int32(tour_size),
                                            np.int32(quantity))


##################################### GP MATH #####################################
class Operator:
    def _write(self,
               *args) -> str:
        formula = self._formula.format(*args)
        return formula


class Cos(Operator):
    def __init__(self) -> None:
        self._formula = 'cos({})'
        self.__name__ = 'cos'
        self._sign = 'cos'

    def __call__(self,
                 x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = np.cos(x)
        return result


class Sin(Operator):
    def __init__(self) -> None:
        self._formula = 'sin({})'
        self.__name__ = 'sin'
        self._sign = 'sin'

    def __call__(self,
                 x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = np.sin(x)
        return result


class Add(Operator):
    def __init__(self) -> None:
        self._formula = '({} + {})'
        self.__name__ = 'add'
        self._sign = '+'

    def __call__(self,
                 x: Union[float, np.ndarray],
                 y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = x + y
        return result


class Sub(Operator):
    def __init__(self) -> None:
        self._formula = '({} - {})'
        self.__name__ = 'sub'
        self._sign = '-'

    def __call__(self,
                 x: Union[float, np.ndarray],
                 y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = x - y
        return result


class Neg(Operator):
    def __init__(self) -> None:
        self._formula = '-{}'
        self.__name__ = 'neg'
        self._sign = '-'

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = -x
        return result


class Mul(Operator):
    def __init__(self) -> None:
        self._formula = '({} * {})'
        self.__name__ = 'mul'
        self._sign = '*'

    def __call__(self,
                 x: Union[float, np.ndarray],
                 y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = x * y
        return result


class Pow2(Operator):
    def __init__(self) -> None:
        self._formula = '({}**2)'
        self.__name__ = 'pow2'
        self._sign = '**2'

    def __call__(self,
                 x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = np.clip(x**2, min_value, max_value)
        return result


class Div(Operator):
    def __init__(self) -> None:
        self._formula = '({}/{})'
        self.__name__ = 'div'
        self._sign = '/'

    def __call__(self,
                 x: Union[float, np.ndarray],
                 y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if type(y) == np.ndarray:
            result = np.divide(x, y, out=np.ones_like(
                y, dtype=np.float64), where=y != 0)
        else:
            if y == 0:
                result = 0
            else:
                result = x/y
        result = np.clip(result, min_value, max_value)
        return result


class Inv(Operator):
    def __init__(self) -> None:
        self._formula = '(1/{})'
        self.__name__ = 'Inv'
        self._sign = '1/'

    def __call__(self,
                 y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if type(y) == np.ndarray:
            result = np.divide(1, y, out=np.ones_like(
                y, dtype=np.float64), where=y != 0)
        else:
            if y == 0:
                result = 1
            else:
                result = 1/y
        result = np.clip(result, min_value, max_value)
        return result


class LogAbs(Operator):
    def __init__(self) -> None:
        self._formula = 'log(abs({}))'
        self.__name__ = 'log(abs)'
        self._sign = 'log(abs)'

    def __call__(self,
                 y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        y_ = np.abs(y)
        if type(y_) == np.ndarray:
            result = np.log(y_, out=np.ones_like(
                y_, dtype=np.float64), where=y_ != 0)
        else:
            if y_ == 0:
                result = 1
            else:
                result = np.log(y_)
        return result


class Exp(Operator):
    def __init__(self) -> None:
        self._formula = 'exp({})'
        self.__name__ = 'exp'
        self._sign = 'exp'

    def __call__(self,
                 x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = np.clip(np.exp(x), min_value, max_value)
        return result


class SqrtAbs(Operator):
    def __init__(self) -> None:
        self._formula = 'sqrt(abs({}))'
        self.__name__ = 'sqrt(abs)'
        self._sign = 'sqrt(abs)'

    def __call__(self,
                 x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = np.sqrt(np.abs(x))
        return result


class Abs(Operator):
    def __init__(self) -> None:
        self._formula = 'abs({})'
        self.__name__ = 'abs()'
        self._sign = 'abs()'

    def __call__(self,
                 x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        result = np.abs(x)
        return result
