import numpy as np
from ..tools import protect_norm
from ._base import Tree


def empty_crossover(individs, fitness, rank):
    return np.random.choice(individs)[0]


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


def binomial(individ, mutant, CR):
    individ = individ.copy()
    j = np.random.choice(range(len(individ)), size=1)[0]
    mask_random = np.random.random(len(individ)) <= CR
    mask_j = np.arange(len(individ)) == j
    mask_union = mask_random | mask_j
    individ[mask_union] = mutant[mask_union].copy()
    return individ


def standart_crossover(individs, fitness, rank):
    individ_1 = individs[0]
    individ_2 = individs[1]
    first_point = np.random.randint(1,  len(individ_1.nodes))
    second_point = np.random.randint(1,  len(individ_2.nodes))

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
