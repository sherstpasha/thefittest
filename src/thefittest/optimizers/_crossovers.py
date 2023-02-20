import numpy as np
from ..tools import protect_norm
from ._base import Tree


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


def common_region(trees):
    terminate = False
    indexes = []
    common_indexes = []
    border_indexes = []
    for tree in trees:
        indexes.append(list(range(len(tree.nodes))))
        common_indexes.append([])
        border_indexes.append([])

    while not terminate:
        inner_break = False
        iters = np.min(list(map(len, indexes)))

        for i in range(iters):
            first_n_args = trees[0].nodes[indexes[0][i]].n_args
            common_indexes[0].append(indexes[0][i])
            for j in range(1, len(indexes)):
                common_indexes[j].append(indexes[j][i])
                if first_n_args != trees[j].nodes[indexes[j][i]].n_args:
                    inner_break = True
                    

            if inner_break:
                for j in range(0, len(indexes)):
                    border_indexes[j].append(indexes[j][i])
                break

        for j in range(len(indexes)):
            _, right = trees[j].subtree(common_indexes[j][-1])
            delete_to = indexes[j].index(right-1) + 1
            indexes[j] = indexes[j][delete_to:]

            if len(indexes[j]) < 1:
                terminate = True
                break

    return common_indexes, border_indexes


def one_point_crossoverGP(individs, fitness, rank, max_level):
    individ_1 = individs[0].copy()
    individ_2 = individs[1].copy()
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
    individ_1 = individs[0].copy()
    individ_2 = individs[1].copy()
    common_indexes, border = common_region([individ_1, individ_2])
    
    for i in range(len(common_indexes[0])):
        if common_indexes[0][i] in border[0]:
            # print(common_indexes[0][i], 'граничная')
            if np.random.random() < 0.5:
                
                id_ = common_indexes[0][i]
                left, right = individ_1.subtree(index = id_)
                new_nodes.extend(individ_1.nodes[left:right])
                # print('0', individ_1.subtree(index = id_, return_class = True))
            else:
                id_ = common_indexes[1][i]
                left, right = individ_2.subtree(index = id_)
                new_nodes.extend(individ_2.nodes[left:right])
                # print('1', individ_2.subtree(index = id_, return_class = True))
        else:
            # print(common_indexes[0][i], 'обычная')
            if np.random.random() < 0.5:
                id_ = common_indexes[0][i]
                new_nodes.append(individ_1.nodes[id_])
                # print('0', new_nodes[-1].name)
            else:
                id_ = common_indexes[1][i]
                new_nodes.append(individ_2.nodes[id_])
                # print('1', new_nodes[-1].name)
    to_return = Tree(new_nodes.copy(), None)
    to_return.levels = to_return.get_levels()
    return to_return
        

