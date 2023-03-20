import numpy as np
from numba import njit
from numba import int32
from numba import float32


@njit(int32(int32, int32[:]))
def find_end_subtree_from_i(index, n_args_array):
    n_index = index + 1
    possible_steps = n_args_array[index]
    while possible_steps:
        possible_steps += n_args_array[n_index] - 1
        n_index += 1
    return n_index


@njit(int32[:](int32, int32[:]))
def find_id_args_from_i(index, n_args_array):
    out = np.empty(n_args_array[index], dtype=np.int32)
    out[0] = index + 1
    for i in range(1, len(out)):
        next_stop = find_end_subtree_from_i(out[i-1], n_args_array)
        out[i] = next_stop
    return out


@njit(int32[:](int32, int32[:]))
def get_levels_tree_from_i(origin, n_args_array):
    d_i = -1
    s = [1]
    d = [-1]
    result_list = []
    for i, n_arg in enumerate(n_args_array[origin:]):
        s[-1] = s[-1] - 1
        if s[-1] == 0:
            s.pop()
            d_i = d.pop() + 1
        else:
            d_i = d[-1] + 1
        result_list.append(d_i)
        if n_arg > 0:
            s.append(n_arg)
            d.append(d_i)
        if len(s) == 0:
            break
    return np.array(result_list, dtype=np.int32)


@njit(int32[:](float32[:], int32, int32))
def select_quantity_id_by_tournament(fitness, tour_size, quantity):
    indexes = np.arange(len(fitness))
    choosen = np.empty(quantity, dtype=np.int32)
    for i in range(quantity):
        tournament = np.random.choice(indexes, size=tour_size, replace=False)
        argmax = np.argmax(fitness[tournament])
        choosen[i] = tournament[argmax]
    return choosen


@njit(int32(int32[:], int32[:]))
def find_first_difference_between_two(array_1, array_2):
    for i in range(min(len(array_1), len(array_2))):
        if array_1[i] != array_2[i]:
            break
    return i
