from numba import njit
from numba import int32


@njit(int32(int32, int32[:]))
def find_end_subtree_from_i(index, n_args_array):
    n_index = index + 1
    possible_steps = n_args_array[index]
    while possible_steps:
        possible_steps += n_args_array[n_index] - 1
        n_index += 1
    return n_index
