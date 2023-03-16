import numpy as np
from numba import njit
from numba import int32
from numba import void
from numba.typed import List



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

@njit
def find_common_id_in_two_trees(n_args_list):
    terminate = False
    indexes = []
    common_indexes = []
    border_indexes = []
    for i, n_args in enumerate(n_args_list):
        indexes.append(list(range(len(n_args))))
        common_indexes.append([0])
        common_indexes[-1].pop()
        border_indexes.append([0])
        border_indexes[-1].pop()

    while not terminate:
        inner_break = False
        iters = min(list(map(len, indexes)))

        for i in range(iters):
            first_n_args = n_args_list[0][indexes[0][i]]
            common_indexes[0].append(indexes[0][i])
            for j in range(1, len(indexes)):
                common_indexes[j].append(indexes[j][i])
                if first_n_args != n_args_list[j][indexes[j][i]]:
                    inner_break = True

            if inner_break:
                for j in range(0, len(indexes)):
                    border_indexes[j].append(indexes[j][i])
                break

        for j in range(len(indexes)):
            right = find_end_subtree_from_i(common_indexes[j][-1], n_args_list[j])
            delete_to = indexes[j].index(right-1) + 1
            indexes[j] = indexes[j][delete_to:]

            if len(indexes[j]) < 1:
                terminate = True
                break

    return common_indexes, border_indexes