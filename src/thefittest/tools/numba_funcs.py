import numpy as np
import random
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


@njit
def random_sample_set(arr, k=-1):
    n = arr.size
    if k < 0:
        k = arr.size
    seen = {0}
    seen.clear()
    index = np.empty(k, dtype=arr.dtype)
    for i in range(k):
        j = random.randint(i, n - 1)
        while j in seen:
            j = random.randint(0, n - 1)
        seen.add(j)
        index[i] = j
    return arr[index]


@njit
def nb_choice(max_n, k=1, weights=None, replace=False):
    '''
    https://stackoverflow.com/questions/64135020/speed-up-random-weighted-choice-without-replacement-in-python
    Choose k samples from max_n values, with optional weights and replacement.
    Args:
        max_n (int): the maximum index to choose
        k (int): number of samples
        weights (array): weight of each index, if not uniform
        replace (bool): whether to sample with replacement
    '''
    # Get cumulative weights
    if weights is None:
        weights = np.full(int(max_n), 1.0)
    cumweights = np.cumsum(weights)

    maxweight = cumweights[-1]  # Total of weights
    # Arrays of sample and sampled indices
    inds = np.full(k, -1, dtype=np.int64)

    # Sample
    i = 0
    while i < k:
        # Find the index
        r = maxweight * np.random.rand()  # Pick random weight value
        # Get corresponding index
        ind = np.searchsorted(cumweights, r, side='right')

        # Optionally sample without replacement
        found = False
        if not replace:
            for j in range(i):
                if inds[j] == ind:
                    found = True
                    continue
        if not found:
            inds[i] = ind
            i += 1

    return inds
