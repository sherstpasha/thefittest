from typing import Any
from typing import Callable
from typing import Tuple
from typing import List
from typing import Union

from numba import boolean
from numba import float64
from numba import int64
from numba import njit
from numba.typed import Dict as numbaDict
from numba.types import List as numbaListType

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray


MIN_VALUE = np.finfo(np.float64).min
MAX_VALUE = np.finfo(np.float64).max


def common_region(trees: Union[List, NDArray]) -> Tuple:
    terminate = False
    indexes = []
    common_indexes: List[List[int]] = []
    border_indexes: List[List[int]] = []
    for tree in trees:
        indexes.append(list(range(len(tree._nodes))))
        common_indexes.append([])
        border_indexes.append([])

    while not terminate:
        inner_break = False
        iters = min(list(map(len, indexes)))

        for i in range(iters):
            first_n_args = trees[0]._nodes[indexes[0][i]]._n_args
            common_indexes[0].append(indexes[0][i])
            for j in range(1, len(indexes)):
                common_indexes[j].append(indexes[j][i])
                if first_n_args != trees[j]._nodes[indexes[j][i]]._n_args:
                    inner_break = True

            if inner_break:
                for j in range(0, len(indexes)):
                    border_indexes[j].append(indexes[j][i])
                break

        for j in range(len(indexes)):
            _, right = trees[j].subtree_id(common_indexes[j][-1])
            delete_to = indexes[j].index(right - 1) + 1
            indexes[j] = indexes[j][delete_to:]

            if len(indexes[j]) < 1:
                terminate = True
                break

    return common_indexes, border_indexes


def common_region_two_trees(
    n_args_array_1: NDArray[np.int64], n_args_array_2: NDArray[np.int64]
) -> Tuple:
    index_list_1 = range(len(n_args_array_1))
    index_list_2 = range(len(n_args_array_2))
    index_1 = np.int64(0)
    index_2 = np.int64(0)

    common_1: List[List[int]] = []
    common_2: List[List[int]] = []
    border_1 = []
    border_2 = []

    while True:
        if index_1 < len(n_args_array_1) and index_2 < len(n_args_array_2):
            id_1 = index_1
            id_2 = index_2
            end = find_first_difference_between_two(n_args_array_1[id_1:], n_args_array_2[id_2:])
            index_1 = index_1 + end
            index_2 = index_2 + end
            common_1.extend(index_list_1[id_1 : index_1 + 1])
            common_2.extend(index_list_2[id_2 : index_2 + 1])

        if len(n_args_array_1) - 1 > index_1 or len(n_args_array_2) - 1 > index_2:
            border_1.append(index_1)
            border_2.append(index_2)
            index_1 = find_end_subtree_from_i(index_1, n_args_array_1)
            index_2 = find_end_subtree_from_i(index_2, n_args_array_2)
        else:
            break

    return [common_1, common_2], [border_1, border_2]


@njit(int64(int64, int64[:]))
def find_end_subtree_from_i(index: np.int64, n_args_array: NDArray[np.int64]) -> np.int64:
    n_index = index + 1
    possible_steps = n_args_array[index]
    while possible_steps:
        possible_steps += n_args_array[n_index] - 1
        n_index += 1
    return n_index


@njit(int64[:](int64, int64[:]))
def find_id_args_from_i(index: np.int64, n_args_array: NDArray[np.int64]) -> NDArray[np.int64]:
    out = np.empty(n_args_array[index], dtype=np.int64)
    out[0] = index + 1
    for i in range(1, len(out)):
        next_stop = find_end_subtree_from_i(out[i - 1], n_args_array)
        out[i] = next_stop
    return out


@njit(int64[:](int64, int64[:]))
def get_levels_tree_from_i(origin: np.int64, n_args_array: NDArray[np.int64]) -> NDArray[np.int64]:
    d_i = -1
    s = [1]
    d = [-1]
    result_list = []
    for n_arg in n_args_array[origin:]:
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
    return np.array(result_list, dtype=np.int64)


@njit(int64(int64[:], int64[:]))
def find_first_difference_between_two(
    array_1: NDArray[np.int64], array_2: NDArray[np.int64]
) -> np.int64:
    for i in np.arange(min(len(array_1), len(array_2)), dtype=np.int64):
        if array_1[i] != array_2[i]:
            break
    return i


@njit(int64(float64, float64[:]))
def binary_search_interval(value: np.float64, intervals: NDArray[np.float64]) -> int:
    if value <= intervals[0]:
        ind = 0
    else:
        left = 0
        right = len(intervals) - 1
        while right - left > 1:
            mid = (left + right) // 2
            if value <= intervals[mid]:
                right = mid
            else:
                left = mid
        ind = right
    return ind


@njit(boolean(int64, int64[:], int64))
def check_for_value(value: np.int64, index_array: NDArray[np.int64], end: np.int64) -> bool:
    found = False
    for i in range(end):
        if value == index_array[i]:
            found = True
            break
    return found


@njit(int64[:](float64[:], int64))
def argsort_k(array: NDArray[np.float64], k: np.int64) -> NDArray[np.int64]:
    size = len(array)
    array_copy = array.copy()
    to_return = np.arange(size, dtype=np.int64)
    for i in range(k):
        max_ = array_copy[i]
        for j in range(i, size):
            if array_copy[j] > max_:
                max_ = array_copy[j]
                max_id = j
        array_copy[i], array_copy[max_id] = array_copy[max_id], array_copy[i]
        to_return[i], to_return[max_id] = to_return[max_id], to_return[i]
    return to_return


@njit(int64[:](float64[:], float64))
def find_pbest_id(array: NDArray[np.float64], p: np.float64) -> NDArray[np.int64]:
    size = len(array)
    count = max(np.int64(1), np.int64(p * size))
    argsort = argsort_k(array, count)
    to_return = argsort[:count]
    return to_return


@njit(float64[:, :](float64[:, :]))
def max_axis(array: NDArray[np.float64]) -> NDArray[np.float64]:
    res = np.zeros((array.shape[0], 1), dtype=np.float64)
    for i in range(array.shape[0]):
        res[i] = np.max(array[i])
    return res


@njit(int64[:](int64[:, :]))
def voting(argmax_y: NDArray[np.int64]) -> NDArray[np.int64]:
    winner = np.empty(argmax_y.shape[1], dtype=np.int64)
    for i in range(argmax_y.shape[1]):
        counter = numbaDict.empty(key_type=int64, value_type=int64)

        for argmax_y_i_j in argmax_y[:, i]:
            if argmax_y_i_j not in counter:
                counter[argmax_y_i_j] = 1
            else:
                counter[argmax_y_i_j] += 1

        max_key = -1
        max_value = 0
        for key, value in counter.items():
            if value > max_value:
                max_value = value
                max_key = key

        winner[i] = max_key
    return winner


@njit(float64[:, :](float64[:, :]))
def softmax_numba(X: NDArray[np.float64]) -> NDArray[np.float64]:
    exps = np.exp(X - max_axis(X))
    sum_ = np.sum(exps, axis=1)
    for j in range(sum_.shape[0]):
        if sum_[j] == 0:
            sum_[j] = 1
    result = ((exps).T / sum_).T
    return result


@njit(float64[:, :](float64[:, :], int64))
def multiactivation2d(X: NDArray[np.float64], activ_id: np.int64) -> NDArray[np.float64]:
    if activ_id == 0:
        result = 1 / (1 + np.exp(-X))
    elif activ_id == 1:
        result = X * (X > 0)
    elif activ_id == 2:
        result = np.exp(-(X**2))
    elif activ_id == 3:
        result = np.tanh(X)
    elif activ_id == 4:
        result = X
    elif activ_id == 5:
        result = softmax_numba(X)
    return result


@njit
def forward(
    weights: NDArray[np.float64],
    nodes: NDArray[np.float64],
    from_: numbaListType(NDArray[np.int64]),
    to_: numbaListType(NDArray[np.int64]),
    weights_id: numbaListType(NDArray[np.int64]),
    activs_code: numbaListType(NDArray[np.int64]),
    activs_nodes: numbaListType(numbaListType(NDArray[np.int64])),
) -> NDArray[np.float64]:
    for from_i, to_i, weights_id_i, a_code_i, a_nodes_i in zip(
        from_, to_, weights_id, activs_code, activs_nodes
    ):
        arr_mask = weights[weights_id_i.flatten()]
        weights_i = arr_mask.reshape(weights_id_i.shape)
        out = np.dot(nodes[from_i].T, weights_i.T)
        nodes[to_i] = out.T

        for a_code_i_i, a_nodes_i_i in zip(a_code_i, a_nodes_i):
            nodes[a_nodes_i_i] = multiactivation2d(nodes[a_nodes_i_i].T, a_code_i_i).T

    return nodes


@njit
def forward2d(
    X: NDArray[np.float64],
    inputs: NDArray[np.int64],
    n_hiddens: np.int64,
    outputs: NDArray[np.int64],
    from_: numbaListType(NDArray[np.int64]),
    to_: numbaListType(NDArray[np.int64]),
    weights_id: numbaListType(NDArray[np.int64]),
    activs_code: numbaListType(NDArray[np.int64]),
    activs_nodes: numbaListType(numbaListType(NDArray[np.int64])),
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    outs = np.empty(shape=(len(weights), X.shape[0], len(outputs)))
    num_nodes = X.shape[1] + n_hiddens + len(outputs)
    shape = (num_nodes, len(X))
    nodes = np.empty(shape, dtype=np.float64)
    nodes[inputs] = X.T[inputs]

    for n in range(outs.shape[0]):
        forward(weights[n], nodes, from_, to_, weights_id, activs_code, activs_nodes)

        outs[n] = nodes[outputs].T
    return outs


def numpy_group_by(group: NDArray, by: NDArray) -> Tuple:
    argsort = np.argsort(by)
    group = group[argsort]
    by = by[argsort]

    keys, cut_index = np.unique(by, return_index=True)
    groups = np.split(group, cut_index)[1:]
    return keys, groups


def array_like_to_numpy_X_y(
    X: ArrayLike, y: ArrayLike, y_numeric=True
) -> Tuple[NDArray[np.float64]]:
    X = np.array(X, dtype=np.float64)
    if y_numeric:
        y = np.array(y, dtype=np.float64)
    else:
        y = np.array(y, dtype=np.int64)
    return X, y


def save_div(x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    result: Union[float, np.ndarray]
    if isinstance(y, np.ndarray):
        result = np.divide(x, y, out=np.ones_like(y, dtype=np.float64), where=y != 0)
    else:
        if y == 0:
            result = 0.0
        else:
            result = x / y
    result = np.clip(result, MIN_VALUE, MAX_VALUE)
    return result


def logabs(y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    y_ = np.abs(y)
    if isinstance(y_, np.ndarray):
        result = np.log(y_, out=np.ones_like(y_, dtype=np.float64), where=y_ != 0)
    else:
        if y_ == 0:
            result = 1
        else:
            result = np.log(y_)
    return result


def save_exp(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    result = np.clip(np.exp(x), MIN_VALUE, MAX_VALUE)
    return result


def sqrtabs(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    result = np.sqrt(np.abs(x))
    return result


def cos(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.cos(x)


def sin(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.sin(x)


class Operator:
    def __init__(self, formula: str, name: str, sign: str, operation: Callable) -> None:
        self._formula = formula
        self.__name__ = name
        self._sign = sign
        self._operation = operation

    def _write(self, *args: Any) -> str:
        formula = self._formula.format(*args)
        return formula

    def __call__(self, *args: Any) -> Any:
        return self._operation(*args)


def create_operator(formula: str, name: str, sign: str, operation: Callable) -> Operator:
    """
    Creates a new mathematical operator for performing a given operation.

    This factory function allows defining a new operator with a custom function
    that performs a mathematical operation. The defined operator can be used to
    perform the operation on scalar values or arrays.

    Parameters
    ----------
    formula : str
        The string representation of the operator's formula, where the operation's arguments
        are denoted by curly braces. For example, "({} + {})" for an addition operation.
    name : str
        The name of the operator, used for identification.
    sign : str
        The symbol of the operator, used for representation in formulas.
    operation : Callable
        The function that performs the operation. It must accept as many arguments as
        indicated by `formula`, and return the result of the operation.

    Returns
    -------
    Operator
        An instance of the Operator class, representing the new mathematical operator.

    Examples
    --------
    Creating an operator to calculate the cosine:

    >>> cos = create_operator("cos({})", "cos", "cos", np.cos)
    >>> print(cos(np.pi))
    -1.0

    Creating an operator for adding two numbers:

    >>> add = create_operator("({} + {})", "add", "+", lambda x, y: x + y)
    >>> print(add(5, 3))
    8

    Creating an operator for subtraction:

    >>> sub = create_operator("({} - {})", "sub", "-", lambda x, y: x - y)
    >>> print(sub(10, 4))
    6
    """
    return Operator(formula, name, sign, operation)


def if_single_or_array_to_float_array(
    single_or_array: Union[float, int, np.number, NDArray[np.number]], num_variables: int
) -> NDArray[np.float64]:
    """
    Convert a single value or an array to a 1D float array.

    Parameters
    ----------
    single_or_array : Union[float, int, np.number, NDArray[np.number]]
        A single numeric value or an array of numeric values.

    Returns
    -------
    NDArray[np.float64]
        1D float array.

    """
    if isinstance(single_or_array, (int, float, np.number)):
        array = np.full(shape=num_variables, fill_value=single_or_array, dtype=np.float64)
    else:
        assert len(single_or_array) == num_variables
        array = single_or_array.astype(np.float64)

    return array


def if_single_or_array_to_int_array(
    single_or_array: Union[float, int, np.number, NDArray[np.number]], num_variables: int
) -> NDArray[np.int64]:
    """
    Convert a single value or an array to a 1D int array.

    Parameters
    ----------
    single_or_array : Union[float, int, np.number, NDArray[np.number]]
        A single numeric value or an array of numeric values.

    Returns
    -------
    NDArray[np.int64]
        1D int array.

    """
    if isinstance(single_or_array, (int, float, np.number)):
        array = np.full(shape=num_variables, fill_value=single_or_array, dtype=np.int64)
    else:
        assert len(single_or_array) == num_variables
        array = single_or_array.astype(np.int64)

    return array
