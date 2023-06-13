import numpy as np
from numba import njit
from numba import float64
from numba import int64
from numba import boolean


@njit(int64[:](int64[:], boolean[:, :]))
def mask2d(X, cond):
    to_return = np.empty(shape = np.sum(cond), dtype=np.int64)
    begin = 0
    for i in range(cond.shape[0]):
        values = X[cond[i]]
        end = begin + len(values)
        to_return[begin: end] = values
        begin = end
    return to_return

        
        

X = np.array([0, 1, 2, 3, 4], dtype = np.int64)

cond = np.array([[True, False, True, False, True],
                 [False, False, True, False, True],
                 [True, False, False, True, False]], dtype = np.bool8)


test = mask2d(X, cond)
print(test)