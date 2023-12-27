import numpy as np
from numpy.typing import NDArray
from typing import Optional
import time
from numba import njit



int_array = np.random.randint(0, 256, size = (1000, 1000), dtype = np.int64)



def numpy_int_to_bit(int_array: NDArray[np.int64]) -> NDArray[np.byte]:
    result = []
    bit = int_array % 2
    remains = int_array // 2
    result.append(bit)
    while np.sum(remains) > 0:
        bit = remains % 2
        remains = remains // 2
        result.append(bit)
    # bit_array = np.array(result, dtype=np.int8)[::-1].T
    return result

@njit
def numpy_int_to_bit2(int_array: NDArray[np.int64]) -> NDArray[np.byte]:
    result = []
    bit = int_array % 2
    remains = int_array // 2
    result.append(bit)
    while np.sum(remains) > 0:
        bit = remains % 2
        remains = remains // 2
        result.append(bit)
    # bit_array = np.asarray(result, dtype=np.int8)[::-1].T
    return result


numpy_int_to_bit2(int_array)

n = 100

t1 = time.time()
for i in range(n):
    res1 = numpy_int_to_bit(int_array)
t2 = time.time()
print(t2 - t1)


t1 = time.time()
for i in range(n):
    numpy_int_to_bit2(int_array)
t2 = time.time()
print(t2 - t1)

# t1 = time.time()
# for i in range(n):
#     res2 = numpy_bit_to_int3(bit_array)
# t2 = time.time()
# print(t2 - t1)

# t1 = time.time()
# for i in range(n):
#     res3 = numpy_bit_to_int4(bit_array)
# t2 = time.time()
# print(t2 - t1)

# print(np.allclose(res1, res2))
# print(np.allclose(res1, res3))