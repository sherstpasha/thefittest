import numpy as np
from numpy.typing import NDArray
from typing import Optional
import time
from numba import njit



bit_array = np.random.randint(0, 2, size = (150, 100), dtype = np.int8)



def numpy_bit_to_int(
    bit_array: NDArray[np.int64], powers: Optional[NDArray[np.int64]] = None
) -> NDArray[np.int64]:
    if powers is None:
        powers = 2 ** np.arange(bit_array.shape[1], dtype=np.int64)
    arange_ = powers[: bit_array.shape[1]][::-1]
    int_array = np.sum(bit_array * arange_, axis=1)
    return int_array

@njit
def numpy_bit_to_int2(
    bit_array: NDArray[np.int64], powers: Optional[NDArray[np.int64]] = None
) -> NDArray[np.int64]:
    if powers is None:
        powers = 2 ** np.arange(bit_array.shape[1], dtype=np.int64)
    arange_ = powers[: bit_array.shape[1]][::-1]
    int_array = np.sum(bit_array * arange_, axis=1)
    return int_array

def numpy_bit_to_int3(
    bit_array: NDArray[np.int64], powers: Optional[NDArray[np.int64]] = None
) -> NDArray[np.int64]:
    if powers is None:
        powers = 2 ** np.arange(bit_array.shape[1], dtype=np.int64)
    arange_ = powers[: bit_array.shape[1]][::-1]

    int_array = np.dot(bit_array, arange_)
    return int_array

def numpy_bit_to_int4(
    bit_array: NDArray[np.int64], reversed_powers: Optional[NDArray[np.int64]] = None
) -> NDArray[np.int64]:

    if reversed_powers is None:
        powers = 2 ** np.arange(bit_array.shape[1], dtype=np.int64)
        reversed_powers = np.flip(powers)

    int_array = np.dot(bit_array, reversed_powers)
    return int_array

powers = 2 ** np.arange(bit_array.shape[1], dtype=np.int64)
reversed_powers = np.flip(powers)
numpy_bit_to_int2(bit_array)

n = 100000

t1 = time.time()
for i in range(n):
    res1 = numpy_bit_to_int(bit_array)
t2 = time.time()
print(t2 - t1)


t1 = time.time()
for i in range(n):
    numpy_bit_to_int2(bit_array)
t2 = time.time()
print(t2 - t1)

t1 = time.time()
for i in range(n):
    res2 = numpy_bit_to_int3(bit_array)
t2 = time.time()
print(t2 - t1)

t1 = time.time()
for i in range(n):
    res3 = numpy_bit_to_int4(bit_array)
t2 = time.time()
print(t2 - t1)

print(np.allclose(res1, res2))
print(np.allclose(res1, res3))