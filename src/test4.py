# import numpy as np
# from numba import njit
# import timeit
# from numba import boolean
# from numba import float64
# from numba import int64
# from numba import njit

# import numpy as np
# from numba import njit
# import timeit
# import random


# # Ваша функция с использованием numba
# @njit(int64[:](int64, int64, int64))
# def randint_numba(low: np.int64, high: np.int64, size: np.int64):
#     return np.random.randint(low, high, size)


# # Ваша функция с использованием numba
# @njit(int64[:](int64, int64, int64))
# def randint_numba2(low: np.int64, high: np.int64, size: np.int64):
#     return np.random.uniform(low, high, size).astype(np.int64)


# @njit(int64[:](int64, int64, int64))
# def numba_randint(low, high, size):
#     """
#     Generate an array of random integers from a discrete uniform distribution.

#     Parameters
#     ----------
#     low : int
#         The lowest integer to be drawn from the distribution.
#     high : int
#         The highest integer to be drawn from the distribution.
#     size : int
#         The number of integers to generate.

#     Returns
#     -------
#     NDArray[int64]
#         An array of random integers.

#     Examples
#     --------
#     >>> from numba import jit
#     >>> import numpy as np
#     >>>
#     >>> # Example of generating random integers
#     >>> result = numba_randint(low=1, high=10, size=5)
#     >>> print("Random Integers:", result)
#     Random Integers: ...

#     Notes
#     -----
#     The generated integers follow a discrete uniform distribution.
#     """
#     result = np.empty(size, dtype=np.int64)

#     for i in range(size):
#         result[i] = low + np.int64(np.floor((high - low) * random.random()))

#     return result


# # Параметры для сравнения
# low = 0
# high = 100
# size = 1000000  # Размер выборки

# # Используйте timeit для измерения времени выполнения вашей функции
# time_numba_randint = timeit.timeit(lambda: randint_numba(low, high, size), number=100)

# # Используйте timeit для измерения времени выполнения вашей функции
# time_numba_randint2 = timeit.timeit(lambda: numba_randint(low, high, size), number=100)

# # Используйте timeit для измерения времени выполнения вашей функции
# time_numba_randint3 = timeit.timeit(lambda: randint_numba2(low, high, size), number=100)

# # Измерьте время выполнения np.random.randint()
# time_numpy_randint = timeit.timeit(lambda: np.random.randint(low, high, size), number=100)


# # Выведите результаты
# print(f"Время выполнения вашей функции: {time_numba_randint:.6f} сек")
# print(f"Время выполнения вашей функции2: {time_numba_randint2:.6f} сек")
# print(f"Время выполнения вашей функции3: {time_numba_randint3:.6f} сек")
# print(f"Время выполнения np.random.randint(): {time_numpy_randint:.6f} сек")

import numpy as np
from thefittest.utils.random import sattolo_shuffle

# Example with a list
my_list = [1, 2, 3, 4, 5]
sattolo_shuffle(my_list)
print("Shuffled List:", my_list)

# Example with a NumPy array
my_array = np.array([1, 2, 3, 4, 5])
sattolo_shuffle(my_array)
print("Shuffled NumPy Array:", my_array)
