from numba import njit
from numba import int64
from numba import float64
from numba import boolean
import numpy as np
import matplotlib.pyplot as plt
from thefittest.tools.numba_funcs import nb_choice
from thefittest.tools.operators import protect_norm


'''взять за основу nb_choice 
weights обязателен, max_n не нужен
может своя реализация searchsorted быстрее (функция поиска индекса интервала, в который входит точка)

        found = False
        if not replace:
            for j in range(i): # этот цикл короткий и увеличивается в процессе генерации. Почему тут continue а не break?
                if inds[j] == ind:
                    found = True
                    continue # может continue адресован к while?
        if not found:
            inds[i] = ind
            i += 1
            этот кусок записать в отдельную функцию "функция опеределения есть ли точка в массиве"

'''




# @njit(int64[:](float64[:], int64, boolean))
# def random_weighted(weights, quantity=1, replace=True):
#     to_return = np.empty(shape=quantity, dtype=np.int64)
#     pool = list(range(len(weights)))

#     for i in range(quantity):
#         if len(pool) == 1:
#             to_return[i] = pool[0]
#         else:
#             mask_pool = np.array(pool)
#             cumsumweights = np.cumsum(weights[mask_pool])
#             sumweights = cumsumweights[-1]

#             roll = sumweights*np.random.rand()
#             ind = np.searchsorted(cumsumweights, roll, side='right') 
#             to_return[i] = pool[ind]
#             if not replace:
#                 assert len(weights) >= quantity
#                 pool.pop(ind)
#     return to_return


proba = np.array([0.1, 0.2, 0.4, 0.2, 0.05, 0.05], dtype = np.float64)
intervals = np.cumsum(proba)
print(intervals)
value = 0.6

def find_interval(value, intervals):
    length = len(intervals)
    # intervals_sorted = np.sorted(intervals)
    if value <= intervals[0]:
        ind = 0
    else:
        for i in range(1, length):
            if intervals[i - 1] < value <= intervals[i]:
                ind = i
                break
    return ind



test = find_interval(value, intervals)

print(test)



# proba = np.random.random(size = 7).astype(np.float64)
# proba = protect_norm(proba)


# # # print(proba)
# k = 2
# test = random_weighted(proba, k, False)

# test2 = nb_choice(len(proba), k, proba, False)
# # print(test)
# plt.hist(test)
# plt.savefig('test.png')
# plt.close()
# plt.hist(test2)
# plt.savefig('test2.png')
# # # print(randrange_weighted(0, 10, proba, 10))

# import time 


# n = 1000

# begin = time.time()
# for i in range(n):
#     test = random_weighted(proba, k, False)
# print(time.time() - begin)

# begin = time.time()
# for i in range(n):
#     test2 = nb_choice(len(proba), k, proba, False)
# print(time.time() - begin)
