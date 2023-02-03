import numpy as np
from thefittest.tools import numpy_group_by


x = np.array([[1, 275, 23],
              [2, 275, 33],
              [1, 494, 15],
              [1, 593, 2],
              [2, 679, 679],
              [2, 533, -2],
              [2, 686, 533],
              [3, 559, 13],
              [3, 219, 23],
              [3, 455, 43],
              [4, 455, 54],
              [4, 468, 23],
              [4, 275, 33],
              [5, 613, 9]])

keys, groups = numpy_group_by(group = x[:,1:], by = x[:,0])
# print(keys)
# print(group)

x2 = dict(zip(keys, groups))
print(x2)