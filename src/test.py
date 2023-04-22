import numpy as np
from thefittest.optimizers import SelfCGP
from thefittest.tools.transformations import donothing
from thefittest.base import FunctionalNode
from thefittest.base import TerminalNode
from thefittest.base import EphemeralNode
from thefittest.base import UniversalSet
from thefittest.base._tree import Node
from thefittest.tools.operators import Mul
from thefittest.tools.operators import Add
from thefittest.tools.operators import Inv
from thefittest.tools.operators import Pow2
from thefittest.tools.operators import Neg
from thefittest.tools.operators import Cos
from thefittest.tools.operators import Sin
from thefittest.tools.transformations import coefficient_determination
from thefittest.benchmarks.symbolicregression17 import problems_dict
from thefittest.tools.generators import full_growing_method


def generator1():
    return np.round(np.random.uniform(0, 10), 4)


def generator2():
    return np.random.randint(0, 10)


def problem(x):
    return 3*x[:, 0]**2 + 2*x[:, 0] + 5


function = problems_dict['F1']['function']
left_border = problems_dict['F1']['bounds'][0]
right_border = problems_dict['F1']['bounds'][1]
sample_size = 300
n_dimension = problems_dict['F1']['n_vars']

number_of_iterations = 300
population_size = 500

X = np.array([np.linspace(left_border, right_border, sample_size)
              for _ in range(n_dimension)]).T
y = function(X)


functional_set = [FunctionalNode(Add()),
                  FunctionalNode(Mul()),
                  FunctionalNode(Neg()),
                  FunctionalNode(Inv()),
                  FunctionalNode(Pow2()),
                  FunctionalNode(Cos()),
                  FunctionalNode(Sin())]


terminal_set = [TerminalNode(X[:, i], f'x{i}') for i in range(n_dimension)]
terminal_set.extend([EphemeralNode(generator1), EphemeralNode(generator2)])
uniset = UniversalSet(functional_set, terminal_set)


tree = full_growing_method(uniset, 5)
print(tree)
print(tree.compile())
# print(uniset._functional_set)



# print(tree)

# arr = list(range(10000))
# range_ = range(10000)
# import time
# import random
# n = 10000

# begin = time.time()
# for i in range(n):
#     for jj, j in enumerate(arr):
#         j
# print((time.time() - begin))

# arr = tuple(arr)
# begin = time.time()
# for i in range(n):
#     for jj, j in enumerate(arr):
#         j
# print((time.time() - begin))

# begin = time.time()
# for i in range(n):
#     for j in range_:
#         j
# print((time.time() - begin))

