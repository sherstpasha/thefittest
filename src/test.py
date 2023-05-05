import time
import numpy as np
from numba import njit
import numpy as np
from thefittest.optimizers import GeneticProgramming
from thefittest.tools import donothing
from thefittest.base import FunctionalNode
from thefittest.base import TerminalNode
from thefittest.base import EphemeralNode
from thefittest.base import UniversalSet
from thefittest.tools.operators import Mul
from thefittest.tools.operators import Add
from thefittest.tools.operators import Div
from thefittest.tools.operators import Neg
from thefittest.tools.random import half_and_half
from numba import float64
from numba import int64
from numpy.typing import NDArray
from thefittest.classifiers import MLPClassifierEA
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
# from thefittest.tools.transformations import clip
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


# data = load_digits()
# X = data.data
# y = data.target


# model = MLPClassifierEA(iters=100, pop_size=100,
#                         hidden_layers=(100,), show_progress_each=10)

# model.fit(X, y)

# y_pred = model.predict(X)





# y_pred2d = np.array([y_pred]*100, dtype=np.int64)

# n = 100
# begin = time.time()
# for i in range(n):
#     test = np.array([f1_score(y, y_pred_i, average='macro')
#                     for y_pred_i in y_pred2d], dtype=np.float64)
# print(time.time() - begin)
# print(test)

# begin = time.time()
# for i in range(n):
#     test = f1_score2d(y.astype(np.int64), y_pred2d.astype(np.int64))
# print(time.time() - begin)
# # print('recall_score', test)
# print(test)
