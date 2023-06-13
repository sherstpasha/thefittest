import time
import matplotlib.pyplot as plt
from thefittest.tools.print import print_net
from thefittest.classifiers import GeneticProgrammingNeuralNetClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from thefittest.tools.print import print_tree
from thefittest.tools.random import train_test_split_stratified
from thefittest.tools.metrics import confusion_matrix
from thefittest.tools.metrics import f1_score
from thefittest.tools.random import full_growing_method
import numpy as np
from numpy.typing import NDArray
from numba import float64
from numba import int64
from numba import njit
from numba import boolean
from numba import prange
from thefittest.tools._numba_funcs import max_axis
from thefittest.tools.operators import forward_softmax2
from thefittest.tools.operators import forward_softmax

data = load_iris()
X = data.data
y = data.target

n_outputs = len(set(y))
eye = np.eye(n_outputs)
proba_train = eye[y]
# X_train, X_test, y_train, y_test = train_test_split_stratified(
#     X, y, 0.3)

model = GeneticProgrammingNeuralNetClassifier(1, 10,
                                              show_progress_each=1,
                                              input_block_size=1)


uniset = model._define_uniset(X)

tree = full_growing_method(uniset, 3)

net = model._genotype_to_phenotype_tree(X.shape[1], len(set(y)), tree)

net._get_order2()

# _weights = np.array([net._weights]*2, dtype=np.float64)


# print(res)
# print(res2)

# n = 100

# begin = time.time()
# for i in range(n):
#     res = forward_softmax2(X,
#                            net._forward_inputs_array,
#                            net._forward_outputs_array,
#                            net._forward_cond_h,
#                            net._forward_cond_o,
#                            net._forward_culc_order_h,
#                            net._forward_culc_order_o,
#                            net._weights,
#                            net._connects[:, 0],
#                            net._forward_activ_code)

# print(time.time() - begin)
# # print(res)

# begin = time.time()
# for i in range(n):
#     res2 = forward_softmax(X,
#                            net._forward_inputs_array,
#                            net._forward_outputs_array,
#                            net._forward_cond_h,
#                            net._forward_cond_o,
#                            net._forward_culc_order_h,
#                            net._forward_culc_order_o,
#                            net._weights,
#                            net._connects[:, 0],
#                            net._forward_activ_code)

# print(time.time() - begin)
# # print(res)

fig, ax = plt.subplots(2)
print_tree(tree, ax[0])
print_net(net, ax[1])
fig.savefig('net.png')
