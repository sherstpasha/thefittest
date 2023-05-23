import time
import matplotlib.pyplot as plt
from thefittest.tools.print import print_net
from thefittest.classifiers import GeneticProgrammingNeuralNetClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
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


data = load_digits()
X = data.data
y = data.target

n_outputs = len(set(y))
eye = np.eye(n_outputs)
proba_train = eye[y]
# X_train, X_test, y_train, y_test = train_test_split_stratified(
#     X, y, 0.3)

model = GeneticProgrammingNeuralNetClassifier(1, 10,
                                              show_progress_each=1,
                                              input_block_size=8)


uniset = model._define_uniset(X)

tree = full_growing_method(uniset, 6)

net = model._genotype_to_phenotype_tree(X.shape[1], len(set(y)), tree)


@njit(float64(float64[:, :], float64[:, :]))
def categorical_crossentropy(target: NDArray[np.float64],
                             output: NDArray[np.float64]) -> np.float64:
    output_c = np.clip(output, 1e-7, 1 - 1e-7)
    to_return = np.mean(np.sum(target*(-np.log(output_c)), axis=1))
    return to_return


@njit(float64[:](float64[:, :], float64[:, :, :]))
def categorical_crossentropy3d(
        target: NDArray[np.float64],
        output3d: NDArray[np.float64]) -> NDArray[np.float64]:
    size = len(output3d)
    to_return = np.empty(size, dtype=np.float64)
    for i in range(size):
        to_return[i] = categorical_crossentropy(target, output3d[i])
    return to_return


@njit(float64[:, :](float64[:, :]))
def softmax_numba(X: NDArray[np.float64]) -> NDArray[np.float64]:
    exps = np.exp(X - max_axis(X))
    sum_ = np.sum(exps, axis=1)
    for j in range(sum_.shape[0]):
        if sum_[j] == 0:
            sum_[j] = 1
    result = ((exps).T/sum_).T
    return result


@njit(float64[:](float64[:], int64))
def multiactivation(X: NDArray[np.float64],
                    activ_id: np.int64) -> NDArray[np.float64]:
    if activ_id == 0:
        result = 1/(1+np.exp(-X))
    elif activ_id == 1:
        result = X*(X > 0)
    elif activ_id == 2:
        result = np.exp(-(X**2))
    elif activ_id == 3:
        result = np.tanh(X)
    return result


@njit(float64[:, :](float64[:, :], int64[:], int64[:], boolean[:, :],
                    boolean[:, :], int64[:], int64[:],
                    float64[:], int64[:], int64[:]))
def forward_softmax(X: NDArray[np.float64],
                    inputs: NDArray[np.int64],
                    outputs: NDArray[np.int64],
                    h_conds: NDArray[np.bool8],
                    o_conds: NDArray[np.bool8],
                    order_h: NDArray[np.int64],
                    order_o: NDArray[np.int64],
                    weights: NDArray[np.float64],
                    from_: NDArray[np.int64],
                    activs: NDArray[np.int64]) -> NDArray[np.float64]:
    num_nodes = X.shape[1] + len(h_conds) + len(outputs)
    shape = (num_nodes, len(X))
    nodes = np.empty(shape, dtype=np.float64)

    nodes[inputs] = X.T[inputs]
    for i, node_i in enumerate(order_h):
        from_i = from_[h_conds[i]]
        weight_i = weights[h_conds[i]]
        i_dot_w_sum = np.dot(nodes[from_i].T, weight_i)
        nodes[node_i] = multiactivation(i_dot_w_sum, activs[i])

    for i, order_o_i in enumerate(order_o):
        from_i = from_[o_conds[i]]
        weight_i = weights[o_conds[i]]
        i_dot_w_sum = np.dot(nodes[from_i].T, weight_i)
        nodes[order_o_i] = i_dot_w_sum

    return softmax_numba(nodes[outputs].T)


@njit(float64[:, :, :](float64[:, :], int64[:], int64[:], boolean[:, :],
                       boolean[:, :], int64[:], int64[:],
                       float64[:, :], int64[:], int64[:]))
def forward_softmax2d(X: NDArray[np.float64],
                      inputs: NDArray[np.int64],
                      outputs: NDArray[np.int64],
                      h_conds: NDArray[np.bool8],
                      o_conds: NDArray[np.bool8],
                      order_h: NDArray[np.int64],
                      order_o: NDArray[np.int64],
                      weights: NDArray[np.float64],
                      from_: NDArray[np.int64],
                      activs: NDArray[np.int64]) -> NDArray[np.float64]:

    outs = np.empty(shape=(len(weights), X.shape[0], len(outputs)))
    for n in range(outs.shape[0]):
        outs[n] = forward_softmax(X,
                                  inputs,
                                  outputs,
                                  h_conds,
                                  o_conds,
                                  order_h,
                                  order_o,
                                  weights[n],
                                  from_,
                                  activs)
    return outs


@njit(float64[:](float64[:, :], int64[:], int64[:], boolean[:, :],
                 boolean[:, :], int64[:], int64[:],
                 float64[:, :], int64[:], int64[:], float64[:,:]))
def forward_softmax2d_eval(X: NDArray[np.float64],
                           inputs: NDArray[np.int64],
                           outputs: NDArray[np.int64],
                           h_conds: NDArray[np.bool8],
                           o_conds: NDArray[np.bool8],
                           order_h: NDArray[np.int64],
                           order_o: NDArray[np.int64],
                           weights: NDArray[np.float64],
                           from_: NDArray[np.int64],
                           activs: NDArray[np.int64],
                           target: NDArray[np.float64]) -> NDArray[np.float64]:

    to_return = np.empty(len(weights), dtype=np.float64)
    for i in range(to_return.shape[0]):
        output = forward_softmax(X,
                                 inputs,
                                 outputs,
                                 h_conds,
                                 o_conds,
                                 order_h,
                                 order_o,
                                 weights[i],
                                 from_,
                                 activs)
        to_return[i] = categorical_crossentropy(target, output)
    return to_return


net._get_order()
_weights = np.array([net._weights]*300, dtype=np.float64)


n = 10

begin = time.time()
for i in range(n):
    outputs1 = forward_softmax2d(X,
                                 net._forward_inputs_array,
                                 net._forward_outputs_array,
                                 net._forward_cond_h,
                                 net._forward_cond_o,
                                 net._forward_culc_order_h,
                                 net._forward_culc_order_o,
                                 _weights,
                                 net._connects[:, 0],
                                 net._forward_activ_code)
    res = categorical_crossentropy3d(proba_train, outputs1)

print(time.time() - begin)
# print(res)

begin = time.time()
for i in range(n):
    res = forward_softmax2d_eval(X,
                                 net._forward_inputs_array,
                                 net._forward_outputs_array,
                                 net._forward_cond_h,
                                 net._forward_cond_o,
                                 net._forward_culc_order_h,
                                 net._forward_culc_order_o,
                                 _weights,
                                 net._connects[:, 0],
                                 net._forward_activ_code,
                                 proba_train)

print(time.time() - begin)
# print(res)

fig, ax = plt.subplots(2)
print_tree(tree, ax[0])
print_net(net, ax[1])
fig.savefig('net.png')
