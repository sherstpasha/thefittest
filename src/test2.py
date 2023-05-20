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
from thefittest.tools._numba_funcs import map_dot, max_axis
from thefittest.tools.operators import multiactivation2d
from thefittest.tools.operators import softmax_numba3d


data = load_iris()
X = data.data
y = data.target

# X_train, X_test, y_train, y_test = train_test_split_stratified(
#     X, y, 0.3)

model = GeneticProgrammingNeuralNetClassifier(1, 10,
                                              show_progress_each=1,
                                              input_block_size=8)


uniset = model._define_uniset(X)

tree = full_growing_method(uniset, 6)

net = model._genotype_to_phenotype_tree(X.shape[1], len(set(y)), tree)


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

    num_nodes = X.shape[1] + len(h_conds) + len(outputs)
    shape = (len(weights), num_nodes, len(X))
    nodes = np.zeros(shape, dtype=np.float64)

    for i in range(weights.shape[0]):
        nodes[i][inputs] = X.T[inputs]

    k = 0
    for i in order_h:
        from_i = from_[h_conds[k]]
        weight_i = weights[:, h_conds[k]]

        i_dot_w_sum = map_dot(nodes[:, from_i], weight_i)
        i_dot_w_sum = np.clip(i_dot_w_sum, -700, 700)
        nodes[:, i] = multiactivation2d(i_dot_w_sum, activs[k])
        k += 1

    k = 0
    for i in order_o:
        from_i = from_[o_conds[k]]
        weight_i = weights[:, o_conds[k]]
        i_dot_w_sum = map_dot(nodes[:, from_i], weight_i)
        i_dot_w_sum = np.clip(i_dot_w_sum, -700, 700)
        nodes[:, i] = i_dot_w_sum
        k += 1

    out = softmax_numba3d(nodes[:, outputs].T)

    return np.transpose(out, axes=(2, 0, 1))


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
def forward_softmax2d3(X: NDArray[np.float64],
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

@njit(float64[:, :, :](float64[:, :], int64[:], int64[:], boolean[:, :],
                       boolean[:, :], int64[:], int64[:],
                       float64[:, :], int64[:], int64[:]))
def forward_softmax2d2(X: NDArray[np.float64],
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
    num_nodes = X.shape[1] + len(h_conds) + len(outputs)
    for n in range(outs.shape[0]):
        shape = (num_nodes, len(X))
        nodes = np.empty(shape, dtype=np.float64)
        nodes[inputs] = X.T[inputs]

        for i, node_i in enumerate(order_h):
            from_i = from_[h_conds[i]]
            weight_i = weights[n][h_conds[i]]
            i_dot_w_sum = np.dot(nodes[from_i].T, weight_i)
            i_dot_w_sum = np.clip(i_dot_w_sum, -700, 700)
            nodes[node_i] = multiactivation(i_dot_w_sum, activs[i])

        for i, order_o_i in enumerate(order_o):
            from_i = from_[o_conds[i]]
            weight_i = weights[n][o_conds[i]]
            i_dot_w_sum = np.dot(nodes[from_i].T, weight_i)
            i_dot_w_sum = np.clip(i_dot_w_sum, -700, 700)
            nodes[order_o_i] = i_dot_w_sum

        outs[n] = softmax_numba(nodes[outputs].T)
    return outs


net._get_order()
_weights = np.array([net._weights]*100, dtype=np.float64)


n = 100

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
print(time.time() - begin)

begin = time.time()
for i in range(n):
    outputs1 = forward_softmax2d2(X,
                                  net._forward_inputs_array,
                                  net._forward_outputs_array,
                                  net._forward_cond_h,
                                  net._forward_cond_o,
                                  net._forward_culc_order_h,
                                  net._forward_culc_order_o,
                                  _weights,
                                  net._connects[:, 0],
                                  net._forward_activ_code)
print(time.time() - begin)

begin = time.time()
for i in range(n):
    outputs1 = forward_softmax2d3(X,
                                  net._forward_inputs_array,
                                  net._forward_outputs_array,
                                  net._forward_cond_h,
                                  net._forward_cond_o,
                                  net._forward_culc_order_h,
                                  net._forward_culc_order_o,
                                  _weights,
                                  net._connects[:, 0],
                                  net._forward_activ_code)
print(time.time() - begin)

fig, ax = plt.subplots(2)
print_tree(tree, ax[0])
print_net(net, ax[1])
fig.savefig('net.png')
