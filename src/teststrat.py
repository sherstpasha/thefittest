import time
import matplotlib.pyplot as plt
from thefittest.tools.print import print_net
from thefittest.classifiers import GeneticProgrammingNeuralNetClassifier
from sklearn.datasets import load_iris
from thefittest.tools.print import print_tree
from thefittest.tools.random import train_test_split_stratified
from thefittest.tools.metrics import confusion_matrix
from thefittest.tools.metrics import f1_score
from thefittest.tools.random import full_growing_method
import numpy as np
from numba.typed import Dict as numbaDict
from numba import int64
from numba import njit


def order_net(hidden, from_, to_):

    bool_hidden = to_[:, np.newaxis] == np.array(list(hidden))
    order = {j: set(from_[bool_hidden[:, i]])
             for i, j in enumerate(hidden)}
    return order


@njit
def order_net2(hidden, from_, to_):
    keys = np.empty(len(hidden), dtype = np.int64)
    values = []
    for i, hidden_i in enumerate(hidden):
        cond = to_ == hidden_i
        keys[i] = hidden_i
        values.append(from_[cond])
    return keys, values


data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split_stratified(
    X, y, 0.3)

model = GeneticProgrammingNeuralNetClassifier(50, 20, show_progress_each=1)

uniset = model._define_uniset(X)

tree = full_growing_method(uniset, 3)

net = model._genotype_to_phenotype_tree(X.shape[1], len(set(y)), tree)


hidden = net._assemble_hiddens()
from_ = net._connects[:, 0]
to_ = net._connects[:, 1]

hidden_n = np.array(list(hidden), dtype=np.int64)
res2 = order_net2(hidden_n, from_, to_)

n = 1

begin = time.time()
for i in range(n):
    res1 = order_net(hidden, from_, to_)
    print(res1)
print(time.time() - begin)

begin = time.time()
for i in range(n):
    hidden_n = np.array(list(hidden), dtype=np.int64)
    key2, value2 = order_net2(hidden_n, from_, to_)
    print(key2, value2)
print(time.time() - begin)

fig, ax = plt.subplots(2)
print_tree(tree, ax[0])
print_net(net, ax[1])
fig.savefig('net.png')
