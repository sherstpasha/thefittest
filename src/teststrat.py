from sklearn.datasets import load_iris
import numpy as np
from thefittest.tools.transformations import numpy_group_by
from thefittest.tools.random import random_sample
import matplotlib.pyplot as plt


data = load_iris()
X = data.data
y = data.target






X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, 0.04)

print(X_train.shape)
print(X_test.shape)
# test = stratified_sample(y,  0.9)

# sample = y[test]
# print(sample)

# plt.hist(y)
# plt.hist(sample)
# plt.savefig('hist.png')