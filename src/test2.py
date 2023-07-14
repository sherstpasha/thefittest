
import matplotlib.pyplot as plt
from thefittest.tools.print import print_net
from thefittest.classifiers import GeneticProgrammingNeuralNetClassifier
from thefittest.tools.print import print_tree
from thefittest.tools.random import train_test_split_stratified
from thefittest.tools.metrics import confusion_matrix
from thefittest.tools.metrics import f1_score
from thefittest.optimizers import SHADE
from thefittest.optimizers import SelfCGA
from thefittest.optimizers import SaDE2005
from thefittest.benchmarks import DigitsDataset
from thefittest.benchmarks import IrisDataset
from thefittest.benchmarks import WineDataset
from thefittest.tools.transformations import scale_data
from thefittest.classifiers import MLPClassifierEA
import numpy as np


data = WineDataset()
# print(data.feature_names)
X = data.get_X()
y = data.get_y()
X = scale_data(X)


X_train, X_test, y_train, y_test = train_test_split_stratified(
    X, y, 0.3)
number_of_iterations = 10000
model = MLPClassifierEA(iters=number_of_iterations, pop_size=300, activation='tanh',
                        hidden_layers=(10, 10, 10), offset=True,
                        show_progress_each=1, keep_history=True,
                        optimizer_weights=SelfCGA)

model.fit(X_train, y_train)

stats = model._optimizer_weights.get_stats()
# print(stats.keys())
# print(np.array(stats['fitness_max']))

y_pred = model.predict(X_test)
print(f1_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

fig, ax = plt.subplots(1)
print_net(model._net, ax)
plt.tight_layout()
plt.show()
