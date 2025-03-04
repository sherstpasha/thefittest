from thefittest.classifiers import MLPEAClassifier, GeneticProgrammingNeuralNetClassifier
from thefittest.benchmarks import DigitsDataset
import numpy as np

from sklearn.metrics import confusion_matrix

import time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = DigitsDataset()
X = data.get_X()
y = data.get_y()

X = torch.tensor(X, device=device, dtype=torch.float64)

model = MLPEAClassifier(
    iters=10000,
    pop_size=10,
    offset=True,
    optimizer_args={"show_progress_each": 1},
)

begin = time.time()

model.fit(X, y)

y_pred = model.predict(X)

print(time.time() - begin)
print(confusion_matrix(y, y_pred))

# net = model.get_net()

# print(net)

# weights = []
# for i in range(99):
#     weights.append(np.random.uniform(-5, 5, size=net._weights.shape))
# weights = np.array(weights, dtype=np.float64)
# print(weights.shape)

# res1 = net.forward(X, weights)
# print("res1", res1)


# res2 = net.build_computation_graph(X, weights)
# print("res2", res2)
