from thefittest.base._net import Net
from thefittest.benchmarks import BanknoteDataset
from thefittest.classifiers import MLPEAClassifier
from thefittest.optimizers import SHAGA

data = BanknoteDataset()
X = data.get_X()
y = data.get_y()

model = MLPEAClassifier(
    iters=10,
    pop_size=10,
    weights_optimizer=SHAGA,
    weights_optimizer_args={"iters": 100, "pop_size": 100},
)

n_inputs = X.shape[1]
n_outputs = len(set(y))

net = model._defitne_net(n_inputs, n_outputs)

print(net)
