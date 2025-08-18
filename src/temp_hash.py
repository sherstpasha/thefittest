# ==== imports ====
import numpy as np
import torch
from thefittest.optimizers import GeneticProgramming
from thefittest.benchmarks import IrisDataset
from thefittest.classifiers import GeneticProgrammingNeuralNetClassifier
from thefittest.base._tree import init_net_uniset

# ==== прототип кеша ====
class NetCache:
    def __init__(self):
        self._store: dict[str, torch.Tensor] = {}

    def get(self, net) -> torch.Tensor | None:
        sig = net.signature()
        return self._store.get(sig, None)

    def add(self, net, weights: torch.Tensor):
        sig = net.signature()
        self._store[sig] = weights.detach().cpu().clone()

# ==== простая "заглушка" для тренировки весов ====
def fake_train_weights(net, X, y):
    # просто генерируем случайные веса той же длины
    return torch.randn(len(net), dtype=torch.float32)

# ==== данные ====
data = IrisDataset()
X = data.get_X()
y = data.get_y()

# ==== init ====
model = GeneticProgrammingNeuralNetClassifier(n_iter=5, pop_size=5)
uniset = init_net_uniset(
    n_variables=X.shape[1],
    input_block_size=2,
    max_hidden_block_size=4,
    offset=True,
)

# создаём популяцию из 2 особей
pop = GeneticProgramming.half_and_half(2, uniset, 7)

# первая сеть
net1 = model.genotype_to_phenotype_tree(
    tree=pop[0], n_variables=X.shape[1], n_outputs=3,
    output_activation="softmax", offset=True
)
# вторая сеть — её копия
net2 = net1.copy()
# третья сеть — другая структура
net3 = model.genotype_to_phenotype_tree(
    tree=pop[1], n_variables=X.shape[1], n_outputs=3,
    output_activation="softmax", offset=True
)

nets = [net1, net2, net3]

# ==== кеширование ====
cache = NetCache()
trained = {}

for i, net in enumerate(nets, 1):
    sig = net.signature()
    w = cache.get(net)
    if w is None:
        print(f"[{i}] Новая структура, тренируем...")
        w = fake_train_weights(net, X, y)
        cache.add(net, w)
    else:
        print(f"[{i}] Нашли в кеше, подставляем готовые веса")
    trained[sig] = w

print("\nСколько уникальных структур обучено:", len(cache._store))
print("Размеры весов по сигнатурам:")
for sig, w in trained.items():
    print(sig[:8], "->", tuple(w.shape))
