# ==== imports & setup ====
from thefittest.optimizers import GeneticProgramming
from thefittest.benchmarks import IrisDataset
from thefittest.classifiers import GeneticProgrammingNeuralNetClassifier
from thefittest.base._tree import init_net_uniset

import numpy as np
import torch
import matplotlib.pyplot as plt

# для воспроизводимости (по желанию)
np.random.seed(0)
torch.manual_seed(0)

# ==== данные ====
data = IrisDataset()
X = data.get_X()                   # (150, 4)
y = data.get_y()
from sklearn.preprocessing import minmax_scale
X_scaled = minmax_scale(X)

# ==== эволюционная часть (генерим одну особь и строим две сети) ====
model = GeneticProgrammingNeuralNetClassifier(n_iter=500, pop_size=500)

uniset = init_net_uniset(
    n_variables=X.shape[1],
    input_block_size=3,
    max_hidden_block_size=8,
    offset=True,
)

pop = GeneticProgramming.half_and_half(100, uniset, 15)

# строим новую и старую версии сети из одного и того же генотипа
net = model.genotype_to_phenotype_tree(
    tree=pop[0], n_variables=X.shape[1], n_outputs=3,
    output_activation='softmax', offset=True
)
net_old = model.genotype_to_phenotype_tree_old(
    tree=pop[0], n_variables=X.shape[1], n_outputs=3,
    output_activation='softmax', offset=True
)

# чтобы сравнение было корректным — копируем веса в старую сеть
net_old._weights = net._weights.detach().cpu().numpy().astype(np.float64)

# ==== отрисовка графов (слева new, справа old) ====
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
net.plot(ax=axes[0]); axes[0].set_title("New Net"); axes[0].axis("off")
net_old.plot(ax=axes[1]); axes[1].set_title("Old Net"); axes[1].axis("off")
plt.show()

# ==== один раз переводим данные в тензор (dtype/девайс весов) ====
X_t = torch.as_tensor(X_scaled, device=net._weights.device, dtype=net._weights.dtype)

# ==== прямой проход и сравнение ====
net.compile_torch()
with torch.no_grad():
    y_new = net.forward(X_t)  # torch, shape (N, C)

# старый возвращает numpy (1, N, C) -> в torch, на тот же девайс/тип
y_old_np = net_old.forward(X_scaled)             # (1, N, C)
y_old = torch.from_numpy(y_old_np).squeeze(0).to(device=y_new.device, dtype=y_new.dtype)

# метрики расхождения
abs_diff = (y_new - y_old).abs()
report = {
    "ok": bool(abs_diff.max().item() < 1e-6),
    "max_abs": abs_diff.max().item(),
    "mean_abs": abs_diff.mean().item(),
    "shape_new": tuple(y_new.shape),
    "shape_old": tuple(y_old_np.shape),
}
print(report)

# покажем по несколько строк выходов (new vs old) для наглядности
k = 8
print("\nNEW (first rows):\n", y_new[:k].cpu().numpy())
print("\nOLD (first rows):\n", y_old[:k].cpu().numpy())

# по желанию: совпадают ли классы (аргмакс)?
pred_new = y_new.argmax(dim=1).cpu().numpy()
pred_old = y_old.argmax(dim=1).cpu().numpy()
print("\nargmax equal =", bool(np.all(pred_new == pred_old)))
