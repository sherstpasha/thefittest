import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from collections import Counter
from thefittest.regressors._symbolicregressiongp_dual import SymbolicRegressionGP_DUAL
from thefittest.optimizers._pdpshagp import PDPSHAGP
from thefittest.base import FunctionalNode, TerminalNode, EphemeralNode
from thefittest.base._tree import DualNode

# --- Генерация данных
np.random.seed(42)
X = np.linspace(-5, 5, 200).reshape(-1, 1)
y = (
    np.sin(2 * X[:, 0])
    + 0.3 * X[:, 0] ** 2
    - np.cos(X[:, 0])
    + np.exp(-0.1 * X[:, 0] ** 2)
    + np.random.normal(0, 0.1, size=len(X))
)

# Разделение
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Параметры эксперимента
p_dual_values = np.arange(0.0, 1.01, 0.15)
n_runs = 20  # количество прогонов для усреднения


# --- Функция одного прогона
def run_single_model(p_dual, seed):
    np.random.seed(seed)

    model = SymbolicRegressionGP_DUAL(
        iters=500,
        pop_size=100,
        aggregation="weighted",
        optimizer_args={"show_progress_each": None, "max_level": 5},
        optimizer=PDPSHAGP,
        p_dual=p_dual,
    )

    model.fit(X_train.astype(np.float32), y_train.astype(np.float32))

    y_pred = model.predict(X_test.astype(np.float32))
    r2 = r2_score(y_test, y_pred)
    n_trees = len(model.get_optimizer().get_fittest()["phenotype"])
    tree_len = len(model.get_optimizer().get_fittest()["genotype"])

    return r2, n_trees, tree_len


# --- Многократный запуск и усреднение
results = []
for p in p_dual_values:
    print(f"▶️ p_dual = {p:.1f} — запуск {n_runs} экспериментов...")

    outputs = Parallel(n_jobs=-1)(  # -1 = использовать все ядра
        delayed(run_single_model)(p, seed) for seed in range(n_runs)
    )

    r2s, n_trees_list, tree_lens = zip(*outputs)

    results.append(
        {
            "p_dual": p,
            "r2": np.mean(r2s),
            "n_trees": np.mean(n_trees_list),
            "length": np.mean(tree_lens),
        }
    )

# --- Графики
results = sorted(results, key=lambda r: r["p_dual"])
p_vals = [r["p_dual"] for r in results]
r2_vals = [r["r2"] for r in results]
n_trees_vals = [r["n_trees"] for r in results]
length_vals = [r["length"] for r in results]

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

axs[0].plot(p_vals, length_vals, marker="o")
axs[0].set_title("Средняя длина дерева vs p_dual")
axs[0].set_xlabel("p_dual")
axs[0].set_ylabel("Узлы (в среднем)")

axs[1].plot(p_vals, n_trees_vals, marker="o")
axs[1].set_title("Среднее количество деревьев vs p_dual")
axs[1].set_xlabel("p_dual")
axs[1].set_ylabel("Поддеревья")

axs[2].plot(p_vals, r2_vals, marker="o")
axs[2].set_title("Средний R² на тесте vs p_dual")
axs[2].set_xlabel("p_dual")
axs[2].set_ylabel("R²")

plt.tight_layout()
plt.show()
