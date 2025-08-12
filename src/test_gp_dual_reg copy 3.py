import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from thefittest.regressors._symbolicregressiongp_dual import SymbolicRegressionGP_DUAL
from thefittest.tools.print import print_trees, print_tree
from thefittest.optimizers._pdpshagp import PDPSHAGP

from collections import Counter
from typing import Union
from thefittest.base import FunctionalNode, TerminalNode, EphemeralNode
from thefittest.base._tree import DualNode


# 🎲 Генерация данных
np.random.seed(42)
X = np.linspace(-5, 5, 200).reshape(-1, 1)
y = (
    np.sin(2 * X[:, 0])
    + 0.3 * X[:, 0] ** 2
    - np.cos(X[:, 0])
    + np.exp(-0.1 * X[:, 0] ** 2)
    + np.random.normal(0, 0.1, size=len(X))
)

# 🔀 Разделение
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ⚙️ Модель
model = SymbolicRegressionGP_DUAL(
    iters=1000,
    pop_size=100,
    optimizer_args={"show_progress_each": 5, "max_level": 3, "no_increase_num": 100},
    optimizer=PDPSHAGP,
    p_dual=0.1,
    meta_model=LinearRegression(),
)

# 🧠 Обучение
model.fit(X_train.astype(np.float32), y_train.astype(np.float32))
uniset = model._get_uniset(X_train)

# 🔮 Предсказания
y_pred = model.predict(X_test.astype(np.float32))

# 🎯 Метрики
print("R2 score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred))

# 🌲 Деревья
trees = model.get_optimizer().get_fittest()["phenotype"]
tree_ = model.get_optimizer().get_fittest()["genotype"]
print_trees(trees)

# 📉 Графики
x_vis = np.linspace(-5, 5, 500).reshape(-1, 1)
true_func = (
    np.sin(2 * x_vis[:, 0])
    + 0.3 * x_vis[:, 0] ** 2
    - np.cos(x_vis[:, 0])
    + np.exp(-0.1 * x_vis[:, 0] ** 2)
)

trees_pred = [tree.set_terminals(x0=x_vis[:, 0]) for tree in trees]
component_outputs = np.array([tree() * np.ones(len(x_vis[:, 0])) for tree in trees_pred])

# 📈 Предсказание
y_vis_pred = (
    model._fitted_meta_model.predict(component_outputs.T)
    if model._fitted_meta_model is not None
    else np.mean(component_outputs, axis=0)
)

# 📊 Отрисовка
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].plot(x_vis, true_func, label="Истинная функция", linewidth=2)
axs[0].plot(x_vis, y_vis_pred, label="Предсказание модели", linestyle="--")
axs[0].set_title("Истинная функция vs Модель")
axs[0].legend()
axs[0].grid(True)

plt.tight_layout()
print_tree(tree_)
plt.show()
