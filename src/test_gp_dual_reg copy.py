import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from thefittest.regressors._symbolicregressiongp_dual import SymbolicRegressionGP_DUAL
from thefittest.tools.print import print_trees, print_tree
from thefittest.optimizers._pdpshagp import PDPSHAGP

import matplotlib.pyplot as plt
from collections import Counter
from typing import Union
from thefittest.base import FunctionalNode, TerminalNode, EphemeralNode
from thefittest.base._tree import DualNode


def visualize_universal_set(uniset):
    # ------- Терминалы -------
    terminal_names = []
    for term in uniset._terminal_set:
        if isinstance(term, TerminalNode):
            terminal_names.append(term._name)
        elif isinstance(term, EphemeralNode):
            terminal_names.append(f"Ephemeral({term._generator.__name__})")

    terminal_counts = Counter(terminal_names)

    # ------- Функционалы -------
    func_names = []
    for arity_group in uniset._functional_set.values():
        for func_node in arity_group:
            val = func_node._value
            if isinstance(val, DualNode):
                name = f"Dual({val._top_node._name}, {val._bottom_node._value.__class__.__name__})"
            else:
                name = val.__class__.__name__
            func_names.append(name)

    func_counts = Counter(func_names)

    # ------- Рисуем -------
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Терминальные узлы
    axs[0].bar(terminal_counts.keys(), terminal_counts.values())
    axs[0].set_title("Распределение терминальных узлов")
    axs[0].set_ylabel("Частота")
    axs[0].tick_params(axis="x", rotation=45)

    # Функциональные узлы
    axs[1].bar(func_counts.keys(), func_counts.values())
    axs[1].set_title("Распределение функциональных узлов")
    axs[1].set_ylabel("Частота")
    axs[1].tick_params(axis="x", rotation=90)

    plt.tight_layout()
    plt.show()


# 📊 Генерация данных: сложная одномерная зависимость
np.random.seed(42)
X = np.linspace(-5, 5, 200).reshape(-1, 1)
y = (
    np.sin(2 * X[:, 0])
    + 0.3 * X[:, 0] ** 2
    - np.cos(X[:, 0])
    + np.exp(-0.1 * X[:, 0] ** 2)
    + np.random.normal(0, 0.1, size=len(X))
)
# y = np.sin(X[:, 0])

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ⚙️ Создание модели
model = SymbolicRegressionGP_DUAL(
    iters=1000,
    pop_size=100,
    aggregation="weighted",  # или "mean"
    optimizer_args={"show_progress_each": 5, "max_level": 5, "no_increase_num": 100},
    optimizer=PDPSHAGP,
    p_dual=0.1,
)

# Обучение модели
model.fit(X_train.astype(np.float32), y_train.astype(np.float32))

uniset = model._get_uniset(X_train)
visualize_universal_set(uniset)

# Строковое представление модели
print("📘 Модель:\n", model.print_expression())

# Предсказания
y_pred = model.predict(X_test.astype(np.float32))

# 🎯 Метрики
print("R2 score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

# 🌳 Отображение деревьев
trees = model.get_optimizer().get_fittest()["phenotype"]
tree_ = model.get_optimizer().get_fittest()["genotype"]
print_trees(trees)

# 📈 Построение графиков
x_vis = np.linspace(-5, 5, 500).reshape(-1, 1)
true_func = (
    np.sin(2 * x_vis[:, 0])
    + 0.3 * x_vis[:, 0] ** 2
    - np.cos(x_vis[:, 0])
    + np.exp(-0.1 * x_vis[:, 0] ** 2)
)
# true_func = np.sin(x_vis[:, 0])

# Получаем подвыражения
trees_pred = [tree.set_terminals(x0=x_vis[:, 0]) for tree in trees]
component_outputs = np.array([tree() * np.ones(len(x_vis[:, 0])) for tree in trees_pred])

# Предсказание итоговое
if model._aggregation == "mean":
    y_vis_pred = np.mean(component_outputs, axis=0)
elif model._aggregation == "weighted":
    y_vis_pred = model._linear_model.predict(component_outputs.T)
else:
    raise ValueError("Unknown aggregation")

# 📊 Рисование графиков
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# --- 1. Истинные и предсказанные значения ---
axs[0].plot(x_vis, true_func, label="Истинная функция", linewidth=2)
axs[0].plot(x_vis, y_vis_pred, label="Предсказание модели", linestyle="--")
axs[0].set_title("Истинная функция vs Модель")
axs[0].legend()
axs[0].grid(True)

# --- 2. Отдельные компоненты ---
# axs[1].plot(x_vis, true_func, label="Истинная функция", linewidth=2)
# for i, comp in enumerate(component_outputs):
#     axs[1].plot(x_vis, comp, label=f"Компонент {i+1}")
# axs[1].set_title("Компоненты модели")
# axs[1].legend()
# axs[1].grid(True)

plt.tight_layout()
print_tree(tree_)
plt.show()
