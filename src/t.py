import numpy as np


def problem(x):
    return np.sin(x[:, 0] * 3) * x[:, 0] * 0.5 + np.cos(x[:, 1] * 2) * x[:, 1] * 0.3


import matplotlib.pyplot as plt

from thefittest.regressors import GeneticProgrammingRegressor
from thefittest.optimizers._cshagp import CSHAGP
from thefittest.optimizers import PDPGP


n_dimension = 2
left_border = -4.5
right_border = 4.5
sample_size = 100


X = np.array([np.linspace(left_border, right_border, sample_size) for _ in range(n_dimension)]).T
y = problem(X)


model = GeneticProgrammingRegressor(
    n_iter=500,
    pop_size=1000,
    optimizer=CSHAGP,
    optimizer_args={
        "show_progress_each": 10,
        "max_level": 9,
        "keep_history": True,
    },
)

model.fit(X, y)
predict = model.predict(X)

tree = model.get_tree()
print("The fittest individ:", tree)

# Получаем статистику
stats = model.get_stats()

# Вычисляем средние H_MR и H_CR по поколениям
stats_H_MR = [np.mean(h) for h in stats["H_MR"]]
stats_H_CR = [np.mean(h) for h in stats["H_CR"]]
stats_best_fit = stats["max_fitness"]

# Найдём, где появился NaN
first_nan_H_MR = next((i for i, v in enumerate(stats_H_MR) if np.isnan(v)), None)
first_nan_H_CR = next((i for i, v in enumerate(stats_H_CR) if np.isnan(v)), None)

print(f"\nПервый NaN в H_MR: поколение {first_nan_H_MR}")
print(f"Первый NaN в H_CR: поколение {first_nan_H_CR}")

if first_nan_H_CR is not None and first_nan_H_CR > 0:
    print(f"\nH_CR до NaN (поколение {first_nan_H_CR - 1}):")
    print(stats["H_CR"][first_nan_H_CR - 1])
    print(f"\nH_CR в момент NaN (поколение {first_nan_H_CR}):")
    print(stats["H_CR"][first_nan_H_CR])

fig, ax = plt.subplots(figsize=(14, 8), ncols=2, nrows=2)

# График предсказаний
ax[0, 0].plot(X[:, 0], y, label="True y")
ax[0, 0].plot(X[:, 0], predict, label="Predict y")
ax[0, 0].legend()
ax[0, 0].set_title("Predictions")

# Дерево
tree.plot(ax=ax[0, 1])
ax[0, 1].set_title("Tree")

# H_MR и H_CR
ax[1, 0].plot(stats_H_MR, label="H_MR mean")
ax[1, 0].plot(stats_H_CR, label="H_CR mean")
ax[1, 0].legend()
ax[1, 0].set_xlabel("Generation")
ax[1, 0].set_title("H_MR and H_CR (history means)")

# Best fitness
ax[1, 1].plot(stats_best_fit, label="Best fitness")
ax[1, 1].legend()
ax[1, 1].set_xlabel("Generation")
ax[1, 1].set_title("Best Fitness")

plt.tight_layout()
plt.show()

# Вывод статистики
print("\n=== Statistics ===")
print(f"Количество поколений: {len(stats['max_fitness'])}")
print(f"Лучший fitness (первый): {stats['max_fitness'][0]:.6f}")
print(f"Лучший fitness (последний): {stats['max_fitness'][-1]:.6f}")
print(f"\nH_MR (средние): начало={stats_H_MR[0]:.4f}, конец={stats_H_MR[-1]:.4f}")
print(f"H_CR (средние): начало={stats_H_CR[0]:.4f}, конец={stats_H_CR[-1]:.4f}")
