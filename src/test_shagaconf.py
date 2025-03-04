from collections import defaultdict
from thefittest.optimizers._shaga_conf import SHAGACONF
from thefittest.benchmarks import Rastrigin
import matplotlib.pyplot as plt
import numpy as np
from thefittest.tools.transformations import GrayCode

# Параметры
n_dimension = 100
left_border = -5.0
right_border = 5.0
n_bits_per_variable = 32
number_of_iterations = 10000
population_size = 1000
num_experiments = 1000  # Количество повторений

# Границы и преобразование
left_border_array = np.full(shape=n_dimension, fill_value=left_border, dtype=np.float64)
right_border_array = np.full(shape=n_dimension, fill_value=right_border, dtype=np.float64)
parts = np.full(shape=n_dimension, fill_value=n_bits_per_variable, dtype=np.int64)

genotype_to_phenotype = GrayCode(fit_by="parts").fit(
    left=left_border_array, right=right_border_array, arg=parts
)

best_fitnesses = []  # Список для хранения лучших значений фитнеса

for exp in range(num_experiments):
    print(f"Запуск {exp + 1} из {num_experiments}")
    optimizer = SHAGACONF(
        fitness_function=Rastrigin(),
        genotype_to_phenotype=genotype_to_phenotype.transform,
        iters=number_of_iterations,
        pop_size=population_size,
        str_len=sum(parts),
        show_progress_each=1,
        minimization=True,
        selection="tournament_k",
        crossover="empty",
        tour_size=2,
        # keep_history=True,
    )

    optimizer.fit()
    fittest = optimizer.get_fittest()
    best_fitnesses.append(fittest["fitness"])

# Визуализация результатов
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_experiments + 1), best_fitnesses, marker="o", linestyle="-")
plt.xlabel("Эксперимент")
plt.ylabel("Лучшая фитнес-функция")
plt.title("Лучшие найденные решения в каждом эксперименте")
plt.grid()
plt.savefig("optimization_results.png")
plt.show()

print("Среднее лучшее значение фитнес-функции:", np.mean(best_fitnesses))
print("Минимальное лучшее значение фитнес-функции:", np.min(best_fitnesses))
print("Максимальное лучшее значение фитнес-функции:", np.max(best_fitnesses))
