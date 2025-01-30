from collections import defaultdict
from thefittest.optimizers._selfcga_i import SelfCGA_i
from thefittest.benchmarks import Sphere
import matplotlib.pyplot as plt

import numpy as np
from thefittest.tools.transformations import GrayCode
from thefittest.benchmarks import Rastrigin
import seaborn as sns


n_dimension = 10
left_border = -5.0
right_border = 5.0
n_bits_per_variable = 32

number_of_iterations = 300
population_size = 500

left_border_array = np.full(shape=n_dimension, fill_value=left_border, dtype=np.float64)
right_border_array = np.full(shape=n_dimension, fill_value=right_border, dtype=np.float64)
parts = np.full(shape=n_dimension, fill_value=n_bits_per_variable, dtype=np.int64)

genotype_to_phenotype = GrayCode(fit_by="parts").fit(
    left=left_border_array, right=right_border_array, arg=parts
)
optimizer = SelfCGA_i(
    fitness_function=Rastrigin(),
    genotype_to_phenotype=genotype_to_phenotype.transform,
    iters=number_of_iterations,
    pop_size=population_size,
    str_len=sum(parts),
    # show_progress_each=1,
    minimization=True,
    selections=("tournament_k", "rank", "proportional"),
    crossovers=("two_point", "one_point", "uniform_2", "uniform_rank_2"),
    mutations=("weak", "strong"),
    tour_size=5,
    keep_history=True,
)


optimizer.fit()

fittest = optimizer.get_fittest()
stats = optimizer.get_stats()

# print(stats["s_max"])
# print(stats["c_max"])
# print(stats["m_max"])

print("The fittest individ:", fittest["genotype"])
print("The fittest individ:", fittest["phenotype"])
print("with fitness", fittest["fitness"])
fig, ax = plt.subplots(figsize=(16, 10), ncols=3, nrows=3)

# График Fitness
ax[0][0].plot(range(number_of_iterations), stats["max_fitness"])
ax[0][0].set_title("Fitness")
ax[0][0].set_ylabel("Fitness value")
ax[0][0].set_xlabel("Iterations")

# Графики вероятностей выбора селекции, кроссовера и мутации
selectiom_proba = defaultdict(list)
for i in range(number_of_iterations):
    for key, value in stats["s_proba"][i].items():
        selectiom_proba[key].append(value)
for key, value in selectiom_proba.items():
    ax[0][1].plot(range(number_of_iterations), value, label=key)
ax[0][1].set_title("Selection Probability")
ax[0][1].legend()

crossover_proba = defaultdict(list)
for i in range(number_of_iterations):
    for key, value in stats["c_proba"][i].items():
        crossover_proba[key].append(value)
for key, value in crossover_proba.items():
    ax[0][2].plot(range(number_of_iterations), value, label=key)
ax[0][2].set_title("Crossover Probability")
ax[0][2].legend()

mutation_proba = defaultdict(list)
for i in range(number_of_iterations):
    for key, value in stats["m_proba"][i].items():
        mutation_proba[key].append(value)
for key, value in mutation_proba.items():
    ax[1][0].plot(range(number_of_iterations), value, label=key)
ax[1][0].set_title("Mutation Probability")
ax[1][0].legend()

# Получаем все уникальные операторы из s_max, c_max и m_max
# Получаем уникальные операторы для каждого типа отдельно
unique_selection = list(set(stats["s_max"]))
unique_crossover = list(set(stats["c_max"]))
unique_mutation = list(set(stats["m_max"]))

# Создаём индексы для каждого типа
selection_indices = {op: i for i, op in enumerate(unique_selection)}
crossover_indices = {op: i for i, op in enumerate(unique_crossover)}
mutation_indices = {op: i for i, op in enumerate(unique_mutation)}

# Создаём матрицы
s_max_matrix = np.zeros((len(unique_selection), number_of_iterations))
c_max_matrix = np.zeros((len(unique_crossover), number_of_iterations))
m_max_matrix = np.zeros((len(unique_mutation), number_of_iterations))

# Заполняем матрицы
for i, op in enumerate(stats["s_max"]):
    s_max_matrix[selection_indices[op], i] = 1
for i, op in enumerate(stats["c_max"]):
    c_max_matrix[crossover_indices[op], i] = 1
for i, op in enumerate(stats["m_max"]):
    m_max_matrix[mutation_indices[op], i] = 1

# Тепловая карта для селекций
sns.heatmap(
    s_max_matrix,
    cmap="Blues",
    cbar=False,
    xticklabels=50,
    yticklabels=unique_selection,
    ax=ax[1][1],
)
ax[1][1].set_title("Selection Max Operators")

# Тепловая карта для кроссоверов
sns.heatmap(
    c_max_matrix,
    cmap="Greens",
    cbar=False,
    xticklabels=50,
    yticklabels=unique_crossover,
    ax=ax[2][0],
)
ax[2][0].set_title("Crossover Max Operators")

# Тепловая карта для мутаций
sns.heatmap(
    m_max_matrix,
    cmap="Reds",
    cbar=False,
    xticklabels=50,
    yticklabels=unique_mutation,
    ax=ax[2][1],
)
ax[2][1].set_title("Mutation Max Operators")

plt.tight_layout()
plt.show()
