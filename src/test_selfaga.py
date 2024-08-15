from collections import defaultdict
from thefittest.optimizers._my_adapt_ga_var2 import MyAdaptGAVar2
from thefittest.optimizers import SelfCGA
from thefittest.benchmarks import Sphere
import matplotlib.pyplot as plt

import numpy as np
from thefittest.tools.transformations import GrayCode
from thefittest.benchmarks import Rastrigin


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
optimizer = MyAdaptGAVar2(
    fitness_function=Rastrigin(),
    genotype_to_phenotype=genotype_to_phenotype.transform,
    iters=number_of_iterations,
    pop_size=population_size,
    str_len=sum(parts),
    show_progress_each=1,
    minimization=True,
    selections=("tournament_k",
                 "rank",
                   "proportional"
                   ),
    crossovers=("two_point",
                 "one_point",
                   "uniform_2"
                   ),
    # mutations=("weak", "average", "strong"),
    tour_size=5,
    keep_history=True,
    adaptation_operator="rank",
    mutate_operator_proba=0.01
    # adaptation_tour_size=2,
)


optimizer.fit()

fittest = optimizer.get_fittest()
stats = optimizer.get_stats()

print(stats.keys())

print("The fittest individ:", fittest["genotype"])
print("The fittest individ:", fittest["phenotype"])
print("with fitness", fittest["fitness"])
fig, ax = plt.subplots(figsize=(14, 7), ncols=2, nrows=3)

ax[0][0].plot(range(number_of_iterations), stats["max_fitness"])
ax[0][0].set_title("Fitness")
ax[0][0].set_ylabel("Fitness value")
ax[0][0].set_xlabel("Iterations")

selection_fitness = defaultdict(list)
for i in range(number_of_iterations):
    for key, value in stats["s_mean_fit"][i].items():
        selection_fitness[key].append(value)
for key, value in selection_fitness.items():
    ax[2][0].plot(range(number_of_iterations), value, label=key)
ax[2][0].legend()

crossover_fitness = defaultdict(list)
for i in range(number_of_iterations):
    for key, value in stats["c_mean_fit"][i].items():
        crossover_fitness[key].append(value)
for key, value in crossover_fitness.items():
    ax[2][1].plot(range(number_of_iterations), value, label=key)
ax[2][1].legend()


selectiom_proba = defaultdict(list)
for i in range(number_of_iterations):
    for key, value in stats["s_used"][i].items():
        selectiom_proba[key].append(value)

for key, value in selectiom_proba.items():
    ax[0][1].plot(range(number_of_iterations), value, label=key)
ax[0][1].legend()

crossover_proba = defaultdict(list)
for i in range(number_of_iterations):
    for key, value in stats["c_used"][i].items():
        crossover_proba[key].append(value)

for key, value in crossover_proba.items():
    ax[1][0].plot(range(number_of_iterations), value, label=key)
ax[1][0].legend()

# mutation_proba = defaultdict(list)
# for i in range(number_of_iterations):
#     for key, value in stats["m_probas"][i].items():
#         mutation_proba[key].append(value)

# for key, value in mutation_proba.items():
#     ax[1][1].plot(range(number_of_iterations), value, label=key)
# ax[1][1].legend()

ax[1][1].plot(range(len(stats["m_probas"])), np.array(stats["m_probas"]).mean(axis = 1))
ax[1][1].set_title("m_probas")
ax[1][1].set_ylabel("m_probas value")
ax[1][1].set_xlabel("Iterations")

plt.tight_layout()
plt.savefig("selfaga.png")
