from thefittest.optimizers import PDPGA
from thefittest.benchmarks import OneMax
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import defaultdict
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
optimizer = PDPGA(
    fitness_function=Sphere(),
    genotype_to_phenotype=genotype_to_phenotype.transform,
    iters=number_of_iterations,
    pop_size=population_size,
    str_len=sum(parts),
    show_progress_each=1,
    minimization=True,
    # selections=("tournament_k", "rank", "proportional"),
    # crossovers=("two_point", "one_point", "uniform_2", "uniform_rank_2"),
    # mutations=("weak", "average", "strong"),
    tour_size=3,
    keep_history=True,
)

optimizer.fit()

fittest = optimizer.get_fittest()

stats = optimizer.get_stats()

print("The fittest individ:", fittest["genotype"])
print("The fittest individ:", fittest["phenotype"])
print("with fitness", fittest["fitness"])

fig, ax = plt.subplots(figsize=(14, 7), ncols=2, nrows=2)

ax[0][0].plot(range(number_of_iterations), stats["max_fitness"])
ax[0][0].set_title("Fitness")
ax[0][0].set_ylabel("Fitness value")
ax[0][0].set_xlabel("Iterations")

selectiom_proba = defaultdict(list)
for i in range(number_of_iterations):
    for key, value in stats["s_proba"][i].items():
        selectiom_proba[key].append(value)

for key, value in selectiom_proba.items():
    ax[0][1].plot(range(number_of_iterations), value, label=key)
ax[0][1].legend()

crossover_proba = defaultdict(list)
for i in range(number_of_iterations):
    for key, value in stats["c_proba"][i].items():
        crossover_proba[key].append(value)

for key, value in crossover_proba.items():
    ax[1][0].plot(range(number_of_iterations), value, label=key)
ax[1][0].legend()

mutation_proba = defaultdict(list)
for i in range(number_of_iterations):
    for key, value in stats["m_proba"][i].items():
        mutation_proba[key].append(value)

for key, value in mutation_proba.items():
    ax[1][1].plot(range(number_of_iterations), value, label=key)
ax[1][1].legend()

plt.tight_layout()
plt.savefig("test.png")
