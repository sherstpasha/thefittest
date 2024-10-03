from collections import defaultdict
from thefittest.optimizers import SelfCGA
from thefittest.benchmarks._optproblems import OneMax, Jump, BalancedString, LeadingOnes, ZeroOne
import matplotlib.pyplot as plt

import numpy as np
from thefittest.tools.transformations import GrayCode
from thefittest.benchmarks import Rastrigin


number_of_iterations = 500

optimizer = SelfCGA(
    fitness_function=Jump(k=5),
    iters=number_of_iterations,
    pop_size=500,
    str_len=1000,
    show_progress_each=1,
    minimization=False,
    # selections=("tournament_k",

    #                ),
    # crossovers=(
    #                "one_point",
    #                ),
    # mutations=("strong",),
    elitism=False,
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
plt.savefig("selfcga.png")
