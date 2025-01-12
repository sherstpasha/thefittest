from collections import defaultdict
from thefittest.optimizers._selfcshaga import SelfCSHAGA
from thefittest.benchmarks._optproblems import F15, F11, OneMax
import matplotlib.pyplot as plt
import numpy as np

number_of_iterations = 200
population_size = 250
num_runs = 10  # Number of runs to average over

# Dictionary to accumulate statistics across runs
accumulated_stats = {
    "max_fitness": np.zeros(number_of_iterations),
    "H_MR": np.zeros((num_runs, number_of_iterations)),
    "H_CR": np.zeros((num_runs, number_of_iterations)),
    "s_proba": [defaultdict(list) for _ in range(number_of_iterations)],
    "c_proba": [defaultdict(list) for _ in range(number_of_iterations)],
}

for run in range(num_runs):
    optimizer = SelfCSHAGA(
        fitness_function=OneMax(),
        iters=number_of_iterations,
        pop_size=population_size,
        str_len=1000,
        show_progress_each=1,
        minimization=False,
        keep_history=True,
        elitism=False,
    )

    optimizer.fit()
    stats = optimizer.get_stats()

    # Accumulate max fitness values
    accumulated_stats["max_fitness"] += stats["max_fitness"]

    # Accumulate mutation and crossover rates
    accumulated_stats["H_MR"][run] = np.array(stats["H_MR"]).mean(axis=1)
    accumulated_stats["H_CR"][run] = np.array(stats["H_CR"]).mean(axis=1)

    # Accumulate selection and crossover probabilities
    for i in range(number_of_iterations):
        for key, value in stats["s_proba"][i].items():
            accumulated_stats["s_proba"][i][key].append(value)
        for key, value in stats["c_proba"][i].items():
            accumulated_stats["c_proba"][i][key].append(value)

# Compute averages across runs
accumulated_stats["max_fitness"] /= num_runs
mean_H_MR = accumulated_stats["H_MR"].mean(axis=0)
mean_H_CR = accumulated_stats["H_CR"].mean(axis=0)

# Compute average probabilities for selection and crossover
average_s_proba = [
    {key: np.mean(value) for key, value in iteration.items()}
    for iteration in accumulated_stats["s_proba"]
]
average_c_proba = [
    {key: np.mean(value) for key, value in iteration.items()}
    for iteration in accumulated_stats["c_proba"]
]

# Plotting results
fig, ax = plt.subplots(figsize=(14, 7), ncols=2, nrows=3)

# Plot average fitness over runs
ax[0][0].plot(
    range(number_of_iterations), accumulated_stats["max_fitness"], label="Average Max Fitness"
)
ax[0][0].set_title("Fitness")
ax[0][0].set_ylabel("Fitness value")
ax[0][0].set_xlabel("Iterations")
ax[0][0].legend()

# Plot average selection probabilities
for key in average_s_proba[0].keys():
    ax[0][1].plot(
        range(number_of_iterations),
        [average_s_proba[i][key] for i in range(number_of_iterations)],
        label=key,
    )
ax[0][1].set_title("Selection Probabilities")
ax[0][1].legend()

# Plot average crossover probabilities
for key in average_c_proba[0].keys():
    ax[1][0].plot(
        range(number_of_iterations),
        [average_c_proba[i][key] for i in range(number_of_iterations)],
        label=key,
    )
ax[1][0].set_title("Crossover Probabilities")
ax[1][0].legend()

# Plot average mutation rates
ax[1][1].plot(range(number_of_iterations), mean_H_MR, label="Average Mutation Rate")
ax[1][1].set_title("Mutation Rate")
ax[1][1].legend()

# Plot average crossover rates
ax[2][1].plot(range(number_of_iterations), mean_H_CR, label="Average Crossover Rate")
ax[2][1].set_title("Crossover Rate")
ax[2][1].legend()

plt.tight_layout()
plt.savefig("selfcshaga_avg_100_runs.png")
plt.show()

# Print fittest individual's details from the final run
fittest = optimizer.get_fittest()
print("The fittest individ (last run):", fittest["genotype"])
print("The fittest individ (last run):", fittest["phenotype"])
print("with fitness (last run):", fittest["fitness"])
