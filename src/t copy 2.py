from thefittest.optimizers import PDPGA
from thefittest.utils.transformations import GrayCode
from thefittest.benchmarks import Rastrigin

import numpy as np


n_dimension = 10
left_border = -5.0
right_border = 5.0
number_of_generations = 500
population_size = 500

genotype_to_phenotype = GrayCode()
genotype_to_phenotype.fit(
    left_border=left_border,
    right_border=right_border,
    num_variables=n_dimension,
    h_per_variable=0.001,
)
num_bits = genotype_to_phenotype.get_bits_per_variable().sum()

selection_proba_history = []
crossover_proba_history = []
mutation_proba_history = []
selection_share_history = []
crossover_share_history = []
mutation_share_history = []


def _calc_share(operators, keys):
    counts = {key: 0 for key in keys}
    for op in operators:
        counts[op] += 1
    total = len(operators)
    return {key: counts[key] / total for key in keys}


def on_generation(optimizer):
    s_keys = list(optimizer._selection_proba.keys())
    c_keys = list(optimizer._crossover_proba.keys())
    m_keys = list(optimizer._mutation_proba.keys())

    selection_proba_history.append(optimizer._selection_proba.copy())
    crossover_proba_history.append(optimizer._crossover_proba.copy())
    mutation_proba_history.append(optimizer._mutation_proba.copy())

    selection_share_history.append(_calc_share(optimizer._selection_operators, s_keys))
    crossover_share_history.append(_calc_share(optimizer._crossover_operators, c_keys))
    mutation_share_history.append(_calc_share(optimizer._mutation_operators, m_keys))


optimizer = PDPGA(
    fitness_function=Rastrigin(),
    genotype_to_phenotype=genotype_to_phenotype.transform,
    iters=number_of_generations,
    pop_size=population_size,
    str_len=num_bits,
    show_progress_each=30,
    minimization=True,
    optimal_value=0.0,
    keep_history=True,
    on_generation=on_generation,
)

optimizer.fit()

fittest = optimizer.get_fittest()
stats = optimizer.get_stats()


print("The fittest individ:", fittest["genotype"])
print("The fittest individ:", fittest["phenotype"])
print("with fitness", fittest["fitness"])
import matplotlib.pyplot as plt

for i in range(len(selection_proba_history)):
    epoch = i + 1
    print(f"Epoch {epoch}")
    print("  selection proba:", selection_proba_history[i])
    print("  selection share:", selection_share_history[i])
    print("  crossover proba:", crossover_proba_history[i])
    print("  crossover share:", crossover_share_history[i])
    print("  mutation proba:", mutation_proba_history[i])
    print("  mutation share:", mutation_share_history[i])


def plot_mean_history(ax, history, title, ylabel="Mean value"):
    if not history:
        ax.set_visible(False)
        return
    values = np.array([np.mean(v) for v in history], dtype=np.float64)
    x = np.arange(len(values))
    ax.plot(x, values)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)


def plot_proba_history(ax, history, title, ylabel="Probability"):
    if not history:
        ax.set_visible(False)
        return
    keys = list(history[0].keys())
    x = np.arange(len(history))
    for key in keys:
        ax.plot(x, [h[key] for h in history], label=key)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, ncol=2)


fig_stats, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12))
axes = axes.ravel()

plot_proba_history(axes[0], selection_proba_history, "Selection probabilities")
plot_proba_history(axes[1], selection_share_history, "Selection usage share", ylabel="Share")
plot_proba_history(axes[2], crossover_proba_history, "Crossover probabilities")
plot_proba_history(axes[3], crossover_share_history, "Crossover usage share", ylabel="Share")
plot_proba_history(axes[4], mutation_proba_history, "Mutation probabilities")
plot_proba_history(axes[5], mutation_share_history, "Mutation usage share", ylabel="Share")

plt.tight_layout()
plt.show()
