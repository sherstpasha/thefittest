import numpy as np
import matplotlib.pyplot as plt

from thefittest.benchmarks import Griewank
from thefittest.optimizers import DifferentialEvolution


n_dimension = 100
left_border = -100.0
right_border = 100.0
number_of_iterations = 500
population_size = 500

optimizer = DifferentialEvolution(
    fitness_function=Griewank(),
    iters=number_of_iterations,
    pop_size=population_size,
    left=-100,
    right=100,
    num_variables=n_dimension,
    show_progress_each=10,
    minimization=True,
    mutation="rand_1",
    F=0.1,
    CR=0.5,
    keep_history=True,
)

optimizer.fit()

fittest = optimizer.get_fittest()
stats = optimizer.get_stats()

print("The fittest individ:", fittest["phenotype"])
print("with fitness", fittest["fitness"])

fig, ax = plt.subplots(figsize=(14, 7), ncols=1, nrows=1)
ax.plot(range(number_of_iterations), stats["max_fitness"], label="max_fitness")
ax.set_title("Fitness")
ax.set_ylabel("Fitness value")
ax.set_xlabel("Iterations")
ax.legend()

plt.tight_layout()
plt.show()
