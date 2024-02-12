import numpy as np
import matplotlib.pyplot as plt

from thefittest.benchmarks import Griewank
from thefittest.optimizers import DifferentialEvolution, SHADE, jDE


n_dimension = 10
left_border = -100.0
right_border = 100.0
number_of_iterations = 50
population_size = 50


left_border_array = np.full(shape=n_dimension, fill_value=left_border, dtype=np.float64)
right_border_array = np.full(shape=n_dimension, fill_value=right_border, dtype=np.float64)

optimizer = SHADE(
    fitness_function=Griewank(),
    iters=number_of_iterations,
    pop_size=population_size,
    left=left_border_array,
    right=right_border_array,
    # show_progress_each=10,
    minimization=True,
    keep_history=True,
    random_state=18,
)

optimizer.fit()

print(optimizer.get_fittest())

optimizer = SHADE(
    fitness_function=Griewank(),
    iters=number_of_iterations,
    pop_size=population_size,
    left=left_border_array,
    right=right_border_array,
    # show_progress_each=10,
    minimization=True,
    keep_history=True,
    random_state=18,
)

optimizer.fit()

print(optimizer.get_fittest())
