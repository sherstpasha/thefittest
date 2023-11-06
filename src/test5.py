from thefittest.optimizers import GeneticAlgorithm
from thefittest.benchmarks import OneMax
import time as time
import numpy as np


number_of_iterations = 5
population_size = 1000
string_length = 1000

optimizer = GeneticAlgorithm(
    fitness_function=OneMax(),
    iters=number_of_iterations,
    pop_size=population_size,
    str_len=string_length,
    show_progress_each=10,
    n_jobs=4,
)
begin = time.time()

optimizer.fit()

print(time.time() - begin)


fittest = optimizer.get_fittest()
