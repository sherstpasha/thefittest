from thefittest.optimizers import GeneticAlgorithm, SelfCGA
from thefittest.benchmarks import OneMax
import time as time
import numpy as np


def problem(population):
    res = []
    for individ in population:
        time.sleep(5)
        res.append(np.sum(individ))
    res = np.array(res, dtype=np.float64)
    return res


number_of_iterations = 2
population_size = 50
string_length = 1000

optimizer = SelfCGA(
    fitness_function=problem,
    iters=number_of_iterations,
    pop_size=population_size,
    str_len=string_length,
    show_progress_each=1,
    n_jobs=4,
)

# optimizer._first_generation()

# pop = optimizer._population_g_i
# optimizer._split_population(pop)
begin = time.time()

optimizer.fit()

print(time.time() - begin)


# fittest = optimizer.get_fittest()
