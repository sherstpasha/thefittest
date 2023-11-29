from thefittest.optimizers import GeneticAlgorithm
from thefittest.benchmarks import OneMax

number_of_iterations = 100
population_size = 200
string_length = 1000


optimizer = GeneticAlgorithm(
    fitness_function=OneMax(),
    iters=number_of_iterations,
    pop_size=population_size,
    str_len=string_length,
    show_progress_each=10,
    n_jobs=2,
)

optimizer.fit()

fittest = optimizer.get_fittest()
