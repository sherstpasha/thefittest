import numpy as np
from thefittest.optimizers import GeneticAlgorithm
from thefittest.benchmarks import OneMax
from thefittest.tools.transformations import donothing

number_of_iterations = 100
population_size = 200
string_length = 1000

model = GeneticAlgorithm(fitness_function=OneMax(),
                         genotype_to_phenotype=donothing,
                         iters=number_of_iterations,
                         pop_size=population_size,
                         str_len=string_length,
                         show_progress_each=10,
                         minimization=False)

model.set_strategy(selection_oper='tournament_k',
                   crossover_oper='uniform2',
                   mutation_oper='custom_rate',
                   elitism_param=True,
                   parents_num_param=7,
                   mutation_rate_param=0.001)

model.fit()

print(model.get_remains_calls())