import numpy as np
from thefittest.utils.transformations import GrayCode
from thefittest.benchmarks import Sphere
from thefittest.optimizers import GeneticAlgorithm


n_dimension = 10
left_border = -5.0
right_border = 5.0
n_bits_per_variable = 32

number_of_iterations = 300
population_size = 500

left_border_array = np.full(shape=n_dimension, fill_value=left_border, dtype=np.float64)
right_border_array = np.full(shape=n_dimension, fill_value=right_border, dtype=np.float64)
parts = np.full(shape=n_dimension, fill_value=n_bits_per_variable, dtype=np.int64)

genotype_to_phenotype = GrayCode().fit(
    left_border=left_border_array,
    right_border=right_border_array,
    bits_per_variable=parts,
    num_variables=n_dimension,
)

optimizer = GeneticAlgorithm(
    fitness_function=Sphere(),
    genotype_to_phenotype=genotype_to_phenotype.transform,
    iters=number_of_iterations,
    pop_size=population_size,
    str_len=sum(parts),
    show_progress_each=30,
    minimization=True,
    selection="tournament_k",
    crossover="two_point",
    mutation="weak",
    tour_size=6,
    optimal_value=0.0,
)

optimizer.fit()

fittest = optimizer.get_fittest()

print("The fittest individ:", fittest["genotype"])
print("The fittest individ:", fittest["phenotype"])
print("with fitness", fittest["fitness"])
