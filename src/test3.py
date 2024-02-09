# from thefittest.base._ea import EvolutionaryAlgorithm

# def function(x):
#     return np.sum(x)


# model = EvolutionaryAlgorithm(fitness_function = function,
#         iters = 100,
#         pop_size = 100,
#         elitism = True,
#         init_population = None,
#         genotype_to_phenotype = None,
#         optimal_value = None,
#         termination_error_value = 0.0,
#         no_increase_num = None,
#         minimization = False,
#         show_progress_each = None,
#         keep_history = False,
#         n_jobs = 1,
#         fitness_function_args = None,
#         genotype_to_phenotype_args = None,
#         random_state = None)

# model.fit()


from thefittest.optimizers import GeneticAlgorithm, SHAGA
from thefittest.benchmarks import OneMax

number_of_iterations = 10
population_size = 10
string_length = 50

optimizer = SHAGA(fitness_function=OneMax(),
                         iters=number_of_iterations,
                         pop_size=population_size,
                         str_len=string_length,
                         show_progress_each=None,
                         random_state=18)

optimizer.fit()

print(optimizer.get_fittest())

# optimizer.fit()

# fittest = optimizer.get_fittest()