import numpy as np
from thefittest.optimizers import SelfAGA
from thefittest.benchmarks import OneMax
from thefittest.tools import donothing
from collections import defaultdict
import matplotlib.pyplot as plt
from thefittest.tools.transformations import GrayCode
from thefittest.benchmarks import Rastrigin


n_dimension = 30
left_border = -5.
right_border = 5.
n_bits_per_variable = 32

number_of_iterations = 300
population_size = 500
n = 50
all_fitness = []

for i in range(n):
    print(i)
    left_border_array = np.full(
        shape=n_dimension, fill_value=left_border, dtype=np.float64)
    right_border_array = np.full(
        shape=n_dimension, fill_value=right_border, dtype=np.float64)
    parts = np.full(
        shape=n_dimension, fill_value=n_bits_per_variable, dtype=np.int64)

    genotype_to_phenotype = GrayCode(fit_by='parts').fit(left=left_border_array,
                                                         right=right_border_array,
                                                         arg=parts)
    model = SelfAGA(fitness_function=Rastrigin(),
                    genotype_to_phenotype=genotype_to_phenotype.transform,
                    iters=number_of_iterations,
                    pop_size=population_size,
                    str_len=sum(parts),
                    show_progress_each=10,
                    minimization=True)

    model.set_strategy(
        adapting_selection_operator='rank',
        p_operator_param=0.3,
        p_mutate_param=0.3,
    )

    model.fit()

    fittest = model.get_fittest()
    genotype, phenotype, fitness = fittest.get()
    all_fitness.append(fitness*(-1))
    # stats = model.get_stats()
print(np.max(all_fitness), np.mean(all_fitness), np.min(all_fitness))
# s_opers_dict = defaultdict(list)
# for s_opers in stats['s_opers']:
#     values, counts = np.unique(s_opers, return_counts=True)
#     for s_oper in model._selection_set.keys():
#         s_opers_dict[s_oper].append(0)
#     for values_i, counts_i in zip(values, counts):
#         s_opers_dict[values_i][-1] += counts_i

# c_opers_dict = defaultdict(list)
# for c_opers in stats['c_opers']:
#     values, counts = np.unique(c_opers, return_counts=True)
#     for c_oper in model._crossover_set.keys():
#         c_opers_dict[c_oper].append(0)
#     for values_i, counts_i in zip(values, counts):
#         c_opers_dict[values_i][-1] += counts_i

# mutation_mean = []
# mutation_median = []
# for m_rate_i in stats['m_proba']:
#     mutation_mean.append(np.mean(m_rate_i))
#     mutation_median.append(np.median(m_rate_i))

# fig, ax = plt.subplots(figsize=(14, 7), ncols=2, nrows=2)

# ax[0][0].plot(range(number_of_iterations), stats['fitness_max'])
# ax[0][0].set_title('Fitness')
# ax[0][0].set_ylabel('Fitness value')
# ax[0][0].set_xlabel("Iterations")

# for key, value in s_opers_dict.items():
#     ax[0][1].plot(range(number_of_iterations), value, label=key)
# ax[0][1].legend(loc="upper left")
# ax[0][1].set_title('Selection operators')
# ax[0][1].set_ylabel('Operators calls')
# ax[0][1].set_xlabel("Iterations")

# for key, value in c_opers_dict.items():
#     ax[1][0].plot(range(number_of_iterations), value, label=key)
# ax[1][0].legend(loc="upper left")
# ax[1][0].set_title('Crossover operators')
# ax[1][0].set_ylabel('Operators calls')
# ax[1][0].set_xlabel("Iterations")

# ax[1][1].plot(range(number_of_iterations), mutation_mean, label='mean')
# ax[1][1].plot(range(number_of_iterations), mutation_median, label='median')
# ax[1][1].set_title('Mutation rate')
# ax[1][1].set_ylabel('Mutation rate')
# ax[1][1].set_xlabel("Iterations")
# ax[1][1].legend()

# plt.tight_layout()
# plt.show()
# fittest = model.get_fittest()
# genotype, phenotype, fitness = fittest.get()
# print('The fittest individ:', phenotype)
# print('with fitness', fitness)
