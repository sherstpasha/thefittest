import numpy as np
from thefittest.optimizers import SHADE, JADE, DifferentialEvolution, jDE, SaDE2005
from thefittest.benchmarks import Rastrigin
from thefittest.tools.transformations import donothing


n_dimension = 30
left_border = -4.5
right_border = 4.5


number_of_iterations = 1500
population_size = 100


left_border_array = np.full(
    shape=n_dimension, fill_value=left_border, dtype=np.float64)
right_border_array = np.full(
    shape=n_dimension, fill_value=right_border, dtype=np.float64)

model = SHADE(fitness_function=Rastrigin(),
              genotype_to_phenotype=donothing,
              iters=number_of_iterations,
              pop_size=population_size,
              left=left_border_array,
              right=right_border_array,
              show_progress_each=100,
              minimization=True,
              keep_history=True)

model.fit()

print(model.get_remains_calls())

# stats = model.get_stats()
# # print('The fittest individ:', model.thefittest.phenotype)
# # print('with fitness', model.thefittest.fitness)
# import matplotlib.pyplot as plt


# m_proba = {}
# for m_proba_i in stats['m_proba']:
#         for key, value in m_proba_i.items():
#             if key not in m_proba.keys():
#                 m_proba[key] = [value]
#             else:
#                 m_proba[key].append(value)

# fig, ax = plt.subplots(figsize=(14, 7), ncols=1, nrows=3)

# ax[0].plot(range(number_of_iterations), stats['fitness_max'])
# ax[0].set_title('Fitness')
# ax[0].set_ylabel('Fitness value')
# ax[0].set_xlabel("Iterations")


# for key, value in m_proba.items():
#     ax[1].plot(range(number_of_iterations), value, label=key)
# ax[1].legend(loc="upper left")
# ax[1].set_title('Mutation operators')
# ax[1].set_ylabel('Operators proba')
# ax[1].set_xlabel("Iterations")

# ax[2].plot(range(number_of_iterations), stats['CRm'])
# ax[2].set_title('CRm')
# ax[2].set_ylabel('CRm value')
# ax[2].set_xlabel("Iterations")


# plt.tight_layout()
# plt.show()
