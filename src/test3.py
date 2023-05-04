from scipy.optimize import differential_evolution
import numpy as np
from thefittest.optimizers import DifferentialEvolution, JADE, SHADE, jDE, SaDE2005
from thefittest.benchmarks.CEC2005 import ShiftedSphere
from thefittest.benchmarks import Sphere
from thefittest.tools import donothing


n_dimension = 30
left_border = -100.
right_border = 100.


number_of_iterations = 100
population_size = 50


left_border_array = np.full(
    shape=n_dimension, fill_value=left_border, dtype=np.float64)
right_border_array = np.full(
    shape=n_dimension, fill_value=right_border, dtype=np.float64)

model = DifferentialEvolution(fitness_function=Sphere(),
                              genotype_to_phenotype=donothing,
                              iters=number_of_iterations,
                              pop_size=population_size,
                              left=left_border_array,
                              right=right_border_array,
                              show_progress_each=1,
                              minimization=True)

model.set_strategy(mutation_oper='rand_1', F_param=0.5, CR_param=0.5, elitism_param=True)

model.fit()


def sphere(x):
    return np.sum(x**2)


# bounds = [(left_border, right_border) for _ in range(n_dimension)]
# result = differential_evolution(sphere,
#                                 bounds, mutation=0.5, recombination=0.5, strategy='rand1bin', disp=True,
#                                 maxiter=number_of_iterations, popsize=population_size, polish = False,
#                                 updating = 'deferred', init='random')
