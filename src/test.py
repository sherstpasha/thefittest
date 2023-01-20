import numpy as np
from thefittest.optimizers import SelfCGA
from thefittest.testfuncs import CEC2005
from thefittest.testfuncs import Sphere

from thefittest.tools import SamplingGrid
from thefittest.tools import GrayCode

n = 10
left = np.full(n, -100, dtype=np.float64)
right = np.full(n, 100, dtype=np.float64)

h = np.full(n, 1e-6, dtype=np.float64)

grid = GrayCode(fit_by='h').fit(left=left, right=right, arg=h)

parts = grid.parts


problem = CEC2005.ShiftedRotatedExpandedScaffes_F6()
model = SelfCGA(
    problem,
    grid.transform, 500, 500, np.sum(parts),
    tour_size=5,
    show_progress_each=10,
    optimal_value=problem.global_optimum,
    termination_error_value=10e-8,
    minimization=True)

model.operators_selector(crossover_opers=['one_point',
                                          'two_point',
                                          'uniform2',
                                          'uniform7',
                                          'uniform_prop2',
                                          'uniform_prop7',
                                          'uniform_rank2',
                                          'uniform_rank7',
                                          'uniform_tour3',
                                          'uniform_tour7'])


# print(CEC2005.evaluation(model, 25))
fittest = model.fit()
print(fittest.fitness)
