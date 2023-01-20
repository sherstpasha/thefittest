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


problem = CEC2005.ShiftedSphere()
model = SelfCGA(
    problem,
    grid.transform, 100, 100, np.sum(parts),
    tour_size=25,
    show_progress_each=30,
    optimal_value=problem.global_optimum,
    termination_error_value=problem.fixed_accuracy)


fittest = model.fit()
print(fittest.fitness)

