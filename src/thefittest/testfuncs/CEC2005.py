import numpy as np
#Unimodal (5)
from ._problems import ShiftedSphere  # 1
from ._problems import ShiftedSchwefe1_2  # 2
from ._problems import ShiftedRotatedHighConditionedElliptic  # 3
from ._problems import ShiftedSchwefe1_2WithNoise  # 4
from ._problems import Schwefel2_6  # 5
#Multimodal (20)
# Basic Functions (7)
from ._problems import ShiftedRosenbrock  # 6
from ._problems import ShiftedRotatedGriewank  # 7
from ._problems import ShiftedRotatedAckley  # 8
from ._problems import ShiftedRastrigin  # 9
from ._problems import ShiftedRotatedRastrigin  # 10
from ._problems import ShiftedRotatedWeierstrass  # 11
from ._problems import Schwefel2_13  # 12
# Expanded Functions (2)
# 13
# 14
# Hybrid Composition Functions (11)
# 15
# 16
# 17
# 18
# 19
# 20
# 21
# 22
# 23
# 24
# 25


def evaluation(opt_model, n_runs):
    global_opt = opt_model.fitness_function.global_optimum
    fixed_opt = opt_model.fitness_function.fixed_accuracy

    errors = np.zeros(n_runs)
    success = np.full(n_runs, False)
    calls = np.zeros(n_runs)

    for i in range(n_runs):
        fittest = opt_model.fit()
        errors[i] = fittest.fitness - global_opt
        success[i] = (fittest.fitness - global_opt) <= fixed_opt
        calls[i] = opt_model.calls

    fe_for_successful = calls[success]
    if len(fe_for_successful):
        success_perf = (np.mean(fe_for_successful)*n_runs)/len(fe_for_successful)
    else:
        success_perf = 0
    success_rate = np.sum(success)/n_runs

    to_return = {'median error': np.median(errors),
                 'success_perf': success_perf,
                 'success_rate': success_rate,
                 'errors': errors}
    return to_return
