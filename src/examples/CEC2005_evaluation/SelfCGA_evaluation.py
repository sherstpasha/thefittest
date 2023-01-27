import numpy as np

from thefittest.optimizers import SelfCGA
from thefittest.testfuncs import CEC2005
from thefittest.tools import GrayCode


print(CEC2005)
# for problem_i in CEC2005.problems_dict:
#     print(problem_i)
# def evaluation(opt_model, n_runs):
#     global_opt = opt_model.fitness_function.fbias
#     fixed_opt = opt_model.fitness_function.fixed_accuracy

#     errors = np.zeros(n_runs)
#     success = np.full(n_runs, False)
#     calls = np.zeros(n_runs)

#     for i in range(n_runs):
#         fittest = opt_model.fit()
#         errors[i] = fittest.fitness - global_opt
#         success[i] = (fittest.fitness - global_opt) <= fixed_opt
#         calls[i] = opt_model.calls

#     fe_for_successful = calls[success]
#     if len(fe_for_successful):
#         success_perf = (np.mean(fe_for_successful)*n_runs) / \
#             len(fe_for_successful)
#     else:
#         success_perf = 0
#     success_rate = np.sum(success)/n_runs

#     to_return = {'median error': np.median(errors),
#                  'success_perf': success_perf,
#                  'success_rate': success_rate,
#                  'errors': errors}
#     return to_return
