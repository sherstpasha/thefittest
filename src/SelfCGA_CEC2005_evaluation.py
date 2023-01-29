import numpy as np
import pandas as pd
from thefittest.optimizers import SelfCGA
from thefittest.testfuncs import CEC2005
from thefittest.tools import GrayCode, SamplingGrid
import multiprocessing


def run_model(some_model):
    return some_model.fit()


if __name__ == '__main__':
    runs = 25
    vars_and_fes = {10: (1000, 10000, 100000),
                    30: (1000, 10000, 100000, 300000),
                    50: (1000, 10000, 100000, 300000, 500000)}
    # vars_and_fes = {10: (1000, 10000),
    #                 30: (1000, 10000),
    #                 50: (1000, 10000)}
    termination = 10e-8

    evaluation_result_error = pd.DataFrame(
        columns=list(CEC2005.problems_dict.keys()))
    evaluation_result_srate = pd.DataFrame(
        columns=list(CEC2005.problems_dict.keys()))
    for f_number, problem_i in CEC2005.problems_dict.items():

        problem = problem_i['function']()
        temp_series_error = pd.Series([], dtype=np.float64)
        temp_series_srate = pd.Series([], dtype=np.float64)
        for vars_i, max_fes_list_i in vars_and_fes.items():

            for max_fes_i in max_fes_list_i:
                prefix = f'{vars_i}_{max_fes_i}_'
                print(f_number + '_' + prefix)

                indexes = ['1st_(Best)', '7th', '13th_(Median)',
                           '19th', '25th_(Worst)', 'Mean', 'Std']
                prefix_indexes = [prefix+ind for ind in indexes]

                left = np.full(vars_i, problem_i['bounds'][0])
                right = np.full(vars_i, problem_i['bounds'][1])
                h = np.full(vars_i, 1e-6)

                grid = GrayCode(fit_by='h').fit(left, right, h)
                parts = grid.parts

                iters = np.sqrt(max_fes_i).astype(np.int64)
                pop_size = (max_fes_i/iters).astype(np.int64)

                pool_obj = multiprocessing.Pool(processes=13)

                result = np.array(pool_obj.map(run_model, [SelfCGA(fitness_function=problem,
                                                                   genotype_to_phenotype=grid.transform,
                                                                   iters=iters, pop_size=pop_size,
                                                                   str_len=np.sum(parts), tour_size=5,
                                                                   optimal_value=problem_i['optimum'],
                                                                   termination_error_value=termination,
                                                                   minimization=True)
                                                           for _ in range(runs)]))
                errors = np.array(
                    [result_i.fitness - problem_i['optimum'] for result_i in result])
                errors = np.sort(errors)
                
                success = np.array([(result_i.fitness - problem_i['optimum'])
                                   <= problem_i['fix_accuracy'] for result_i in result], dtype=np.float64)
                success_rate = np.mean(success)

                new_series_error = pd.Series(data=(errors[0], errors[6], errors[12], errors[18], errors[24], np.mean(errors), np.std(errors)),
                                             index=prefix_indexes)
                new_series_srate = pd.Series(
                    data=[success_rate], index=[prefix])
                temp_series_error = pd.concat(
                    [temp_series_error, new_series_error])
                temp_series_srate = pd.concat(
                    [temp_series_srate, new_series_srate])

        evaluation_result_error[f_number] = temp_series_error
        evaluation_result_srate[f_number] = temp_series_srate
        break

    # print(evaluation_result_error)
    evaluation_result_error.to_excel(
        'C:/Users/user/Мой диск/evaluation_result_error.xlsx')
    evaluation_result_srate.to_excel(
        'C:/Users/user/Мой диск/evaluation_result_srate.xlsx')

# left =
# print(problem)
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
