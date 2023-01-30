import numpy as np
import pandas as pd
from thefittest.optimizers import SelfCGA
from thefittest.testfuncs import CEC2005
from thefittest.tools import GrayCode
import multiprocessing
from functools import partial


def create_and_run(some_vars_i, some_max_fes_i, some_problem_i, i):
    left = np.full(some_vars_i, some_problem_i['bounds'][0])
    right = np.full(some_vars_i, some_problem_i['bounds'][1])
    h = np.full(some_vars_i, 1e-6)

    grid = GrayCode(fit_by='h').fit(left, right, h)
    parts = grid.parts

    iters = np.sqrt(some_max_fes_i).astype(np.int64)
    pop_size = (some_max_fes_i/iters).astype(np.int64)
    
    some_model = SelfCGA(fitness_function=some_problem_i['function'](), genotype_to_phenotype=grid.transform,
                         iters=iters, pop_size=pop_size,
                         str_len=np.sum(parts), tour_size=5,
                         optimal_value=some_problem_i['optimum'],
                         termination_error_value=some_problem_i['fix_accuracy'],
                         minimization=True)
    if some_problem_i.get('init_bounds'):
        init_population_float = np.random.uniform(some_problem_i['init_bounds'][0],
                                                  some_problem_i['init_bounds'][1],
                                                  size=(pop_size, some_vars_i))
        int_bit = grid.inverse_transform(init_population_float)

        return some_model.fit(int_bit)

    return some_model.fit()


if __name__ == '__main__':

    runs = 25
    vars_and_fes = {10: (1000, 10000, 100000),
                    30: (1000, 10000, 100000, 300000),
                    50: (1000, 10000, 100000, 300000, 500000)}
    indexes = ['1st_(Best)', '7th', '13th_(Median)',
               '19th', '25th_(Worst)', 'Mean', 'Std']
    pool_obj = multiprocessing.Pool(processes=13)

    evaluation_result_error = pd.DataFrame(
        columns=list(CEC2005.problems_dict.keys()))
    evaluation_result_srate = pd.DataFrame(
        columns=indexes + ['success_rate', 'success_performance'])
    errors_for_stats = pd.DataFrame()

    for f_number, problem_i in CEC2005.problems_dict.items():

        temp_series_error = pd.Series([], dtype=np.float64)

        for vars_i, max_fes_list_i in vars_and_fes.items():

            for max_fes_i in max_fes_list_i:
                prefix = f'{vars_i}_{max_fes_i}_'
                print(f_number + '_' + prefix)

                prefix_indexes = [prefix+ind for ind in indexes]

                run_models = partial(
                    create_and_run, vars_i, max_fes_i, problem_i)

                result = np.array(pool_obj.map(run_models, range(25)))

                errors = np.array([result_i.thefittest.fitness - problem_i['optimum']
                                   for result_i in result])
                success = errors <= problem_i['fix_accuracy']

                errors = np.sort(errors)
                errors_value = (errors[0], errors[6], errors[12], errors[18], errors[24], np.mean(
                    errors), np.std(errors))
                new_series_error = pd.Series(data=errors_value,
                                             index=prefix_indexes)

                success_rate = np.mean(success)
                calls = np.array([result_i.calls for result_i in result])
                down = np.sum([success])
                if down == 0:
                    success_perf = 0
                else:
                    success_perf = (np.mean(calls)*runs)/down

                num_fes = np.zeros(shape=(1, 9))
                calls_argsort = np.argsort(calls)
                calls = calls[calls_argsort]

                num_fes[:, :7] = np.array([calls[0], calls[6], calls[12], calls[18],
                                           calls[24], np.mean(calls), np.std(calls)])
                num_fes[:, 7] = success_rate
                num_fes[:, 8] = success_perf

                new_df_srate = pd.DataFrame(data=num_fes,
                                            columns=indexes +
                                            ['success_rate', 'success_performance'],
                                            index=[f_number + '_' + prefix])
                new_df_error_stats = pd.DataFrame(data=errors.reshape(1, -1), index=[f_number + '_' + prefix])

                temp_series_error = pd.concat(
                    [temp_series_error, new_series_error])
                evaluation_result_srate = pd.concat(
                    [evaluation_result_srate, new_df_srate])

                errors_for_stats = pd.concat(
                    [errors_for_stats, new_df_error_stats])

        evaluation_result_error[f_number] = temp_series_error
        break

    evaluation_result_error.to_excel(
        'C:/Users/user/Мой диск/evaluation_result_error.xlsx')
    evaluation_result_srate.to_excel(
        'C:/Users/user/Мой диск/evaluation_result_srate.xlsx')
    errors_for_stats.to_excel(
        'C:/Users/user/Мой диск/evaluation_result_stats.xlsx')
