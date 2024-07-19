import numpy as np

from thefittest.benchmarks.CEC2005 import problems_dict
from thefittest.optimizers import GeneticAlgorithm
from thefittest.tools.transformations import GrayCode


def find_solution_error(solution, true_solution):
    return np.abs(solution - true_solution) 


n_vars_list = [2, 10, 30]
eps = 0.01
n_runs = 1000
iters_pop = {"F1": 30, "F2": 35}

for F, content in problems_dict.items():
    
    F =  "F2"
    content = problems_dict[F]

    for n_vars in n_vars_list:
        reliability = 0.
        speed_sum = 0
        range_left = np.nan
        range_right = np.nan
        find_count = 0

        for run_i in range(n_runs):

            left = np.array([content["bounds"][0]]*n_vars, dtype = np.float64)
            right = np.array([content["bounds"][1]]*n_vars, dtype = np.float64)
            h = np.array([eps]*n_vars, dtype = np.float64)

            genotype_to_phenotype = GrayCode().fit(left, right, h)

            str_let = genotype_to_phenotype.parts.sum()

            optimizer = GeneticAlgorithm(fitness_function=content["function"](),
                                        genotype_to_phenotype=genotype_to_phenotype.transform,
                                        iters=iters_pop[F], 
                                        pop_size=iters_pop[F],
                                        str_len=str_let,
                                        elitism=False,
                                        selection="tournament_5",
                                        crossover="uniform_2",
                                        mutation="weak",
                                        keep_history=True,
                                        minimization=True)        
            optimizer.fit()

            stat = optimizer.get_stats()

            speed_i = find_solution_with_precision(stat["max_ph"], content["optimum_x"][:n_vars], h)



        print("F:", F)
        print("n_vars:", n_vars)
        print("iters:", iters_pop[F])
        print("pop_size:", iters_pop[F])
        print("reliability:", reliability)
        print("speed:", speed)
        print("range:", range_left, range_right)

        break
    break
        

        




