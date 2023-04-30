import time
import numpy as np
import numba
import random
print(np.version.version)
print(numba._version.get_versions())


#0.1.11
from thefittest.tools.transformations import donothing
from thefittest.optimizers import GeneticAlgorithm
def custom_problem(x):
    return np.sum(x, axis=1)
number_of_iterations = 500
population_size = 500
string_length = 1000


def custom_problem(x):
    return np.sum(x, axis=1)


n = 100
begin = time.time()
for i in range(n):
    print(i)
    model = GeneticAlgorithm(fitness_function=custom_problem,
                             genotype_to_phenotype=donothing,
                             iters=number_of_iterations,
                             pop_size=population_size,
                             str_len=string_length,
                            #  show_progress_each=10,
                             minimization=False)

    s_oper = random.choice(['proportional',
                            'rank',
                            'tournament_3',
                            'tournament_5',
                            'tournament_7'])
    
    c_oper = random.choice(['empty',
                            'one_point',
                            'two_point',
                            'uniform2',
                            # 'uniform7',
                            # 'uniform_prop2',
                            # 'uniform_prop7',
                            # 'uniform_rank2',
                            # 'uniform_rank7',
                            # 'uniform_tour3',
                            # 'uniform_tour7'
                            ])
    
    m_oper = random.choice(['weak',
                            'average',
                            'strong'])
    
    model.set_strategy(mutation_oper=m_oper, crossover_oper=c_oper, selection_oper=s_oper)

    model.fit()
print((time.time() - begin)/n)