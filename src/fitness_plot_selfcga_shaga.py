import time
import plotly.express as px
import numpy as np
import pandas as pd

from thefittest.optimizers import SHAGA
from thefittest.optimizers import SelfCGA
from thefittest.testfuncs import OneMax
from thefittest.tools import donothing


problem = OneMax()
iters = 100
pop_size = 100
str_len = 100

model_selfcga = SelfCGA(fitness_function=problem,
                        genotype_to_phenotype=donothing,
                        iters=iters,
                        pop_size=pop_size,
                        str_len=str_len,
                        show_progress_each=20,
                        minimization=False,
                        keep_history=True)

# model_selfcga.set_strategy(crossover_opers=['uniform2',
#                                     'uniform7',
#                                     'uniform_prop2',
#                                     'uniform_prop7',
#                                     'uniform_rank2',
#                                     'uniform_rank7',
#                                     'uniform_tour3',
#                                     'uniform_tour7'])

model_shaga = SHAGA(fitness_function=problem,
                    genotype_to_phenotype=donothing,
                    iters=iters,
                    pop_size=pop_size,
                    str_len=str_len,
                    show_progress_each=20,
                    minimization=False,
                    keep_history=True)
model_selfcga.fit()
model_shaga.fit()

fitness_selfcga = model_selfcga.stats.fitness
fitness_shaga = model_shaga.stats.fitness

for i in range(30):

    model_selfcga = SelfCGA(fitness_function=problem,
                            genotype_to_phenotype=donothing,
                            iters=iters,
                            pop_size=pop_size,
                            str_len=str_len,
                            show_progress_each=20,
                            minimization=False,
                            keep_history=True)

    # model_selfcga.set_strategy(crossover_opers=['uniform2',
    #                                 'uniform7',
    #                                 'uniform_prop2',
    #                                 'uniform_prop7',
    #                                 'uniform_rank2',
    #                                 'uniform_rank7',
    #                                 'uniform_tour3',
    #                                 'uniform_tour7'])

    model_shaga = SHAGA(fitness_function=problem,
                        genotype_to_phenotype=donothing,
                        iters=iters,
                        pop_size=pop_size,
                        str_len=str_len,
                        show_progress_each=20,
                        minimization=False,
                        keep_history=True)
    model_selfcga.fit()
    model_shaga.fit()

    fitness_selfcga += model_selfcga.stats.fitness
    fitness_shaga += model_shaga.stats.fitness

fitness_selfcga = fitness_selfcga/31
fitness_shaga = fitness_shaga/31
# print('done')
# time.sleep(3)
x = np.arange(len(fitness_selfcga))
data1 = pd.DataFrame({'selfcga': fitness_selfcga,
                      'shaga': fitness_shaga})

fig = px.line(data1)
fig.write_html("C:/Users/user/Desktop/file1.html")
# print('done')
# time.sleep(3)
