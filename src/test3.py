import numpy as np
from thefittest.optimizers import DifferentialEvolution
from thefittest.benchmarks import Griewank
from thefittest.tools.transformations import donothing


n_dimension = 100
left_border = -100.
right_border = 100.


number_of_iterations = 500
population_size = 500


left_border_array = np.full(
    shape=n_dimension, fill_value=left_border, dtype=np.float64)
right_border_array = np.full(
    shape=n_dimension, fill_value=right_border, dtype=np.float64)

model = DifferentialEvolution(fitness_function=Griewank(),
                              genotype_to_phenotype=donothing,
                              iters=number_of_iterations,
                              pop_size=population_size,
                              left=left_border_array,
                              right=right_border_array,
                            #   show_progress_each=10,
                              minimization=True)

# model.set_strategy(mutation_oper='rand_1', F_param=0.1, CR_param=0.5)



from line_profiler import LineProfiler
lp = LineProfiler()
lp_wrapper = lp(model.fit)
lp_wrapper()
lp.print_stats()

model = DifferentialEvolution(fitness_function=Griewank(),
                              genotype_to_phenotype=donothing,
                              iters=number_of_iterations,
                              pop_size=population_size,
                              left=left_border_array,
                              right=right_border_array,
                            #   show_progress_each=10,
                              minimization=True)

model.set_strategy(mutation_oper='best_1')



from line_profiler import LineProfiler
lp = LineProfiler()
lp_wrapper = lp(model.fit)
lp_wrapper()
lp.print_stats()