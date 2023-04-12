import numpy as np
# from thefittest.optimizers import GeneticAlgorithm
# from thefittest.optimizers import SelfCGA
# from thefittest.optimizers import DifferentialEvolution
# from thefittest.optimizers import JADE
# from thefittest.optimizers import jDE
# from thefittest.optimizers import SaDE2005
# from thefittest.optimizers import SHADE
# from thefittest.optimizers import SHAGA
# from thefittest.tools.transformations import GrayCode
# from thefittest.benchmarks import Rastrigin
# from thefittest.benchmarks import Griewank
from thefittest.tools.transformations import donothing
from thefittest.base._ea import Statistics
from thefittest.base import FunctionalNode
from thefittest.base import TerminalNode
from thefittest.base import EphemeralNode
from thefittest.base import UniversalSet
from thefittest.optimizers import GeneticProgramming
from thefittest.optimizers import SelfCGP
from thefittest.tools.operators import Mul
from thefittest.tools.operators import Add
from thefittest.tools.transformations import coefficient_determination
from thefittest.benchmarks.symbolicregression17 import problems_dict
from thefittest.tools.operators import Pow2
from thefittest.tools.operators import Div
from thefittest.tools.operators import Neg
from thefittest.tools.operators import Sin
from thefittest.tools.operators import Cos


def problem(x):
    return np.sin(x[:, 0])


def generator1():
    return np.round(np.random.uniform(0, 10), 4)


def generator2():
    return np.random.randint(0, 10)


F = 'F3'
function = problems_dict[F]['function']
left = problems_dict[F]['bounds'][0]
right = problems_dict[F]['bounds'][1]
# function = problem
# left = -4.75
# right = 4.75
size = 300
n_vars = problems_dict[F]['n_vars']
# n_vars = 1
X = np.array([np.linspace(left, right, size) for _ in range(n_vars)]).T

y = function(X)


def fitness_function(trees):
    fitness = []
    for tree in trees:
        y_pred = tree.compile()*np.ones(len(y))
        fitness.append(coefficient_determination(y, y_pred))
    fitness = np.array(fitness)
    return np.clip(fitness, -1e32, 1)


functional_set = [FunctionalNode(Add()),
                  FunctionalNode(Mul()),
                  FunctionalNode(Pow2()),
                  FunctionalNode(Div()),
                  FunctionalNode(Neg()),
                  FunctionalNode(Sin()),
                  FunctionalNode(Cos()),
                  ]


terminal_set = [TerminalNode(X[:, i], f'x{i}') for i in range(n_vars)]

ephemeral_set = [EphemeralNode(generator1), EphemeralNode(generator2)]

uniset = UniversalSet(functional_set, terminal_set, ephemeral_set)

model = SelfCGP(fitness_function=fitness_function,
                genotype_to_phenotype=donothing,
                uniset=uniset,
                pop_size=500, iters=100,
                show_progress_each=1,
                minimization=False,
                no_increase_num=300,
                keep_history=True)
model.fit()
print(model.stats)
# n_dimension = 10
# left_border = -5.
# right_border = 5.
# n_bits_per_variable = 16

# number_of_iterations = 300
# population_size = 300

# left_border_array = np.full(
#     shape=n_dimension, fill_value=left_border, dtype=np.float64)
# right_border_array = np.full(
#     shape=n_dimension, fill_value=right_border, dtype=np.float64)
# parts = np.full(
#     shape=n_dimension, fill_value=n_bits_per_variable, dtype=np.int64)

# genotype_to_phenotype = GrayCode(fit_by='parts').fit(left=left_border_array,
#                                                      right=right_border_array,
#                                                      arg=parts)
# model = GeneticAlgorithm(fitness_function=Rastrigin(),
#                          genotype_to_phenotype=genotype_to_phenotype.transform,
#                          iters=number_of_iterations,
#                          pop_size=population_size,
#                          str_len=sum(parts),
#                          show_progress_each=None,
#                          minimization=True,
#                          keep_history=True)

# model.set_strategy(selection_oper='rank',
#                    crossover_oper='two_point',
#                    mutation_oper='average')

# model.fit()

# print('The fittest individ:', model.thefittest.phenotype)
# print('with fitness', model.thefittest.fitness)
# print(model.stats)


# number_of_iterations = 500
# population_size = 500


# left_border_array = np.full(
#     shape=n_dimension, fill_value=left_border, dtype=np.float64)
# right_border_array = np.full(
#     shape=n_dimension, fill_value=right_border, dtype=np.float64)

# model = DifferentialEvolution(fitness_function=Griewank(),
#                               genotype_to_phenotype=donothing,
#                               iters=number_of_iterations,
#                               pop_size=population_size,
#                               left=left_border_array,
#                               right=right_border_array,
#                               show_progress_each=None,
#                               minimization=True,
#                               keep_history=True)

# model.set_strategy(mutation_oper='rand_1', F_param=0.1, CR_param=0.5)

# model.fit()

# model = JADE(fitness_function=Griewank(),
#                               genotype_to_phenotype=donothing,
#                               iters=number_of_iterations,
#                               pop_size=population_size,
#                               left=left_border_array,
#                               right=right_border_array,
#                               show_progress_each=None,
#                               minimization=True,
#                               keep_history=True)

# model.fit()


# model = jDE(fitness_function=Griewank(),
#                               genotype_to_phenotype=donothing,
#                               iters=number_of_iterations,
#                               pop_size=population_size,
#                               left=left_border_array,
#                               right=right_border_array,
#                               show_progress_each=None,
#                               minimization=True,
#                               keep_history='full')

# model.fit()

# print('The fittest individ:', model.thefittest.phenotype)
# print('with fitness', model.thefittest.fitness)


# model = SaDE2005(fitness_function=Griewank(),
#                               genotype_to_phenotype=donothing,
#                               iters=50,
#                               pop_size=50,
#                               left=left_border_array,
#                               right=right_border_array,
#                               show_progress_each=None,
#                               minimization=True,
#                               keep_history=True)

# model.fit()


# model = SelfCGA(fitness_function=Rastrigin(),
#                 genotype_to_phenotype=genotype_to_phenotype.transform,
#                 iters=number_of_iterations,
#                 pop_size=population_size,
#                 str_len=sum(parts),
#                 show_progress_each=None,
#                 minimization=True,
#                 keep_history=False)

# model.fit()

# print('The fittest individ:', model.thefittest.phenotype)
# print('with fitness', model.thefittest.fitness)
# print(model.stats)

# n_dimension = 100
# left_border = -100.
# right_border = 100.


# model = SHADE(fitness_function=Griewank(),
#                               genotype_to_phenotype=donothing,
#                               iters=number_of_iterations,
#                               pop_size=population_size,
#                               left=left_border_array,
#                               right=right_border_array,
#                               show_progress_each=None,
#                               minimization=True,
#                               keep_history=True)

# model.fit()


# print('The fittest individ:', model.thefittest.phenotype)
# print('with fitness', model.thefittest.fitness)
# print(model.stats)

# model = SHAGA(fitness_function=Rastrigin(),
#               genotype_to_phenotype=genotype_to_phenotype.transform,
#               iters=number_of_iterations,
#               pop_size=population_size,
#               str_len=sum(parts),
#               show_progress_each=None,
#               minimization=True,
#               keep_history=True)

# model.fit()

# print('The fittest individ:', model.thefittest.phenotype)
# print('with fitness', model.thefittest.fitness)
# print(model.stats)
