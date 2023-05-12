import numpy as np
import time
import random

#0.1.11
from thefittest.optimizers import GeneticProgramming
from thefittest.tools import donothing
from thefittest.base import FunctionalNode
from thefittest.base import TerminalNode
from thefittest.base import EphemeralNode
from thefittest.base import UniversalSet
from thefittest.tools.operators import Mul
from thefittest.tools.operators import Add
from thefittest.tools.operators import Div
from thefittest.tools.operators import Neg
from thefittest.tools.operators import Cos
from thefittest.tools.operators import Sin
from thefittest.tools.metrics import coefficient_determination


def generator1():
    return np.round(np.random.uniform(0, 10), 4)


def generator2():
    return np.random.randint(0, 10)


def problem(x):
    return 3*x[:, 0]**2 + 2*x[:,0] + 5 - 3*np.cos(2*x[:,0])


function = problem
left_border = -4.5
right_border = 4.5
sample_size = 300
n_dimension = 1

number_of_iterations = 200
population_size = 500

X = np.array([np.linspace(left_border, right_border, sample_size)
              for _ in range(n_dimension)]).T
y = function(X)


functional_set = [FunctionalNode(Add()),
                  FunctionalNode(Mul()),
                  FunctionalNode(Neg()),
                  FunctionalNode(Div()),
                  FunctionalNode(Cos()),
                  FunctionalNode(Sin())]


terminal_set = [TerminalNode(X[:, i], f'x{i}') for i in range(n_dimension)]
terminal_set.extend([EphemeralNode(generator1), EphemeralNode(generator2)])
uniset = UniversalSet(functional_set, terminal_set)


def fitness_function(trees):
    fitness = []
    for tree in trees:
        y_pred = tree()*np.ones(len(y))
        fitness.append(coefficient_determination(y, y_pred))
    return np.array(fitness)

n = 1
begin = time.time()
for i in range(n):
    print(i)
    model = GeneticProgramming(fitness_function=fitness_function,
                            genotype_to_phenotype=donothing,
                            uniset=uniset,
                            pop_size=population_size,
                            iters=number_of_iterations,
                            minimization=False,
                            show_progress_each=10,
                            keep_history=False)

    s_oper = random.choice(['proportional',
                            'rank',
                            'tournament_3',
                            'tournament_5',
                            'tournament_7'])
    
    c_oper = random.choice(['empty',
                            'standart',
                            'one_point',
                            'uniform2',
                            'uniform_prop2',
                            'uniform_prop7',
                            'uniform_rank2',
                            'uniform_rank7',
                            'uniform_tour3',
                            'uniform_tour7'])
    
    m_oper = random.choice(['weak_point',
                            'average_point',
                            'strong_point',
                            'weak_grow',
                            'average_grow',
                            'strong_grow'])
    
    model.set_strategy(mutation_oper=m_oper, crossover_oper=c_oper, selection_oper=s_oper)

    model.fit()
print((time.time() - begin)/n)