import numpy as np
from thefittest.optimizers import SelfCGP
from thefittest.tools.transformations import donothing
from thefittest.base import FunctionalNode
from thefittest.base import TerminalNode
from thefittest.base import EphemeralNode
from thefittest.base import UniversalSet
from thefittest.tools.operators import Mul
from thefittest.tools.operators import Add
from thefittest.tools.operators import Inv
from thefittest.tools.operators import Pow2
from thefittest.tools.operators import Neg
from thefittest.tools.operators import Cos
from thefittest.tools.operators import Sin
from thefittest.tools.transformations import coefficient_determination
from thefittest.benchmarks.symbolicregression17 import problems_dict


def generator1():
    return np.round(np.random.uniform(0, 10), 4)


def generator2():
    return np.random.randint(0, 10)

function = problems_dict['F1']['function']
left_border = problems_dict['F1']['bounds'][0]
right_border = problems_dict['F1']['bounds'][1]
sample_size = 300
n_dimension = problems_dict['F1']['n_vars']

number_of_iterations = 300
population_size = 500

X = np.array([np.linspace(left_border, right_border, sample_size)
              for _ in range(n_dimension)]).T
y = function(X)


functional_set = [FunctionalNode(Add()),
                  FunctionalNode(Mul()),
                  FunctionalNode(Neg()),
                  FunctionalNode(Inv()),
                  FunctionalNode(Pow2()),
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


model = SelfCGP(fitness_function=fitness_function,
                genotype_to_phenotype=donothing,
                uniset=uniset,
                pop_size=population_size,
                iters=number_of_iterations,
                show_progress_each=10,
                minimization=False,
                keep_history=True)

model.fit()

