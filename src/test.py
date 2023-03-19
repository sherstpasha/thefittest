from thefittest.experimental.selfcgpo import SelfCGPO
from thefittest.optimizers._base import UniversalSetO
from thefittest.optimizers._base import FunctionalNode
from thefittest.optimizers._base import TerminalNode
from thefittest.optimizers._base import EphemeralNode
from thefittest.tools.operators import Add
from thefittest.tools.operators import Mul
from thefittest.tools.operators import Inv
from thefittest.tools.operators import Neg
from thefittest.tools.operators import Cos
from thefittest.tools.operators import Sin
from thefittest.tools.operators import SqrtAbs
from thefittest.tools.operators import Pow2
from thefittest.tools.transformations import donothing
from thefittest.tools.transformations import coefficient_determination
from thefittest.benchmarks.symbolicregression17 import problems_dict
import numpy as np


def generator1():
    return np.round(np.random.uniform(0, 10), 4)


F = 'F3'
function = problems_dict[F]['function']
left = problems_dict[F]['bounds'][0]
right = problems_dict[F]['bounds'][1]
size = 300
n_vars = problems_dict[F]['n_vars']
X = np.array([np.linspace(left, right, size) for _ in range(n_vars)]).T

y = function(X)

functional_set = [FunctionalNode(Add()),
                  FunctionalNode(Mul()),
                  FunctionalNode(Pow2()),
                  FunctionalNode(Inv()),
                  FunctionalNode(Neg()),
                  FunctionalNode(SqrtAbs()),
                  FunctionalNode(Cos()),
                  FunctionalNode(Sin())]
terminal_set = [TerminalNode(X[:, i], f'x{i}') for i in range(n_vars)]
ephemeral_set = [EphemeralNode(generator1)]

uniset = UniversalSetO(functional_set, terminal_set, ephemeral_set)


def fitness_function(trees):
    fitness = []
    for tree in trees:
        y_pred = tree.compile()*np.ones(len(y))
        fitness.append(coefficient_determination(y, y_pred))
    fitness = np.array(fitness)
    return np.clip(fitness, -1e10, 1)


# print(uniset.functional_set)
print(uniset.random_functional())
# model = SelfCGPO(fitness_function=fitness_function,
#                 genotype_to_phenotype=donothing,
#                 uniset=uniset,
#                 pop_size=10, iters=10,
#                 show_progress_each=1,
#                 minimization=False,
#                 no_increase_num=300,
#                 keep_history='full')

# # uniset.update_o_set(1)
# # print(model.o_sets)
# # # print(model.o_sets_2)

# model.fit()