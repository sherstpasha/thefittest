from thefittest.base._ea import MutliGenomeEA
from thefittest.base._tree import UniversalSet
from thefittest.base._tree import FunctionalNode
from thefittest.base._tree import TerminalNode
from thefittest.optimizers import DifferentialEvolution
from thefittest.optimizers import GeneticAlgorithm
from thefittest.optimizers import GeneticProgramming
from thefittest.benchmarks import Sphere
from thefittest.tools.operators import Add
from thefittest.tools.operators import Sub
import numpy as np


def problem(phenotype):
    return np.ones(shape=len(phenotype))


n_vars = 10
left = np.full(n_vars, -5, dtype=np.float64)
right = np.full(n_vars, 5, dtype=np.float64)
parts = np.full(n_vars, 12, dtype=np.int64)


functional_set = (FunctionalNode(Add()), FunctionalNode(Sub()))
terminal_set = (TerminalNode(1, "x"), TerminalNode(2, "y"))
uniset = UniversalSet(functional_set=functional_set, terminal_set=terminal_set)

optimizers = [GeneticAlgorithm, GeneticProgramming]
optimizers_args = ({"str_len": sum(parts)}, {"uniset": uniset})

model = MutliGenomeEA(
    optimizers=optimizers,
    optimizers_args=optimizers_args,
    fitness_function=problem,
    iters=10,
    pop_size=5,
)

pop = model._first_generation()
