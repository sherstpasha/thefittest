from thefittest.optimizers._base import Tree
from thefittest.optimizers._base import UniversalSet
from thefittest.optimizers._operators import Mul
from thefittest.optimizers._operators import Add3
from thefittest.optimizers._operators import Add
from thefittest.optimizers._operators import Pow
from thefittest.optimizers._operators import Cos
from thefittest.optimizers._operators import Sin
from thefittest.optimizers._operators import Neg
from thefittest.optimizers._initializations import growing_method, full_growing_method
import numpy as np


uniset = UniversalSet(functional_set=(Add(),
                                      Cos(),
                                      #   Sin(),
                                      Mul(),
                                      Neg()
                                      ),
                      terminal_set={'x0': np.array([1, 2, 3])},
                      constant_set=(1, 3, 5, 7))

tree_1 = growing_method(uniset, 15)
tree_2 = tree_1.copy()

print(tree_1)
print(tree_2)

print(tree_1 == tree_2)
