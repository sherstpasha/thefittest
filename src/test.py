import numpy as np
from thefittest.utils.crossovers import empty_crossoverGP
from thefittest.base import Tree
from thefittest.base import init_symbolic_regression_uniset


# Example
X = np.array([[0.3, 0.7], [0.3, 1.1], [3.5, 11.0]], dtype=np.float64)
functional_set_names = ("add", "mul", "neg", "inv")
max_tree_level = 5

# Initialize Universal Set for Symbolic Regression
universal_set = init_symbolic_regression_uniset(X, functional_set_names)

# Define the parents, fitness values, ranks, and maximum allowed depth
parents = np.array([Tree.random_tree(universal_set, max_tree_level)], dtype=object)
fitness_values = np.array([0.8], dtype=np.float64)
ranks = np.array([1.0], dtype=np.float64)
max_depth = 7

# Perform empty crossover for genetic programming
offspring = empty_crossoverGP(parents, fitness_values, ranks, max_depth)

print("Original Individual:", parents[0])
print("Offspring After Empty Crossover (GP):", offspring)
