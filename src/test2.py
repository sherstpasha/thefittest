import numpy as np


from thefittest.utils.random import numba_seed
from thefittest.utils.random import random_sample
from thefittest.utils.mutations import point_mutation


import numpy as np
from thefittest.base import Tree
from thefittest.base import init_symbolic_regression_uniset
from thefittest.utils.mutations import point_mutation
# Example Parameters
X = np.array([[0.3, 0.7], [0.3, 1.1], [3.5, 11.0]], dtype=np.float64)
functional_set_names = ("add", "mul", "neg", "inv")
max_tree_level = 4
mutation_probability = 1
# Initialize Universal Set for Symbolic Regression
universal_set = init_symbolic_regression_uniset(X, functional_set_names)
# Generate a Random Tree
for _ in range(10000):
    tree = Tree.random_tree(universal_set, max_tree_level)
    # Perform Point Mutation

    numba_seed(10)
    mutated_tree1 = point_mutation(tree, universal_set, mutation_probability, max_tree_level)

    numba_seed(10)
    mutated_tree2 = point_mutation(tree, universal_set, mutation_probability, max_tree_level)
    # print("Original Tree:", mutated_tree1)

    # print("Mutated Tree:", mutated_tree2)

    # print(mutated_tree1 == mutated_tree2)
    if mutated_tree1 != mutated_tree2:
        print(mutated_tree1)
        print(mutated_tree2)
        
