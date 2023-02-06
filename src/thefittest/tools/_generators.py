import numpy as np


def cauchy_distribution(loc=0, scale=1, size=1):
    x_ = np.random.standard_cauchy(size = size)
    return loc + scale*x_