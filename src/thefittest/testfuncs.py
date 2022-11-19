import numpy as np


def onemax(x): 
    '''
    Type: binary
    Dimensions: any
    Global maximum is equal to the lenght of a binary string
    '''
    return np.sum(x, axis=1)

def sphere(x):
    '''
    Type: float
    Dimensions: any
    Function has no local minima except the global one.
    The function is usually evaluated on the square xi ∈ [-10, 10],
    for all i = 1, 2.
    Global minimum f(x*) = 0 at x* = (0,...,0)
    '''
    return np.sum(x**2, axis=1)

def matyas(x):
    '''
    Type: float
    Dimensions: 2
    xi ∈ [-10, 10], for all i = 1, 2.
    Global minimum f(x*) = 0 at x* = (0, 0)
    '''
    return 0.26*(x[:, 0]**2 + x[:, 1]**2) - 0.48*x[:, 0]*x[:, 1]
