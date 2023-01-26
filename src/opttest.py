import optproblems.cec2005 as cec
import optproblems
from thefittest.testfuncs import CEC2005
import numpy as np

D = 2
x = np.random.uniform(-5, 5, size = (10, D))

def phenome(x):
    return x

other = cec.F22(D)
my = CEC2005.HybridCompositionFunction3H()

def eval(func, x):
    res = []
    for x_i in x:
        res.append(func(x_i))
    return res


test_1 = eval(other, x)
test_2 = my(x)

print(test_1)
print(test_2)

print(np.mean(np.sum((test_1 - test_2)**2)))
