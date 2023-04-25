import numpy as np
from thefittest.base._net import MultilayerPerceptron

n_vars = 3
X = np.ones((500, n_vars))
model = MultilayerPerceptron(n_vars, (100,), 3)

out = model.forward(X)
print(1, out)

# out2 = model.forward(X)
# print(2, out2)

# print(out == out2)

# import time
# n = 100

# begin = time.time()
# for i in range(n):
#     out = model.forward2(X)
# print(time.time() - begin)


# begin = time.time()
# for i in range(n):
#     out = model.forward(X)
# print(time.time() - begin)