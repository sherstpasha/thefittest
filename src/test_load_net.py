from thefittest.base._net import Net
from thefittest.benchmarks import BanknoteDataset
from sklearn.preprocessing import minmax_scale
import numpy as np


data = BanknoteDataset()
X = data.get_X()
y = data.get_y()

X_scaled = minmax_scale(X)







net = Net.load_from_file("net.pkl")


X = np.hstack([X_scaled, np.ones((X_scaled.shape[0], 1))])


res = net.forward(X)

print(res)