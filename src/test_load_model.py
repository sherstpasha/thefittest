from thefittest.base._model import Model
from thefittest.base._net import Net, NetEnsemble
from thefittest.base._ea import MultiGenome
from thefittest.benchmarks import BanknoteDataset
from sklearn.preprocessing import minmax_scale
import numpy as np


data = BanknoteDataset()
X = data.get_X()
y = data.get_y()

X_scaled = minmax_scale(X)



X_scaled = np.hstack([X_scaled, np.ones((X_scaled.shape[0], 1))])


ens = NetEnsemble.load_from_file("ens.pkl")

ct = MultiGenome.load_from_file("common_tree.pkl")

print(ens.forward(X_scaled))


print(ens._meta_algorithm)

print(ct._genotypes)