from thefittest.benchmarks import TwoNormDataset


data = TwoNormDataset()
X = data.get_X()
y = data.get_y()


print(X)
print(y)
