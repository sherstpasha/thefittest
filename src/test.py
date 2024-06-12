from thefittest.benchmarks import TextureDataset


data = TextureDataset()
X = data.get_X()
y = data.get_y()


print(X)
print(y)

print(data.get_y_names())
