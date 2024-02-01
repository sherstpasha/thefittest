from sklearn.preprocessing import minmax_scale


def minmax_scale(data):
    data_copy = data.copy()
    max_value = data_copy.max()
    min_value = data_copy.min()
    if max_value == min_value:
        scaled_data = np.ones_like(data_copy, dtype=np.float64)
    else:
        scaled_data = ((data_copy - min_value) / (max_value - min_value)).astype(np.float64)
    return scaled_data



import numpy as np

# Example data
example_data = np.random.uniform(-100, 100, size = 10).astype(np.float64)*0 + 1

# Scale the data using the scale_data function
scaled_data = scale_data(example_data)
scaled_data2 = minmax_scale(example_data)

# Display original and scaled data
print("Original Data:", example_data)
print("Scaled Data:", scaled_data)
print("Scaled Data 2:", scaled_data2)

from thefittest.utils.random import random_weighted_sample

print(random_weighted_sample(scaled_data, 10, False))

print(random_weighted_sample(scaled_data2, 10, False))