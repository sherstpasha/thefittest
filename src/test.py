import numpy as np
from thefittest.utils.transformations import SamplingGrid

# Инициализация и подгонка сетки выборки
grid = SamplingGrid()
grid.fit(left_border=-5.0, right_border=5.0, num_variables=3, h_per_variable=0.1)

# Получение длины строки для бинарного представления
string_length = grid.get_bits_per_variable().sum()

# Генерация бинарной популяции
binary_population = np.random.randint(2, size=(5, string_length), dtype=np.int8)

# Трансформация бинарной популяции в массив чисел с плавающей точкой
transformed_population = grid.transform(binary_population)
print("Transformed Population:", transformed_population)
