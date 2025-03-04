import numpy as np
from opfunu.cec_based.cec2022 import (
    F12022,
    F22022,
    F32022,
    F42022,
    F52022,
    F62022,
    F72022,
    F82022,
    F92022,
    F102022,
    F112022,
    F122022,
)

# Определяем стандартные границы для размерности 10: массив 10 пар [-100, 100]
default_bounds = np.array([[-100.0, 100.0] for _ in range(10)]).T

# Создаём экземпляры функций с нужными параметрами сразу
f1 = F12022(ndim=10, bounds=default_bounds)
f2 = F22022(ndim=10, bounds=default_bounds)
f3 = F32022(ndim=10, bounds=default_bounds)
f4 = F42022(ndim=10, bounds=default_bounds)
f5 = F52022(ndim=10, bounds=default_bounds)
f6 = F62022(ndim=10, bounds=default_bounds)
f7 = F72022(ndim=10, bounds=default_bounds)
f8 = F82022(ndim=10, bounds=default_bounds)
f9 = F92022(ndim=10, bounds=default_bounds)
f10 = F102022(ndim=10, bounds=default_bounds)
f11 = F112022(ndim=10, bounds=default_bounds)
f12 = F122022(ndim=10, bounds=default_bounds)

problems_tuple = (
    {  # Функция 1: Shifted and Rotated High Conditioned Elliptic Function
        "function": f1,
        "bounds": (-100, 100),
        "optimum": f1.f_global,
        "optimum_x": f1.x_global[:10],
        "dimention": 10,
        "iters": 304,
        "pop_size": 304,
    },
    {  # Функция 2: Shifted and Rotated Bent Cigar Function
        "function": f2,
        "bounds": (-100, 100),
        "optimum": f2.f_global,
        "optimum_x": f2.x_global[:10],
        "dimention": 10,
        "iters": 0,
        "pop_size": 0,
    },
    {  # Функция 3: Shifted and Rotated Discus Function
        "function": f3,
        "bounds": (-100, 100),
        "optimum": f3.f_global,
        "optimum_x": f3.x_global[:10],
        "dimention": 10,
        "iters": 0,
        "pop_size": 0,
    },
    {  # Функция 4: Shifted and Rotated Different Powers Function
        "function": f4,
        "bounds": (-100, 100),
        "optimum": f4.f_global,
        "optimum_x": f4.x_global[:10],
        "dimention": 10,
        "iters": 0,
        "pop_size": 0,
    },
    {  # Функция 5: Shifted and Rotated Rosenbrock’s Function
        "function": f5,
        "bounds": (-100, 100),
        "optimum": f5.f_global,
        "optimum_x": f5.x_global[:10],
        "dimention": 10,
        "iters": 0,
        "pop_size": 0,
    },
    {  # Функция 6: Shifted and Rotated Ackley’s Function
        "function": f6,
        "bounds": (-100, 100),
        "optimum": f6.f_global,
        "optimum_x": f6.x_global[:10],
        "dimention": 10,
        "iters": 0,
        "pop_size": 0,
    },
    {  # Функция 7: Shifted and Rotated Weierstrass Function
        "function": f7,
        "bounds": (-100, 100),
        "optimum": f7.f_global,
        "optimum_x": f7.x_global[:10],
        "dimention": 10,
        "iters": 0,
        "pop_size": 0,
    },
    {  # Функция 8: Shifted and Rotated Griewank’s Function
        "function": f8,
        "bounds": (-100, 100),
        "optimum": f8.f_global,
        "optimum_x": f8.x_global[:10],
        "dimention": 10,
        "iters": 0,
        "pop_size": 0,
    },
    {  # Функция 9: Shifted and Rotated Rastrigin’s Function
        "function": f9,
        "bounds": (-100, 100),
        "optimum": f9.f_global,
        "optimum_x": f9.x_global[:10],
        "dimention": 10,
        "iters": 0,
        "pop_size": 0,
    },
    {  # Функция 10: Shifted and Rotated Modified Schwefel’s Function
        "function": f10,
        "bounds": (-100, 100),
        "optimum": f10.f_global,
        "optimum_x": f10.x_global[:10],
        "dimention": 10,
        "iters": 0,
        "pop_size": 0,
    },
    {  # Функция 11: Shifted and Rotated Step Function
        "function": f11,
        "bounds": (-100, 100),
        "optimum": f11.f_global,
        "optimum_x": f11.x_global[:10],
        "dimention": 10,
        "iters": 0,
        "pop_size": 0,
    },
    {  # Функция 12: Shifted and Rotated Katsuura Function
        "function": f12,
        "bounds": (-100, 100),
        "optimum": f12.f_global,
        "optimum_x": f12.x_global[:10],
        "dimention": 10,
        "iters": 0,
        "pop_size": 0,
    },
)
