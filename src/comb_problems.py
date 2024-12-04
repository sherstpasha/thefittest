# import numpy as np

# from thefittest.benchmarks._optproblems import OneMax  
# from thefittest.benchmarks._optproblems import ZeroOne
# from thefittest.benchmarks._optproblems import LeadingOnes
# from thefittest.benchmarks._optproblems import Jump

# problems_tuple = (
#     {  # 1
#         "function": OneMax,
#         "str_len": 1000,
#         "optimum": 1000,
#         "optimum_x": np.ones(shape=1000),
#         "iters": 146,
#         "pop_size": 146,
#     },
#     {  # 2
#         "function": ZeroOne,
#         "str_len": 1000,
#         "optimum": 500,
#         "optimum_x": np.tile([0, 1], 500),
#         "iters": 230,
#         "pop_size": 230,
#     },
#     {  # 3
#         "function": LeadingOnes,
#         "str_len": 1000,
#         "optimum": 1000,
#         "optimum_x": np.ones(shape=1000),
#         "iters": 1600,
#         "pop_size": 1600,
#     },
#     {  # 4
#         "function": Jump,
#         "str_len": 1000,
#         "optimum": 1000,
#         "k": 5,
#         "optimum_x": np.ones(shape=1000),
#         "iters": 256,
#         "pop_size": 256,
#     }
# )


import numpy as np

from thefittest.benchmarks._optproblems import OneMax  
from thefittest.benchmarks._optproblems import ZeroOne
from thefittest.benchmarks._optproblems import LeadingOnes
from thefittest.benchmarks._optproblems import Jump
from thefittest.benchmarks._optproblems import F11
from thefittest.benchmarks._optproblems import F12
from thefittest.benchmarks._optproblems import F13
from thefittest.benchmarks._optproblems import F14
from thefittest.benchmarks._optproblems import F15


problems_tuple = (
    {  # 1
        "function": OneMax,
        "str_len": 100,
        "optimum": 100,
        # "optimum_x": np.ones(shape=100),
        "iters": 200,
        "pop_size": 100,
    },
    {  # 2
        "function": F11,
        "str_len": 30,
        "optimum": 5,
        # "optimum_x": np.ones(shape=1000),
        "iters": 200,
        "pop_size": 250,
    },
    {  # 3
        "function": F12,
        "str_len": 30,
        "optimum": 5,
        # "optimum_x": np.tile([0, 1], 500),
        "iters": 200,
        "pop_size": 250,
    },
    {  # 4
        "function": F13,
        "str_len": 24,
        "optimum": 30,
        # "optimum_x": np.ones(shape=1000),
        "iters": 200,
        "pop_size": 250,
    },
    {  # 5
        "function": F14,
        "str_len": 30,
        "optimum": 30,
        # "optimum_x": np.ones(shape=1000),
        "iters": 200,
        "pop_size": 250,
    },
    {  # 6
        "function": F15,
        "str_len": 30,
        "optimum": 5,
        # "optimum_x": np.ones(shape=1000),
        "iters": 200,
        "pop_size": 250,
    },
)