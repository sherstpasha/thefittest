import numpy as np

from thefittest.benchmarks._optproblems import OneMax  
from thefittest.benchmarks._optproblems import ZeroOne
from thefittest.benchmarks._optproblems import LeadingOnes
from thefittest.benchmarks._optproblems import Jump

problems_tuple = (
    {  # 1
        "function": OneMax,
        "str_len": 1000,
        "optimum": 1000,
        "optimum_x": np.ones(shape=1000),
        "iters": 146,
        "pop_size": 146,
    },
    {  # 2
        "function": ZeroOne,
        "str_len": 1000,
        "optimum": 500,
        "optimum_x": np.tile([0, 1], 500),
        "iters": 230,
        "pop_size": 230,
    },
    {  # 3
        "function": LeadingOnes,
        "str_len": 1000,
        "optimum": 1000,
        "optimum_x": np.ones(shape=1000),
        "iters": 1600,
        "pop_size": 1600,
    },
    {  # 4
        "function": Jump,
        "str_len": 1000,
        "optimum": 1000,
        "k": 5,
        "optimum_x": np.ones(shape=1000),
        "iters": 256,
        "pop_size": 256,
    }
)