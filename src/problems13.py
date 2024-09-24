import numpy as np
from thefittest.benchmarks._optproblems import Func1
from thefittest.benchmarks._optproblems import Func2
from thefittest.benchmarks._optproblems import Func3
from thefittest.benchmarks._optproblems import Func4
from thefittest.benchmarks._optproblems import Func5
from thefittest.benchmarks._optproblems import Func6
from thefittest.benchmarks._optproblems import Func7
from thefittest.benchmarks._optproblems import Func8
from thefittest.benchmarks._optproblems import Func9
from thefittest.benchmarks._optproblems import Func10
from thefittest.benchmarks._optproblems import Func11
from thefittest.benchmarks._optproblems import Func12
from thefittest.benchmarks._optproblems import Func13

problems_tuple = (
    {  # 1
        "function": Func1,
        "bounds": (-10, 10),
        "optimum": 0,
        "optimum_x": np.array([0.]),
        "dimention": 1,
        "iters": 10,
        "pop_size": 10,
    },
    {  # 2
        "function": Func2,
        "bounds": (-10, 10),
        "optimum": -10,
        "optimum_x": np.array([0.]),
        "dimention": 1,
        "iters": 22,
        "pop_size": 22,
    },
    {  # 3
        "function": Func3,
        "bounds": (-4, 4),
        "optimum": 0,
        "optimum_x": np.array([0., 0.]),
        "dimention": 2,
        "iters": 22,
        "pop_size": 22,
    },
    {  # 4
        "function": Func4,
        "bounds": (0, 4),
        "optimum": 0,
        "optimum_x": np.array([0., 0.]),
        "dimention": 2,
        "iters": 19,
        "pop_size": 19,
    },
    {  # 5
        "function": Func5,
        "bounds": (-16, 16),
        "optimum": 0,
        "optimum_x": np.array([0., 0.]),
        "dimention": 2,
        "iters": 31,
        "pop_size": 31,
    },
    {  # 6
        "function": Func6,
        "bounds": (-16, 16),
        "optimum": 0,
        "optimum_x": np.array([0., 0.]),
        "dimention": 2,
        "iters": 37,
        "pop_size": 37,
    },
    {  # 7
        "function": Func7,
        "bounds": (-2, 2),
        "optimum": 0,
        "optimum_x": np.array([1., 1.]),
        "dimention": 2,
        "iters": 88,
        "pop_size": 88,
    },
    {  # 8
        "function": Func8,
        "bounds": (-16, 16),
        "optimum": 0,
        "optimum_x": np.array([0., 0.]),
        "dimention": 2,
        "iters": 44,
        "pop_size": 44,
    },
    {  # 9
        "function": Func9,
        "bounds": (-5, 5),
        "optimum": 0,
        "optimum_x": np.array([1., 1.]),
        "dimention": 2,
        "iters": 126,
        "pop_size": 126,
    },
    {  # 10
        "function": Func10,
        "bounds": (-10, 10),
        "optimum": -1,
        "optimum_x": np.array([0., 0.]),
        "dimention": 2,
        "iters": 772,
        "pop_size": 772,
    },
    {  # 11
        "function": Func11,
        "bounds": (-10, 10),
        "optimum": 0,
        "optimum_x": np.array([0., 0.]),
        "dimention": 2,
        "iters": 31,
        "pop_size": 31,
    },
    {  # 12
        "function": Func12,
        "bounds": (0, 4),
        "optimum": -60.91894,
        "optimum_x": np.array([1.9952, 1.9952]),
        "dimention": 2,
        "iters": 16,
        "pop_size": 16,
    },
    {  # 13
        "function": Func13,
        "bounds": (0, 4),
        "optimum": -15.610118,
        "optimum_x": np.array([1.9952, 1.9952]),
        "dimention": 2,
        "iters": 16,
        "pop_size": 16,
    },
)
