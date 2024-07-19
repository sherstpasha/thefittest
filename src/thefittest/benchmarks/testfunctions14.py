from ._optproblems import Func1
from ._optproblems import Func2
from ._optproblems import Func3
from ._optproblems import Func4
from ._optproblems import Func5
import numpy as np


problems_dict = {
    "F1": {
        "function": Func1,
        "bounds": (-10, 10),
        "optimum": 0,
        "optimum_x": np.array([0.]),
        "dimentions": [1],
    },
    "F2": {
        "function": Func2,
        "bounds": (-10, 10),
        "optimum": 0,
        "optimum_x": np.array([0.]),
        "dimentions": [1],
    },
    "F3": {
        "function": Func3,
        "bounds": (-4, 4),
        "optimum": 0,
        "optimum_x": np.array([0.]),
        "dimentions": [2],
    },

    "F4": {
        "function": Func4,
        "bounds": (0, 4),
        "optimum": 0,
        "optimum_x": np.array([0.]),
        "dimentions": [2],
    },

    "F5": {
        "function": Func5,
        "bounds": (0, 4),
        "optimum": 0,
        "optimum_x": np.array([0.]),
        "dimentions": [2],
    },
    }