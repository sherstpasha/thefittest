from ._optproblems import Func1
from ._optproblems import Func2
from ._optproblems import Func3
from ._optproblems import Func4
from ._optproblems import Func5
from ._optproblems import Func6
from ._optproblems import Func7
from ._optproblems import Func8
from ._optproblems import Func9
from ._optproblems import Func10
from ._optproblems import Func11
from ._optproblems import Func12
from ._optproblems import Func13
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
        "optimum": -10,
        "optimum_x": np.array([0.,]),
        "dimentions": [1],
    },
    "F3": {
        "function": Func3,
        "bounds": (-4, 4),
        "optimum": 0,
        "optimum_x": np.array([0., 0.]),
        "dimentions": [2],
    },

    "F4": {
        "function": Func4,
        "bounds": (0, 4),
        "optimum": 0,
        "optimum_x": np.array([0., 0.]),
        "dimentions": [2],
    },

    "F5": {
        "function": Func5,
        "bounds": (-16, 16),
        "optimum": 0,
        "optimum_x": np.array([0., 0.]),
        "dimentions": [2],
    },

    "F6": {
        "function": Func6,
        "bounds": (-16, 16),
        "optimum": 0,
        "optimum_x": np.array([0., 0.]),
        "dimentions": [2],
    },

    "F7": {
        "function": Func7,
        "bounds": (-2, 2),
        "optimum": 0,
        "optimum_x": np.array([1., 1.]),
        "dimentions": [2],
    },

    "F8": {
        "function": Func8,
        "bounds": (-16, 16),
        "optimum": 0,
        "optimum_x": np.array([0., 0.]),
        "dimentions": [2],
    },

    "F9": {
        "function": Func9,
        "bounds": (-5, 5),
        "optimum": 0,
        "optimum_x": np.array([1., 1.]),
        "dimentions": [2],
    },

    "F10": {
        "function": Func10,
        "bounds": (-10, 10),
        "optimum": -1,
        "optimum_x": np.array([0., 0.]),
        "dimentions": [2],
    },

    "F11": {
        "function": Func11,
        "bounds": (-10, 10),
        "optimum": 0,
        "optimum_x": np.array([0., 0.]),
        "dimentions": [2],
    },

    "F12": {
        "function": Func12,
        "bounds": (0, 4),
        "optimum": -60.91894,
        "optimum_x": np.array([1.9952, 1.9952]),
        "dimentions": [2],
    },

    "F13": {
        "function": Func13,
        "bounds": (0, 4),
        "optimum": -15.610118,
        "optimum_x": np.array([1.9952, 1.9952]),
        "dimentions": [2],
    },
    }
