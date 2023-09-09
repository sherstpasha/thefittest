import numpy as np

from ._optproblems import ShiftedSphere  # 1 Unimodal (5)
from ._optproblems import ShiftedSchwefe1_2  # 2
from ._optproblems import ShiftedRotatedHighConditionedElliptic  # 3
from ._optproblems import ShiftedSchwefe1_2WithNoise  # 4
from ._optproblems import Schwefel2_6  # 5

from ._optproblems import ShiftedRosenbrock  # 6 Multimodal (20) Basic Functions (7)
from ._optproblems import ShiftedRotatedGriewank  # 7
from ._optproblems import ShiftedRotatedAckley  # 8
from ._optproblems import ShiftedRastrigin  # 9
from ._optproblems import ShiftedRotatedRastrigin  # 10
from ._optproblems import ShiftedRotatedWeierstrass  # 11
from ._optproblems import Schwefel2_13  # 12

from ._optproblems import ShiftedExpandedGriewankRosenbrock  # 13 Expanded Functions (2)
from ._optproblems import ShiftedRotatedExpandedScaffes_F6  # 14

from ._optproblems import HybridCompositionFunction1  # 15 Hybrid Composition Functions (11)
from ._optproblems import RotatedVersionHybridCompositionFunction1  # 16
from ._optproblems import RotatedVersionHybridCompositionFunction1Noise  # 17
from ._optproblems import RotatedHybridCompositionFunction  # 18
from ._optproblems import RotatedHybridCompositionFunctionNarrowBasin  # 19
from ._optproblems import RotatedHybridCompositionFunctionOptimalBounds  # 20
from ._optproblems import HybridCompositionFunction3  # 21
from ._optproblems import HybridCompositionFunction3H  # 22
from ._optproblems import NonContinuousHybridCompositionFunction3  # 23
from ._optproblems import HybridCompositionFunction4  # 24
from ._optproblems import HybridCompositionFunction4withoutbounds  # 25


'''Suganthan, Ponnuthurai & Hansen, Nikolaus & Liang, Jing & Deb, Kalyan & Chen, Ying-ping & Auger,
Anne & Tiwari, Santosh. (2005). Problem Definitions and Evaluation Criteria for the CEC 2005 Special
Session on Real-Parameter Optimization. Natural Computing. 341-357'''

problems_dict = {'F1': {'function': ShiftedSphere,
                        'bounds': (-100, 100),
                        'fix_accuracy': 1e-6,
                        'optimum': -450,
                        'dimentions': range(2, 101)},
                 'F2': {'function': ShiftedSchwefe1_2,
                        'bounds': (-100, 100),
                        'fix_accuracy': 1e-6,
                        'optimum': -450,
                        'dimentions': range(2, 101)},
                 'F3': {'function': ShiftedRotatedHighConditionedElliptic,
                        'bounds': (-100, 100),
                        'fix_accuracy': 1e-6,
                        'optimum': -450,
                        'dimentions': (2, 10, 30, 50)},
                 'F4': {'function': ShiftedSchwefe1_2WithNoise,
                        'bounds': (-100, 100),
                        'fix_accuracy': 1e-6,
                        'optimum': -450,
                        'dimentions': range(2, 101)},
                 'F5': {'function': Schwefel2_6,
                        'bounds': (-100, 100),
                        'fix_accuracy': 1e-6,
                        'optimum': -310,
                        'dimentions': range(2, 101)},
                 'F6': {'function': ShiftedRosenbrock,
                        'bounds': (-100, 100),
                        'fix_accuracy': 1e-2,
                        'optimum': 390,
                        'dimentions': range(2, 101)},
                 'F7': {'function': ShiftedRotatedGriewank,
                        'bounds': (-1000, 1000),
                        'fix_accuracy': 1e-2,
                        'optimum': -180,
                        'init_bounds': (0, 600),
                        'dimentions': (2, 10, 30, 50)},
                 'F8': {'function': ShiftedRotatedAckley,
                        'bounds': (-32, 32),
                        'fix_accuracy': 1e-2,
                        'optimum': -140,
                        'dimentions': (2, 10, 30, 50)},
                 'F9': {'function': ShiftedRastrigin,
                        'bounds': (-5, 5),
                        'fix_accuracy': 1e-2,
                        'optimum': -330,
                        'dimentions': range(2, 51)},
                 'F10': {'function': ShiftedRotatedRastrigin,
                         'bounds': (-5, 5),
                         'fix_accuracy': 1e-2,
                         'optimum': -330,
                         'dimentions': (2, 10, 30, 50)},
                 'F11': {'function': ShiftedRotatedWeierstrass,
                         'bounds': (-0.5, 0.5),
                         'fix_accuracy': 1e-2,
                         'optimum': 90,
                         'dimentions': (2, 10, 30, 50)},
                 'F12': {'function': Schwefel2_13,
                         'bounds': (-np.pi, np.pi),
                         'fix_accuracy': 1e-2,
                         'optimum': -460,
                         'dimentions': range(2, 101)},
                 'F13': {'function': ShiftedExpandedGriewankRosenbrock,
                         'bounds': (-3, 1),
                         'fix_accuracy': 1e-2,
                         'optimum': -130,
                         'dimentions': range(2, 101)},
                 'F14': {'function': ShiftedRotatedExpandedScaffes_F6,
                         'bounds': (-100, 100),
                         'fix_accuracy': 1e-2,
                         'optimum': -300,
                         'dimentions': (2, 10, 30, 50)},
                 'F15': {'function': HybridCompositionFunction1,
                         'bounds': (-5, 5),
                         'fix_accuracy': 1e-2,
                         'optimum': 120,
                         'dimentions': (2, 10, 30, 50)},
                 'F16': {'function': RotatedVersionHybridCompositionFunction1,
                         'bounds': (-5, 5),
                         'fix_accuracy': 1e-2,
                         'optimum': 120,
                         'dimentions': (2, 10, 30, 50)},
                 'F17': {'function': RotatedVersionHybridCompositionFunction1Noise,
                         'bounds': (-5, 5),
                         'fix_accuracy': 1e-1,
                         'optimum': 120,
                         'dimentions': (2, 10, 30, 50)},
                 'F18': {'function': RotatedHybridCompositionFunction,
                         'bounds': (-5, 5),
                         'fix_accuracy': 1e-1,
                         'optimum': 10,
                         'dimentions': (2, 10, 30, 50)},
                 'F19': {'function': RotatedHybridCompositionFunctionNarrowBasin,
                         'bounds': (-5, 5),
                         'fix_accuracy': 1e-1,
                         'optimum': 10,
                         'dimentions': (2, 10, 30, 50)},
                 'F20': {'function': RotatedHybridCompositionFunctionOptimalBounds,
                         'bounds': (-5, 5),
                         'fix_accuracy': 1e-1,
                         'optimum': 10,
                         'dimentions': (2, 10, 30, 50)},
                 'F21': {'function': HybridCompositionFunction3,
                         'bounds': (-5, 5),
                         'fix_accuracy': 1e-1,
                         'optimum': 360,
                         'dimentions': (2, 10, 30, 50)},
                 'F22': {'function': HybridCompositionFunction3H,
                         'bounds': (-5, 5),
                         'fix_accuracy': 1e-1,
                         'optimum': 360,
                         'dimentions': (2, 10, 30, 50)},
                 'F23': {'function': NonContinuousHybridCompositionFunction3,
                         'bounds': (-5, 5),
                         'fix_accuracy': 1e-1,
                         'optimum': 360,
                         'dimentions': (2, 10, 30, 50)},
                 'F24': {'function': HybridCompositionFunction4,
                         'bounds': (-5, 5),
                         'fix_accuracy': 1e-1,
                         'optimum': 260,
                         'dimentions': (2, 10, 30, 50)},
                 'F25': {'function': HybridCompositionFunction4withoutbounds,
                         'bounds': (-10, 10),
                         'fix_accuracy': 1e-1,
                         'optimum': 260,
                         'init_bounds': (2, 5),
                         'dimentions': (2, 10, 30, 50)}}
