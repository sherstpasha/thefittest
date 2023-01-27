import numpy as np
# Unimodal (5)
from ._problems import ShiftedSphere  # 1
from ._problems import ShiftedSchwefe1_2  # 2
from ._problems import ShiftedRotatedHighConditionedElliptic  # 3
from ._problems import ShiftedSchwefe1_2WithNoise  # 4
from ._problems import Schwefel2_6  # 5
# Multimodal (20)
# Basic Functions (7)
from ._problems import ShiftedRosenbrock  # 6
from ._problems import ShiftedRotatedGriewank  # 7
from ._problems import ShiftedRotatedAckley  # 8
from ._problems import ShiftedRastrigin  # 9
from ._problems import ShiftedRotatedRastrigin  # 10
from ._problems import ShiftedRotatedWeierstrass  # 11
from ._problems import Schwefel2_13  # 12
# Expanded Functions (2)
from ._problems import ShiftedExpandedGriewankRosenbrock  # 13
from ._problems import ShiftedRotatedExpandedScaffes_F6  # 14
# Hybrid Composition Functions (11)
from ._problems import HybridCompositionFunction1  # 15
from ._problems import RotatedVersionHybridCompositionFunction1  # 16
from ._problems import RotatedVersionHybridCompositionFunction1Noise  # 17
from ._problems import RotatedHybridCompositionFunction  # 18
from ._problems import RotatedHybridCompositionFunctionNarrowBasin  # 19
from ._problems import RotatedHybridCompositionFunctionOptimalBounds  # 20
from ._problems import HybridCompositionFunction3  # 21
from ._problems import HybridCompositionFunction3H  # 22
from ._problems import NonContinuousHybridCompositionFunction3  # 23
from ._problems import HybridCompositionFunction4  # 24
from ._problems import HybridCompositionFunction4withoutbounds  # 25


problems_dict = {'F1': {'function': ShiftedSphere,
                        'bounds': (-100, 100), 'accuracy': -450 + 1e-6},
                 'F2': {'function': ShiftedSchwefe1_2,
                        'bounds': (-100, 100), 'accuracy': -450 + 1e-6},
                 'F3': {'function': ShiftedRotatedHighConditionedElliptic,
                        'bounds': (-100, 100), 'accuracy': -450 + 1e-6},
                 'F4': {'function': ShiftedSchwefe1_2WithNoise,
                        'bounds': (-100, 100), 'accuracy': -450 + 1e-6},
                 'F5': {'function': Schwefel2_6,
                        'bounds': (-100, 100), 'accuracy': -310 + 1e-6},
                 'F6': {'function': ShiftedRosenbrock,
                        'bounds': (-100, 100), 'accuracy': 390 + 1e-2},
                 'F7': {'function': ShiftedRotatedGriewank,
                        'bounds': (-1000, 1000), 'accuracy': -180 + 1e-2,
                        'init_bounds': (0, 600)},
                 'F8': {'function': ShiftedRotatedAckley,
                        'bounds': (-32, 32), 'accuracy': -140 + 1e-2},
                 'F9': {'function': ShiftedRastrigin,
                        'bounds': (-5, 5), 'accuracy': -330 + 1e-2},              
                 'F10': {'function': ShiftedRotatedRastrigin,
                        'bounds': (-5, 5), 'accuracy': -330 + 1e-2}, 
                 'F11': {'function': ShiftedRotatedWeierstrass,
                        'bounds': (-0.5, 0.5), 'accuracy': 90 + 1e-2}, 
                 'F12': {'function': Schwefel2_13,
                        'bounds': (-np.pi, np.pi), 'accuracy': -460 + 1e-2},  
                 'F13': {'function': ShiftedExpandedGriewankRosenbrock,
                        'bounds': (-3, 1), 'accuracy': -130 + 1e-2},  
                 'F14': {'function': ShiftedRotatedExpandedScaffes_F6,
                        'bounds': (-100, 100), 'accuracy': -300 + 1e-2},  
                 'F15': {'function': HybridCompositionFunction1,
                        'bounds': (-5, 5), 'accuracy': 120 + 1e-2},  
                 'F16': {'function': RotatedVersionHybridCompositionFunction1,
                        'bounds': (-5, 5), 'accuracy': 120 + 1e-2},  
                 'F17': {'function': RotatedVersionHybridCompositionFunction1Noise,
                        'bounds': (-5, 5), 'accuracy': 120 + 1e-1},  
                 'F18': {'function': RotatedHybridCompositionFunction,
                        'bounds': (-5, 5), 'accuracy': 10 + 1e-1},  
                 'F19': {'function': RotatedHybridCompositionFunctionNarrowBasin,
                        'bounds': (-5, 5), 'accuracy': 10 + 1e-1},  
                 'F20': {'function': RotatedHybridCompositionFunctionOptimalBounds,
                        'bounds': (-5, 5), 'accuracy': 10 + 1e-1},  
                 'F21': {'function': HybridCompositionFunction3,
                        'bounds': (-5, 5), 'accuracy': 360 + 1e-1},  
                 'F22': {'function': HybridCompositionFunction3H,
                        'bounds': (-5, 5), 'accuracy': 360 + 1e-1},  
                 'F23': {'function': NonContinuousHybridCompositionFunction3,
                        'bounds': (-5, 5), 'accuracy': 360 + 1e-1},  
                 'F24': {'function': HybridCompositionFunction4,
                        'bounds': (-5, 5), 'accuracy': 260 + 1e-1},  
                 'F25': {'function': HybridCompositionFunction4withoutbounds,
                        'bounds': (-10, 10), 'accuracy': 260 + 1e-1,
                        'init_bounds': (2, 5)}}

