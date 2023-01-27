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


# problems_dict = {'F1': {'function': ShiftedSphere,
# 'bounds': (-100, 100),
# 'bounds_init_population': (-100, 100)}



# }
# problems_list = (ShiftedSphere,
#                  ShiftedSchwefe1_2,
#                  ShiftedRotatedHighConditionedElliptic,
#                  ShiftedSchwefe1_2WithNoise,
#                  Schwefel2_6,
#                  ShiftedRosenbrock,
#                  ShiftedRotatedGriewank,
#                  ShiftedRotatedAckley,
#                  )
