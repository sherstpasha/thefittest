import numpy as np

from thefittest.benchmarks._optproblems import ShiftedSphere  # 1 Unimodal (5)
from thefittest.benchmarks._optproblems import ShiftedSchwefe1_2  # 2
from thefittest.benchmarks._optproblems import ShiftedRotatedHighConditionedElliptic  # 3
from thefittest.benchmarks._optproblems import ShiftedSchwefe1_2WithNoise  # 4

from thefittest.benchmarks._optproblems import (
    ShiftedRosenbrock,
)  # 6 Multimodal (20) Basic Functions (7)
from thefittest.benchmarks._optproblems import ShiftedRotatedGriewank  # 7
from thefittest.benchmarks._optproblems import ShiftedRastrigin  # 9
from thefittest.benchmarks._optproblems import ShiftedRotatedRastrigin  # 10
from thefittest.benchmarks._optproblems import Schwefel2_13  # 12

from thefittest.benchmarks._optproblems import (
    ShiftedExpandedGriewankRosenbrock,
)  # 13 Expanded Functions (2)
from thefittest.benchmarks._optproblems import ShiftedRotatedExpandedScaffes_F6  # 14

from thefittest.benchmarks._optproblems import (
    HybridCompositionFunction1,
)  # 15 Hybrid Composition Functions (11)
from thefittest.benchmarks._optproblems import RotatedVersionHybridCompositionFunction1  # 16
from thefittest.benchmarks._optproblems import RotatedVersionHybridCompositionFunction1Noise  # 17
from thefittest.benchmarks._optproblems import RotatedHybridCompositionFunction  # 18
from thefittest.benchmarks._optproblems import RotatedHybridCompositionFunctionNarrowBasin  # 19
from thefittest.benchmarks._optproblems import HybridCompositionFunction3  # 21
from thefittest.benchmarks._optproblems import HybridCompositionFunction3H  # 22
from thefittest.benchmarks._optproblems import NonContinuousHybridCompositionFunction3  # 23
from thefittest.benchmarks._optproblems import HybridCompositionFunction4  # 24
from thefittest.benchmarks._optproblems import HybridCompositionFunction4withoutbounds  # 25

from thefittest.benchmarks import Sphere
from thefittest.benchmarks import Schwefe1_2
from thefittest.benchmarks import Rosenbrock
from thefittest.benchmarks import Rastrigin
from thefittest.benchmarks import Ackley
from thefittest.benchmarks import Weierstrass
from thefittest.benchmarks import HighConditionedElliptic
from thefittest.benchmarks import Griewank
from thefittest.benchmarks._optproblems import ExpandedScaffers_F6


problems_tuple = (
    {  # 1
        "function": ShiftedRosenbrock,
        "bounds": (-100, 100),
        "optimum": 390,
        "optimum_x": ShiftedRosenbrock().x_shift[:2],
        "dimention": 2,
        "iters": 270,
        "pop_size": 270,
    },
    {  # 2
        "function": ShiftedRotatedGriewank,
        "bounds": (-1000, 1000),
        "optimum": -180,
        "optimum_x": ShiftedRotatedGriewank().x_shift[:2],
        "dimention": 2,
        "iters": 650,
        "pop_size": 650,
    },
    {  # 3
        "function": ShiftedExpandedGriewankRosenbrock,
        "bounds": (-3, 1),
        "optimum": -130,
        "optimum_x": ShiftedExpandedGriewankRosenbrock().x_shift[:2],
        "dimention": 2,
        "iters": 110,
        "pop_size": 110,
    },
    {  # 4
        "function": RotatedVersionHybridCompositionFunction1,
        "bounds": (-5, 5),
        "optimum": 120,
        "optimum_x": RotatedVersionHybridCompositionFunction1().x_shift[0][:2],
        "dimention": 2,
        "iters": 130,
        "pop_size": 130,
    },
    {  # 5
        "function": RotatedVersionHybridCompositionFunction1Noise,
        "bounds": (-5, 5),
        "optimum": 120,
        "optimum_x": RotatedVersionHybridCompositionFunction1Noise().x_shift[0][:2],
        "dimention": 2,
        "iters": 130,
        "pop_size": 130,
    },
    {  # 6
        "function": RotatedHybridCompositionFunction,
        "bounds": (-5, 5),
        "optimum": 10,
        "optimum_x": RotatedHybridCompositionFunction().x_shift[0][:2],
        "dimention": 2,
        "iters": 250,
        "pop_size": 250,
    },
    {  # 7
        "function": HybridCompositionFunction3,
        "bounds": (-5, 5),
        "optimum": 360,
        "optimum_x": HybridCompositionFunction3().x_shift[0][:2],
        "dimention": 2,
        "iters": 477,
        "pop_size": 477,
    },
    {  # 8
        "function": HybridCompositionFunction3H,
        "bounds": (-5, 5),
        "optimum": 360,
        "optimum_x": HybridCompositionFunction3H().x_shift[0][:2],
        "dimention": 2,
        "iters": 477,
        "pop_size": 477,
    },
    {  # 9
        "function": NonContinuousHybridCompositionFunction3,
        "bounds": (-5, 5),
        "optimum": 360,
        "optimum_x": NonContinuousHybridCompositionFunction3().x_shift[0][:2],
        "dimention": 2,
        "iters": 477,
        "pop_size": 477,
    },
    {  # 10
        "function": HybridCompositionFunction4,
        "bounds": (-5, 5),
        "optimum": 260,
        "optimum_x": HybridCompositionFunction4().x_shift[0][:2],
        "dimention": 2,
        "iters": 799,
        "pop_size": 799,
    },
    {  # 11
        "function": HybridCompositionFunction4withoutbounds,
        "bounds": (-10, 10),
        "optimum": 260,
        "optimum_x": HybridCompositionFunction4withoutbounds().x_shift[0][:2],
        "dimention": 2,
        "iters": 1000,
        "pop_size": 1000,
    },
    {  # 12
        "function": Rosenbrock,
        "bounds": (-2.048, 2.048),
        "optimum": 0,
        "optimum_x": np.ones(shape=2, dtype=np.float64),
        "dimention": 2,
        "iters": 100,
        "pop_size": 100,
    },
    {  # 13
        "function": ExpandedScaffers_F6,
        "bounds": (-100, 100),
        "optimum": 0,
        "optimum_x": np.zeros(shape=2, dtype=np.float64),
        "dimention": 2,
        "iters": 100,
        "pop_size": 100,
    },
    {  # 14
        "function": Weierstrass,
        "bounds": (-1, 1),
        "optimum": 0,
        "optimum_x": np.zeros(shape=5, dtype=np.float64),
        "dimention": 5,
        "iters": 1157,
        "pop_size": 1157,
    },
    {  # 15
        "function": ShiftedSphere,
        "bounds": (-100, 100),
        "optimum": -450,
        "optimum_x": ShiftedSphere().x_shift[:10],
        "dimention": 10,
        "iters": 125,
        "pop_size": 125,
    },
    {  # 16
        "function": ShiftedSchwefe1_2,
        "bounds": (-100, 100),
        "optimum": -450,
        "optimum_x": ShiftedSchwefe1_2().x_shift[:10],
        "dimention": 10,
        "iters": 728,
        "pop_size": 728,
    },
    {  # 17
        "function": ShiftedSchwefe1_2WithNoise,
        "bounds": (-100, 100),
        "optimum": -450,
        "optimum_x": ShiftedSchwefe1_2WithNoise().x_shift[:10],
        "dimention": 10,
        "iters": 757,
        "pop_size": 757,
    },
    {  # 18
        "function": ShiftedRastrigin,
        "bounds": (-5, 5),
        "optimum": -330,
        "optimum_x": ShiftedRastrigin().x_shift[:10],
        "dimention": 10,
        "iters": 470,
        "pop_size": 470,
    },
    {  # 19
        "function": ShiftedRotatedRastrigin,
        "bounds": (-5, 5),
        "optimum": -330,
        "optimum_x": ShiftedRotatedRastrigin().x_shift[:10],
        "dimention": 10,
        "iters": 463,
        "pop_size": 463,
    },
    {  # 20
        "function": HybridCompositionFunction1,
        "bounds": (-5, 5),
        "optimum": 120,
        "optimum_x": HybridCompositionFunction1().x_shift[0][:10],
        "dimention": 10,
        "iters": 799,
        "pop_size": 799,
    },
    {  # 21
        "function": Sphere,
        "bounds": (-5.12, 5.12),
        "optimum": 0,
        "optimum_x": np.zeros(shape=30, dtype=np.float64),
        "dimention": 30,
        "iters": 210,
        "pop_size": 210,
    },
    {  # 22
        "function": HighConditionedElliptic,
        "bounds": (-100, 100),
        "optimum": 0,
        "optimum_x": np.zeros(shape=30, dtype=np.float64),
        "dimention": 30,
        "iters": 355,
        "pop_size": 355,
    },
    {  # 23
        "function": Griewank,
        "bounds": (-600, 600),
        "optimum": 0,
        "optimum_x": np.zeros(shape=30, dtype=np.float64),
        "dimention": 30,
        "iters": 600,
        "pop_size": 600,
    },
    {  # 24
        "function": Ackley,
        "bounds": (-32.768, 32.768),
        "optimum": 0,
        "optimum_x": np.zeros(shape=30, dtype=np.float64),
        "dimention": 30,
        "iters": 247,
        "pop_size": 247,
    },
    {  # 25
        "function": Rastrigin,
        "bounds": (-5.12, 5.12),
        "optimum": 0,
        "optimum_x": np.zeros(shape=30, dtype=np.float64),
        "dimention": 30,
        "iters": 799,
        "pop_size": 799,
    },
)
