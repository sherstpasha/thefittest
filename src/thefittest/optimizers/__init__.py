from typing import Union

from ._differentialevolution import DifferentialEvolution
from ._geneticalgorithm import GeneticAlgorithm
from ._geneticprogramming import GeneticProgramming
from ._jade import JADE
from ._jde import jDE
from ._sade2005 import SaDE2005
from ._selfaga import SelfAGA
from ._selfcga import SelfCGA
from ._selfcgp import SelfCGP
from ._shade import SHADE
from ._shaga import SHAGA


_optimizer_tree_type = [GeneticProgramming, SelfCGP]

OptimizerTreeType = Union[GeneticProgramming, SelfCGP]


optimizer_real_coded = [DifferentialEvolution, JADE, SHADE, jDE, SaDE2005]

OptimizerRealCoded = Union[DifferentialEvolution, JADE, SHADE, jDE, SaDE2005]

optimizer_binary_coded = [GeneticAlgorithm, SelfCGA, SHAGA, SelfAGA]

OptimizerBinaryCoded = Union[GeneticAlgorithm, SelfCGA, SHAGA, SelfAGA]

optimizer_string_type = optimizer_real_coded + optimizer_binary_coded

OptimizerStringType = Union[OptimizerRealCoded, OptimizerBinaryCoded]

OptimizerAnyType = Union[OptimizerTreeType, OptimizerStringType]

optimizer_any_type = _optimizer_tree_type + optimizer_string_type

__all__ = [
    "DifferentialEvolution",
    "GeneticAlgorithm",
    "GeneticProgramming",
    "JADE",
    "jDE",
    "SaDE2005",
    "SelfAGA",
    "SelfCGA",
    "SelfCGP",
    "SHADE",
    "SHAGA",
]
