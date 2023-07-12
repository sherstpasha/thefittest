from ._geneticalgorithm import GeneticAlgorithm
from ._selfcga import SelfCGA
from ._geneticprogramming import GeneticProgramming
from ._selfcgp import SelfCGP
from ._differentialevolution import DifferentialEvolution
from ._sade2005 import SaDE2005
from ._jde import jDE
from ._jade import JADE
from ._shade import SHADE
from ._shaga import SHAGA
from typing import Union


_optimizer_tree_type = [GeneticProgramming,
                        SelfCGP]

OptimizerTreeType = Union[GeneticProgramming,
                          SelfCGP]


optimizer_real_coded = [DifferentialEvolution,
                        JADE,
                        SHADE,
                        jDE,
                        SaDE2005]

OptimizerRealCoded = Union[DifferentialEvolution,
                           JADE,
                           SHADE,
                           jDE,
                           SaDE2005]

optimizer_binary_coded = [GeneticAlgorithm, SelfCGA, SHAGA]

OptimizerBinaryCoded = Union[GeneticAlgorithm, SelfCGA, SHAGA]

optimizer_string_type = optimizer_real_coded + optimizer_binary_coded

OptimizerStringType = Union[OptimizerRealCoded, OptimizerBinaryCoded]

OptimizerAnyType = Union[OptimizerTreeType, OptimizerStringType]

optimizer_any_type = _optimizer_tree_type + optimizer_string_type
