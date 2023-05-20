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


OptimizerTreeType = Union[GeneticProgramming,
                          SelfCGP]

OptimizerStringType = Union[DifferentialEvolution,
                            JADE,
                            SHADE,
                            jDE,
                            SaDE2005,
                            GeneticAlgorithm,
                            SelfCGA,
                            SHAGA]
