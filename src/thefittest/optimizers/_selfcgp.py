import numpy as np
from typing import Optional
from typing import Callable
from typing import Any
from ._geneticprogramming import GeneticProgramming
from ..tools import scale_data
from ..tools import rank_data
from functools import partial
from ._base import TheFittest
from ._base import Statistics
from ._base import LastBest
from ..tools import numpy_group_by


