from typing import Any
import numpy as np
from numpy.typing import NDArray
from ._numba_funcs import find_end_subtree_from_i
from ._numba_funcs import find_id_args_from_i
from ._numba_funcs import get_levels_tree_from_i
from ._numba_funcs import find_first_difference_between_two
from ._numba_funcs import binary_search_interval
from ._numba_funcs import check_for_value
from ._numba_funcs import argsort_k
from ._numba_funcs import find_pbest_id
from ._numba_funcs import find_pbest_id


def donothing(x: Any) -> Any:
    return x
