from typing import Any

from ._numba_funcs import argsort_k
from ._numba_funcs import binary_search_interval
from ._numba_funcs import check_for_value
from ._numba_funcs import find_end_subtree_from_i
from ._numba_funcs import find_first_difference_between_two
from ._numba_funcs import find_id_args_from_i
from ._numba_funcs import find_pbest_id
from ._numba_funcs import get_levels_tree_from_i


def donothing(x: Any) -> Any:
    return x


__all__ = [
    "find_end_subtree_from_i",
    "find_id_args_from_i",
    "get_levels_tree_from_i",
    "find_first_difference_between_two",
    "binary_search_interval",
    "check_for_value",
    "argsort_k",
    "find_pbest_id",
]
