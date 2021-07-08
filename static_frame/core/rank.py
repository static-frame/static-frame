import typing as tp

from enum import Enum
import numpy as np

from static_frame.core.util import DEFAULT_STABLE_SORT_KIND
from static_frame.core.util import DTYPE_INT_DEFAULT


class RankMethod(str, Enum):
    AVERAGE = 'average'
    MIN = 'min'
    MAX = 'max'
    DENSE = 'dense'
    ORDINAL = 'ordinal'

# after https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html
def rank_1d(
        array: np.ndarray,
        method: RankMethod,
        ascending: bool = True,
        ):
    index_sorted = np.argsort(array, kind=DEFAULT_STABLE_SORT_KIND)
    if not ascending:
        index_sorted = (len(index_sorted) - 1) - index_sorted
    if method == RankMethod.ORDINAL:
        return index_sorted


    array_sorted = array[index_sorted] # order
    obs = np.r_[True, array_sorted[1:] != array_sorted[:-1]]
    dense = obs.cumsum()[index_sorted]

    if method == RankMethod.DENSE:
        return dense

    # cumulative counts of each unique value
    count = np.r_[np.nonzero(obs)[0], len(obs)]

    if method == RankMethod.MAX:
        return count[dense]

    if method == RankMethod.MIN:
        return count[dense - 1] + 1

    if method == RankMethod.AVERAGE:
        return .5 * (count[dense] + count[dense - 1] + 1)

    raise NotImplementedError(f'no handling for {method}')

    # range_array = np.arange(len(array), dtype=DTYPE_INT_DEFAULT)


