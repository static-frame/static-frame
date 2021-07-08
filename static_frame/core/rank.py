import typing as tp

from enum import Enum, unique
import numpy as np

from static_frame.core.util import DEFAULT_STABLE_SORT_KIND, union1d
from static_frame.core.util import DTYPE_INT_DEFAULT
from static_frame.core.util import DTYPE_BOOL

class RankMethod(str, Enum):
    AVERAGE = 'average'
    MIN = 'min'
    MAX = 'max'
    DENSE = 'dense'
    ORDINAL = 'ordinal'

def rank_1d(
        array: np.ndarray,
        method: RankMethod,
        ascending: bool = True,
        start: int = 0,
        ):
    '''
    Rank 1D array. Basedon the the scipy implementation:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html


    average: The average of the ranks that would have been assigned to
    all the tied values is assigned to each value.
    min: The minimum of the ranks that would have been assigned to all
    the tied values is assigned to each value.  (This is also
    referred to as "competition" ranking.)
    max: The maximum of the ranks that would have been assigned to all
    the tied values is assigned to each value.
    dense: Like 'min', but the rank of the next highest element is
    assigned the rank immediately after those assigned to the tied
    elements.
    ordinal: All values are given a distinct rank, corresponding to
    the order that the values occur in `a`.
    '''
    size = len(array)
    index_sorted = np.argsort(array, kind=DEFAULT_STABLE_SORT_KIND)
    if method == RankMethod.ORDINAL:
        # NOTE: not sure if this works for other rank types
        if not ascending:
            index_sorted = (size - 1) - index_sorted
        return index_sorted + start

    array_sorted = array[index_sorted] # order array
    # createa a Boolean array showing unique values, first value is always True
    is_unique = np.full(size, True, dtype=DTYPE_BOOL)
    is_unique[1:] = array_sorted[1:] != array_sorted[:-1]
    # cumsum used on is_unique to only increment when unique; then re-order; this always has 1 as the lowest value
    dense = is_unique.cumsum()[index_sorted]

    if method == RankMethod.DENSE:
        return dense - 1 + start

    # indices where unique is true
    unique_pos = np.nonzero(is_unique)[0]
    size_unique = len(unique_pos)
    # cumulative counts of each unique value, adding length as last value
    count = np.full(size_unique + 1, size)
    count[:size_unique] = unique_pos

    # import ipdb; ipdb.set_trace()
    if method == RankMethod.MAX:
        return count[dense] - 1 + start

    if method == RankMethod.MIN:
        return count[dense - 1] + start

    if method == RankMethod.AVERAGE:
        ranks0 = (.5 * (count[dense] + count[dense - 1] + 1)) - 1
        if not ascending:
            ranks0 = (size - 1) - ranks0
        return ranks0 + start

    raise NotImplementedError(f'no handling for {method}')

    # range_array = np.arange(len(array), dtype=DTYPE_INT_DEFAULT)


