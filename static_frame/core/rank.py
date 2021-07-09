import typing as tp

from enum import Enum
import numpy as np

from static_frame.core.util import DEFAULT_STABLE_SORT_KIND, PositionsAllocator
from static_frame.core.util import DTYPE_INT_DEFAULT
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import DTYPE_FLOAT_DEFAULT

class RankMethod(str, Enum):
    AVERAGE = 'average'
    MIN = 'min'
    MAX = 'max'
    DENSE = 'dense'
    ORDINAL = 'ordinal'


def rank_1d(
        array: np.ndarray,
        method: tp.Union[str, RankMethod],
        ascending: bool = True,
        start: int = 0,
        ) -> np.ndarray:
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
    ranks0_max = size - 1

    index_sorted = np.argsort(array, kind=DEFAULT_STABLE_SORT_KIND)

    ordinal = np.empty(array.size, dtype=DTYPE_INT_DEFAULT)
    ordinal[index_sorted] = np.arange(array.size, dtype=DTYPE_INT_DEFAULT)

    if method == RankMethod.ORDINAL:
        ranks0 = ordinal
    else:
        array_sorted = array[index_sorted] # order array
        # createa a Boolean array showing unique values, first value is always True
        is_unique = np.full(size, True, dtype=DTYPE_BOOL)
        is_unique[1:] = array_sorted[1:] != array_sorted[:-1]

        # is_unique = np.r_[True, array_sorted[1:] != array_sorted[:-1]]
        # cumsum used on is_unique to only increment when unique; then re-order; this always has 1 as the lowest value
        dense = is_unique.cumsum()[ordinal]
        if method == RankMethod.DENSE:
            ranks0 = dense - 1
            ranks0_max = ranks0.max()
        else:
            # indices where unique is true
            unique_pos = np.nonzero(is_unique)[0]
            size_unique = len(unique_pos)
            # cumulative counts of each unique value, adding length as last value
            count = np.full(size_unique + 1, size)
            count[:size_unique] = unique_pos

            if ((method == RankMethod.MAX and ascending)
                    or (method == RankMethod.MIN and not ascending)):
                ranks0 = count[dense] - 1
            elif ((method == RankMethod.MIN and ascending)
                    or (method == RankMethod.MAX and not ascending)):
                ranks0 = count[dense - 1]
            elif method == RankMethod.AVERAGE:
                # take the average of min and max selections
                ranks0 = (.5 * (count[dense] + count[dense - 1] + 1)) - 1
            else:
                raise NotImplementedError(f'no handling for {method}')

    if not ascending:
        ranks0 = ranks0_max - ranks0
    return ranks0 + start


def rank_2d(
        array: np.ndarray,
        method: tp.Union[str, RankMethod],
        ascending: bool = True,
        start: int = 0,
        axis: int = 0,
        ) -> np.ndarray:
    '''
    Args:
        axis: if 0, columns are sorted, if 1, rows are sorted
    '''
    # scipy uses np.apply_along_axis, but that handles many more cases than needed

    if method == RankMethod.AVERAGE:
        dtype = DTYPE_FLOAT_DEFAULT
    else:
        dtype = DTYPE_INT_DEFAULT
    shape = array.shape
    post = np.empty(shape, dtype=dtype)
    if axis == 0: # apply by column
        for i in range(shape[1]):
            post[:, i] = rank_1d(array[:, i], method, ascending, start)
    elif axis == 1: # apply by row
        for i in range(shape[0]):
            post[i] = rank_1d(array[i], method, ascending, start)
    return post




