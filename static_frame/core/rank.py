import typing as tp

from enum import Enum
import numpy as np

from static_frame.core.util import DEFAULT_STABLE_SORT_KIND
from static_frame.core.util import DTYPE_INT_DEFAULT
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import DTYPE_FLOAT_DEFAULT
from static_frame.core.util import EMPTY_ARRAY
from static_frame.core.util import EMPTY_ARRAY_INT

class RankMethod(str, Enum):
    MEAN = 'mean'
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
    Rank 1D array. Based on the the scipy implementation:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html

    mean: The mean of the ranks that would have been assigned to
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
    if size == 0:
        return EMPTY_ARRAY if method == RankMethod.MEAN else EMPTY_ARRAY_INT

    index_sorted = np.argsort(array, kind=DEFAULT_STABLE_SORT_KIND)
    ordinal = np.empty(array.size, dtype=DTYPE_INT_DEFAULT)
    ordinal[index_sorted] = np.arange(array.size, dtype=DTYPE_INT_DEFAULT)

    if method == RankMethod.ORDINAL:
        ranks0 = ordinal
        if not ascending:
            ranks0_max = size - 1
    else:
        array_sorted = array[index_sorted] # order array
        # createa a Boolean array showing unique values, first value is always True
        is_unique = np.full(size, True, dtype=DTYPE_BOOL)
        is_unique[1:] = array_sorted[1:] != array_sorted[:-1]

        # cumsum used on is_unique to only increment when unique; then re-order; this always has 1 as the lowest value
        dense = is_unique.cumsum()[ordinal]
        if method == RankMethod.DENSE:
            ranks0 = dense - 1
        else:
            # indices where unique is true
            unique_pos = np.nonzero(is_unique)[0]
            size_unique = len(unique_pos)
            # cumulative counts of each unique value, adding length as last value
            count = np.empty(size_unique + 1, dtype=DTYPE_INT_DEFAULT)
            count[:size_unique] = unique_pos
            count[size_unique] = size

            if ((method == RankMethod.MAX and ascending)
                    or (method == RankMethod.MIN and not ascending)):
                ranks0 = count[dense] - 1
            elif ((method == RankMethod.MIN and ascending)
                    or (method == RankMethod.MAX and not ascending)):
                ranks0 = count[dense - 1]
            elif method == RankMethod.MEAN:
                # take the mean of min and max selections
                ranks0 = 0.5 * ((count[dense] - 1) + count[dense - 1])
            else:
                raise NotImplementedError(f'no handling for {method}')

        if not ascending:
            # determine max after selection and shift
            ranks0_max = ranks0.max()

    if not ascending:
        ranks0 = ranks0_max - ranks0
    if start != 0:
        ranks0 = ranks0 + start

    ranks0.flags.writeable = False
    return ranks0


def rank_2d(
        array: np.ndarray,
        method: tp.Union[str, RankMethod],
        ascending: bool = True,
        start: int = 0,
        axis: int = 0,
        ) -> np.ndarray:
    '''
    Args:
        axis: if 0, columns are ranked, if 1, rows are ranked
    '''
    # scipy uses np.apply_along_axis, but that handles many more cases than needed

    if method == RankMethod.MEAN:
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

    post.flags.writeable = False
    return post




