'''
This module us for utilty functions that take as input and / or return Container subclasses such as Index, Series, or Frame, and that need to be shared by multiple such Container classes.
'''

import numpy as np
import typing as tp

if tp.TYPE_CHECKING:
    from static_frame.core.series import Series #pylint: disable=W0611
    from static_frame.core.frame import Frame #pylint: disable=W0611

from static_frame.core.util import IndexConstructor
from static_frame.core.util import IndexInitializer
from static_frame.core.util import STATIC_ATTR

from static_frame.core.index_base import IndexBase


def index_from_optional_constructor(
        value: IndexInitializer,
        *,
        default_constructor: IndexConstructor,
        explicit_constructor: tp.Optional[IndexConstructor] = None,
        ) -> IndexBase:
    '''
    Given a value that is an IndexInitializer (which means it might be an Index), determine if that value is really an Index, and if so, determine if a copy has to be made; otherwise, use the default_constructor. If an explicit_constructor is given, that is always used.
    '''
    if explicit_constructor:
        return explicit_constructor(value)

    # default constructor could be a function with a STATIC attribute
    if isinstance(value, IndexBase) and hasattr(default_constructor, STATIC_ATTR):
        # if default is STATIC, and value is not STATIC, get an immutabel
        if default_constructor.STATIC: # type: ignore
            if not value.STATIC:
                # v: ~S, dc: S, use immutable alternative
                return value._IMMUTABLE_CONSTRUCTOR(value)
            # v: S, dc: S, both immutable
            return value
        else: # default constructor is mutable
            if not value.STATIC:
                # v: ~S, dc: ~S, both are mutable
                return value.copy()
            # v: S, dc: ~S, return a mutable version of something that is not mutable
            return default_constructor(value)

    # cannot always deterine satic status from constructors; fallback on using default constructor
    return default_constructor(value)



# matmul of two series reduces to a single value

def matmul(
        lhs: tp.Union['Series', 'Frame', tp.Iterable],
        rhs: tp.Union['Series', 'Frame', tp.Iterable],
        ) -> tp.Any: #tp.Union['Series', 'Frame']:
    '''
    Implementation of matrix multiplication for Series and Frame
    '''
    from static_frame.core.series import Series
    from static_frame.core.frame import Frame

    # for a @ b = c
    # if a is 2D: a.columns must align b.index
    # if b is 1D, a.columns bust align with b.index
    # if a is 1D: len(a) == b.index (len of b), returns w columns of B

    if not isinstance(rhs, (np.ndarray, Series, Frame)):
        # try to make it into an array
        rhs = np.array(rhs)

    if not isinstance(lhs, (np.ndarray, Series, Frame)):
        # try to make it into an array
        lhs = np.array(lhs)

    if isinstance(lhs, np.ndarray):
        lhs_type = np.ndarray
    elif isinstance(lhs, Series):
        lhs_type = Series
    else: # normalize subclasses
        lhs_type = Frame

    if isinstance(rhs, np.ndarray):
        rhs_type = np.ndarray
    elif isinstance(rhs, Series):
        rhs_type = Series
    else: # normalize subclasses
        rhs_type = Frame

    if rhs_type == np.ndarray and lhs_type == np.ndarray:
        return np.matmul(lhs, rhs)


    own_index = True
    constructor = None

    if lhs.ndim == 1: # Series, 1D array
        # result will be 1D or 0D
        columns = None

        if lhs_type == Series and (rhs_type == Series or rhs_type == Frame):
            aligned = lhs._index.union(rhs._index)
            # if the aligned shape is not the same size as the originals, we do not have the same values in each and cannot proceed (all values go to NaN)
            if len(aligned) != len(lhs._index) or len(aligned) != len(rhs._index):
                raise RuntimeError('shapes not alignable for matrix multiplication')

        if lhs_type == Series:
            if rhs_type == np.ndarray:
                if lhs.shape[0] != rhs.shape[0]: # works for 1D and 2D
                    raise RuntimeError('shapes not alignable for matrix multiplication')
                ndim = rhs.ndim - 1 # if 2D, result is 1D, of 1D, result is 0
                left = lhs.values
                right = rhs # already np
                if ndim == 1:
                    index = None # force auto increment integer
                    own_index = False
                    constructor = lhs.__class__
                # else:
                #     index = lhs.index
            elif rhs_type == Series:
                ndim = 0
                left = lhs.reindex(aligned).values
                right = rhs.reindex(aligned).values
                # index = aligned
            else: # rhs is Frame
                ndim = 1
                left = lhs.reindex(aligned).values
                right = rhs.reindex(index=aligned).values
                index = rhs._columns
                constructor = lhs.__class__
        else: # lhs is 1D array
            left = lhs
            right = rhs.values
            if rhs_type == Series:
                ndim = 0
            else: # rhs is Frame, len(lhs) == len(rhs.index)
                ndim = 1
                index = rhs._columns
                constructor = Series # cannot get from argument

    elif lhs.ndim == 2: # Frame, 2D array

        if lhs_type == Frame and (rhs_type == Series or rhs_type == Frame):
            aligned = lhs._columns.union(rhs._index)
            # if the aligned shape is not the same size as the originals, we do not have the same values in each and cannot proceed (all values go to NaN)
            if len(aligned) != len(lhs._columns) or len(aligned) != len(rhs._index):
                raise RuntimeError('shapes not alignable for matrix multiplication')

        if lhs_type == Frame:
            if rhs_type == np.ndarray:
                if lhs.shape[1] != rhs.shape[0]: # works for 1D and 2D
                    raise RuntimeError('shapes not alignable for matrix multiplication')
                ndim = rhs.ndim
                left = lhs.values
                right = rhs # already np
                index = lhs._index

                if ndim == 1:
                    constructor = Series
                else:
                    constructor = lhs.__class__
                    columns = None # force auto increment index
            elif rhs_type == Series:
                # a.columns must align with b.index
                ndim = 1
                left = lhs.reindex(columns=aligned).values
                right = rhs.reindex(aligned).values
                index = lhs._index  # this axis is not changed
                constructor = rhs.__class__
            else: # rhs is Frame
                # a.columns must align with b.index
                ndim = 2
                left = lhs.reindex(columns=aligned).values
                right = rhs.reindex(index=aligned).values
                index = lhs._index
                columns = rhs._columns
                constructor = lhs.__class__ # give left precedence
        else: # lhs is 2D array
            left = lhs
            right = rhs.values
            if rhs_type == Series: # returns unindexed Series
                ndim = 1
                index = None
                own_index = False
                constructor = rhs.__class__
            else: # rhs is Frame, lhs.shape[1] == rhs.shape[0]
                if lhs.shape[1] != rhs.shape[0]: # works for 1D and 2D
                    raise RuntimeError('shapes not alignable for matrix multiplication')
                ndim = 2
                index = None
                own_index = False
                columns = rhs._columns
                constructor = rhs.__class__

    # import ipdb; ipdb.set_trace()

    # NOTE: np.matmul is not the same as np.dot for some arguments
    data = np.matmul(left, right)
    # import ipdb; ipdb.set_trace()

    if ndim == 0:
        return data

    data.flags.writeable = False
    if ndim == 1:
        return constructor(data,
                index=index,
                own_index=own_index,
                )
    return constructor(data,
            index=index,
            own_index=own_index,
            columns=columns
            )