from collections import abc
from collections import defaultdict
from collections import namedtuple
from enum import Enum
from functools import partial
from functools import reduce
from io import StringIO
from itertools import chain
from itertools import zip_longest
from os import PathLike
from urllib import request
from copy import deepcopy

import contextlib
import datetime
import operator
import os
import tempfile
import typing as tp

from automap import FrozenAutoMap  # pylint: disable = E0611
import numpy as np


if tp.TYPE_CHECKING:
    from static_frame.core.index_base import IndexBase #pylint: disable=W0611 #pragma: no cover
    from static_frame.core.index import Index #pylint: disable=W0611 #pragma: no cover
    from static_frame.core.series import Series #pylint: disable=W0611 #pragma: no cover
    from static_frame.core.frame import Frame #pylint: disable=W0611 #pragma: no cover
    from static_frame.core.frame import FrameAsType #pylint: disable=W0611 #pragma: no cover
    from static_frame.core.type_blocks import TypeBlocks #pylint: disable=W0611 #pragma: no cover

# dtype.kind
#     A character code (one of ‘biufcmMOSUV’) identifying the general kind of data.
#     b 	boolean
#     i 	signed integer
#     u 	unsigned integer
#     f 	floating-point
#     c 	complex floating-point
#     m 	timedelta
#     M 	datetime
#     O 	object
#     S 	(byte-)string
#     U 	Unicode
#     V 	void

# https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.scalars.html


DEFAULT_SORT_KIND = 'mergesort'
DEFAULT_STABLE_SORT_KIND = 'mergesort'

DTYPE_DATETIME_KIND = 'M'
DTYPE_TIMEDELTA_KIND = 'm'
DTYPE_COMPLEX_KIND = 'c'
DTYPE_FLOAT_KIND = 'f'
DTYPE_OBJECT_KIND = 'O'
DTYPE_BOOL_KIND = 'b'

DTYPE_STR_KINDS = ('U', 'S') # S is np.bytes_
DTYPE_INT_KINDS = ('i', 'u') # signed and unsigned
DTYPE_INEXACT_KINDS = (DTYPE_FLOAT_KIND, DTYPE_COMPLEX_KIND) # kinds that support NaN values
DTYPE_NAT_KINDS = (DTYPE_DATETIME_KIND, DTYPE_TIMEDELTA_KIND)

# all kinds that can have NaN, NaT, or None
DTYPE_NA_KINDS = frozenset((
        DTYPE_FLOAT_KIND,
        DTYPE_COMPLEX_KIND,
        DTYPE_DATETIME_KIND,
        DTYPE_TIMEDELTA_KIND,
        DTYPE_OBJECT_KIND,
        ))

# all kinds that can use tolist() to go to a compatible Python type
DTYPE_OBJECTABLE_KINDS = frozenset((
        DTYPE_FLOAT_KIND,
        DTYPE_COMPLEX_KIND,
        DTYPE_OBJECT_KIND,
        DTYPE_BOOL_KIND,
        'U', 'S', # str kinds
        'i', 'u' # int kinds
        ))

DTYPE_OBJECT = np.dtype(object)
DTYPE_BOOL = np.dtype(bool)
DTYPE_STR = np.dtype(str)
DTYPE_INT_DEFAULT = np.dtype(np.int64)
DTYPE_FLOAT_DEFAULT = np.dtype(np.float64)
DTYPE_COMPLEX_DEFAULT = np.dtype(np.complex128)

DTYPES_BOOL = (DTYPE_BOOL,)
DTYPES_INEXACT = (DTYPE_FLOAT_DEFAULT, DTYPE_COMPLEX_DEFAULT)

NULL_SLICE = slice(None) # gathers everything
UNIT_SLICE = slice(0, 1)
EMPTY_SLICE = slice(0, 0) # gathers nothing

SLICE_START_ATTR = 'start'
SLICE_STOP_ATTR = 'stop'
SLICE_STEP_ATTR = 'step'
SLICE_ATTRS = (SLICE_START_ATTR, SLICE_STOP_ATTR, SLICE_STEP_ATTR)

STATIC_ATTR = 'STATIC'

ELEMENT_TUPLE = (None,)

EMPTY_TUPLE = ()
EMPTY_SET: tp.FrozenSet[tp.Any] = frozenset()

# defaults to float64
EMPTY_ARRAY = np.array(EMPTY_TUPLE, dtype=None)
EMPTY_ARRAY.flags.writeable = False

EMPTY_ARRAY_BOOL = np.array(EMPTY_TUPLE, dtype=DTYPE_BOOL)
EMPTY_ARRAY_BOOL.flags.writeable = False

EMPTY_ARRAY_INT = np.array(EMPTY_TUPLE, dtype=DTYPE_INT_DEFAULT)
EMPTY_ARRAY_INT.flags.writeable = False

NAT = np.datetime64('nat')
NAT_STR = 'NaT'

# define missing for timedelta as an untyped 0
EMPTY_TIMEDELTA = np.timedelta64(0)

# _DICT_STABLE = sys.version_info >= (3, 6)

# map from datetime.timedelta attrs to np.timedelta64 codes
TIME_DELTA_ATTR_MAP = (
        ('days', 'D'),
        ('seconds', 's'),
        ('microseconds', 'us')
        )

# ufunc functions that will not work with DTYPE_STR_KINDS, but do work if converted to object arrays
UFUNC_AXIS_STR_TO_OBJ = {np.min, np.max, np.sum}

#-------------------------------------------------------------------------------
# utility type groups

INT_TYPES = (int, np.integer) # np.integer catches all np int
FLOAT_TYPES = (float, np.floating) # np.floating catches all np float
COMPLEX_TYPES = (complex, np.complexfloating) # np.complexfloating catches all np complex
INEXACT_TYPES = (float, complex, np.inexact) # inexact matches floating, complexfloating
InexactTypes = tp.Union[float, complex, np.inexact]
NUMERIC_TYPES = (int, float, complex, np.number)

BOOL_TYPES = (bool, np.bool_)
DICTLIKE_TYPES = (abc.Set, dict, FrozenAutoMap)


# iterables that cannot be used in NP array constructors; asumes that dictlike types have already been identified
INVALID_ITERABLE_FOR_ARRAY = (abc.ValuesView, abc.KeysView)
NON_STR_TYPES = {int, float, bool}

# integers above this value will occassionally, once coerced to a float (64 or 128) in an NP array, will not match a hash lookup as a key in a dictionary; an NP array of int or object will work
INT_MAX_COERCIBLE_TO_FLOAT = 1_000_000_000_000_000

# for getitem / loc selection
KEY_ITERABLE_TYPES = (list, np.ndarray)
KeyIterableTypes = tp.Union[tp.Iterable[tp.Any], np.ndarray]

# types of keys that return multiple items, even if the selection reduces to 1
KEY_MULTIPLE_TYPES = (slice, list, np.ndarray)

# for type hinting
# keys once dimension has been isolated
GetItemKeyType = tp.Union[
        int,
        np.integer,
        slice,
        tp.List[tp.Any],
        None,
        'Index',
        'Series',
        np.ndarray
        ]

# keys that might include a multiple dimensions speciation; tuple is used to identify compound extraction
GetItemKeyTypeCompound = tp.Union[
        tp.Tuple[tp.Any, ...],
        int,
        np.integer,
        slice,
        tp.List[tp.Any],
        None,
        'Index',
        'Series',
        np.ndarray]

KeyTransformType = tp.Optional[tp.Callable[[GetItemKeyType], GetItemKeyType]]
NameType = tp.Optional[tp.Hashable]
TupleConstructorType = tp.Callable[[tp.Iterator[tp.Any]], tp.Tuple[tp.Any, ...]]

Bloc2DKeyType = tp.Union['Frame', np.ndarray]
# Bloc1DKeyType = tp.Union['Series', np.ndarray]

UFunc = tp.Callable[..., np.ndarray]
AnyCallable = tp.Callable[..., tp.Any]

Mapping = tp.Union[tp.Mapping[tp.Hashable, tp.Any], 'Series']
CallableOrMapping = tp.Union[AnyCallable, tp.Mapping[tp.Hashable, tp.Any], 'Series']


def is_mapping(value: tp.Any) -> bool:
    from static_frame import Series
    return isinstance(value, (dict, Series))

def is_callable_or_mapping(value: CallableOrMapping) -> bool:
    from static_frame import Series
    return callable(value) or isinstance(value, dict) or isinstance(value, Series)

CallableOrCallableMap = tp.Union[AnyCallable, tp.Mapping[tp.Hashable, AnyCallable]]

# for explivitl selection hashables, or things that will be converted to lists of hashables (explicitly lists)
KeyOrKeys = tp.Union[tp.Hashable, tp.Iterable[tp.Hashable]]

PathSpecifier = tp.Union[str, PathLike]
PathSpecifierOrFileLike = tp.Union[str, PathLike, tp.TextIO]
PathSpecifierOrFileLikeOrIterator = tp.Union[str, PathLike, tp.TextIO, tp.Iterator[str]]


DtypeSpecifier = tp.Optional[tp.Union[str, np.dtype, type]]

DTYPE_SPECIFIER_TYPES = (str, np.dtype, type)

def is_dtype_specifier(value: tp.Any) -> bool:
    return isinstance(value, DTYPE_SPECIFIER_TYPES)

# support an iterable of specifiers, or mapping based on column names
DtypesSpecifier = tp.Optional[tp.Union[
        DtypeSpecifier,
        tp.Iterable[DtypeSpecifier],
        tp.Dict[tp.Hashable, DtypeSpecifier]
        ]]

# specifiers that are equivalent to object
DTYPE_SPECIFIERS_OBJECT = {DTYPE_OBJECT, object, tuple}

DepthLevelSpecifier = tp.Union[int, tp.Iterable[int]]

CallableToIterType = tp.Callable[[], tp.Iterable[tp.Any]]

IndexSpecifier = tp.Union[int, tp.Hashable] # specify a postiion in an index
IndexInitializer = tp.Union[
        'IndexBase',
        tp.Iterable[tp.Hashable],
        tp.Iterable[tp.Sequence[tp.Hashable]], # only for IndexHierarhcy
        ]
IndexConstructor = tp.Callable[..., 'IndexBase']

IndexConstructors = tp.Sequence[IndexConstructor]


# take integers for size; otherwise, extract size from any other index initializer

SeriesInitializer = tp.Union[
        tp.Iterable[tp.Any],
        np.ndarray,
        tp.Mapping[tp.Hashable, tp.Any],
        int, float, str, bool]

# support single items, or numpy arrays, or values that can be made into a 2D array
FRAME_INITIALIZER_DEFAULT = object()

FrameInitializer = tp.Union[
        tp.Iterable[tp.Iterable[tp.Any]],
        np.ndarray,
        ] # need to add FRAME_INITIALIZER_DEFAULT

DateInitializer = tp.Union[str, datetime.date, np.datetime64]
YearMonthInitializer = tp.Union[str, datetime.date, np.datetime64]
YearInitializer = tp.Union[str, datetime.date, np.datetime64]

#-------------------------------------------------------------------------------
FILL_VALUE_DEFAULT = object()
NAME_DEFAULT = object()
STORE_LABEL_DEFAULT = object()


#-------------------------------------------------------------------------------
# join utils

class Join(Enum):
    INNER = 0
    LEFT = 1
    RIGHT = 2
    OUTER = 3

class Pair(tuple): #type: ignore
    pass

class PairLeft(Pair):
    pass

class PairRight(Pair):
    pass

#-------------------------------------------------------------------------------

def mloc(array: np.ndarray) -> int:
    '''Return the memory location of an array.
    '''
    return tp.cast(int, array.__array_interface__['data'][0])


def immutable_filter(src_array: np.ndarray) -> np.ndarray:
    '''Pass an immutable array; otherwise, return an immutable copy of the provided array.
    '''
    if src_array.flags.writeable:
        dst_array = src_array.copy()
        dst_array.flags.writeable = False
        return dst_array
    return src_array # keep it as is

def name_filter(name: NameType) -> NameType:
    '''
    For name attributes on containers, only permit recursively hashable objects.
    '''
    try:
        hash(name)
    except TypeError:
        raise TypeError('unhashable name attribute', name)
    return name

def shape_filter(array: np.ndarray) -> tp.Tuple[int, int]:
    '''Represent a 1D array as a 2D array with length as rows of a single-column array.

    Return:
        row, column count for a block of ndim 1 or ndim 2.
    '''
    if array.ndim == 1:
        return array.shape[0], 1
    return array.shape #type: ignore

def column_2d_filter(array: np.ndarray) -> np.ndarray:
    '''Reshape a flat ndim 1 array into a 2D array with one columns and rows of length. This is used (a) for getting string representations and (b) for using np.concatenate and np binary operators on 1D arrays.
    '''
    # it is not clear when reshape is a copy or a view
    if array.ndim == 1:
        return np.reshape(array, (array.shape[0], 1))
    return array

def column_1d_filter(array: np.ndarray) -> np.ndarray:
    '''
    Ensure that a column that might be 2D or 1D is returned as a 1D array.
    '''
    if array.ndim == 2:
        # could assert that array.shape[1] == 1, but this will raise if does not fit
        return np.reshape(array, array.shape[0])
    return array

def row_1d_filter(array: np.ndarray) -> np.ndarray:
    '''
    Ensure that a row that might be 2D or 1D is returned as a 1D array.
    '''
    if array.ndim == 2:
        # could assert that array.shape[0] == 1, but this will raise if does not fit
        return np.reshape(array, array.shape[1])
    return array

def duplicate_filter(values: tp.Iterable[tp.Any]) -> tp.Iterator[tp.Any]:
    '''
    Assuming ordered values, yield one of each unique value as determined by __eq__ comparison.
    '''
    v_iter = iter(values)
    try:
        v = next(v_iter)
    except StopIteration:
        return
    yield v
    last = v
    for v in v_iter:
        if v != last:
            yield v
        last = v

def _gen_skip_middle(
        forward_iter: CallableToIterType,
        forward_count: int,
        reverse_iter: CallableToIterType,
        reverse_count: int,
        center_sentinel: tp.Any) -> tp.Iterator[tp.Any]:
    '''
    Provide a generator to yield the count values from each side.
    '''
    assert forward_count > 0 and reverse_count > 0
    # for the forward gen, we take one more column to serve as the center column ellipsis; thus, we start enumeration at 0
    for idx, value in enumerate(forward_iter(), start=1):
        yield value
        if idx == forward_count:
            break
    # center sentinel
    yield center_sentinel

    values = []
    for idx, col in enumerate(reverse_iter(), start=1):
        values.append(col)
        if idx == reverse_count:
            break
    yield from reversed(values)


def dtype_from_element(value: tp.Optional[tp.Hashable]) -> np.dtype:
    '''Given an arbitrary hashable to be treated as an element, return the appropriate dtype. This was created to avoid using np.array(value).dtype, which for a Tuple does not return object.
    '''
    if value is np.nan:
        # NOTE: this will not catch all NaN instances, but will catch any default NaNs in function signatures that reference the same NaN object found on the NP root namespace
        return DTYPE_FLOAT_DEFAULT
    if value is None:
        return DTYPE_OBJECT
    if isinstance(value, tuple):
        return DTYPE_OBJECT
    if hasattr(value, 'dtype'):
        return value.dtype #type: ignore
    # NOTE: calling array and getting dtype on np.nan is faster than combining isinstance, isnan calls
    return np.array(value).dtype

def resolve_dtype(dt1: np.dtype, dt2: np.dtype) -> np.dtype:
    '''
    Given two dtypes, return a compatible dtype that can hold both contents without truncation.
    '''
    # NOTE: this is not taking into account endianness; it is not clear if this is important
    # NOTE: np.dtype(object) == np.object_, so we can return np.object_

    # if the same, return that dtype
    if dt1 == dt2:
        return dt1

    # if either is object, we go to object
    if dt1.kind == 'O' or dt2.kind == 'O':
        return DTYPE_OBJECT

    dt1_is_str = dt1.kind in DTYPE_STR_KINDS
    dt2_is_str = dt2.kind in DTYPE_STR_KINDS
    if dt1_is_str and dt2_is_str:
        # if both are string or string-like, we can use result type to get the longest string
        return np.result_type(dt1, dt2)

    dt1_is_dt = dt1.kind == DTYPE_DATETIME_KIND
    dt2_is_dt = dt2.kind == DTYPE_DATETIME_KIND
    if dt1_is_dt and dt2_is_dt:
        # if both are datetime, result type will work
        return np.result_type(dt1, dt2)

    dt1_is_tdelta = dt1.kind == DTYPE_TIMEDELTA_KIND
    dt2_is_tdelta = dt2.kind == DTYPE_TIMEDELTA_KIND
    if dt1_is_tdelta and dt2_is_tdelta:
        # this may or may not work
        # TypeError: Cannot get a common metadata divisor for NumPy datetime metadata [D] and [Y] because they have incompatible nonlinear base time units
        try:
            return np.result_type(dt1, dt2)
        except TypeError:
            return DTYPE_OBJECT

    dt1_is_bool = dt1.type is np.bool_
    dt2_is_bool = dt2.type is np.bool_

    # if any one is a string or a bool, we have to go to object; we handle both cases being the same above; result_type gives a string in mixed cases
    if (dt1_is_str or dt2_is_str
            or dt1_is_bool or dt2_is_bool
            or dt1_is_dt or dt2_is_dt
            or dt1_is_tdelta or dt2_is_tdelta
            ):
        return DTYPE_OBJECT

    # if not a string or an object, can use result type
    return np.result_type(dt1, dt2)

def resolve_dtype_iter(dtypes: tp.Iterable[np.dtype]) -> np.dtype:
    '''Given an iterable of one or more dtypes, do pairwise comparisons to determine compatible overall type. Once we get to object we can stop checking and return object.

    Args:
        dtypes: iterable of one or more dtypes.
    '''
    dtypes = iter(dtypes)
    dt_resolve = next(dtypes)

    for dt in dtypes:
        dt_resolve = resolve_dtype(dt_resolve, dt)
        if dt_resolve == DTYPE_OBJECT:
            return dt_resolve
    return dt_resolve



def concat_resolved(
        arrays: tp.Iterable[np.ndarray],
        axis: int = 0) -> np.ndarray:
    '''
    Concatenation of 2D arrays that uses resolved dtypes to avoid truncation.

    Axis 0 stacks rows (extends columns); axis 1 stacks columns (extends rows).

    No shape manipulation will happen, so it is always assumed that all dimensionalities will be common.
    '''
    #all the input array dimensions except for the concatenation axis must match exactly
    if axis is None:
        raise NotImplementedError('no handling of concatenating flattened arrays')

    # first pass to determine shape and resolved type
    arrays_iter = iter(arrays)
    first = next(arrays_iter)

    # ndim = first.ndim
    dt_resolve = first.dtype
    shape = list(first.shape)

    for array in arrays_iter:
        if dt_resolve != DTYPE_OBJECT:
            dt_resolve = resolve_dtype(array.dtype, dt_resolve)
        shape[axis] += array.shape[axis]

    out = np.empty(shape=shape, dtype=dt_resolve)
    np.concatenate(arrays, out=out, axis=axis)
    out.flags.writeable = False
    return out


def full_for_fill(
        dtype: tp.Optional[np.dtype],
        shape: tp.Union[int, tp.Tuple[int, ...]],
        fill_value: object,
        ) -> np.ndarray:
    '''
    Return a "full" NP array for the given fill_value
    Args:
        dtype: target dtype, which may or may not be possible given the fill_value. This can be set to None to only use the fill_value to determine dtype.
    '''
    dtype_element = dtype_from_element(fill_value)
    if dtype is not None:
        dtype_final = resolve_dtype(dtype, dtype_element)
    else:
        dtype_final = dtype_element
    # NOTE: we do not make this array immutable as we sometimes need to mutate it before adding it to TypeBlocks
    if dtype_final != DTYPE_OBJECT:
        return np.full(shape, fill_value, dtype=dtype_final)

    # for tuples and other objects, better to create and fill
    array = np.empty(shape, dtype=dtype_final)
    if fill_value is None:
        return array # None is already set for empty object arrays

    for iloc in np.ndindex(shape):
        array[iloc] = fill_value
    return array


def dtype_to_fill_value(dtype: DtypeSpecifier) -> tp.Any:
    '''Given a dtype, return an appropriate and compatible null value. This used to provide temporary, "dummy" fill values that reduce type coercions.
    '''
    if not isinstance(dtype, np.dtype):
        # we permit things like object, float, etc.
        dtype = np.dtype(dtype)

    kind = dtype.kind

    if kind in DTYPE_INT_KINDS:
        return 0 # cannot support NaN
    if kind == DTYPE_BOOL_KIND:
        return False
    if kind in DTYPE_INEXACT_KINDS:
        return np.nan
    if kind == DTYPE_OBJECT_KIND:
        return None
    if kind in DTYPE_STR_KINDS:
        return ''
    if kind in DTYPE_DATETIME_KIND:
        return NAT
    if kind in DTYPE_TIMEDELTA_KIND:
        return EMPTY_TIMEDELTA
    raise NotImplementedError('no support for this dtype', kind)

def dtype_kind_to_na(kind: str) -> tp.Any:
    '''Given a dtype kind, return an a NA value to do the least invasive type coercion.
    '''
    if kind in DTYPE_INEXACT_KINDS:
        return np.nan
    if kind in DTYPE_INT_KINDS:
        # allow integers to go to float rather than object
        return np.nan
    if kind in DTYPE_NAT_KINDS:
        return NAT
    return None

def ufunc_axis_skipna(
        array: np.ndarray,
        *,
        skipna: bool,
        axis: int,
        ufunc: UFunc,
        ufunc_skipna: UFunc,
        out: tp.Optional[np.ndarray]=None
        ) -> np.ndarray:
    '''For ufunc array application, when two ufunc versions are available. Expected to always reduce dimensionality.
    '''

    if array.dtype.kind == 'O':
        # replace None with nan
        if skipna:
            is_not_none = np.not_equal(array, None)

        if array.ndim == 1:
            if skipna:
                v = array[is_not_none]
                if len(v) == 0: # all values were None
                    return np.nan
            else:
                v = array
        else:
            # for 2D array, replace None with NaN
            if skipna:
                v = array.copy() # already an object type
                v[~is_not_none] = np.nan
            else:
                v = array

    elif array.dtype.kind == 'M' or array.dtype.kind == 'm':
        # dates do not support skipna functions
        return ufunc(array, axis=axis, out=out)

    elif array.dtype.kind in DTYPE_STR_KINDS and ufunc in UFUNC_AXIS_STR_TO_OBJ:
        v = array.astype(object)
    else:
        v = array

    if skipna:
        return ufunc_skipna(v, axis=axis, out=out)
    return ufunc(v, axis=axis, out=out)


def ufunc_unique(
        array: np.ndarray,
        *,
        axis: tp.Optional[int] = None,
        non_array_type: type = frozenset,
        ) -> np.ndarray:
    '''
    Extended functionality of the np.unique ufunc, to handle cases of mixed typed objects, where NP will fail in finding unique values for a hetergenous object type.

    Args:
        non_array_type: for cases where unique will not work, determine type to return. This can be frozenset or a
    '''
    if array.dtype.kind == 'O':
        if axis is None or array.ndim < 2:
            try:
                return np.unique(array)
            except TypeError: # if unorderable types
                # np.unique will give TypeError: The axis argument to unique is not supported for dtype object
                pass
            # this may or may not work, depending on contained types
            if array.ndim > 1: # axis is None, need to flatten
                array_iter = array.flat
            else:
                array_iter = array
        else:
            # ndim == 2 and axis is not None
            if axis == 0:
                array_iter = array2d_to_tuples(array)
            else:
                array_iter = array2d_to_tuples(array.T)

        # Use a dict to retain order; this will break for non hashables
        store = dict.fromkeys(array_iter)
        array = np.empty(len(store), dtype=object)
        array[:] = tuple(store)
        return array

    # all other types, use the main ufunc
    return np.unique(array, axis=axis)


def roll_1d(array: np.ndarray,
            shift: int
            ) -> np.ndarray:
    '''
    Specialized form of np.roll that, by focusing on the 1D solution, is at least four times faster.
    '''
    size = len(array)
    if size <= 1:
        return array.copy()

    # result will be positive
    shift = shift % size
    if shift == 0:
        return array.copy()

    post = np.empty(size, dtype=array.dtype)

    post[0:shift] = array[-shift:]
    post[shift:] = array[0:-shift]
    return post


def roll_2d(array: np.ndarray,
            shift: int,
            axis: int
            ) -> np.ndarray:
    '''
    Specialized form of np.roll that, by focusing on the 2D solution
    '''
    post = np.empty(array.shape, dtype=array.dtype)

    if axis == 0: # roll rows
        size = array.shape[0]
        if size <= 1:
            return array.copy()

        # result will be positive
        shift = shift % size
        if shift == 0:
            return array.copy()

        post[0:shift, :] = array[-shift:, :]
        post[shift:, :] = array[0:-shift, :]
        return post

    elif axis == 1: # roll columns
        size = array.shape[1]
        if size <= 1:
            return array.copy()

        # result will be positive
        shift = shift % size
        if shift == 0:
            return array.copy()

        post[:, 0:shift] = array[:, -shift:]
        post[:, shift:] = array[:, 0:-shift]
        return post

    raise NotImplementedError()

#-------------------------------------------------------------------------------

def _argminmax_1d(
        array: np.ndarray,
        ufunc: UFunc,
        ufunc_skipna: UFunc,
        skipna: bool = True,
        ) -> tp.Any: # tp.Union[int, float]:
    '''
    Perform argmin or argmax, handling NaN as needed.
    '''
    # always need to to check for nans, even if skipna is False, as np will raise if all NaN, and will not return Nan if there skipna is false
    isna = isna_array(array)

    if isna.all():
        return np.nan

    if isna.any():
        if not skipna:
            return np.nan
        # always use skipna ufunc if any NaNs are present, as otherwise the wrong indices are returned when a nan is encountered (rather than a nan)
        return ufunc_skipna(array)

    return ufunc(array)


argmin_1d = partial(_argminmax_1d, ufunc=np.argmin, ufunc_skipna=np.nanargmin)
argmax_1d = partial(_argminmax_1d, ufunc=np.argmax, ufunc_skipna=np.nanargmax)


def _argminmax_2d(
        array: np.ndarray,
        ufunc: UFunc,
        ufunc_skipna: UFunc,
        skipna: bool = True,
        axis: int = 0
        ) -> np.ndarray: # int or float array
    '''
    Perform argmin or argmax, handling NaN as needed.
    '''
    # always need to to check for nans, even if skipna is False, as np will raise if all NaN, and will not return Nan if there skipna is false
    isna = isna_array(array)

    isna_axis = isna.any(axis=axis)
    if isna_axis.all(): # nan in every axis remaining position
        if not skipna:
            return np.full(isna_axis.shape, np.nan, dtype=DTYPE_FLOAT_DEFAULT)

    if isna_axis.any():
        # always use skipna ufunc if any NaNs are present, as otherwise the wrong indices are returned when a nan is encountered (rather than a nan)
        post = ufunc_skipna(array, axis=axis)
        if not skipna:
            post = post.astype(DTYPE_FLOAT_DEFAULT  )
            post[isna_axis] = np.nan
        return post

    return ufunc(array, axis=axis)

argmin_2d = partial(_argminmax_2d, ufunc=np.argmin, ufunc_skipna=np.nanargmin)
argmax_2d = partial(_argminmax_2d, ufunc=np.argmax, ufunc_skipna=np.nanargmax)

#-------------------------------------------------------------------------------
# array constructors


def is_gen_copy_values(values: tp.Iterable[tp.Any]) -> tp.Tuple[bool, bool]:
    '''
    Returns:
        copy_values: True if values cannot be used in an np.array constructor.
    '''
    is_gen = not hasattr(values, '__len__')
    copy_values = is_gen
    if not is_gen:
        is_dictlike = isinstance(values, DICTLIKE_TYPES)
        copy_values |= is_dictlike
        if not is_dictlike:
            is_iifa = isinstance(values, INVALID_ITERABLE_FOR_ARRAY)
            copy_values |= is_iifa
    return is_gen, copy_values


def prepare_iter_for_array(
        values: tp.Iterable[tp.Any],
        restrict_copy: bool = False
        ) -> tp.Tuple[DtypeSpecifier, bool, tp.Sequence[tp.Any]]:
    '''
    Determine an appropriate DtypeSpecifier for values in an iterable. This does not try to determine the actual dtype, but instead, if the DtypeSpecifier needs to be object rather than None (which lets NumPy auto detect). This is expected to only operate on 1D data.

    Args:
        values: can be a generator that will be exhausted in processing; if a generator, a copy will be made and returned as values.
        restrict_copy: if True, reject making a copy, even if a generator is given
    Returns:
        resolved, has_tuple, values
    '''

    is_gen, copy_values = is_gen_copy_values(values)

    if not is_gen and len(values) == 0: #type: ignore
        return None, False, values #type: ignore

    if restrict_copy:
        copy_values = False

    v_iter = values if is_gen else iter(values)

    if copy_values:
        values_post = []

    resolved = None # None is valid specifier if the type is not ambiguous

    has_tuple = False
    has_str = False
    has_non_str = False
    has_inexact = False
    has_big_int = False

    for v in v_iter:
        if copy_values:
            # if a generator, have to make a copy while iterating
            values_post.append(v)

        value_type = v.__class__

        if (value_type is str
                or value_type is np.str_
                or value_type is bytes
                or value_type is np.bytes_):
            # must compare to both string types
            has_str = True
        elif hasattr(v, '__len__'):
            # identify SF types, lists, or tuples
            has_tuple = True
            resolved = object
            break
        elif isinstance(v, Enum):
            # must check isinstance, as Enum types are always derived from Enum
            resolved = object
            break
        else:
            has_non_str = True
            if value_type in INEXACT_TYPES:
                has_inexact = True
            elif value_type is int and abs(v) > INT_MAX_COERCIBLE_TO_FLOAT:
                has_big_int = True

        if (has_str and has_non_str) or (has_big_int and has_inexact):
            resolved = object
            break

    if copy_values:
        # v_iter is an iter, we need to finish it
        values_post.extend(v_iter)
        return resolved, has_tuple, values_post
    return resolved, has_tuple, values #type: ignore


def iterable_to_array_1d(
        values: tp.Iterable[tp.Any],
        dtype: DtypeSpecifier = None
        ) -> tp.Tuple[np.ndarray, bool]:
    '''
    Convert an arbitrary Python iterable to a 1D NumPy array without any undesirable type coercion.

    Returns:
        pair of array, Boolean, where the Boolean can be used when necessary to establish uniqueness.
    '''
    if values.__class__ is np.ndarray:
        if values.ndim != 1: #type: ignore
            raise RuntimeError('expected 1d array')
        if dtype is not None and dtype != values.dtype: #type: ignore
            raise RuntimeError(f'Supplied dtype {dtype} not set on supplied array.')
        return values, len(values) <= 1 #type: ignore

    if isinstance(values, range):
        # translate range to np.arange to avoid iteration
        array = np.arange(start=values.start,
                stop=values.stop,
                step=values.step,
                dtype=dtype)
        array.flags.writeable = False
        return array, True

    if hasattr(values, 'values') and hasattr(values, 'index'):
        raise RuntimeError(f'Supplied iterable {type(values)} appears to be labeled, though labels are ignored in this context. Convert to an array.')

    values_for_construct: tp.Sequence[tp.Any]

    # values for construct will only be a copy when necessary in iteration to find type
    if isinstance(values, str):
        # if a we get a single string, it is an iterable of characters, but if given to NumPy will produce an array of a single element; here, we convert it to an iterable of a single element; let dtype argument (if passed or None) be used in creation, below
        has_tuple = False
        values_for_construct = (values,)
    elif dtype is None:
        # this gives as dtype only None, or object, letting array constructor do the rest
        dtype, has_tuple, values_for_construct = prepare_iter_for_array(values)
        if len(values_for_construct) == 0:
            return EMPTY_ARRAY, True # no dtype given, so return empty float array
    else:
        is_gen, copy_values = is_gen_copy_values(values)
        if copy_values:
            # we have to realize into sequence for numpy creation
            values_for_construct = tuple(values)
        else:
            values_for_construct = values #type: ignore

        if len(values_for_construct) == 0:
            # dtype was given, return an empty array with that dtype
            v = np.empty(0, dtype=dtype)
            v.flags.writeable = False
            return v, True
        #as we have not iterated iterable, assume that there might be tuples if the dtype is object
        has_tuple = dtype in DTYPE_SPECIFIERS_OBJECT

    if len(values_for_construct) == 1 or isinstance(values, DICTLIKE_TYPES):
        # check values for dictlike, not values_for_construct
        is_unique = True
    else:
        is_unique = False

    # construction
    if has_tuple:
        # this matches cases where dtype is given and dtype is an object specifier
        # this is the only way to assign from a sequence that contains a tuple; this does not work for dict or set (they must be copied into an iterable), and is little slower than creating array directly
        v = np.empty(len(values_for_construct), dtype=DTYPE_OBJECT)
        v[NULL_SLICE] = values_for_construct
    elif dtype == int:
        # large python ints can overflow default NumPy int type
        try:
            v = np.array(values_for_construct, dtype=dtype)
        except OverflowError:
            v = np.array(values_for_construct, dtype=DTYPE_OBJECT)
    else:
        # if dtype was None, we might have discovered this was object but has no tuples; faster to do this constructor instead of null slice assignment
        v = np.array(values_for_construct, dtype=dtype)

    v.flags.writeable = False
    return v, is_unique


def iterable_to_array_2d(
        values: tp.Iterable[tp.Iterable[tp.Any]],
        ) -> np.ndarray:
    '''
    Convert an arbitrary Python iterable of iterables to a 2D NumPy array without any undesirable type coercion. Useful IndexHierarchy construction.

    Returns:
        pair of array, Boolean, where the Boolean can be used when necessary to establish uniqueness.
    '''
    if values.__class__ is np.ndarray:
        if values.ndim != 2: #type: ignore
            raise RuntimeError('expected 2d array')
        return values

    if hasattr(values, 'values') and hasattr(values, 'index'):
        raise RuntimeError(f'Supplied iterable {type(values)} appears to be labeled, though labels are ignored in this context. Convert to an array.')

    # consume values into a tuple
    if not hasattr(values, '__len__'):
        values = tuple(values)

    # if we provide whole generator to prepare_iter_for_array, it will copy the entire sequence unless restrict copy is True
    dtype, _, _ = prepare_iter_for_array(
            (y for z in values for y in z),
            restrict_copy=True
            )

    array = np.array(values, dtype=dtype)
    if array.ndim != 2:
        raise RuntimeError('failed to convert iterable to 2d array')

    array.flags.writeable = False
    return array

def iterable_to_array_nd(
        values: tp.Any,
        ) -> np.ndarray:
    '''
    Attempt to determine if a value is 0, 1, or 2D array; this will interpret lists of tuples as 2D, as NumPy does.
    '''
    if hasattr(values, '__iter__') and not isinstance(values, str):

        values = iter(values)
        try:
            first = next(values)
        except StopIteration:
            return EMPTY_ARRAY

        if hasattr(first, '__iter__') and not isinstance(first, str):
            return iterable_to_array_2d(chain((first,), values))

        array, _ = iterable_to_array_1d(chain((first,), values))
        return array
    # its an element
    return np.array(values)

#-------------------------------------------------------------------------------

def slice_to_ascending_slice(
        key: slice,
        size: int
        ) -> slice:
    '''
    Given a slice, return a slice that, with ascending integers, covers the same values.

    Args:
        size: the length of the container on this axis
    '''
    # NOTE: a slice can have start > stop, and None as step: should that case be handled here?

    if key.step is None or key.step > 0:
        return key

    stop = key.start if key.start is None else key.start + 1

    if key.step == -1:
        # if 6, 1, -1, then
        start = key.stop if key.stop is None else key.stop + 1
        return slice(start, stop, 1)

    step = abs(key.step)
    start = size - 1 if key.start is None else min(size - 1, key.start)

    if key.stop is None:
        start = start - (step * (start // step))
    else:
        start = start - (step * ((start - key.stop - 1) // step))

    return slice(start, stop, step)

def slice_to_inclusive_slice(
        key: slice,
        offset: int = 0,
        ) -> slice:
    '''Make a stop exclusive key inclusive by adding one to the stop value.
    '''
    start = None if key.start is None else key.start + offset
    stop = None if key.stop is None else key.stop + 1 + offset
    return slice(start, stop, key.step)



#-------------------------------------------------------------------------------
# dates

DT64_YEAR = np.dtype('datetime64[Y]')
DT64_MONTH = np.dtype('datetime64[M]')
DT64_DAY = np.dtype('datetime64[D]')
DT64_H = np.dtype('datetime64[h]')
DT64_M = np.dtype('datetime64[m]')
DT64_S = np.dtype('datetime64[s]')
DT64_MS = np.dtype('datetime64[ms]')
DT64_US = np.dtype('datetime64[us]')
DT64_NS = np.dtype('datetime64[ns]')
DT64_PS = np.dtype('datetime64[ps]')
DT64_FS = np.dtype('datetime64[fs]')
DT64_AS = np.dtype('datetime64[as]')

TD64_YEAR = np.timedelta64(1, 'Y')
TD64_MONTH = np.timedelta64(1, 'M')
TD64_DAY = np.timedelta64(1, 'D')
TD64_H = np.timedelta64(1, 'h')
TD64_M = np.timedelta64(1, 'm')
TD64_S = np.timedelta64(1, 's')
TD64_MS = np.timedelta64(1, 'ms')
TD64_US = np.timedelta64(1, 'us')
TD64_NS = np.timedelta64(1, 'ns')

_DT_NOT_FROM_INT = (DT64_DAY, DT64_MONTH)

DTU_PYARROW = set(('ns', 'D', 's'))

def to_datetime64(
        value: DateInitializer,
        dtype: tp.Optional[np.dtype] = None
        ) -> np.datetime64:
    '''
    Convert a value ot a datetime64; this must be a datetime64 so as to be hashable.
    '''
    # for now, only support creating from a string, as creation from integers is based on offset from epoch
    if not isinstance(value, np.datetime64):
        if dtype is None:
            # let constructor figure it out
            dt = np.datetime64(value)
        else: # assume value is single value;
            # note that integers will be converted to units from epoch
            if isinstance(value, int):
                if dtype == DT64_YEAR:
                    # convert to string as that is likely what is wanted
                    value = str(value)
                elif dtype in _DT_NOT_FROM_INT:
                    raise RuntimeError('attempting to create {} from an integer, which is generally not desired as the result will be offset from the epoch.'.format(dtype))
            # cannot use the datetime directly
            if dtype != np.datetime64:
                dt = np.datetime64(value, np.datetime_data(dtype)[0])
            else: # cannot use a generic datetime type
                dt = np.datetime64(value)
    else: # if a dtype was explicitly given, check it
        # value is an instance of a datetime64, and has a dtype attr
        dt = value
        if dtype:
            # dtype can be either generic, or a matching specific dtype
            if dtype != np.datetime64 and dtype != dt.dtype:
                raise RuntimeError(f'value ({dt}) is not a supported dtype ({dtype})')
    return dt

def to_timedelta64(value: datetime.timedelta) -> np.timedelta64:
    '''
    Convert a datetime.timedelta into a NumPy timedelta64. This approach is better than using np.timedelta64(value), as that reduces all values to microseconds.
    '''
    return reduce(operator.add,
        (np.timedelta64(getattr(value, attr), code) for attr, code in TIME_DELTA_ATTR_MAP if getattr(value, attr) > 0))

def _slice_to_datetime_slice_args(key: slice,
        dtype: tp.Optional[np.dtype] = None
        ) -> tp.Iterator[tp.Optional[np.datetime64]]:
    '''
    Given a slice representing a datetime region, convert to arguments for a new slice, possibly using the appropriate dtype for conversion.
    '''
    for attr in SLICE_ATTRS:
        value = getattr(key, attr)
        if value is None:
            yield None
        elif attr == SLICE_STEP_ATTR:
            # steps are never transformed
            yield value
        else:
            yield to_datetime64(value, dtype=dtype)

def key_to_datetime_key(
        key: GetItemKeyType,
        dtype: np.dtype = np.datetime64,
        ) -> GetItemKeyType:
    '''
    Given an get item key for a Date index, convert it to np.datetime64 representation.
    '''
    if isinstance(key, slice):
        return slice(*_slice_to_datetime_slice_args(key, dtype=dtype))

    if isinstance(key, (datetime.date, datetime.datetime)):
        return np.datetime64(key)

    if isinstance(key, np.datetime64):
        return key

    if isinstance(key, str):
        return to_datetime64(key, dtype=dtype)

    if isinstance(key, np.ndarray):
        if key.dtype.kind == 'b' or key.dtype.kind == 'M':
            return key
        return key.astype(dtype)

    if hasattr(key, '__len__'):
        # use dtype via array constructor to determine type; or just use datetime64 to parse to the passed-in representation
        return np.array(key, dtype=dtype)

    if hasattr(key, '__next__'): # a generator-like
        return np.array(tuple(key), dtype=dtype) #type: ignore

    # for now, return key unaltered
    return key

#-------------------------------------------------------------------------------

def array_to_groups_and_locations(
        array: np.ndarray,
        unique_axis: tp.Optional[int] = 0) -> tp.Tuple[np.ndarray, np.ndarray]:
    '''Locations are index positions for each group.
    '''
    try:
        groups, locations = np.unique(
                array,
                return_inverse=True,
                axis=unique_axis)
    except TypeError:
        # group by string representations, necessary when types are not comparable
        _, group_index, locations = np.unique(
                array.astype(str),
                return_index=True,
                return_inverse=True,
                axis=unique_axis)
        # groups here are the strings; need to restore to values
        groups = array[group_index]

    return groups, locations


def isna_element(value: tp.Any) -> bool:
    '''Return Boolean if value is an NA. This does not yet handle pd.NA
    '''
    try:
        return np.isnan(value) #type: ignore
    except TypeError:
        pass
    try:
        return np.isnat(value) #type: ignore
    except TypeError:
        pass
    return value is None


def isna_array(array: np.ndarray,
        include_none: bool = True,
        ) -> np.ndarray:
    '''Given an np.ndarray, return a Boolean array setting True for missing values.

    Note: the returned array is not made immutable.
    '''
    kind = array.dtype.kind
    # matches all floating point types
    if kind in DTYPE_INEXACT_KINDS:
        return np.isnan(array)
    elif kind in DTYPE_NAT_KINDS:
        return np.isnat(array)
    # match everything that is not an object; options are: biufcmMOSUV
    elif kind != 'O':
        return np.full(array.shape, False, dtype=bool)
    # only check for None if we have an object type
    # NOTE: this will not work for Frames contained within a Series
    if include_none:
        return np.not_equal(array, array) | np.equal(array, None)
    return np.not_equal(array, array)


def binary_transition(
        array: np.ndarray,
        axis: int = 0
        ) -> np.ndarray:
    '''
    Given a Boolean 1D array, return the index positions (integers) at False values where that False was previously True, or will be True

    Returns:
        For a 1D input, a 1D array of integers; for a 2D input, a 1D object array of lists, where each position corresponds to a found index position. Returning a list is undesirable, but more efficient as a list will be neede for selection downstream.
    '''

    if len(array) == 0:
        # NOTE: on some platforms this may not be the same dtype as returned from np.nonzero
        return EMPTY_ARRAY_INT

    not_array = ~array

    if array.ndim == 1:
        # non-nan values that go (from left to right) to NaN
        target_sel_leading = (array ^ roll_1d(array, -1)) & not_array
        target_sel_leading[-1] = False # wrap around observation invalid
        # non-nan values that were previously NaN (from left to right)
        target_sel_trailing = (array ^ roll_1d(array, 1)) & not_array
        target_sel_trailing[0] = False # wrap around observation invalid

        return np.nonzero(target_sel_leading | target_sel_trailing)[0]

    elif array.ndim == 2:
        # if axis == 0, we compare rows going down/up, looking at column values
        # non-nan values that go (from left to right) to NaN
        target_sel_leading = (array ^ roll_2d(array, -1, axis=axis)) & not_array
        # non-nan values that were previously NaN (from left to right)
        target_sel_trailing = (array ^ roll_2d(array, 1, axis=axis)) & not_array

        # wrap around observation invalid
        if axis == 0:
            # process an entire row
            target_sel_leading[-1, :] = False
            target_sel_trailing[0, :] = False
        else:
            # process entire column
            target_sel_leading[:, -1] = False
            target_sel_trailing[:, 0] = False

        # this dictionary could be very sparse compared to axis dimensionality
        indices_by_axis: tp.DefaultDict[int, tp.List[int]] = defaultdict(list)
        for y, x in zip(*np.nonzero(target_sel_leading | target_sel_trailing)):
            if axis == 0:
                # store many rows values for each column
                indices_by_axis[x].append(y)
            else:
                indices_by_axis[y].append(x)

        # if axis is 0, return column width, else return row height
        post = np.empty(dtype=object, shape=array.shape[not axis])
        for k, v in indices_by_axis.items():
            post[k] = v

        return post

    raise NotImplementedError(f'no handling for array with ndim: {array.ndim}')

#-------------------------------------------------------------------------------

def array_deepcopy(
        array: np.ndarray,
        memo: tp.Optional[tp.Dict[int, tp.Any]],
        ) -> np.ndarray:
    '''
    Create a deepcopy of an array, handling memo lookup, insertion, and object arrays.
    '''
    ident = id(array)
    if memo is not None and ident in memo:
        return memo[ident]

    if array.dtype == DTYPE_OBJECT:
        post = deepcopy(array, memo)
    else:
        post = array.copy()

    if post.ndim > 0:
        post.flags.writeable = array.flags.writeable

    if memo is not None:
        memo[ident] = post
    return post

#-------------------------------------------------------------------------------
# tools for handling duplicates

def _array_to_duplicated_hashable(
        array: np.ndarray,
        axis: int = 0,
        exclude_first: bool = False,
        exclude_last: bool = False) -> np.ndarray:
    '''
    Algorithm for finding duplicates in unsortable arrays for hashables. This will always be an object array.
    '''
    # np.unique fails under the same conditions that sorting fails, so there is no need to try np.unique: must go to set drectly.
    len_axis = array.shape[axis]

    if array.ndim == 1:
        value_source = array
        to_hashable = None
    else:
        if axis == 0:
            value_source = array # will iterate rows
        else:
            value_source = (array[:, i] for i in range(len_axis))
        # values will be arrays; must convert to tuples to make hashable
        to_hashable = tuple


    is_dupe = np.full(len_axis, False)

    # could exit early with a set, but would have to hash all array twice to go to set and dictionary
    # creating a list for each entry and tracking indices would be very expensive

    unique_to_first: tp.Dict[tp.Hashable, int] = {} # value to first occurence
    dupe_to_first: tp.Dict[tp.Hashable, int] = {}
    dupe_to_last: tp.Dict[tp.Hashable, int] = {}

    for idx, v in enumerate(value_source):

        if to_hashable:
            v = to_hashable(v)

        if v not in unique_to_first:
            unique_to_first[v] = idx
        else:
            # v has been seen before; upate Boolean array
            is_dupe[idx] = True

            # if no entry in dupe to first, no update with value in unique to first, which is the index this values was first seen
            if v not in dupe_to_first:
                dupe_to_first[v] = unique_to_first[v]
            # always update last
            dupe_to_last[v] = idx

    if exclude_last: # overwrite with False
        is_dupe[list(dupe_to_last.values())] = False

    if not exclude_first: # add in first values
        is_dupe[list(dupe_to_first.values())] = True

    return is_dupe


def _array_to_duplicated_sortable(
        array: np.ndarray,
        axis: int = 0,
        exclude_first: bool = False,
        exclude_last: bool = False) -> np.ndarray:
    '''
    Algorithm for finding duplicates in sortable arrays. This may or may not be an object array, as some object arrays (those of compatible types) are sortable.
    '''
    # based in part on https://stackoverflow.com/questions/11528078/determining-duplicate-values-in-an-array
    # https://stackoverflow.com/a/43033882/388739
    # indices to sort and sorted array
    # a right roll on the sorted array, comparing to the original sorted array. creates a boolean array, with all non-first duplicates marked as True

    # NOTE: this is not compatible with heterogenous typed object arrays, raises TypeError

    if array.ndim == 1:
        o_idx = np.argsort(array, axis=None, kind=DEFAULT_STABLE_SORT_KIND)
        array_sorted = array[o_idx]
        opposite_axis = 0
        # f_flags is True where there are duplicated values in the sorted array
        f_flags = array_sorted == roll_1d(array_sorted, 1)
    else:
        if axis == 0: # sort rows
            # first should be last
            arg = [array[:, x] for x in range(array.shape[1] - 1, -1, -1)]
            o_idx = np.lexsort(arg)
            array_sorted = array[o_idx]
        elif axis == 1: # sort columns
            arg = [array[x] for x in range(array.shape[0] - 1, -1, -1)]
            o_idx = np.lexsort(arg)
            array_sorted = array[:, o_idx]
        else:
            raise NotImplementedError(f'no handling for axis: {axis}')

        opposite_axis = int(not bool(axis))
        # rolling axis 1 rotates columns; roll axis 0 rotates rows
        match = array_sorted == roll_2d(array_sorted, 1, axis=axis)
        f_flags = match.all(axis=opposite_axis)

    if not f_flags.any():
        # we always return a 1 dim array
        return np.full(len(f_flags), False)

    # The first element of f_flags should always be False.
    # In certain edge cases, this doesn't happen naturally.
    # Index 0 should always exist, due to `.any()` behavior.
    f_flags[0] = np.False_

    if exclude_first and not exclude_last:
        dupes = f_flags
    else:
        # non-LAST duplicates is a left roll of the non-first flags.
        l_flags = roll_1d(f_flags, -1)

        if not exclude_first and exclude_last:
            dupes = l_flags
        elif not exclude_first and not exclude_last:
            # all duplicates is the union.
            dupes = f_flags | l_flags
        else:
            # all non-first, non-last duplicates is the intersection.
            dupes = f_flags & l_flags

    # undo the sort: get the indices to extract Booleans from dupes; in some cases r_idx is the same as o_idx, but not all
    r_idx = np.argsort(o_idx, axis=None, kind=DEFAULT_STABLE_SORT_KIND)
    return dupes[r_idx]



def array_to_duplicated(
        array: np.ndarray,
        axis: int = 0,
        exclude_first: bool = False,
        exclude_last: bool = False,
        ) -> np.ndarray:
    '''Given a numpy array (1D or 2D), return a Boolean array along the specified axis that shows which values are duplicated. By default, all duplicates are indicated. For 2d arrays, axis 0 compares rows and returns a row-length Boolean array; axis 1 compares columns and returns a column-length Boolean array.

    Args:
        exclude_first: Mark as True all duplicates except the first encountared.
        exclude_last: Mark as True all duplicates except the last encountared.
    '''
    try:
        return _array_to_duplicated_sortable(
                array=array,
                axis=axis,
                exclude_first=exclude_first,
                exclude_last=exclude_last
                )
    except TypeError: # raised if not sorted
        return _array_to_duplicated_hashable(
                array=array,
                axis=axis,
                exclude_first=exclude_first,
                exclude_last=exclude_last
                )


#-------------------------------------------------------------------------------
def array_shift(*,
        array: np.ndarray,
        shift: int,
        axis: int, # 0 is rows, 1 is columns
        wrap: bool,
        fill_value: tp.Any = np.nan) -> np.ndarray:
    '''
    Apply an np-style roll to a 1D or 2D array; if wrap is False, fill values out-shifted values with fill_value.

    Args:
        fill_value: only used if wrap is False.
    '''

    # works for all shapes
    if shift > 0:
        shift_mod = shift % array.shape[axis]
    elif shift < 0:
        # do negative modulo to force negative value
        shift_mod = shift % -array.shape[axis]
    else:
        shift_mod = 0

    if (not wrap and shift == 0) or (wrap and shift_mod == 0):
        # must copy so as not let caller mutate arguement
        return array.copy()

    if wrap:
        # roll functions will handle finding noop rolls
        if array.ndim == 1:
            return roll_1d(array, shift_mod)
        return roll_2d(array, shift_mod, axis=axis)

    # will insure that the result can contain the fill and the original values
    result = full_for_fill(array.dtype, array.shape, fill_value)

    if axis == 0:
        if shift > 0:
            result[shift:] = array[:-shift]
        elif shift < 0:
            result[:shift] = array[-shift:]
    elif axis == 1:
        if shift > 0:
            result[:, shift:] = array[:, :-shift]
        elif shift < 0:
            result[:, :shift] = array[:, -shift:]

    return result

def array2d_to_tuples(array: np.ndarray) -> tp.Iterator[tp.Tuple[tp.Any, ...]]:
    yield from map(tuple, array)

def array2d_to_array1d(array: np.ndarray) -> np.ndarray:
    post = np.empty(array.shape[0], dtype=object)
    for i, row in enumerate(array):
        post[i] = tuple(row)
    post.flags.writeable = False
    return post

def array1d_to_last_contiguous_to_edge(array: np.ndarray) -> int:
    '''
    Given a Boolean array, return the start index where of the last range, through to the end, of contiguous True values.
    '''
    length = len(array)
    if len(array) == 0:
        return 1
    if array[-1] == False: #pylint: disable=C0121
        # if last values is False, no contiguous region
        return length
    count = array.sum()
    if count == 0: # if not any are true, no regions found
        return length
    if count == length: # if all are true
        return 0

    transitions = np.empty(length, dtype=bool)
    transitions[0] = False # first value not a transition
    # compare current to previous; do not compare first
    np.not_equal(array[:-1], array[1:], out=transitions[1:])
    # transition_idx must always contain at least one index from here
    transition_idx: tp.Sequence[int] = np.nonzero(transitions)[0]
    # last element must be True, so there will always be one transition, and the last transition will mark the boundary of a contiguous region
    return transition_idx[-1]

    # NOTE: last checks are not necessary
    # if array[last_idx:].all():
    #     return last_idx
    # return length

#-------------------------------------------------------------------------------
# extension to union and intersection handling

def _ufunc_set_1d(
        func: tp.Callable[[np.ndarray, np.ndarray], np.ndarray],
        array: np.ndarray,
        other: np.ndarray,
        *,
        assume_unique: bool = False
        ) -> np.ndarray:
    '''
    Peform 1D set operations. When possible, short-circuit comparison and return array with original order.

    NOTE: there are known issues with how NP handles NaN and NaT with set operations, for example:

    >>> np.intersect1d((np.nan, 3), (np.nan, 3))
    array([3.])
    >>> np.union1d((np.nan, 3), (np.nan, 3))
    array([ 3., nan, nan])

    For unions, and for float and datetime64 types, a correction is made, but of object types, no efficient solution is available.

    Args:
        assume_unique: if arguments are assumed unique, can implement optional identity filtering, which retains order (un sorted) for opperands that are equal. This is important in numerous operations on the matching Indices where order should not be perterbed.
    '''
    is_union = func == np.union1d
    is_intersection = func == np.intersect1d
    is_difference = func == np.setdiff1d

    if not (is_union or is_intersection or is_difference):
        raise NotImplementedError('unexpected func', func)

    dtype = resolve_dtype(array.dtype, other.dtype)

    # optimizations for empty arrays
    if is_intersection:
        if len(array) == 0 or len(other) == 0:
            # not sure what DTYPE is correct to return here
            post = np.array(EMPTY_TUPLE, dtype=dtype)
            post.flags.writeable = False
            return post
    elif is_difference:
        if len(array) == 0:
            post = np.array(EMPTY_TUPLE, dtype=dtype)
            post.flags.writeable = False
            return post

    if assume_unique:
        # can only return arguments, and use length to determine unique comparison condition, if arguments are assumed to already be unique
        if is_union:
            if len(array) == 0:
                return other
            elif len(other) == 0:
                return array
        elif is_difference:
            if len(other) == 0:
                return array

        if len(array) == len(other):
            arrays_are_equal = False
            compare = array == other
            # if sizes are the same, the result of == is mostly a bool array; comparison to some arrays (e.g. string), will result in a single Boolean, but it should always be False
            if isinstance(compare, BOOL_TYPES) and compare:
                arrays_are_equal = True #pragma: no cover
            elif isinstance(compare, np.ndarray) and compare.all(axis=None):
                arrays_are_equal = True

            if arrays_are_equal:
                if is_difference:
                    post = np.array(EMPTY_TUPLE, dtype=dtype)
                    post.flags.writeable = False
                    return post
                return array

    array_is_str = array.dtype.kind in DTYPE_STR_KINDS
    other_is_str = other.dtype.kind in DTYPE_STR_KINDS
    set_compare = array_is_str ^ other_is_str

    if set_compare or dtype.kind == 'O':
        if is_union:
            result = frozenset(array) | frozenset(other)
        elif is_intersection:
            result = frozenset(array) & frozenset(other)
        else:
            result = frozenset(array).difference(frozenset(other))
        # NOTE: try to sort, as set ordering is not stable
        try:
            result = sorted(result) #type: ignore
        except TypeError:
            pass
        post, _ = iterable_to_array_1d(result, dtype) # return immutable array
        return post

    if is_union:
        post = func(array, other)
    else:
        post = func(array, other, assume_unique=assume_unique) #type: ignore

    post.flags.writeable = False
    return post

def _ufunc_set_2d(
        func: tp.Callable[[np.ndarray, np.ndarray], np.ndarray],
        array: np.ndarray,
        other: np.ndarray,
        *,
        assume_unique: bool=False
        ) -> np.ndarray:
    '''
    Peform 2D set operations. When possible, short-circuit comparison and return array with original order.

    Args:
        func: a 1d set operation
        array: can be a 2D array, or a 1D object array of tuples.
        other: can be a 2D array, or a 1D object array of tuples.
        assume_unique: if True, array operands are assumed unique and order is preserved for matching operands.
    Returns:
        Either a 2D array (if both operands are 2D), or a 1D object array of tuples (if one or both are 1d tuple arrays).
    '''
    # NOTE: diversity if returned values may be a problem; likely should always return 2D array, or follow pattern that if both operands are 2D, a 2D array is returned

    is_union = func == np.union1d
    is_intersection = func == np.intersect1d
    is_difference = func == np.setdiff1d
    is_2d = array.ndim == 2 and other.ndim == 2

    if not (is_union or is_intersection or is_difference):
        raise NotImplementedError('unexpected func', func)

    # if either are object, or combination resovle to object, get object
    dtype = resolve_dtype(array.dtype, other.dtype)

    # optimizations for empty arrays
    if is_intersection: # intersection with empty
        if len(array) == 0 or len(other) == 0:
            post = np.array(EMPTY_TUPLE, dtype=dtype)
            if is_2d:
                post = post.reshape(0, 0)
            post.flags.writeable = False
            return post
    elif is_difference:
        if len(array) == 0:
            post = np.array(EMPTY_TUPLE, dtype=dtype)
            if is_2d:
                post = post.reshape(0, 0)
            post.flags.writeable = False
            return post

    if assume_unique:
        # can only return arguments, and use length to determine unique comparison condition, if arguments are assumed to already be unique
        if is_union:
            if len(array) == 0:
                return other
            elif len(other) == 0:
                return array
        elif is_difference:
            if len(other) == 0:
                return array

        if array.shape == other.shape:
            arrays_are_equal = False
            compare = array == other
            # will not match a 2D array of integers and 1D array of tuples containing integers (would have to do a post-set comparison, but would loose order)
            if isinstance(compare, BOOL_TYPES) and compare:
                arrays_are_equal = True #pragma: no cover
            elif isinstance(compare, np.ndarray) and compare.all(axis=None):
                arrays_are_equal = True
            if arrays_are_equal:
                if is_difference:
                    post = np.array(EMPTY_TUPLE, dtype=dtype)
                    if is_2d:
                        post = post.reshape(0, 0)
                    post.flags.writeable = False
                    return post
                return array

    if dtype.kind == 'O':
        # assume that 1D arrays arrays are arrays of tuples
        if array.ndim == 1:
            array_set = frozenset(array)
        else: # assume row-wise comparison
            array_set = frozenset(tuple(row) for row in array)

        if other.ndim == 1:
            other_set = frozenset(other)
        else: # assume row-wise comparison
            other_set = frozenset(tuple(row) for row in other)

        if is_union:
            result = array_set | other_set
        elif is_intersection:
            result = array_set & other_set
        else:
            result = array_set.difference(other_set)

        # NOTE: this sort may not always be successful
        try:
            values: tp.Sequence[tp.Tuple[tp.Hashable, ...]] = sorted(result)
        except TypeError:
            values = tuple(result)

        if is_2d:
            if len(values) == 0:
                post = np.array(EMPTY_TUPLE, dtype=dtype).reshape(0, 0)
            else:
                post = np.array(values, dtype=object)
            post.flags.writeable = False
            return post

        post = np.empty(len(values), dtype=object)
        post[:] = values
        post.flags.writeable = False
        return post

    # from here, we assume we have two 2D arrays
    if not is_2d:
        raise RuntimeError('non-object arrays have to both be 2D')

    # number of columns must be the same, as doing row-wise comparison, and determines the length of each row
    assert array.shape[1] == other.shape[1]
    width = array.shape[1]

    if array.dtype != dtype:
        array = array.astype(dtype)
    if other.dtype != dtype:
        other = other.astype(dtype)

    func_kwargs = {} if is_union else dict(assume_unique=assume_unique)

    if width == 1:
        # let the function flatten the array, then reshape into 2D
        post = func(array, other, **func_kwargs)  # type: ignore
        post = post.reshape(len(post), width)
        post.flags.writeable = False
        return post

    # this approach based on https://stackoverflow.com/questions/9269681/intersection-of-2d-numpy-ndarrays
    # we can use a the 1D function on the rows, once converted to a structured array

    dtype_view = [('', array.dtype)] * width
    # creates a view of tuples for 1D operation
    array_view = array.view(dtype_view)
    other_view = other.view(dtype_view)
    post = func(array_view, other_view, **func_kwargs).view(dtype).reshape(-1, width) # type: ignore
    post.flags.writeable = False
    return post

def union1d(array: np.ndarray,
        other: np.ndarray,
        assume_unique: bool = False
        ) -> np.ndarray:
    '''
    Union on 1D array, handling diverse types and short-circuiting to preserve order where appropriate.
    '''
    return _ufunc_set_1d(np.union1d,
            array,
            other,
            assume_unique=assume_unique)

def intersect1d(
        array: np.ndarray,
        other: np.ndarray,
        assume_unique: bool = False
        ) -> np.ndarray:
    '''
    Intersect on 1D array, handling diverse types and short-circuiting to preserve order where appropriate.
    '''
    return _ufunc_set_1d(np.intersect1d,
            array,
            other,
            assume_unique=assume_unique)

def setdiff1d(
        array: np.ndarray,
        other: np.ndarray,
        assume_unique: bool = False
        ) -> np.ndarray:
    '''
    Difference on 1D array, handling diverse types and short-circuiting to preserve order where appropriate
    '''
    return _ufunc_set_1d(np.setdiff1d,
        array,
        other,
        assume_unique=assume_unique)

def union2d(
        array: np.ndarray,
        other: np.ndarray,
        assume_unique: bool = False
        ) -> np.ndarray:
    '''
    Union on 2D array, handling diverse types and short-circuiting to preserve order where appropriate.
    '''
    return _ufunc_set_2d(np.union1d,
            array,
            other,
            assume_unique=assume_unique)

def intersect2d(
        array: np.ndarray,
        other: np.ndarray,
        assume_unique: bool = False
        ) -> np.ndarray:
    '''
    Intersect on 2D array, handling diverse types and short-circuiting to preserve order where appropriate.
    '''
    return _ufunc_set_2d(np.intersect1d,
            array,
            other,
            assume_unique=assume_unique)

def setdiff2d(
        array: np.ndarray,
        other: np.ndarray,
        assume_unique: bool = False
        ) -> np.ndarray:
    '''
    Difference on 2D array, handling diverse types and short-circuiting to preserve order where appropriate.
    '''
    return _ufunc_set_2d(np.setdiff1d,
        array,
        other,
        assume_unique=assume_unique)

def ufunc_set_iter(
        arrays: tp.Iterable[np.ndarray],
        union: bool = False,
        assume_unique: bool = False
        ) -> np.ndarray:
    '''
    Iteratively apply a set operation ufunc to 1D or 2D arrays; if all are equal, no operation is performed and order is retained.

    Args:
        arrays: iterator of arrays; can be a Generator.
        union: if True, a union is taken, else, an intersection.
    '''
    arrays = iter(arrays)
    result = next(arrays)

    # will detect ndim by first value, but insure that all other arrays have the same ndim
    if result.ndim == 1:
        ufunc = union1d if union else intersect1d
        ndim = 1
    else: # ndim == 2
        ufunc = union2d if union else intersect2d
        ndim = 2

    for array in arrays:
        if array.ndim != ndim:
            raise RuntimeError('arrays do not all have the same ndim')

        # to retain order on identity, assume_unique must be True
        result = ufunc(result, array, assume_unique=assume_unique)

        if not union and len(result) == 0:
            # short circuit intersection that results in no common values
            break

    result.flags.writeable = False
    return result


def _isin_1d(
        array: np.ndarray,
        other: tp.FrozenSet[tp.Any]
        ) -> np.ndarray:
    '''
    Iterate over an 1D array to build a 1D Boolean ndarray representing whether or not the original element is in the set

    Args:
        array: The source array
        other: The set of elements being looked for
    '''
    result: np.ndarray = np.empty(array.shape, dtype=DTYPE_BOOL)

    for i, element in enumerate(array):
        result[i] = element in other

    result.flags.writeable = False
    return result


def _isin_2d(
        array: np.ndarray,
        other: tp.FrozenSet[tp.Any]
        ) -> np.ndarray:
    '''
    Iterate over an 2D array to build a 2D, immutable, Boolean ndarray representing whether or not the original element is in the set

    Args:
        array: The source array
        other: The set of elements being looked for
    '''
    result: np.ndarray = np.empty(array.shape, dtype=DTYPE_BOOL)

    for (i, j), v in np.ndenumerate(array):
        result[i, j] = v in other

    result.flags.writeable = False
    return result


def isin_array(*,
        array: np.ndarray,
        array_is_unique: bool,
        other: np.ndarray,
        other_is_unique: bool,
        ) -> np.ndarray:
    '''Core isin processing after other has been converted to an array.
    '''
    if array.dtype == DTYPE_OBJECT or other.dtype == DTYPE_OBJECT:
        # both funcs return immutable arrays
        func = _isin_1d if array.ndim == 1 else _isin_2d
        try:
            return func(array, frozenset(other))
        except TypeError: # only occur when something is unhashable.
            pass

    assume_unique = array_is_unique and other_is_unique
    func = np.in1d if array.ndim == 1 else np.isin

    result = func(array, other, assume_unique=assume_unique) #type: ignore
    result.flags.writeable = False

    return result


def isin(
        array: np.ndarray,
        other: tp.Iterable[tp.Any],
        array_is_unique: bool = False,
        ) -> np.ndarray:
    '''
    Builds a same-size, immutable, Boolean ndarray representing whether or not the original element is in another ndarray

    numpy's has very poor isin performance, as it converts both arguments to array-like objects.
    This implementation optimizes that by converting the lookup argument into a set, providing constant comparison time.

    Args:
        array: The source array
        other: The elements being looked for
        array_is_unique: if array is known to be unique
    '''
    result: tp.Optional[np.ndarray] = None

    if hasattr(other, '__len__') and len(other) == 0: #type: ignore
        result = np.full(array.shape, False, dtype=DTYPE_BOOL)
        result.flags.writeable = False
        return result

    other, other_is_unique = iterable_to_array_1d(other)

    return isin_array(array=array,
            array_is_unique=array_is_unique,
            other=other,
            other_is_unique=other_is_unique,
            )

#-------------------------------------------------------------------------------
def _ufunc_logical_skipna(
        array: np.ndarray,
        ufunc: AnyCallable,
        skipna: bool,
        axis: int = 0,
        out: tp.Optional[np.ndarray] = None
        ) -> np.ndarray:
    '''
    Given a logical (and, or) ufunc that does not support skipna, implement skipna behavior.
    '''
    if ufunc != np.all and ufunc != np.any:
        raise NotImplementedError(f'unsupported ufunc ({ufunc}); use np.all or np.any')

    if len(array) == 0:
        # TODO: handle if this is ndim == 2 and has no length
        # any() of an empty array is False
        return ufunc == np.all

    kind = array.dtype.kind

    #---------------------------------------------------------------------------
    # types that cannot have NA
    if kind == 'b':
        return ufunc(array, axis=axis, out=out)
    if kind in DTYPE_INT_KINDS:
        return ufunc(array, axis=axis, out=out)
    if kind in DTYPE_STR_KINDS:
        # only string in object arrays can be converted to bool, where the empty string will be evaluated as False; here, manually check
        return ufunc(array != '', axis=axis, out=out)

    #---------------------------------------------------------------------------
    # types that can have NA

    if kind in DTYPE_INEXACT_KINDS:
        isna = isna_array(array)
        hasna = isna.any() # returns single value for 1d, 2d
        if hasna and skipna:
            fill_value = 0.0 if ufunc == np.any else 1.0
            v = array.copy()
            v[isna] = fill_value
            return ufunc(v, axis=axis, out=out)
        elif hasna and not skipna:
            # if array.ndim == 1:
            #     return np.nan
            raise TypeError('cannot propagate NaN without expanding to object array result')
        return ufunc(array, axis=axis, out=out)

    if kind in DTYPE_NAT_KINDS:
        isna = isna_array(array)
        hasna = isna.any() # returns single value for 1d, 2d
        # all dates are truthy, special handling only to propagate NaNs
        if hasna and not skipna:
            # if array.ndim == 1:
            #     return NAT
            raise TypeError('cannot propagate NaN without expanding to object array result')
        # to ignore NaN, simply fall back on all-truth behavior, below

    if kind == 'O':
        # all object types: convert to boolean aray then process
        isna = isna_array(array)
        hasna = isna.any() # returns single value for 1d, 2d
        if hasna and skipna:
            # supply True for np.all, False for np.any
            fill_value = False if ufunc == np.any else True
            v = array.copy()
            v = v.astype(bool) # nan will be converted to True
            v[isna] = fill_value
        elif hasna and not skipna:
            # if array.ndim == 1:
            #     return np.nan
            raise TypeError('cannot propagate NaN without expanding to object array result')
        else:
            v = array.astype(bool)
        return ufunc(v, axis=axis, out=out)

    # all types other than strings or objects assume truthy
    if array.ndim == 1:
        return True
    return np.full(array.shape[0 if axis else 1], fill_value=True, dtype=bool)


def ufunc_all(array: np.ndarray,
        axis: int = 0,
        out: tp.Optional[np.ndarray] = None
        ) -> np.ndarray:
    return _ufunc_logical_skipna(array,
            ufunc=np.all,
            skipna=False,
            axis=axis,
            out=out)

ufunc_all.__doc__ = np.all.__doc__

def ufunc_any(array: np.ndarray,
        axis: int = 0,
        out: tp.Optional[np.ndarray] = None
        ) -> np.ndarray:
    return _ufunc_logical_skipna(array,
            ufunc=np.any,
            skipna=False,
            axis=axis,
            out=out)

ufunc_any.__doc__ = np.any.__doc__

def ufunc_nanall(array: np.ndarray,
        axis: int = 0,
        out: tp.Optional[np.ndarray] = None
        ) -> np.ndarray:
    return _ufunc_logical_skipna(array,
            ufunc=np.all,
            skipna=True,
            axis=axis,
            out=out)

def ufunc_nanany(array: np.ndarray,
        axis: int = 0,
        out: tp.Optional[np.ndarray] = None
        ) -> np.ndarray:
    return _ufunc_logical_skipna(array,
            ufunc=np.any,
            skipna=True,
            axis=axis,
            out=out)

#-------------------------------------------------------------------------------

def array_from_element_attr(*,
        array: np.ndarray,
        attr_name: str,
        dtype: np.dtype
        ) -> np.array:
    '''
    Handle element-wise attribute acesss on arrays of Python date/datetime objects.
    '''
    if array.ndim == 1:
        post = np.fromiter(
                (getattr(d, attr_name) for d in array),
                count=len(array),
                dtype=dtype,
                )
    else:
        post = np.empty(shape=array.shape, dtype=dtype)
        for iloc, e in np.ndenumerate(array):
            post[iloc] = getattr(e, attr_name)

    post.flags.writeable = False
    return post

def array_from_element_apply(*,
        array: np.ndarray,
        func: AnyCallable,
        dtype: np.dtype
        ) -> np.array:
    '''
    Handle element-wise function application.
    '''
    if array.ndim == 1 and dtype != DTYPE_OBJECT and dtype.kind not in DTYPE_STR_KINDS:
        post = np.fromiter(
                (func(d) for d in array),
                count=len(array),
                dtype=dtype,
                )
    else:
        post = np.empty(shape=array.shape, dtype=dtype)
        for iloc, e in np.ndenumerate(array):
            post[iloc] = func(e)

    post.flags.writeable = False
    return post


def array_from_element_method(*,
        array: np.ndarray,
        method_name: str,
        args: tp.Tuple[tp.Any, ...],
        dtype: np.dtype,
        pre_insert: tp.Optional[AnyCallable] = None,
        ) -> np.array:
    '''
    Handle element-wise method calling on arrays of Python date/datetime objects.

    Args:
        pre_insert:
    '''
    if dtype == DTYPE_STR:
        # build into a list first, then construct array to determine size
        if array.ndim == 1:
            if pre_insert:
                proto = [pre_insert(getattr(d, method_name)(*args)) for d in array]
            else:
                proto = [getattr(d, method_name)(*args) for d in array]
        else:
            proto = [[None for _ in range(array.shape[1])]
                    for _ in range(array.shape[0])]
            if pre_insert:
                for (y, x), e in np.ndenumerate(array):
                    proto[y][x] = pre_insert(getattr(e, method_name)(*args))
            else:
                for (y, x), e in np.ndenumerate(array):
                    proto[y][x] = getattr(e, method_name)(*args)
        post = np.array(proto, dtype=dtype)

    else:
        if array.ndim == 1 and dtype != DTYPE_OBJECT:
            # NOTE: can I get the method off the clas and pass self
            if pre_insert:
                post = np.fromiter(
                        (pre_insert(getattr(d, method_name)(*args)) for d in array),
                        count=len(array),
                        dtype=dtype,
                        )
            else:
                post = np.fromiter(
                        (getattr(d, method_name)(*args) for d in array),
                        count=len(array),
                        dtype=dtype,
                        )
        else:
            post = np.empty(shape=array.shape, dtype=dtype)
            if pre_insert:
                for iloc, e in np.ndenumerate(array):
                    post[iloc] = pre_insert(getattr(e, method_name)(*args))
            else:
                for iloc, e in np.ndenumerate(array):
                    post[iloc] = getattr(e, method_name)(*args)

    post.flags.writeable = False
    return post


def array_from_iterator(iterator: tp.Iterator[tp.Any],
        count: int,
        dtype: DtypeSpecifier,
        ) -> np.ndarray:
    '''Given an iterator/generaotr of known size and dtype, load it into an array.
    '''
    if dtype != object:
        array = np.fromiter(iterator,
                count=count,
                dtype=dtype,
                )
    else:
        array = np.empty(count, dtype=dtype)
        for i, v in enumerate(iterator):
            array[i] = v

    array.flags.writeable = False
    return array


#-------------------------------------------------------------------------------

class PositionsAllocator:
    '''Resource for re-using a single array of contiguous ascending integers for common applications in IndexBase.
    '''

    _size: int = 1024 # 1048576
    _array: np.ndarray = np.arange(_size, dtype=DTYPE_INT_DEFAULT)
    _array.flags.writeable = False

    @classmethod
    def get(cls, size: int) -> np.ndarray:
        if size > cls._size:
            cls._size = size * 2
            cls._array = np.arange(cls._size, dtype=DTYPE_INT_DEFAULT)
            cls._array.flags.writeable = False
        # slices of immutable arrays are immutable
        return cls._array[:size]


def array_sample(
        array: np.ndarray,
        count: int,
        seed: tp.Optional[int] = None,
        sort: bool = False,
        ) -> np.ndarray:
    '''
    Given an array 1D or 2D array, randomly sample ``count`` components of that array. Sampling is always done "without replacment", meaning that the resulting array will never have duplicates. For 2D array, selection happens along axis 0.
    '''
    if array.ndim != 1:
        raise NotImplementedError(f'no handling for axis {array.ndim}')

    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)

    post = np.random.choice(array, size=count, replace=False)
    if sort:
        post.sort(kind=DEFAULT_SORT_KIND)

    if seed is not None:
        np.random.set_state(state)

    post.flags.writeable = False
    return post


#-------------------------------------------------------------------------------

def slices_from_targets(
        target_index: tp.Sequence[int],
        target_values: tp.Sequence[tp.Any],
        length: int,
        directional_forward: bool,
        limit: int,
        slice_condition: tp.Callable[[slice], bool]
        ) -> tp.Iterator[tp.Tuple[slice, tp.Any]]:
    '''
    Utility function used in fillna_directional implementations for Series and Frame. Yields slices and values for setting contiguous ranges of values.

    NOTE: slice_condition is still needed to check if a slice actually has missing values; see if there is a way to determine these cases in advance, so as to not call a function on each slice.

    Args:
        target_index: iterable of integers, where integers are positions where (as commonly used) values along an axis were previously NA, or will be NA. Often the result of binary_transition()
        target_values: values found at the index positions
        length: the maximum length in the target array
        directional_forward: determine direction
        limit: set a max size for all slices
        slice_condition: optional function for filtering slices.
    '''
    if directional_forward:
        target_slices = (
                slice(start+1, stop)
                for start, stop in
                zip_longest(target_index, target_index[1:], fillvalue=length)
                )
    else:
        # use None to signal first value, but store 0
        target_slices = (
                slice((start+1 if start is not None else 0), stop)
                for start, stop in
                zip(chain(ELEMENT_TUPLE, target_index[:-1]), target_index)
                )

    for target_slice, value in zip(target_slices, target_values):
        # asserts not necessary as slices are created above; but mypy needs them
        assert target_slice.start is not None and target_slice.stop is not None

        # all conditions that are noop slices
        if target_slice.start == target_slice.stop: #pylint: disable=R1724
            # matches case where start is 0 and stop is 0
            continue
        elif directional_forward and target_slice.start >= length:
            continue

        # only process if first value of slice is NaN
        if slice_condition(target_slice):

            if limit > 0:
                # get the length of the range resulting from the slice; if bigger than limit, reduce the by that amount
                shift = len(range(*target_slice.indices(length))) - limit
                if shift > 0:

                    if directional_forward:
                        target_slice = slice(
                                target_slice.start,
                                target_slice.stop - shift)
                    else:
                        target_slice = slice(
                                (target_slice.start or 0) + shift,
                                target_slice.stop)

            yield target_slice, value



#-------------------------------------------------------------------------------
# URL handling, file downloading, file writing

def path_filter(fp: PathSpecifierOrFileLikeOrIterator) -> tp.Union[str, tp.TextIO]:
    '''Realize Path objects as strings, let TextIO pass through, if given.
    '''
    if fp is None:
        raise ValueError('None cannot be interpreted as a file path')
    if isinstance(fp, PathLike):
        return str(fp)
    return fp #type: ignore [return-value]


def _read_url(fp: str) -> str:
    '''
    Read a URL into memory, return a decoded string.
    '''
    with request.urlopen(fp) as response: #pragma: no cover
        return tp.cast(str, response.read().decode('utf-8')) #pragma: no cover


def write_optional_file(
        content: str,
        fp: tp.Optional[PathSpecifierOrFileLike] = None,
        ) -> tp.Optional[str]:

    if fp is not None:
        fp = path_filter(fp)

    fd = f = None
    if not fp: # get a temp file
        fd, fp = tempfile.mkstemp(suffix='.html', text=True)
    elif isinstance(fp, StringIO):
        f = fp
        fp = None
    # nothing to do if we have an fp

    if f is None: # do not have a file object
        try:
            assert isinstance(fp, str)
            with tp.cast(StringIO, open(fp, 'w')) as f:
                f.write(content)
        finally:
            if fd is not None:
                os.close(fd)
    else: # string IO
        f.write(content)
        f.seek(0)
    return fp #type: ignore


@contextlib.contextmanager
def file_like_manager(
        file_like: PathSpecifierOrFileLikeOrIterator,
        encoding: tp.Optional[str] = None,
        mode: str = 'r',
        ) -> tp.Iterator[tp.Iterator[str]]:
    '''
    Return an file or file-like object. Manage closing of file if necessary.
    '''
    file_like = path_filter(file_like)

    is_file = False
    try:
        if isinstance(file_like, str):
            f = open(file_like, mode=mode, encoding=encoding)
            is_file = True
        else:
            f = file_like # assume an open file-like object
        yield f

    finally:
        if is_file:
            f.close()

#-------------------------------------------------------------------------------
# trivial, non NP util

def get_tuple_constructor(
        fields: np.ndarray,
        ) -> TupleConstructorType:
    '''
    Given fields, try to create a Namedtuple and return the `_make` method
    '''
    # this will raise if attrs are invalid
    try:
        return namedtuple('Axis', fields)._make #type: ignore
    except ValueError:
        pass
    raise ValueError('invalid fields for namedtuple; pass `tuple` as constructor')


def key_normalize(key: KeyOrKeys) -> tp.List[tp.Hashable]:
    '''
    Normalizing a key that might be a single element or an iterable of keys; expected return is always a list, as it will be used for getitem selection.
    '''
    if isinstance(key, str) or not hasattr(key, '__len__'):
        return [key]
    return key if isinstance(key, list) else list(key) # type: ignore
