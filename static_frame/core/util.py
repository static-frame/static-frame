import sys
import typing as tp
import os
import operator
import struct

from collections import abc
from collections import defaultdict

from itertools import chain
from io import StringIO
from io import BytesIO
import datetime
from urllib import request
import tempfile
from functools import reduce
from itertools import zip_longest

import numpy as np  # type: ignore


if tp.TYPE_CHECKING:

    from static_frame.core.index_base import IndexBase
    from static_frame.core.index import Index
    from static_frame.core.series import Series
    from static_frame.core.frame import Frame
    from static_frame.core.frame import FrameAsType
    from static_frame.core.type_blocks import TypeBlocks


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

DEFAULT_INT_DTYPE = np.int64 # default for SF construction

# ARCHITECTURE_SIZE = struct.calcsize('P') * 8 # size of pointer
# ARCHITECTURE_INT_DTYPE = np.int64 if ARCHITECTURE_SIZE == 64 else np.int32

DEFAULT_STABLE_SORT_KIND = 'mergesort'
DTYPE_STR_KIND = ('U', 'S') # S is np.bytes_
DTYPE_INT_KIND = ('i', 'u') # signed and unsigned
DTYPE_NAN_KIND = ('f', 'c') # kinds taht support NaN values
DTYPE_DATETIME_KIND = 'M'
DTYPE_TIMEDELTA_KIND = 'm'
DTYPE_NAT_KIND = ('M', 'm')

DTYPE_OBJECT = np.dtype(object)
DTYPE_BOOL = np.dtype(bool)

NULL_SLICE = slice(None)
UNIT_SLICE = slice(0, 1)
SLICE_START_ATTR = 'start'
SLICE_STOP_ATTR = 'stop'
SLICE_STEP_ATTR = 'step'
SLICE_ATTRS = (SLICE_START_ATTR, SLICE_STOP_ATTR, SLICE_STEP_ATTR)

STATIC_ATTR = 'STATIC'

EMPTY_TUPLE = ()

# defaults to float64
EMPTY_ARRAY = np.array(EMPTY_TUPLE, dtype=None)
EMPTY_ARRAY.flags.writeable = False

EMPTY_ARRAY_BOOL = np.array(EMPTY_TUPLE, dtype=DTYPE_BOOL)
EMPTY_ARRAY_BOOL.flags.writeable = False

EMPTY_ARRAY_INT = np.array(EMPTY_TUPLE, dtype=DEFAULT_INT_DTYPE)
EMPTY_ARRAY_INT.flags.writeable = False

NAT = np.datetime64('nat')
# define missing for timedelta as an untyped 0
EMPTY_TIMEDELTA = np.timedelta64(0)

# _DICT_STABLE = sys.version_info >= (3, 6)

# map from datetime.timedelta attrs to np.timedelta64 codes
TIME_DELTA_ATTR_MAP = (
        ('days', 'D'),
        ('seconds', 's'),
        ('microseconds', 'us')
        )

# ufunc functions that will not work with DTYPE_STR_KIND, but do work if converted to object arrays; see UFUNC_AXIS_SKIPNA for the matching functions
UFUNC_AXIS_STR_TO_OBJ = {np.min, np.max, np.sum}

#-------------------------------------------------------------------------------
# utility type groups

INT_TYPES = (int, np.integer) # np.integer catches all np int types

BOOL_TYPES = (bool, np.bool_)

DICTLIKE_TYPES = (abc.Set, dict)

# iterables that cannot be used in NP array constructors; asumes that dictlike types have already been identified
INVALID_ITERABLE_FOR_ARRAY = (abc.ValuesView, abc.KeysView)
NON_STR_TYPES = {int, float, bool}


# for getitem / loc selection
KEY_ITERABLE_TYPES = (list, np.ndarray)

# types of keys that return muultiple items, even if the selection reduces to 1
KEY_MULTIPLE_TYPES = (slice, list, np.ndarray)

# for type hinting
# keys once dimension has been isolated
GetItemKeyType = tp.Union[
        int, np.integer, slice, tp.List[tp.Any], None, 'Index', 'Series', np.ndarray]

# keys that might include a multiple dimensions speciation; tuple is used to identify compound extraction
GetItemKeyTypeCompound = tp.Union[
        tp.Tuple[tp.Any, ...], int, np.integer, slice, tp.List[tp.Any], None, 'Index', 'Series', np.ndarray]

UFunc = tp.Callable[[np.ndarray], np.ndarray]
AnyCallable = tp.Callable[..., tp.Any]

CallableOrMapping = tp.Union[AnyCallable, tp.Mapping[tp.Hashable, tp.Any], 'Series']
KeyOrKeys = tp.Union[tp.Hashable, tp.Iterable[tp.Hashable]]
FilePathOrFileLike = tp.Union[str, tp.TextIO]

DtypeSpecifier = tp.Optional[tp.Union[str, np.dtype, type]]

# support an iterable of specifiers, or mapping based on column names
DtypesSpecifier = tp.Optional[
        tp.Union[tp.Iterable[DtypeSpecifier], tp.Dict[tp.Hashable, DtypeSpecifier]]]

# specifiers that are equivalent to object
DTYPE_SPECIFIERS_OBJECT = {DTYPE_OBJECT, object, tuple}

DepthLevelSpecifier = tp.Union[int, tp.Iterable[int]]

CallableToIterType = tp.Callable[[], tp.Iterable[tp.Any]]

IndexSpecifier = tp.Union[int, str]
IndexInitializer = tp.Union[
        tp.Iterable[tp.Hashable],
        tp.Generator[tp.Hashable, None, None]]
IndexConstructor = tp.Callable[[IndexInitializer], 'IndexBase']


SeriesInitializer = tp.Union[
        tp.Iterable[tp.Any],
        np.ndarray,
        tp.Mapping[tp.Hashable, tp.Any],
        int, float, str, bool]

# support single items, or numpy arrays, or values that can be made into a 2D array
FrameInitializer = tp.Union[
        tp.Iterable[tp.Iterable[tp.Any]],
        np.ndarray,
        ]

FRAME_INITIALIZER_DEFAULT = object()

DateInitializer = tp.Union[str, datetime.date, np.datetime64]
YearMonthInitializer = tp.Union[str, datetime.date, np.datetime64]
YearInitializer = tp.Union[str, datetime.date, np.datetime64]


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

def name_filter(name: tp.Hashable) -> tp.Hashable:
    '''
    For name attributes on containers, only permit recursively hashable objects.
    '''
    try:
        hash(name)
    except TypeError:
        raise TypeError('unhashable name attribute', name)
    return name

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


def resolve_dtype(dt1: np.dtype, dt2: np.dtype) -> np.dtype:
    '''
    Given two dtypes, return a compatible dtype that can hold both contents without truncation.
    '''
    # NOTE: np.dtype(object) == np.object_, so we can return np.object_
    # if the same, return that detype
    if dt1 == dt2:
        return dt1

    # if either is object, we go to object
    if dt1.kind == 'O' or dt2.kind == 'O':
        return DTYPE_OBJECT

    dt1_is_str = dt1.kind in DTYPE_STR_KIND
    dt2_is_str = dt2.kind in DTYPE_STR_KIND
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

    # first pass to determine shape and resolvved type
    arrays_iter = iter(arrays)
    first = next(arrays_iter)

    ndim = first.ndim
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
        dtype: np.dtype,
        shape: tp.Union[int, tp.Tuple[int, ...]],
        fill_value: object) -> np.ndarray:
    '''
    Return a "full" NP array for the given fill_value
    Args:
        dtype: target dtype, which may or may not be possible given the fill_value.
    '''
    dtype = resolve_dtype(dtype, np.array(fill_value).dtype)
    return np.full(shape, fill_value, dtype=dtype)


def dtype_to_na(dtype: DtypeSpecifier) -> tp.Any:
    '''Given a dtype, return an appropriate and compatible null value.
    '''
    if not isinstance(dtype, np.dtype):
        # we permit things like object, float, etc.
        dtype = np.dtype(dtype)

    kind = dtype.kind

    if kind in DTYPE_INT_KIND:
        return 0 # cannot support NaN
    elif kind == 'b':
        return False
    elif kind in DTYPE_NAN_KIND:
        return np.nan
    elif kind == 'O':
        return None
    elif kind in DTYPE_STR_KIND:
        return ''
    elif kind in DTYPE_DATETIME_KIND:
        return NAT
    elif kind in DTYPE_TIMEDELTA_KIND:
        return EMPTY_TIMEDELTA

    raise NotImplementedError('no support for this dtype', kind)


def ufunc_skipna_1d(*,
        array: np.ndarray,
        skipna: bool,
        ufunc: UFunc,
        ufunc_skipna: UFunc) -> np.ndarray:
    '''For one dimensional ufunc array application. Expected to always reduce to single element.
    '''
    if array.dtype.kind == 'O':
        # replace None with nan
        v = array[np.not_equal(array, None)]
        if len(v) == 0: # all values were None
            return np.nan
    elif array.dtype.kind == 'M':
        # dates do not support skipna functions
        return ufunc(array)
    elif array.dtype.kind in DTYPE_STR_KIND and ufunc in UFUNC_AXIS_STR_TO_OBJ:
        v = array.astype(object)
    else:
        v = array

    if skipna:
        return ufunc_skipna(v)
    return ufunc(v)


def ufunc_unique(
        array: np.ndarray,
        axis: tp.Optional[int] = None
        ) -> tp.Union[tp.FrozenSet[tp.Any], np.ndarray]:
    '''
    Extended functionality of the np.unique ufunc, to handle cases of mixed typed objects, where NP will fail in finding unique values for a hetergenous object type.
    '''
    if array.dtype.kind == 'O':
        if axis is None or array.ndim < 2:
            try:
                return np.unique(array)
            except TypeError: # if unorderable types
                pass
            # this may or may not work, depending on contained types
            if array.ndim > 1: # need to flatten
                array_iter = array.flat
            else:
                array_iter = array
            return frozenset(array_iter)

        # ndim == 2 and axis is not None
        # np.unique will give TypeError: The axis argument to unique is not supported for dtype object
        if axis == 0:
            array_iter = array
        else:
            array_iter = array.T
        return frozenset(tuple(x) for x in array_iter)
    # all other types, use the main ufunc
    return np.unique(array, axis=axis)


def roll_1d(array: np.ndarray,
            shift: int
            ) -> np.ndarray:
    '''
    Specialized form of np.roll that, by focusing on the 1D solution, is at least four times faster.
    '''
    size = len(array)
    if size == 0:
        return array.copy()

    shift = shift % size
    if shift == 0:
        return array.copy()

    post = np.empty(size, dtype=array.dtype)
    if shift > 0:
        post[0:shift] = array[-shift:]
        post[shift:] = array[0:-shift]
        return post
    # shift is negative, negate to flip
    post[0:size+shift] = array[-shift:]
    post[size+shift:None] = array[:-shift]
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
        if size == 0: # cannot mod zero
            return array.copy()

        shift = shift % size
        if shift == 0:
            return array.copy()

        if shift > 0:
            post[0:shift, :] = array[-shift:, :]
            post[shift:, :] = array[0:-shift, :]
            return post
        # shift is negative, negate to flip
        post[0:size+shift, :] = array[-shift:, :]
        post[size+shift:None, :] = array[:-shift, :]

    elif axis == 1: # roll columns
        size = array.shape[1]
        if size == 0: # cannot mod zero
            return array.copy()

        shift = shift % size
        if shift == 0:
            return array.copy()

        if shift > 0:
            post[:, 0:shift] = array[:, -shift:]
            post[:, shift:] = array[:, 0:-shift]
            return post
        # shift is negative, negate to flip
        post[:, 0:size+shift] = array[:, -shift:]
        post[:, size+shift:None] = array[:, :-shift]
        return post

    raise NotImplementedError()


#-------------------------------------------------------------------------------
# array constructors

# def resolve_type(
#         value: tp.Any,
#         resolved: tp.Optional[type]=None
#         ) -> tp.Tuple[type, bool]:
#     '''Return a type, suitable for usage as a DtypeSpecifier, that will not truncate when used in array creation.
#     Returns:
#         type, is_tuple
#     '''
#     if resolved == object:
#         # clients should stop iteration once ann object is returned
#         raise RuntimeError('already resolved to object')

#     value_type = type(value)

#     is_tuple = False

#     # normalize NP types to python types
#     if issubclass(value_type, np.integer):
#         value_type = int
#     elif issubclass(value_type, np.floating):
#         value_type = float
#     elif issubclass(value_type, np.complexfloating):
#         value_type = complex
#     elif value_type == tuple:
#         is_tuple = True

#     # anything that gets converted to object
#     if is_tuple:
#         # NOTE: we do not convert other conntainers to object here, as they are not common as elements, and if it is an iterable of set, list, etc, array constructor will treat that argument the same as object
#         return object, is_tuple

#     if resolved is None: # first usage
#         return value_type, is_tuple

#     if value_type == resolved:
#         # fine to return set, list here;
#         return value_type, is_tuple

#     if ((resolved == float and value_type == int)
#             or (resolved == int and value_type == float)
#             ):
#         # if not the same (float or int), promote value of int to float
#         # if value is float and resolved
#         return float, is_tuple

#     # resolved is not None, and this value_type is not equal to the resolved
#     return object, is_tuple


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


def resolve_type_iter(
        values: tp.Iterable[tp.Any],
        sample_size: int = 10,
        ) -> tp.Tuple[DtypeSpecifier, bool, tp.Sequence[tp.Any]]:
    '''
    Determine an appropriate DtypeSpecifier for values in an iterable. This does not try to determine the actual dtype, but instead, if the DtypeSpecifier needs to be object rather than None (which lets NumPy auto detect).

    Args:
        values: can be a generator that will be exhausted in processing; if a generator, a copy will be made and returned as values
        sample_size: number or elements to examine to determine DtypeSpecifier.
    Returns:
        resolved, has_tuple, values
    '''

    is_gen, copy_values = is_gen_copy_values(values)

    if not is_gen:
        values = tp.cast(tp.Sequence[tp.Any], values)
        if len(values) == 0:
            return None, False, values

    v_iter = iter(values)

    if copy_values:
        # will copy in loop below; check for empty iterables and exit early
        try:
            front = next(v_iter)
        except StopIteration:
            # if no values, can return a float-type array
            return None, False, EMPTY_TUPLE

        v_iter = chain((front,), v_iter)
        # do not create list unless we are sure we have more than 1 value
        values_post = []

    resolved = None # None is valid specifier if the type is not ambiguous
    has_tuple = False
    has_str = False
    has_non_str = False

    for i, v in enumerate(v_iter, start=1):
        if copy_values:
            # if a generator, have to make a copy while iterating
            # for array construcdtion, cannot use dictlike, so must convert to list
            values_post.append(v)

        if resolved != object:

            value_type = type(v)

            if value_type == tuple:
                has_tuple = True
            elif value_type == str or value_type == np.str_:
                # must compare to both sring types
                has_str = True
            else:
                has_non_str = True

            if has_tuple or (has_str and has_non_str):
                resolved = object

        else: # resolved is object, can exit once has_tuple is known
            if has_tuple:
                # can end if we have found a tuple
                if copy_values:
                    values_post.extend(v_iter)
                break

        if i >= sample_size:
            if copy_values:
                values_post.extend(v_iter)
            break

    # NOTE: we break before finding a tuple, but our treatment of object types, downstream, will always assign them in the appropriate way
    if copy_values:
        return resolved, has_tuple, values_post

    return resolved, has_tuple, tp.cast(tp.Sequence[tp.Any], values)




def iterable_to_array(
        values: tp.Iterable[tp.Any],
        dtype: DtypeSpecifier=None
        ) -> tp.Tuple[np.ndarray, bool]:
    '''
    Convert an arbitrary Python iterable to a NumPy array without any undesirable type coercion.

    Returns:
        pair of array, Boolean, where the Boolean can be used when necessary to establish uniqueness.
    '''
    if isinstance(values, np.ndarray):
        if dtype is not None and dtype != values.dtype:
            raise RuntimeError('supplied dtype not set on supplied array')
        return values, len(values) <= 1

    # values for construct will only be a copy when necessary in iteration to find type
    if dtype is None:
        # this gives as dtype only None, or object, letting array constructor do the rest
        dtype, has_tuple, values_for_construct = resolve_type_iter(values)

        if len(values_for_construct) == 0:
            return EMPTY_ARRAY, True # no dtype given, so return empty float array

        # dtype_is_object = dtype in DTYPE_SPECIFIERS_OBJECT

    else: # dtype given, do not do full iteration
        is_gen, copy_values = is_gen_copy_values(values)

        if copy_values:
            # we have to realize into sequence for numpy creation
            values_for_construct = tuple(values)
        else:
            values_for_construct = tp.cast(tp.Sequence[tp.Any], values)

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
        # this is the only way to assign from a sequence that contains a tuple; this does not work for dict or set (they must be copied into an iterabel), and is little slower than creating array directly
        v = np.empty(len(values_for_construct), dtype=DTYPE_OBJECT)
        v[NULL_SLICE] = values_for_construct

    elif dtype == int:
        # large python ints can overflow default NumPy int type
        try:
            v = np.array(values_for_construct, dtype=dtype)
        except OverflowError:
            v = np.array(values_for_construct, dtype=DTYPE_OBJECT)
    else:
        # if dtype was None, we might have discovered this was object and but no tuples; faster to do this constructor instead of null slice assignment
        v = np.array(values_for_construct, dtype=dtype)

    v.flags.writeable = False
    return v, is_unique

#-------------------------------------------------------------------------------

def slice_to_ascending_slice(
        key: slice,
        size: int
        ) -> slice:
    '''
    Given a slice, return a slice that, with ascending integers, covers the same values.
    '''
    if key.step is None or key.step > 0:
        return key

    stop = key.start if key.start is None else key.start + 1

    if key.step == -1:
        # if 6, 1, -1, then
        start = key.stop if key.stop is None else key.stop + 1
        return slice(start, stop, 1)

    # if 6, 1, -2: 6, 4, 2; then
    start = next(reversed(range(*key.indices(size))))
    return slice(start, stop, -key.step)

#-------------------------------------------------------------------------------
# dates

_DT64_DAY = np.dtype('datetime64[D]')
_DT64_MONTH = np.dtype('datetime64[M]')
_DT64_YEAR = np.dtype('datetime64[Y]')
_DT64_S = np.dtype('datetime64[s]')
_DT64_MS = np.dtype('datetime64[ms]')

_TD64_DAY = np.timedelta64(1, 'D')
_TD64_MONTH = np.timedelta64(1, 'M')
_TD64_YEAR = np.timedelta64(1, 'Y')
_TD64_S = np.timedelta64(1, 's')
_TD64_MS = np.timedelta64(1, 'ms')

_DT_NOT_FROM_INT = (_DT64_DAY, _DT64_MONTH)

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
                if dtype == _DT64_YEAR:
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
                raise RuntimeError('not supported dtype', dt, dtype)
    return dt

def to_timedelta64(value: datetime.timedelta) -> np.timedelta64:
    '''
    Convert a datetime.timedelta into a NumPy timedelta64. This approach is better than using np.timedelta64(value), as that reduces all values to microseconds.
    '''
    return reduce(operator.add,
        (np.timedelta64(getattr(value, attr), code) for attr, code in TIME_DELTA_ATTR_MAP if getattr(value, attr) > 0))

def _slice_to_datetime_slice_args(key: slice, dtype: tp.Optional[np.dtype] = None) -> tp.Iterator[tp.Optional[np.datetime64]]:
    '''
    Given a slice representing a datetime region, convert to arguments for a new slice, possibly using the appropriate dtype for conversion.
    '''
    for attr in SLICE_ATTRS:
        value = getattr(key, attr)
        if value is None:
            yield None
        else:
            yield to_datetime64(value, dtype=dtype)

def key_to_datetime_key(
        key: GetItemKeyType,
        dtype: np.dtype = np.datetime64) -> GetItemKeyType:
    '''
    Given an get item key for a Date index, convert it to np.datetime64 representation.
    '''
    if isinstance(key, slice):
        return slice(*_slice_to_datetime_slice_args(key, dtype=dtype))

    if isinstance(key, np.datetime64):
        return key

    if isinstance(key, str):
        return to_datetime64(key, dtype=dtype)

    if isinstance(key, np.ndarray):
        if key.dtype.kind == 'b' or key.dtype.kind == 'M':
            return key
        return key.astype(dtype)

    if hasattr(key, '__len__'):
        # use dtype via array constructor to determine type; or just use datetime64 to parse to the passed-in representationn
        return np.array(key, dtype=dtype)

    if hasattr(key, '__next__'): # a generator-like
        return np.array(tuple(tp.cast(tp.Iterator[tp.Any], key)), dtype=dtype)

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
    '''Return Boolean if value is an NA.
    '''

    try:
        return tp.cast(bool, np.isnan(value))
    except TypeError:
        pass
    try:
        return tp.cast(bool, np.isnat(value))
    except TypeError:
        pass

    return value is None


def isna_array(array: np.ndarray) -> np.ndarray:
    '''Given an np.ndarray, return a bolean array setting True for missing values.

    Note: the returned array is not made immutable.
    '''
    kind = array.dtype.kind
    # matches all floating point types
    if kind in DTYPE_NAN_KIND:
        return np.isnan(array)
    elif kind in DTYPE_NAT_KIND:
        return np.isnat(array)
    # match everything that is not an object; options are: biufcmMOSUV
    elif kind != 'O':
        return np.full(array.shape, False, dtype=bool)
    # only check for None if we have an object type
    return np.not_equal(array, array) | np.equal(array, None)

    # try: # this will only work for arrays that do not have strings
    #     # astype: None gets converted to nan if possible
    #     # cannot use can_cast to reliabily identify arrays with non-float-castable elements
    #     return np.isnan(array.astype(float))
    # except ValueError:
    #     # this Exception means there was a character or something not castable to float
    #     # this is a big perforamnce hit; problem is cannot find np.nan in numpy object array
    #     if array.ndim == 1:
    #         return np.fromiter((x is None or x is np.nan for x in array),
    #                 count=array.size,
    #                 dtype=bool)

    #     return np.fromiter((x is None or x is np.nan for x in array.flat),
    #             count=array.size,
    #             dtype=bool).reshape(array.shape)



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

def array_to_duplicated(
        array: np.ndarray,
        axis: int = 0,
        exclude_first: bool = False,
        exclude_last: bool = False) -> np.ndarray:
    '''Given a numpy array (1D or 2D), return a Boolean array along the specified axis that shows which values are duplicated. By default, all duplicates are indicated. For 2d arrays, axis 0 compares rows and returns a row-length Boolean array; axis 1 compares columns and returns a column-length Boolean array.

    Args:
        exclude_first: Mark as True all duplicates except the first encountared.
        exclude_last: Mark as True all duplicates except the last encountared.
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

def array_shift(
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


def array_set_ufunc_many(
        arrays: tp.Iterable[np.ndarray],
        union: bool = False) -> np.ndarray:
    '''
    Iteratively apply a set operation unfunc to a arrays; if all are equal, no operation is performed and order is retained.

    Args:
        union: if True, a union is taken, else, an intersection.
    '''
    arrays = iter(arrays)
    result = next(arrays)

    if result.ndim == 1:
        ufunc = np.union1d if union else np.intersect1d
        ndim = 1
    else: # ndim == 2
        ufunc = union2d if union else intersect2d
        ndim = 2

    for array in arrays:
        if array.ndim != ndim:
            raise RuntimeError('arrays do not all have the same ndim')
        # is the new array different
        if ndim == 1:
            # if lengths are not comparable, or comparable and the values are not the same
            if len(array) != len(result) or (array != result).any():
                result = ufunc(result, array)
            # otherwise, arrays are identical and can skip ufunc application
        else:
            result = ufunc(result, array)

        if not union and len(result) == 0:
            # short circuit intersection that results in no common values
            return result

    return result

def array2d_to_tuples(array: np.ndarray) -> tp.Iterator[tp.Tuple[tp.Any, ...]]:
    for row in array: # assuming 2d
        yield tuple(row)

#-------------------------------------------------------------------------------
# extension to union and intersection handling

def union1d(array: np.ndarray, other: np.ndarray) -> np.ndarray:

    set_compare = False
    array_is_str = array.dtype.kind in DTYPE_STR_KIND
    other_is_str = other.dtype.kind in DTYPE_STR_KIND

    if array_is_str ^ other_is_str:
        # if only one is string
        set_compare = True

    if set_compare or array.dtype.kind == 'O' or other.dtype.kind == 'O':
        result = set(array) | set(other)
        dtype = resolve_dtype(array.dtype, other.dtype)
        v, _ = iterable_to_array(result, dtype)
        return v

    return np.union1d(array, other)


def intersect1d(
        array: np.ndarray,
        other: np.ndarray) -> np.ndarray:
    '''
    Extend ufunc version to handle cases where types cannot be sorted.
    '''
    set_compare = False
    array_is_str = array.dtype.kind in DTYPE_STR_KIND
    other_is_str = other.dtype.kind in DTYPE_STR_KIND

    if array_is_str ^ other_is_str:
        # if only one is string
        set_compare = True

    if set_compare or array.dtype.kind == 'O' or other.dtype.kind == 'O':
        # if a 2D array gets here, a hashability error will be raised
        result = set(array) & set(other)
        dtype = resolve_dtype(array.dtype, other.dtype)
        v, _ = iterable_to_array(result, dtype)
        return v

    return np.intersect1d(array, other)

def set_ufunc2d(
        func: tp.Callable[[np.ndarray, np.ndarray], np.ndarray],
        array: np.ndarray,
        other: np.ndarray) -> np.ndarray:
    '''
    Given a 1d set operation, convert to structured array, perform operation, then restore original shape.
    '''
    if array.dtype.kind == 'O' or other.dtype.kind == 'O':
        if array.ndim == 1:
            array_set = set(array)
        else:
            array_set = set(tuple(row) for row in array)
        if other.ndim == 1:
            other_set = set(other)
        else:
            other_set = set(tuple(row) for row in other)

        if func is np.union1d:
            result = array_set | other_set
        elif func is np.intersect1d:
            result = array_set & other_set
        else:
            raise NotImplementedError('unexpected func', func)
        # sort so as to duplicate results from NP functions
        # NOTE: this sort may not always be necssary
        return np.array(sorted(result), dtype=object)

    assert array.shape[1] == other.shape[1]
    # this does will work if dyptes are differently sized strings, such as U2 and U3
    dtype = resolve_dtype(array.dtype, other.dtype)
    if array.dtype != dtype:
        array = array.astype(dtype)
    if other.dtype != dtype:
        other = other.astype(dtype)

    width = array.shape[1]
    array_view = array.view([('', array.dtype)] * width)
    other_view = other.view([('', other.dtype)] * width)
    return func(array_view, other_view).view(dtype).reshape(-1, width)


def intersect2d(array: np.ndarray,
        other: np.ndarray) -> np.ndarray:
    return set_ufunc2d(np.intersect1d, array, other)

def union2d(array: np.ndarray, other: np.ndarray) -> np.ndarray:
    return set_ufunc2d(np.union1d, array, other)

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
        length: the maximum lengh in the target array
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
        # NOTE: usage of None here is awkward; try to use zero
        target_slices = (
                slice((start+1 if start is not None else start), stop)
                for start, stop in
                zip(chain((None,), target_index[:-1]), target_index)
                )

    for target_slice, value in zip(target_slices, target_values):

        # all conditions that are noop slices
        if target_slice.start == target_slice.stop:
            continue
        elif (directional_forward
                and target_slice.start is not None
                and target_slice.start >= length):
            continue
        elif (not directional_forward
                and target_slice.start is None
                and target_slice.stop == 0):
            continue
        elif target_slice.stop is None:
            # stop value should never be None
            raise NotImplementedError('unexpected slice', target_slice)

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

def _read_url(fp: str) -> str:
    with request.urlopen(fp) as response:
        return tp.cast(str, response.read().decode('utf-8'))


def write_optional_file(
        content: str,
        fp: tp.Optional[FilePathOrFileLike] = None,
        ) -> tp.Optional[str]:

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
    return tp.cast(str, fp)

#-------------------------------------------------------------------------------

# Options:
# 1. store GetItem instance on object: created at init regardless of use, storage on container
# 2. use descripter to return a GetItem instance lazily created and stored on the container: no init creation, incurs extra branching on call, stored once used.

# NOTE: object creation can be faster than getattr and branching
# In [3]: %timeit sf.GetItem('')
# 249 ns ± 3.1 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

# In [4]: x = object()

# In [5]: %timeit getattr(x, '', None); 0 if x else 1
# 316 ns ± 1.29 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)


TContainer = tp.TypeVar('TContainer', 'Index', 'Series', 'Frame', 'TypeBlocks')
GetItemFunc = tp.TypeVar('GetItemFunc', bound=tp.Callable[[GetItemKeyType], TContainer])

#TODO: rename InterfaceGetItem
class GetItem(tp.Generic[TContainer]):

    __slots__ = ('_func',)

    def __init__(self, func: tp.Callable[[GetItemKeyType], TContainer]) -> None:
        self._func: tp.Callable[[GetItemKeyType], TContainer] = func

    def __getitem__(self, key: GetItemKeyType) -> TContainer:
        return self._func(key)

#-------------------------------------------------------------------------------

class InterfaceSelection1D(tp.Generic[TContainer]):
    '''An instance to serve as an interface to all of iloc and loc
    '''

    __slots__ = (
            '_func_iloc',
            '_func_loc',
            )

    def __init__(self, *,
            func_iloc: GetItemFunc,
            func_loc: GetItemFunc) -> None:

        self._func_iloc = func_iloc
        self._func_loc = func_loc

    @property
    def iloc(self) -> GetItem[TContainer]:
        return GetItem(self._func_iloc)

    @property
    def loc(self) -> GetItem[TContainer]:
        return GetItem(self._func_loc)


#-------------------------------------------------------------------------------

class InterfaceSelection2D(tp.Generic[TContainer]):
    '''An instance to serve as an interface to all of iloc, loc, and __getitem__ extractors.
    '''

    __slots__ = (
            '_func_iloc',
            '_func_loc',
            '_func_getitem'
            )

    def __init__(self, *,
            func_iloc: GetItemFunc,
            func_loc: GetItemFunc,
            func_getitem: GetItemFunc) -> None:

        self._func_iloc = func_iloc
        self._func_loc = func_loc
        self._func_getitem = func_getitem

    def __getitem__(self, key: GetItemKeyType) -> tp.Any:
        return self._func_getitem(key)

    @property
    def iloc(self) -> GetItem[TContainer]:
        return GetItem(self._func_iloc)

    @property
    def loc(self) -> GetItem[TContainer]:
        return GetItem(self._func_loc)

#-------------------------------------------------------------------------------

class InterfaceAsType:
    '''An instance to serve as an interface to __getitem__ extractors.
    '''

    __slots__ = ('_func_getitem',)

    def __init__(self, func_getitem: tp.Callable[[GetItemKeyType], 'FrameAsType']) -> None:
        '''
        Args:
            _func_getitem: a callable that expects a _func_getitem key and returns a FrameAsType interface; for example, Frame._extract_getitem_astype.
        '''
        self._func_getitem = func_getitem

    def __getitem__(self, key: GetItemKeyType) -> 'FrameAsType':
        return self._func_getitem(key)

    def __call__(self, dtype: np.dtype) -> 'Frame':
        return self._func_getitem(NULL_SLICE)(dtype)


#-------------------------------------------------------------------------------
# index utilities


class IndexCorrespondence:
    '''
    All iloc data necessary for reindexing.
    '''

    __slots__ = (
            'has_common',
            'is_subset',
            'iloc_src',
            'iloc_dst',
            'size',
            )

    has_common: bool
    is_subset: bool
    iloc_src: GetItemKeyType
    iloc_dst: GetItemKeyType
    size: int

    @classmethod
    def from_correspondence(cls,
            src_index: 'Index',
            dst_index: 'Index') -> 'IndexCorrespondence':
        '''
        Return an IndexCorrespondence instance from the correspondence of two Index or IndexHierarchy objects.
        '''
        mixed_depth = False
        if src_index.depth == dst_index.depth:
            depth = src_index.depth
        else:
            # if dimensions are mixed, the only way there can be a match is if the 1D index is of object type (so it can hold a tuple); otherwise, there can be no matches;
            if src_index.depth == 1 and src_index.values.dtype.kind == 'O':
                depth = dst_index.depth
                mixed_depth = True
            elif dst_index.depth == 1 and dst_index.values.dtype.kind == 'O':
                depth = src_index.depth
                mixed_depth = True
            else:
                depth = 0

        # need to use lower level array methods go get intersection, rather than Index methods, as need arrays, not Index objects
        if depth == 1:
            # NOTE: this can fail in some cases: comparing two object arrays with NaNs and strings.
            common_labels = intersect1d(src_index.values, dst_index.values)
            has_common = len(common_labels) > 0
            assert not mixed_depth
        elif depth > 1:
            # if either values arrays are object, we have to covert all values to tuples
            common_labels = intersect2d(src_index.values, dst_index.values)
            if mixed_depth:
                # when mixed, on the 1D index we have to use loc_to_iloc with tuples
                common_labels = list(array2d_to_tuples(common_labels))
            has_common = len(common_labels) > 0
        else:
            has_common = False

        size = len(dst_index.values)

        # either a reordering or a subset
        if has_common:

            if len(common_labels) == len(dst_index):
                # use new index to retain order
                iloc_src = src_index.loc_to_iloc(dst_index.values)
                iloc_dst = np.arange(size)
                return cls(has_common=has_common,
                        is_subset=True,
                        iloc_src=iloc_src,
                        iloc_dst=iloc_dst,
                        size=size
                        )

            # these will be equal sized
            iloc_src = src_index.loc_to_iloc(common_labels)
            iloc_dst = dst_index.loc_to_iloc(common_labels)

            # if iloc_src.dtype != int:
            #     import ipdb; ipdb.set_trace()
            return cls(has_common=has_common,
                    is_subset=False,
                    iloc_src=iloc_src,
                    iloc_dst=iloc_dst,
                    size=size)

        return cls(has_common=has_common,
                is_subset=False,
                iloc_src=None,
                iloc_dst=None,
                size=size)


    def __init__(self,
            has_common: bool,
            is_subset: bool,
            iloc_src: GetItemKeyType,
            iloc_dst: GetItemKeyType,
            size: int) -> None:
        '''
        Args:
            has_common: True if any of the indices align
            is_subset: True if the destination is a reordering or subset
            iloc_src: An iterable of iloc values to be taken from the source
            iloc_dst: An iterable of iloc values to be written to
            size: The size of the destination.
        '''
        self.has_common = has_common
        self.is_subset = is_subset
        self.iloc_src = iloc_src
        self.iloc_dst = iloc_dst
        self.size = size

    def iloc_src_fancy(self) -> tp.List[tp.List[int]]:
        '''
        Convert an iloc iterable of integers into one that is combitable with fancy indexing.
        '''
        return [[x] for x in tp.cast(tp.Iterable[int], self.iloc_src)]




#-------------------------------------------------------------------------------
