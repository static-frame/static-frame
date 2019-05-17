import sys
import typing as tp
import os

from collections import OrderedDict
from collections import abc
from itertools import chain
from io import StringIO
from io import BytesIO
import datetime
from urllib import request
import tempfile


import numpy as np



# min/max fail on object arrays
# handle nan in object blocks with skipna processing on ufuncs
# allow columns asignment with getitem on FrameGO from an integer
# bloc() to select / assign into an 2D array with Boolean mask selection
# roll() on TypeBlocks (can be used in duplicate discovery on blocks)


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

DEFAULT_SORT_KIND = 'mergesort'
_DEFAULT_STABLE_SORT_KIND = 'mergesort'
_DTYPE_STR_KIND = ('U', 'S') # S is np.bytes_
_DTYPE_INT_KIND = ('i', 'u') # signed and unsigned
DTYPE_OBJECT = np.dtype(object)

NULL_SLICE = slice(None)
_UNIT_SLICE = slice(0, 1)
SLICE_STOP_ATTR = 'stop'
SLICE_STEP_ATTR = 'step'
SLICE_ATTRS = ('start', SLICE_STOP_ATTR, SLICE_STEP_ATTR)
STATIC_ATTR = 'STATIC'

# defaults to float64
EMPTY_ARRAY = np.array((), dtype=None)
EMPTY_ARRAY.flags.writeable = False

_DICT_STABLE = sys.version_info >= (3, 6)



#-------------------------------------------------------------------------------
# utility

INT_TYPES = (int, np.int_)
_BOOL_TYPES = (bool, np.bool_)

# for getitem / loc selection
KEY_ITERABLE_TYPES = (list, np.ndarray)

# types of keys that return muultiple items, even if the selection reduces to 1
KEY_MULTIPLE_TYPES = (slice, list, np.ndarray)

# for type hinting
# keys once dimension has been isolated
GetItemKeyType = tp.Union[
        int, slice, list, None, 'Index', 'Series', np.ndarray]

# keys that might include a multiple dimensions speciation; tuple is used to identify compound extraction
GetItemKeyTypeCompound = tp.Union[
        tuple, int, slice, list, None, 'Index', 'Series', np.ndarray]

CallableOrMapping = tp.Union[tp.Callable, tp.Mapping]
KeyOrKeys = tp.Union[tp.Hashable, tp.Iterable[tp.Hashable]]
FilePathOrFileLike = tp.Union[str, StringIO, BytesIO]

DtypeSpecifier = tp.Optional[tp.Union[str, np.dtype, type]]

# support an iterable of specifiers, or mapping based on column names
DtypesSpecifier = tp.Optional[
        tp.Union[tp.Iterable[DtypeSpecifier], tp.Dict[tp.Hashable, DtypeSpecifier]]]

DepthLevelSpecifier = tp.Union[int, tp.Iterable[int]]

CallableToIterType = tp.Callable[[], tp.Iterable[tp.Any]]

IndexSpecifier = tp.Union[int, str]
IndexInitializer = tp.Union[
        tp.Iterable[tp.Hashable],
        tp.Generator[tp.Hashable, None, None]]

SeriesInitializer = tp.Union[
        tp.Iterable[tp.Any],
        np.ndarray,
        tp.Mapping[tp.Hashable, tp.Any],
        int, float, str, bool]

FrameInitializer = tp.Union[
        tp.Iterable[tp.Iterable[tp.Any]],
        np.ndarray,
        tp.Mapping[tp.Hashable, tp.Iterable[tp.Any]]
        ]

DateInitializer = tp.Union[str, datetime.date, np.datetime64]
YearMonthInitializer = tp.Union[str, datetime.date, np.datetime64]
YearInitializer = tp.Union[str, datetime.date, np.datetime64]

def mloc(array: np.ndarray) -> int:
    '''Return the memory location of an array.
    '''
    return array.__array_interface__['data'][0]


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
        center_sentinel: tp.Any) -> tp.Generator:
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


def _resolve_dtype(dt1: np.dtype, dt2: np.dtype) -> np.dtype:
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

    dt1_is_str = dt1.kind in _DTYPE_STR_KIND
    dt2_is_str = dt2.kind in _DTYPE_STR_KIND

    # if both are string or string-like, we can use result type to get the longest string
    if dt1_is_str and dt2_is_str:
        return np.result_type(dt1, dt2)

    dt1_is_bool = dt1.type is np.bool_
    dt2_is_bool = dt2.type is np.bool_

    # if any one is a string or a bool, we have to go to object; result_type gives a string in mixed cases
    if dt1_is_str or dt2_is_str or dt1_is_bool or dt2_is_bool:
        return DTYPE_OBJECT

    # if not a string or an object, can use result type
    return np.result_type(dt1, dt2)

def resolve_dtype_iter(dtypes: tp.Iterable[np.dtype]):
    '''Given an iterable of dtypes, do pairwise comparisons to determine compatible overall type. Once we get to object we can stop checking and return object
    '''
    dtypes = iter(dtypes)
    dt_resolve = next(dtypes)
    for dt in dtypes:
        dt_resolve = _resolve_dtype(dt_resolve, dt)
        if dt_resolve == DTYPE_OBJECT:
            return dt_resolve
    return dt_resolve

def concat_resolved(arrays: tp.Iterable[np.ndarray],
        axis=0):
    '''
    Concatenation that uses resolved dtypes to avoid truncation.

    Axis 0 stacks rows (extends columns); axis 1 stacks columns (extends rows).

    Now shape manipulation will happen, so it is always assumed that all dimensionalities will be common.
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
            dt_resolve = _resolve_dtype(array.dtype, dt_resolve)
        shape[axis] += array.shape[axis]

    out = np.empty(shape=shape, dtype=dt_resolve)
    np.concatenate(arrays, out=out, axis=axis)
    out.flags.writeable = False
    return out


def full_for_fill(
        dtype: np.dtype,
        shape: tp.Union[int, tp.Tuple[int, int]],
        fill_value: object) -> np.ndarray:
    '''
    Return a "full" NP array for the given fill_value
    Args:
        dtype: target dtype, which may or may not be possible given the fill_value.
    '''
    dtype = _resolve_dtype(dtype, np.array(fill_value).dtype)
    return np.full(shape, fill_value, dtype=dtype)


def _dtype_to_na(dtype: np.dtype):
    '''Given a dtype, return an appropriate and compatible null value.
    '''
    if not isinstance(dtype, np.dtype):
        # we permit things like object, float, etc.
        dtype = np.dtype(dtype)

    if dtype.kind in _DTYPE_INT_KIND:
        return 0 # cannot support NaN
    elif dtype.kind == 'b':
        return False
    elif dtype.kind == 'f':
        return np.nan
    elif dtype.kind == 'O':
        return None
    elif dtype.kind in _DTYPE_STR_KIND:
        return ''
    raise NotImplementedError('no support for this dtype', dtype.kind)

# ufunc functions that will not work with _DTYPE_STR_KIND, but do work if converted to object arrays; see _UFUNC_AXIS_SKIPNA for the matching functions
_UFUNC_AXIS_STR_TO_OBJ = {np.min, np.max, np.sum}

def ufunc_skipna_1d(*, array, skipna, ufunc, ufunc_skipna):
    '''For one dimensional ufunc array application. Expected to always reduce to single element.
    '''
    # if len(array) == 0:
    #     # np returns 0 for sum of an empty array
    #     return 0.0
    if array.dtype.kind == 'O':
        # replace None with nan
        v = array[np.not_equal(array, None)]
        if len(v) == 0: # all values were None
            return np.nan
    elif array.dtype.kind == 'M':
        # dates do not support skipna functions
        return ufunc(array)
    elif array.dtype.kind in _DTYPE_STR_KIND and ufunc in _UFUNC_AXIS_STR_TO_OBJ:
        v = array.astype(object)
    else:
        v = array

    if skipna:
        return ufunc_skipna(v)
    return ufunc(v)


def ufunc_unique(array: np.ndarray,
        axis: tp.Optional[int] = None
        ) -> tp.Union[frozenset, np.ndarray]:
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
            # return np.array(sorted(frozenset(array_iter)), dtype=object)

        # ndim == 2 and axis is not None
        # the normal np.unique will give TypeError: The axis argument to unique is not supported for dtype object
        if axis == 0:
            array_iter = array
        else:
            array_iter = array.T
        return frozenset(tuple(x) for x in array_iter)
    # all other types, use the main ufunc
    return np.unique(array, axis=axis)


def iterable_to_array(other) -> tp.Tuple[np.ndarray, bool]:
    '''Utility method to take arbitary, heterogenous typed iterables and realize them as an NP array. As this is used in isin() functions, identifying cases where we can assume that this array has only unique values is useful. That is done here by type, where Set-like types are marked as assume_unique.
    '''
    v_iter = None

    if isinstance(other, np.ndarray):
        v = other # not making this immutable; clients can decide
        # could look if unique, not sure if too much overhead
        assume_unique = False

    else: # must determine if we need an object type
        # we can use assume_unique if `other` is a set, keys view, frozen set, another index, as our _labels is always unique
        if isinstance(other, (abc.Set, dict)): # mathches set, frozenset, keysview
            v_iter = iter(other)
            assume_unique = True
        else: # generators and other iteraables, lists, etc
            v_iter = iter(other)
            assume_unique = False

        # must determine if we have heterogenous types, as if we have string and float, for example, all numbers become quoted
        try:
            x = next(v_iter)
        except StopIteration:
            return EMPTY_ARRAY, True

        dtype = type(x)
        array_values = [x]
        for x in v_iter:
            array_values.append(x)
            # if it is not the same as previous, return object
            if dtype != object and dtype != type(x):
                dtype = object

        v = np.array(array_values, dtype=dtype)
        v.flags.writeable = False

    if len(v) == 1:
        assume_unique = True

    return v, assume_unique


def _slice_to_ascending_slice(key: slice, size: int) -> slice:
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

def _to_datetime64(
        value: DateInitializer,
        dtype: tp.Optional[np.dtype] = None
        ) -> np.datetime64:
    '''
    Convert a value ot a datetime64; this must be a datetime64 so as to be hashable.
    '''
    # for now, only support creating from a string, as creation from integers is based on offset from epoch
    if not isinstance(value, np.datetime64):
        if dtype is None:
            # let this constructor figure it out
            dt = np.datetime64(value)
        else: # assume value is single value;
            # note that integers will be converted to units from epoch
            if isinstance(value, int):
                if dtype == _DT64_YEAR:
                    # convert to string as that is likely what is wanted
                    value = str(value)
                elif dtype in _DT_NOT_FROM_INT:
                    raise RuntimeError('attempting to create {} from an integer, which is generally desired as the result will be offset from the epoch.'.format(dtype))
            # cannot use the datetime directly
            if dtype != np.datetime64:
                dt = np.datetime64(value, np.datetime_data(dtype)[0])
            else: # cannot use a generic datetime type
                dt = np.datetime64(value)
    else: # if a dtype was explicitly given, check it
        if dtype and dt.dtype != dtype:
            raise RuntimeError('not supported dtype', dt, dtype)
    return dt

def _slice_to_datetime_slice_args(key, dtype=None):
    for attr in SLICE_ATTRS:
        value = getattr(key, attr)
        if value is None:
            yield None
        else:
            yield _to_datetime64(value, dtype=dtype)

def key_to_datetime_key(
        key: GetItemKeyType,
        dtype=np.datetime64) -> GetItemKeyType:
    '''
    Given an get item key for a Date index, convert it to np.datetime64 representation.
    '''
    if isinstance(key, slice):
        return slice(*_slice_to_datetime_slice_args(key, dtype=dtype))

    if isinstance(key, np.datetime64):
        return key

    if isinstance(key, str):
        return _to_datetime64(key, dtype=dtype)

    if isinstance(key, np.ndarray):
        if key.dtype.kind == 'b' or key.dtype.kind == 'M':
            return key
        return key.astype(dtype)

    if hasattr(key, '__len__'):
        # use array constructor to determine type
        return np.array(key, dtype=dtype)

    if hasattr(key, '__next__'): # a generator-like
        return np.array(tuple(key), dtype=dtype)

    # for now, return key unaltered
    return key

#-------------------------------------------------------------------------------


def _dict_to_sorted_items(
            mapping: tp.Dict) -> tp.Generator[
            tp.Tuple[tp.Hashable, tp.Any], None, None]:
    '''
    Convert a dict into two arrays. Note that sorting is only necessary in Python 3.5, and should not be done if an ordered dict
    '''
    if isinstance(mapping, OrderedDict) or _DICT_STABLE:
        # cannot use fromiter as do not know type
        keys = mapping.keys()
    else:
        keys = sorted(mapping.keys())
    for k in keys:
        yield k, mapping[k]


def _array_to_groups_and_locations(
        array: np.ndarray,
        unique_axis: int = 0):
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

def _isna(array: np.ndarray) -> np.ndarray:
    '''Utility function that, given an np.ndarray, returns a bolean arrea setting True nulls. Note: the returned array is not made immutable
    '''
    # matches all floating point types
    if array.dtype.kind == 'f':
        return np.isnan(array)
    # match everything that is not an object; options are: biufcmMOSUV
    elif array.dtype.kind != 'O':
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


def _array_to_duplicated(
        array: np.ndarray,
        axis: int = 0,
        exclude_first=False,
        exclude_last=False):
    '''Given a numpy array, return a Boolean array along the specified axis that shows which values are duplicated. By default, all duplicates are indicated. For 2d arrays, axis 0 compares rows and returns a row-length Boolean array; axis 1 compares columns and returns a column-length Boolean array.

    Args:
        exclude_first: Mark as True all duplicates except the first encountared.
        exclude_last: Mark as True all duplicates except the last encountared.
    '''
    # based in part on https://stackoverflow.com/questions/11528078/determining-duplicate-values-in-an-array
    # https://stackoverflow.com/a/43033882/388739
    # indices to sort and sorted array
    # a right roll on the sorted array, comparing to the original sorted array. creates a boolean array, with all non-first duplicates marked as True

    if array.ndim == 1:
        o_idx = np.argsort(array, axis=None, kind=_DEFAULT_STABLE_SORT_KIND)
        array_sorted = array[o_idx]
        opposite_axis = 0
        f_flags = array_sorted == np.roll(array_sorted, 1)

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
            raise NotImplementedError('no handling for axis')
        opposite_axis = int(not bool(axis))
        # rolling axis 1 rotates columns; roll axis 0 rotates rows
        match = array_sorted == np.roll(array_sorted, 1, axis=axis)
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
        l_flags = np.roll(f_flags, -1)

        if not exclude_first and exclude_last:
            dupes = l_flags
        elif not exclude_first and not exclude_last:
            # all duplicates is the union.
            dupes = f_flags | l_flags
        else:
            # all non-first, non-last duplicates is the intersection.
            dupes = f_flags & l_flags

    # undo the sort: get the indices to extract Booleans from dupes; in some cases r_idx is the same as o_idx, but not all
    r_idx = np.argsort(o_idx, axis=None, kind=_DEFAULT_STABLE_SORT_KIND)
    return dupes[r_idx]

def array_shift(array: np.ndarray,
        shift: int,
        axis: int, # 0 is rows, 1 is columns
        wrap: bool,
        fill_value=np.nan) -> np.ndarray:
    '''
    Apply an np-style roll to an array; if wrap is False, fill values out-shifted values with fill_value.

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
        # standard np roll works fine
        return np.roll(array, shift_mod, axis=axis)

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
        union: bool = False):
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

def array2d_to_tuples(array: np.ndarray) -> tp.Generator[tp.Tuple, None, None]:
    for row in array: # assuming 2d
        yield tuple(row)

#-------------------------------------------------------------------------------
# extension to union and intersection handling

def _ufunc2d(
        func: tp.Callable[[np.ndarray], np.ndarray],
        array: np.ndarray,
        other: np.ndarray):
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
            raise Exception('unexpected func', func)
        # sort so as to duplicate results from NP functions
        # NOTE: this sort may not always be necssary
        return np.array(sorted(result), dtype=object)

    assert array.shape[1] == other.shape[1]
    # this does will work if dyptes are differently sized strings, such as U2 and U3
    dtype = _resolve_dtype(array.dtype, other.dtype)
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
    return _ufunc2d(np.intersect1d, array, other)

def union2d(array, other) -> np.ndarray:
    return _ufunc2d(np.union1d, array, other)

def intersect1d(array: np.ndarray,
        other: np.ndarray) -> np.ndarray:
    '''
    Extend ufunc version to handle cases where types cannot be sorted.
    '''
    try:
        return np.intersect1d(array, other)
    except TypeError:
        result = set(array) & set(other)

    dtype = _resolve_dtype(array.dtype, other.dtype)
    if dtype.kind == 'O':
        # np fromiter does not work with object types
        return np.array(tuple(result), dtype=dtype)
    return np.fromiter(result, count=len(result), dtype=dtype)



#-------------------------------------------------------------------------------
# URL handling, file downloading, file writing

def _read_url(fp: str):
    with request.urlopen(fp) as response:
        return response.read().decode('utf-8')


def write_optional_file(
        content: str,
        fp: tp.Optional[FilePathOrFileLike] = None,
        ):

    fd = f = None
    if not fp: # get a temp file
        fd, fp = tempfile.mkstemp(suffix='.html', text=True)
    elif isinstance(fp, StringIO):
        f = fp
        fp = None
    # nothing to do if we have an fp

    if f is None: # do not have a file object
        try:
            with open(fp, 'w') as f:
                f.write(content)
        finally:
            if fd is not None:
                os.close(fd)
    else: # string IO
        f.write(content)
        f.seek(0)
    return fp

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


TContainer = tp.TypeVar('Container', 'Index', 'Series', 'Frame')

#TODO: rename InterfaceGetItem
class GetItem:
    __slots__ = ('_func',)

    def __init__(self,
            func: tp.Callable[[GetItemKeyType], TContainer]
            ) -> None:
        self._func = func

    def __getitem__(self, key: GetItemKeyType) -> TContainer:
        return self._func(key)

#-------------------------------------------------------------------------------

class InterfaceSelection1D:
    '''An instance to serve as an interface to all of iloc and loc
    '''

    __slots__ = (
            '_func_iloc',
            '_func_loc',
            )

    def __init__(self, *,
            func_iloc: str,
            func_loc: str) -> None:

        self._func_iloc = func_iloc
        self._func_loc = func_loc

    @property
    def iloc(self) -> GetItem:
        return GetItem(self._func_iloc)

    @property
    def loc(self) -> GetItem:
        return GetItem(self._func_loc)


#-------------------------------------------------------------------------------

class InterfaceSelection2D:
    '''An instance to serve as an interface to all of iloc, loc, and __getitem__ extractors.
    '''

    __slots__ = (
            '_func_iloc',
            '_func_loc',
            '_func_getitem'
            )

    def __init__(self, *,
            func_iloc: str,
            func_loc: str,
            func_getitem: str) -> None:

        self._func_iloc = func_iloc
        self._func_loc = func_loc
        self._func_getitem = func_getitem

    def __getitem__(self, key):
        return self._func_getitem(key)

    @property
    def iloc(self) -> GetItem:
        return GetItem(self._func_iloc)

    @property
    def loc(self) -> GetItem:
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

    def __getitem__(self, key) -> 'FrameAsType':
        return self._func_getitem(key)

    def __call__(self, dtype):
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
            iloc_src: tp.Iterable[int],
            iloc_dst: tp.Iterable[int],
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

    def iloc_src_fancy(self):
        '''
        Convert an iloc iterable of integers into one that is combitable with fancy indexing.
        '''
        return [[x] for x in self.iloc_src]





#-------------------------------------------------------------------------------
