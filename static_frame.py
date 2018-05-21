from types import GeneratorType
import typing as tp
import sys
import json
import os
import csv


from collections import OrderedDict
from collections import namedtuple
from collections import abc

from itertools import chain
from itertools import zip_longest
from itertools import product
from functools import wraps
from functools import partial
from itertools import repeat
from enum import Enum
from io import StringIO
from io import BytesIO

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

import operator as operator_mod

import numpy as np
from numpy.ma import MaskedArray


__version__ = '0.1.4'

_module = sys.modules[__name__]

# target features for 0.1a1 release

# handle nan in object blocks with skipna processing on ufuncs
# allow columns asignment with getitem on FrameGO from an integer
# bloc() to select / assign into an 2D array with Boolean mask selection
# read_csv needs nan types; empty strings might need to be loaded as nans


# future features
# drop on TypeBlocks (if needed, update set_index to use); drop on Frame, Series
# Series in from_concat
# roll() on TypeBlocks (can be used in duplicate discovery on blocks)
# shift as non-wrapping roll
# astype: TypeBlocks, Series, Frame
#   Frame.astype[a:b](int)

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

_DEFAULT_SORT_KIND = 'mergesort'
_DEFAULT_STABLE_SORT_KIND = 'mergesort'
_DTYPE_STR_KIND = ('U', 'S') # S is np.bytes_
_DTYPE_INT_KIND = ('i', 'u') # signed and unsigned

_UFUNC_UNARY_OPERATORS = (
        '__pos__',
        '__neg__',
        '__abs__',
        '__invert__')

_UFUNC_BINARY_OPERATORS = (
        '__add__',
        '__sub__',
        '__mul__',
        '__matmul__',
        '__truediv__',
        '__floordiv__',
        '__mod__',
        #'__divmod__', this returns two np.arrays when called on an np array
        '__pow__',
        '__lshift__',
        '__rshift__',
        '__and__',
        '__xor__',
        '__or__',
        '__lt__',
        '__le__',
        '__eq__',
        '__ne__',
        '__gt__',
        '__ge__',
        )

# all reverse are binary
_REVERSE_OPERATOR_MAP = {
        '__radd__': '__add__',
        '__rsub__': '__sub__',
        '__rmul__': '__mul__',
        '__rtruediv__': '__truediv__',
        '__rfloordiv__': '__floordiv__',
        }


def _ufunc_logical_skipna(array: np.ndarray,
        ufunc: tp.Callable,
        skipna: bool,
        axis: int=0,
        out=None
        ) -> np.ndarray:
    '''
    Given a logical (and, or) ufunc that does not support skipna, implement skipna behavior.
    '''
    if ufunc != np.all and ufunc != np.any:
        raise Exception('unsupported ufunc')

    if len(array) == 0:
        # TODO: handle if this is ndim == 2 and has no length
        if ufunc == np.all:
            return True
        return False # any() of an empty array is False

    if array.dtype.kind == 'b':
        # if boolean execute first
        return ufunc(array, axis=axis, out=out)
    elif array.dtype.kind == 'f':
        if skipna:
            # replace nans with nonzero value; faster to use masked array?
            v = array.copy()
            v[np.isnan(array)] = 0
            return ufunc(v, axis=axis, out=out)
        return ufunc(array, axis=axis, out=out)
    elif array.dtype.kind in _DTYPE_INT_KIND:
        return ufunc(array, axis=axis, out=out)

    # all types other than strings or objects" assume truthy
    elif array.dtype.kind != 'O' and array.dtype.kind not in _DTYPE_STR_KIND:
        if array.ndim == 1:
            return True
        return np.full(array.shape[0 if axis else 1], fill_value=True, dtype=bool)

    # convert to boolean aray then process
    if skipna:
        v = np.fromiter(((False if x is np.nan else bool(x)) for x in array.flat),
                count=array.size,
                dtype=bool).reshape(array.shape)
    else:
        v = np.fromiter((bool(x) for x in array.flat),
                count=array.size,
                dtype=bool).reshape(array.shape)
    return ufunc(v, axis=axis, out=out)


def _all(array, axis=0, out=None):
    return _ufunc_logical_skipna(array, ufunc=np.all, skipna=False, axis=axis, out=out)

_all.__doc__ = np.all.__doc__

def _any(array, axis=0, out=None):
    return _ufunc_logical_skipna(array, ufunc=np.any, skipna=False, axis=axis, out=out)

_any.__doc__ = np.any.__doc__

def _nanall(array, axis=0, out=None):
    return _ufunc_logical_skipna(array, ufunc=np.all, skipna=True, axis=axis, out=out)

def _nanany(array, axis=0, out=None):
    return _ufunc_logical_skipna(array, ufunc=np.any, skipna=True, axis=axis, out=out)


# TODO: specify the out dtype of these functions for bool
_UFUNC_AXIS_SKIPNA = {
        'all': (_all, _nanall, bool),
        'any': (_any, _nanany, bool),
        'sum': (np.sum, np.nansum, None),
        'min': (np.min, np.nanmin, None),
        'max': (np.max, np.nanmax, None),
        'mean': (np.mean, np.nanmean, None),
        'std': (np.std, np.nanstd, None),
        'var': (np.var, np.nanvar, None),
        'prod': (np.prod, np.nanprod, None),
        'cumsum': (np.cumsum, np.nancumsum, None),
        'cumprod': (np.cumprod, np.nancumprod, None)
        }

class MetaOperatorDelegate(type):
    '''Auto-populate binary and unary methods based on instance methods named `_ufunc_unary_operator` and `_ufunc_binary_operator`.
    '''

    @staticmethod
    def create_ufunc_operator(func_name, opperand_count=1, reverse=False):
        # operator module defines alias to funcs with names like __add__, etc
        if not reverse:
            operator_func = getattr(operator_mod, func_name)
            func_wrapper = operator_func
        else:
            unreversed_operator_func = getattr(operator_mod, _REVERSE_OPERATOR_MAP[func_name])
            # flip the order of the arguments
            operator_func = lambda rhs, lhs: unreversed_operator_func(lhs, rhs)
            func_wrapper = unreversed_operator_func

        if opperand_count == 1:
            assert not reverse # cannot reverse a single opperand
            def func(self):
                return self._ufunc_unary_operator(operator_func)
        elif opperand_count == 2:
            def func(self, other):
                return self._ufunc_binary_operator(operator=operator_func, other=other)
        else:
            raise NotImplementedError()

        f = wraps(func_wrapper)(func)
        f.__name__ = func_name
        return f

    @staticmethod
    def create_ufunc_axis_skipna(func_name):
        ufunc, ufunc_skipna, dtype = _UFUNC_AXIS_SKIPNA[func_name]

        # these become the common defaults for all of these functions
        def func(self, axis=0, skipna=True, **kwargs):
            return self._ufunc_axis_skipna(
                    axis=axis,
                    skipna=skipna,
                    ufunc=ufunc,
                    ufunc_skipna=ufunc_skipna,
                    dtype=dtype)

        f = wraps(ufunc)(func) # not sure if this is correct
        f.__name__ = func_name
        return f

    def __new__(mcs, name, bases, attrs):
        '''
        Create and assign all autopopulated functions.
        '''
        for opperand_count, func_name in chain(
                product((1,), _UFUNC_UNARY_OPERATORS),
                product((2,), _UFUNC_BINARY_OPERATORS)):
            attrs[func_name] = mcs.create_ufunc_operator(
                    func_name,
                    opperand_count=opperand_count)
        for func_name in _REVERSE_OPERATOR_MAP:
            attrs[func_name] = mcs.create_ufunc_operator(
                    func_name,
                    opperand_count=2,
                    reverse=True)

        for func_name in _UFUNC_AXIS_SKIPNA:
            attrs[func_name] = mcs.create_ufunc_axis_skipna(func_name)

        return type.__new__(mcs, name, bases, attrs)

#-------------------------------------------------------------------------------
# utility

# for getitem / loc selection
_KEY_ITERABLE_TYPES = (list, np.ndarray)

# types of keys that return muultiple items, even if the selection reduces to 1
_KEY_MULTIPLE_TYPES = (slice, list, np.ndarray)

# for type hinting
# keys once dimension has been isolated
GetItemKeyType = tp.Union[int, slice, list, None]
# keys that might include a multiple dimensions speciation; tuple is used to identify compound extraction
GetItemKeyTypeCompound = tp.Union[tuple, int, slice, list, None]

CallableOrMapping = tp.Union[tp.Callable, tp.Mapping]
KeyOrKeys = tp.Union[tp.Hashable, tp.Iterable[tp.Hashable]]
FilePathOrFileLike = tp.Union[str, StringIO, BytesIO]
DtypeSpecifier = tp.Optional[tp.Union[str, np.dtype, type]]

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

def mloc(array: np.ndarray) -> int:
    '''Return the memory location of an array.
    '''
    return array.__array_interface__['data'][0]


def _gen_skip_middle(
        forward_iter: tp.Iterable[tp.Any],
        forward_count: int,
        reverse_iter: tp.Iterable[tp.Any],
        reverse_count: int,
        center_sentinel: tp.Any):
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


def _full_for_fill(
        dtype: np.dtype,
        shape: tp.Union[int, tp.Tuple[int, int]],
        fill_value):
    '''
    Args:
        dtype: target dtype, which may or may not be possible given the fill_value.
    '''

    try:
        fill_can_cast = np.can_cast(fill_value, dtype)
    except TypeError: # happens when fill is None and dtype is float
        fill_can_cast = False

    dtype = object if not fill_can_cast else dtype
    return np.full(shape, fill_value, dtype=dtype)

def _resolve_dtype(dt1, dt2) -> np.dtype:
    '''
    Given two dtypes, return a compatible dtype that can hold both contents without truncation.
    '''
    # NOTE: np.dtype(object) == np.object_, so we can return np.object_
    # if the same, return that detype
    if dt1 == dt2:
        return dt1

    # if either is object, we go to object
    if dt1.kind == 'O' or dt2.kind == 'O':
        return np.object_

    dt1_is_str = dt1.kind in _DTYPE_STR_KIND
    dt2_is_str = dt2.kind in _DTYPE_STR_KIND

    # if both are string or string-lie, we can use result type to get the longest string
    if dt1_is_str and dt2_is_str:
        return np.result_type(dt1, dt2)

    dt1_is_bool = dt1.type is np.bool_
    dt2_is_bool = dt2.type is np.bool_

    # if any one is a string or a bool, we have to go to object; result_type gives a string in mixed cases
    if dt1_is_str or dt2_is_str or dt1_is_bool or dt2_is_bool:
        return np.object_

    # if not a string or an object, can use result type
    return np.result_type(dt1, dt2)

def _resolve_dtype_iter(dtypes: tp.Iterable[np.dtype]):
    '''Given an iterable of dtypes, do pairwise comparisons to determine compatible overall type. Once we get to object we can stop checking and return object
    '''
    dtypes = iter(dtypes)
    dt_resolve = next(dtypes)
    for dt in dtypes:
        dt_resolve = _resolve_dtype(dt_resolve, dt)
        if dt_resolve == np.object_:
            return dt_resolve
    return dt_resolve


def _dtype_to_na(dtype):
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

def _ufunc_skipna_1d(*, array, skipna, ufunc, ufunc_skipna):
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
    else:
        v = array
    if skipna:
        return ufunc_skipna(v)
    return ufunc(v)


def _iterable_to_array(other) -> tp.Tuple[np.ndarray, bool]:
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
        x = next(v_iter)
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


def _dict_to_sorted_items(dict: tp.Dict) -> tp.Generator[
            tp.Tuple[tp.Any, tp.Any], None, None]:
    '''
    Convert a dict into two arrays. Note that sorting is only necessary in Python 3.5, and should not be done if an ordered dict
    '''
    if isinstance(dict, OrderedDict):
        # cannot use fromiter as do not know type
        keys = dict.keys()
    else:
        keys = sorted(dict.keys())
    for k in keys:
        yield k, dict[k]


def _array_to_groups_and_locations(
        array: np.ndarray,
        unique_axis: int=None):
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
    # TODO: this function is a bottleneck: perhaps use numba?
    # moatches all floating point types
    if array.dtype.kind == 'f':
        return np.isnan(array)
    # match everything that is not an object; options are: biufcmMOSUV
    elif array.dtype.kind != 'O':
        return np.full(array.shape, False, dtype=bool)
    # only check for None if we have an object type
    # astype: None gets converted to nan if possible
    try: # this will only work for arrays that do not have strings
        # cannot use can_cast to reliabily identify arrays with non-float-castable elements
        return np.isnan(array.astype(float))
    except ValueError:
        # this means there was a character or something not castable to float; have to prceed slowly
        # TODO: this is a big perforamnce hit; problem is cannot find np.nan in numpy object array
        if array.ndim == 1:
            return np.fromiter((x is None or x is np.nan for x in array),
                    count=array.size,
                    dtype=bool)

        return np.fromiter((x is None or x is np.nan for x in array.flat),
                count=array.size,
                dtype=bool).reshape(array.shape)


def _array_to_duplicated(
        array: np.ndarray,
        axis: int=0,
        exclude_first=False,
        exclude_last=False):
    '''Given a numpy array, return a Boolean array along the specified axis that shows which values are duplicated. By default, all duplicates are indicated. For 2d arrays, axis 0 compares rows and returns a row-length Boolean array; axis 1 compares  colimns and returns a column-length Boolean array.

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


def _array_set_ufunc_many(arrays, ufunc=np.intersect1d):
    '''
    Iteratively apply a set operation unfunc to a arrays; if all are equal, no operation is performed and order is retained.
    '''
    arrays = iter(arrays)
    result = next(arrays)
    for array in arrays:
        # is the new array different
        if len(array) != len(result) or (array != result).any():
            result = ufunc(result, array)
        if ufunc == np.intersect1d and len(result) == 0:
            # short circuit intersection
            return result
    return result

#-------------------------------------------------------------------------------

class GetItem:
    __slots__ = ('callback',)

    def __init__(self, callback) -> None:
        self.callback = callback

    def __getitem__(self, key: GetItemKeyType):
        return self.callback(key)


class ExtractInterface:
    '''An instance to serve as an interface to all of iloc, loc, and __getitem__ extractors.
    '''
    __slots__ = ('iloc', 'loc', 'getitem')

    def __init__(self, *,
            iloc: GetItem,
            loc: GetItem,
            getitem: tp.Callable) -> None:
        self.iloc = iloc
        self.loc = loc
        self.getitem = getitem

    def __getitem__(self, key):
        return self.getitem(key)



#-------------------------------------------------------------------------------
# display infrastructure

class DisplayConfig:

    __slots__ = (
        'type_show',
        'type_color',
        'type_delimiter',
        'display_columns',
        'display_rows',
        'cell_max_width',
        'cell_align_left'
        )

    @classmethod
    def from_json(cls, str) -> 'DisplayConfig':
        args = json.loads(str.strip())
        return cls(**args)

    @classmethod
    def from_file(cls, fp):
        with open(fp) as f:
            return cls.from_json(f.read())

    def write(self, fp):
        '''Write a JSON file.
        '''
        with open(fp, 'w') as f:
            f.write(self.to_json() + '\n')

    @classmethod
    def from_default(cls, **kwargs):
        return cls(**kwargs)

    def __init__(self,
            type_show=True,
            type_color=False,
            type_delimiter='<>',
            display_columns: tp.Optional[int]=12,
            display_rows: tp.Optional[int]=36,
            cell_max_width: int=20,
            cell_align_left: bool=True
            ) -> None:
        self.type_show = type_show
        self.type_color = type_color
        self.type_delimiter = type_delimiter
        self.display_columns = display_columns
        self.display_rows = display_rows
        self.cell_max_width = cell_max_width
        self.cell_align_left = cell_align_left

    def __repr__(self):
        return '<' + self.__class__.__name__ + ' ' + ' '.join(
                '{k}={v}'.format(k=k, v=getattr(self, k))
                for k in self.__slots__) + '>'

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}

    def to_json(self) -> str:
        return(json.dumps(self.to_dict()))

    def to_transpose(self) -> 'DisplayConfig':
        args = self.to_dict()
        args['display_columns'], args['display_rows'] = (
                args['display_rows'], args['display_columns'])
        return self.__class__(**args)


_module._display_active = DisplayConfig()

class DisplayActive:
    '''Utility interface for setting module-level display configuration.
    '''
    FILE_NAME = '.static_frame.conf'

    @staticmethod
    def set(dc: DisplayConfig):
        _module._display_active = dc

    @staticmethod
    def get():
        return _module._display_active

    @classmethod
    def update(cls, **kwargs):
        args = cls.get().to_dict()
        args.update(kwargs)
        cls.set(DisplayConfig(**args))

    @classmethod
    def _default_fp(cls):
        return os.path.join(os.path.expanduser('~'), cls.FILE_NAME)

    @classmethod
    def write(cls, fp=None):
        fp = fp or cls._default_fp()
        dc = cls.get()
        dc.write(fp)

    @classmethod
    def read(cls, fp=None):
        fp = fp or cls._default_fp()
        cls.set(DisplayConfig.from_file(fp))



class Display:
    '''
    A Display is a string representation of a table, encoded as a list of lists, where list components are equal-width strings, keyed by row index
    '''
    __slots__ = ('_rows', '_config')

    CHAR_MARGIN = 1
    CELL_EMPTY = ('', 0)
    ELLIPSIS = '...'
    CELL_ELLIPSIS = (ELLIPSIS, len(ELLIPSIS))
    ELLIPSIS_INDICES = (None,)
    DATA_MARGINS = 2 # columns / rows that seperate data
    ELLIPSIS_CENTER_SENTINEL = object()

    @staticmethod
    def type_delimiter(dtype: np.dtype, config: DisplayConfig):
        dtype_str = str(dtype) if isinstance(dtype, np.dtype) else dtype
        if config.type_delimiter:
            return config.type_delimiter[0] + dtype_str + config.type_delimiter[1]
        return dtype_str

    @staticmethod
    def type_color(dtype: np.dtype):
        dtype_str = str(dtype) if isinstance(dtype, np.dtype) else dtype
        return '\033[90m' + dtype_str + '\033[0m'

    @classmethod
    def to_cell(cls,
            value: tp.Any,
            config: DisplayConfig,
            is_dtype=False) -> tp.Tuple[str, int]:

        if is_dtype:
            type_str = cls.type_delimiter(value, config=config)
            type_length = len(type_str)
            if config.type_color:
                type_str = cls.type_color(type_str)
            return (type_str, type_length)

        msg = str(value)
        return (msg, len(msg))

    @classmethod
    def from_values(cls,
            values: np.ndarray,
            header: str,
            config: DisplayConfig=None) -> 'Display':
        '''
        Given a 1 or 2D ndarray, return a Display instance. Generally 2D arrays are passed here only from TypeBlocks.
        '''
        # return a list of lists, where each inner list represents multiple columns
        config = config or DisplayActive.get()

        msg = header.strip()

        # create a list of lists, always starting with the header
        rows = [[(msg, len(msg))]]

        if isinstance(values, np.ndarray) and values.ndim == 2:
            # get rows from numpy string formatting
            np_rows = np.array_str(values).split('\n')
            last_idx = len(np_rows) - 1
            for idx, row in enumerate(np_rows):
                # trim brackets
                end_slice_len = 2 if idx == last_idx else 1
                row = row[2: len(row) - end_slice_len].strip()
                rows.append([cls.to_cell(row, config=config)])
        else:
            count_max = config.display_rows - cls.DATA_MARGINS
            # print('comparing values to count_max', len(values), count_max)
            if len(values) > config.display_rows:
                data_half_count = Display.truncate_half_count(count_max)
                value_gen = partial(_gen_skip_middle,
                        forward_iter=values.__iter__,
                        forward_count=data_half_count,
                        reverse_iter=partial(reversed, values),
                        reverse_count=data_half_count,
                        center_sentinel=cls.ELLIPSIS_CENTER_SENTINEL
                        )
            else:
                value_gen = values.__iter__

            for v in value_gen():
                if v is cls.ELLIPSIS_CENTER_SENTINEL: # center sentinel
                    rows.append([cls.CELL_ELLIPSIS])
                else:
                    rows.append([cls.to_cell(v, config=config)])

        # add the types to the last row
        if isinstance(values, np.ndarray) and config.type_show:
            rows.append([cls.to_cell(values.dtype, config=config, is_dtype=True)])
        else: # this is an object
            rows.append([cls.CELL_EMPTY])

        return cls(rows, config=config)


    @staticmethod
    def truncate_half_count(count_target: int) -> int:
        '''Given a target number of rows or columns, return the count of half as found in presentation where one column is used for the elipsis. The number returned will always be odd. For example, given a target of 5 we allocate 2 per half (plus 1 reserved for middle).
        '''
        if count_target <= 4:
            return 1 # practical floor for all values of 4 or less
        return (count_target - 1) // 2

    @classmethod
    def _truncate_indices(cls, count_target: int, indices):

        # if have 5 data cols, 7 total, and target was 6
        # half count of 2, 5 total out, with 1 meta, 1 data, elipsis, data, meta

        # if have 5 data cols, 7 total, and target was 7
        # half count of 3, 7 total out, with 1 meta, 2 data, elipsis, 2 data, 1 meta

        # if have 6 data cols, 8 total, and target was 6
        # half count of 2, 5 total out, with 1 meta, 1 data, elipsis, data, meta

        # if have 6 data cols, 8 total, and target was 7
        # half count of 3, 7 total out, with 1 meta, 2 data, elipsis, 2 data, 1 meta

        if count_target and len(indices) > count_target:
            half_count = cls.truncate_half_count(count_target)
            # replace with array from_iter? with known size?
            return tuple(chain(
                    indices[:half_count],
                    cls.ELLIPSIS_INDICES,
                    indices[-half_count:]))
        return indices

    @classmethod
    def _to_rows(cls, display: 'Display', config: DisplayConfig=None) -> tp.Iterable[str]:
        '''
        Given already defined rows, align them to left or right.
        '''
        config = config or DisplayActive.get()

        # find max columns for all defined rows
        col_count_src = max(len(row) for row in display._rows)
        col_last_src = col_count_src - 1

        row_count_src = len(display._rows)
        row_indices = tuple(range(row_count_src))

        rows = [[] for _ in row_indices]

        for col_idx_src in range(col_count_src):
            # for each column, get the max width
            max_width = 0
            for row_idx_src in row_indices:
                # get existing max width, up to the max
                if row_idx_src is not None:
                    row = display._rows[row_idx_src]
                    if col_idx_src >= len(row): # this row does not have this column
                        continue
                    cell = row[col_idx_src]

                    max_width = max(max_width, cell[1])
                else:
                    max_width = max(max_width, len(cls.ELLIPSIS))
                # if we have already exceeded max width, can stop iterating
                if max_width >= config.cell_max_width:
                    break
            max_width = min(max_width, config.cell_max_width)

            if ((config.cell_align_left is True and col_idx_src == col_last_src) or
                    (config.cell_align_left is False and col_idx_src == 0)):
                pad_width = max_width
            else:
                pad_width = max_width + cls.CHAR_MARGIN

            for row_idx_src in row_indices:
                row = display._rows[row_idx_src]
                if col_idx_src >= len(row):
                    cell = cls.CELL_EMPTY
                else:
                    cell = row[col_idx_src]
                # msg may have been ljusted before, so we strip again here
                # cannot use ljust here, as the cell might have more characters for coloring
                if cell[1] > max_width:
                    cell_content = cell[0].strip()[:max_width - 3] + cls.ELLIPSIS
                    cell_fill_width = cls.CHAR_MARGIN # should only be margin left
                else:
                    cell_content = cell[0].strip()
                    cell_fill_width = pad_width - cell[1] # this includes margin

                # print(col_idx, row_idx, cell, max_width, pad_width, cell_fill_width)
                if config.cell_align_left:
                    # must manually add space as color chars make ljust not
                    msg = cell_content + ' ' * cell_fill_width
                else:
                    msg =  ' ' * cell_fill_width + cell_content

                rows[row_idx_src].append(msg)

        # rstrip to remove extra white space on last column
        return [''.join(row).rstrip() for row in rows]


    def __init__(self,
            rows: tp.List[tp.List[tp.Tuple[str, int]]],
            config: DisplayConfig=None) -> None:
        '''Define rows as a list of strings; the strings may be of different size, but they are expected to be aligned vertically in final presentation.
        '''
        config = config or DisplayActive.get()
        self._rows = rows
        self._config = config


    def __repr__(self):
        return '\n'.join(self._to_rows(self, self._config))

    def to_rows(self) -> tp.Iterable[str]:
        return self._to_rows(self, self._config)

    def __iter__(self):
        for row in self._rows:
            yield [cell[0] for cell in row]

    def __len__(self):
        return len(self._rows)

    #---------------------------------------------------------------------------
    # in place mutation

    def append_display(self, display: 'Display') -> None:
        '''
        Mutate this display by appending the passed display.
        '''
        # NOTE: do not want to pass config or call format here as we call this for each column or block we add
        for row_idx, row in enumerate(display._rows):
            self._rows[row_idx].extend(row)

    def append_iterable(self,
            iterable: tp.Iterable[tp.Any],
            header: str) -> None:
        '''
        Add an iterable of strings
        '''
        self._rows[0].append(self.to_cell(header, config=self._config))

        # truncate iterable if necessary
        count_max = self._config.display_rows - self.DATA_MARGINS

        if len(iterable) > count_max:
            data_half_count = self.truncate_half_count(count_max)
            value_gen = partial(_gen_skip_middle,
                    forward_iter = iterable.__iter__,
                    forward_count = data_half_count,
                    reverse_iter = partial(reversed, iterable),
                    reverse_count = data_half_count,
                    center_sentinel = self.ELLIPSIS_CENTER_SENTINEL
                    )
        else:
            value_gen = iterable.__iter__

        # start at 1 as 0 is header
        for idx, value in enumerate(value_gen(), start=1):
            if value is self.ELLIPSIS_CENTER_SENTINEL:
                self._rows[idx].append(self.CELL_ELLIPSIS)
            else:
                self._rows[idx].append(self.to_cell(value, config=self._config))

        if isinstance(iterable, np.ndarray):
            if self._config.type_show:
                self._rows[idx + 1].append(self.to_cell(iterable.dtype,
                        config=self._config,
                        is_dtype=True))

    def append_ellipsis(self):
        '''Append an ellipsis over all rows.
        '''
        for row in self._rows:
            row.append(self.CELL_ELLIPSIS)

    def insert_rows(self, *displays: tp.Iterable['Display']):
        '''
        Insert rows on top of existing rows.
        args:
            Each arg in args is an instance of Display
        '''
        # each arg is a list, to be a new row
        # assume each row in display becomes a column
        new_rows = []
        for display in displays:
            new_rows.extend(display._rows)
        # slow for now: make rows a dict to make faster
        new_rows.extend(self._rows)
        self._rows = new_rows

    #---------------------------------------------------------------------------
    # return a new display

    def flatten(self) -> 'Display':
        row = []
        for part in self._rows:
            row.extend(part)
        rows = [row]
        return self.__class__(rows, config=self._config)

#-------------------------------------------------------------------------------
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
    def from_correspondence(cls, src_index, dst_index) -> 'IndexCorrespondence':
        # sorts results
        common_labels = np.intersect1d(src_index.values, dst_index.values)
        has_common = len(common_labels) > 0
        size = len(dst_index.values)

        # either a reordering or a subsetfrom_index
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

            # these will be equal sized; not sure if they will be in order
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
            iloc_pairs: generate corresponding pairs of iloc postions between src and dst; may be empty or have less constituents than index.
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
class TypeBlocks(metaclass=MetaOperatorDelegate):
    '''An ordered collection of potentially heterogenous, immutable NumPy arrays, providing an external array-like interface of a single, 2D array. Used by :py:class:`Frame` for core, unindexed array management.
    '''
    # related to Pandas BlockManager
    __slots__ = (
            '_blocks',
            '_dtypes',
            '_index',
            '_shape',
            '_row_dtype',
            'iloc',
            )

    @staticmethod
    def immutable_filter(src_array: np.ndarray) -> np.ndarray:
        '''Pass an immutable array; otherwise, return an immutable copy of the provided array.
        '''
        if src_array.flags.writeable:
            dst_array = src_array.copy()
            dst_array.flags.writeable = False
            return dst_array
        return src_array # keep it as is

    @staticmethod
    def single_column_filter(array: np.ndarray) -> np.ndarray:
        '''Reshape a flat ndim 1 array into a 2D array with one columns and rows of length. This is only used (a) for getting string representations and (b) for using np.concatenate and np binary operators on 1D arrays.
        '''
        # it is not clear when reshape is a copy or a view
        if array.ndim == 1:
            return np.reshape(array, (array.shape[0], 1))
        return array

    @staticmethod
    def shape_filter(array: np.ndarray) -> tp.Tuple[int, int]:
        '''Reprsent a 1D array as a 2D array with length as rows of a single-column array.

        Return:
            row, column count for a block of ndim 1 or ndim 2.
        '''
        if array.ndim == 1:
            return array.shape[0], 1
        return array.shape

    #---------------------------------------------------------------------------
    # constructors

    @classmethod
    def from_blocks(cls,
            raw_blocks: tp.Iterable[np.ndarray]) -> 'TypeBlocks':
        '''
        The order of the blocks defines the order of the columns contained.

        Args:
            raw_blocks: iterable (generator compatible) of NDArrays.
        '''
        blocks = [] # ordered blocks
        index = [] # columns position to blocks key
        dtypes = [] # column position to dtype
        block_count = 0

        # if a single block, no need to loop
        if isinstance(raw_blocks, np.ndarray):
            row_count, column_count = cls.shape_filter(raw_blocks)
            blocks.append(cls.immutable_filter(raw_blocks))
            for i in range(column_count):
                index.append((block_count, i))
                dtypes.append(raw_blocks.dtype)

        else: # an iterable of blocks
            row_count = 0
            column_count = 0
            for block in raw_blocks:
                assert isinstance(block, np.ndarray), 'found non array block: %s' % block
                if block.ndim > 2:
                    raise Exception('cannot include array with more than 2 dimensions')

                r, c = cls.shape_filter(block)
                # check number of rows is the same for all blocks
                if row_count:
                    assert r == row_count, 'mismatched row count: %s: %s' % (r, row_count)
                else: # assign on first
                    row_count = r

                blocks.append(cls.immutable_filter(block))

                # store position to key of block, block columns
                for i in range(c):
                    index.append((block_count, i))
                    dtypes.append(block.dtype)
                column_count += c
                block_count += 1

        # blocks cam be empty
        return cls(blocks=blocks,
                dtypes=dtypes,
                index=index,
                shape=(row_count, column_count),
                )


    @classmethod
    def from_element_items(cls, items, shape, dtype):
        '''Given a generator of pairs of iloc coords and values, return a TypeBlock of the desired shape and dtype.
        '''
        a = np.full(shape, fill_value=_dtype_to_na(dtype), dtype=dtype)
        for iloc, v in items:
            a[iloc] = v
        a.flags.writeable = False
        return cls.from_blocks(a)


    @classmethod
    def from_none(cls):
        return cls(blocks=list(), dtypes=list(), index=list(), shape=(0, 0))

    #---------------------------------------------------------------------------

    def __init__(self, *,
            blocks: tp.Iterable[np.ndarray],
            dtypes: tp.Iterable[np.dtype],
            index: tp.Iterable[tp.Tuple[int, int]],
            shape: tp.Tuple[int, int] # could be derived
            ) -> None:
        '''
        Args:
            blocks: A list of one or two-dimensional NumPy arrays
            dtypes: list of dtypes per external column
            index: list of tuple coordinates, where list index is external column and the tuple is the block, intra-block column
            shape: two-element tuple defining row and column count. A (0, 0) shape is permitted for empty TypeBlocks.
        '''
        self._blocks = blocks
        self._dtypes = dtypes
        self._index = index # list where index, as column, gets block, offset
        self._shape = shape

        if self._blocks:
            self._row_dtype = _resolve_dtype_iter(b.dtype for b in self._blocks)
        else:
            self._row_dtype = None

        # assert len(self._dtypes) == len(self._index) == self._shape[1]

        # set up callbacks
        self.iloc = GetItem(self._extract_iloc)


    def copy(self) -> 'TypeBlocks':
        '''
        Return a new TypeBlocks. Underlying arrays do not need to be copied.
        '''
        return self.__class__(
                blocks=[b for b in self._blocks],
                dtypes=self._dtypes.copy(), # list
                index=self._index.copy(),
                shape=self._shape)

    #---------------------------------------------------------------------------
    # new properties

    @property
    def dtypes(self) -> np.ndarray:
        '''
        Return an immutable array that, for each realizable column (not each block), the dtype is given.
        '''
        # this creates a new array every time it is called; could cache
        a = np.array(self._dtypes, dtype=np.dtype)
        a.flags.writeable = False
        return a

    # consider renaming pointers
    @property
    def mloc(self) -> np.ndarray:
        '''Return an immutable ndarray of NP array memory location integers.
        '''
        a = np.fromiter(
                (mloc(b) for b in self._blocks),
                count=len(self._blocks),
                dtype=int)
        a.flags.writeable = False
        return a

    @property
    def unified(self) -> bool:
        return len(self._blocks) <= 1

    #---------------------------------------------------------------------------
    # common NP-style properties

    @property
    def shape(self) -> tp.Tuple[int, int]:
        # make this a property so as to be immutable
        return self._shape

    @property
    def ndim(self) -> int:
        return 2

    @property
    def size(self) -> int:
        return sum(b.size for b in self._blocks)

    @property
    def nbytes(self) -> int:
        return sum(b.nbytes for b in self._blocks)

    #---------------------------------------------------------------------------
    # value extraction

    @staticmethod
    def _blocks_to_array(*, blocks, shape, row_dtype) -> np.ndarray:
        '''
        Given blocks and a combined shape, return a consolidated single array.
        '''
        if len(blocks) == 1:
            return blocks[0]

        # get empty array and fill parts
        if shape[0] == 1:
            # greturn 1 column TypeBlock as a 1D array with length equal to the number of columns
            array = np.empty(shape[1], dtype=row_dtype)
        else: # get ndim 2 shape array
            array = np.empty(shape, dtype=row_dtype)

        # can we use a np.concatenate, but need to handle 1D arrrays and need to converty type before concatenate

        pos = 0
        for block in blocks:
            if block.ndim == 1:
                end = pos + 1
            else:
                end = pos + block.shape[1]

            if array.ndim == 1:
                array[pos: end] = block[:] # gets a row from array
            else:
                if block.ndim == 1:
                    array[:, pos] = block[:] # a 1d array
                else:
                    array[:, pos: end] = block[:] # gets a row / row slice from array
            pos = end

        array.flags.writeable = False
        return array


    @property
    def values(self) -> np.ndarray:
        '''Returns a consolidated NP array of the all blocks.
        '''
        return self._blocks_to_array(
                blocks=self._blocks,
                shape=self._shape,
                row_dtype=self._row_dtype)

    def axis_values(self, axis=0, reverse=False) -> tp.Generator[np.ndarray, None, None]:
        '''Generator of arrays produced along an axis.

        Args:
            axis: 0 iterates over columns, 1 iterates over rows
        '''
        # NOTE: can add a reverse argument here and iterate in reverse; this could be useful if we need to pass rows/cols to lexsort, as in _array_to_duplicated
        if axis == 1: # iterate over rows
            unified = self.unified
            # iterate over rows; might be faster to create entire values
            if not reverse:
                row_idx_iter = range(self._shape[0])
            else:
                row_idx_iter = range(self._shape[0] - 1, -1, -1)

            for i in row_idx_iter:
                if unified:
                    yield self._blocks[0][i]
                else:
                    # cannot use a generator w/ np concat
                    # use == for type comparisons
                    parts = []
                    for b in self._blocks:
                        if b.ndim == 1:
                            # get a slice to permit concatenation
                            key = slice(i, i+1)
                        else:
                            key = i
                        if b.dtype == self._row_dtype:
                            parts.append(b[key])
                        else:
                            parts.append(b[key].astype(self._row_dtype))
                    yield np.concatenate(parts)

        elif axis == 0: # iterate over columns
            if not reverse:
                block_column_iter = self._index
            else:
                block_column_iter = reversed(self._index)

            for block_idx, column in block_column_iter:
                b = self._blocks[block_idx]
                if b.ndim == 1:
                    yield b
                else:
                    yield b[:, column]
        else:
            raise NotImplementedError()


    def element_items(self) -> tp.Generator[
            tp.Tuple[tp.Tuple[int, int], tp.Any], None, None]:
        '''
        Generator of pairs of iloc locations, values accross entire TypeBlock.
        '''
        #np.ndindex(self._shape) # get all target indices
        offsets = np.cumsum([b.shape[1] if b.ndim == 2 else 1 for b in self._blocks])
        gens = [np.ndenumerate(self.single_column_filter(b)) for b in self._blocks]

        for iloc in np.ndindex(self._shape):
            block_idx, column = self._index[iloc[1]]
            b = self._blocks[block_idx]
            if b.ndim == 1:
                yield iloc, b[iloc[0]]
            else:
                yield iloc, b[iloc[0], column]

    #---------------------------------------------------------------------------
    # methods for evaluating compatibility with other blocks, and reblocking

    def block_compatible(self,
            other: 'TypeBlocks',
            by_shape=True) -> bool:
        '''Block compatible means that the blocks are the same shape and the same (or compatible) dtype.

        Args:
            by_shape: If True, the full shape is compared; if False, only the columns width iis compared.
        '''
        for a, b in zip_longest(self._blocks, other._blocks, fillvalue=None):
            if a is None or b is None:
                return False
            if by_shape:
                if self.shape_filter(a) != self.shape_filter(b):
                    return False
            else:
                if self.shape_filter(a)[1] != self.shape_filter(b)[1]:
                    return False
            # this does not show us if the types can be operated on;
            # similarly, np.can_cast, np.result_type do not telll us if an operation will succeede
            # if not a.dtype is b.dtype:
            #     return False
        return True

    def reblock_compatible(self, other: 'TypeBlocks') -> bool:
        '''
        Return True if post reblocking these TypeBlocks are compatible. This only compares columns in blocks, not the entire shape.
        '''
        # we only compare size, not the type
        if any(a is None or b is None or a[1] != b[1]
                for a, b in zip_longest(
                self._reblock_signature(),
                other._reblock_signature())):
            return False
        return True

    @classmethod
    def _concatenate_blocks(cls, group: tp.Iterable[np.ndarray]):
        '''
        '''
        return np.concatenate([cls.single_column_filter(x) for x in group], axis=1)

    @classmethod
    def consolidate_blocks(cls,
            raw_blocks: tp.Iterable[np.ndarray]) -> tp.Generator[np.ndarray, None, None]:
        '''
        Generator consumer, generator producer of np.ndarray, consolidating if types are exact matches. Possible improvement to discover when a type can correctly old another type.
        '''
        group_dtype = None # store type found along contiguous blocks
        group = []

        for block in raw_blocks:
            if group_dtype is None: # first block of a type
                group_dtype = block.dtype
                group.append(block)
                continue

            if block.dtype != group_dtype:
                # new group found, return stored
                if len(group) == 1: # return reference without copy
                    yield group[0]
                else: # combine groups
                    # could pre allocating and assing as necessary for large groups
                    yield cls._concatenate_blocks(group)
                group_dtype = block.dtype
                group = [block]
            else: # new block has same group dtype
                group.append(block)

        # get anything leftover
        if group:
            if len(group) == 1:
                yield group[0]
            else:
                yield cls._concatenate_blocks(group)


    def _reblock(self) -> tp.Generator[np.ndarray, None, None]:
        '''Generator of new block that consolidate adjacent types that are the same.
        '''
        yield from self.consolidate_blocks(raw_blocks=self._blocks)

    def consolidate(self) -> 'TypeBlocks':
        '''Return a new TypeBlocks that unifies all adjacent types.
        '''
        # note: not sure if we have a single block if we should return a new TypeBlocks instance (as done presently), or simply return self; either way, no new np arrays will be created
        return self.from_blocks(self.consolidate_blocks(raw_blocks=self._blocks))


    def _reblock_signature(self) -> tp.Generator[tp.Tuple[np.dtype, int], None, None]:
        '''For anticipating if a reblock will result in a compatible block configuration for operator application, get the reblock signature, providing the dtype and size for each block without actually reblocking.

        This is a generator to permit lazy pairwise comparison.
        '''
        group_dtype = None # store type found along contiguous blocks
        group_cols = 0
        for block in self._blocks:
            if group_dtype is None: # first block of a type
                group_dtype = block.dtype
                if block.ndim == 1:
                    group_cols += 1
                else:
                    group_cols += block.shape[1]
                continue
            if block.dtype != group_dtype:
                yield (group_dtype, group_cols)
                group_dtype = block.dtype
                group_cols = 0
            if block.ndim == 1:
                group_cols += 1
            else:
                group_cols += block.shape[1]
        if group_cols > 0:
            yield (group_dtype, group_cols)

    def resize_blocks(self, *,
            index_ic: tp.Optional[IndexCorrespondence],
            columns_ic = tp.Optional[IndexCorrespondence],
            fill_value = tp.Any
            ) -> tp.Generator[np.ndarray, None, None]:
        '''
        Given index and column IndexCorrespondence objects, return a generator of resized blocks, extracting from self based on correspondence. Used for Frame.reindex()
        '''
        if columns_ic is None and index_ic is None:
            for b in self._blocks:
                yield b

        elif columns_ic is None and index_ic is not None:
            for b in self._blocks:
                if index_ic.is_subset:
                    # works for both 1d and 2s arrays
                    yield b[index_ic.iloc_src]
                else:
                    shape = index_ic.size if b.ndim == 1 else (index_ic.size, b.shape[1])
                    values = _full_for_fill(b.dtype, shape, fill_value)
                    if index_ic.has_common:
                        values[index_ic.iloc_dst] = b[index_ic.iloc_src]
                    values.flags.writeable = False
                    yield values

        elif columns_ic is not None and index_ic is None:
            if not columns_ic.has_common:
                # just return an empty frame; what type it shold be is not clear
                shape = self.shape[0], columns_ic.size
                values = _full_for_fill(self._row_dtype, shape, fill_value)
                values.flags.writeable = False
                yield values
            else:
                if self.unified and columns_ic.is_subset:
                    b = self._blocks[0]
                    if b.ndim == 1:
                        yield b
                    else:
                        yield b[:, columns_ic.iloc_src]
                else:
                    dst_to_src = dict(zip(columns_ic.iloc_dst, columns_ic.iloc_src))
                    for idx in range(columns_ic.size):
                        if idx in dst_to_src:
                            block_idx, block_col = self._index[dst_to_src[idx]]
                            b = self._blocks[block_idx]
                            if b.ndim == 1:
                                yield b
                            else:
                                yield b[:, block_col]
                        else:
                            # just get an empty position
                            # dtype should be the same as the column replacing?
                            values = _full_for_fill(self._row_dtype,
                                    self.shape[0],
                                    fill_value)
                            values.flags.writeable = False
                            yield values

        else: # both defined
            if not columns_ic.has_common and not index_ic.has_common:
                # just return an empty frame; what type it shold be is not clear
                shape = index_ic.size, columns_ic.size
                values = _full_for_fill(self._row_dtype, shape, fill_value)
                values.flags.writeable = False
                yield values
            else:
                if self.unified and index_ic.is_subset and columns_ic.is_subset:
                    b = self._blocks[0]
                    if b.ndim == 1:
                        yield b[index_ic.iloc_src]
                    else:
                        yield b[index_ic.iloc_src_fancy(), columns_ic.iloc_src]
                else:
                    columns_dst_to_src = dict(zip(columns_ic.iloc_dst, columns_ic.iloc_src))

                    for idx in range(columns_ic.size):
                        if idx in columns_dst_to_src:
                            block_idx, block_col = self._index[columns_dst_to_src[idx]]
                            b = self._blocks[block_idx]

                            if index_ic.is_subset:
                                if b.ndim == 1:
                                    yield b[index_ic.iloc_src]
                                else:
                                    yield b[index_ic.iloc_src, block_col]
                            else: # need an empty to fill
                                values = _full_for_fill(self._row_dtype,
                                        index_ic.size,
                                        fill_value)
                                if b.ndim == 1:
                                    values[index_ic.iloc_dst] = b[index_ic.iloc_src]
                                else:
                                    values[index_ic.iloc_dst] = b[index_ic.iloc_src, block_col]
                                values.flags.writeable = False
                                yield values
                        else:
                            values = _full_for_fill(self._row_dtype,
                                        index_ic.size,
                                        fill_value)
                            values.flags.writeable = False
                            yield values


    def group(self,
            axis,
            key) -> tp.Generator[tp.Tuple[tp.Hashable, np.ndarray, np.ndarray], None, None]:
        '''
        Args:
            key: iloc selector on opposite axis

        Returns:
            Generator of group, selection pairs, where selection is an np.ndaarray
        '''
        # in worse case this will make a copy of the values extracted; this is probably still cheaper than iterating manually through rows/columns
        unique_axis = None
        if axis == 0:
            # axis 0 means we return row groups; key is a column key
            group_source = self._extract_array(column_key=key)
            if group_source.ndim > 1:
                unique_axis = 0
        elif axis == 1:
            # axis 1 means we return column groups; key is a row key
            group_source = self._extract_array(row_key=key)
            if group_source.ndim > 1:
                unique_axis = 1

        groups, locations = _array_to_groups_and_locations(
                group_source,
                unique_axis)

        for idx, g in enumerate(groups):
            selection = locations == idx
            if axis == 0: # return row extractions
                yield g, selection, self._extract(row_key=selection)
            elif axis == 1: # return columns extractions
                yield g, selection, self._extract(column_key=selection)


    def block_apply_axis(self, func, *, axis, dtype=None) -> np.ndarray:
        '''Apply a function that reduces blocks to a single axis.

        Args:
            dtype: if we know the return type of func, we can provide it here to avoid having to use the row dtype.

        Returns:
            As this is a reduction of axis where the caller (a Frame) is likely to return a Series, this function is not a generator of blocks, but instead just returns a consolidated 1d array.
        '''
        assert axis < 2

        if self.unified:
            # TODO: not sure if we need dim filter here
            result = func(self._blocks[0], axis=axis)
            result.flags.writeable = False
            return result
        else:
            # need to have good row dtype here, so that ints goes to floats
            if axis == 0:
                # reduce all rows to 1d with column width
                shape = self._shape[1]
                pos = 0
            else:
                # reduce all columns to 2d blocks with 1 column
                shape = (self._shape[0], len(self._blocks))

            # this will be uninitialzied and thuse, if a value is not assigned, will have garbage
            out = np.empty(shape, dtype=dtype or self._row_dtype)
            for idx, b in enumerate(self._blocks):
                if axis == 0:
                    if b.ndim == 1:
                        end = pos + 1
                        out[pos] = func(b, axis=axis)
                    else:
                        end = pos + b.shape[1]
                        temp = func(b, axis=axis)
                        if len(temp) != end - pos:
                            raise Exception('unexpected.')
                        func(b, axis=axis, out=out[pos: end])
                    pos = end
                else:
                    if b.ndim == 1: # cannot process yet
                        # if this is a numeric single columns we just copy it and process it later; but if this is a logical application (and, or) then out is already boolean
                        if out.dtype == bool and b.dtype != bool:
                            # making 2D with axis 0 func will result in element-wise operation
                            out[:, idx] = func(self.single_column_filter(b), axis=0)
                        else: # otherwise, keep as is
                            out[:, idx] = b
                    else:
                        func(b, axis=axis, out=out[:, idx])

        if axis == 0: # nothing more to do
            out.flags.writeable = False
            return out
        # must call function one more time on remaining components
        result = func(out, axis=axis)
        result.flags.writeable = False
        return result

    #---------------------------------------------------------------------------
    def __len__(self):
        '''Length, as with NumPy and Pandas, is the number of rows.
        '''
        return self._shape[0]


    def display(self, config: DisplayConfig=None) -> Display:
        config = config or DisplayActive.get()

        h = '<' + self.__class__.__name__ + '>'
        d = None
        for idx, block in enumerate(self._blocks):
            block = self.single_column_filter(block)
            header = '' if idx > 0 else h
            display = Display.from_values(block, header, config=config)
            if not d: # assign first
                d = display
            else:
                d.append_display(display)
        return d

    def __repr__(self) -> str:
        return repr(self. display())


    #---------------------------------------------------------------------------
    # extraction utilities

    @staticmethod
    def _cols_to_slice(indices: tp.Sequence[int]) -> slice:
        '''Translate an iterable of contiguous integers into a slice
        '''
        start_idx = indices[0]
        # can always represetn a singel column a single slice
        if len(indices) == 1:
            return slice(start_idx, start_idx + 1)

        stop_idx = indices[-1]
        if stop_idx > start_idx:            # ascending indices
            return slice(start_idx, stop_idx + 1)

        if stop_idx == 0:
            return slice(start_idx, None, -1)
        # stop is less than start, need to reduce by 1 to cover range
        return slice(start_idx, stop_idx - 1, -1)


    @classmethod
    def _indices_to_contiguous_pairs(cls, indices) -> tp.Generator:
        '''Indices are pairs of (block_idx, value); convert these to pairs of (block_idx, slice) when we identify contiguous indices.

        Args:
            indices: can be a generator
        '''
        # store pairs of block idx, ascending col list
        last = None
        for block_idx, col in indices:
            if not last:
                last = (block_idx, col)
                bundle = [col]
                continue
            if last[0] == block_idx and abs(col - last[1]) == 1:
                # if contiguous, update last, add to bundle
                last = (block_idx, col)
                bundle.append(col)
                continue
            # either new block, or not contiguous on same block
            # store what we have so far
            assert len(bundle) > 0
            # yield a pair of block_idx, contiguous slice
            yield (last[0], cls._cols_to_slice(bundle))
            # but this one on bundle
            bundle = [col]
            last = (block_idx, col)
        # last might be None if we
        if last and bundle:
            yield (last[0], cls._cols_to_slice(bundle))


    def _key_to_block_slices(self, key) -> tp.Generator[
                tp.Tuple[int, tp.Union[slice, int]], None, None]:
        '''
        For a column key (an integer, slice, or iterable), generate pairs of (block_idx, slice or integer) to cover all extractions. First, get the relevant index values (pairs of block id, column id), then convert those to contiguous slices.

        Returns:
            A generator iterable of pairs, where values are block index, slice or column index
        '''

        # do type checking on slice v others, as with others we need to sort once iterable of keys
        if isinstance(key, (int, np.int_)):
            # the index has the pair block, column integer
            yield self._index[key]
        else:
            if isinstance(key, slice):
                indices = self._index[key] # slice the index
                # already sorted
            elif isinstance(key, np.ndarray) and key.dtype == bool:
                indices = (self._index[idx]
                        for idx, v in enumerate(key) if v == True)

            elif isinstance(key, _KEY_ITERABLE_TYPES):
                # an iterable of keys, may not have contiguous regions; provide in the order given; set as a generator; self._index is a list, not an np.array, so cannot slice self._index; requires iteration in passed generator anyways so probably this is as fast as it can be.
                indices = (self._index[x] for x in key)
            elif key is None: # get all
                indices = self._index
            else:
                raise NotImplementedError('got key', key)
            yield from self._indices_to_contiguous_pairs(indices)


    def _mask_blocks(self,
            row_key=None,
            column_key=None) -> tp.Generator[np.ndarray, None, None]:
        '''Return Boolean blocks of the same size and shape, where key selection sets values to True.
        '''

        # this selects the columns; but need to return all bloics
        block_slices = iter(self._key_to_block_slices(column_key))
        target_block_idx = target_slice = None

        for block_idx, b in enumerate(self._blocks):
            mask = np.full(b.shape, False, dtype=bool)

            while True:
                if target_block_idx is None: # can be zero
                    try:
                        target_block_idx, target_slice = next(block_slices)
                    except StopIteration:
                        break

                if block_idx != target_block_idx:
                    break # need to advance blocks

                if b.ndim == 1: # given 1D array, our row key is all we need
                    mask[row_key] = True
                else:
                    if row_key is None:
                        mask[:, target_slice] = True
                    else:
                        mask[row_key, target_slice] = True

                target_block_idx = target_slice = None

            yield mask


    def _assign_blocks_from_keys(self,
            row_key=None,
            column_key=None,
            value=None) -> tp.Generator[np.ndarray, None, None]:
        '''Assign value into all blocks, returning blocks of the same size and shape.
        '''
        if isinstance(value, np.ndarray):
            value_dtype = value.dtype
        else:
            value_dtype = np.array(value).dtype

        # this selects the columns; but need to return all blocks
        block_slices = iter(self._key_to_block_slices(column_key))
        target_block_idx = target_slice = None

        for block_idx, b in enumerate(self._blocks):

            assigned = None
            while True:
                if target_block_idx is None: # can be zero
                    try:
                        target_block_idx, target_slice = next(block_slices)
                    except StopIteration:
                        break

                if block_idx != target_block_idx:
                    break # need to advance blocks, keep targets

                # from here, we have a target we need to apply
                if assigned is None:
                    assigned_dtype = _resolve_dtype(value_dtype, b.dtype)
                    if b.dtype == assigned_dtype:
                        assigned = b.copy()
                    else:
                        assigned = b.astype(assigned_dtype)

                # match sliceable, when target_slice is a slice (can be an integer)
                if (isinstance(target_slice, slice) and
                        not isinstance(value, str)
                        and hasattr(value, '__len__')):
                    if b.ndim == 1:
                        width = 1
                        value_piece = value[0] # do not want to slice
                    else:
                        width = len(range(*target_slice.indices(assigned.shape[1])))
                        value_piece = value[slice(0, width)]
                    # reassign remainder for next iteration
                    value = value[slice(width, None)]
                else: # not sliceable
                    value_piece = value
                    value = value
                if b.ndim == 1: # given 1D array, our row key is all we need
                    assigned[row_key] = value_piece
                else:
                    if row_key is None:
                        assigned[:, target_slice] = value_piece
                    else:
                        assigned[row_key, target_slice] = value_piece

                target_block_idx = target_slice = None

            if assigned is None:
                yield b # no change
            else:
                # disable writing so clients can keep the array
                assigned.flags.writeable = False
                yield assigned


    def _assign_blocks_from_boolean_blocks(self,
            targets: tp.Iterable[np.ndarray],
            value=None) -> tp.Generator[np.ndarray, None, None]:
        '''Assign value into all blocks based on a Bolean arrays of shape equal to each block in these blocks, returning blocks of the same size and shape. Value is set where the Boolean is True.

        Args:
            value: Must be a single value, rather than an array
        '''
        if isinstance(value, np.ndarray):
            raise Exception('cannot assign an array with Boolean targets')
        else:
            value_dtype = np.array(value).dtype

        for block, target in zip_longest(self._blocks, targets):
            if block is None or target is None:
                raise Exception('blocks or targets do not align')

            if not target.any():
                yield block

            assigned_dtype = _resolve_dtype(value_dtype, block.dtype)

            if block.dtype == assigned_dtype:
                assigned = block.copy()
            else:
                assigned = block.astype(assigned_dtype)

            assert assigned.shape == target.shape
            assigned[target] = value
            yield assigned


    def _slice_blocks(self,
            row_key=None,
            column_key=None) -> tp.Generator[np.ndarray, None, None]:
        '''
        Generator of sliced blocks, given row and column key selectors.
        The result is suitable for pass to TypeBlocks constructor.
        '''
        single_row = False
        if isinstance(row_key, int):
            single_row = True
        elif isinstance(row_key, _KEY_ITERABLE_TYPES) and len(row_key) == 1:
            # an iterable of index integers is expected here
            single_row = True
        elif isinstance(row_key, slice):
            # need to determine if there is only one index returned by range (after getting indices from the slice); do this without creating a list/tuple, or walking through the entire range; get constant time look-up of range length after uses slice.indicies
            if len(range(*row_key.indices(self._shape[0]))) == 1:
                single_row = True
        elif isinstance(row_key, np.ndarray) and row_key.dtype == bool:
            # TODO: need fastest way to find if there is more than one boolean
            if row_key.sum() == 1:
                single_row = True

        # convert column_key into a series of block slices; we have to do this as we stride blocks; do not have to convert row_key as can use directly per block slice
        for block_idx, slc in self._key_to_block_slices(column_key):
            b = self._blocks[block_idx]
            if b.ndim == 1: # given 1D array, our row key is all we need
                if row_key is None:
                    block_sliced = b
                else:
                    block_sliced = b[row_key]
            else: # given 2D, use row key and column slice
                if row_key is None:
                    block_sliced = b[:, slc]
                else:
                    block_sliced = b[row_key, slc]

            # optionally, apply additional selection, reshaping, or adjustments to what we got out of the block
            if isinstance(block_sliced, np.ndarray):
                # if we have a single row and the thing we sliced is 1d, we need to rotate it
                if single_row and block_sliced.ndim == 1:
                    block_sliced = block_sliced.reshape(1, block_sliced.shape[0])
                # if we have a single column as 2d, unpack it; however, we have to make sure this is not a single row in a 2d
                elif (block_sliced.ndim == 2
                        and block_sliced.shape[0] == 1
                        and not single_row):
                    block_sliced = block_sliced[0]
            else: # a single element, wrap back up in array
                block_sliced = np.array((block_sliced,), dtype=b.dtype)

            yield block_sliced


    def _extract_array(self,
            row_key=None,
            column_key=None) -> np.ndarray:
        '''Alternative extractor that returns just an np array, concatenating blocks as necessary. Used by internal clients that want to process row/column with an array.
        '''
        # identifying column_key as integer, then we only access one block, and can return directly without iterating over blocks
        if isinstance(column_key, int):
            block_idx, column = self._index[column_key]
            b = self._blocks[block_idx]
            if b.ndim == 1:
                if row_key is None:
                    return b
                return b[row_key]
            if row_key is None:
                return b[:, column]
            return b[row_key, column]

        # pass a generator to from_block; will return a TypeBlocks or a single element
        # TODO: figure out shape from keys so as to not accumulate?
        blocks = []
        rows = 0
        columns = 0
        for b in tuple(self._slice_blocks( # a generator
                row_key=row_key,
                column_key=column_key)):
            if b.ndim == 1: # it is a single column
                if not rows: # assume all the same after first
                    # if 1d, then the length should be the number of rows
                    rows = b.shape[0]
                columns += 1
            else:
                if not rows: # assume all the same after first
                    rows = b.shape[0]
                columns += b.shape[1]
            blocks.append(b)

        return self._blocks_to_array(
                blocks=blocks,
                shape=(rows, columns),
                row_dtype=self._row_dtype)

    def _extract(self,
            row_key: GetItemKeyType=None,
            column_key: GetItemKeyType=None) -> 'TypeBlocks': # but sometimes an element
        '''
        Return a TypeBlocks after performing row and column selection using iloc selection.

        Row and column keys can be:
            integer: single row/column selection
            slices: one or more contiguous selections
            iterable of integers: one or more non-contiguous and/or repeated selections

        Note: Boolean-based selection is not (yet?) implemented here, but instead will be implemented at the `loc` level. This might imply that Boolean selection is only available with `loc`.

        Returns:
            TypeBlocks, or a single element if both are coordinats
        '''
        # identifying column_key as integer, then we only access one block, and can return directly without iterating over blocks
        if isinstance(column_key, int):
            block_idx, column = self._index[column_key]
            b = self._blocks[block_idx]
            if b.ndim == 1:
                if row_key is None: # return a column
                    return TypeBlocks.from_blocks(b)
                elif isinstance(row_key, int):
                    return b[row_key] # return single item
                return TypeBlocks.from_blocks(b[row_key])

            if row_key is None: # return a column
                return TypeBlocks.from_blocks(b[:, column])
            elif isinstance(row_key, int):
                return b[row_key, column] # return single item
            return TypeBlocks.from_blocks(b[row_key, column])

        # pass a generator to from_block; will return a TypeBlocks or a single element
        return self.from_blocks(self._slice_blocks(
                row_key=row_key,
                column_key=column_key))


    def _extract_iloc(self,
            key: GetItemKeyTypeCompound) -> 'TypeBlocks':
        if self.unified:
            # perform slicing directly on block if possible
            return self.from_blocks(self._blocks[0][key])
        if isinstance(key, tuple):
            return self._extract(*key)
        return self._extract(row_key=key)

    def extract_iloc_mask(self,
            key: GetItemKeyTypeCompound) -> 'TypeBlocks':
        if isinstance(key, tuple):
            return TypeBlocks.from_blocks(self._mask_blocks(*key))
        return TypeBlocks.from_blocks(self._mask_blocks(row_key=key))

    def extract_iloc_assign(self,
            key: GetItemKeyTypeCompound,
            value) -> 'TypeBlocks':
        if isinstance(key, tuple):
            return TypeBlocks.from_blocks(self._assign_blocks_from_keys(*key, value=value))
        return TypeBlocks.from_blocks(self._assign_blocks_from_keys(row_key=key, value=value))


    def __getitem__(self, key) -> 'TypeBlocks':
        '''
        Returns a column, or a column slice.
        '''
        # NOTE: if key is a tuple it means that multiple indices are being provided; this should probably raise an error
        if isinstance(key, tuple):
            raise KeyError('__getitem__ does not support multiple indexers')
        return self._extract(row_key=None, column_key=key)

    #---------------------------------------------------------------------------
    # operators

    def _ufunc_unary_operator(self, operator: tp.Callable) -> 'TypeBlocks':
        # for now, do no reblocking; though, in many cases, operating on a unified block will be faster
        def operation():
            for b in self._blocks:
                result = operator(b)
                result.flags.writeable = False
                yield result

        return self.from_blocks(operation())

    #---------------------------------------------------------------------------

    def _block_shape_slices(self) -> tp.Generator[slice, None, None]:
        '''Generator of slices necessary to slice a 1d array of length equal to the number of columns into a lenght suitable for each block.
        '''
        start = 0
        for b in self._blocks:
            end = start + (1 if b.ndim == 1 else b.shape[1])
            yield slice(start, end)
            start = end

    def _ufunc_binary_operator(self, *, operator: tp.Callable, other) -> 'TypeBlocks':
        if isinstance(other, TypeBlocks):
            if self.block_compatible(other):
                # this means that the blocks are the same size; we do not check types
                self_opperands = self._blocks
                other_opperands = other._blocks
            elif self._shape == other._shape:
                # if the result of reblock does not result in compatible shapes, we have to use .values as opperands; the dtypes can be different so we only have to check that they columns sizes, the second element of the signature, all match.
                if not self.reblock_compatible(other):
                    self_opperands = (self.values,)
                    other_opperands = (other.values,)
                else:
                    self_opperands = self._reblock()
                    other_opperands = other._reblock()
            else: # raise same error as NP
                raise NotImplementedError('cannot apply binary operators to arbitrary TypeBlocks')

            def operation():
                for a, b in zip_longest(
                        (self.single_column_filter(op) for op in self_opperands),
                        (self.single_column_filter(op) for op in other_opperands)
                        ):
                    result = operator(a, b)
                    result.flags.writeable = False # own the data
                    yield result
        else:
            # process other as an array
            self_opperands = self._blocks
            if not isinstance(other, np.ndarray):
                # this maybe expensive for a single scalar
                other = np.array(other) # this will work with a single scalar too

            # handle dimensions
            if other.ndim == 0 or (other.ndim == 1 and len(other) == 1):
                # a scalar: reference same value for each block position
                other_opperands = (other for _ in range(len(self._blocks)))
            elif other.ndim == 1 and len(other) == self._shape[1]:
                # if given a 1d array
                # one dimensional array of same size: chop to block width
                other_opperands = (other[s] for s in self._block_shape_slices())
            else:
                raise NotImplementedError('cannot apply binary operators to arbitrary np arrays.')

            def operation():
                for a, b in zip_longest(self_opperands, other_opperands):
                    result = operator(a, b)
                    result.flags.writeable = False # own the data
                    yield result

        return self.from_blocks(operation())



    def _ufunc_axis_skipna(self, *, axis, skipna, ufunc, ufunc_skipna, dtype):
        # not sure if these make sense on TypeBlocks, as they reduce dimensionality
        raise NotImplementedError()

    #---------------------------------------------------------------------------
    # transformations resulting in the same dimensionality

    def transpose(self) -> 'TypeBlocks':
        '''Return a new TypeBlocks that transposes and concatenates all blocks.
        '''
        blocks = []
        for b in self._blocks:
            b = self.single_column_filter(b).transpose()
            if b.dtype != self._row_dtype:
                b = b.astype(self._row_dtype)
            blocks.append(b)
        a = np.concatenate(blocks)
        a.flags.writeable = False # keep this array
        return self.from_blocks(a)


    def isna(self) -> 'TypeBlocks':
        '''Return a Boolean TypeBlocks where True is NaN or None.
        '''
        def blocks():
            for b in self._blocks:
                bool_block = _isna(b)
                bool_block.flags.writeable = False
                yield bool_block

        return self.from_blocks(blocks())


    def notna(self) -> 'TypeBlocks':
        '''Return a Boolean TypeBlocks where True is not NaN or None.
        '''
        def blocks():
            for b in self._blocks:
                bool_block = np.logical_not(_isna(b))
                bool_block.flags.writeable = False
                yield bool_block

        return self.from_blocks(blocks())


    def dropna_to_keep_locations(self,
            axis: int=0,
            condition: tp.Callable[[np.ndarray], bool]=np.all) -> 'TypeBlocks':
        '''
        Return the row and column slices to extract the new TypeBlock. This is to be used by Frame, where the slices will be needed on the indices as well.

        Args:
            axis: Dimension to drop, where 0 will drop rows and 1 will drop columns based on the condition function applied to a Boolean array.
        '''
        # get a unified boolean array; as iisna will always return a Boolean, we can simply take the firtst block out of consolidation
        unified = next(self.consolidate_blocks(_isna(b) for b in self._blocks))

        # flip axis to condition funcion
        condition_axis = 0 if axis else 1
        to_drop = condition(unified, axis=condition_axis)
        to_keep = np.logical_not(to_drop)

        if axis == 1:
            row_key = None
            column_key = to_keep
        else:
            row_key = to_keep
            column_key = None

        return row_key, column_key


    def fillna(self, value) -> 'TypeBlocks':
        '''
        Return a new TypeBlocks instance that fills missing values with the passed value.
        '''
        return self.from_blocks(
                self._assign_blocks_from_boolean_blocks(
                        targets=(_isna(b) for b in self._blocks),
                        value=value)
                )


    #---------------------------------------------------------------------------
    # mutate

    def append(self, block: np.ndarray):
        '''Add a block; an array copy will not be made unless the passed in block is not immutable'''
        # shape can be 0, 0 if empty
        row_count = self._shape[0]

        # update shape
        if block.ndim == 1:
            if row_count:
                assert len(block) == row_count, 'mismatched row count'
            else:
                row_count = len(block)
            block_columns = 1
        else:
            if row_count:
                assert block.shape[0] == row_count, 'mismatched row count'
            else:
                row_count = block.shape[0]
            block_columns = block.shape[1]


        # extend shape, or define it if not yet set
        self._shape = (row_count, self._shape[1] + block_columns)

        # add block, dtypes, index
        block_idx = len(self._blocks) # next block
        for i in range(block_columns):
            self._index.append((block_idx, i))
            self._dtypes.append(block.dtype)

        # make immutable copy if necessary before appending
        self._blocks.append(self.immutable_filter(block))

        # if already aligned, nothing to do
        if not self._row_dtype: # if never set as shape is empty
            self._row_dtype = block.dtype
        elif block.dtype != self._row_dtype:
            # we do not use _resolve_dtype here as we want to preserve types, not safely cooerce them (i.e., int to float)
            self._row_dtype = object

    def extend(self, other: 'TypeBlocks'):
        '''Extend this TypeBlock with the contents of another.
        '''

        if isinstance(other, TypeBlocks):
            if self._shape[0]:
                assert self._shape[0] == other._shape[0]
            blocks = other._blocks
        else: # accept iterables of np.arrays
            blocks = other
        # row count must be the same
        for block in blocks:
            self.append(block)


#-------------------------------------------------------------------------------
class LocMap:

    @staticmethod
    def map_slice_args(label_to_pos, key: slice):
        '''Given a slice and a label to position mapping, yield each argument necessary to create a new slice.

        Args:
            label_to_pos: mapping, no order dependency
        '''
        # TODO: just iter over (key.start) etc.
        for field in ('start', 'stop', 'step'):
            attr = getattr(key, field)
            if attr is None:
                yield None
            else:
                if field == 'stop':
                    # loc selections are inclusive, so iloc gets one more
                    yield label_to_pos[attr] + 1
                else:
                    yield label_to_pos[attr]

    @classmethod
    def loc_to_iloc(cls,
            label_to_pos: tp.Dict,
            positions: np.ndarray,
            key: GetItemKeyType) -> GetItemKeyType:
        '''
        Returns:
            A integer mapped slice, or GetItemKey type that is based on integers, compatible with TypeBlocks
        '''

        if isinstance(key, slice):
            return slice(*cls.map_slice_args(label_to_pos, key))

        elif isinstance(key, _KEY_ITERABLE_TYPES):

            # can be an iterable of labels (keys) or an iterable of Booleans
            # if len(key) == len(label_to_pos) and isinstance(key[0], (bool, np.bool_)):
            if isinstance(key, np.ndarray) and key.dtype == bool:
                return positions[key]

            # map labels to integer positions
            # NOTE: we miss the opportunity to get a reference from values when we have contiguous keys
            return [label_to_pos[x] for x in key]

        # if a single element (an integer, string, or date, we just get the integer out of the map
        return label_to_pos[key]



#-------------------------------------------------------------------------------

class Index(metaclass=MetaOperatorDelegate):
    '''A mapping of labels to positions, immutable and of fixed size. Used in :py:class:`Series` and as index and columns in :py:class:`Frame`.

    Args:
        labels: Iterable of values to be used as the index.
        loc_is_iloc: Optimization for when a contiguous integer index is provided as labels. Generally only set by internal clients.
        dtype: Optional dytpe to be used for labels.
    '''

    __slots__ = (
            '_map',
            '_labels',
            '_positions',
            '_recache',
            '_loc_is_iloc',
            'loc',
            'iloc',
            )

    #---------------------------------------------------------------------------
    # methods used in __init__ that are customized in dervied classes; there, we need to mutate instance state, this these are instance methods
    @staticmethod
    def _extract_labels(
            mapping,
            labels,
            dtype=None) -> tp.Tuple[tp.Iterable[int], tp.Iterable[tp.Any]]:
        '''Derive labels, a cache of the mapping keys in a sequence type (either an ndarray or a list).

        If the labels passed at instantiation are an ndarray, they are used after immutable filtering. Otherwise, the mapping keys are used to create an ndarray.

        This method is overridden in the derived class.

        Args:
            labels: might be an expired Generator, but if it is an immutable npdarry, we can use it without a copy
        '''
        # pre-fetching labels for faster get_item construction
        if isinstance(labels, np.ndarray): # if an np array can handle directly
            labels = TypeBlocks.immutable_filter(labels)
        elif hasattr(labels, '__len__'): # not a generator, not an array
            labels = np.array(labels, dtype)
            labels.flags.writeable = False
        else: # labels may be an expired generator
            # until all Python dictionaries are ordered, we cannot just take keys()
            # labels = np.array(tuple(mapping.keys()))
            # assume object type so as to not create a temporary list
            labels = np.empty(len(mapping),
                    dtype=dtype if dtype else object)
            for k, v in mapping.items():
                labels[v] = k
            labels.flags.writeable = False

        return labels

    @staticmethod
    def _extract_positions(
            mapping,
            positions):
        # positions is either None or an ndarray
        if isinstance(positions, np.ndarray): # if an np array can handle directly
            return TypeBlocks.immutable_filter(positions)
        positions = np.arange(len(mapping))
        positions.flags.writeable = False
        return positions


    def _update_array_cache(self):
        '''Derived classes can use this to set stored arrays, self._labels and self._positions.
        '''
        pass

    #---------------------------------------------------------------------------
    def __init__(self,
            labels: IndexInitializer,
            loc_is_iloc: bool=False,
            dtype: DtypeSpecifier=None) -> None:

        self._recache = False

        positions = None

        if issubclass(labels.__class__, Index):
            # get a reference to the immutable arrays
            # even if this is an IndexGO index, we can take the cached arrays, assuming they are up to date
            if labels._recache:
                labels._update_array_cache()
            positions = labels._positions
            loc_is_iloc = labels._loc_is_iloc
            labels = labels._labels

        # map provided values to integer positions; do only one iteration of labels to support generators
        # collections.abs.sized
        if hasattr(labels, '__len__'):
            # dict() function shown to be faster then gen expression
            if positions is not None:
                self._map = dict(zip(labels, positions))
            else:
                self._map = dict(zip(labels, range(len(labels))))
        else: # handle generators
            # dict() function shown slower in this case
            # self._map = dict((v, k) for k, v in enumerate(labels))
            self._map = {v: k for k, v in enumerate(labels)}

        # this might be NP array, or a list, depending on if static or grow only
        self._labels = self._extract_labels(self._map, labels, dtype)
        self._positions = self._extract_positions(self._map, positions)

        # NOTE:  automatic discovery is possible but not sure that the cost of a
        self._loc_is_iloc = loc_is_iloc

        if len(self._map) != len(self._labels):
            raise KeyError('labels have non-unique values')

        self.loc = GetItem(self._extract_loc)
        self.iloc = GetItem(self._extract_iloc)


    def display(self, config: DisplayConfig=None) -> Display:
        config = config or DisplayActive.get()

        if self._recache:
            self._update_array_cache()

        return Display.from_values(self.values,
                header='<' + self.__class__.__name__ + '>',
                config=config)

    def __repr__(self) -> str:
        return repr(self.display())


    def loc_to_iloc(self, key: GetItemKeyType) -> GetItemKeyType:
        '''
        Returns:
            Return GetItemKey type that is based on integers, compatible with TypeBlocks
        '''
        if isinstance(key, Series):
            key = key.values
        if self._recache:
            self._update_array_cache()

        if self._loc_is_iloc:
            return key

        return LocMap.loc_to_iloc(self._map, self._positions, key)

    def __len__(self) -> int:
        if self._recache:
            self._update_array_cache()
        return len(self._labels)


    def __iter__(self):
        '''We iterate over labels.
        '''
        if self._recache:
            self._update_array_cache()
        return self._labels.__iter__()

    def __contains__(self, value) -> bool:
        '''Return True if value in the labels.
        '''
        return self._map.__contains__(value)


    @property
    def values(self) -> np.ndarray:
        '''Return the immutable labels array
        '''
        if self._recache:
            self._update_array_cache()
        return self._labels

    @property
    def mloc(self):
        '''Memory location
        '''
        if self._recache:
            self._update_array_cache()
        return mloc(self._labels)

    def copy(self) -> 'Index':
        # this is not a complete deepcopy, as _labels here is an immutable np array (a new map will be created); if this is an IndexGO, we will pass the cached, immutable NP array
        if self._recache:
            self._update_array_cache()
        return self.__class__(labels=self._labels)


    def relabel(self, mapper: CallableOrMapping) -> 'Index':
        '''
        Return a new Index with labels replaced by the callable or mapping; order will be retained. If a mapping is used, the mapping need not map all origin keys.
        '''
        if not callable(mapper):
            # if a mapper, it must support both __getitem__ and __contains__
            getitem = getattr(mapper, '__getitem__')
            return self.__class__(getitem(x) if x in mapper else x for x in self._labels)

        return self.__class__(mapper(x) for x in self._labels)

    #---------------------------------------------------------------------------
    # set operations

    def intersection(self, other) -> 'Index':
        if self._recache:
            self._update_array_cache()

        if isinstance(other, np.ndarray):
            opperand = other
        else: # assume we can get it from a .values attribute
            opperand = other.values

        return self.__class__(labels=np.intersect1d(self._labels, opperand))

    def union(self, other) -> 'Index':
        if self._recache:
            self._update_array_cache()

        if isinstance(other, np.ndarray):
            opperand = other
        else: # assume we can get it from a .values attribute
            opperand = other.values

        return self.__class__(labels=np.union1d(self._labels, opperand))

    #---------------------------------------------------------------------------
    # extraction and selection

    def _extract_iloc(self, key) -> 'Index':
        '''Extract a new index given an iloc key
        '''
        if self._recache:
            self._update_array_cache()

        if isinstance(key, slice):
            # if labels is an np array, this will be a view; if a list, a copy
            labels = self._labels[key]
        elif isinstance(key, _KEY_ITERABLE_TYPES):
            # we assume Booleans have been normalized to integers here
            # can select directly from _labels[key] if if key is a list
            labels = self._labels[key]
        elif key is None:
            labels = self._labels
        else: # select a single label value
            labels = (self._labels[key],)
        return self.__class__(labels=labels)

    def _extract_loc(self, key: GetItemKeyType) -> 'Index':
        return self._extract_iloc(self.loc_to_iloc(key))

    def __getitem__(self, key: GetItemKeyType) -> 'Index':
        '''Extract a new index given an iloc key (this is the same as Pandas).
        '''
        return self._extract_iloc(key)

    #---------------------------------------------------------------------------
    # operators

    def _ufunc_unary_operator(self, operator: tp.Callable) -> np.ndarray:
        '''Always return an NP array.
        '''
        if self._recache:
            self._update_array_cache()

        array = operator(self._labels)
        array.flags.writeable = False
        return array

    def _ufunc_binary_operator(self, *, operator: tp.Callable, other) -> np.ndarray:
        '''
        Binary operators applied to an index always return an NP array. This deviates from Pandas, where some operations (multipling an int index by an int) result in a new Index, while other operations result in a np.array (using == on two Index).
        '''
        if self._recache:
            self._update_array_cache()

        if issubclass(other.__class__, Index):
            other = other.values # operate on labels to labels
        array = operator(self._labels, other)
        array.flags.writeable = False
        return array


    def _ufunc_axis_skipna(self, *, axis, skipna, ufunc, ufunc_skipna, dtype=None):
        '''Axis argument is required but is irrelevant.

        Args:
            dtype: Not used in 1D application, but collected here to provide a uniform signature.
        '''
        return _ufunc_skipna_1d(
                array=self._labels,
                skipna=skipna,
                ufunc=ufunc,
                ufunc_skipna=ufunc_skipna)

    #---------------------------------------------------------------------------
    # utility functions

    def sort(self,
            ascending: bool=True,
            kind: str=_DEFAULT_SORT_KIND) -> 'Index':
        '''Return a new Index with the labels sorted.

        Args:
            kind: Sort algorithm passed to NumPy.
        '''
        v = np.sort(self._labels, kind=kind)
        if not ascending:
            v = v[::-1]
        v.flags.writeable = False
        return __class__(v)

    def isin(self, other: tp.Iterable[tp.Any]) -> np.ndarray:
        '''Return a Boolean array showing True where a label is found in other. If other is a multidimensional array, it is flattened.
        '''
        if self._recache:
            self._update_array_cache()
        v, assume_unique = _iterable_to_array(other)
        return np.in1d(self._labels, v, assume_unique=assume_unique)


class IndexGO(Index):
    '''
    A mapping of labels to positions, immutable with grow-only size. Used as columns in :py:class:`FrameGO`. Initialization arguments are the same as for :py:class:`Index`.
    '''

    __slots__ = (
            '_map',
            '_labels_mutable',
            '_positions_mutable_count',
            '_labels',
            '_positions',
            '_recache',
            'iloc',
            )

    def _extract_labels(self,
            mapping,
            labels,
            dtype) -> tp.Iterable[tp.Any]:
        '''Called in Index.__init__(). This creates and populates mutable storage as a side effect of array derivation.
        '''
        labels = super(IndexGO, self)._extract_labels(mapping, labels, dtype)
        self._labels_mutable = labels.tolist()
        return labels

    def _extract_positions(self, mapping, positions) -> tp.Iterable[tp.Any]:
        '''Called in Index.__init__(). This creates and populates mutable storage. This creates and populates mutable storage as a side effect of array derivation.
        '''
        positions = super(IndexGO, self)._extract_positions(mapping, positions)
        self._positions_mutable_count = len(positions)
        return positions


    def _update_array_cache(self):
        # this might fail if a sequence is given as a label
        self._labels = np.array(self._labels_mutable)
        self._labels.flags.writeable = False
        self._positions = np.arange(self._positions_mutable_count)
        self._positions.flags.writeable = False
        self._recache = False

    #---------------------------------------------------------------------------
    # grow only mutation

    def append(self, value):
        '''Add a value
        '''
        if value in self._map:
            raise KeyError('duplicate key append attempted', value)
        # the new value is the count
        self._map[value] = self._positions_mutable_count
        self._labels_mutable.append(value)

        # check value before incrementing
        if self._loc_is_iloc:
            if isinstance(value, int) and value == self._positions_mutable_count:
                pass # an increment that keeps loc is iloc relationship
            else:
                self._loc_is_iloc = False

        self._positions_mutable_count += 1
        self._recache = True


    def extend(self, values: _KEY_ITERABLE_TYPES):
        '''Add multiple values
        Args:
            values: can be a generator.
        '''
        for value in values:
            if value in self._map:
                raise KeyError('duplicate key append attempted', value)
            # might bet better performance by calling extend() on _positions and _labels
            self.append(value)



#-------------------------------------------------------------------------------
class Series(metaclass=MetaOperatorDelegate):
    '''
    A one-dimensional ordered, labelled collection, immutable and of fixed size.

    Args:
        values: An iterable of values, or a single object, to be aligned with the supplied (or automatically generated) index. Alternatively, a dictionary of index / value pairs can be provided.
        index: Option index initializer. If provided, lenght must be equal to length of values.
        own_index: Flag index as ownable by Series; primarily for internal clients.
    '''

    __slots__ = (
        'values',
        '_index',
        'iloc',
        'loc',
        'mask',
        'masked_array',
        'assign',
        'iter_group',
        'iter_group_items',
        'iter_element',
        'iter_element_items',
        )

    @classmethod
    def from_items(cls,
            pairs: tp.Iterable[tp.Tuple[tp.Hashable, tp.Any]],
            dtype: DtypeSpecifier=None) -> 'Series':
        '''Series construction from an iterator or generator of pairs, where the first value is the index and the second value is the value.

        Args:
            pairs: Iterable of pairs of index, value.
            dtype: dtype or valid dtype specifier.
        '''
        index = []
        def values():
            for pair in pairs:
                # populate index as side effect of iterating values
                index.append(pair[0])
                yield pair[1]

        return cls(values(), index=index, dtype=dtype)


    def __init__(self,
            values: SeriesInitializer,
            *,
            index: IndexInitializer=None,
            dtype: DtypeSpecifier=None,
            own_index: bool=False
            ) -> None:
        #-----------------------------------------------------------------------
        # values assignment
        # expose .values directly as it is immutable
        if not isinstance(values, np.ndarray):
            if isinstance(values, dict):
                # not sure if we should sort; not sure what to do if index is provided
                if index is not None:
                    raise Exception('cannot create Series from dictionary when index is defined')
                index = []
                def values_gen():
                    for k, v in _dict_to_sorted_items(values):
                        # populate index as side effect of iterating values
                        index.append(k)
                        yield v
                self.values = np.fromiter(values_gen(), dtype=dtype, count=len(values))

            # TODO: not sure if we need to check __iter__ here
            elif (dtype and dtype != object and dtype != str
                    and hasattr(values, '__iter__')
                    and hasattr(values, '__len__')):
                self.values = np.fromiter(values, dtype=dtype, count=len(values))
            elif hasattr(values, '__len__'):
                self.values = np.array(values, dtype=dtype)
            elif hasattr(values, '__next__'): # a generator-like
                self.values = np.array(tuple(values), dtype=dtype)
            else: # it must be a single item
                if not hasattr(index, '__len__'):
                    raise Exception('cannot create a Series from a single item if passed index has no length.')
                self.values = np.full(len(index), values, dtype=dtype)
            self.values.flags.writeable = False
        else: # is numpy
            if dtype is not None and dtype != values.dtype:
                raise Exception('type requested is not the type given') # what to do here?
            self.values = TypeBlocks.immutable_filter(values)

        #-----------------------------------------------------------------------
        # index assignment
        # NOTE: this generally must be done after values assignment, as from_items needs a values generator to be exhausted before looking to values

        if index is None: # create an integer index
            self._index = Index(range(len(self.values)), loc_is_iloc=True)
        elif own_index:
            self._index = index
        elif isinstance(index, IndexGO):
            # if a grow only index need to make immutable
            self._index = Index(index)
        elif isinstance(index, Index):
            # do not make a copy of it is an immutable index
            self._index = index
        else: # let index handle instantiation
            self._index = Index(index)

        if len(self.values) != len(self._index):
            raise Exception('values and index do not match length')

        #-----------------------------------------------------------------------
        # attributes

        self.loc = GetItem(self._extract_loc)
        self.iloc = GetItem(self._extract_iloc)

        self.mask = ExtractInterface(
                iloc=GetItem(self._extract_iloc_mask),
                loc=GetItem(self._extract_loc_mask),
                getitem=self._extract_loc_mask)

        self.masked_array = ExtractInterface(
                iloc=GetItem(self._extract_iloc_masked_array),
                loc=GetItem(self._extract_loc_masked_array),
                getitem=self._extract_loc_masked_array)

        self.assign = ExtractInterface(
                iloc=GetItem(self._extract_iloc_assign),
                loc=GetItem(self._extract_loc_assign),
                getitem=self._extract_loc_assign)


        self.iter_group = IterNode(
                container=self,
                function_items=self._axis_group_items,
                function_values=self._axis_group,
                yield_type=IterNodeType.VALUES
                )
        self.iter_group_items = IterNode(
                container=self,
                function_items=self._axis_group_items,
                function_values=self._axis_group,
                yield_type=IterNodeType.ITEMS
                )


        self.iter_element = IterNode(
                container=self,
                function_items=self._axis_element_items,
                function_values=self._axis_element,
                yield_type=IterNodeType.VALUES
                )
        self.iter_element_items = IterNode(
                container=self,
                function_items=self._axis_element_items,
                function_values=self._axis_element,
                yield_type=IterNodeType.ITEMS
                )

    #---------------------------------------------------------------------------
    # index manipulation

    def _reindex_other_like_iloc(self,
            value: 'Series',
            iloc_key: GetItemKeyType) -> 'Series':
        '''Given a value that is a Series, reindex it to the index components, drawn from this Series, that are specified by the iloc_key.
        '''
        return value.reindex(self._index._extract_iloc(iloc_key))


    def reindex(self,
            index: tp.Union[Index, tp.Sequence[tp.Any]],
            fill_value=np.nan) -> 'Series':
        '''
        Return a new Series based on the passed index.

        Args:
            fill_value: attempted to be used, but may be coerced by the dtype of this Series. `
        '''
        # TODO: implement `method` argument with bfill, ffill options

        # always use the Index constructor for safe aliasing when possible
        index = Index(index)

        ic = IndexCorrespondence.from_correspondence(self.index, index)

        if ic.is_subset: # must have some common
            return self.__class__(self.values[ic.iloc_src],
                    index=index,
                    own_index=True)

        values = _full_for_fill(self.values.dtype, len(index), fill_value)

        # if some intersection of values
        if ic.has_common:
            values[ic.iloc_dst] = self.values[ic.iloc_src]

        # make immutable so a copy is not made
        values.flags.writeable = False
        return self.__class__(values,
                index=index,
                own_index=True)

    def relabel(self, mapper: CallableOrMapping) -> 'Series':
        '''
        Return a new Series based on a mapping (or callable) from old to new index values.
        '''
        return self.__class__(self.values,
                index=self._index.relabel(mapper),
                own_index=True)

    #---------------------------------------------------------------------------
    # na handling

    def isna(self) -> 'Series':
        '''
        Return a same-indexed, Boolean Series indicating which values are NaN or None.
        '''
        # consider returning self if not values.any()?
        values = _isna(self.values)
        values.flags.writeable = False
        return self.__class__(values, index=self._index)

    def notna(self) -> 'Series':
        '''
        Return a same-indexed, Boolean Series indicating which values are NaN or None.
        '''
        values = np.logical_not(_isna(self.values))
        values.flags.writeable = False
        return self.__class__(values, index=self._index)

    def dropna(self) -> 'Series':
        '''
        Return a new Series after removing values of NaN or None.
        '''
        sel = np.logical_not(_isna(self.values))
        if not np.any(sel):
            return self

        values = self.values[sel]
        values.flags.writeable = False
        return self.__class__(values, index=self._index.loc[sel])

    def fillna(self, value) -> 'Series':
        '''Return a new Series after replacing NaN or None values with the supplied value.
        '''
        sel = _isna(self.values)
        if not np.any(sel):
            return self

        if isinstance(value, np.ndarray):
            raise Exception('cannot assign an array to fillna')
        else:
            value_dtype = np.array(value).dtype

        assigned_dtype = _resolve_dtype(value_dtype, self.values.dtype)

        if self.values.dtype == assigned_dtype:
            assigned = self.values.copy()
        else:
            assigned = self.values.astype(assigned_dtype)

        assigned[sel] = value
        assigned.flags.writeable = False
        return self.__class__(assigned, index=self._index)

    #---------------------------------------------------------------------------
    # operators

    def _ufunc_unary_operator(self, operator: tp.Callable) -> 'Series':
        return self.__class__(operator(self.values), index=self._index, dtype=self.dtype)

    def _ufunc_binary_operator(self, *, operator: tp.Callable, other) -> 'Series':

        values = self.values
        index = self._index

        if isinstance(other, Series):
            # if indices are the same, we can simply set other to values and fallback on NP
            if (self.index == other.index).all(): # this is an array
                other = other.values
            else:
                index = self.index.union(other.index)
                # now need to reindex the Series
                values = self.reindex(index).values
                other = other.reindex(index).values

        # if its an np array, we simply fall back on np behavior
        elif isinstance(other, np.ndarray):
            if other.ndim > 1:
                raise NotImplementedError('Operator application to greater dimensionalities will result in an array with more than 1 dimension; it is not clear how such an array should be indexed.')
        # permit single value constants; not sure about filtering other types

        # we want the dtype to be the result of applying the operator; this happends by default
        result = operator(values, other)
        result.flags.writeable = False
        return self.__class__(result, index=index)


    def _ufunc_axis_skipna(self, *, axis, skipna, ufunc, ufunc_skipna, dtype=None):
        '''For a Series, all functions of this type reduce the single axis of the Series to 1d, so Index has no use here.

        Args:
            dtype: not used, part of signature for a commin interface
        '''
        # following pandas convention, we replace Nones with nans so that, if skipna is False, a None can cause a nan result
        return _ufunc_skipna_1d(
                array=self.values,
                skipna=skipna,
                ufunc=ufunc,
                ufunc_skipna=ufunc_skipna)

    #---------------------------------------------------------------------------
    def __len__(self) -> int:
        '''Length of values.
        '''
        return self.values.__len__()

    def display(self, config: DisplayConfig=None) -> Display:
        '''Return a Display of the Series.
        '''
        config = config or DisplayActive.get()

        d = self._index.display(config=config)
        d.append_display(Display.from_values(
                self.values,
                header='<' + self.__class__.__name__ + '>',
                config=config))
        return d

    def __repr__(self):
        return repr(self.display())

    #---------------------------------------------------------------------------
    # common attributes from the numpy array

    @property
    def mloc(self):
        return mloc(self.values)

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def size(self):
        return self.values.size

    @property
    def nbytes(self):
        return self.values.nbytes

    #---------------------------------------------------------------------------
    # extraction

    def _extract_iloc(self, key: GetItemKeyType) -> 'Series':
        # iterable selection should be handled by NP (but maybe not if a tuple)
        return self.__class__(
                self.values[key],
                index=self._index.iloc[key])

    def _extract_loc(self, key: GetItemKeyType) -> 'Series':
        '''
        Compatibility:
            Pandas supports taking in iterables of keys, where some keys are not found in the index; a Series is returned as if a reindex operation was performed. This is undesirable. Better instead is to use reindex()
        '''
        iloc_key = self._index.loc_to_iloc(key)
        values = self.values[iloc_key]

        if not isinstance(values, np.ndarray): # if we have a single element
            return values
        # this might create new index from iloc, and then createa another Index on Index __init__
        return self.__class__(values, index=self._index.iloc[iloc_key])

    def __getitem__(self, key: GetItemKeyType) -> 'Series':
        '''A Loc selection (by index labels).

        Compatibility:
            Pandas supports using both loc and iloc style selections with the __getitem__ interface on Series. This is undesirable, so here we only expose the loc interface (making the Series dictionary like, but unlike the Index, where __getitem__ is an iloc).
        '''
        return self._extract_loc(key)

    #---------------------------------------------------------------------------
    # utilites for alternate extraction: mask and assignment

    def _extract_iloc_mask(self, key: GetItemKeyType) -> 'Series':
        '''Produce a new boolean Series of the same shape, where the values selected via iloc selection are True.
        '''
        mask = np.full(self.shape, False, dtype=bool)
        mask[key] = True
        mask.flags.writeable = False
        # can pass self here as it is immutable (assuming index cannot change)
        return self.__class__(mask, index=self._index)

    def _extract_loc_mask(self, key: GetItemKeyType) -> 'Series':
        '''Produce a new boolean Series of the same shape, where the values selected via loc selection are True.
        '''
        iloc_key = self._index.loc_to_iloc(key)
        return self._extract_iloc_mask(key=iloc_key)


    def _extract_iloc_masked_array(self, key: GetItemKeyType) -> MaskedArray:
        '''Produce a new boolean Series of the same shape, where the values selected via iloc selection are True.
        '''
        mask = self._extract_iloc_mask(key=key)
        return MaskedArray(data=self.values, mask=mask.values)

    def _extract_loc_masked_array(self, key: GetItemKeyType) -> MaskedArray:
        '''Produce a new boolean Series of the same shape, where the values selected via loc selection are True.
        '''
        iloc_key = self._index.loc_to_iloc(key)
        return self._extract_iloc_masked_array(key=iloc_key)

    #---------------------------------------------------------------------------

    def _extract_iloc_assign(self, key: GetItemKeyType) -> 'SeriesAssign':
        return SeriesAssign(data=self, iloc_key=key)

    def _extract_loc_assign(self, key: GetItemKeyType) -> 'SeriesAssign':
        iloc_key = self._index.loc_to_iloc(key)
        return SeriesAssign(data=self, iloc_key=iloc_key)

    #---------------------------------------------------------------------------
    # axis functions

    def _axis_group_items(self, *, axis=0):
        groups, locations = _array_to_groups_and_locations(self.values)
        for idx, g in enumerate(groups):
            selection = locations == idx
            yield g, self._extract_iloc(selection)

    def _axis_group(self, *, axis=0):
        yield from (x for _, x in self._axis_group_items(axis=axis))

    def _axis_element_items(self, *, axis=0):
        '''Generator of index, value pairs, equivalent to Series.items(). Rpeated to have a common signature as other axis functions.
        '''
        return zip(self._index.values, self.values)

    def _axis_element(self, *, axis=0):
        yield from (x for _, x in self._axis_element_items(axis=axis))

    #---------------------------------------------------------------------------

    @property
    def index(self):
        return self._index

    # @index.setter
    # def index(self, value):
    #     if len(value) != len(self._index):
    #         raise Exception('new index must match length of old index')
    #     self._index = Index(value)

    #---------------------------------------------------------------------------
    # dictionary-like interface

    def keys(self) -> Index:
        '''
        Iterator of index labels.
        '''
        return self._index

    def __iter__(self):
        '''
        Iterator of index labels, same as :py:meth:`Series.keys`.
        '''
        return self._index.__iter__()

    def __contains__(self, value) -> bool:
        '''
        Inclusion of value in index labels.
        '''
        return self._index.__contains__(value)

    def items(self) -> tp.Generator[tp.Tuple[tp.Any, tp.Any], None, None]:
        '''Iterator of pairs of index label and value.
        '''
        return zip(self._index.values, self.values)

    def get(self, key, default=None):
        '''
        Return the value found at the index key, else the default if the key is not found.
        '''
        if key not in self._index:
            return default
        return self.__getitem__(key)

    #---------------------------------------------------------------------------
    # transformations resulting in the same dimensionality

    def sort_index(self,
            ascending: bool=True,
            kind: str=_DEFAULT_SORT_KIND) -> 'Series':
        '''
        Return a new Series ordered by the sorted Index.
        '''
        # argsort lets us do the sort once and reuse the results
        order = np.argsort(self._index.values, kind=kind)
        if not ascending:
            order = order[::-1]

        index_values = self._index.values[order]
        index_values.flags.writeable = False
        values = self.values[order]
        values.flags.writeable = False
        return self.__class__(values, index=index_values)

    def sort_values(self,
            ascending: bool=True,
            kind: str=_DEFAULT_SORT_KIND) -> 'Series':
        '''
        Return a new Series ordered by the sorted values.
        '''
        # argsort lets us do the sort once and reuse the results
        order = np.argsort(self.values, kind=kind)
        if not ascending:
            order = order[::-1]

        index_values = self._index.values[order]
        index_values.flags.writeable = False
        values = self.values[order]
        values.flags.writeable = False
        return self.__class__(values, index=index_values)


    def isin(self, other) -> 'Series':
        '''
        Return a same-sized Boolean Series that shows if the same-positoined element is in the iterable passed to the function.
        '''
        # cannot use assume_unique because do not know if values is unique
        v, _ = _iterable_to_array(other)
        array = np.in1d(self.values, v)
        array.flags.writeable = False
        return self.__class__(array, index=self._index)

    def transpose(self) -> 'Series':
        '''The transpositon of a Series is itself.
        '''
        return self

    @property
    def T(self):
        return self.transpose()


    def duplicated(self,
            exclude_first=False,
            exclude_last=False) -> np.ndarray:
        '''
        Return a same-sized Boolean Series that shows True for all b values that are duplicated.
        '''
        # TODO: might be able to do this witnout calling .values and passing in TypeBlocks, but TB needs to support roll
        duplicates = _array_to_duplicated(self.values,
                exclude_first=exclude_first,
                exclude_last=exclude_last)
        duplicates.flags.writeable = False
        return self.__class__(duplicates, index=self._index)

    def drop_duplicated(self,
            exclude_first=False,
            exclude_last=False
            ):
        '''
        Return a Series with duplicated values removed.
        '''
        duplicates = _array_to_duplicated(self.values,
                exclude_first=exclude_first,
                exclude_last=exclude_last)
        keep = ~duplicates
        return self.__class__(self.values[keep], index=self._index[keep])

    #---------------------------------------------------------------------------
    # utility function to numpy array

    def unique(self) -> np.ndarray:
        '''
        Return a NumPy array of unqiue values.
        '''
        return np.unique(self.values)

    #---------------------------------------------------------------------------
    # export

    def to_pairs(self) -> tp.Iterable[tp.Tuple[tp.Hashable, tp.Any]]:
        '''
        Return a tuple of tuples of index label, value.
        '''
        return tuple(zip(self._index._labels, self.values))

    def to_pandas(self):
        '''
        Return a Pandas Series.
        '''
        import pandas
        return pandas.Series(self.values.copy(),
                index=self._index.values.copy())



class SeriesAssign:
    __slots__ = ('data', 'iloc_key')

    def __init__(self, *,
            data: Series,
            iloc_key: GetItemKeyType
            ) -> None:
        self.data = data
        self.iloc_key = iloc_key

    def __call__(self, value):
        if isinstance(value, Series):
            value = self.data._reindex_other_like_iloc(value, self.iloc_key).values

        if isinstance(value, np.ndarray):
            value_dtype = value.dtype
        else:
            value_dtype = np.array(value).dtype
        dtype = _resolve_dtype(self.data.dtype, value_dtype)

        if dtype == self.data.dtype:
            array = self.data.values.copy()
        else:
            array = self.data.values.astype(dtype)

        array[self.iloc_key] = value
        array.flags.writeable = False
        return Series(array, index=self.data.index)


#-------------------------------------------------------------------------------


class IterNodeApplyType(Enum):
    SERIES_ITEMS = 1
    FRAME_ELEMENTS = 2


class IterNodeType(Enum):
    VALUES = 1
    ITEMS = 2


class IterNodeDelegate:
    '''
    Delegate returned from :py:class:`IterNode`, providing iteration as well as a family of apply methods.
    '''

    __slots__ = (
            '_func_values',
            '_func_items',
            '_yield_type',
            '_apply_constructor'
            )

    def __init__(self,
            func_values,
            func_items,
            yield_type: IterNodeType,
            apply_constructor) -> None:
        '''
        Args:
            apply_constructor: Callable (generally a class) used to construct the object returned from apply(); must take an iterator of items.
        '''
        self._func_values = func_values
        self._func_items= func_items
        self._yield_type = yield_type
        self._apply_constructor = apply_constructor

    #---------------------------------------------------------------------------
    # core methods are apply_iter_items, yielding pairs of key, value

    def apply_iter_items(self,
            func: CallableOrMapping) -> tp.Generator[tp.Tuple[tp.Any, tp.Any], None, None]:
        '''
        Generator that applies function to each element iterated and yields the pair of element and the result.

        Args:
            func: A function or a mapping object that defines __getitem__ and __contains__. If a mpping is given and a value is not found in the mapping, the value is returned unchanged (this deviates from Pandas Series.map, which inserts NaNs)
        '''
        condition = None
        if not callable(func):
            # if the key is not in the map, we return the value unaltered
            condition = getattr(func, '__contains__')
            func = getattr(func, '__getitem__')

        # apply always calls the items function
        for k, v in self._func_items():
            if condition and not condition(v):
                if self._yield_type is IterNodeType.VALUES:
                    yield k, v
                else: # items, give both keys and values to function
                    yield k, (k, v)
            else:
                # depend on yield type, we determine what the passed in function expects to take
                if self._yield_type is IterNodeType.VALUES:
                    yield k, func(v)
                else: # items, give both keys and values to function
                    yield k, func(k, v)


    def apply_iter_items_parallel(self,
            func: CallableOrMapping,
            max_workers=4,
            chunksize=20,
            use_threads=False,
            ) -> tp.Generator[tp.Tuple[tp.Any, tp.Any], None, None]:

        '''
        Args:
            func: A function or a mapping object that defines __getitem__. If a mapping is given all values must be found in the mapping (this deviates from Pandas Series.map, which inserts NaNs)
        '''
        pool_executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

        if not callable(func):
            func = getattr(func, '__getitem__')

        # use side effect list population to create keys when iterating over values
        func_keys = []
        if self._yield_type is IterNodeType.VALUES:
            def arg_gen():
                for k, v in self._func_items():
                    func_keys.append(k)
                    yield v
        else:
            def arg_gen():
                for k, v in self._func_items():
                    func_keys.append(k)
                    yield k, v

        with pool_executor(max_workers=max_workers) as executor:
            yield from zip(func_keys,
                    executor.map(func, arg_gen(), chunksize=chunksize)
                    )

    #---------------------------------------------------------------------------
    # utility interfaces

    def apply_iter(self,
            func: CallableOrMapping,
            dtype=None) -> tp.Generator[tp.Any, None, None]:
        '''
        Generator that applies the passed function to each element iterated and yields the result.
        '''
        yield from (v for _, v in self.apply_iter_items(func=func))


    def apply(self,
            func: CallableOrMapping,
            dtype=None,
            max_workers=None,
            chunksize=20,
            use_threads=False,
            ) -> tp.Union[Series, 'Frame']:
        '''
        Apply passed function to each object iterated, where the object depends on the creation of this instance.
        '''
        if max_workers:
            return self._apply_constructor(
                    self.apply_iter_items_parallel(
                            func=func,
                            max_workers=max_workers,
                            chunksize=chunksize,
                            use_threads=use_threads),
                    dtype=dtype)

        return self._apply_constructor(
                self.apply_iter_items(func=func),
                dtype=dtype)

    def __iter__(self):
        '''
        Return a generator based on the yield type.
        '''
        if self._yield_type is IterNodeType.VALUES:
            yield from self._func_values()
        else:
            yield from self._func_items()


class IterNode:
    '''Iterface to a type of iteration on :py:class:`Series` and :py:class:`Frame`.
    '''
    # '''Stores two version of a generator function: one to yield single values, another to yield items pairs. The latter is needed in all cases, as when we use apply we return a Series, and need to have recourse to an index.
    # '''

    __slots__ = ('_container',
            '_func_values',
            '_func_items',
            '_yield_type',
            '_apply_type'
            )

    def __init__(self, *,
            container: tp.Union[Series, 'Frame'],
            function_values,
            function_items,
            yield_type: IterNodeType,
            apply_type: IterNodeApplyType=IterNodeApplyType.SERIES_ITEMS
            ) -> None:
        self._container = container
        self._func_values = function_values
        self._func_items = function_items
        self._yield_type = yield_type
        self._apply_type = apply_type

    def __call__(self, *args, **kwargs):
        '''
        In usage as an iteator, the args passed here are expected to be argument for the core iterators, i.e., axis arguments.
        '''
        func_values = partial(self._func_values, *args, **kwargs)
        func_items = partial(self._func_items, *args, **kwargs)

        if self._apply_type is IterNodeApplyType.SERIES_ITEMS:
            apply_constructor = Series.from_items
        elif self._apply_type is IterNodeApplyType.FRAME_ELEMENTS:
            apply_constructor = partial(Frame.from_element_loc_items,
                    index=self._container._index,
                    columns=self._container._columns)
        else:
            raise NotImplementedError()

        return IterNodeDelegate(
                func_values=func_values,
                func_items=func_items,
                yield_type=self._yield_type,
                apply_constructor=apply_constructor
                )


class Frame(metaclass=MetaOperatorDelegate):
    '''
    A two-dimensional ordered, labelled collection, immutable and of fixed size.

    Args:
        data: An iterable of row iterables, a 2D numpy array, or dictionary mapping column names to column values.
        index: Iterable of index labels, equal in length to the number of records.
        columns: Iterable of column labels, equal in length to the length of each row.
        own_data: Flag data as ownable by Frame; primarily for internal clients.
        own_index: Flag index as ownable by Frame; primarily for internal clients.
        own_columns: Flag columns as ownable by Frame; primarily for internal clients.

    '''

    __slots__ = (
            '_blocks',
            '_columns',
            '_index',
            'iloc',
            'loc',
            'mask',
            'masked_array',
            'assign',
            'iter_array',
            'iter_array_items',
            'iter_tuple',
            'iter_tuple_items',
            'iter_series',
            'iter_series_items',
            'iter_group',
            'iter_group_items',
            'iter_element',
            'iter_element_items'
            )

    _COLUMN_CONSTRUCTOR = Index


    @classmethod
    def from_concat(cls,
            frames: tp.Iterable['Frame'],
            axis: int=0,
            union: bool=True,
            index: IndexInitializer=None,
            columns: IndexInitializer=None):
        '''
        Concatenate multiple Frames into a new Frame. If index or columns are provided and appropriately sized, the resulting Frame will have those indices. If the axis along concatenation (index for axis 0, columns for axis 1) is unique after concatenation, it will be preserved.

        Args:
            frames: Iterable of Frames.
            axis: Integer specifying 0 to concatenate vertically, 1 to concatenate horizontally.
            union: If True, the union of the aligned indices is used; if False, the intersection is used.
            index: Optionally specify a new index.
            columns: Optionally specify new columns.
        '''
        # TODO: should this upport Series?

        if union:
            ufunc = np.union1d
        else:
            ufunc = np.intersect1d

        if axis == 1:
            # index can be the same, columns must be redefined if not unique
            if columns is None:
                # if these are different types unexpected things could happen
                columns = np.concatenate([frame._columns.values for frame in frames])
                columns.flags.writeable = False
                if len(np.unique(columns)) != len(columns):
                    raise Exception('Column names after concatenation are not unique; supply a columns argument.')
            if index is None:
                index = _array_set_ufunc_many(
                        (frame._index.values for frame in frames),
                        ufunc=ufunc)
                index.flags.writeable = False

            def blocks():
                for frame in frames:
                    if len(frame.index) != len(index) or (frame.index != index).any():
                        frame = frame.reindex(index=index)
                    for block in frame._blocks._blocks:
                        yield block

            blocks = TypeBlocks.from_blocks(blocks())
            return cls(blocks, index=index, columns=columns, own_data=True)

        elif axis == 0:
            if index is None:
                # if these are different types unexpected things could happen
                index = np.concatenate([frame._index.values for frame in frames])
                index.flags.writeable = False
                if len(np.unique(index)) != len(index):
                    raise Exception('Index names after concatenation are not unique; supply an index argument.')

            if columns is None:
                columns = _array_set_ufunc_many(
                        (frame._columns.values for frame in frames),
                        ufunc=ufunc)
                columns.flags.writeable = False

            def blocks():
                aligned_frames = []
                previous_frame = None
                block_compatible = True
                reblock_compatible = True

                for frame in frames:
                    if len(frame.columns) != len(columns) or (frame.columns != columns).any():
                        frame = frame.reindex(columns=columns)
                    aligned_frames.append(frame)
                    # column size is all the same by this point
                    # NOTE: this could be implemented on TypeBlock as a vstack opperations
                    if previous_frame is not None:
                        if block_compatible:
                            block_compatible &= frame._blocks.block_compatible(
                                    previous_frame._blocks)
                        if reblock_compatible:
                            reblock_compatible &= frame._blocks.reblock_compatible(
                                    previous_frame._blocks)
                    previous_frame = frame

                if block_compatible or reblock_compatible:
                    if not block_compatible and reblock_compatible:
                        type_blocks = [f._blocks.consolidate() for f in aligned_frames]
                    else:
                        type_blocks = [f._blocks for f in aligned_frames]

                    # all TypeBlocks have the same number of blocks by here
                    for block_idx in range(len(type_blocks[0]._blocks)):
                        block_parts = []
                        for frame_idx in range(len(type_blocks)):
                            b = TypeBlocks.single_column_filter(
                                    type_blocks[frame_idx]._blocks[block_idx])
                            block_parts.append(b)
                        array = np.vstack(block_parts)
                        array.flags.writeable = False
                        yield array
                else:
                    # must just combine .values
                    array = np.vstack(frame.values for frame in frames)
                    array.flags.writeable = False
                    yield array

            blocks = TypeBlocks.from_blocks(blocks())
            return cls(blocks, index=index, columns=columns, own_data=True)

        else:
            raise NotImplementedError('no support for axis', axis)

    @classmethod
    def from_records(cls,
            records: tp.Iterable[tp.Any],
            *,
            index: IndexInitializer,
            columns: IndexInitializer):
        '''Frame constructor from an iterable of rows.

        Args:
            records: Iterable of row values.
            index: Iterable of index labels, equal in length to the number of records.
            columns: Iterable of column labels, equal in length to the length of each row.
        '''
        # derive_columns = False
        if columns is None:
            # derive_columns = True
            columns = [] # TODO: not sure if this works

        # if records is np; we can just pass it to constructor, as is alrady a consolidate type
        if isinstance(records, np.ndarray):
            return cls(records, index=index, columns=columns)

        def blocks():
            rows = list(records)
            # derive types form first rows, but cannot do strings
            # string type requires size, so cannot use np.fromiter
            types = [(type(x) if not isinstance(x, str) else None) for x in rows[0]]
            row_count = len(rows)
            for idx in range(len(rows[0])):
                column_type = types[idx]
                if column_type is None:
                    values = np.array([row[idx] for row in rows])
                else:
                    values = np.fromiter(
                            (row[idx] for row in rows),
                            count=row_count,
                            dtype=column_type)
                values.flags.writeable = False
                yield values

        return cls(TypeBlocks.from_blocks(TypeBlocks.consolidate_blocks(blocks())),
                index=index,
                columns=columns,
                own_data=True)


    @classmethod
    def from_items(cls,
            pairs: tp.Iterable[tp.Tuple[tp.Hashable, tp.Iterable[tp.Any]]],
            *,
            index: IndexInitializer=None):
        '''Frame constructor from an iterator or generator of pairs, where the first value is the column name and the second value an iterable of column values.

        Args:
            pairs: Iterable of pairs of column name, column values.
            index: Iterable of values to create an Index.
        '''
        columns = []
        def blocks():
            for k, v in pairs:
                columns.append(k) # side effet of generator!
                # if hasattr(v, 'values'): # its could be Series or Frame
                #     values = v.values
                #     assert isinstance(values, np.ndarray)
                #     yield values
                if isinstance(v, np.ndarray):
                    yield v
                else:
                    values = np.array(v)
                    values.flags.writeable = False
                    yield values


        return cls(TypeBlocks.from_blocks(TypeBlocks.consolidate_blocks(blocks())),
                index=index,
                columns=columns,
                own_data=True)

    @classmethod
    def from_structured_array(cls,
            array: np.ndarray,
            *,
            index_column: tp.Optional[IndexSpecifier]=None) -> 'Frame':
        '''
        Convert a NumPy structed array into a Frame.

        Args:
            array: Structured numpy array.
            index_column: Optionally provide the name or position offset of the column to use as the index.
        '''

        names = array.dtype.names
        if isinstance(index_column, int):
            index_name = names[index_column]
        else:
            index_name = index_column

        # assign in generator; requires  reading through gen first
        index_array = None
        # cannot use names of we remove an index; might be a more efficient way as we kmnow the size
        columns = []

        def blocks():
            for name in names:
                if name == index_name:
                    nonlocal index_array
                    index_array = array[name]
                    continue
                columns.append(name)
                # this is not expected to make a copy
                yield array[name]

        return cls(TypeBlocks.from_blocks(TypeBlocks.consolidate_blocks(blocks())),
                columns=columns,
                index=index_array,
                own_data=True)

    @classmethod
    def from_element_iloc_items(cls,
            items,
            *,
            index,
            columns,
            dtype):
        '''
        Given an iterable of pairs of iloc coordinates and values, populate a Frame as defined by the given index and columns. Dtype must be specified.
        '''
        index = Index(index)
        columns = cls._COLUMN_CONSTRUCTOR(columns)
        tb = TypeBlocks.from_element_items(items,
                shape=(len(index), len(columns)),
                dtype=dtype)
        return cls(tb,
                index=index,
                columns=columns,
                own_data=True,
                own_index=True,
                own_columns=True)

    @classmethod
    def from_element_loc_items(cls,
            items,
            *,
            index,
            columns,
            dtype=None):
        index = Index(index)
        columns = cls._COLUMN_CONSTRUCTOR(columns)
        items = (((index.loc_to_iloc(k[0]), columns.loc_to_iloc(k[1])), v)
                for k, v in items)

        dtype = dtype if dtype is not None else object
        tb = TypeBlocks.from_element_items(items,
                shape=(len(index), len(columns)),
                dtype=dtype)
        return cls(tb,
                index=index,
                columns=columns,
                own_data=True,
                own_index=True,
                own_columns=True)

    @classmethod
    def from_csv(cls,
            fp: FilePathOrFileLike,
            *,
            delimiter: str=',',
            index_column: tp.Optional[tp.Union[int, str]]=None,
            skip_header: int=0,
            skip_footer: int=0,
            header_is_columns: bool=True,
            quote_char: str='"',
            dtype: DtypeSpecifier=None,
            encoding: tp.Optional[str]=None
            ) -> 'Frame':
        '''
        Create a Frame from a file path or a file-like object defining a delimited (CSV, TSV) data file.

        Args:
            fp: A file path or a file-like object.
            delimiter: The character used to seperate row elements.
            index_column: Optionally specify a column, by position or name, to become the index.
            skip_header: Number of leading lines to skip.
            skip_footer: Numver of trailing lines to skip.
            header_is_columns: If True, columns names are read from the first line after the first skip_header lines.
            dtype: set to None by default to permit discovery
        '''
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html

        delimiter_native = '\t'

        if delimiter != delimiter_native:
            # this is necessary if there are quoted cells that include the delimiter
            def to_tsv():
                if isinstance(fp, str):
                    with open(fp, 'r') as f:
                        for row in csv.reader(f, delimiter=delimiter, quotechar=quote_char):
                            yield delimiter_native.join(row)
                else:
                    # handling file like object works for stringio but not for bytesio
                    for row in csv.reader(fp, delimiter=delimiter, quotechar=quote_char):
                        yield delimiter_native.join(row)

            file_like = to_tsv()
        else:
            file_like = fp


        array = np.genfromtxt(file_like,
                delimiter=delimiter_native,
                skip_header=skip_header,
                skip_footer=skip_footer,
                names=header_is_columns,
                dtype=dtype,
                encoding=encoding,
                invalid_raise=False,
                missing_values={''},
                )
        # can own this array so set it as immutable
        array.flags.writeable = False
        return cls.from_structured_array(array,
                index_column=index_column,
                )

    @classmethod
    def from_tsv(cls, fp, **kwargs):
        '''
        Specialized version of :py:meth:`Frame.from_csv` for TSV files.
        '''
        return cls.from_csv(fp, delimiter='\t', **kwargs)

    #---------------------------------------------------------------------------

    def __init__(self,
            data: FrameInitializer=None,
            *,
            index: IndexInitializer=None,
            columns: IndexInitializer=None,
            own_data: bool=False,
            own_index: bool=False,
            own_columns: bool=False
            ) -> None:
        '''
        Args:
            own_data: if True, assume that the data being based in can be owned entirely by this Frame; that is, that a copy does not need to made.
        '''
        if isinstance(data, TypeBlocks):
            if own_data:
                self._blocks = data
            else:
                # assume we need to create a new TB instance; this will not copy underlying arrays as all blocks are immutable
                self._blocks = TypeBlocks.from_blocks(data._blocks)

        elif isinstance(data, dict):
            if columns is not None:
                raise Exception('cannot create Frame from dictionary when columns is defined')

            columns = []
            def blocks():
                for k, v in _dict_to_sorted_items(data):
                    columns.append(k)
                    if isinstance(v, np.ndarray):
                        yield v
                    else:
                        values = np.array(v)
                        values.flags.writeable = False
                        yield values
            self._blocks = TypeBlocks.from_blocks(blocks())

        elif data is not None:
            def blocks():
                # need to identify Series-like things and handle as single column
                if hasattr(data, 'ndim') and data.ndim == 1:
                    # if derive_columns:
                    #     if hasattr(data, 'name') and data.name:
                    #         columns.append(data.name)
                    if hasattr('values'):
                        yield data.values
                    else:
                        yield data
                elif isinstance(data, np.ndarray):
                    if own_data:
                        data.flags.writeable = False
                    yield data
                else: # try to make it into array
                    a = np.array(data)
                    a.flags.writeable = False
                    yield a

            self._blocks = TypeBlocks.from_blocks(blocks())
        else:
            # will have shape of 0,0
            self._blocks = TypeBlocks.from_none()

        row_count, col_count = self._blocks.shape

        # columns could be an np array, or an Index instance
        if own_columns:
            self._columns = columns
        elif columns is not None:
            self._columns = self._COLUMN_CONSTRUCTOR(columns)
        else:
            self._columns = self._COLUMN_CONSTRUCTOR(range(col_count),
                    loc_is_iloc=True)
        if len(self._columns) != col_count:
            raise Exception('columns provided do not have correct size')

        if own_index:
            self._index = index
        elif index is not None:
            self._index = Index(index)
        else:
            self._index = Index(range(row_count),
                    loc_is_iloc=True)
        # permit bypassing this check if the row_count is zero
        if row_count and len(self._index) != row_count:
            raise Exception('index provided do not have correct size')

        self.loc = GetItem(self._extract_loc)
        self.iloc = GetItem(self._extract_iloc)

        self.mask = ExtractInterface(
                iloc=GetItem(self._extract_iloc_mask),
                loc=GetItem(self._extract_loc_mask),
                getitem=self._extract_getitem_mask)

        self.masked_array = ExtractInterface(
                iloc=GetItem(self._extract_iloc_masked_array),
                loc=GetItem(self._extract_loc_masked_array),
                getitem=self._extract_getitem_masked_array)

        self.assign = ExtractInterface(
                iloc=GetItem(self._extract_iloc_assign),
                loc=GetItem(self._extract_loc_assign),
                getitem=self._extract_getitem_assign)

        # generators
        self.iter_array = IterNode(
                container=self,
                function_values=self._axis_array,
                function_items=self._axis_array_items,
                yield_type=IterNodeType.VALUES
                )
        self.iter_array_items = IterNode(
                container=self,
                function_values=self._axis_array,
                function_items=self._axis_array_items,
                yield_type=IterNodeType.ITEMS
                )

        self.iter_tuple = IterNode(
                container=self,
                function_values=self._axis_tuple,
                function_items=self._axis_tuple_items,
                yield_type=IterNodeType.VALUES
                )
        self.iter_tuple_items = IterNode(
                container=self,
                function_values=self._axis_tuple,
                function_items=self._axis_tuple_items,
                yield_type=IterNodeType.ITEMS
                )

        self.iter_series = IterNode(
                container=self,
                function_values=self._axis_series,
                function_items=self._axis_series_items,
                yield_type=IterNodeType.VALUES
                )
        self.iter_series_items = IterNode(
                container=self,
                function_values=self._axis_series,
                function_items=self._axis_series_items,
                yield_type=IterNodeType.ITEMS
                )

        self.iter_group = IterNode(
                container=self,
                function_values=self._axis_group_loc,
                function_items=self._axis_group_loc_items,
                yield_type=IterNodeType.VALUES
                )
        self.iter_group_items = IterNode(
                container=self,
                function_values=self._axis_group_loc,
                function_items=self._axis_group_loc_items,
                yield_type=IterNodeType.ITEMS
                )

        self.iter_element = IterNode(
                container=self,
                function_values=self._iter_element_loc,
                function_items=self._iter_element_loc_items,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.FRAME_ELEMENTS
                )
        self.iter_element_items = IterNode(
                container=self,
                function_values=self._iter_element_loc,
                function_items=self._iter_element_loc_items,
                yield_type=IterNodeType.ITEMS,
                apply_type=IterNodeApplyType.FRAME_ELEMENTS
                )


    #---------------------------------------------------------------------------
    # index manipulation

    def _reindex_other_like_iloc(self,
            value: tp.Union[Series, 'Frame'],
            iloc_key: GetItemKeyTypeCompound) -> 'Frame':
        '''Given a value that is a Series or Frame, reindex it to the index components, drawn from this Frame, that are specified by the iloc_key.
        '''
        if isinstance(iloc_key, tuple):
            row_key, column_key = iloc_key
        else:
            row_key, column_key = iloc_key, None

        # within this frame, get Index objects by extracting based on passed-in iloc keys
        axis_nm = self._extract_axis_not_multi(row_key, column_key)
        v = None

        if axis_nm[0] and not axis_nm[1]:
            # only column is multi
            if isinstance(value, Series):
                v = value.reindex(self._columns._extract_iloc(column_key))
        elif not axis_nm[0] and axis_nm[1]:
            # only row is multi
            if isinstance(value, Series):
                v = value.reindex(self._index._extract_iloc(row_key))
        elif not axis_nm[0] and not axis_nm[1]:
            # both multi, must be a DF
            if isinstance(value, Frame):
                target_column_index = self._columns._extract_iloc(column_key)
                target_row_index = self._index._extract_iloc(row_key)
                v = value.reindex(index=target_row_index,
                        columns=target_column_index)
        if v is None:
            raise Exception(('cannot assign '
                    + value.__class__.__name__
                    + ' with key configuration'), axis_nm)
        return v


    def reindex(self,
            index: tp.Union[Index, tp.Sequence[tp.Any]]=None,
            columns: tp.Union[Index, tp.Sequence[tp.Any]]=None,
            fill_value=np.nan) -> 'Frame':
        '''
        Return a new Frame based on the passed index and/or columns.
        '''
        if index is None and columns is None:
            raise Exception('must specify one of index or columns')

        if index is not None:
            index = Index(index)
            index_ic = IndexCorrespondence.from_correspondence(self._index, index)
        else:
            index = self._index
            index_ic = None

        if columns is not None:
            columns = self._COLUMN_CONSTRUCTOR(columns)
            columns_ic = IndexCorrespondence.from_correspondence(self._columns, columns)
        else:
            columns = self._columns
            columns_ic = None

        return self.__class__(
                TypeBlocks.from_blocks(self._blocks.resize_blocks(
                        index_ic=index_ic,
                        columns_ic=columns_ic,
                        fill_value=fill_value)),
                        index=index,
                        columns=columns,
                        own_data=True)


    def relabel(self,
            index: CallableOrMapping=None,
            columns: CallableOrMapping=None) -> 'Frame':
        '''
        Return a new Series based on a mapping (or callable) from old to new index values.
        '''
        # create new index objects in both cases so as to call with own*
        index = self._index.relabel(index) if index else self._index.copy()
        columns = self._columns.relabel(columns) if columns else self._columns.copy()

        return self.__class__(
                self._blocks.copy(),
                index=index,
                columns=columns,
                own_data=True,
                own_index=True,
                own_columns=True)


    #---------------------------------------------------------------------------
    # na handling

    def isna(self) -> 'Frame':
        '''
        Return a same-indexed, Boolean Frame indicating True which values are NaN or None.
        '''
        # always return a Frame, even if this is a FrameGO
        return Frame(self._blocks.isna(),
                index=self._index,
                columns=self._columns,
                own_data=True)


    def notna(self) -> 'Frame':
        '''
        Return a same-indexed, Boolean Frame indicating True which values are not NaN or None.
        '''
        # always return a Frame, even if this is a FrameGO
        return Frame(self._blocks.notna(),
                index=self._index,
                columns=self._columns,
                own_data=True)

    def dropna(self,
            axis: int=0,
            condition: tp.Callable[[np.ndarray], bool]=np.all) -> 'Frame':
        '''
        Return a new Frame after removing rows (axis 0) or columns (axis 1) where condition is True, where condition is an NumPy ufunc that process the Boolean array returned by isna().
        '''
        # returns Boolean areas that define axis to keep
        row_key, column_key = self._blocks.dropna_to_keep_locations(
                axis=axis,
                condition=condition)

        # NOTE: if not values to drop and this is a Frame (not a FrameGO) we can return self as it is immutable
        if self.__class__ is Frame:
            if (row_key is not None and column_key is not None
                    and row_key.all() and column_key.all()):
                return self

        return self._extract(row_key, column_key)

    def fillna(self, value) -> 'Frame':
        '''Return a new Frame after replacing NaN or None values with the supplied value.
        '''
        return self.__class__(self._blocks.fillna(value),
                index=self._index,
                columns=self._columns,
                own_data=True)



    #---------------------------------------------------------------------------

    def __len__(self) -> int:
        '''Length of rows in values.
        '''
        return self._blocks.shape[0]

    def display(self, config: DisplayConfig=None) -> Display:
        config = config or DisplayActive.get()

        d = self._index.display(config=config)

        # print('comparing blocks to display columns', self._blocks.shape[1], config.display_columns)
        if self._blocks.shape[1] > config.display_columns:
            # columns as they will look after application of truncation and insertion of ellipsis
            # get target column count in the absence of meta data, subtracting 2
            data_half_count = Display.truncate_half_count(
                    config.display_columns - Display.DATA_MARGINS)

            column_gen = partial(_gen_skip_middle,
                    forward_iter = partial(self._blocks.axis_values, axis=0),
                    forward_count = data_half_count,
                    reverse_iter = partial(self._blocks.axis_values, axis=0, reverse=True),
                    reverse_count = data_half_count,
                    center_sentinel = Display.ELLIPSIS_CENTER_SENTINEL
                    )
        else:
            column_gen = partial(self._blocks.axis_values, axis=0)

        for column in column_gen():
            if column is Display.ELLIPSIS_CENTER_SENTINEL:
                d.append_ellipsis()
            else:
                d.append_iterable(column, header='')

        cls_display = Display.from_values((),
                header='<' + self.__class__.__name__ + '>',
                config=config)
        # add two rows, one for class, another for columns

        # need to apply the column config such that it truncates it based on the the max columns, not the max rows
        config_column = config.to_transpose()
        d.insert_rows(
                cls_display.flatten(),
                self._columns.display(config=config_column).flatten(),
                )
        return d

    def __repr__(self) -> str:
        return repr(self.display())

    #---------------------------------------------------------------------------
    # accessors

    @property
    def values(self):
        return self._blocks.values

    @property
    def index(self):
        return self._index

    # @index.setter
    # def index(self, value):
    #     if len(value) != len(self._index):
    #         raise Exception('new index must match length of old index')
    #     self._index = Index(value)

    @property
    def columns(self):
        return self._columns

    # @columns.setter
    # def columns(self, value):
    #     if len(value) != len(self._columns):
    #         raise Exception('new columns must match length of old index')
    #     self._columns = self._COLUMN_CONSTRUCTOR(value)

    #---------------------------------------------------------------------------
    # common attributes from the numpy array

    @property
    def dtypes(self) -> Series:
        '''Return a Series of dytpes for each realizable column.
        '''
        return Series(self._blocks.dtypes, index=self._columns.values)

    @property
    def mloc(self) -> np.ndarray:
        '''Return an immutable ndarray of NP array memory location integers.
        '''
        return self._blocks.mloc

    #---------------------------------------------------------------------------

    @property
    def shape(self) -> tp.Tuple[int, int]:
        return self._blocks.shape

    @property
    def ndim(self) -> int:
        return self._blocks.ndim

    @property
    def size(self) -> int:
        return self._blocks.size

    @property
    def nbytes(self) -> int:
        return self._blocks.nbytes

    #---------------------------------------------------------------------------
    @staticmethod
    def _extract_axis_not_multi(row_key, column_key) -> tp.Tuple[bool, bool]:
        '''
        If either row or column is given with a non-multiple type of selection, reduce dimensionality.
        '''
        row_nm = False
        column_nm = False
        if row_key is not None and not isinstance(row_key, _KEY_MULTIPLE_TYPES):
            row_nm = True # axis 0
        if column_key is not None and not isinstance(column_key, _KEY_MULTIPLE_TYPES):
            column_nm = True # axis 1
        return row_nm, column_nm


    def _extract(self,
            row_key: GetItemKeyType=None,
            column_key: GetItemKeyType=None) -> tp.Union['Frame', Series]:
        '''
        Extract based on iloc selection.
        '''
        blocks = self._blocks._extract(row_key=row_key, column_key=column_key)

        if not isinstance(blocks, TypeBlocks):
            return blocks # reduced to element

        if row_key is not None: # have to accept 9!
            index = self._index[row_key]
        else:
            index = self._index

        if column_key is not None:
            columns = self._columns[column_key]
        else:
            columns = self._columns

        axis_nm = self._extract_axis_not_multi(row_key, column_key)

        if blocks._shape == (1, 1):
            # if TypeBlocks did not return an element, need to determine which axis to use for Series index
            if axis_nm[0]: # if row not multi
                return Series(blocks.values[0], index=columns)
            elif axis_nm[1]:
                return Series(blocks.values[0], index=index)
            # if both are multi, we return a Fram
        elif blocks._shape[0] == 1: # if one row
            if axis_nm[0]: # if row key not multi
                # best to use blocks.values, as will need to consolidate if necessary
                block = blocks.values
                if block.ndim == 1:
                    return Series(block, index=columns)
                else: # 2d block, get teh first row
                    return Series(block[0], index=columns)
        elif blocks._shape[1] == 1: # if one column
            if axis_nm[1]: # if column key is not multi
                return Series(blocks.values, index=index)

        return self.__class__(blocks, index=index, columns=columns)


    def _extract_iloc(self, key: GetItemKeyTypeCompound) -> 'Frame':
        if isinstance(key, tuple):
            return self._extract(*key)
        return self._extract(row_key=key)

    def _compound_loc_to_iloc(self,
            key: GetItemKeyTypeCompound) -> tp.Tuple[GetItemKeyType, GetItemKeyType]:
        '''
        Given a compound iloc key, return a tuple of row, column keys. Assumes the first argument is always a row extractor.
        '''
        if isinstance(key, tuple):
            loc_row_key, loc_column_key = key
            iloc_column_key = self._columns.loc_to_iloc(loc_column_key)
        else:
            loc_row_key = key
            iloc_column_key = None

        iloc_row_key = self._index.loc_to_iloc(loc_row_key)
        return iloc_row_key, iloc_column_key

    def _compound_loc_to_getitem_iloc(self,
            key: GetItemKeyTypeCompound) -> tp.Tuple[GetItemKeyType, GetItemKeyType]:
        '''Handle a potentially compound key in the style of __getitem__
        '''
        if isinstance(key, tuple):
            raise KeyError('__getitem__ does not support multiple indexers')
        iloc_column_key = self._columns.loc_to_iloc(key)
        return None, iloc_column_key


    def _extract_loc(self, key: GetItemKeyTypeCompound) -> 'Frame':
        iloc_row_key, iloc_column_key = self._compound_loc_to_iloc(key)
        return self._extract(row_key=iloc_row_key,
                column_key=iloc_column_key)


    def __getitem__(self, key: GetItemKeyType):
        return self._extract(*self._compound_loc_to_getitem_iloc(key))


    #---------------------------------------------------------------------------

    def _extract_iloc_mask(self, key: GetItemKeyTypeCompound) -> 'Frame':
        masked_blocks = self._blocks.extract_iloc_mask(key)
        return self.__class__(masked_blocks,
                columns=self._columns.values,
                index=self._index,
                own_data=True)

    def _extract_loc_mask(self, key: GetItemKeyTypeCompound) -> 'Frame':
        key = self._compound_loc_to_iloc(key)
        return self._extract_iloc_mask(key=key)

    def _extract_getitem_mask(self, key: GetItemKeyTypeCompound) -> 'Frame':
        key = self._compound_loc_to_getitem_iloc(key)
        return self._extract_iloc_mask(key=key)



    def _extract_iloc_masked_array(self, key: GetItemKeyTypeCompound) -> MaskedArray:
        masked_blocks = self._blocks.extract_iloc_mask(key)
        return MaskedArray(data=self.values, mask=masked_blocks.values)

    def _extract_loc_masked_array(self, key: GetItemKeyTypeCompound) -> MaskedArray:
        key = self._compound_loc_to_iloc(key)
        return self._extract_iloc_masked_array(key=key)

    def _extract_getitem_masked_array(self, key: GetItemKeyTypeCompound) -> 'Frame':
        key = self._compound_loc_to_getitem_iloc(key)
        return self._extract_iloc_masked_array(key=key)

    #---------------------------------------------------------------------------

    def _extract_iloc_assign(self, key: GetItemKeyTypeCompound) -> 'FrameAssign':
        return FrameAssign(data=self, iloc_key=key)

    def _extract_loc_assign(self, key: GetItemKeyTypeCompound) -> 'FrameAssign':
        # extract if tuple, then pack back again
        key = self._compound_loc_to_iloc(key)
        return self._extract_iloc_assign(key=key)

    def _extract_getitem_assign(self, key: GetItemKeyTypeCompound) -> 'FrameAssign':
        # extract if tuple, then pack back again
        key = self._compound_loc_to_getitem_iloc(key)
        return self._extract_iloc_assign(key=key)

    #---------------------------------------------------------------------------
    # dictionary-like interface

    def keys(self):
        '''Iterator of column labels.
        '''
        return self._columns

    def __iter__(self):
        '''
        Iterator of column labels, same as :py:meth:`Frame.keys`.
        '''
        return self._columns.__iter__()

    def __contains__(self, value) -> bool:
        '''
        Inclusion of value in column labels.
        '''
        return self._columns.__contains__(value)

    def items(self) -> tp.Generator[tp.Tuple[tp.Any, Series], None, None]:
        '''Iterator of pairs of column label and corresponding column :py:class:`Series`.
        '''
        return zip(self._columns.values,
                (Series(v, index=self._index) for v in self._blocks.axis_values(0)))

    def get(self, key, default=None):
        '''
        Return the value found at the columns key, else the default if the key is not found.
        '''
        if key not in self._columns:
            return default
        return self.__getitem__(key)


    #---------------------------------------------------------------------------
    # operator functions

    def _ufunc_unary_operator(self, operator: tp.Callable) -> 'Frame':
        # call the unary operator on _blocks
        return self.__class__(
                self._blocks._ufunc_unary_operator(operator=operator),
                index=self._index,
                columns=self._columns)

    def _ufunc_binary_operator(self, *, operator, other):
        if isinstance(other, Frame):
            # reindex both dimensions to union indices
            columns = self._columns.union(other._columns)
            index = self._index.union(other._index)
            self_tb = self.reindex(columns=columns, index=index)._blocks
            other_tb = other.reindex(columns=columns, index=index)._blocks
            return self.__class__(self_tb._ufunc_binary_operator(
                    operator=operator, other=other_tb),
                    index=index,
                    columns=columns,
                    own_data=True
                    )
        elif isinstance(other, Series):
            columns = self._columns.union(other._index)
            self_tb = self.reindex(columns=columns)._blocks
            other_array = other.reindex(columns).values
            return self.__class__(self_tb._ufunc_binary_operator(
                    operator=operator, other=other_array),
                    index=self._index,
                    columns=columns,
                    own_data=True
                    )
        # handle single values and lists that can be converted to appropriate arrays
        if not isinstance(other, np.ndarray) and hasattr(other, '__iter__'):
            other = np.array(other)
        # assume we will keep dimensionality
        return self.__class__(self._blocks._ufunc_binary_operator(
                operator=operator, other=other),
                index=self._index,
                columns=self._columns,
                own_data=True
                )

    #---------------------------------------------------------------------------
    # axis functions

    def _ufunc_axis_skipna(self, *, axis, skipna, ufunc, ufunc_skipna, dtype):
        # axis 0 sums ros, deliveres column index
        # axis 1 sums cols, delivers row index
        assert axis < 2

        # TODO: need to handle replacing None with nan in object blocks!
        if skipna:
            post = self._blocks.block_apply_axis(ufunc_skipna, axis=axis, dtype=dtype)
        else:
            post = self._blocks.block_apply_axis(ufunc, axis=axis, dtype=dtype)
        # post has been made immutable so Series will own
        if axis == 0:
            return Series(post, index=self._columns)
        return Series(post, index=self._index)

    #---------------------------------------------------------------------------
    # axis iterators
    # NOTE: if there is more than one argument, the axis argument needs to be key-word only

    def _axis_array(self, axis):
        '''Generator of arrays across an axis
        '''
        yield from self._blocks.axis_values(axis)

    def _axis_array_items(self, axis):
        keys = self._index if axis == 1 else self._columns
        yield from zip(keys, self._blocks.axis_values(axis))


    def _axis_tuple(self, axis):
        '''Generator of named tuples across an axis.

        Args:
            axis: 0 iterates over columns (index axis), 1 iterates over rows (column axis)
        '''
        if axis == 1:
            Tuple = namedtuple('Axis', self._columns.values)
        elif axis == 0:
            Tuple = namedtuple('Axis', self._index.values)
        else:
            raise NotImplementedError()

        for axis_values in self._blocks.axis_values(axis):
            yield Tuple(*axis_values)

    def _axis_tuple_items(self, axis):
        keys = self._index if axis == 1 else self._columns
        yield from zip(keys, self._axis_tuple(axis=axis))


    def _axis_series(self, axis):
        '''Generator of Series across an axis
        '''
        if axis == 1:
            index = self._columns.values
        elif axis == 0:
            index = self._index
        for axis_values in self._blocks.axis_values(axis):
            yield Series(axis_values, index=index)

    def _axis_series_items(self, axis):
        keys = self._index if axis == 1 else self._columns
        yield from zip(keys, self._axis_series(axis=axis))


    #---------------------------------------------------------------------------
    # grouping methods naturally return their "index" as the group element

    def _axis_group_iloc_items(self, key, *, axis):
        for group, selection, tb in self._blocks.group(axis=axis, key=key):
            if axis == 0:
                # axis 0 is a row iter, so need to slice index, keep columns
                yield group, self.__class__(tb,
                        index=self._index[selection],
                        columns=self._columns.values,
                        own_data=True)
            elif axis == 1:
                # axis 1 is a column iterators, so need to slice columns, keep index
                yield group, self.__class__(tb,
                        index=self._index,
                        columns=self._columns[selection],
                        own_data=True)
            else:
                raise NotImplementedError()

    def _axis_group_loc_items(self, key, *, axis=0):
        if axis == 0: # row iterator, selecting columns for group by
            key = self._columns.loc_to_iloc(key)
        elif axis == 1: # column iterator, selecting rows for group by
            key = self._index.loc_to_iloc(key)
        else:
            raise NotImplementedError()
        yield from self._axis_group_iloc_items(key=key, axis=axis)

    def _axis_group_loc(self, key, *, axis=0):
        yield from (x for _, x in self._axis_group_loc_items(key=key, axis=axis))


    #---------------------------------------------------------------------------

    def _iter_element_iloc_items(self):
        yield from self._blocks.element_items()

    def _iter_element_iloc(self):
        yield from (x for _, x in self._iter_element_iloc_items())

    def _iter_element_loc_items(self):
        yield from (
                ((self._index._labels[k[0]], self._columns._labels[k[1]]), v)
                for k, v in self._blocks.element_items())

    def _iter_element_loc(self):
        yield from (x for _, x in self._iter_element_loc_items())


    #---------------------------------------------------------------------------
    # transformations resulting in the same dimensionality

    def sort_index(self,
            ascending: bool=True,
            kind: str=_DEFAULT_SORT_KIND) -> 'Frame':
        '''
        Return a new Frame ordered by the sorted Index.
        '''
        # argsort lets us do the sort once and reuse the results
        order = np.argsort(self._index.values, kind=kind)
        if not ascending:
            order = order[::-1]

        index_values = self._index.values[order]
        index_values.flags.writeable = False
        blocks = self._blocks.iloc[order]
        return self.__class__(blocks,
                index=index_values,
                columns=self._columns,
                own_data=True)

    def sort_columns(self,
            ascending: bool=True,
            kind: str=_DEFAULT_SORT_KIND) -> 'Frame':
        '''
        Return a new Frame ordered by the sorted Columns.
        '''
        # argsort lets us do the sort once and reuse the results
        order = np.argsort(self._columns.values, kind=kind)
        if not ascending:
            order = order[::-1]

        columns_values = self._columns.values[order]
        columns_values.flags.writeable = False
        blocks = self._blocks[order]
        return self.__class__(blocks,
                index=self._index,
                columns=columns_values,
                own_data=True)

    def sort_values(self,
            key: KeyOrKeys,
            ascending: bool=True,
            axis: int=1,
            kind=_DEFAULT_SORT_KIND) -> 'Frame':
        '''
        Return a new Frame ordered by the sorted values, where values is given by one or more columns.
        '''
        # argsort lets us do the sort once and reuse the results
        if axis == 0: # get a column ordering
            if key in self._index:
                iloc_key = self._index.loc_to_iloc(key)
                order = np.argsort(self._blocks.iloc[iloc_key].values, kind=kind)
            else: # assume an iterable of keys
                # order so that highest priority is last
                iloc_keys = (self._index.loc_to_iloc(key) for key in reversed(key))
                order = np.lexsort([self._blocks.iloc[key].values for key in iloc_keys])
        elif axis == 1:
            if key in self._columns:
                iloc_key = self._columns.loc_to_iloc(key)
                order = np.argsort(self._blocks[iloc_key].values, kind=kind)
            else: # assume an iterable of keys
                # order so that highest priority is last
                iloc_keys = (self._columns.loc_to_iloc(key) for key in reversed(key))
                order = np.lexsort([self._blocks[key].values for key in iloc_keys])
        else:
            raise NotImplementedError()

        if not ascending:
            order = order[::-1]

        if axis == 0:
            column_values = self._columns.values[order]
            column_values.flags.writeable = False
            blocks = self._blocks[order]
            return self.__class__(blocks,
                    index=self._index,
                    columns=column_values,
                    own_data=True)

        index_values = self._index.values[order]
        index_values.flags.writeable = False
        blocks = self._blocks.iloc[order]
        return self.__class__(blocks,
                index=index_values,
                columns=self._columns,
                own_data=True)

    def isin(self, other) -> 'Frame':
        '''
        Return a same-sized Boolean Frame that shows if the same-positioned element is in the iterable passed to the function.
        '''
        # cannot use assume_unique because do not know if values is unique
        v, _ = _iterable_to_array(other)
        # TODO: is it faster to do this at the block level and return blocks?
        array = np.isin(self.values, v)
        array.flags.writeable = False
        return self.__class__(array, columns=self._columns, index=self._index)

    def transpose(self) -> 'Frame':
        '''Return a tansposed version of the Frame.
        '''
        return self.__class__(self._blocks.transpose(),
                index=self._columns,
                columns=self._index,
                own_data=True)

    @property
    def T(self) -> 'Frame':
        return self.transpose()


    def duplicated(self,
            axis=0,
            exclude_first=False,
            exclude_last=False) -> 'Series':
        '''
        Return an axis-sized Boolean Series that shows True for all rows (axis 0) or columns (axis 1) duplicated.
        '''
        # NOTE: can avoid calling .vaalues with extensions to TypeBlocks
        duplicates = _array_to_duplicated(self.values,
                axis=axis,
                exclude_first=exclude_first,
                exclude_last=exclude_last)
        duplicates.flags.writeable = False
        if axis == 0: # index is index
            return Series(duplicates, index=self._index)
        return Series(duplicates, index=self._columns)

    def drop_duplicated(self,
            axis=0,
            exclude_first: bool=False,
            exclude_last: bool=False
            ) -> 'Frame':
        '''
        Return a Frame with duplicated values removed.
        '''
        # NOTE: can avoid calling .vaalues with extensions to TypeBlocks
        duplicates = _array_to_duplicated(self.values,
                axis=axis,
                exclude_first=exclude_first,
                exclude_last=exclude_last)

        if not duplicates.any():
            return self

        keep = ~duplicates
        if axis == 0: # return rows with index indexed
            return self.__class__(self.values[keep],
                    index=self._index[keep],
                    columns=self._columns)
        return self.__class__(self.values[:, keep],
                index=self._index,
                columns=self._columns[keep])

    def set_index(self, column: tp.Optional[tp.Union[int, str]],
            drop: bool=False) -> 'Frame':
        '''
        Return a new frame produced by setting the given column as the index, optionally removing that column from the new Frame.
        '''
        if isinstance(column, int):
            column_iloc = column
        else:
            column_iloc = self._columns.loc_to_iloc(column)

        if drop:
            # NOTE: not sure if there is a faster way; perhaps with a drop interface on TypeBlocks
            selection = np.fromiter(
                    (x != column_iloc for x in range(self._blocks.shape[1])),
                    count=self._blocks.shape[1],
                    dtype=bool)
            blocks = self._blocks[selection]
            own_data = True
            columns = self._columns.values[selection]
        else:
            blocks = self._blocks
            own_data = False
            columns = self._columns

        index = self._blocks._extract_array(column_key=column_iloc)

        return self.__class__(blocks,
                columns=columns,
                index=index,
                own_data=own_data)

    #---------------------------------------------------------------------------
    # transformations resulting in reduced dimensionality

    def head(self, count: int=5) -> 'Frame':
        '''Return a Frame consisting only of the top rows as specified by ``count``.
        '''
        return self.iloc[:count]

    def tail(self, count: int=5) -> 'Frame':
        '''Return a Frame consisting only of the bottom rows as specified by ``count``.
        '''
        return self.iloc[-count:]


    #---------------------------------------------------------------------------
    # utility function to numpy array

    def unique(self, axis=None) -> np.ndarray:
        '''
        Return a NumPy array of unqiue values. If the axis argument is provied, uniqueness is determined by columns or row.
        '''
        return np.unique(self.values, axis=axis)

    #---------------------------------------------------------------------------
    # exporters

    def to_pairs(self, axis) -> tp.Iterable[
            tp.Tuple[tp.Hashable, tp.Iterable[tp.Tuple[tp.Hashable, tp.Any]]]]:
        '''
        Return a tuple of major axis key, minor axis key vlaue pairs, where major axis is determined by the axis argument.
        '''

        if axis == 1:
            major = self._index.values
            minor = self._columns.values
        elif axis == 0:
            major = self._columns.values
            minor = self._index.values
        else:
            raise NotImplementedError()

        return tuple(
                zip(major, (tuple(zip(minor, v))
                for v in self._blocks.axis_values(axis))))

    def to_pandas(self):
        '''
        Return a Pandas DataFrame.
        '''
        import pandas
        return pandas.DataFrame(self.values.copy(),
                index=self._index.values.copy(),
                columns=self._columns.values.copy())

    def to_csv(self,
            fp: FilePathOrFileLike,
            sep: str=',',
            include_index: bool=True,
            include_columns: bool=True,
            encoding: tp.Optional[str]=None,
            line_terminator: str='\n'
            ):
        '''
        Given a file path or file-like object, write the Frame as delimited text.
        '''
        to_str = str

        if isinstance(fp, str):
            f = open(fp, 'w', encoding=encoding)
            is_file = True
        else:
            f = fp # assume an open file like
            is_file = False
        try:
            if include_columns:
                if include_index:
                    f.write('index' + sep)
                # iter directly over columns in case it is an IndexGO and needs to update cache
                f.write(sep.join(to_str(x) for x in self._columns))
                f.write(line_terminator)

            col_idx_last = self.shape[1] - 1
            # avoid row creation to avoid joining types; avoide creating a list for each row
            row_current_idx = None
            for (row_idx, col_idx), element in self._iter_element_iloc_items():
                if row_idx != row_current_idx:
                    if row_current_idx is not None:
                        f.write(line_terminator)
                    if include_index:
                        f.write(self._index._labels[row_idx] + sep)
                    row_current_idx = row_idx
                f.write(to_str(element))
                if col_idx != col_idx_last:
                    f.write(sep)
            # not sure if we need a final line terminator
        except:
            raise
        finally:
            if is_file:
                f.close()
        if is_file:
            f.close()

    def to_tsv(self,
            fp: FilePathOrFileLike, **kwargs):
        return self.to_csv(fp=fp, sep='\t', **kwargs)


class FrameGO(Frame):
    '''A two-dimensional, ordered, labelled collection, immutable with grow-only columns. Initialization arguments are the same as for :py:class:`Frame`.
    '''

    __slots__ = (
        '_blocks',
        '_columns',
        '_index',
        'iloc',
        'loc',
        )

    _COLUMN_CONSTRUCTOR = IndexGO


    def __setitem__(self, key, value):
        '''For adding a single column, one column at a time.
        '''
        if key in self._columns:
            raise Exception('key already defined in columns; use .assign to get new Frame')

        if isinstance(value, Series):
            # TODO: performance test if it is faster to compare indices and not call reindex() if we can avoid it?
            # select only the values matching our index
            self._blocks.append(value.reindex(self.index).values)
        else: # unindexed array
            if not isinstance(value, np.ndarray):
                if isinstance(value, GeneratorType):
                    value = np.array(list(value))
                elif not hasattr(value, '__len__') or isinstance(value, str):
                    value = np.full(self._blocks.shape[0], value)
                else:
                    # for now, we assume all values make sense to covnert to NP array
                    value = np.array(value)
                value.flags.writeable = False

            if value.ndim != 1 or len(value) != self._blocks.shape[0]:
                raise Exception('incorrectly sized, unindexed value')
            self._blocks.append(value)

        # this might fail if key is a sequence
        self._columns.append(key)


    def extend_columns(self,
            keys: tp.Iterable,
            values: tp.Iterable):
        '''Extend the FrameGO (add more columns) by providing two iterables, one for column names and antoher for appropriately sized iterables.
        '''
        for k, v in zip_longest(keys, values):
            self.__setitem__(k, v)

    def extend_blocks(self,
            keys: tp.Iterable,
            values: tp.Iterable[np.ndarray]):
        '''Extend the FrameGO (add more columns) by providing two iterables, one of needed column names (not nested), and an iterable of blocks (definting one or more columns in an ndarray).
        '''
        self._columns.extend(keys)
        # TypeBlocks only accepts ndarrays; can try to convert here if lists or tuples given
        for value in values:
            if not isinstance(value, np.ndarray):
                value = np.array(value)
                value.flags.writeable = False
            self._blocks.append(value)

        if len(self._columns) != self._blocks.shape[1]:
            raise Exception('incompatible keys and values')


    def extend(self, frame: 'FrameGO'):
        '''Extend by simply extending this frames blocks.
        '''
        raise NotImplementedError()


class FrameAssign:
    __slots__ = ('data', 'iloc_key',)

    def __init__(self, *,
            data: Frame,
            iloc_key: GetItemKeyTypeCompound
            ) -> None:
        self.data = data
        self.iloc_key = iloc_key

    def __call__(self, value):
        if isinstance(value, (Series, Frame)):
            value = self.data._reindex_other_like_iloc(value, self.iloc_key).values

        blocks = self.data._blocks.extract_iloc_assign(self.iloc_key, value)
        # can own the newly created block given by extract
        return self.data.__class__(
                data=blocks,
                columns=self.data.columns.values,
                index=self.data.index,
                own_data=True)





