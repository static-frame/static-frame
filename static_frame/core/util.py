from __future__ import annotations

import ast
import contextlib
import datetime
import math
import operator
import os
import re
import tempfile
import warnings
from collections import Counter
from collections import abc
from collections import defaultdict
from collections import namedtuple
from collections.abc import Mapping
from enum import Enum
from fractions import Fraction
from functools import partial
from functools import reduce
from io import StringIO
from itertools import chain
from itertools import zip_longest
from os import PathLike
from types import TracebackType

import numpy as np
import typing_extensions as tp
from arraykit import array_to_tuple_iter
from arraykit import column_2d_filter
from arraykit import first_true_1d
from arraykit import isna_element
from arraykit import mloc
from arraykit import nonzero_1d
from arraykit import resolve_dtype
from arraymap import FrozenAutoMap  # pylint: disable = E0611

from static_frame.core.exception import ErrorNotTruthy
from static_frame.core.exception import InvalidDatetime64Comparison
from static_frame.core.exception import InvalidDatetime64Initializer
from static_frame.core.exception import LocInvalid

if tp.TYPE_CHECKING:
    from concurrent.futures import Executor  # pragma: no cover

    from static_frame.core.frame import Frame  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.index import Index  # pylint: disable=W0611 #pragma: no cover
    # from static_frame.core.index_auto import IndexAutoFactory  #pragma: no cover
    from static_frame.core.index_auto import IndexAutoConstructorFactory  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.index_auto import IndexConstructorFactoryBase  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.index_base import IndexBase  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchy  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.series import Series  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.type_blocks import TypeBlocks  # pylint: disable=W0611 #pragma: no cover

TNDArrayAny = np.ndarray[tp.Any, tp.Any]
TNDArrayBool = np.ndarray[tp.Any, np.dtype[np.bool_]]
TNDArrayObject = np.ndarray[tp.Any, np.dtype[np.object_]]
TNDArrayIntDefault = np.ndarray[tp.Any, np.dtype[np.int64]]

TDtypeAny = np.dtype[tp.Any]
TDtypeObject = np.dtype[np.object_] #pragma: no cover
TOptionalArrayList = tp.Optional[tp.List[TNDArrayAny]]

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

TSortKinds = tp.Literal['quicksort', 'mergesort']
DEFAULT_SORT_KIND: tp.Literal['mergesort'] = 'mergesort'
DEFAULT_STABLE_SORT_KIND: tp.Literal['mergesort'] = 'mergesort' # for when results will be in correct if not used
DEFAULT_FAST_SORT_KIND: tp.Literal['quicksort'] = 'quicksort' # for when fastest is all that we want

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

DTYPE_NAN_NAT_KINDS = frozenset((
        DTYPE_FLOAT_KIND,
        DTYPE_COMPLEX_KIND,
        DTYPE_DATETIME_KIND,
        DTYPE_TIMEDELTA_KIND,
        ))

# this is all kinds except 'V'
# DTYPE_FALSY_KINDS = frozenset((
#         DTYPE_FLOAT_KIND,
#         DTYPE_COMPLEX_KIND,
#         DTYPE_DATETIME_KIND,
#         DTYPE_TIMEDELTA_KIND,
#         DTYPE_OBJECT_KIND,
#         DTYPE_BOOL_KIND,
#         'U', 'S', # str kinds
#         'i', 'u' # int kinds
#         ))

# all kinds that can use tolist() to go to a compatible Python type
DTYPE_OBJECTABLE_KINDS = frozenset((
        DTYPE_FLOAT_KIND,
        DTYPE_COMPLEX_KIND,
        DTYPE_OBJECT_KIND,
        DTYPE_BOOL_KIND,
        'U', 'S', # str kinds
        'i', 'u' # int kinds
        ))

# all dt64 units that tolist() to go to a compatible Python type. Note that datetime.date.MINYEAR, MAXYEAR sets a limit that is more narrow than dt64
# NOTE: similar to DT64_EXCLUDE_YEAR_MONTH_SUB_MICRO
DTYPE_OBJECTABLE_DT64_UNITS = frozenset((
        'D', 'h', 'm', 's', 'ms', 'us',
        ))

def is_objectable_dt64(array: TNDArrayAny) -> bool:
    if np.datetime_data(array.dtype)[0] not in DTYPE_OBJECTABLE_DT64_UNITS:
        return False
    years = array.astype(DT64_YEAR).astype(DTYPE_INT_DEFAULT) + 1970
    if np.any(years < datetime.MINYEAR):
        return False
    if np.any(years > datetime.MAXYEAR):
        return False
    return True

# all numeric types, plus bool
DTYPE_NUMERICABLE_KINDS = frozenset((
        DTYPE_FLOAT_KIND,
        DTYPE_COMPLEX_KIND,
        DTYPE_BOOL_KIND,
        'i', 'u' # int kinds
))


DTYPE_OBJECT = np.dtype(object)
DTYPE_BOOL = np.dtype(bool)
DTYPE_STR = np.dtype(str)
DTYPE_INT_DEFAULT = np.dtype(np.int64)
DTYPE_UINT_DEFAULT = np.dtype(np.uint64)
# DTYPE_INT_PLATFORM = np.dtype(int) # 32 on windows

DTYPE_FLOAT_DEFAULT = np.dtype(np.float64)
DTYPE_COMPLEX_DEFAULT = np.dtype(np.complex128)
DTYPE_YEAR_MONTH_STR = np.dtype('U7')
DTYPE_YEAR_QUARTER_STR = np.dtype('U7')

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
EMPTY_SET: tp.FrozenSet[tp.Any] = frozenset()
EMPTY_TUPLE: tp.Tuple[()] = ()

# defaults to float64
EMPTY_ARRAY: TNDArrayAny = np.array((), dtype=None)
EMPTY_ARRAY.flags.writeable = False

EMPTY_ARRAY_BOOL: TNDArrayAny = np.array((), dtype=DTYPE_BOOL)
EMPTY_ARRAY_BOOL.flags.writeable = False

EMPTY_ARRAY_INT: TNDArrayAny = np.array((), dtype=DTYPE_INT_DEFAULT)
EMPTY_ARRAY_INT.flags.writeable = False


UNIT_ARRAY_INT: TNDArrayAny = np.array((0,), dtype=DTYPE_INT_DEFAULT)
UNIT_ARRAY_INT.flags.writeable = False

EMPTY_ARRAY_OBJECT = np.array((), dtype=DTYPE_OBJECT)
EMPTY_ARRAY_OBJECT.flags.writeable = False

EMPTY_FROZEN_AUTOMAP = FrozenAutoMap()

NAT = np.datetime64('nat')
NAT_STR = 'NaT'

# this is a different NAT but can be treated the same
NAT_TD64 = np.timedelta64('nat')

# define missing for timedelta as an untyped 0
EMPTY_TIMEDELTA = np.timedelta64(0)

# map from datetime.timedelta attrs to np.timedelta64 codes
TIME_DELTA_ATTR_MAP = (
        ('days', 'D'),
        ('seconds', 's'),
        ('microseconds', 'us')
        )

# ufunc functions that will not work with DTYPE_STR_KINDS, but do work if converted to object arrays
UFUNC_AXIS_STR_TO_OBJ = frozenset((np.min, np.max, np.sum))

FALSY_VALUES = frozenset((0, '', None, ()))

#-------------------------------------------------------------------------------
# utility type groups

INT_TYPES = (int, np.integer) # np.integer catches all np int
FLOAT_TYPES = (float, np.floating) # np.floating catches all np float
COMPLEX_TYPES = (complex, np.complexfloating) # np.complexfloating catches all np complex
INEXACT_TYPES = (float, complex, np.inexact) # inexact matches floating, complexfloating
NUMERIC_TYPES = (int, float, complex, np.number)
BOOL_TYPES = (bool, np.bool_)
DICTLIKE_TYPES = (abc.Set, dict, FrozenAutoMap)

# iterables that cannot be used in NP array constructors; assumes that dictlike types have already been identified
INVALID_ITERABLE_FOR_ARRAY = (abc.ValuesView, abc.KeysView)
NON_STR_TYPES = {int, float, bool}

# integers above this value will occassionally, once coerced to a float (64 or 128) in an NP array, will not match a hash lookup as a key in a dictionary; an NP array of int or object will work
INT_MAX_COERCIBLE_TO_FLOAT = 1_000_000_000_000_000
INT64_MAX = np.iinfo(np.int64).max

# for getitem / loc selection
KEY_ITERABLE_TYPES = (list, np.ndarray)
TKeyIterable = tp.Union[tp.Iterable[tp.Any], TNDArrayAny]

# types of keys that return multiple items, even if the selection reduces to 1
KEY_MULTIPLE_TYPES = (np.ndarray, list, slice)

TILocSelectorOne = tp.Union[int, np.integer[tp.Any]]
TILocSelectorMany = tp.Union[TNDArrayAny, tp.List[int], slice, None]
TILocSelector = tp.Union[TILocSelectorOne, TILocSelectorMany]
TILocSelectorCompound = tp.Union[TILocSelector, tp.Tuple[TILocSelector, TILocSelector]]

# NOTE: slice is not hashable
# NOTE: this is TLocSelectorOne
TLabel = tp.Union[
        tp.Hashable,
        int,
        bool,
        np.bool_,
        np.integer[tp.Any],
        float,
        complex,
        np.inexact[tp.Any],
        str,
        bytes,
        None,
        np.datetime64,
        np.timedelta64,
        datetime.date,
        datetime.datetime,
        tp.Tuple['TLabel', ...],
]

TLocSelectorMany = tp.Union[
        slice,
        tp.List[TLabel],
        TNDArrayAny,
        'IndexBase',
        'Series',
        ]

TLocSelectorNonContainer = tp.Union[
        TLabel,
        slice,
        tp.List[TLabel],
        TNDArrayAny,
        ]

# keys once dimension has been isolated
TLocSelector = tp.Union[
        TLabel,
        TLocSelectorMany
        ]

# keys that might include a multiple dimensions speciation; tuple is used to identify compound extraction
TLocSelectorCompound = tp.Union[TLocSelector, tp.Tuple[TLocSelector, TLocSelector]]

TKeyTransform = tp.Optional[tp.Callable[[TLocSelector], TLocSelector]]
TName = TLabel # include name default?

TTupleCtor = tp.Union[tp.Callable[[tp.Iterable[tp.Any]], tp.Sequence[tp.Any]], tp.Type[tp.Tuple[tp.Any]]]

TBlocKey = tp.Union['Frame', TNDArrayAny, None]
# Bloc1DKeyType = tp.Union['Series', np.ndarray]

TUFunc = tp.Callable[..., TNDArrayAny]
TCallableAny = tp.Callable[..., tp.Any]

TMapping = tp.Union[tp.Mapping[TLabel, tp.Any], 'Series']
TCallableOrMapping = tp.Union[TCallableAny, tp.Mapping[TLabel, tp.Any], 'Series']

TShape = tp.Union[int, tp.Tuple[int, ...]]

# mloc, shape, and strides
TArraySignature = tp.Tuple[int, tp.Tuple[int, ...], tp.Tuple[int, ...]]

def array_signature(value: TNDArrayAny) -> TArraySignature:
    return mloc(value), value.shape, value.strides

def is_mapping(value: tp.Any) -> bool:
    from static_frame import Series
    return isinstance(value, (Mapping, Series))

def is_callable_or_mapping(value: tp.Any) -> bool:
    from static_frame import Series
    return callable(value) or isinstance(value, Mapping) or isinstance(value, Series)

TCallableOrCallableMap = tp.Union[TCallableAny, tp.Mapping[TLabel, TCallableAny]]

# for explivitl selection hashables, or things that will be converted to lists of hashables (explicitly lists)
TKeyOrKeys = tp.Union[TLabel, tp.Iterable[TLabel]]
TBoolOrBools = tp.Union[bool, tp.Iterable[bool]]

TPathSpecifier = tp.Union[str, PathLike[tp.Any]]
TPathSpecifierOrIO = tp.Union[str, PathLike[tp.Any], tp.IO[tp.Any]]
TPathSpecifierOrBinaryIO = tp.Union[str, PathLike[tp.Any], tp.BinaryIO]
TPathSpecifierOrTextIO = tp.Union[str, PathLike[tp.Any], tp.TextIO]
TPathSpecifierOrTextIOOrIterator = tp.Union[str, PathLike[tp.Any], tp.TextIO, tp.Iterator[str]]

TDtypeSpecifier = tp.Union[str, TDtypeAny, type, None]
TDtypeOrDT64 = tp.Union[TDtypeAny, tp.Type[np.datetime64]]

def validate_dtype_specifier(value: tp.Any) -> TDtypeSpecifier:
    if value is None or isinstance(value, np.dtype):
        return value

    dt = np.dtype(value)
    if dt == DTYPE_OBJECT and value is not object and value not in ('object', '|O'):
        # fail on implicit conversion to object dtype
        raise TypeError(f'Implicit NumPy conversion of a type {value!r} to an object dtype; use `object` instead.')
    return dt


DTYPE_SPECIFIER_TYPES = (str, np.dtype, type)

def is_dtype_specifier(value: tp.Any) -> bool:
    return isinstance(value, DTYPE_SPECIFIER_TYPES)

def is_neither_slice_nor_mask(value: TLocSelectorNonContainer) -> bool:
    is_slice = value.__class__ is slice
    is_mask = value.__class__ is np.ndarray and value.dtype == DTYPE_BOOL # type: ignore
    return not is_slice and not is_mask

def is_strict_int(value: tp.Any) -> bool:
    '''Strict check that does not include bools as an int
    '''
    if value is None:
        return False
    if value.__class__ is bool or value.__class__ is np.bool_:
        return False
    return isinstance(value, INT_TYPES)

def depth_level_from_specifier(
        key: TDepthLevelSpecifier,
        size: int,
        ) -> TDepthLevel:
    '''Determine if a key is strictly an ILoc-style key. This is used in `IndexHierarchy`, where at times we select "columns" (or depths) by integer (not name or per-depth names, as such attributes are not required), and we cannot assume the caller gives us integers, as some types of inputs (Python lists of Booleans) might work due to low-level duckyness.

    This does not permit selection by tuple elements at this time, as that is not possible for IndexHierarchy depth selection.
    '''
    if key is None:
        return list(range(size))
    elif key.__class__ is np.ndarray:
        # let object dtype use iterable path
        if key.dtype.kind in DTYPE_INT_KINDS: #type: ignore
            return key.tolist() # type: ignore
        elif key.dtype == DTYPE_BOOL: # type: ignore
            return PositionsAllocator.get(size)[key].tolist() # type: ignore
        elif key.dtype.kind == DTYPE_OBJECT_KIND: # type: ignore
            for e in key: # type: ignore
                if not is_strict_int(e):
                    raise KeyError(f'Cannot select depths by non integer: {e!r}')
            return key.tolist() # type: ignore
        raise KeyError(f'Cannot select depths by NumPy array of dtype: {key.dtype!r}') # type: ignore
    elif key.__class__ is slice:
        if key.start is not None and not is_strict_int(key.start): # type: ignore
            raise KeyError(f'Cannot select depths by non integer slices: {key!r}')
        if key.stop is not None and not is_strict_int(key.stop): # type: ignore
            raise KeyError(f'Cannot select depths by non integer slices: {key!r}')
        return list(range(*key.indices(size))) # type: ignore
    elif isinstance(key, list):
        # an iterable, or an object dtype array
        for e in key:
            if not is_strict_int(e):
                raise KeyError(f'Cannot select depths by non integer: {e!r}')
        return key
    if not is_strict_int(key):
        raise KeyError(f'Cannot select depths by non integer: {key!r}')
    return key # type: ignore

# support an iterable of specifiers, or mapping based on column names
TDtypesSpecifier = tp.Optional[tp.Union[
        TDtypeSpecifier,
        tp.Iterable[TDtypeSpecifier],
        tp.Dict[TLabel, TDtypeSpecifier]
        ]]

TDepthLevelSpecifierOne = int
TDepthLevelSpecifierMany = tp.Union[tp.List[int], slice, TNDArrayIntDefault, None]
TDepthLevelSpecifier = tp.Union[TDepthLevelSpecifierOne, TDepthLevelSpecifierMany]

TDepthLevel = tp.Union[int, tp.List[int]]

TCallableToIter = tp.Callable[[], tp.Iterable[tp.Any]]

TIndexSpecifier = tp.Union[int, TLabel] # specify a position in an index
TIndexInitializer = tp.Union[
        'IndexBase',
        tp.Iterable[TLabel],
        tp.Iterable[tp.Sequence[TLabel]], # only for IndexHierarchy
        # tp.Type['IndexAutoFactory'],
        ]

# NOTE: this should include tp.Type['IndexAutoConstructorFactory']
TIndexCtor = tp.Union[tp.Callable[..., 'IndexBase'], tp.Type['Index'],]
TIndexHierarchyCtor = tp.Union[tp.Callable[..., 'IndexHierarchy'], tp.Type['IndexHierarchy']]

TIndexCtorSpecifier = tp.Optional[TIndexCtor]

TIndexCtorSpecifiers = tp.Union[TIndexCtorSpecifier,
        tp.Sequence[TIndexCtorSpecifier],
        tp.Iterable[TIndexCtorSpecifier],
        TNDArrayObject, # object array of constructors
        None,
        tp.Type['IndexAutoConstructorFactory'],
        ]

TExplicitIndexCtor = tp.Union[
        TIndexCtorSpecifier,
        'IndexConstructorFactoryBase',
        tp.Type['IndexConstructorFactoryBase'],
        None,
        ]
# take integers for size; otherwise, extract size from any other index initializer

TSeriesInitializer = tp.Union[
        tp.Iterable[tp.Any],
        TNDArrayAny,
        ]

# support single items, or numpy arrays, or values that can be made into a 2D array
FRAME_INITIALIZER_DEFAULT = object()
CONTINUATION_TOKEN_INACTIVE = object()
ZIP_LONGEST_DEFAULT = object()

TFrameInitializer = tp.Union[
        tp.Iterable[tp.Iterable[tp.Any]],
        TNDArrayAny,
        'TypeBlocks',
        'Frame',
        'Series',
        ]

TDateInitializer = tp.Union[int, np.integer[tp.Any], str, datetime.date, np.datetime64]
TYearMonthInitializer = tp.Union[int, np.integer[tp.Any], str, datetime.date, np.datetime64]
TYearInitializer = tp.Union[int, np.integer[tp.Any], str, datetime.date, np.datetime64]

#-------------------------------------------------------------------------------
FILL_VALUE_DEFAULT = object()
NAME_DEFAULT = object()
STORE_LABEL_DEFAULT = object()

#-------------------------------------------------------------------------------
NOT_IN_CACHE_SENTINEL = object()

#-------------------------------------------------------------------------------
# operator mod does not have r methods; create complete method reference
OPERATORS: tp.Dict[str, TUFunc] = { # pyright: ignore
    '__pos__': operator.__pos__,
    '__neg__': operator.__neg__,
    '__abs__': operator.__abs__,
    '__invert__': operator.__invert__,

    '__add__': operator.__add__,
    '__sub__': operator.__sub__,
    '__mul__': operator.__mul__,
    '__matmul__': operator.__matmul__,
    '__truediv__': operator.__truediv__,
    '__floordiv__': operator.__floordiv__,
    '__mod__': operator.__mod__,

    '__pow__': operator.__pow__,
    '__lshift__': operator.__lshift__,
    '__rshift__': operator.__rshift__,
    '__and__': operator.__and__,
    '__xor__': operator.__xor__,
    '__or__': operator.__or__,
    '__lt__': operator.__lt__,
    '__le__': operator.__le__,
    '__eq__': operator.__eq__,
    '__ne__': operator.__ne__,
    '__gt__': operator.__gt__,
    '__ge__': operator.__ge__,
}

# extend r methods
for attr in ('__add__', '__sub__', '__mul__', '__matmul__', '__truediv__', '__floordiv__'):
    func = getattr(operator, attr)
    # bind func from closure
    rfunc = lambda rhs, lhs, func=func: func(lhs, rhs)
    rfunc.__name__ = 'r' + func.__name__
    rattr = '__r' + attr[2:]
    OPERATORS[rattr] = rfunc

UFUNC_TO_REVERSE_OPERATOR: tp.Dict[TUFunc, TUFunc] = {
    # '__pos__': operator.__pos__,
    # '__neg__': operator.__neg__,
    # '__abs__': operator.__abs__,
    # '__invert__': operator.__invert__,
    np.add: OPERATORS['__radd__'],
    np.subtract: OPERATORS['__rsub__'],
    np.multiply: OPERATORS['__rmul__'],
    np.matmul: OPERATORS['__rmatmul__'],
    np.true_divide: OPERATORS['__rtruediv__'],
    np.floor_divide: OPERATORS['__rfloordiv__'],
    # '__mod__': operator.__mod__,
    # '__pow__': operator.__pow__,
    # '__lshift__': operator.__lshift__,
    # '__rshift__': operator.__rshift__,
    # '__and__': operator.__and__,
    # '__xor__': operator.__xor__,
    # '__or__': operator.__or__,
    np.less: OPERATORS['__gt__'],
    np.less_equal: OPERATORS['__ge__'],
    np.equal: OPERATORS['__eq__'],
    np.not_equal: OPERATORS['__ne__'],
    np.greater: OPERATORS['__lt__'],
    np.greater_equal: OPERATORS['__le__'],
}


class IterNodeType(Enum):
    VALUES = 1
    ITEMS = 2

#-------------------------------------------------------------------------------
class WarningsSilent:
    '''Alternate context manager for silencing warnings with less overhead.
    '''
    __slots__ = ('previous_warnings',)

    FILTER = [('ignore', None, Warning, None, 0)]

    def __enter__(self) -> None:
        self.previous_warnings = warnings.filters
        warnings.filters = self.FILTER

    def __exit__(self,
            type: tp.Type[BaseException],
            value: BaseException,
            traceback: TracebackType,
            ) -> None:
        warnings.filters = self.previous_warnings



#-------------------------------------------------------------------------------

def _ufunc_logical_skipna(
        array: TNDArrayAny,
        ufunc: TCallableAny,
        skipna: bool,
        axis: int = 0,
        out: tp.Optional[TNDArrayAny] = None
        ) -> tp.Any:
    '''
    Given a logical (and, or) ufunc that does not support skipna, implement skipna behavior.
    '''
    if ufunc != np.all and ufunc != np.any:
        raise NotImplementedError(f'unsupported ufunc ({ufunc}); use np.all or np.any')

    if len(array) == 0:
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
        # NOTE: NaN will be interpreted as True
        return ufunc(array, axis=axis, out=out)

    if kind == 'O':
        # all object types: convert to boolean aray then process
        isna = isna_array(array)
        hasna = isna.any() # returns single value for 1d, 2d
        if hasna and skipna:
            # supply True for np.all, False for np.any
            fill_value = False if ufunc == np.any else True
            v = array.astype(bool) # nan will be converted to True
            v[isna] = fill_value
        else:
            # NOTE: NaN will be converted to True, None will be converted to False
            v = array.astype(bool)
        return ufunc(v, axis=axis, out=out)

    # all other types assume truthy
    # if kind in DTYPE_NAT_KINDS: # all dates are truthy, NAT is truthy

    if array.ndim == 1:
        return True
    return np.full(array.shape[0 if axis else 1], fill_value=True, dtype=bool)

def ufunc_all(array: TNDArrayAny,
        axis: int = 0,
        out: tp.Optional[TNDArrayAny] = None
        ) -> tp.Any:
    return _ufunc_logical_skipna(array,
            ufunc=np.all,
            skipna=False,
            axis=axis,
            out=out)

ufunc_all.__doc__ = np.all.__doc__

def ufunc_any(array: TNDArrayAny,
        axis: int = 0,
        out: tp.Optional[TNDArrayAny] = None
        ) -> tp.Any:
    return _ufunc_logical_skipna(array,
            ufunc=np.any,
            skipna=False,
            axis=axis,
            out=out)

ufunc_any.__doc__ = np.any.__doc__

def ufunc_nanall(array: TNDArrayAny,
        axis: int = 0,
        out: tp.Optional[TNDArrayAny] = None
        ) -> tp.Any:
    return _ufunc_logical_skipna(array,
            ufunc=np.all,
            skipna=True,
            axis=axis,
            out=out)

def ufunc_nanany(array: TNDArrayAny,
        axis: int = 0,
        out: tp.Optional[TNDArrayAny] = None
        ) -> tp.Any:
    return _ufunc_logical_skipna(array,
            ufunc=np.any,
            skipna=True,
            axis=axis,
            out=out)

#-------------------------------------------------------------------------------

def _ufunc_numeric_skipna(
        array: TNDArrayAny,
        axis: int,
        allna: float,
        func: TUFunc,
        allna_default: float,
        out: tp.Optional[TNDArrayAny],
        ) -> TNDArrayAny:
    '''Alternate func that permits specifying `allna`.
    '''
    out_provided = out is not None

    if allna == allna_default: # NumPy default, use default
        return func(array, axis, out=out)

    if out_provided:
        func(array, axis, out=out)
    else:
        out = func(array, axis)
    assert out is not None

    if array.ndim == 1:
        if out == allna_default: # might be all NaN
            if isna_array(array).all():
                if out_provided:
                    out[None] = allna
                    return out
                return allna # type: ignore
        return out

    # ndim == 2
    if (out == allna_default).any():
        out[isna_array(array).all(axis)] = allna

    out.flags.writeable = False
    return out

def ufunc_nanprod(
        array: TNDArrayAny,
        axis: int = 0,
        allna: float = 1,
        out: tp.Optional[TNDArrayAny] = None,
        ) -> TNDArrayAny:
    '''Alternate nanprod that permits specifying `allna`.
    '''
    return _ufunc_numeric_skipna(
            array,
            axis,
            allna,
            np.nanprod,
            1,
            out,
            )

def ufunc_nansum(
        array: TNDArrayAny,
        axis: int = 0,
        allna: float = 1,
        out: tp.Optional[TNDArrayAny] = None,
        ) -> TNDArrayAny:
    '''Alternate nansum that permits specifying `allna`.
    '''
    return _ufunc_numeric_skipna(
            array,
            axis,
            allna,
            np.nansum,
            0,
            out,
            )

#-------------------------------------------------------------------------------
class UFuncCategory(Enum):
    BOOL = 0
    SELECTION = 1
    STATISTICAL = 2 # go to default float type if int, float/complex keep size
    CUMMULATIVE = 3 # go to max size if int, float/complex keep size
    SUMMING = 4 # same except bool goes to max int


UFUNC_MAP: tp.Dict[TCallableAny, UFuncCategory] = {
    all: UFuncCategory.BOOL,
    any: UFuncCategory.BOOL,
    np.all: UFuncCategory.BOOL,
    np.any: UFuncCategory.BOOL,
    ufunc_all: UFuncCategory.BOOL,
    ufunc_any: UFuncCategory.BOOL,
    ufunc_nanall: UFuncCategory.BOOL,
    ufunc_nanany: UFuncCategory.BOOL,

    sum: UFuncCategory.SUMMING,
    np.sum: UFuncCategory.SUMMING,
    np.nansum: UFuncCategory.SUMMING,
    ufunc_nansum: UFuncCategory.SUMMING,

    min: UFuncCategory.SELECTION,
    np.min: UFuncCategory.SELECTION,
    np.nanmin: UFuncCategory.SELECTION,
    max: UFuncCategory.SELECTION,
    np.max: UFuncCategory.SELECTION,
    np.nanmax: UFuncCategory.SELECTION,

    np.mean: UFuncCategory.STATISTICAL,
    np.nanmean: UFuncCategory.STATISTICAL,
    np.median: UFuncCategory.STATISTICAL,
    np.nanmedian: UFuncCategory.STATISTICAL,
    np.std: UFuncCategory.STATISTICAL,
    np.nanstd: UFuncCategory.STATISTICAL,
    np.var: UFuncCategory.STATISTICAL,
    np.nanvar: UFuncCategory.STATISTICAL,

    np.prod: UFuncCategory.CUMMULATIVE,
    np.nanprod: UFuncCategory.CUMMULATIVE,
    ufunc_nanprod: UFuncCategory.CUMMULATIVE,
    np.cumsum: UFuncCategory.CUMMULATIVE,
    np.nancumsum: UFuncCategory.CUMMULATIVE,
    np.cumprod: UFuncCategory.CUMMULATIVE,
    np.nancumprod: UFuncCategory.CUMMULATIVE,
}

def ufunc_to_category(func: tp.Union[TUFunc, partial[TUFunc]]) -> tp.Optional[UFuncCategory]:
    if func.__class__ is partial:
        # std, var partialed
        func = func.func #type: ignore
    return UFUNC_MAP.get(func, None)

def ufunc_dtype_to_dtype(func: TUFunc, dtype: TDtypeAny) -> tp.Optional[TDtypeAny]:
    '''Given a common TUFunc and dtype, return the expected return dtype, or None if not possible.
    '''
    rt = ufunc_to_category(func)
    if rt is None:
        return None

    if rt is UFuncCategory.BOOL:
        if dtype == DTYPE_OBJECT:
            return None # an object array returns a value from the array
        return DTYPE_BOOL

    if rt is UFuncCategory.SELECTION:
        if dtype == DTYPE_OBJECT:
            return None # cannot be sure
        else:
            return dtype

    if rt is UFuncCategory.SUMMING:
        if dtype == DTYPE_OBJECT:
            return None # cannot be sure
        if dtype == DTYPE_BOOL or dtype.kind in DTYPE_INT_KINDS:
            return DTYPE_INT_DEFAULT
        if dtype.kind in DTYPE_INEXACT_KINDS:
            if func is sum:
                if dtype.kind == DTYPE_COMPLEX_KIND:
                    if dtype.itemsize <= DTYPE_COMPLEX_DEFAULT.itemsize:
                        return DTYPE_COMPLEX_DEFAULT
                    return dtype
                if dtype.kind == DTYPE_FLOAT_KIND:
                    if dtype.itemsize <= DTYPE_FLOAT_DEFAULT.itemsize:
                        return DTYPE_FLOAT_DEFAULT
                    return dtype
            return dtype # keep same size

    if rt is UFuncCategory.STATISTICAL:
        if dtype == DTYPE_OBJECT or dtype == DTYPE_BOOL:
            return DTYPE_FLOAT_DEFAULT
        if dtype.kind in DTYPE_INT_KINDS:
            return DTYPE_FLOAT_DEFAULT
        if dtype.kind in DTYPE_INEXACT_KINDS:
            if func in (np.std, np.nanstd, np.var, np.nanvar):
                if dtype.kind == DTYPE_COMPLEX_KIND:
                    return DTYPE_FLOAT_DEFAULT
            return dtype # keep same size

    if rt is UFuncCategory.CUMMULATIVE:
        if dtype == DTYPE_OBJECT:
            return None
        elif dtype == DTYPE_BOOL:
            return DTYPE_INT_DEFAULT
        elif dtype.kind in DTYPE_INT_KINDS:
            return DTYPE_INT_DEFAULT
        elif dtype.kind in DTYPE_INEXACT_KINDS:
            return dtype # keep same size

    return None

#-------------------------------------------------------------------------------
TVFGItem = tp.TypeVar('TVFGItem')

class FrozenGenerator:
    '''
    A wrapper of an iterator (or iterable) that stores values iterated for later recall; this never iterates the iterator unbound, but always iterates up to a target.
    '''
    __slots__ = (
        '_gen',
        '_src',
        )

    def __init__(self, gen: tp.Iterable[TVFGItem]):
        # NOTE: while generally called with an iterator, some iterables such as dict_values need to be converted to an iterator
        self._gen: tp.Iterator[TVFGItem]
        if hasattr(gen, '__next__'):
            self._gen = gen #type: ignore
        else:
            self._gen = iter(gen)
        self._src: tp.List[TVFGItem] = []

    def __getitem__(self, key: int) -> TVFGItem: # type: ignore
        start = len(self._src)
        if key >= start:
            for k in range(start, key + 1):
                try:
                    self._src.append(next(self._gen))
                except StopIteration:
                    raise IndexError(k) from None
        return self._src[key] # type: ignore

#-------------------------------------------------------------------------------
def get_concurrent_executor(
        *,
        use_threads: bool,
        max_workers: tp.Optional[int],
        mp_context: tp.Optional[str],
        ) -> tp.Type[Executor]:
    # NOTE: these imports are conditional as these modules are not supported in pyodide
    from concurrent.futures import Executor
    exe: tp.Callable[..., Executor]
    if use_threads:
        from concurrent.futures import ThreadPoolExecutor
        exe = partial(ThreadPoolExecutor,
                max_workers=max_workers)
    else:
        from concurrent.futures import ProcessPoolExecutor
        exe = partial(ProcessPoolExecutor,
                max_workers=max_workers,
                mp_context=mp_context) # pyright: ignore
    return exe #type: ignore

#-------------------------------------------------------------------------------
# join utils

class Join(Enum):
    INNER = 0
    LEFT = 1
    RIGHT = 2
    OUTER = 3

#-------------------------------------------------------------------------------

def bytes_to_size_label(size_bytes: int) -> str:
    if size_bytes == 0:
        return '0 B'
    size_name = ('B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB')
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s: tp.Union[int, float]
    if size_name[i] == 'B':
        s = size_bytes
    else:
        s = round(size_bytes / p, 2)
    return f'{s} {size_name[i]}'

#-------------------------------------------------------------------------------

_T = tp.TypeVar('_T', bound=TLabel)

def frozenset_filter(src: tp.Iterable[_T]) -> tp.FrozenSet[_T]:
    '''
    Return a frozenset of `src` if not already a frozenset.
    '''
    if isinstance(src, frozenset):
        return src
    return frozenset(src)

# def mloc(array: np.ndarray) -> int:
#     '''Return the memory location of an array.
#     '''
#     return tp.cast(int, array.__array_interface__['data'][0])


# def immutable_filter(src_array: np.ndarray) -> np.ndarray:
#     '''Pass an immutable array; otherwise, return an immutable copy of the provided array.
#     '''
#     if src_array.flags.writeable:
#         dst_array = src_array.copy()
#         dst_array.flags.writeable = False
#         return dst_array
#     return src_array # keep it as is


# def name_filter(name: TName) -> TName:
#     '''
#     For name attributes on containers, only permit recursively hashable objects.
#     '''
#     try:
#         hash(name)
#     except TypeError:
#         raise TypeError('unhashable name attribute', name)
#     return name


# def shape_filter(array: np.ndarray) -> tp.Tuple[int, int]:
#     '''Represent a 1D array as a 2D array with length as rows of a single-column array.

#     Return:
#         row, column count for a block of ndim 1 or ndim 2.
#     '''
#     if array.ndim == 1:
#         return array.shape[0], 1
#     return array.shape #type: ignore


# def column_2d_filter(array: np.ndarray) -> np.ndarray:
#     '''Reshape a flat ndim 1 array into a 2D array with one columns and rows of length. This is used (a) for getting string representations and (b) for using np.concatenate and np binary operators on 1D arrays.
#     '''
#     # it is not clear when reshape is a copy or a view
#     if array.ndim == 1:
#         return np.reshape(array, (array.shape[0], 1))
#     return array


# def column_1d_filter(array: np.ndarray) -> np.ndarray:
#     '''
#     Ensure that a column that might be 2D or 1D is returned as a 1D array.
#     '''
#     if array.ndim == 2:
#         # could assert that array.shape[1] == 1, but this will raise if does not fit
#         return np.reshape(array, array.shape[0])
#     return array


# def row_1d_filter(array: np.ndarray) -> np.ndarray:
#     '''
#     Ensure that a row that might be 2D or 1D is returned as a 1D array.
#     '''
#     if array.ndim == 2:
#         # could assert that array.shape[0] == 1, but this will raise if does not fit
#         return np.reshape(array, array.shape[1])
#     return array

# NOTE: no longer needed
# def duplicate_filter(values: tp.Iterable[tp.Any]) -> tp.Iterator[tp.Any]:
#     '''
#     Assuming ordered values, yield one of each unique value as determined by __eq__ comparison.
#     '''
#     v_iter = iter(values)
#     try:
#         v = next(v_iter)
#     except StopIteration:
#         return
#     yield v
#     last = v
#     for v in v_iter:
#         if v != last:
#             yield v
#         last = v

def gen_skip_middle(
        forward_iter: TCallableToIter,
        forward_count: int,
        reverse_iter: TCallableToIter,
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

def dtype_from_element(
        value: tp.Any,
        ) -> TDtypeAny:
    '''Given an arbitrary hashable to be treated as an element, return the appropriate dtype. This was created to avoid using np.array(value).dtype, which for a Tuple does not return object.
    '''
    if value is np.nan:
        # NOTE: this will not catch all NaN instances, but will catch any default NaNs in function signatures that reference the same NaN object found on the NP root namespace
        return DTYPE_FLOAT_DEFAULT
    if value is None:
        return DTYPE_OBJECT
    # we want to match np.array elements; they have __len__ but it raises when called
    if value.__class__ is np.ndarray and value.ndim == 0:
        return value.dtype # type: ignore
    # all arrays, or SF containers, should be treated as objects when elements
    # NOTE: might check for __iter__?
    if hasattr(value, '__len__') and not isinstance(value, str) and not isinstance(value, bytes):
        return DTYPE_OBJECT
    # NOTE: calling array and getting dtype on np.nan is faster than combining isinstance, isnan calls
    return np.array(value).dtype

# def resolve_dtype(dt1: np.dtype, dt2: np.dtype) -> np.dtype:
#     '''
#     Given two dtypes, return a compatible dtype that can hold both contents without truncation.
#     '''
#     # NOTE: this is not taking into account endianness; it is not clear if this is important
#     # NOTE: np.dtype(object) == np.object_, so we can return np.object_

#     # if the same, return that dtype
#     if dt1 == dt2:
#         return dt1

#     # if either is object, we go to object
#     if dt1.kind == 'O' or dt2.kind == 'O':
#         return DTYPE_OBJECT

#     dt1_is_str = dt1.kind in DTYPE_STR_KINDS
#     dt2_is_str = dt2.kind in DTYPE_STR_KINDS
#     if dt1_is_str and dt2_is_str:
#         # if both are string or string-like, we can use result type to get the longest string
#         return np.result_type(dt1, dt2)

#     dt1_is_dt = dt1.kind == DTYPE_DATETIME_KIND
#     dt2_is_dt = dt2.kind == DTYPE_DATETIME_KIND
#     if dt1_is_dt and dt2_is_dt:
#         # if both are datetime, result type will work
#         return np.result_type(dt1, dt2)

#     dt1_is_tdelta = dt1.kind == DTYPE_TIMEDELTA_KIND
#     dt2_is_tdelta = dt2.kind == DTYPE_TIMEDELTA_KIND
#     if dt1_is_tdelta and dt2_is_tdelta:
#         # this may or may not work
#         # TypeError: Cannot get a common metadata divisor for NumPy datetime metadata [D] and [Y] because they have incompatible nonlinear base time units
#         try:
#             return np.result_type(dt1, dt2)
#         except TypeError:
#             return DTYPE_OBJECT

#     dt1_is_bool = dt1.type is np.bool_
#     dt2_is_bool = dt2.type is np.bool_

#     # if any one is a string or a bool, we have to go to object; we handle both cases being the same above; result_type gives a string in mixed cases
#     if (dt1_is_str or dt2_is_str
#             or dt1_is_bool or dt2_is_bool
#             or dt1_is_dt or dt2_is_dt
#             or dt1_is_tdelta or dt2_is_tdelta
#             ):
#         return DTYPE_OBJECT

#     # if not a string or an object, can use result type
#     return np.result_type(dt1, dt2)


# def resolve_dtype_iter(dtypes: tp.Iterable[np.dtype]) -> np.dtype:
#     '''Given an iterable of one or more dtypes, do pairwise comparisons to determine compatible overall type. Once we get to object we can stop checking and return object.

#     Args:
#         dtypes: iterable of one or more dtypes.
#     '''
#     dtypes = iter(dtypes)
#     dt_resolve = next(dtypes)

#     for dt in dtypes:
#         dt_resolve = resolve_dtype(dt_resolve, dt)
#         if dt_resolve == DTYPE_OBJECT:
#             return dt_resolve
#     return dt_resolve

def concat_resolved(
        arrays: tp.Iterable[TNDArrayAny],
        axis: int = 0,
        ) -> TNDArrayAny:
    '''
    Concatenation of 1D or 2D arrays that uses resolved dtypes to avoid truncation. Both axis are supported.

    Axis 0 stacks rows (extends columns); axis 1 stacks columns (extends rows).

    No shape manipulation will happen, so it is always assumed that all dimensionalities will be common.
    '''
    if axis is None:
        raise NotImplementedError('no handling of concatenating flattened arrays')

    arrays_seq: tp.Sequence[TNDArrayAny]
    if not hasattr(arrays, '__len__'): # a generator
        arrays_seq = list(arrays)
    else:
        arrays_seq = arrays # type: ignore

    shape: tp.Sequence[int]
    if len(arrays_seq) == 2: # assume we have a sequence
        # faster path when we have two in a sequence
        a1, a2 = arrays_seq
        dt_resolve = resolve_dtype(a1.dtype, a2.dtype)
        size = a1.shape[axis] + a2.shape[axis]
        if a1.ndim == 1:
            shape = (size,)
        else:
            shape = (size, a1.shape[1]) if axis == 0 else (a1.shape[0], size)
    else: # first pass to determine shape and resolved type
        arrays_iter = iter(arrays_seq)
        first = next(arrays_iter)
        dt_resolve = first.dtype
        shape = list(first.shape)

        for array in arrays_iter:
            if dt_resolve != DTYPE_OBJECT:
                dt_resolve = resolve_dtype(array.dtype, dt_resolve)
            shape[axis] += array.shape[axis]

    out: TNDArrayAny = np.empty(shape=shape, dtype=dt_resolve)
    np.concatenate(arrays_seq, out=out, axis=axis)
    out.flags.writeable = False
    return out

def blocks_to_array_2d(
        blocks: tp.Iterable[TNDArrayAny], # can be iterator
        shape: tp.Optional[tp.Tuple[int, int]] = None,
        dtype: tp.Optional[TDtypeAny] = None,
        ) -> TNDArrayAny:
    '''
    Given an iterable of blocks, return a consolidatd array. This is assumed to be an axis 1 style concatenation.
    This is equivalent but more efficient than:
        TypeBlocks.from_blocks(blocks).values
    '''
    discover_dtype = dtype is None
    discover_shape = shape is None
    blocks_is_gen = not hasattr(blocks, '__len__')
    blocks_post: TOptionalArrayList = None

    if discover_shape or discover_dtype:
        # if we have to discover shape or types, we have to do two iterations, and then must load an iterator of `blocks` into a list
        if blocks_is_gen:
            blocks_post = []

        if discover_shape:
            rows = -1
            columns = 0

        for b in blocks:
            if discover_shape:
                if rows == -1:
                    rows = len(b) # works for 1D and 2D
                elif len(b) != rows:
                    raise RuntimeError(f'Invalid block shape {len(b)}')
                if b.ndim == 1:
                    columns += 1
                else:
                    columns += b.shape[1]
            if discover_dtype:
                if dtype is None:
                    dtype = b.dtype
                elif dtype != DTYPE_OBJECT:
                    dtype = resolve_dtype(dtype, b.dtype)
            if blocks_post is not None:
                blocks_post.append(b)

        if discover_shape:
            shape = (rows, columns) #if discover_shape else shape

    if blocks_post is None:
        # blocks might be an iterator if we did not need to discover shape or dtype
        if not blocks_is_gen and len(blocks) == 1: # type: ignore
            return column_2d_filter(blocks[0]) # type: ignore
        blocks_post = blocks #type: ignore
    elif len(blocks_post) == 1:
        # blocks_post is filled; block might be 1d so use filter
        return column_2d_filter(blocks_post[0])

    # NOTE: this is an axis 1 np.concatenate with known shape, dtype
    array: TNDArrayAny = np.empty(shape, dtype=dtype) # type: ignore
    pos = 0
    for b in blocks_post: #type: ignore
        if b.ndim == 1:
            array[NULL_SLICE, pos] = b
            pos += 1
        else:
            end = pos + b.shape[1]
            array[NULL_SLICE, pos: end] = b
            pos = end

    array.flags.writeable = False
    return array

def full_for_fill(
        dtype: tp.Optional[TDtypeAny],
        shape: tp.Union[int, tp.Tuple[int, ...]],
        fill_value: object,
        resolve_fill_value_dtype: bool = True,
        ) -> TNDArrayAny:
    '''
    Return a "full" NP array for the given fill_value
    Args:
        dtype: target dtype, which may or may not be possible given the fill_value. This can be set to None to only use the fill_value to determine dtype.
    '''
    # NOTE: this will treat all no-str iterables as
    if resolve_fill_value_dtype:
        dtype_element = dtype_from_element(fill_value)
        dtype_final = dtype_element if dtype is None else resolve_dtype(dtype, dtype_element)
    else:
        assert dtype is not None
        dtype_final = dtype

    # NOTE: we do not make this array immutable as we sometimes need to mutate it before adding it to TypeBlocks
    if dtype_final != DTYPE_OBJECT:
        return np.full(shape, fill_value, dtype=dtype_final)

    # for tuples and other objects, better to create and fill
    array: TNDArrayAny = np.empty(shape, dtype=DTYPE_OBJECT)

    if fill_value is None:
        return array # None is already set for empty object arrays

    # if we have a generator, None, string, or other simple types, can directly assign
    if isinstance(fill_value, str) or not hasattr(fill_value, '__len__'):
        array[NULL_SLICE] = fill_value
    else:
        for iloc in np.ndindex(shape):
            array[iloc] = fill_value

    return array

def dtype_to_fill_value(dtype: TDtypeSpecifier) -> tp.Any:
    '''Given a dtype, return an appropriate and compatible null value. This is used to provide temporary, "dummy" fill values that reduce type coercions.
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

def array_ufunc_axis_skipna(
        array: TNDArrayAny,
        *,
        skipna: bool,
        axis: int,
        ufunc: TUFunc,
        ufunc_skipna: TUFunc,
        out: tp.Optional[TNDArrayAny] = None
        ) -> tp.Any:
    '''For ufunc array application, when two ufunc versions are available. Expected to always reduce dimensionality.
    '''
    kind = array.dtype.kind
    if kind in DTYPE_NUMERICABLE_KINDS:
        v = array
    elif kind == 'O':
        # replace None with nan
        if skipna:
            is_not_none: TNDArrayAny = np.not_equal(array, None) # type: ignore

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
                v[~is_not_none] = np.nan # pyright: ignore
            else:
                v = array

    elif kind == 'M' or kind == 'm':
        # dates do not support skipna functions
        return ufunc(array, axis=axis, out=out)

    elif kind in DTYPE_STR_KINDS and ufunc in UFUNC_AXIS_STR_TO_OBJ:
        v = array.astype(object)
    else: # normal string dtypes
        v = array

    if skipna:
        return ufunc_skipna(v, axis=axis, out=out)
    return ufunc(v, axis=axis, out=out)

#-------------------------------------------------------------------------------
# unique value discovery; based on NP's arraysetops.py

def argsort_array(
        array: TNDArrayAny,
        kind: TSortKinds = DEFAULT_STABLE_SORT_KIND,
        ) -> TNDArrayAny:
    # NOTE: must use stable sort when returning positions
    if array.dtype.kind == 'O':
        try:
            return array.argsort(kind=kind)
        except TypeError: # if unorderable types
            pass

        array_sortable = np.empty(array.shape, dtype=DTYPE_INT_DEFAULT)

        indices: tp.Dict[tp.Any, int] = {}
        for i, v in enumerate(array):
            array_sortable[i] = indices.setdefault(v, len(indices))
        del indices

        return np.argsort(array_sortable, kind=kind)

    return array.argsort(kind=kind)

def ufunc_unique1d(array: TNDArrayAny) -> TNDArrayAny:
    '''
    Find the unique elements of an array, ignoring shape. Optimized from NumPy implementation based on assumption of 1D array.
    '''
    if array.dtype.kind == 'O':
        try: # some 1D object arrays are sortable
            array = np.sort(array, kind=DEFAULT_FAST_SORT_KIND)
            sortable = True
        except TypeError: # if unorderable types
            sortable = False
        if not sortable:
            # Use a dict to retain order; this will break for non hashables
            store = dict.fromkeys(array)
            array = np.empty(len(store), dtype=object)
            array[:] = tuple(store)
            return array
    else:
        array = np.sort(array, kind=DEFAULT_FAST_SORT_KIND)

    mask = np.empty(array.shape, dtype=DTYPE_BOOL)
    mask[:1] = True # using a slice handles empty mask case
    mask[1:] = array[1:] != array[:-1]

    array = array[mask]
    array.flags.writeable = False
    return array

def ufunc_unique1d_indexer(array: TNDArrayAny,
        ) -> tp.Tuple[TNDArrayAny, TNDArrayAny]:
    '''
    Find the unique elements of an array. Optimized from NumPy implementation based on assumption of 1D array. Returns unique values as well as index positions of those values in the original array.
    '''
    positions = argsort_array(array)

    # get the sorted array
    array = array[positions]

    mask = np.empty(array.shape, dtype=DTYPE_BOOL)
    mask[:1] = True
    mask[1:] = array[1:] != array[:-1]

    values = array[mask] # get unique values
    values.flags.writeable = False
    if len(values) <= 1: # we have only one item
        return values, np.full(mask.shape, 0, dtype=DTYPE_INT_DEFAULT)

    indexer = np.empty(mask.shape, dtype=DTYPE_INT_DEFAULT)
    indexer[positions] = np.cumsum(mask) - 1
    indexer.flags.writeable = False

    return values, indexer

def ufunc_unique1d_positions(array: TNDArrayAny,
        ) -> tp.Tuple[TNDArrayAny, TNDArrayAny]:
    '''
    Find the unique elements of an array. Optimized from NumPy implementation based on assumption of 1D array. Does not return the unique values, but the positions in the original index of those values, as well as the locations of the unique values.
    '''
    positions = argsort_array(array)

    array = array[positions]

    mask = np.empty(array.shape, dtype=DTYPE_BOOL)
    mask[:1] = True
    mask[1:] = array[1:] != array[:-1]

    indexer = np.empty(mask.shape, dtype=np.intp)
    indexer[positions] = np.cumsum(mask) - 1
    indexer.flags.writeable = False

    return positions[mask], indexer


def ufunc_unique1d_counts(array: TNDArrayAny,
        ) -> tp.Tuple[TNDArrayAny, TNDArrayAny]:
    '''
    Find the unique elements of an array. Optimized from NumPy implementation based on assumption of 1D array. Returns unique values as well as the counts of those unique values from the original array.
    '''
    if array.dtype.kind == 'O':
        try: # some 1D object arrays are sortable
            array = np.sort(array, kind=DEFAULT_STABLE_SORT_KIND)
            sortable = True
        except TypeError: # if unorderable types
            sortable = False

        if not sortable:
            # Use a dict to retain order; this will break for non hashables
            store: tp.Dict[TLabel, int] = Counter(array)

            counts = np.empty(len(store), dtype=DTYPE_INT_DEFAULT)
            array = np.empty(len(store), dtype=DTYPE_OBJECT)

            counts[NULL_SLICE] = tuple(store.values())
            array[NULL_SLICE] = tuple(store)

            return array, counts
    else:
        array = np.sort(array, kind=DEFAULT_STABLE_SORT_KIND)

    mask = np.empty(array.shape, dtype=DTYPE_BOOL)
    mask[:1] = True
    mask[1:] = array[1:] != array[:-1]

    pos = nonzero_1d(mask)
    index_of_last_occurrence = np.empty(len(pos) + 1, dtype=pos.dtype)
    index_of_last_occurrence[:-1] = pos
    index_of_last_occurrence[-1] = mask.size

    return array[mask], np.diff(index_of_last_occurrence)

def ufunc_unique_enumerated(
        array: TNDArrayAny,
        *,
        retain_order: bool = False,
        func: tp.Optional[tp.Callable[[tp.Any], bool]] = None,
        ) -> tp.Tuple[TNDArrayAny, TNDArrayAny]:
    # see doc_str.unique_enumerated

    is_2d = array.ndim == 2
    if not is_2d and array.ndim != 1:
        raise ValueError('Only 1D and 2D arrays supported.')

    if not retain_order and not func:
        if is_2d:
            uniques, indexer = ufunc_unique1d_indexer(array.flatten(order='F'))
        else:
            uniques, indexer = ufunc_unique1d_indexer(array)
    else:
        indexer = np.empty(array.size, dtype=DTYPE_INT_DEFAULT)
        indices: tp.Dict[tp.Any, int] = {}

        eiter: tp.Iterator[tp.Tuple[int, tp.Any]]
        if is_2d:
            # NOTE: force F ordering so 2D arrays observe order by column; this returns array elements that need to be converted to Python objects with item()
            eiter = ((i, e.item()) for i, e in enumerate( # type: ignore
                    np.nditer(array, order='F', flags=('refs_ok',))))
        else:
            eiter = enumerate(array)

        if not func:
            for i, v in eiter:
                indexer[i] = indices.setdefault(v, len(indices))
        else:
            for i, v in eiter:
                if func(v):
                    indexer[i] = -1
                else:
                    indexer[i] = indices.setdefault(v, len(indices))

        if array.dtype != DTYPE_OBJECT:
            uniques = np.fromiter(indices.keys(), count=len(indices), dtype=array.dtype)
        else:
            uniques = np.array(list(indices.keys()), dtype=DTYPE_OBJECT)

    if is_2d:
        indexer = indexer.reshape(array.shape, order='F')

    return indexer, uniques

def view_2d_as_1d(array: TNDArrayAny) -> TNDArrayAny:
    '''Given a 2D array, reshape it as a consolidated 1D arrays
    '''
    assert array.ndim == 2
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    # NOTE: this could be cached
    dtype = [(f'f{i}', array.dtype) for i in range(array.shape[1])]
    return array.view(dtype)[NULL_SLICE, 0]

def ufunc_unique2d(array: TNDArrayAny,
        axis: int = 0,
    ) -> TNDArrayAny:
    '''
    Optimized from NumPy implementation.
    '''
    if array.dtype.kind == 'O':
        if axis == 0:
            array_iter = array_to_tuple_iter(array)
        else:
            array_iter = array_to_tuple_iter(array.T)
        # Use a dict to retain order; this will break for non hashables
        # NOTE: could try to sort tuples and do matching, but might fail comparison
        store = dict.fromkeys(array_iter)
        array = np.empty(len(store), dtype=object)
        array[:] = tuple(store)
        return array

    # if not an object array, we can create a 1D structure array and sort that
    if axis == 1:
        array = array.T

    consolidated = view_2d_as_1d(array)
    values = ufunc_unique1d(consolidated)
    values = values.view(array.dtype).reshape(-1, array.shape[1]) # restore dtype, shape
    if axis == 1:
        return values.T
    return values

def ufunc_unique2d_indexer(array: TNDArrayAny,
        axis: int = 0,
        ) -> tp.Tuple[TNDArrayAny, TNDArrayAny]:
    '''
    Find the unique elements of an array and provide an indexer that shows their locations in the original.
    '''
    if axis == 1:
        array = array.T

    if array.dtype.kind == 'O':
        # we must convert to string in order to extract positions data
        consolidated = view_2d_as_1d(array.astype(str))
        positions, indexer = ufunc_unique1d_positions(consolidated)
        values = array[positions] # restore original values

    else:
        consolidated = view_2d_as_1d(array)
        values, indexer = ufunc_unique1d_indexer(consolidated)
        values = values.view(array.dtype).reshape(-1, array.shape[1])

    if axis == 1:
        return values.T, indexer
    return values, indexer

def ufunc_unique(
        array: TNDArrayAny,
        *,
        axis: tp.Optional[int] = None,
        ) -> TNDArrayAny:
    '''
    Extended functionality of the np.unique ufunc, to handle cases of mixed typed objects, where NP will fail in finding unique values for a heterogenous object type.

    Args:

    '''
    if axis is None and array.ndim == 2:
        return ufunc_unique1d(array.reshape(-1)) # reshape over flatten return a view if possible
    elif array.ndim == 1:
        return ufunc_unique1d(array)
    return ufunc_unique2d(array, axis=axis) # type: ignore

def roll_1d(array: TNDArrayAny,
            shift: int
            ) -> TNDArrayAny:
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

    post: TNDArrayAny = np.empty(size, dtype=array.dtype)
    post[0:shift] = array[-shift:]
    post[shift:] = array[0:-shift]
    return post

def roll_2d(array: TNDArrayAny,
            shift: int,
            axis: int
            ) -> TNDArrayAny:
    '''
    Specialized form of np.roll that, by focusing on the 2D solution
    '''
    post: TNDArrayAny = np.empty(array.shape, dtype=array.dtype)

    if axis == 0: # roll rows
        size = array.shape[0]
        if size <= 1:
            return array.copy()

        # result will be positive
        shift = shift % size
        if shift == 0:
            return array.copy()

        post[0:shift, NULL_SLICE] = array[-shift:, NULL_SLICE]
        post[shift:, NULL_SLICE] = array[0:-shift, NULL_SLICE]
        return post

    elif axis == 1: # roll columns
        size = array.shape[1]
        if size <= 1:
            return array.copy()

        # result will be positive
        shift = shift % size
        if shift == 0:
            return array.copy()

        post[NULL_SLICE, 0:shift] = array[NULL_SLICE, -shift:]
        post[NULL_SLICE, shift:] = array[NULL_SLICE, 0:-shift]
        return post

    raise NotImplementedError()

#-------------------------------------------------------------------------------

def _argminmax_1d(
        array: TNDArrayAny,
        ufunc: TUFunc,
        ufunc_skipna: TUFunc,
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
        array: TNDArrayAny,
        ufunc: TUFunc,
        ufunc_skipna: TUFunc,
        skipna: bool = True,
        axis: int = 0
        ) -> TNDArrayAny: # int or float array
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
        ) -> tp.Tuple[TDtypeSpecifier, bool, tp.Sequence[tp.Any]]:
    '''
    Determine an appropriate TDtypeSpecifier for values in an iterable. This does not try to determine the actual dtype, but instead, if the TDtypeSpecifier needs to be object rather than None (which lets NumPy auto detect). This is expected to only operate on 1D data.

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
        dtype: TDtypeSpecifier = None,
        count: tp.Optional[int] = None,
        ) -> tp.Tuple[TNDArrayAny, bool]:
    '''
    Convert an arbitrary Python iterable to a 1D NumPy array without any undesirable type coercion.

    Args:
        count: if provided, can be used to optimize some array creation scenarios.

    Returns:
        pair of array, Boolean, where the Boolean can be used when necessary to establish uniqueness.
    '''
    dtype = None if dtype is None else np.dtype(dtype) # convert dtype specifier to a dtype

    if values.__class__ is np.ndarray:
        if values.ndim != 1: #type: ignore
            raise RuntimeError('expected 1d array')
        if dtype is not None and dtype != values.dtype: #type: ignore
            raise RuntimeError(f'Supplied dtype {dtype} not set on supplied array.')
        return values, len(values) <= 1 #type: ignore

    if values.__class__ is range:
        # translate range to np.arange to avoid iteration
        array = np.arange(start=values.start, #type: ignore
                stop=values.stop, #type: ignore
                step=values.step, #type: ignore
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
        # this returns as dtype only None, or object, letting array constructor do the rest
        dtype, has_tuple, values_for_construct = prepare_iter_for_array(values)
        if len(values_for_construct) == 0:
            return EMPTY_ARRAY, True # no dtype given, so return empty float array
    else: # dtype is provided
        is_gen, copy_values = is_gen_copy_values(values)

        if is_gen and count and dtype.kind not in DTYPE_STR_KINDS:
            if dtype.kind != DTYPE_OBJECT_KIND:
                # if dtype is int this might raise OverflowError
                array = np.fromiter(values,
                        count=count,
                        dtype=dtype,
                        )
            else: # object dtypes
                array = np.empty(count, dtype=dtype)
                for i, element in enumerate(values):
                    array[i] = element
            array.flags.writeable = False
            return array, False

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
        has_tuple = dtype == DTYPE_OBJECT

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
        ) -> TNDArrayAny:
    '''
    Convert an arbitrary Python iterable of iterables to a 2D NumPy array without any undesirable type coercion. Useful IndexHierarchy construction.

    Returns:
        pair of array, Boolean, where the Boolean can be used when necessary to establish uniqueness.
    '''
    if values.__class__ is np.ndarray:
        if values.ndim != 2: #type: ignore
            raise RuntimeError('expected 2d array')
        return values # type: ignore

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

    array: TNDArrayAny = np.array(values, dtype=dtype)
    if array.ndim != 2:
        raise RuntimeError('failed to convert iterable to 2d array')

    array.flags.writeable = False
    return array

def iterable_to_array_nd(
        values: tp.Any,
        ) -> TNDArrayAny:
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

# def slice_to_ascending_slice(
#         key: slice,
#         size: int
#         ) -> slice:
#     '''
#     Given a slice, return a slice that, with ascending integers, covers the same values.

#     Args:
#         size: the length of the container on this axis
#     '''
#     key_step = key.step
#     key_start = key.start
#     key_stop = key.stop

#     if key_step is None or key_step > 0:
#         return key

#     # will get rid of all negative values greater than the size; but will replace None with an appropriate number for usage in range
#     norm_key_start, norm_key_stop, norm_key_step = key.indices(size)

#     # everything else should be descending, but we might have non-descending start, stop
#     if key_start is not None and key_stop is not None:
#         if norm_key_start <= norm_key_stop: # an ascending range
#             return EMPTY_SLICE

#     norm_range = range(norm_key_start, norm_key_stop, norm_key_step)

#     # derive stop
#     if key_start is None:
#         stop = None
#     else:
#         stop = norm_range[0] + 1

#     if key_step == -1:
#         # gets last realized value, not last range value
#         return slice(None if key_stop is None else norm_range[-1], stop, 1)

#     return slice(norm_range[-1], stop, key_step * -1)

def pos_loc_slice_to_iloc_slice(
        key: slice,
        length: int,
        ) -> slice:
    '''Make a positional (integer) exclusive stop key inclusive by adding one to the stop value.
    '''
    if key == NULL_SLICE:
        return key

    # NOTE: we are not validating that this is an integer here
    start = None if key.start is None else key.start

    if key.stop is None:
        stop = None
    else:
        try:
            if key.stop >= length:
                # while a valid slice of positions, loc lookups do not permit over-stating boundaries
                raise LocInvalid(f'Invalid loc: {key}')
        except TypeError as e: # if stop is not an int
            raise LocInvalid(f'Invalid loc: {key}') from e

        stop = key.stop + 1
    return slice(start, stop, key.step)

def key_to_str(key: TLocSelector) -> str:
    if key.__class__ is not slice:
        return str(key)
    if key == NULL_SLICE:
        return ':'

    result = ':' if key.start is None else f'{key.start}:' # type: ignore [union-attr]

    if key.stop is not None: # type: ignore [union-attr]
        result += str(key.stop) # type: ignore [union-attr]
    if key.step is not None and key.step != 1: # type: ignore [union-attr]
        result += f':{key.step}' # type: ignore [union-attr]

    return result

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

DT_NOT_FROM_INT = (DT64_DAY, DT64_MONTH) # year is handled separately

DTU_PYARROW = frozenset(('ns', 'D', 's'))

def to_datetime64(
        value: TDateInitializer,
        dtype: tp.Optional[TDtypeOrDT64] = None
        ) -> np.datetime64:
    '''
    Convert a value ot a datetime64; this must be a datetime64 so as to be hashable.

    Args:
        dtype: Provide the expected dtype of the returned value.
    '''
    if not isinstance(value, np.datetime64):
        if dtype is None:
            # let constructor figure it out; if value is an integer it will raise
            dt = np.datetime64(value) # type: ignore
        else: # assume value is single value;
            # integers will be converted to units from epoch
            if isinstance(value, INT_TYPES):
                if dtype == DT64_YEAR: # convert to string
                    value = str(value)
                elif dtype in DT_NOT_FROM_INT:
                    raise InvalidDatetime64Initializer(f'Attempting to create {dtype} from an integer, which is generally not desired as the result will be an offset from the epoch.')
            # cannot use the datetime directly
            if dtype != np.datetime64:
                dt = np.datetime64(value, np.datetime_data(dtype)[0]) # type: ignore
                # permit NaNs to pass
                if not np.isnan(dt) and dtype == DT64_YEAR:
                    dt_naive = np.datetime64(value) # type: ignore
                    if dt_naive.dtype != dt.dtype:
                        raise InvalidDatetime64Initializer(f'value ({value}) will not be converted to dtype ({dtype})')
            else: # cannot use a generic datetime type
                dt = np.datetime64(value) # type: ignore
    else: # if a dtype was explicitly given, check it
        # value is an instance of a datetime64, and has a dtype attr
        dt = value
        if dtype:
            # dtype can be either generic, or a matching specific dtype
            if dtype != np.datetime64 and dtype != dt.dtype:
                raise InvalidDatetime64Initializer(f'value ({dt}) is not a supported dtype ({dtype})')
    return dt

def to_timedelta64(value: datetime.timedelta) -> np.timedelta64:
    '''
    Convert a datetime.timedelta into a NumPy timedelta64. This approach is better than using np.timedelta64(value), as that reduces all values to microseconds.
    '''
    return reduce(operator.add,
        (np.timedelta64(getattr(value, attr), code) for attr, code in TIME_DELTA_ATTR_MAP if getattr(value, attr) > 0))

def datetime64_not_aligned(array: TNDArrayAny, other: TNDArrayAny) -> bool:
    '''Return True if both arrays are dt64 and they are not aligned by unit. Used in property tests that must skip this condition.
    '''
    array_is_dt64 = array.dtype.kind == DTYPE_DATETIME_KIND
    other_is_dt64 = other.dtype.kind == DTYPE_DATETIME_KIND
    if array_is_dt64 and other_is_dt64:
        return np.datetime_data(array.dtype)[0] != np.datetime_data(other.dtype)[0]
    return False

def timedelta64_not_aligned(array: TNDArrayAny, other: TNDArrayAny) -> bool:
    '''Return True if both arrays are dt64 and they are not aligned by unit. Used in property tests that must skip this condition.
    '''
    array_is_td64 = array.dtype.kind == DTYPE_TIMEDELTA_KIND
    other_is_td64 = other.dtype.kind == DTYPE_TIMEDELTA_KIND
    if array_is_td64 and other_is_td64:
        return np.datetime_data(array.dtype)[0] != np.datetime_data(other.dtype)[0]
    return False


def _slice_to_datetime_slice_args(key: slice,
        dtype: tp.Optional[TDtypeOrDT64] = None
        ) -> tp.Iterator[tp.Optional[np.datetime64]]:
    '''
    Given a slice representing a datetime region, convert to arguments for a new slice, possibly using the appropriate dtype for conversion.
    '''
    for attr in SLICE_ATTRS:
        value = getattr(key, attr)
        if value is None:
            yield None
        elif attr is SLICE_STEP_ATTR:
            # steps are never transformed
            yield value
        else:
            yield to_datetime64(value, dtype=dtype)

def key_to_datetime_key(
        key: TLocSelector,
        dtype: TDtypeOrDT64 = np.datetime64,
        ) -> TLocSelector:
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

    if isinstance(key, INT_TYPES):
        return to_datetime64(key, dtype=dtype)

    if isinstance(key, np.ndarray):
        if key.dtype.kind == 'b' or key.dtype.kind == 'M':
            return key
        if dtype == DT64_YEAR and key.dtype.kind in DTYPE_INT_KINDS:
            key = key.astype(DTYPE_STR)
        return key.astype(dtype)

    if hasattr(key, '__len__'):
        if dtype == DT64_YEAR:
            return np.array([to_datetime64(v, dtype) for v in key], dtype=dtype) # type: ignore
        # use dtype via array constructor to determine type; or just use datetime64 to parse to the passed-in representation
        return np.array(key, dtype=dtype)

    if hasattr(key, '__iter__'): # a generator-like
        if dtype == DT64_YEAR:
            return np.array([to_datetime64(v, dtype) for v in key], dtype=dtype) # pyright: ignore
        return np.array(tuple(key), dtype=dtype) #type: ignore

    # could be None
    return key

#-------------------------------------------------------------------------------

def array_to_groups_and_locations(
        array: TNDArrayAny,
        unique_axis: int = 0,
        ) -> tp.Tuple[TNDArrayAny, TNDArrayAny]:
    '''Locations are index positions for each group.
    Args:
        unique_axis: only used if ndim > 1
    '''
    if array.ndim == 1:
        return ufunc_unique1d_indexer(array)
    return ufunc_unique2d_indexer(array, axis=unique_axis)

# def isna_element(value: tp.Any) -> bool:
#     '''Return Boolean if value is an NA. This does not yet handle pd.NA
#     '''
#     try:
#         return np.isnan(value) #type: ignore
#     except TypeError:
#         pass
#     try:
#         return np.isnat(value) #type: ignore
#     except TypeError:
#         pass
#     return value is None

def isna_array(array: TNDArrayAny,
        include_none: bool = True,
        ) -> TNDArrayAny:
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
    elif kind != DTYPE_OBJECT_KIND:
        return np.full(array.shape, False, dtype=DTYPE_BOOL)

    # only check for None if we have an object type
    with WarningsSilent():
        try:
            if include_none:
                return np.not_equal(array, array) | np.equal(array, None) # type: ignore
            return np.not_equal(array, array)
        except ErrorNotTruthy:
            pass

    # no other option than to do elementwise evaluation
    return np.fromiter(
            (isna_element(e, include_none) for e in array),
            dtype=DTYPE_BOOL,
            count=len(array),
            )

def isfalsy_array(array: TNDArrayAny) -> TNDArrayAny:
    '''
    Return a Boolean array indicating the presence of Falsy or NA values.

    Args:
        array: 1D or 2D array.
    '''
    # NOTE: compare to dtype_to_fill_value
    kind = array.dtype.kind
    # matches all floating point types
    if kind in DTYPE_INEXACT_KINDS:
        return np.isnan(array) | (array == 0.0) # type: ignore
    elif kind == DTYPE_DATETIME_KIND:
        return np.isnat(array)
    elif kind == DTYPE_TIMEDELTA_KIND:
        return np.isnat(array) | (array == EMPTY_TIMEDELTA) # type: ignore
    elif kind == DTYPE_BOOL_KIND:
        return ~array
    elif kind in DTYPE_STR_KINDS:
        return array == '' # type: ignore
    elif kind in DTYPE_INT_KINDS:
        return array == 0 # type: ignore
    elif kind != 'O':
        return np.full(array.shape, False, dtype=DTYPE_BOOL)

    # NOTE: an ArrayKit implementation might out performthis
    post: TNDArrayAny = np.empty(array.shape, dtype=DTYPE_BOOL)
    for coord, v in np.ndenumerate(array):
        post[coord] = not bool(v)
    # or with NaN observations
    return post | np.not_equal(array, array) # type: ignore

def arrays_equal(array: TNDArrayAny,
        other: TNDArrayAny,
        *,
        skipna: bool,
        ) -> bool:
    '''
    Given two arrays, determine if they are equal; support skipping Na comparisons and handling dt64
    '''
    if id(array) == id(other):
        return True

    if (mloc(array) == mloc(other)
            and array.shape == other.shape
            and array.strides == other.strides):
        # NOTE: this implements an TArraySignature check that will short-circuit. A columnar slice from a 2D array will always have a unique array id(); however, two slices from the same 2D array will have the same mloc, shape, and strides; we can identify those cases here
        return True

    if array.dtype.kind == DTYPE_DATETIME_KIND and other.dtype.kind == DTYPE_DATETIME_KIND:
        if np.datetime_data(array.dtype)[0] != np.datetime_data(other.dtype)[0]:
            # do not permit True result between 2021 and 2021-01-01
            return False

    with WarningsSilent():
        # FutureWarning: elementwise comparison failed; returning scalar instead...
        eq = array == other

    # NOTE: will only be False, or an array
    if eq is False:
        return eq

    if skipna:
        isna_both = (isna_array(array, include_none=False)
                & isna_array(other, include_none=False))
        eq[isna_both] = True

    if not eq.all(): # avoid returning a NumPy Bool
        return False
    return True

def binary_transition(
        array: TNDArrayAny,
        axis: int = 0
        ) -> TNDArrayAny:
    '''
    Given a Boolean 1D array, return the index positions (integers) at False values where that False was previously True, or will be True

    Returns:
        For a 1D input, a 1D array of integers; for a 2D input, a 1D object array of lists, where each position corresponds to a found index position. Returning a list is undesirable, but more efficient as a list will be neede for selection downstream.
    '''

    if len(array) == 0:
        return EMPTY_ARRAY_INT

    not_array = ~array

    if array.ndim == 1:
        # non-nan values that go (from left to right) to NaN
        target_sel_leading = (array ^ roll_1d(array, -1)) & not_array
        target_sel_leading[-1] = False # wrap around observation invalid
        # non-nan values that were previously NaN (from left to right)
        target_sel_trailing = (array ^ roll_1d(array, 1)) & not_array
        target_sel_trailing[0] = False # wrap around observation invalid

        return nonzero_1d(target_sel_leading | target_sel_trailing)

    elif array.ndim == 2:
        # if axis == 0, we compare rows going down/up, looking at column values
        # non-nan values that go (from left to right) to NaN
        target_sel_leading = (array ^ roll_2d(array, -1, axis=axis)) & not_array
        # non-nan values that were previously NaN (from left to right)
        target_sel_trailing = (array ^ roll_2d(array, 1, axis=axis)) & not_array

        # wrap around observation invalid
        if axis == 0:
            # process an entire row
            target_sel_leading[-1, NULL_SLICE] = False
            target_sel_trailing[0, NULL_SLICE] = False
        else:
            # process entire column
            target_sel_leading[NULL_SLICE, -1] = False
            target_sel_trailing[NULL_SLICE, 0] = False

        # this dictionary could be very sparse compared to axis dimensionality
        indices_by_axis: tp.DefaultDict[int, tp.List[int]] = defaultdict(list)
        for y, x in zip(*np.nonzero(target_sel_leading | target_sel_trailing)): # pyright: ignore
            if axis == 0:
                # store many rows values for each column
                indices_by_axis[x].append(y)
            else:
                indices_by_axis[y].append(x)

        # if axis is 0, return column width, else return row height
        post: TNDArrayAny = np.empty(dtype=DTYPE_OBJECT, shape=array.shape[not axis])
        for k, v in indices_by_axis.items():
            post[k] = v
        return post

    raise NotImplementedError(f'no handling for array with ndim: {array.ndim}')

#-------------------------------------------------------------------------------

# def array_deepcopy(
#         array: np.ndarray,
#         memo: tp.Optional[tp.Dict[int, tp.Any]],
#         ) -> np.ndarray:
#     '''
#     Create a deepcopy of an array, handling memo lookup, insertion, and object arrays.
#     '''
#     ident = id(array)
#     if memo is not None and ident in memo:
#         return memo[ident]

#     if array.dtype == DTYPE_OBJECT:
#         post = deepcopy(array, memo)
#     else:
#         post = array.copy()

#     if post.ndim > 0:
#         post.flags.writeable = array.flags.writeable

#     if memo is not None:
#         memo[ident] = post
#     return post

#-------------------------------------------------------------------------------
# tools for handling duplicates

def _array_to_duplicated_hashable(
        array: TNDArrayAny,
        axis: int = 0,
        exclude_first: bool = False,
        exclude_last: bool = False) -> TNDArrayAny:
    '''
    Algorithm for finding duplicates in unsortable arrays for hashables. This will always be an object array.
    '''
    # np.unique fails under the same conditions that sorting fails, so there is no need to try np.unique: must go to set drectly.
    len_axis = array.shape[axis]

    value_source: tp.Iterable[tp.Any]
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


    is_dupe: TNDArrayAny = np.full(len_axis, False)

    # could exit early with a set, but would have to hash all array twice to go to set and dictionary
    # creating a list for each entry and tracking indices would be very expensive

    unique_to_first: tp.Dict[TLabel, int] = {} # value to first occurence
    dupe_to_first: tp.Dict[TLabel, int] = {}
    dupe_to_last: tp.Dict[TLabel, int] = {}

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
        array: TNDArrayAny,
        axis: int = 0,
        exclude_first: bool = False,
        exclude_last: bool = False) -> TNDArrayAny:
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

    dupes: TNDArrayAny
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
        array: TNDArrayAny,
        axis: int = 0,
        exclude_first: bool = False,
        exclude_last: bool = False,
        ) -> TNDArrayAny:
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
        array: TNDArrayAny,
        shift: int,
        axis: int, # 0 is rows, 1 is columns
        wrap: bool,
        fill_value: tp.Any = np.nan) -> TNDArrayAny:
    '''
    Apply an np-style roll to a 1D or 2D array; if wrap is False, fill values out-shifted values with fill_value.

    Args:
        fill_value: only used if wrap is False.
    '''

    # works for all shapes
    if shift == 0 or array.shape[axis] == 0:
        shift_mod = 0
    elif shift > 0:
        shift_mod = shift % array.shape[axis]
    elif shift < 0:
        # do negative modulo to force negative value
        shift_mod = shift % -array.shape[axis]

    if (not wrap and shift == 0) or (wrap and shift_mod == 0):
        # must copy so as not let caller mutate argument
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

# def array2d_to_tuples(array: TNDArrayAny) -> tp.Iterator[tp.Tuple[tp.Any, ...]]:
#     yield from map(tuple, array)

# def array_to_tuple_array(array: TNDArrayAny) -> TNDArrayAny:
#     post: TNDArrayAny = np.empty(array.shape[0], dtype=object)
#     for i, row in enumerate(array):
#         post[i] = tuple(row)
#     post.flags.writeable = False
#     return post

def array1d_to_last_contiguous_to_edge(array: TNDArrayAny) -> int:
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
    # count will always be > 0, as all False will hae the last value as False, which is handled above
    if count == length: # if all are true
        return 0

    transitions = np.empty(length, dtype=bool)
    transitions[0] = False # first value not a transition
    # compare current to previous; do not compare first
    np.not_equal(array[:-1], array[1:], out=transitions[1:])
    # last element must be True, so there will always be one transition, and the last transition will mark the boundary of a contiguous region
    return first_true_1d(transitions, forward=False)

#-------------------------------------------------------------------------------
# extension to union and intersection handling

class ManyToOneType(Enum):
    CONCAT = 0
    UNION = 1
    INTERSECT = 2
    DIFFERENCE = 3


def _ufunc_set_1d(
        func: tp.Callable[[TNDArrayAny, TNDArrayAny], TNDArrayAny],
        array: TNDArrayAny,
        other: TNDArrayAny,
        *,
        assume_unique: bool = False
        ) -> TNDArrayAny:
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
    is_union = func is np.union1d
    is_intersection = func is np.intersect1d
    is_difference = func is np.setdiff1d

    if not (is_union or is_intersection or is_difference):
        raise NotImplementedError('unexpected func', func)

    dtype = resolve_dtype(array.dtype, other.dtype)
    post: TNDArrayAny

    # optimizations for empty arrays
    if is_intersection:
        if len(array) == 0 or len(other) == 0:
            # not sure what DTYPE is correct to return here
            post = np.array((), dtype=dtype)
            post.flags.writeable = False
            return post
    elif is_difference:
        if len(array) == 0:
            post = np.array((), dtype=dtype)
            post.flags.writeable = False
            return post

    # np.intersect1d will not handle different dt64 units correctly, but rather "downcast" to the lowest unit, which is not what we want; so, only use np.intersect1d if the units are the same
    array_is_dt64 = array.dtype.kind == DTYPE_DATETIME_KIND
    other_is_dt64 = other.dtype.kind == DTYPE_DATETIME_KIND

    if array_is_dt64 and other_is_dt64:
        if np.datetime_data(array.dtype)[0] != np.datetime_data(other.dtype)[0]:
            raise InvalidDatetime64Comparison()

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
            # NOTE: if these are both dt64 of different units but "aligned" they will return equal
            with WarningsSilent():
                compare = array == other
            # if sizes are the same, the result of == is mostly a bool array; comparison to some arrays (e.g. string), will result in a single Boolean, but it will always be False
            if compare.__class__ is np.ndarray and compare.all(axis=None):
                if is_difference:
                    post = np.array((), dtype=dtype)
                    post.flags.writeable = False
                    return post
                return array

    array_is_str = array.dtype.kind in DTYPE_STR_KINDS
    other_is_str = other.dtype.kind in DTYPE_STR_KINDS

    if (array_is_str ^ other_is_str) or dtype.kind == 'O':
        # NOTE: we convert applicable dt64 types to objects to permit date object to dt64 comparisons when possible
        if array_is_dt64 and is_objectable_dt64(array):
            array = array.astype(DTYPE_OBJECT)
        elif other_is_dt64 and is_objectable_dt64(other):
            # the case of both is handled above
            other = other.astype(DTYPE_OBJECT)

        # NOTE: taking a frozenset of dt64 arrays does not force elements to date/datetime objects, which is what we want here
        with WarningsSilent():
            # NOTE: dt64 element comparisons will warn about elementwise comparison, even those they are elements
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
        func: tp.Callable[[TNDArrayAny, TNDArrayAny], TNDArrayAny],
        array: TNDArrayAny,
        other: TNDArrayAny,
        *,
        assume_unique: bool=False
        ) -> TNDArrayAny:
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

    if is_2d:
        cols = array.shape[1]
        if cols != other.shape[1]:
            raise RuntimeError("cannot perform set operation on arrays with different number of columns")

    # if either are object, or combination resovle to object, get object
    dtype = resolve_dtype(array.dtype, other.dtype)
    post: TNDArrayAny

    # optimizations for empty arrays
    if is_intersection: # intersection with empty
        if len(array) == 0 or len(other) == 0:
            post = np.array((), dtype=dtype)
            if is_2d:
                post = post.reshape(0, cols)
            post.flags.writeable = False
            return post
    elif is_difference:
        if len(array) == 0:
            post = np.array((), dtype=dtype)
            if is_2d:
                post = post.reshape(0, cols)
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
            with WarningsSilent():
                compare = array == other
            # will not match a 2D array of integers and 1D array of tuples containing integers (would have to do a post-set comparison, but would loose order)
            if isinstance(compare, BOOL_TYPES) and compare:
                arrays_are_equal = True #pragma: no cover
            elif isinstance(compare, np.ndarray) and compare.all(axis=None):
                arrays_are_equal = True
            if arrays_are_equal:
                if is_difference:
                    post = np.array((), dtype=dtype)
                    if is_2d:
                        post = post.reshape(0, cols)
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
            with WarningsSilent():
                values: tp.Sequence[tp.Tuple[TLabel, ...]] = sorted(result)
        except TypeError:
            values = tuple(result)

        if is_2d:
            if len(values) == 0:
                post = np.array((), dtype=dtype).reshape(0, cols)
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
        post = func(array, other, **func_kwargs)
        post = post.reshape(len(post), width)
        post.flags.writeable = False
        return post

    # this approach based on https://stackoverflow.com/questions/9269681/intersection-of-2d-numpy-ndarrays
    # we can use a the 1D function on the rows, once converted to a structured array

    dtype_view = [('', array.dtype)] * width
    # creates a view of tuples for 1D operation
    array_view = array.view(dtype_view)
    other_view = other.view(dtype_view)
    post = func(array_view, other_view, **func_kwargs).view(dtype).reshape(-1, width)
    post.flags.writeable = False
    return post

def union1d(array: TNDArrayAny,
        other: TNDArrayAny,
        assume_unique: bool = False
        ) -> TNDArrayAny:
    '''
    Union on 1D array, handling diverse types and short-circuiting to preserve order where appropriate.
    '''
    return _ufunc_set_1d(np.union1d,
            array,
            other,
            assume_unique=assume_unique)

def intersect1d(
        array: TNDArrayAny,
        other: TNDArrayAny,
        assume_unique: bool = False
        ) -> TNDArrayAny:
    '''
    Intersect on 1D array, handling diverse types and short-circuiting to preserve order where appropriate.
    '''
    return _ufunc_set_1d(np.intersect1d,
            array,
            other,
            assume_unique=assume_unique)

def setdiff1d(
        array: TNDArrayAny,
        other: TNDArrayAny,
        assume_unique: bool = False
        ) -> TNDArrayAny:
    '''
    Difference on 1D array, handling diverse types and short-circuiting to preserve order where appropriate
    '''
    return _ufunc_set_1d(np.setdiff1d,
        array,
        other,
        assume_unique=assume_unique)

def union2d(
        array: TNDArrayAny,
        other: TNDArrayAny,
        assume_unique: bool = False
        ) -> TNDArrayAny:
    '''
    Union on 2D array, handling diverse types and short-circuiting to preserve order where appropriate.
    '''
    return _ufunc_set_2d(np.union1d,
            array,
            other,
            assume_unique=assume_unique)

def intersect2d(
        array: TNDArrayAny,
        other: TNDArrayAny,
        assume_unique: bool = False
        ) -> TNDArrayAny:
    '''
    Intersect on 2D array, handling diverse types and short-circuiting to preserve order where appropriate.
    '''
    return _ufunc_set_2d(np.intersect1d,
            array,
            other,
            assume_unique=assume_unique)

def setdiff2d(
        array: TNDArrayAny,
        other: TNDArrayAny,
        assume_unique: bool = False
        ) -> TNDArrayAny:
    '''
    Difference on 2D array, handling diverse types and short-circuiting to preserve order where appropriate.
    '''
    return _ufunc_set_2d(np.setdiff1d,
        array,
        other,
        assume_unique=assume_unique)

MANY_TO_ONE_MAP = {
        (1, ManyToOneType.UNION): union1d,
        (1, ManyToOneType.INTERSECT): intersect1d,
        (1, ManyToOneType.DIFFERENCE): setdiff1d,
        (2, ManyToOneType.UNION): union2d,
        (2, ManyToOneType.INTERSECT): intersect2d,
        (2, ManyToOneType.DIFFERENCE): setdiff2d,
        }

def ufunc_set_iter(
        arrays: tp.Iterable[TNDArrayAny],
        many_to_one_type: ManyToOneType,
        assume_unique: bool = False
        ) -> TNDArrayAny:
    '''
    Iteratively apply a set operation ufunc to 1D or 2D arrays; if all are equal, no operation is performed and order is retained.

    Args:
        arrays: iterator of arrays; can be a Generator.
        union: if True, a union is taken, else, an intersection.
    '''
    # will detect ndim by first value, but insure that all other arrays have the same ndim

    if hasattr(arrays, '__len__') and len(arrays) == 2: # type: ignore
        a1, a2 = arrays
        if a1.ndim != a2.ndim:
            raise RuntimeError('arrays do not all have the same ndim')
        ufunc = MANY_TO_ONE_MAP[(a1.ndim, many_to_one_type)] # pyright: ignore
        result = ufunc(a1, a2, assume_unique=assume_unique)
    else:
        arrays = iter(arrays)
        result = next(arrays)
        ufunc = MANY_TO_ONE_MAP[(result.ndim, many_to_one_type)] # pyright: ignore

        # skip processing for the same array instance
        array_id = id(result)
        for array in arrays:
            if array.ndim != result.ndim:
                raise RuntimeError('arrays do not all have the same ndim')
            if id(array) == array_id:
                continue
            # to retain order on identity, assume_unique must be True
            result = ufunc(result, array, assume_unique=assume_unique)

            if len(result) == 0 and (
                    many_to_one_type is ManyToOneType.INTERSECT
                    or many_to_one_type is ManyToOneType.DIFFERENCE):
                # short circuit for ops with no common values
                break

    result.flags.writeable = False
    return result

def _isin_1d(
        array: TNDArrayAny,
        other: tp.FrozenSet[tp.Any]
        ) -> TNDArrayAny:
    '''
    Iterate over an 1D array to build a 1D Boolean ndarray representing whether or not the original element is in the set

    Args:
        array: The source array
        other: The set of elements being looked for
    '''
    result: TNDArrayAny = np.empty(array.shape, dtype=DTYPE_BOOL)

    for i, element in enumerate(array):
        result[i] = element in other

    result.flags.writeable = False
    return result

def _isin_2d(
        array: TNDArrayAny,
        other: tp.FrozenSet[tp.Any]
        ) -> TNDArrayAny:
    '''
    Iterate over an 2D array to build a 2D, immutable, Boolean ndarray representing whether or not the original element is in the set

    Args:
        array: The source array
        other: The set of elements being looked for
    '''
    result: TNDArrayAny = np.empty(array.shape, dtype=DTYPE_BOOL)

    for (i, j), v in np.ndenumerate(array):
        result[i, j] = v in other

    result.flags.writeable = False
    return result

def isin_array(*,
        array: TNDArrayAny,
        array_is_unique: bool,
        other: TNDArrayAny,
        other_is_unique: bool,
        ) -> TNDArrayAny:
    '''Core isin processing after other has been converted to an array.
    '''
    func: TUFunc
    if array.dtype == DTYPE_OBJECT or other.dtype == DTYPE_OBJECT:
        # both funcs return immutable arrays
        func = _isin_1d if array.ndim == 1 else _isin_2d
        try:
            return func(array, frozenset(other))
        except TypeError: # only occur when something is unhashable.
            pass

    assume_unique = array_is_unique and other_is_unique
    # func = np.in1d if array.ndim == 1 else np.isin
    func = np.isin

    result: TNDArrayBool
    if len(other) == 1:
        # this alternative was implmented due to strange behavior in NumPy when using np.isin with "other" that is one element and an unsigned int
        result = array == other
        if not result.__class__ is np.ndarray:
            result = np.full(array.shape, result, dtype=DTYPE_BOOL)
    else:
        with WarningsSilent():
            # FutureWarning: elementwise comparison failed;
            result = func(array, other, assume_unique=assume_unique)

    result.flags.writeable = False
    return result

def isin(
        array: TNDArrayAny,
        other: tp.Iterable[tp.Any],
        array_is_unique: bool = False,
        ) -> TNDArrayAny:
    '''
    Builds a same-size, immutable, Boolean ndarray representing whether or not the original element is in another ndarray

    numpy's has very poor isin performance, as it converts both arguments to array-like objects.
    This implementation optimizes that by converting the lookup argument into a set, providing constant comparison time.

    Args:
        array: The source array
        other: The elements being looked for
        array_is_unique: if array is known to be unique
    '''
    if hasattr(other, '__len__') and len(other) == 0: #type: ignore
        result: TNDArrayAny = np.full(array.shape, False, dtype=DTYPE_BOOL)
        result.flags.writeable = False
        return result

    other, other_is_unique = iterable_to_array_1d(other)

    return isin_array(array=array,
            array_is_unique=array_is_unique,
            other=other,
            other_is_unique=other_is_unique,
            )

#-------------------------------------------------------------------------------

def array_from_element_attr(*,
        array: TNDArrayAny,
        attr_name: str,
        dtype: TDtypeAny
        ) -> TNDArrayAny:
    '''
    Handle element-wise attribute acesss on arrays of Python date/datetime objects.
    '''
    post: TNDArrayAny
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

def array_from_element_apply(
        array: TNDArrayAny,
        func: TCallableAny,
        dtype: TDtypeAny
        ) -> TNDArrayAny:
    '''
    Handle element-wise function application.
    '''
    post: TNDArrayAny
    if (array.ndim == 1 and dtype != DTYPE_OBJECT
            and dtype.kind not in DTYPE_STR_KINDS):
        post = np.fromiter(
                (func(d) for d in array),
                count=len(array),
                dtype=dtype,
                )
    elif dtype.kind not in DTYPE_STR_KINDS:
        post = np.empty(shape=array.shape, dtype=dtype)
        for iloc, e in np.ndenumerate(array):
            post[iloc] = func(e)
    else: # a string kind that is unsized
        post = np.array([func(e) for e in np.ravel(array)], dtype=dtype).reshape(array.shape)

    post.flags.writeable = False
    return post

def array_from_element_method(*,
        array: TNDArrayAny,
        method_name: str,
        args: tp.Tuple[tp.Any, ...],
        dtype: TDtypeAny,
        pre_insert: tp.Optional[TCallableAny] = None,
        ) -> TNDArrayAny:
    '''
    Handle element-wise method calling on arrays of Python objects. For input arrays of strings or bytes, a string method can be extracted from the appropriate Python type. For other input arrays, the method will be extracted and called for each element.

    Args:
        pre_insert: A function called on each element after the method is called.
        dtype: dtype of array to be returned.
    '''
    # when we know the type of the element, pre-fetch the Python class
    cls_element: tp.Optional[tp.Type[tp.Any]]
    if array.dtype.kind == 'U':
        cls_element = str
    elif array.dtype.kind == 'S':
        cls_element = bytes
    else:
        cls_element = None

    post: TNDArrayAny

    if dtype == DTYPE_STR:
        # if destination is a string, must build into a list first, then construct array to determine dtype size
        if cls_element is not None: # if we can extract function from object first
            func = getattr(cls_element, method_name) #type: ignore
            if array.ndim == 1:
                if pre_insert:
                    proto = [pre_insert(func(d, *args)) for d in array]
                else:
                    proto = [func(d, *args) for d in array]
            else:
                proto = [[None for _ in range(array.shape[1])]
                        for _ in range(array.shape[0])]
                if pre_insert:
                    for (y, x), e in np.ndenumerate(array):
                        proto[y][x] = pre_insert(func(e, *args))
                else:
                    for (y, x), e in np.ndenumerate(array):
                        proto[y][x] = func(e, *args)
        else: # must call getattr for each element
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

    else: # returned dtype is not a string
        if cls_element is not None: # if we can extract function from object first
            func = getattr(cls_element, method_name) #type: ignore
            if array.ndim == 1 and dtype != DTYPE_OBJECT:
                if pre_insert:
                    post = np.fromiter(
                            (pre_insert(func(d, *args)) for d in array),
                            count=len(array),
                            dtype=dtype,
                            )
                else:
                    post = np.fromiter(
                            (func(d, *args) for d in array),
                            count=len(array),
                            dtype=dtype,
                            )
            else: # PERF: slower to always use ndenumerate
                post = np.empty(shape=array.shape, dtype=dtype)
                if pre_insert:
                    for iloc, e in np.ndenumerate(array):
                        post[iloc] = pre_insert(func(e, *args))
                else:
                    for iloc, e in np.ndenumerate(array):
                        post[iloc] = func(e, *args)

        else:
            if array.ndim == 1 and dtype != DTYPE_OBJECT:
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

#-------------------------------------------------------------------------------

class PositionsAllocator:
    '''Resource for re-using a single array of contiguous ascending integers for common applications in IndexBase.
    '''
    _size: int = 1024 # 1048576
    _array: TNDArrayIntDefault = np.arange(_size, dtype=DTYPE_INT_DEFAULT)
    _array.flags.writeable = False

    # NOTE: preliminary tests of using lru-style caching on these instances has not shown a general benfit

    @classmethod
    def get(cls, size: int) -> TNDArrayIntDefault:
        if size == 1:
            return UNIT_ARRAY_INT

        if size > cls._size:
            cls._size = size * 2
            cls._array = np.arange(cls._size, dtype=DTYPE_INT_DEFAULT)
            cls._array.flags.writeable = False
        # slices of immutable arrays are immutable
        return cls._array[:size]


def array_sample(
        array: TNDArrayAny,
        count: int,
        seed: tp.Optional[int] = None,
        sort: bool = False,
        ) -> TNDArrayAny:
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

def run_length_1d(array: TNDArrayAny) -> tp.Tuple[TNDArrayAny, TNDArrayAny]:
    '''Given an array of values, discover contiguous values and their length.

    Return:
        np.ndarray: a value per contiguous width
        np.ndarray: a width per contiguous value
    '''
    assert array.ndim == 1

    size = len(array)
    if size == 0:
        return EMPTY_ARRAY, EMPTY_ARRAY_INT
    if size == 1:
        return array[:1], np.array((size,), dtype=DTYPE_INT_DEFAULT)

    # this provides one True for the start of each region, including the first
    transitions = np.empty(size, dtype=DTYPE_BOOL)
    transitions[0] = True
    transitions[1:] = (array != np.roll(array, 1))[1:]

    # get the index at the the transition for each width
    idx = PositionsAllocator.get(size)[transitions]

    # use the difference in positions to get widths; we need the width from the last transition to the full length in the last position
    widths = np.empty(len(idx), dtype=DTYPE_INT_DEFAULT)
    widths[-1] = size - idx[-1]
    widths[:-1] = (idx - np.roll(idx, 1))[1:]

    return array[transitions], widths

#-------------------------------------------------------------------------------
# json utils

class JSONFilter:
    '''Note: this is filter for encoding NumPy arrays and Python objects in generally readible format by naive consumers. Thus all dates are represented as date strings. This differs from using `__repr__` and attempting to reanimate Python types.
    '''

    @staticmethod
    def encode_element(obj: tp.Any) -> tp.Any:
        '''Convert non-JSON compatible objects to JSON compatible objects or strings. This will recursively process components of iterables.
        '''
        if obj is None:
            return obj
        if isinstance(obj, (str, int, float)):
            return obj
        if isinstance(obj, datetime.date):
            return obj.isoformat()
        if isinstance(obj, (Fraction, complex, np.timedelta64, np.datetime64)):
            return str(obj)
        if hasattr(obj, 'dtype'):
            if obj.dtype.kind in ('c', 'M', 'm'):
                if obj.ndim == 0:
                    return str(obj)
                if obj.ndim == 1:
                    return [str(e) for e in obj]
                return [[str(e) for e in row] for row in obj]
            if obj.ndim == 0:
                return obj.item()
            return obj.tolist()

        if isinstance(obj, dict):
            ee = JSONFilter.encode_element
            return {ee(k): ee(v) for k, v in obj.items()}
        if hasattr(obj, '__iter__'):
            ee = JSONFilter.encode_element
            return [ee(e) for e in obj]
        # let pass and raise on JSON encoding
        return obj

    @classmethod
    def encode_items(cls,
            items: tp.Iterator[tp.Tuple[TLabel, tp.Any]],
            ) -> tp.Any:
        '''Return a key-value mapping. Saves on isinstance checks when we no what the outer container is.
        '''
        ee = cls.encode_element
        return {ee(k): ee(v) for k, v in items}

    @classmethod
    def encode_iterable(cls,
            iterable: tp.Iterable[tp.Any],
            ) -> tp.Any:
        '''Return an iterable. Saves on isinstance checks when we no what the outer container is.
        '''
        ee = cls.encode_element
        return [ee(v) for v in iterable]


class Reanimate:
    RE: tp.Pattern[str]

    @classmethod
    def filter(cls, value: str) -> tp.Any:
        raise NotImplementedError() #pragma: no cover


class ReanimateDT64(Reanimate):
    # NOTE: in NumPy 2, the string repr is changed to use "np" instead of "numpy"
    RE = re.compile(r"(?:numpy|np)\.datetime64\('([-.T:0-9]+)'\)")

    @classmethod
    def filter(cls, value: str) -> tp.Any:
        post = cls.RE.fullmatch(value)
        if post is None:
            return value
        return np.datetime64(post.group(1))


class ReanimateDTD(Reanimate):
    RE = re.compile(r"datetime.date\(([ ,0-9]+)\)")

    @classmethod
    def filter(cls, value: str) -> tp.Any:
        post = cls.RE.fullmatch(value)
        if post is None:
            return value
        args = ast.literal_eval(post.group(1))
        return datetime.date(*args)


class JSONTranslator(JSONFilter):
    '''JSON encoding of select types to permit reanimation. Let fail types that are not encodable and are not explicitly handled.
    '''
    REANIMATEABLE = (ReanimateDT64, ReanimateDTD)

    @staticmethod
    def encode_element(obj: tp.Any) -> tp.Any:
        '''From a Python object, pre-JSON encoding, replacing any Python objects with discoverable strings.
        '''
        if obj is None:
            return obj

        if isinstance(obj, str):
            return obj

        if isinstance(obj, (np.datetime64, datetime.date)):
            return repr(obj) # take repr for encoding / decoding

        if isinstance(obj, dict):
            ee = JSONTranslator.encode_element
            return {ee(k): ee(v) for k, v in obj.items()}
        if hasattr(obj, '__iter__'):
            ee = JSONTranslator.encode_element
            # all iterables must be lists for JSON encoding
            return [ee(e) for e in obj]

        return obj

    @classmethod
    def decode_element(cls, obj: tp.Any) -> tp.Any:
        '''Given an object post JSON conversion, check all strings for strings that can be converted to python objects. Also, all lists are converted to tuples
        '''
        if obj is None:
            return obj

        if isinstance(obj, str):
            # test regular expressions
            for reanimate in cls.REANIMATEABLE:
                post = reanimate.filter(obj)
                if post is not obj:
                    return post
            return obj

        if obj.__class__ is list: # post JSON, only ever be lists
            de = cls.decode_element
            # realize all things JSON gives as lists to tuples
            return tuple(de(e) for e in obj)
        if isinstance(obj, dict):
            de = cls.decode_element
            return {de(k): de(v) for k, v in obj.items()}

        return obj

#-------------------------------------------------------------------------------

def slices_from_targets(
        target_index: tp.Sequence[int] | TNDArrayAny,
        target_values: tp.Iterable[tp.Any],
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

def path_filter(fp: TPathSpecifierOrTextIOOrIterator) -> tp.Union[str, tp.TextIO]:
    '''Realize Path objects as strings, let TextIO pass through, if given.
    '''
    if fp is None:
        raise ValueError('None cannot be interpreted as a file path')
    if isinstance(fp, PathLike):
        return str(fp)
    return fp #type: ignore [return-value]

def write_optional_file(
        content: str,
        fp: tp.Optional[TPathSpecifierOrTextIO] = None,
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
            with tp.cast(StringIO, open(fp, 'w', encoding='utf-8')) as f:
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
        file_like: TPathSpecifierOrTextIOOrIterator,
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
        fields: TNDArrayAny,
        ) -> TTupleCtor:
    '''
    Given fields, try to create a Namedtuple and return the `_make` method
    '''
    # this will raise if attrs are invalid
    try:
        return namedtuple('Axis', fields)._make #type: ignore
    except ValueError:
        pass
    raise ValueError('invalid fields for namedtuple; pass `tuple` as constructor')

def key_normalize(key: TKeyOrKeys) -> tp.List[TLabel]:
    '''
    Normalizing a key that might be a single element or an iterable of keys; expected return is always a list, as it will be used for getitem selection.
    '''
    if isinstance(key, str) or not hasattr(key, '__len__'):
        return [key] # type: ignore
    return key if isinstance(key, list) else list(key) # type: ignore

def iloc_to_insertion_iloc(
        key: int | np.integer[tp.Any],
        size: int,
        ) -> int | np.integer[tp.Any]:
    '''
    Given an iloc (possibly bipolar), return the appropriate insertion iloc (always positive)
    '''
    if key < -size or key >= size:
        raise IndexError(f'index {key} out of range for length {size} container.')
    return key % size
