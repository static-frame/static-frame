import sys
import typing as tp


from collections import OrderedDict
from collections import abc
from io import StringIO
from io import BytesIO
import datetime
from urllib import request


import numpy as np



# min/max fail on object arrays
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

_NULL_SLICE = slice(None)
SLICE_STOP_ATTR = 'stop'
SLICE_STEP_ATTR = 'step'
SLICE_ATTRS = ('start', SLICE_STOP_ATTR, SLICE_STEP_ATTR)

# defaults to float64
_EMPTY_ARRAY = np.array((), dtype=None)
_EMPTY_ARRAY.flags.writeable = False

_DICT_STABLE = sys.version_info >= (3, 6)

#-------------------------------------------------------------------------------
# utility

_INT_TYPES = (int, np.int_)

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


def _gen_skip_middle(
        forward_iter: CallableToIterType,
        forward_count: int,
        reverse_iter: CallableToIterType,
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
    elif array.dtype.kind == 'M':
        # dates do not support skipna functions
        return ufunc(array)
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


def _slice_to_datetime_slice_args(key):
    for attr in SLICE_ATTRS:
        value = getattr(key, attr)
        if value is None:
            yield None
        else:
            yield np.datetime64(value)

def _key_to_datetime_key(key: GetItemKeyType) -> GetItemKeyType:
    '''
    Given an get item key for a Date index, convert it to np.datetime64 representation.
    '''
    if isinstance(key, slice):
        return slice(*_slice_to_datetime_slice_args(key))

    if isinstance(key, np.datetime64):
        return key

    if isinstance(key, str):
        # not using self._DTYPE to coerce type further
        return np.datetime64(key)

    if isinstance(key, np.ndarray):
        if key.dtype.kind == 'b':
            # return Boolean unaltered
            return key
        elif key.dtype.kind == 'M':
            return key
        else:
            return key.astype(np.datetime64)

    if hasattr(key, '__iter__') and hasattr(key, '__len__'):
        return np.fromiter(key, dtype=np.datetime64, count=len(key))

    if hasattr(key, '__len__'):
        return np.array(key, dtype=np.datetime64)

    if hasattr(key, '__next__'): # a generator-like
        return np.array(tuple(key), dtype=np.datetime64)

    # for now, return key unaltered
    return key

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
    # TODO: this needs to handle 2D arrays for hierarchical indices
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

def _array2d_to_tuples(array: np.ndarray) -> tp.Generator[tp.Tuple, None, None]:
    for row in array: # assuming 2d
        yield tuple(row)


def _ufunc2d(func, array, other):
    '''
    Given a 1d set operation, convert to structured array, perform operation, then restore original shape.
    '''
    if array.dtype.kind == 'O' or other.dtype.kind == 'O':
        # TODO: support !D object arrays here:
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
        return np.array(sorted(result), dtype=object)
    else:
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


def _intersect2d(array, other) -> np.ndarray:
    return _ufunc2d(np.intersect1d, array, other)

def _union2d(array, other) -> np.ndarray:
    return _ufunc2d(np.union1d, array, other)


#-------------------------------------------------------------------------------
# URL handling, file downloading

# def _is_url(fp: str) -> bool:
#     # mathc http, https
#     return str.startswith('http')

def _read_url(fp: str):
    with request.urlopen(fp) as response:
        return response.read().decode('utf-8')


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
            common_labels = np.intersect1d(src_index.values, dst_index.values)
            has_common = len(common_labels) > 0
            assert not mixed_depth
        elif depth > 1:
            # if either values arrays are object, we have to covert all values to tuples
            common_labels = _intersect2d(src_index.values, dst_index.values)
            if mixed_depth:
                # when mixed, on the 1D index we have to use loc_to_iloc with tuples
                common_labels = list(_array2d_to_tuples(common_labels))
            has_common = len(common_labels) > 0
        else:
            has_common = False

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

