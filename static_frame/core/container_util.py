'''
This module us for utilty functions that take as input and / or return Container subclasses such as Index, Series, or Frame, and that need to be shared by multiple such Container classes.
'''

from collections import defaultdict

import numpy as np
import typing as tp

if tp.TYPE_CHECKING:
    from static_frame.core.series import Series #pylint: disable=W0611 #pragma: no cover
    from static_frame.core.frame import Frame #pylint: disable=W0611 #pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchy #pylint: disable=W0611 #pragma: no cover
    from static_frame.core.index_auto import IndexAutoFactoryType #pylint: disable=W0611 #pragma: no cover

from static_frame.core.util import IndexConstructor
from static_frame.core.util import IndexConstructors
from static_frame.core.util import IndexInitializer
from static_frame.core.util import STATIC_ATTR
from static_frame.core.util import AnyCallable
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import Bloc2DKeyType
from static_frame.core.util import DtypesSpecifier
from static_frame.core.util import slice_to_ascending_slice
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import DEFAULT_SORT_KIND
from static_frame.core.util import iterable_to_array_1d

from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import DTYPE_STR
from static_frame.core.util import DTYPE_BOOL

from static_frame.core.index_base import IndexBase

def dtypes_mappable(dtypes: DtypesSpecifier):
    '''
    Determine if the dtypes argument can be used by name lookup, rather than index.
    '''
    from static_frame.core.series import Series
    return isinstance(dtypes, (dict, Series))


def is_static(value: IndexConstructor) -> bool:
    try:
        # if this is a class constructor
        return getattr(value, STATIC_ATTR)
    except AttributeError:
        pass
    # assume this is a class method
    return getattr(value.__self__, STATIC_ATTR)


def pandas_version_under_1() -> bool:
    import pandas
    return not hasattr(pandas, 'NA') # object introduced in 1.0

def pandas_to_numpy(
        container: tp.Any,
        own_data: bool,
        fill_value: tp.Any = np.nan
        ) -> np.ndarray:
    '''Convert Pandas container to a numpy array in pandas 1.0, where we might have Pandas extension dtypes that may have pd.NA. If no pd.NA, can go back to numpy types.

    If coming from a Pandas extension type, will convert pd.NA to `fill_value` in the resulting object array. For object dtypes, pd.NA may pass on into SF; the only way to find them is an expensive iteration and `is` comparison, which we are not sure we want to do at this time.

    Args:
        fill_value: if replcaing pd.NA, what to replace it with. Ultimately, this can use FillValueAuto to avoid going to object in all cases.
    '''
    # NOTE: only to be used with pandas 1.0 and greater

    if container.ndim == 1: # Series, Index
        dtype_src = container.dtype
        ndim = 1
    elif container.ndim == 2: # DataFrame, assume contiguous dtypes
        dtypes = container.dtypes.unique()
        assert len(dtypes) == 1
        dtype_src = dtypes[0]
        ndim = 2
    else:
        raise NotImplementedError(f'no handling for ndim {container.ndim}')

    if isinstance(dtype_src, np.dtype):
        dtype = dtype_src
        is_extension_dtype = False
    elif hasattr(dtype_src, 'numpy_dtype'):
        # only int, uint dtypes have this attribute
        dtype = dtype_src.numpy_dtype
        is_extension_dtype = True
    else:
        dtype = None # resolve below
        is_extension_dtype = True

    if is_extension_dtype:
        isna = container.isna() # returns a NumPy Boolean type
        hasna = isna.values.any() # will work for ndim 1 and 2

        from pandas import StringDtype #pylint: disable=E0611
        from pandas import BooleanDtype #pylint: disable=E0611
        # from pandas import DatetimeTZDtype
        # from pandas import Int8Dtype
        # from pandas import Int16Dtype
        # from pandas import Int32Dtype
        # from pandas import Int64Dtype
        # from pandas import UInt16Dtype
        # from pandas import UInt32Dtype
        # from pandas import UInt64Dtype
        # from pandas import UInt8Dtype

        if isinstance(dtype_src, BooleanDtype):
            dtype = DTYPE_OBJECT if hasna else DTYPE_BOOL
        elif isinstance(dtype_src, StringDtype):
            # trying to use a dtype argument for strings results in a converting pd.NA to a string "<NA>"
            dtype = DTYPE_OBJECT if hasna else DTYPE_STR
        else:
            # if an extension type and it hasna, have to go to object; otherwise, set to None or the dtype obtained above
            dtype = DTYPE_OBJECT if hasna else dtype

        try:
            array = container.to_numpy(copy=not own_data, dtype=dtype)
        except (ValueError, TypeError):
            # cannot convert to '<class 'int'>'-dtype NumPy array with missing values. Specify an appropriate 'na_value' for this dtype; this will go to object
            # TypeError: boolean value of NA is ambiguous
            array = container.to_numpy(copy=not own_data)

        if hasna:
            # if hasna and extension dtype, should be an object array; please pd.NA objects with fill_value (np.nan)
            assert array.dtype == DTYPE_OBJECT
            array[isna] = fill_value

    else: # not an extension dtype
        if own_data:
            array = container.values
        else:
            array = container.values.copy()

    array.flags.writeable = False
    return array








def index_from_optional_constructor(
        value: IndexInitializer,
        *,
        default_constructor: IndexConstructor,
        explicit_constructor: tp.Optional[IndexConstructor] = None,
        ) -> IndexBase:
    '''
    Given a value that is an IndexInitializer (which means it might be an Index), determine if that value is really an Index, and if so, determine if a copy has to be made; otherwise, use the default_constructor. If an explicit_constructor is given, that is always used.
    '''
    # NOTE: this might return an own_index flag to show callers when a new index has been created

    if explicit_constructor:
        return explicit_constructor(value)

    # default constructor could be a function with a STATIC attribute
    if isinstance(value, IndexBase):
        # if default is STATIC, and value is not STATIC, get an immutabel
        if is_static(default_constructor): # type: ignore
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
            return value._MUTABLE_CONSTRUCTOR(value)

    # cannot always deterine satic status from constructors; fallback on using default constructor
    return default_constructor(value)

def index_constructor_empty(index: tp.Union[IndexInitializer, 'IndexAutoFactoryType']):
    '''
    Determine if an index is empty (if possible) or an IndexAutoFactory.
    '''
    from static_frame.core.index_auto import IndexAutoFactory

    return index is None or index is IndexAutoFactory or (
            hasattr(index, '__len__') and len(index) == 0)

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

    # NOTE: np.matmul is not the same as np.dot for some arguments
    data = np.matmul(left, right)

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


def axis_window_items( *,
        source: tp.Union['Series', 'Frame'],
        size: int,
        axis: int = 0,
        step: int = 1,
        window_sized: bool = True,
        window_func: tp.Optional[AnyCallable] = None,
        window_valid: tp.Optional[AnyCallable] = None,
        label_shift: int = 0,
        start_shift: int = 0,
        size_increment: int = 0,
        as_array: bool = False,
        ) -> tp.Iterator[tp.Tuple[tp.Hashable, tp.Any]]:
    '''Generator of index, window pairs pairs.

    Args:
        size: integer greater than 0
        step: integer greater than 0 to determine the step size between windows. A step of 1 shifts the window 1 data point; a step equal to window size results in non-overlapping windows.
        window_sized: if True, windows that do not meet the size are skipped.
        window_func: Array processor of window values, pre-function application; useful for applying weighting to the window.
        window_valid: Function that, given an array window, returns True if the window meets requirements and should be returned.
        label_shift: shift, relative to the right-most data point contained in the window, to derive the label to be paired with the window; e.g., to return the first label of the window, the shift will be the size minus one.
        start_shift: shift from 0 to determine where the collection of windows begins.
        size_increment: value to be added to each window aftert the first, so as to, in combination with setting the step size to 0, permit expanding windows.
        as_array: if True, the window is returned as an array instead of a SF object.
    '''
    if size <= 0:
        raise RuntimeError('window size must be greater than 0')
    if step < 0:
        raise RuntimeError('window step cannot be less than than 0')

    source_ndim = source.ndim

    if source_ndim == 1:
        labels = source._index
        if as_array:
            values = source.values
    else:
        labels = source._index if axis == 0 else source._columns
        if as_array:
            values = source._blocks.values

    if start_shift >= 0:
        count_window_max = len(labels)
    else: # add for iterations when less than 0
        count_window_max = len(labels) + abs(start_shift)

    idx_left_max = count_window_max - 1
    idx_left = start_shift
    count = 0

    while True:
        # idx_left, size can change over iterations
        idx_right = idx_left + size - 1

        # floor idx_left at 0 so as to not wrap
        idx_left_floored = max(idx_left, 0)
        idx_right_floored = max(idx_right, -1) # will add one

        key = slice(idx_left_floored, idx_right_floored + 1)

        if source_ndim == 1:
            if as_array:
                window = values[key]
            else:
                window = source._extract_iloc(key)
        else:
            if axis == 0:
                if as_array:
                    window = values[key]
                else: # use low level iloc selector
                    window = source._extract(row_key=key)
            else:
                if as_array:
                    window = values[NULL_SLICE, key]
                else:
                    window = source._extract(column_key=key)

        valid = True
        try:
            idx_label = idx_right + label_shift
            if idx_label < 0: # do not wrap around
                raise IndexError()
            #if we cannot get a lable, the window is invalid
            label = labels.iloc[idx_label]
        except IndexError: # an invalid label has to be dropped
            valid = False

        if valid and window_sized and window.shape[axis] != size:
            valid = False
        if valid and window_valid and not window_valid(window):
            valid = False

        if valid:
            if window_func:
                window = window_func(window)
            yield label, window

        idx_left += step
        size += size_increment
        count += 1

        # import ipdb; ipdb.set_trace()

        if count > count_window_max or idx_left > idx_left_max or size < 0:
            break


def bloc_key_normalize(
        key: Bloc2DKeyType,
        container: 'Frame'
        ) -> np.ndarray:
    '''
    Normalize and validate a bloc key. Return a same sized Boolean array.
    '''
    from static_frame.core.frame import Frame

    if isinstance(key, Frame):
        bloc_frame = key.reindex(
                index=container._index,
                columns=container._columns,
                fill_value=False
                )
        bloc_key = bloc_frame.values # shape must match post reindex
    elif isinstance(key, np.ndarray):
        bloc_key = key
        if bloc_key.shape != container.shape:
            raise RuntimeError(f'bloc {bloc_key.shape} must match shape {container.shape}')
    else:
        raise RuntimeError(f'invalid bloc_key, must be Frame or array, not {key}')

    if not bloc_key.dtype == bool:
        raise RuntimeError('cannot use non-Bolean dtype as bloc key')

    return bloc_key


def key_to_ascending_key(key: GetItemKeyType, size: int) -> GetItemKeyType:
    '''
    Normalize all types of keys into an ascending formation.

    Args:
        size: the length of the container on this axis
    '''
    from static_frame.core.frame import Frame
    from static_frame.core.series import Series

    if isinstance(key, slice):
        return slice_to_ascending_slice(key, size=size)

    if isinstance(key, str) or not hasattr(key, '__len__'):
        return key

    if isinstance(key, np.ndarray):
        # array first as not truthy
        return np.sort(key, kind=DEFAULT_SORT_KIND)

    if not key:
        return key

    if isinstance(key, list):
        return sorted(key)

    if isinstance(key, Series):
        return key.sort_index()

    if isinstance(key, Frame):
        # for usage in assignment we need columns to be sorted
        return key.sort_columns()

    raise RuntimeError(f'unhandled key {key}')



def rehierarch_and_map(*,
        labels: np.ndarray,
        depth_map: tp.Iterable[int],
        index_constructor: IndexConstructor,
        index_constructors: tp.Optional[IndexConstructors] = None,
        name: tp.Hashable = None,
        ) -> tp.Tuple['IndexHierarchy', tp.Sequence[int]]:
    '''
    Given labels suitable for a hierarchical index, order them into a hierarchy using the given depth_map.
    '''

    depth = labels.shape[1] # number of columns

    if depth != len(depth_map):
        raise RuntimeError('must specify new depths for all depths')
    if set(range(depth)) != set(depth_map):
        raise RuntimeError('all depths must be specified')

    labels_post = labels[NULL_SLICE, list(depth_map)]
    labels_sort = np.full(labels_post.shape, 0)

    # get ordering of vlues found in each level
    order = [defaultdict(int) for _ in range(depth)]

    for idx_row, label in enumerate(labels):
        label = tuple(label)
        for idx_col in range(depth):
            if label[idx_col] not in order[idx_col]:
                # Map label to an integer representing the observed order.
                order[idx_col][label[idx_col]] = len(order[idx_col])
            # Fill array for sorting based on observed order.
            labels_sort[idx_row, idx_col] = order[idx_col][label[idx_col]]

    # Reverse depth_map for lexical sorting, which sorts by rightmost column first.
    order_lex = np.lexsort([labels_sort[NULL_SLICE, i] for i in reversed(depth_map)])
    labels_post = labels_post[order_lex]
    labels_post.flags.writeable = False
    index = index_constructor(labels_post,
            index_constructors=index_constructors,
            name=name,
            )
    return index, order_lex



def array_from_value_iter(
        key: tp.Hashable,
        idx: int,
        get_value_iter: tp.Callable[[tp.Hashable], tp.Iterator[tp.Any]],
        get_col_dtype: tp.Optional[tp.Callable],
        row_count: int,
        ):
    '''
    Return a single array given keys and collections.

    Args:
        get_value_iter: Iterator of a values
        dtypes: if an
        key: hashable for looking up field in `get_value_iter`.
        idx: integer position to extract from dtypes
    '''
    # for each column, try to get a column_type, or None
    if get_col_dtype is None:
        column_type = None
    else: # column_type returned here can be None.
        column_type = get_col_dtype(idx)
        # if this value is None we cannot tell if it was explicitly None or just was not specified

    values = None
    if column_type is not None:
        try:
            values = np.fromiter(
                    get_value_iter(key),
                    count=row_count,
                    dtype=column_type)
            values.flags.writeable = False
        except (ValueError, TypeError):
            # the column_type may not be compatible, so must fall back on using np.array to determine the type, i.e., ValueError: cannot convert float NaN to integer
            pass
    if values is None:
        # returns an immutable array
        values, _ = iterable_to_array_1d(
                get_value_iter(key),
                dtype=column_type
                )

    return values



