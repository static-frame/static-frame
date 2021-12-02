import typing as tp
from enum import Enum

from functools import partial
from functools import lru_cache
from itertools import chain
from itertools import repeat

from hypothesis import strategies as st
from hypothesis.extra import numpy as hypo_np
from hypothesis import settings as hypo_settings
from hypothesis import HealthCheck


import numpy as np

from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import DTYPE_BOOL
# from static_frame.core.util import DTYPE_INEXACT_KINDS

from static_frame import TypeBlocks

from static_frame import Index
from static_frame import IndexGO
from static_frame import IndexYear
from static_frame import IndexYearGO
from static_frame import IndexYearMonth
from static_frame import IndexYearMonthGO
from static_frame import IndexDate
from static_frame import IndexDateGO
from static_frame import IndexHour
from static_frame import IndexHourGO
from static_frame import IndexMinute
from static_frame import IndexMinuteGO
from static_frame import IndexMillisecond
from static_frame import IndexMillisecondGO
from static_frame import IndexMicrosecond
from static_frame import IndexMicrosecondGO
from static_frame import IndexNanosecond
from static_frame import IndexNanosecondGO
from static_frame import IndexHierarchy
from static_frame import IndexHierarchyGO
from static_frame import Series
from static_frame import Frame
from static_frame import FrameGO


MAX_ROWS = 8
MAX_COLUMNS = 10


hypo_settings.register_profile("sf",
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
        )
hypo_settings.load_profile("sf")

#-------------------------------------------------------------------------------
# spacings

def subset_contiguous_sum(target: int) -> tp.Tuple[tp.Tuple[int, ...], ...]:
    '''
    Return an iterabel of integers that sum to the target. This does not find all combinations or permutations, just all combintation of the range from 1 to the number (inclusive).
    '''
    # based on https://stackoverflow.com/questions/4632322/finding-all-possible-combinations-of-numbers-to-reach-a-given-sum

    if target == 0:
        return ()

    if not 0 < target <= 32:
        # over sizes of 60 or so performance is noticieable
        raise RuntimeError(f'target is too large: {target}')

    @lru_cache()
    def subset_sum(
            numbers: tp.Sequence[int], partial: tp.Tuple[int, ...] = (), partial_sum: int = 0,
        ) -> tp.Iterator[tp.Tuple[int, ...]]:
        if partial_sum == target:
            yield partial
        if partial_sum > target:
            return
        for i, n in enumerate(numbers, start=1):
            # get pairs of index (starting at 1) and n (the value at each position)
            yield from subset_sum(numbers[i:], partial + (n,), partial_sum + n)

    return tuple(subset_sum(range(1, target+1)))


def get_spacing(size: int = MAX_COLUMNS) -> st.SearchStrategy:
    # generate permutations of the orderings of the integers
    return st.one_of((st.permutations(c) for c in subset_contiguous_sum(size)))


#-------------------------------------------------------------------------------
# values

# 55203 is just before "high surrogates", and avoids this exception
# UnicodeDecodeError: 'utf-32-le' codec can't decode bytes in position 0-3: code point in surrogate code point range(0xd800, 0xe000)
ST_CODEPOINT_LIMIT = dict(min_codepoint=1, max_codepoint=55203)

ST_TYPES_COMMON: tp.Tuple[tp.Callable[..., st.SearchStrategy], ...] = (
        st.integers,
        # st.decimals,
        st.fractions,
        st.dates,
        st.datetimes,
        partial(st.characters, **ST_CODEPOINT_LIMIT),
        partial(st.text, st.characters(**ST_CODEPOINT_LIMIT))  # type: ignore
        )

ST_TYPES_FLOAT_NAN: tp.Tuple[st.SearchStrategy, ...] = (
        st.floats,
        st.complex_numbers,
        )

filter_nan = lambda x: not np.isnan(x)

ST_TYPES_FLOAT_NO_NAN: tp.Tuple[tp.Callable[[], st.SearchStrategy], ...] = (
        lambda: st.floats().filter(filter_nan),
        lambda: st.complex_numbers().filter(filter_nan)
        )

ST_TYPES_UNARY_BINARY = (st.booleans, st.none)

# common collections

ST_TYPES_FOR_UNIQUE = ST_TYPES_FLOAT_NO_NAN + ST_TYPES_COMMON
ST_TYPES_FOR_UNIQUE_MIXED = ST_TYPES_FLOAT_NO_NAN + ST_TYPES_COMMON + ST_TYPES_UNARY_BINARY
ST_VALUE = ST_TYPES_FLOAT_NAN + ST_TYPES_COMMON + ST_TYPES_UNARY_BINARY

def get_value() -> st.SearchStrategy:
    '''
    Any plausible value.
    '''
    return st.one_of(strat() for strat in ST_VALUE)

def get_label() -> st.SearchStrategy:
    '''
    A hashable suitable for use in an Index. While NaNs are supported as labels in Index objects, the unique constraint used below does not enforce uniqueness for NaNs, and thus we must filter out NaNs in advance.
    '''
    return st.one_of((strat() for strat in ST_TYPES_FOR_UNIQUE_MIXED))


def get_labels(
        min_size: int = 0,
        max_size: int = MAX_ROWS) -> st.SearchStrategy:
    '''
    Labels are suitable for creating non-date Indices (though they might include dates); these labels might force an object array result.
    '''
    def gen() -> tp.Iterator[st.SearchStrategy]:

        yield st.lists(get_label(),
                min_size=min_size,
                max_size=max_size,
                unique=True)

        for strat in ST_TYPES_FOR_UNIQUE:
            yield st.lists(
                    strat(),
                    min_size=min_size,
                    max_size=max_size,
                    unique=True)

    return st.one_of(gen())


#-------------------------------------------------------------------------------
# dtypes


class DTGroup(Enum):
    # NOTE: we branch on these enums in get_array_from_dtype_group to add object dtypes when appropriate

    OBJECT = (partial(st.just, DTYPE_OBJECT),)
    ALL = (hypo_np.scalar_dtypes,)

    NUMERIC = (
            hypo_np.floating_dtypes,
            hypo_np.integer_dtypes,
            hypo_np.complex_number_dtypes
            )

    BOOL = (partial(st.just, DTYPE_BOOL),)
    STRING = (hypo_np.unicode_string_dtypes,)

    YEAR = (partial(hypo_np.datetime64_dtypes, min_period='Y', max_period='Y'),)
    YEAR_MONTH = (partial(hypo_np.datetime64_dtypes, min_period='M', max_period='M'),)
    DATE = (partial(hypo_np.datetime64_dtypes, min_period='D', max_period='D'),)
    HOUR = (partial(hypo_np.datetime64_dtypes, min_period='h', max_period='h'),)
    MINUTE = (partial(hypo_np.datetime64_dtypes, min_period='m', max_period='m'),)
    SECOND = (partial(hypo_np.datetime64_dtypes, min_period='s', max_period='s'),)
    MILLISECOND = (partial(hypo_np.datetime64_dtypes, min_period='ms', max_period='ms'),)
    MICROSECOND = (partial(hypo_np.datetime64_dtypes, min_period='us', max_period='us'),)
    NANOSECOND = (partial(hypo_np.datetime64_dtypes, min_period='ns', max_period='ns'),)

    # derived
    NUMERIC_REAL = (
            hypo_np.floating_dtypes,
            hypo_np.integer_dtypes,
            )
    DATETIME = tuple(chain(
            YEAR,
            YEAR_MONTH,
            DATE,
            HOUR,
            SECOND,
            MILLISECOND,
            MICROSECOND,
            NANOSECOND,
            ))

    BASIC = NUMERIC + BOOL + STRING

    # NOTE: duplicate non-datetime to produce more balanced distribution
    CORE = tuple(chain(
            # OBJECT, # object has to be handled with get_array_from_dtype_group
            NUMERIC, NUMERIC, NUMERIC,
            BOOL, BOOL, BOOL,
            STRING, STRING, STRING,
            DATETIME,
            ))

    ALL_NO_OBJECT = tuple(chain(
            ALL, ALL, ALL,
            NUMERIC, NUMERIC, NUMERIC,
            BOOL, BOOL, BOOL,
            STRING, STRING, STRING,
            DATETIME,
            ))

def get_dtype(dtype_group: DTGroup = DTGroup.ALL) -> st.SearchStrategy:

    def st_dts() -> tp.Iterator[st.SearchStrategy]:
        for st_dt in dtype_group.value:
            yield st_dt()

    return st.one_of(st_dts())

def get_dtypes(
        min_size: int = 0,
        max_size: int = MAX_COLUMNS,
        dtype_group: DTGroup = DTGroup.ALL,
        ) -> st.SearchStrategy:
    return st.lists(get_dtype(dtype_group), min_size=min_size)

def get_dtype_pairs(
        dtype_group: DTGroup = DTGroup.ALL,
        ) -> st.SearchStrategy:
    return st.tuples(get_dtype(dtype_group), get_dtype(dtype_group))

#-------------------------------------------------------------------------------
# shape generation

def get_shape_1d(min_size: int = 0, max_size: int = MAX_ROWS) -> st.SearchStrategy:
    return st.tuples(st.integers(min_value=min_size, max_value=max_size))

def get_shape_2d(
        min_rows: int = 1,
        max_rows: int = MAX_ROWS,
        min_columns: int = 1,
        max_columns: int = MAX_COLUMNS,
        ) -> st.SearchStrategy:
    return st.tuples(
            st.integers(min_value=min_rows, max_value=max_rows),
            st.integers(min_value=min_columns, max_value=max_columns)
            )

def get_shape_1d2d(
        min_rows: int = 1,
        max_rows: int = MAX_ROWS,
        min_columns: int = 1,
        max_columns: int = MAX_COLUMNS) -> st.SearchStrategy:

    return st.one_of(
            get_shape_2d(
                    min_rows=min_rows,
                    max_rows=max_rows,
                    min_columns=min_columns,
                    max_columns=max_columns),
            get_shape_1d(
                    min_size=min_rows,
                    max_size=max_rows)
            )

#-------------------------------------------------------------------------------
# array generation

def get_array_object(
        shape: tp.Tuple[int, ...] = (MAX_ROWS, MAX_COLUMNS),
        unique: bool = True) -> st.SearchStrategy:
    if unique:
        # if unique, cannot use fill
        return hypo_np.arrays(
                shape=shape,
                dtype=get_dtype(DTGroup.OBJECT),
                elements=get_value(),
                fill=st.nothing(),
                unique=unique
                )
    return hypo_np.arrays(
            shape=shape,
            dtype=get_dtype(DTGroup.OBJECT),
            elements=get_value(),
            fill=st.none(),
            unique=unique
            )


def get_array_from_dtype_group(
        dtype_group: DTGroup,
        shape: tp.Tuple[int, ...] = (MAX_ROWS, MAX_COLUMNS),
        unique: bool = True,
        ) -> st.SearchStrategy:
    '''
    Given a dtype group and shape, get array. Handles manually creating and filling object arrays when dtype group is object or ALL.
    '''

    # TODO: can remove floating-point NaNs when necessary with .map call with this function on array generators; can apply based on DTYPE group

    # def fill_na(array: np.ndarray) -> np.ndarray:
    #     if array.dtype.kind in DTYPE_INEXACT_KINDS:
    #         is_nan = np.isnan(array)
    #         if is_nan.any():
    #             fill = np.empty(array.shape, dtype=array.dtype)
    #             array[is_nan] = fill[is_nan]
    #             return array
    #     return array

    array_object = get_array_object(
            shape=shape,
            unique=unique
            )
    array_non_object = hypo_np.arrays(
            get_dtype(dtype_group),
            shape,
            unique=unique
            )

    if dtype_group is DTGroup.OBJECT:
        return array_object
    if dtype_group is DTGroup.ALL or dtype_group is DTGroup.CORE:
        return st.one_of(array_non_object, array_non_object, array_object)
    return array_non_object


def get_array_1d(
        min_size: int = 0,
        max_size: int = MAX_ROWS,
        unique: bool = False,
        dtype_group: DTGroup = DTGroup.ALL
        ) -> st.SearchStrategy:

    shape = get_shape_1d(min_size=min_size, max_size=max_size)
    return get_array_from_dtype_group(
            dtype_group=dtype_group,
            shape=shape,
            unique=unique
            )


def get_array_2d(
        min_rows: int = 1,
        max_rows: int = MAX_ROWS,
        min_columns: int = 1,
        max_columns: int = MAX_COLUMNS,
        unique: bool = False,
        dtype_group: DTGroup = DTGroup.ALL
        ) -> st.SearchStrategy:

    shape = get_shape_2d(
            min_rows=min_rows,
            max_rows=max_rows,
            min_columns=min_columns,
            max_columns=max_columns
            )

    return get_array_from_dtype_group(
            dtype_group=dtype_group,
            shape=shape,
            unique=unique
            )


def get_array_1d2d(
        min_rows: int = 1,
        max_rows: int = MAX_ROWS,
        min_columns: int = 1,
        max_columns: int = MAX_COLUMNS,
        dtype_group: DTGroup = DTGroup.ALL
        ) -> st.SearchStrategy:
    '''
    For convenience in building blocks, treat row constraints as 1d size constraints.
    '''
    array_2d = get_array_2d(
            min_rows=min_rows,
            max_rows=max_rows,
            min_columns=min_columns,
            max_columns=max_columns,
            dtype_group=dtype_group
            )

    if 1 in range(min_columns, max_columns + 1):
        # if min/max columns are given, and column of 1 is not supported, it is incorrect to give back a 1D array (in the context of the usage of this in blocks)
        return st.one_of(
                array_2d,
                get_array_1d(
                        min_size=min_rows,
                        max_size=max_rows,
                        dtype_group=dtype_group
                        )
                )
    return array_2d

#-------------------------------------------------------------------------------
# aligend arrays for concatenation and type blocks

def get_arrays_2d_aligned_columns(
        min_size: int = 1,
        max_size: int = 10) -> st.SearchStrategy:

    return st.integers(min_value=1, max_value=MAX_COLUMNS).flatmap(
        lambda columns: st.lists(
            get_array_2d(
                min_columns=columns,
                max_columns=columns
                ),
            min_size=min_size,
            max_size=max_size
            )
    )

def get_arrays_2d_aligned_rows(
        min_size: int = 1,
        max_size: int = 10) -> st.SearchStrategy:

    return st.integers(min_value=1, max_value=MAX_ROWS).flatmap(
        lambda rows: st.lists(
            get_array_2d(
                min_rows=rows,
                max_rows=rows,
                ),
            min_size=min_size,
            max_size=max_size
            )
    )

def get_arrays_2d_aligned(
        min_size: int = 1,
        max_size: int = 10) -> st.SearchStrategy:

    return get_shape_2d().flatmap(
        lambda shape: st.lists(
            get_array_2d(
                min_rows=shape[0],
                max_rows=shape[0],
                min_columns=shape[1],
                max_columns=shape[1],
                ),
            min_size=min_size,
            max_size=max_size
            )
    )

def get_blocks(
        min_rows: int = 1,
        max_rows: int = MAX_ROWS,
        min_columns: int = 1,
        max_columns: int = MAX_COLUMNS,
        dtype_group: DTGroup = DTGroup.ALL
        ) -> st.SearchStrategy:
    '''
    Args:
        min_columns: number of resultant columns in combination of all arrays.
    '''

    def constructor(rows_column_widths: tp.Tuple[int, tp.Iterator[int]]) -> st.SearchStrategy:
        rows, column_widths = rows_column_widths

        def array_gen() -> tp.Iterator[st.SearchStrategy]:
            for width in column_widths:
                yield get_array_1d2d(
                    min_rows=rows,
                    max_rows=rows,
                    min_columns=width,
                    max_columns=width,
                    dtype_group=dtype_group
                    )

        return st.tuples(*array_gen())

    def get_column_widths(shape: tp.Tuple[int, int]) -> st.SearchStrategy:
        rows, columns = shape
        return st.tuples(st.just(rows), get_spacing(columns)).flatmap(constructor)

    return get_shape_2d(
            min_rows=min_rows,
            max_rows=max_rows,
            min_columns=min_columns,
            max_columns=max_columns).flatmap(get_column_widths)


def get_type_blocks(
        min_rows: int = 0,
        max_rows: int = MAX_ROWS,
        min_columns: int = 0,
        max_columns: int = MAX_COLUMNS,
        dtype_group: DTGroup = DTGroup.ALL
        ) -> st.SearchStrategy:
    return st.builds(TypeBlocks.from_blocks,
            get_blocks(min_rows=min_rows,
                    max_rows=max_rows,
                    min_columns=min_columns,
                    max_columns=max_columns,
                    dtype_group=dtype_group)
            )


get_type_blocks_numeric: tp.Callable[..., st.SearchStrategy] = partial(get_type_blocks, dtype_group=DTGroup.NUMERIC)
get_type_blocks_numeric.__name__ = 'get_type_blocks_numeric'


def get_type_blocks_aligned_array(
        min_rows: int = 0,
        max_rows: int = MAX_ROWS,
        min_columns: int = 0,
        max_columns: int = MAX_COLUMNS,
        dtype_group: DTGroup = DTGroup.ALL
        ) -> st.SearchStrategy:
    '''
    Return TypeBlocks instance, as well as an array aligned by row size.
    '''
    def constructor(shape: tp.Tuple[int, int]) -> st.SearchStrategy:
        rows, columns = shape
        return st.tuples(
                get_type_blocks(
                        min_rows=rows,
                        max_rows=rows,
                        min_columns=columns,
                        max_columns=columns,
                        dtype_group=dtype_group
                        ),
                get_array_1d2d( # let columns be variable3
                        min_rows=rows,
                        max_rows=rows,
                        dtype_group=dtype_group
                        )
                )

    return get_shape_2d(
            min_rows=min_rows,
            max_rows=max_rows,
            min_columns=min_columns,
            max_columns=max_columns,
            ).flatmap(constructor)

def get_type_blocks_aligned_type_blocks(
        min_size: int = 0,
        max_size: int = MAX_ROWS,
        min_rows: int = 0,
        max_rows: int = MAX_ROWS,
        min_columns: int = 0,
        max_columns: int = MAX_COLUMNS,
        dtype_group: DTGroup = DTGroup.ALL
        ) -> st.SearchStrategy:
    '''
    Return an iterable of TypeBlocks instances, all alligned by row count
    '''
    def constructor(shape: tp.Tuple[int, int]) -> st.SearchStrategy:
        rows, columns = shape
        return st.lists(
                get_type_blocks(
                        min_rows=rows,
                        max_rows=rows,
                        dtype_group=dtype_group
                        ),
                min_size=min_size,
                max_size=max_size
                )

    return get_shape_2d(
            min_rows=min_rows,
            max_rows=max_rows,
            min_columns=min_columns,
            max_columns=max_columns,
            ).flatmap(constructor)

#-------------------------------------------------------------------------------
# index objects

_INDEX_CLS_TO_DEFAULT_DT_GROUP = dict((
            (Index, DTGroup.CORE),
            (IndexGO, DTGroup.CORE),
            (IndexYear, DTGroup.YEAR),
            (IndexYearGO, DTGroup.YEAR),
            (IndexYearMonth, DTGroup.YEAR_MONTH),
            (IndexYearMonthGO, DTGroup.YEAR_MONTH),
            (IndexDate, DTGroup.DATE),
            (IndexDateGO, DTGroup.DATE),
            (IndexHour, DTGroup.HOUR),
            (IndexHourGO, DTGroup.HOUR),
            (IndexMinute, DTGroup.MINUTE),
            (IndexMinuteGO, DTGroup.MINUTE),
            (IndexMillisecond, DTGroup.MILLISECOND),
            (IndexMillisecondGO, DTGroup.MILLISECOND),
            (IndexMicrosecond, DTGroup.MICROSECOND),
            (IndexMicrosecondGO, DTGroup.MICROSECOND),
            (IndexNanosecond, DTGroup.NANOSECOND),
            (IndexNanosecondGO, DTGroup.NANOSECOND),
            ))

def get_index(
        min_size: int = 0,
        max_size: int = MAX_ROWS,
        dtype_group: tp.Optional[DTGroup] = None,
        cls: tp.Type[Index] = Index
        ) -> st.SearchStrategy:
    # NOTE: have observed cases where a non-unqiue index is returned: with float/int 0, or two NaNs. Need to filter
    # using get_labels here forces Index construction from lists, rather than from arrays
    if dtype_group is not None:
        # NOTE: we cannot product if the dtype we get will align with a datetime64 type
        return st.builds(cls, get_array_1d(
                min_size=min_size,
                max_size=max_size,
                unique=True,
                dtype_group=dtype_group
                ))
    return st.builds(cls, get_labels(min_size=min_size, max_size=max_size))

get_index_date: tp.Callable[..., st.SearchStrategy] = partial(get_index,
        cls=IndexDate,
        dtype_group=DTGroup.DATE)
get_index_date.__name__ = 'get_index_date'

get_index_year: tp.Callable[..., st.SearchStrategy] = partial(get_index,
        cls=IndexYear,
        dtype_group=DTGroup.YEAR)
get_index_year.__name__ = 'get_index_year'

get_index_go: tp.Callable[..., st.SearchStrategy] = partial(get_index, cls=IndexGO)
get_index_go.__name__ = 'get_index_go'


def get_index_any(
        min_size: int = 0,
        max_size: int = MAX_ROWS,
        ) -> st.SearchStrategy:

    strategies = []
    for cls, dtype_group in _INDEX_CLS_TO_DEFAULT_DT_GROUP.items():
        st_index = get_index(
                min_size=min_size,
                max_size=max_size,
                dtype_group=dtype_group,
                cls=cls,
                )
        strategies.append(st_index)

    return st.one_of(strategies)


def get_index_hierarchy(
        min_size: int = 1,
        max_size: int = MAX_ROWS,
        min_depth: int = 2,
        max_depth: int = 5,
        dtype_group: tp.Optional[DTGroup] = None,
        cls: tp.Callable[..., IndexHierarchy] = IndexHierarchy.from_labels
        ) -> st.SearchStrategy:

    def constructor(
            labels_spacings: tp.Tuple[tp.Sequence[tp.Sequence[str]], tp.Sequence[tp.Iterable[int]]]
            ) -> st.SearchStrategy:
        # returns an iterable of labels
        labels_proto, spacings = labels_spacings
        depth = len(labels_proto)
        size = len(labels_proto[0])

        # update all labels (except the deepest) by repeating values a number of times, as determined by spacings
        labels: tp.List[tp.Optional[tp.Sequence[str]]] = [None for _ in range(depth)]
        for d in range(depth):
            if d >= depth - 1:
                labels[d] = labels_proto[d]
            else:
                spacing = spacings[d]

                def spans() -> tp.Iterator[tp.Iterator[str]]:
                    idx = 0
                    for count in spacing:
                        if count == 0:
                            continue
                        yield repeat(labels_proto[d][idx], count)
                        idx += count

                labels[d] = list(chain.from_iterable(spans()))

        def label_gen() -> tp.Iterator[tp.List[str]]:
            for i in range(size):
                yield [tp.cast(tp.Sequence[str], labels[d])[i] for d in range(depth)]

        return st.builds(
                cls,
                st.just(label_gen()) # can just handle a generator
                )

    # generate depth-sized lists of candidate leabels and spacings
    def get_labels_spacings(depth_size: tp.Tuple[int, int]) -> st.SearchStrategy:
        depth, size = depth_size

        if dtype_group is not None:
            level = get_array_1d(min_size=size,
                    max_size=size,
                    unique=True,
                    dtype_group=dtype_group)
        else:
            level = get_labels(min_size=size, max_size=size)

        labels = st.lists(level, min_size=depth, max_size=depth)
        # could do depth minus 1
        spacings = st.lists(get_spacing(size), min_size=depth, max_size=depth)
        return st.tuples(labels, spacings).flatmap(constructor)

    # generate depth and size, pass to get get_labels
    return st.tuples(
            st.integers(min_value=min_depth, max_value=max_depth),
            st.integers(min_value=min_size, max_value=max_size)
            ).flatmap(get_labels_spacings)


def get_index_hierarchy_any(
        min_size: int = 1,
        max_size: int = MAX_ROWS,
        min_depth: int = 2,
        max_depth: int = 5,
        ) -> st.SearchStrategy:


    def get_labels_spacings(depth_size: tp.Tuple[int, int]) -> st.SearchStrategy:
        depth, size = depth_size
        args = []
        for _ in range(depth):
            args.append(get_index_any(
                    min_size=size,
                    max_size=size,
                    ))

        return st.one_of(
                st.builds(IndexHierarchy.from_product, *args),
                st.builds(IndexHierarchyGO.from_product, *args),
                )

    return st.tuples(
            st.integers(min_value=min_depth, max_value=max_depth),
            st.integers(min_value=min_size, max_value=max_size)
            ).flatmap(get_labels_spacings)



#-------------------------------------------------------------------------------
# series objects

def get_series(
        min_size: int = 0,
        max_size: int = MAX_ROWS,
        cls: tp.Type[Series] = Series,
        dtype_group: DTGroup = DTGroup.ALL,
        index_cls: tp.Type[Index] = Index,
        index_dtype_group: tp.Optional[DTGroup] = None
        ) -> st.SearchStrategy:

    def constructor(shape: tp.Tuple[int]) -> st.SearchStrategy:
        size = shape[0] # tuple len 1

        if issubclass(index_cls, IndexHierarchy):
            index = get_index_hierarchy(
                    cls=index_cls.from_labels,
                    min_size=size,
                    max_size=size,
                    dtype_group=index_dtype_group,
            )
        else:
            index = get_index(
                    cls=index_cls,
                    min_size=size,
                    max_size=size,
                    dtype_group=index_dtype_group,
            )

        return st.builds(cls,
            get_array_1d(
                    min_size=size,
                    max_size=size,
                    dtype_group=dtype_group
                    ),
            index=index
            )

    return get_shape_1d(min_size=min_size, max_size=max_size).flatmap(constructor)

# label index, values
get_series_date_numeric: tp.Callable[..., st.SearchStrategy] = partial(get_series,
        dtype_group=DTGroup.NUMERIC,
        index_cls=IndexDate,
        index_dtype_group=DTGroup.DATE
        )
get_series_date_numeric.__name__ = 'get_series_date_numeric'

# depth greater than 1 index
get_series_str_dgt1_numeric: tp.Callable[..., st.SearchStrategy] = partial(get_series,
        min_size=1,
        dtype_group=DTGroup.NUMERIC,
        index_cls=IndexHierarchy,
        index_dtype_group=DTGroup.STRING
        )
get_series_str_dgt1_numeric.__name__ = 'get_series_str_dgt1_numeric'

get_series_obj_dgt1_numeric: tp.Callable[..., st.SearchStrategy] = partial(get_series,
        min_size=1,
        dtype_group=DTGroup.NUMERIC,
        index_cls=IndexHierarchy,
        index_dtype_group=DTGroup.OBJECT
        )
get_series_obj_dgt1_numeric.__name__ = 'get_series_obj_dgt1_numeric'


#-------------------------------------------------------------------------------
# frames

def get_frame(
        min_rows: int = 1,
        max_rows: int = MAX_ROWS,
        min_columns: int = 1,
        max_columns: int = MAX_COLUMNS,
        cls: tp.Type[Frame] = Frame,
        dtype_group: DTGroup = DTGroup.ALL,
        index_cls: tp.Type[Index] = Index,
        index_dtype_group: tp.Optional[DTGroup] = None,
        columns_cls: tp.Type[Index] = Index,
        columns_dtype_group: tp.Optional[DTGroup] = None
        ) -> st.SearchStrategy:

    def constructor(shape: tp.Tuple[int, int]) -> st.SearchStrategy:

        row_count, column_count = shape

        if issubclass(index_cls, IndexHierarchy):
            index = get_index_hierarchy(
                    cls=index_cls.from_labels,
                    min_size=row_count,
                    max_size=row_count,
                    dtype_group=index_dtype_group,
            )
        else:
            index = get_index(
                    cls=index_cls,
                    min_size=row_count,
                    max_size=row_count,
                    dtype_group=index_dtype_group,
            )

        if issubclass(index_cls, IndexHierarchy):
            columns = get_index_hierarchy(
                    cls=columns_cls.from_labels,
                    min_size=column_count,
                    max_size=column_count,
                    dtype_group=columns_dtype_group,
            )
        else:
            columns = get_index(
                    cls=columns_cls,
                    min_size=column_count,
                    max_size=column_count,
                    dtype_group=columns_dtype_group,
            )

        return st.builds(cls,
                get_type_blocks(
                        min_rows=row_count,
                        max_rows=row_count,
                        min_columns=column_count,
                        max_columns=column_count,
                        dtype_group=dtype_group
                        ),
                index=index,
                columns=columns
                )

    return get_shape_2d(min_rows=min_rows,
            max_rows=max_rows,
            min_columns=min_columns,
            max_columns=max_columns
            ).flatmap(constructor)


# label index, columns, values
get_frame_date_str_numeric: tp.Callable[..., st.SearchStrategy] = partial(get_frame,
        dtype_group=DTGroup.NUMERIC,
        index_cls=IndexDate,
        index_dtype_group=DTGroup.DATE,
        columns_cls=Index,
        columns_dtype_group=DTGroup.STRING
        )
get_frame_date_str_numeric.__name__ = 'get_frame_date_str_numeric'

get_frame_go: tp.Callable[..., st.SearchStrategy] = partial(get_frame, cls=FrameGO)
get_frame_go.__name__ = 'get_frame_go'

def get_frame_or_frame_go(
        min_rows: int = 1,
        max_rows: int = MAX_ROWS,
        min_columns: int = 1,
        max_columns: int = MAX_COLUMNS,
        dtype_group: DTGroup = DTGroup.ALL,
        index_cls: tp.Type[Index] = Index,
        index_dtype_group: tp.Optional[DTGroup] = None,
        columns_cls: tp.Type[Index] = Index,
        columns_dtype_group: tp.Optional[DTGroup] = None
        ) -> st.SearchStrategy:
    '''
    Return either a ``Frame`` or a ``FrameGO``,
    '''
    st_frame = get_frame(
            min_rows=min_rows,
            max_rows=max_rows,
            min_columns=min_columns,
            max_columns=max_columns,
            cls=Frame,
            dtype_group=dtype_group,
            index_cls=index_cls,
            index_dtype_group=index_dtype_group,
            columns_cls=columns_cls,
            columns_dtype_group=columns_dtype_group
            )
    st_frame_go = get_frame(
            min_rows=min_rows,
            max_rows=max_rows,
            min_columns=min_columns,
            max_columns=max_columns,
            cls=FrameGO,
            dtype_group=dtype_group,
            index_cls=index_cls,
            index_dtype_group=index_dtype_group,
            columns_cls=columns_cls,
            columns_dtype_group=columns_dtype_group
            )
    return st.one_of((st_frame, st_frame_go))




if __name__ == '__main__':
    import fnmatch
    from argparse import ArgumentParser
    from static_frame.core.display_color import HexColor

    parser = ArgumentParser()
    parser.add_argument('-n', '--name', default=None)
    parser.add_argument('-c', '--count', default=2, type=int)

    options = parser.parse_args()

    local_items = tuple(locals().items())
    for v in (v for k, v in local_items if callable(v) and k.startswith('get')):

        if options.name:
            if not fnmatch.fnmatch(v.__name__, options.name):
                continue

        print(HexColor.format_terminal('grey', '.' * 50))
        print(HexColor.format_terminal('hotpink', str(v.__name__)))

        for x in range(options.count):
            print(HexColor.format_terminal('grey', '.' * 50))
            example = v().example()
            print(repr(example))
