
from functools import partial
from io import StringIO
from io import BytesIO
from itertools import chain
from itertools import product
from itertools import zip_longest
from copy import deepcopy
from operator import itemgetter
from collections.abc import Set
import csv
import json
import sqlite3
import typing as tp
import warnings

import numpy as np
from numpy.ma import MaskedArray #type: ignore
from arraykit import column_1d_filter
from arraykit import name_filter
from arraykit import resolve_dtype
from arraykit import resolve_dtype_iter


from static_frame.core.assign import Assign
from static_frame.core.container import ContainerOperand
from static_frame.core.container_util import array_from_value_iter
from static_frame.core.container_util import arrays_from_index_frame
from static_frame.core.container_util import axis_window_items
from static_frame.core.container_util import bloc_key_normalize
from static_frame.core.container_util import get_col_dtype_factory
from static_frame.core.container_util import index_constructor_empty
from static_frame.core.container_util import index_from_optional_constructor
from static_frame.core.container_util import index_many_concat
from static_frame.core.container_util import index_many_set
from static_frame.core.container_util import key_to_ascending_key
from static_frame.core.container_util import matmul
from static_frame.core.container_util import pandas_to_numpy
from static_frame.core.container_util import pandas_version_under_1
from static_frame.core.container_util import rehierarch_from_index_hierarchy
from static_frame.core.container_util import rehierarch_from_type_blocks
from static_frame.core.container_util import apex_to_name
from static_frame.core.container_util import MessagePackElement
from static_frame.core.container_util import sort_index_for_order
from static_frame.core.container_util import prepare_values_for_lex

from static_frame.core.display import Display
from static_frame.core.display import DisplayActive
from static_frame.core.display_config import DisplayConfig
from static_frame.core.display_config import DisplayFormats
from static_frame.core.display import DisplayHeader
from static_frame.core.doc_str import doc_inject
from static_frame.core.exception import AxisInvalid
from static_frame.core.exception import ErrorInitFrame
from static_frame.core.exception import ErrorInitIndexNonUnique
from static_frame.core.exception import RelabelInvalid
from static_frame.core.index import _index_initializer_needs_init
from static_frame.core.index import immutable_index_filter
from static_frame.core.index import Index
from static_frame.core.index import IndexGO
from static_frame.core.index_auto import IndexAutoFactory
from static_frame.core.index_auto import IndexDefaultFactory
from static_frame.core.index_auto import IndexAutoFactoryType
from static_frame.core.index_auto import RelabelInput
from static_frame.core.index_base import IndexBase
from static_frame.core.index_correspondence import IndexCorrespondence
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.index_hierarchy import IndexHierarchyGO
from static_frame.core.node_dt import InterfaceDatetime
from static_frame.core.node_iter import IterNodeApplyType
from static_frame.core.node_iter import IterNodeAxis
from static_frame.core.node_iter import IterNodeConstructorAxis
from static_frame.core.node_iter import IterNodeDepthLevelAxis
from static_frame.core.node_iter import IterNodeGroupAxis
from static_frame.core.node_iter import IterNodeType
from static_frame.core.node_iter import IterNodeWindow
from static_frame.core.node_selector import InterfaceAssignQuartet
from static_frame.core.node_selector import InterfaceAsType
from static_frame.core.node_selector import InterfaceGetItem
from static_frame.core.node_selector import InterfaceSelectTrio
from static_frame.core.node_str import InterfaceString
from static_frame.core.node_transpose import InterfaceTranspose
from static_frame.core.node_fill_value import InterfaceFillValue
from static_frame.core.node_fill_value import InterfaceFillValueGO
from static_frame.core.node_re import InterfaceRe
from static_frame.core.series import Series
from static_frame.core.store_filter import STORE_FILTER_DEFAULT
from static_frame.core.store_filter import StoreFilter
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.pivot import pivot_derive_constructors
from static_frame.core.pivot import pivot_index_map
from static_frame.core.pivot import extrapolate_column_fields
from static_frame.core.pivot import pivot_records_items
from static_frame.core.pivot import pivot_records_dtypes
from static_frame.core.pivot import pivot_items
from static_frame.core.util import BOOL_TYPES
from static_frame.core.util import _gen_skip_middle
from static_frame.core.util import _read_url
from static_frame.core.util import AnyCallable
from static_frame.core.util import argmax_2d
from static_frame.core.util import argmin_2d
from static_frame.core.util import array_to_duplicated
from static_frame.core.util import array_to_groups_and_locations
from static_frame.core.util import array2d_to_tuples
from static_frame.core.util import Bloc2DKeyType
from static_frame.core.util import CallableOrCallableMap
from static_frame.core.util import DEFAULT_SORT_KIND
from static_frame.core.util import DepthLevelSpecifier
from static_frame.core.util import DTYPE_FLOAT_DEFAULT
from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import dtype_to_fill_value
from static_frame.core.util import DtypeSpecifier
from static_frame.core.util import DtypesSpecifier
from static_frame.core.util import EMPTY_TUPLE
from static_frame.core.util import EMPTY_ARRAY
from static_frame.core.util import FILL_VALUE_DEFAULT
from static_frame.core.util import FRAME_INITIALIZER_DEFAULT
from static_frame.core.util import FrameInitializer
from static_frame.core.util import get_tuple_constructor
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import GetItemKeyTypeCompound
from static_frame.core.util import IndexConstructor
from static_frame.core.util import IndexConstructors
from static_frame.core.util import IndexInitializer
from static_frame.core.util import IndexSpecifier
from static_frame.core.util import INT_TYPES
from static_frame.core.util import is_callable_or_mapping
from static_frame.core.util import is_dtype_specifier
from static_frame.core.util import is_mapping
from static_frame.core.util import isna_array
from static_frame.core.util import iterable_to_array_1d
from static_frame.core.util import iterable_to_array_nd
from static_frame.core.util import isfalsy_array
from static_frame.core.util import Join
from static_frame.core.util import KEY_MULTIPLE_TYPES
from static_frame.core.util import key_normalize
from static_frame.core.util import KeyOrKeys
from static_frame.core.util import NAME_DEFAULT
from static_frame.core.util import NameType
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import Pair
from static_frame.core.util import PairLeft
from static_frame.core.util import PairRight
from static_frame.core.util import path_filter
from static_frame.core.util import PathSpecifier
from static_frame.core.util import PathSpecifierOrFileLike
from static_frame.core.util import PathSpecifierOrFileLikeOrIterator
from static_frame.core.util import UFunc
from static_frame.core.util import ufunc_unique
from static_frame.core.util import write_optional_file
from static_frame.core.util import dtype_kind_to_na
from static_frame.core.util import DTYPE_DATETIME_KIND
from static_frame.core.util import DTU_PYARROW
from static_frame.core.util import DT64_NS
from static_frame.core.util import DTYPE_INT_DEFAULT
from static_frame.core.util import STORE_LABEL_DEFAULT
from static_frame.core.util import file_like_manager
from static_frame.core.util import array2d_to_array1d
from static_frame.core.util import concat_resolved
from static_frame.core.util import CONTINUATION_TOKEN_INACTIVE
from static_frame.core.util import DTYPE_NA_KINDS
from static_frame.core.util import BoolOrBools

from static_frame.core.rank import rank_1d
from static_frame.core.rank import RankMethod

from static_frame.core.style_config import StyleConfig
from static_frame.core.style_config import STYLE_CONFIG_DEFAULT
from static_frame.core.style_config import style_config_css_factory


if tp.TYPE_CHECKING:
    import pandas #pylint: disable=W0611 #pragma: no cover
    from xarray import Dataset #pylint: disable=W0611 #pragma: no cover #type: ignore [attr-defined]
    import pyarrow #pylint: disable=W0611 #pragma: no cover


class Frame(ContainerOperand):
    '''A two-dimensional ordered, labelled collection, immutable and of fixed size.
    '''
    __slots__ = (
            '__weakref__',
            '_blocks',
            '_columns',
            '_index',
            '_name'
            )

    _blocks: TypeBlocks
    _columns: IndexBase
    _index: IndexBase
    _name: tp.Hashable

    _COLUMNS_CONSTRUCTOR = Index
    _COLUMNS_HIERARCHY_CONSTRUCTOR = IndexHierarchy

    _NDIM: int = 2

    #---------------------------------------------------------------------------
    # constructors

    @classmethod
    def from_series(cls,
            series: Series,
            *,
            name: tp.Hashable = None,
            columns_constructor: IndexConstructor = None,
            ) -> 'Frame':
        '''
        Frame constructor from a Series:

        Args:
            series: A Series instance, to be realized as single column, with the column label taken from the `name` attribute.
        '''
        if not isinstance(series, Series):
            raise RuntimeError('from_series must be called with a Series')
        return cls(TypeBlocks.from_blocks(series.values),
                index=series.index,
                columns=(series.name,),
                name=name,
                columns_constructor=columns_constructor,
                own_data=True,
                own_index=True,
                )

    @classmethod
    def from_element(cls,
            element: tp.Any,
            *,
            index: IndexInitializer,
            columns: IndexInitializer,
            dtype: DtypeSpecifier = None,
            name: tp.Hashable = None,
            index_constructor: IndexConstructor = None,
            columns_constructor: IndexConstructor = None,
            own_index: bool = False,
            own_columns: bool = False
            ) -> 'Frame':
        '''
        Create a Frame from an element, i.e., a single value stored in a single cell. Both ``index`` and ``columns`` are required, and cannot be specified with ``IndexAutoFactory``.
        '''
        if own_columns:
            columns_final = columns
        else:
            columns_final = index_from_optional_constructor(columns,
                    default_constructor=cls._COLUMNS_CONSTRUCTOR,
                    explicit_constructor=columns_constructor
                    )
        if own_index:
            index_final = index
        else:
            index_final = index_from_optional_constructor(index,
                    default_constructor=Index,
                    explicit_constructor=index_constructor
                    )

        shape = (len(index_final), len(columns_final))
        if hasattr(element, '__len__') and not isinstance(element, str):
            array = np.empty(shape, dtype=DTYPE_OBJECT)
            # this is the only way to insert tuples, lists,ranges
            for iloc in np.ndindex(shape):
                array[iloc] = element
        else:
            array = np.full(
                    shape,
                    fill_value=element,
                    dtype=dtype)
        array.flags.writeable = False

        return cls(TypeBlocks.from_blocks(array),
                index=index_final,
                columns=columns_final,
                name=name,
                own_data=True,
                own_index=True,
                own_columns=True,
                )


    @classmethod
    def from_elements(cls,
            elements: tp.Iterable[tp.Any],
            *,
            index: tp.Union[IndexInitializer, IndexAutoFactoryType] = None,
            columns: tp.Union[IndexInitializer, IndexAutoFactoryType] = None,
            dtype: DtypeSpecifier = None,
            name: tp.Hashable = None,
            index_constructor: IndexConstructor = None,
            columns_constructor: IndexConstructor = None,
            own_index: bool = False,
            own_columns: bool = False
            ) -> 'Frame':
        '''
        Create a Frame from an iterable of elements, to be formed into a ``Frame`` with a single column.
        '''

        # will be immutable
        array, _ = iterable_to_array_1d(elements, dtype=dtype)

        columns_empty = index_constructor_empty(columns)
        index_empty = index_constructor_empty(index)

        #-----------------------------------------------------------------------
        if own_columns:
            columns_final = columns
            col_count = len(columns_final)
        elif columns_empty:
            col_count = 1
            columns_final = IndexAutoFactory.from_optional_constructor(
                    col_count, # default to one colmns
                    default_constructor=cls._COLUMNS_CONSTRUCTOR,
                    explicit_constructor=columns_constructor
                    )
        else:
            columns_final = index_from_optional_constructor(columns,
                    default_constructor=cls._COLUMNS_CONSTRUCTOR,
                    explicit_constructor=columns_constructor
                    )
            col_count = len(columns_final)

        #-----------------------------------------------------------------------
        row_count = len(array)
        if own_index:
            index_final = index
        elif index_empty:
            index_final = IndexAutoFactory.from_optional_constructor(
                    row_count,
                    default_constructor=Index,
                    explicit_constructor=index_constructor
                    )
        else:
            index_final = index_from_optional_constructor(index,
                    default_constructor=Index,
                    explicit_constructor=index_constructor
                    )

        #-----------------------------------------------------------------------
        if col_count > 1:
            array = np.tile(array.reshape((row_count, 1)), (1, col_count))
            array.flags.writeable = False

        return cls(TypeBlocks.from_blocks(array),
                index=index_final,
                columns=columns_final,
                name=name,
                own_data=True,
                own_index=True,
                own_columns=True,
                )



    #---------------------------------------------------------------------------
    @classmethod
    @doc_inject(selector='constructor_frame')
    def from_concat(cls,
            frames: tp.Iterable[tp.Union['Frame', Series]],
            *,
            axis: int = 0,
            union: bool = True,
            index: tp.Union[IndexInitializer, IndexAutoFactoryType] = None,
            columns: tp.Union[IndexInitializer, IndexAutoFactoryType] = None,
            name: NameType = None,
            fill_value: object = np.nan,
            consolidate_blocks: bool = False
            ) -> 'Frame':
        '''
        Concatenate multiple Frames into a new Frame. If index or columns are provided and appropriately sized, the resulting Frame will use those indices. If the axis along concatenation (index for axis 0, columns for axis 1) is unique after concatenation, it will be preserved; otherwise, a new index or an :obj:`IndexAutoFactory` must be supplied.

        Args:
            frames: Iterable of Frames.
            axis: Integer specifying 0 to concatenate supplied Frames vertically (aligning on columns), 1 to concatenate horizontally (aligning on rows).
            union: If True, the union of the aligned indices is used; if False, the intersection is used.
            index: Optionally specify a new index.
            columns: Optionally specify new columns.
            {name}f
            {consolidate_blocks}

        Returns:
            :obj:`static_frame.Frame`
        '''

        # when doing axis 1 concat (growin horizontally) Series need to be presented as rows (axis 0)
        # NOTE: might check for Series that do not have names
        frames = [f if isinstance(f, Frame) else f.to_frame(axis) for f in frames]

        own_columns = False
        own_index = False

        if not frames:
            return cls(
                    index=index,
                    columns=columns,
                    name=name,
                    own_columns=own_columns,
                    own_index=own_index)

        own_index = False
        own_columns = False

        if axis == 1: # stacks columns (extends rows horizontally)
            # index can be the same, columns must be redefined if not unique
            if columns is IndexAutoFactory:
                columns = None # let default creation happen
            elif columns is None:
                try:
                    columns = index_many_concat(
                            (f._columns for f in frames),
                            cls._COLUMNS_CONSTRUCTOR,
                            )
                except ErrorInitIndexNonUnique:
                    raise ErrorInitFrame('Column names after horizontal concatenation are not unique; supply a columns argument or IndexAutoFactory.')
                own_columns = True

            if index is IndexAutoFactory:
                raise ErrorInitFrame('for axis 1 concatenation, index must be used for reindexing row alignment: IndexAutoFactory is not permitted')
            elif index is None:
                index = index_many_set(
                        (f._index for f in frames),
                        Index,
                        union=union,
                        )
                own_index = True

            def blocks() -> tp.Iterator[np.ndarray]:
                for frame in frames:
                    if len(frame.index) != len(index) or (frame.index != index).any():
                        frame = frame.reindex(index=index, fill_value=fill_value)
                    for block in frame._blocks._blocks:
                        yield block

        elif axis == 0: # stacks rows (extends columns vertically)
            if index is IndexAutoFactory:
                index = None # let default creation happen
            elif index is None:
                try:
                    index = index_many_concat((f._index for f in frames), Index)
                except ErrorInitIndexNonUnique:
                    raise ErrorInitFrame('Index names after vertical concatenation are not unique; supply an index argument or IndexAutoFactory.')
                own_index = True

            if columns is IndexAutoFactory:
                raise ErrorInitFrame('for axis 0 concatenation, columns must be used for reindexing and column alignment: IndexAutoFactory is not permitted')
            elif columns is None:
                columns = index_many_set(
                        (f._columns for f in frames),
                        cls._COLUMNS_CONSTRUCTOR,
                        union=union,
                        )
                own_columns = True

            def blocks() -> tp.Iterator[np.ndarray]:
                type_blocks = []
                previous_frame = None
                block_compatible = True
                reblock_compatible = True

                for frame in frames:
                    if len(frame.columns) != len(columns) or (frame.columns != columns).any():
                        frame = frame.reindex(columns=columns, fill_value=fill_value)

                    type_blocks.append(frame._blocks)
                    # column size is all the same by this point
                    if previous_frame is not None: # after the first
                        if block_compatible:
                            block_compatible &= frame._blocks.block_compatible(
                                    previous_frame._blocks,
                                    axis=1) # only compare columns
                        if reblock_compatible:
                            reblock_compatible &= frame._blocks.reblock_compatible(
                                    previous_frame._blocks)
                    previous_frame = frame

                yield from TypeBlocks.vstack_blocks_to_blocks(
                        type_blocks=type_blocks,
                        block_compatible=block_compatible,
                        reblock_compatible=reblock_compatible,
                        )
        else:
            raise AxisInvalid(f'no support for {axis}')

        if consolidate_blocks:
            block_gen = lambda: TypeBlocks.consolidate_blocks(blocks())
        else:
            block_gen = blocks

        return cls(TypeBlocks.from_blocks(block_gen()),
                index=index,
                columns=columns,
                name=name,
                own_data=True,
                own_columns=own_columns,
                own_index=own_index)

    @classmethod
    def from_concat_items(cls,
            items: tp.Iterable[tp.Tuple[tp.Hashable, tp.Union['Frame', Series]]],
            *,
            axis: int = 0,
            union: bool = True,
            name: NameType = None,
            fill_value: object = np.nan,
            index_constructor: tp.Optional[IndexConstructor] = None,
            columns_constructor: tp.Optional[IndexConstructor] = None,
            consolidate_blocks: bool = False,
            ) -> 'Frame':
        '''
        Produce a :obj:`Frame` with a hierarchical index from an iterable of pairs of labels, :obj:`Frame`. The :obj:`IndexHierarchy` is formed from the provided labels and the :obj:`Index` if each :obj:`Frame`.

        Args:
            items: Iterable of pairs of label, :obj:`Frame`
            axis:
            union:
            name:
            fill_value:
            index_constructor:
            columns_constructor:
            consolidate_blocks:
        '''
        frames = []

        def gen() -> tp.Iterator[tp.Tuple[tp.Hashable, IndexBase]]:
            # default index construction does not yield elements, but instead yield Index objects for more efficient IndexHierarchy construction
            yield_elements = True
            if axis == 0 and (index_constructor is None or isinstance(index_constructor, IndexDefaultFactory)):
                yield_elements = False
            elif axis == 1 and (columns_constructor is None or isinstance(columns_constructor, IndexDefaultFactory)):
                yield_elements = False

            for label, frame in items:
                # must normalize Series here to avoid down-stream confusion
                if isinstance(frame, Series):
                    frame = frame.to_frame(axis)

                frames.append(frame)
                if axis == 0:
                    if yield_elements:
                        yield from product((label,), frame._index)
                    else:
                        yield label, frame._index
                elif axis == 1:
                    if yield_elements:
                        yield from product((label,), frame._columns)
                    else:
                        yield label, frame._columns

                # we have already evaluated AxisInvalid

        # populates array_values as side effect
        if axis == 0:
            # ih = IndexHierarchy.from_index_items(gen())
            ih = index_from_optional_constructor(
                    gen(),
                    default_constructor=IndexHierarchy.from_index_items,
                    explicit_constructor=index_constructor,
                    )
            if columns_constructor is not None:
                raise NotImplementedError('using columns_constructor for axis 0 not yet supported')
            kwargs = dict(index=ih)
        elif axis == 1:
            # ih = cls._COLUMNS_HIERARCHY_CONSTRUCTOR.from_index_items(gen())
            ih = index_from_optional_constructor(
                    gen(),
                    default_constructor=cls._COLUMNS_HIERARCHY_CONSTRUCTOR.from_index_items,
                    explicit_constructor=columns_constructor,
                    )
            if index_constructor is not None:
                raise NotImplementedError('using index_constructor for axis 1 not yet supported')
            kwargs = dict(columns=ih)
        else:
            raise AxisInvalid(f'invalid axis: {axis}')

        return cls.from_concat(frames,
                axis=axis,
                union=union,
                name=name,
                fill_value=fill_value,
                consolidate_blocks=consolidate_blocks,
                **kwargs
                )

    @classmethod
    def from_overlay(cls,
            containers: tp.Iterable['Frame'],
            *,
            index: tp.Optional[IndexInitializer] = None,
            columns: tp.Optional[IndexInitializer] = None,
            union: bool = True,
            name: NameType = None,
            ) -> 'Frame':
        '''
        Return a new :obj:`Frame` made by overlaying containers, filling in missing values (None or NaN) with aligned values from subsequent containers.

        Args:
            containers: Iterable of :obj:`Frame`.
            index: An optional :obj:`Index`, :obj:`IndexHierarchy`, or index initializer, to be used as the index upon which all containers are aligned. :obj:`IndexAutoFactory` is not supported.
            columns: An optional :obj:`Index`, :obj:`IndexHierarchy`, or columns initializer, to be used as the columns upon which all containers are aligned. :obj:`IndexAutoFactory` is not supported.
            union: If True, and no ``index`` or ``columns`` argument is supplied, a union index or columns from ``containers`` will be used; if False, the intersection index or columns will be used.
        '''
        if not hasattr(containers, '__len__'):
            containers = tuple(containers) # exhaust a generator

        if index is None:
            index = index_many_set(
                    (c.index for c in containers),
                    cls_default=Index,
                    union=union,
                    )
        else:
            index = index_from_optional_constructor(index,
                    default_constructor=Index
                    )
        if columns is None:
            columns = index_many_set(
                    (c.columns for c in containers),
                    cls_default=cls._COLUMNS_CONSTRUCTOR,
                    union=union,
                    )
        else:
            columns = index_from_optional_constructor(columns,
                    default_constructor=cls._COLUMNS_CONSTRUCTOR)

        fill_arrays = dict() # NOTE: we will hash to NaN and NaT, but can assume we are using the same instance

        containers_iter = iter(containers)
        container = next(containers_iter)
        fill_value = dtype_kind_to_na(container._blocks._row_dtype.kind)
        post = container.reindex(
                index=index,
                columns=columns,
                fill_value=fill_value,
                own_index=True,
                own_columns=True,
                )
        for container in containers_iter:
            values = []
            for col, dtype_at_col in post.dtypes.items():
                if col not in container:
                    # get fill value based on previous container
                    fill_value = dtype_kind_to_na(dtype_at_col.kind)
                    if fill_value not in fill_arrays:
                        array = np.full(len(index), fill_value)
                        array.flags.writeable = False
                        fill_arrays[fill_value] = array
                    array = fill_arrays[fill_value]
                else:
                    col_series = container[col]
                    fill_value = dtype_kind_to_na(col_series.dtype.kind)
                    array = col_series.reindex(index, fill_value=fill_value).values
                    array.flags.writeable = False
                values.append(array)

            post = cls(
                    post._blocks.fill_missing_by_values(values, func=isna_array),
                    index=index,
                    columns=columns,
                    name=name,
                    own_data=True,
                    own_index=True,
                    own_columns=True,
                    )

            if not post.isna().any().any():
                break

        return post


    @classmethod
    @doc_inject(selector='constructor_frame')
    def from_records(cls,
            records: tp.Iterable[tp.Any],
            *,
            index: tp.Optional[IndexInitializer] = None,
            columns: tp.Optional[IndexInitializer] = None,
            dtypes: DtypesSpecifier = None,
            name: tp.Hashable = None,
            consolidate_blocks: bool = False,
            index_constructor: IndexConstructor = None,
            columns_constructor: IndexConstructor = None,
            own_index: bool = False,
            own_columns: bool = False
            ) -> 'Frame':
        '''Construct a :obj:`Frame` from an iterable of rows, where rows are defined as iterables, including tuples, lists, and arrays. If each row is a NamedTuple, and ``columns`` is not provided, column names will be derived from the NamedTuple fields.

        Supplying ``dtypes`` will significantly improve performance, as otherwise columnar array types must be derived by element-wise examination.

        For records defined as ``Series``, use ``Frame.from_concat``; for records defined as dictionary, use ``Frame.from_dict_records``; for creating a ``Frame`` from a single dictionary, where keys are column labels and values are columns, use ``Frame.from_dict``.

        Args:
            records: Iterable of row values, where row values are arrays, tuples, lists, or namedtuples. For dictionary records, use ``Frame.from_dict_records``.
            index: Optionally provide an iterable of index labels, equal in length to the number of records. If a generator, this value will not be evaluated until after records are loaded.
            columns: Optionally provide an iterable of column labels, equal in length to the number of elements in a row.
            {dtypes}
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`Frame`
        '''
        # if records is np; we can just pass it to constructor, as is already a consolidated type
        if records.__class__ is np.ndarray:
            if dtypes is not None:
                raise ErrorInitFrame('specifying dtypes when using NP records is not permitted')
            return cls(records,
                    index=index,
                    columns=columns,
                    index_constructor=index_constructor,
                    columns_constructor=columns_constructor,
                    own_index=own_index,
                    own_columns=own_columns,
                    name=name,
                    )

        if not hasattr(records, '__len__'):
            # might be a generator; must convert to sequence
            rows = list(records)
        else: # could be a sequence, or something like a dict view
            rows = records
        row_count = len(rows)

        if not row_count:
            if columns is not None: # we can create a zero-record Frame
                return cls(
                        columns=columns,
                        columns_constructor=columns_constructor,
                        own_columns=own_columns,
                        name=name,
                        )
            raise ErrorInitFrame('no rows available in records, and no columns defined.')

        if hasattr(rows, '__getitem__'):
            rows_to_iter = False
            row_reference = rows[0]
        else: # dict view, or other sized iterable that does not support getitem
            rows_to_iter = True
            row_reference = next(iter(rows))

        if isinstance(row_reference, Series):
            raise ErrorInitFrame('Frame.from_records() does not support Series records. Use Frame.from_concat() instead.')
        if isinstance(row_reference, dict):
            raise ErrorInitFrame('Frame.from_records() does not support dictionary records. Use Frame.from_dict_records() instead.')

        is_dataclass = hasattr(row_reference, '__dataclass_fields__')
        if is_dataclass:
            fields_dc = tuple(row_reference.__dataclass_fields__.keys())

        column_name_getter = None
        # NOTE: even if getter is defined, columns list is needed to be available to get_col_dtype after it is populated
        if columns is None and hasattr(row_reference, '_fields'): # NamedTuple
            column_name_getter = row_reference._fields.__getitem__
            columns = []
        elif columns is None and is_dataclass:
            column_name_getter = fields_dc.__getitem__
            columns = []

        get_col_dtype = None if dtypes is None else get_col_dtype_factory(dtypes, columns)

        # NOTE: row data by definition does not have Index data, so col count is length of row
        if hasattr(row_reference, '__len__'):
            col_count = len(row_reference)
        elif is_dataclass:
            col_count = len(fields_dc) # defined in branch above
        else:
            raise NotImplementedError(f'cannot get col_count from {row_reference}')

        if not is_dataclass:
            def get_value_iter(col_key: tp.Hashable) -> tp.Iterator[tp.Any]:
                rows_iter = rows if not rows_to_iter else iter(rows)
                return (row[col_key] for row in rows_iter)
        else:
            def get_value_iter(col_key: tp.Hashable) -> tp.Iterator[tp.Any]:
                rows_iter = rows if not rows_to_iter else iter(rows)
                return (getattr(row, fields_dc[col_key]) for row in rows_iter)

        def blocks() -> tp.Iterator[np.ndarray]:
            # iterate over final column order, yielding 1D arrays
            for col_idx in range(col_count):
                if column_name_getter: # append as side effect of generator!
                    columns.append(column_name_getter(col_idx))
                values = array_from_value_iter(
                        key=col_idx,
                        idx=col_idx, # integer used
                        get_value_iter=get_value_iter,
                        get_col_dtype=get_col_dtype,
                        row_count=row_count
                        )
                yield values

        if consolidate_blocks:
            block_gen = lambda: TypeBlocks.consolidate_blocks(blocks())
        else:
            block_gen = blocks

        return cls(TypeBlocks.from_blocks(block_gen()),
                index=index,
                columns=columns,
                name=name,
                own_data=True,
                index_constructor=index_constructor,
                columns_constructor=columns_constructor,
                own_index=own_index,
                own_columns=own_columns,
                )

    @classmethod
    @doc_inject(selector='constructor_frame')
    def from_dict_records(cls,
            records: tp.Iterable[tp.Dict[tp.Hashable, tp.Any]],
            *,
            index: tp.Optional[IndexInitializer] = None,
            dtypes: DtypesSpecifier = None,
            name: tp.Hashable = None,
            fill_value: object = np.nan,
            consolidate_blocks: bool = False,
            index_constructor: IndexConstructor = None,
            columns_constructor: IndexConstructor = None,
            own_index: bool = False,
            ) -> 'Frame':
        '''Frame constructor from an iterable of dictionaries, where each dictionary represents a row; column names will be derived from the union of all row dictionary keys.

        Args:
            records: Iterable of row values, where row values are dictionaries.
            index: Optionally provide an iterable of index labels, equal in length to the number of records. If a generator, this value will not be evaluated until after records are loaded.
            {dtypes}
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`static_frame.Frame`
        '''
        columns = []

        get_col_dtype = None if dtypes is None else get_col_dtype_factory(dtypes, columns)

        if not hasattr(records, '__len__'):
            # might be a generator; must convert to sequence
            rows = list(records)
        else: # could be a sequence, or something like a dict view
            rows = records
        row_count = len(rows)

        if not row_count:
            raise ErrorInitFrame('no rows available in records.')

        if hasattr(rows, '__getitem__'):
            rows_to_iter = False
        else: # dict view, or other sized iterable that does not support getitem
            rows_to_iter = True

        row_reference = {}
        for row in rows: # produce a row that has a value for all observed keys
            row_reference.update(row)

        col_count = len(row_reference)

        # define function to get generator of row values; may need to call twice, so need to get fresh row_iter each time
        def get_value_iter(col_key: tp.Hashable) -> tp.Iterator[tp.Any]:
            rows_iter = rows if not rows_to_iter else iter(rows)
            return (row.get(col_key, fill_value) for row in rows_iter)

        def blocks() -> tp.Iterator[np.ndarray]:
            # iterate over final column order, yielding 1D arrays
            for col_idx, col_key in enumerate(row_reference.keys()):
                columns.append(col_key)

                values = array_from_value_iter(
                        key=col_key,
                        idx=col_idx,
                        get_value_iter=get_value_iter,
                        get_col_dtype=get_col_dtype,
                        row_count=row_count
                        )
                yield values

        if consolidate_blocks:
            block_gen = lambda: TypeBlocks.consolidate_blocks(blocks())
        else:
            block_gen = blocks

        return cls(TypeBlocks.from_blocks(block_gen()),
                index=index,
                columns=columns,
                name=name,
                own_data=True,
                index_constructor=index_constructor,
                columns_constructor=columns_constructor,
                own_index=own_index,
                )

    @classmethod
    @doc_inject(selector='constructor_frame')
    def from_records_items(cls,
            items: tp.Iterator[tp.Tuple[tp.Hashable, tp.Iterable[tp.Any]]],
            *,
            columns: tp.Optional[IndexInitializer] = None,
            dtypes: DtypesSpecifier = None,
            name: tp.Hashable = None,
            consolidate_blocks: bool = False,
            index_constructor: IndexConstructor = None,
            columns_constructor: IndexConstructor = None,
            own_columns: bool = False,
            ) -> 'Frame':
        '''Frame constructor from iterable of pairs of index value, row (where row is an iterable).

        Args:
            items: Iterable of pairs of index label, row values, where row values are arrays, tuples, lists, dictionaries, or namedtuples.
            columns: Optionally provide an iterable of column labels, equal in length to the length of each row.
            {dtypes}
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`static_frame.Frame`

        '''
        index = []

        def gen() -> tp.Iterator[np.ndarray]:
            for label, values in items:
                index.append(label)
                yield values

        return cls.from_records(gen(),
                index=index,
                columns=columns,
                dtypes=dtypes,
                name=name,
                consolidate_blocks=consolidate_blocks,
                index_constructor=index_constructor,
                columns_constructor=columns_constructor,
                own_columns=own_columns,
                )

    @classmethod
    @doc_inject(selector='constructor_frame')
    def from_dict_records_items(cls,
            items: tp.Iterator[tp.Tuple[tp.Hashable, tp.Iterable[tp.Any]]],
            *,
            dtypes: DtypesSpecifier = None,
            name: tp.Hashable = None,
            consolidate_blocks: bool = False) -> 'Frame':
        '''Frame constructor from iterable of pairs of index label, row, where row is a dictionary. Column names will be derived from the union of all row dictionary keys.

        Args:
            items: Iterable of pairs of index label, row values, where row values are arrays, tuples, lists, dictionaries, or namedtuples.
            {dtypes}
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`static_frame.Frame`

        '''
        index = []

        def gen() -> tp.Iterator[np.ndarray]:
            for label, values in items:
                index.append(label)
                yield values

        return cls.from_dict_records(gen(),
                index=index,
                dtypes=dtypes,
                name=name,
                consolidate_blocks=consolidate_blocks
                )

    @classmethod
    @doc_inject(selector='constructor_frame')
    def from_items(cls,
            pairs: tp.Iterable[tp.Tuple[tp.Hashable, tp.Iterable[tp.Any]]],
            *,
            index: IndexInitializer = None,
            fill_value: object = np.nan,
            dtypes: DtypesSpecifier = None,
            name: tp.Hashable = None,
            index_constructor: IndexConstructor = None,
            columns_constructor: IndexConstructor = None,
            consolidate_blocks: bool = False
            ) -> 'Frame':
        '''Frame constructor from an iterator of pairs, where the first value is the column label and the second value is an iterable of column values. :obj:`Series` can be provided as values if an ``index`` argument is supplied.

        Args:
            pairs: Iterable of pairs of column name, column values.
            index: Iterable of values to create an Index.
            fill_value: If pairs include Series, they will be reindexed with the provided index; reindexing will use this fill value.
            {dtypes}
            {name}
            index_constructor:
            columns_constructor:
            {consolidate_blocks}

        Returns:
            :obj:`Frame`
        '''
        columns = []

        # if an index initializer is passed, and we expect to get Series, we need to create the index in advance of iterating blocks
        own_index = False
        if _index_initializer_needs_init(index):
            index = index_from_optional_constructor(index,
                    default_constructor=Index,
                    explicit_constructor=index_constructor
                    )
            own_index = True

        get_col_dtype = None if dtypes is None else get_col_dtype_factory(dtypes, columns)

        def blocks() -> tp.Iterator[np.ndarray]:
            for col_idx, (k, v) in enumerate(pairs):
                columns.append(k) # side effect of generator!
                column_type = None if get_col_dtype is None else get_col_dtype(col_idx) #pylint: disable=E1102

                if v.__class__ is np.ndarray:
                    # NOTE: we rely on TypeBlocks constructor to check that these are same sized
                    if column_type is not None:
                        yield v.astype(column_type)
                    else:
                        yield v
                elif isinstance(v, Series):
                    if index is None:
                        raise ErrorInitFrame('can only consume Series in Frame.from_items if an Index is provided.')

                    if not v.index.equals(index):
                        v = v.reindex(index,
                                fill_value=fill_value,
                                check_equals=False,
                                )
                    # return values array post reindexing
                    if column_type is not None:
                        yield v.values.astype(column_type)
                    else:
                        yield v.values

                elif isinstance(v, Frame):
                    raise ErrorInitFrame('Frames are not supported in from_items constructor.')
                else:
                    # returned array is immutable
                    values, _ = iterable_to_array_1d(v, column_type)
                    yield values

        if consolidate_blocks:
            block_gen = lambda: TypeBlocks.consolidate_blocks(blocks())
        else:
            block_gen = blocks

        return cls(TypeBlocks.from_blocks(block_gen()),
                index=index,
                columns=columns,
                name=name,
                own_data=True,
                own_index=own_index,
                columns_constructor=columns_constructor
                )

    @classmethod
    @doc_inject(selector='constructor_frame')
    def from_dict(cls,
            mapping: tp.Dict[tp.Hashable, tp.Iterable[tp.Any]],
            *,
            index: tp.Optional[IndexInitializer] = None,
            fill_value: object = np.nan,
            dtypes: DtypesSpecifier = None,
            name: tp.Hashable = None,
            index_constructor: IndexConstructor = None,
            columns_constructor: IndexConstructor = None,
            consolidate_blocks: bool = False
            ) -> 'Frame':
        '''
        Create a Frame from a dictionary (or any object that has an items() method) where keys are column labels and values are columns values (either sequence types or :obj:`Series`).

        Args:
            mapping: a dictionary or similar mapping interface.
            index:
            fill_value:
            {dtypes}
            {name}
            index_constructor:
            columns_constructor:
            {consolidate_blocks}
        '''
        return cls.from_items(mapping.items(),
                index=index,
                fill_value=fill_value,
                name=name,
                dtypes=dtypes,
                index_constructor=index_constructor,
                columns_constructor=columns_constructor,
                consolidate_blocks=consolidate_blocks,
                )


    @classmethod
    @doc_inject(selector='constructor_frame')
    def from_fields(cls,
            fields: tp.Iterable[tp.Iterable[tp.Any]],
            *,
            index: tp.Optional[IndexInitializer] = None,
            columns: tp.Optional[IndexInitializer] = None,
            fill_value: object = np.nan,
            dtypes: DtypesSpecifier = None,
            name: tp.Hashable = None,
            index_constructor: IndexConstructor = None,
            columns_constructor: IndexConstructor = None,
            own_index: bool = False,
            own_columns: bool = False,
            consolidate_blocks: bool = False
            ) -> 'Frame':
        '''Frame constructor from an iterator of columns, where columns are iterables. :obj:`Series` can be provided as values if an ``index`` argument is supplied.

        Args:
            fields: Iterable of column values.
            index: Iterable of values to create an Index.
            fill_value: If pairs include Series, they will be reindexed with the provided index; reindexing will use this fill value.
            {dtypes}
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`static_frame.Frame`
        '''
        # if an index initializer is passed, and we expect to get Series, we need to create the index in advance of iterating blocks
        if not own_index and _index_initializer_needs_init(index):
            index = index_from_optional_constructor(index,
                    default_constructor=Index,
                    explicit_constructor=index_constructor
                    )
            own_index = True

        get_col_dtype = None if dtypes is None else get_col_dtype_factory(dtypes, columns)

        def blocks() -> tp.Iterator[np.ndarray]:
            for col_idx, v in enumerate(fields):
                column_type = None if get_col_dtype is None else get_col_dtype(col_idx) #pylint: disable=E1102

                if v.__class__ is np.ndarray:
                    if column_type is not None:
                        yield v.astype(column_type)
                    else:
                        yield v
                elif isinstance(v, Series):
                    if index is None:
                        raise ErrorInitFrame('can only consume Series in Frame.from_fields if an Index is provided.')
                    if not v.index.equals(index):
                        v = v.reindex(index,
                                fill_value=fill_value,
                                check_equals=False,
                                )
                    if column_type is not None:
                        yield v.values.astype(column_type)
                    else:
                        yield v.values
                elif isinstance(v, Frame):
                    raise ErrorInitFrame('Frames are not supported in from_fields constructor.')
                else: # returned array is immutable
                    values, _ = iterable_to_array_1d(v, column_type)
                    yield values

        if consolidate_blocks:
            block_gen = lambda: TypeBlocks.consolidate_blocks(blocks())
        else:
            block_gen = blocks

        return cls(TypeBlocks.from_blocks(block_gen()),
                index=index,
                columns=columns,
                name=name,
                own_data=True,
                own_index=own_index,
                own_columns=own_columns,
                columns_constructor=columns_constructor
                )





    @staticmethod
    def _structured_array_to_d_ia_cl(
            array: np.ndarray,
            *,
            index_depth: int = 0,
            index_column_first: tp.Optional[IndexSpecifier] = None,
            dtypes: DtypesSpecifier = None,
            consolidate_blocks: bool = False,
            store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT,
            columns: tp.Optional[IndexBase] = None
            ) -> tp.Tuple[TypeBlocks, tp.Sequence[np.ndarray], tp.Sequence[tp.Hashable]]:
        '''
        Expanded function name: _structure_array_to_data_index_arrays_columns_labels

        Utility function for creating TypeBlocks from structure array (or a 2D array that np.genfromtxt might have returned) while extracting index and columns labels. Does not form Index objects for columns or index, allowing down-stream processes to do so.

        Args:
            index_column_first: optionally name the column that will start the block of index columns.
            columns: optionally provide a columns Index to resolve dtypes specified by name.
        '''
        names = array.dtype.names # using names instead of fields, as this is NP convention
        is_structured_array = True
        if names is None:
            is_structured_array = False
            # raise ErrorInitFrame('array is not a structured array')
            # could use np.rec.fromarrays, but that makes a copy; better to use the passed in array
            # must be a 2D array
            names = tuple(range(array.shape[1]))

        index_start_pos = -1 # will be ignored
        index_end_pos = -1
        if index_column_first is not None:
            if index_depth <= 0:
                raise ErrorInitFrame('index_column_first specified but index_depth is 0')
            elif isinstance(index_column_first, INT_TYPES):
                index_start_pos = index_column_first
            else:
                index_start_pos = names.index(index_column_first) # linear performance
            index_end_pos = index_start_pos + index_depth - 1
        else: # no index_column_first specified, if index depth > 0, set start to 0
            if index_depth > 0:
                index_start_pos = 0
                # Subtract one for inclusive boun
                index_end_pos = index_start_pos + index_depth - 1

        # assign in generator
        index_arrays = []
        # collect whatever labels are found on structured arrays; these may not be the same as the passed in columns, if columns are provided
        columns_labels = []

        index_field_placeholder = object()
        columns_by_col_idx = []

        if columns is None:
            use_dtype_names = True
        else:
            use_dtype_names = False
            columns_idx = 0 # relative position in index object
            # construct columns_by_col_idx from columns, adding sentinal for index columns; this means we cannot get map dtypes from index names
            for i in range(len(names)):
                if i >= index_start_pos and i <= index_end_pos:
                    columns_by_col_idx.append(index_field_placeholder)
                    continue
                columns_by_col_idx.append(columns[columns_idx])
                columns_idx += 1

        get_col_dtype = None if dtypes is None else get_col_dtype_factory(
                dtypes,
                columns_by_col_idx)

        def blocks() -> tp.Iterator[np.ndarray]:
            # iterate over column names and yield one at a time for block construction; collect index arrays and column labels as we go
            for col_idx, name in enumerate(names):
                if use_dtype_names:
                    # append here as we iterate for usage in get_col_dtype
                    columns_by_col_idx.append(name)

                # this is not expected to make a copy
                if is_structured_array:
                    array_final = array[name]
                else: # a 2D array, name is integer for column
                    array_final = array[NULL_SLICE, name]

                # do StoreFilter conversions before dtype
                if array_final.ndim == 0:
                    # some structured arrays give 0 ndim arrays by name
                    array_final = np.reshape(array_final, (1,))

                if store_filter is not None:
                    array_final = store_filter.to_type_filter_array(array_final)

                if get_col_dtype:
                    # dtypes are applied to all columns and can refer to columns that will become part of the Index by name or iloc position: we need to be able to type these before creating Index obejcts
                    dtype = get_col_dtype(col_idx) #pylint: disable=E1102
                    if dtype is not None:
                        array_final = array_final.astype(dtype)

                array_final.flags.writeable = False

                if col_idx >= index_start_pos and col_idx <= index_end_pos:
                    index_arrays.append(array_final)
                    continue

                columns_labels.append(name)
                yield array_final

        if consolidate_blocks:
            data = TypeBlocks.from_blocks(TypeBlocks.consolidate_blocks(blocks()))
        else:
            data = TypeBlocks.from_blocks(blocks())

        return data, index_arrays, columns_labels

    @classmethod
    def _from_data_index_arrays_column_labels(cls,
            data: TypeBlocks,
            index_depth: int,
            index_arrays: tp.Sequence[np.ndarray],
            columns_depth: int,
            columns_labels: tp.Sequence[tp.Hashable],
            name: tp.Hashable,
            ) -> 'Frame':
        '''
        Private constructor used for specialized construction from NP Structured array, as well as StoreHDF5.
        '''
        columns_constructor = None
        if columns_depth == 0:
            columns = None
        elif columns_depth == 1:
            columns = columns_labels
        elif columns_depth > 1:
            # assume deliminted IH extracted from SA labels
            columns = columns_labels
            columns_constructor = partial(
                    cls._COLUMNS_HIERARCHY_CONSTRUCTOR.from_labels_delimited,
                    delimiter=' ')

        kwargs = dict(
                data=data,
                own_data=True,
                columns=columns,
                columns_constructor=columns_constructor,
                name=name
                )

        if index_depth == 0:
            return cls(
                index=None,
                **kwargs)
        if index_depth == 1:
            return cls(
                index=index_arrays[0],
                **kwargs)
        return cls(
                index=zip(*index_arrays),
                index_constructor=IndexHierarchy.from_labels,
                **kwargs
                )

    @classmethod
    @doc_inject(selector='constructor_frame')
    def from_structured_array(cls,
            array: np.ndarray,
            *,
            index_depth: int = 0,
            index_column_first: tp.Optional[IndexSpecifier] = None,
            columns_depth: int = 1,
            dtypes: DtypesSpecifier = None,
            name: tp.Hashable = None,
            consolidate_blocks: bool = False,
            store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ) -> 'Frame':
        '''
        Convert a NumPy structed array into a Frame.

        Args:
            array: Structured NumPy array.
            index_depth: Depth if index levels, where (for example) 0 is no index, 1 is a single column index, and 2 is a two-columns IndexHierarchy.
            index_column_first: Optionally provide the name or position offset of the column to use as the index.
            {dtypes}
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`static_frame.Frame`
        '''
        # from a structured array, we assume we want to get the columns labels
        data, index_arrays, columns_labels = cls._structured_array_to_d_ia_cl(
                array=array,
                index_depth=index_depth,
                index_column_first=index_column_first,
                dtypes=dtypes,
                consolidate_blocks=consolidate_blocks,
                store_filter=store_filter,
                )
        return cls._from_data_index_arrays_column_labels(
                data=data,
                index_depth=index_depth,
                index_arrays=index_arrays,
                columns_depth=columns_depth,
                columns_labels=columns_labels,
                name=name
                )

    #---------------------------------------------------------------------------
    @classmethod
    def from_element_items(cls,
            items: tp.Iterable[tp.Tuple[
                    tp.Tuple[tp.Hashable, tp.Hashable], tp.Any]],
            *,
            index: IndexInitializer,
            columns: IndexInitializer,
            dtype: DtypesSpecifier = None,
            axis: tp.Optional[int] = None,
            name: NameType = None,
            fill_value: object = FILL_VALUE_DEFAULT,
            index_constructor: IndexConstructor = None,
            columns_constructor: IndexConstructor = None,
            own_index: bool = False,
            own_columns: bool = False,
            ) -> 'Frame':
        '''
        Create a :obj:`Frame` from an iterable of key, value, where key is a pair of row, column labels.

        This function is partialed (setting the index and columns) and used by ``IterNodeDelegate`` as the apply constructor for doing application on element iteration.

        Args:
            items: an iterable of pairs of 2-tuples of row, column loc labels and values.
            axis: when None, items can be in an order; when 0, items must be well-formed and ordered row major; when 1, items must be well-formed and ordered columns major.

        Returns:
            :obj:`static_frame.Frame`
        '''
        if not own_index:
            index = index_from_optional_constructor(index,
                    default_constructor=Index,
                    explicit_constructor=index_constructor
                    )
            own_index = True

        if not own_columns:
            columns = index_from_optional_constructor(columns,
                    default_constructor=cls._COLUMNS_CONSTRUCTOR,
                    explicit_constructor=columns_constructor
                    )
            own_columns = True

        # if items are given i n
        if axis is None:
            if not is_dtype_specifier(dtype):
                raise ErrorInitFrame('cannot provide multiple dtypes when creating a Frame from element items and axis is None')
            items = (((index._loc_to_iloc(k[0]), columns._loc_to_iloc(k[1])), v)
                    for k, v in items)
            dtype = dtype if dtype is not None else DTYPE_OBJECT
            tb = TypeBlocks.from_element_items(
                    items,
                    shape=(len(index), len(columns)),
                    dtype=dtype,
                    fill_value=fill_value)
            return cls(tb,
                    index=index,
                    columns=columns,
                    name=name,
                    own_data=True,
                    own_index=own_index, # always true as either provided or created new
                    own_columns=own_columns,
                    )

        elif axis == 0: # row wise, use from-records
            def records() -> tp.Iterator[tp.List[tp.Any]]:
                # do not need to convert loc to iloc
                items_iter = iter(items)
                first = next(items_iter)
                (r_last, _), value = first
                values = [value]
                for (r, c), v in items_iter:
                    if r != r_last:
                        yield values
                        r_last = r
                        values = []
                    values.append(v)
                yield values

            return cls.from_records(records(),
                    index=index,
                    columns=columns,
                    name=name,
                    own_index=own_index,
                    own_columns=own_columns,
                    dtypes=dtype,
                    )

        elif axis == 1: # column wise, use from_items
            def fields() -> tp.Iterator[tp.Tuple[tp.Hashable, tp.List[tp.Any]]]:
                items_iter = iter(items)
                first = next(items_iter)
                (_, c_last), value = first
                values = [value]
                for (r, c), v in items_iter:
                    if c != c_last:
                        yield values
                        c_last = c
                        values = []
                    values.append(v)
                yield values

            return cls.from_fields(fields(),
                    index=index,
                    columns=columns,
                    name=name,
                    own_index=own_index,
                    own_columns=own_columns,
                    dtypes=dtype,
                    )
        raise AxisInvalid(f'no support for axis: {axis}')

    #---------------------------------------------------------------------------
    # file, data format loaders

    @classmethod
    @doc_inject(selector='constructor_frame')
    def from_sql(cls,
            query: str,
            *,
            connection: sqlite3.Connection,
            index_depth: int = 0,
            columns_depth: int = 1,
            columns_select: tp.Optional[tp.Iterable[str]] = None,
            dtypes: DtypesSpecifier = None,
            name: tp.Hashable = None,
            consolidate_blocks: bool = False,
            parameters: tp.Iterable[tp.Any] = (),
            ) -> 'Frame':
        '''
        Frame constructor from an SQL query and a database connection object.

        Args:
            query: A query string.
            connection: A DBAPI2 (PEP 249) Connection object, such as those returned from SQLite (via the sqlite3 module) or PyODBC.
            {dtypes}
            columns_select: An optional iterable of field names to extract from the results of the query.
            {name}
            {consolidate_blocks}
            parameters: Provide a list of values for an SQL query expecting parameter substitution.
        '''
        columns = None
        own_columns = False

        # We cannot assume the cursor object returned by DBAPI Connection to have a context manager, thus all cursor usage needs to be wrapped in a try/finally to insure that the cursor is closed.
        cursor = None
        try:
            cursor = connection.cursor()
            cursor.execute(query, parameters)

            if columns_select:
                columns_select = set(columns_select)
                # selector function defined below
                def filter_row(row: tp.Sequence[tp.Any]) -> tp.Sequence[tp.Any]:
                    post = selector(row)
                    return post if not selector_reduces else (post,)

            if columns_depth >= 1 or columns_select:
                # always need to derive labels if using columns_select
                labels = (col for (col, *_) in cursor.description[index_depth:])

            if columns_depth <= 1 and columns_select:
                iloc_sel, labels = zip(*(
                        pair for pair in enumerate(labels) if pair[1] in columns_select
                        ))
                selector = itemgetter(*iloc_sel)
                selector_reduces = len(iloc_sel) == 1

            if columns_depth == 1:
                columns = cls._COLUMNS_CONSTRUCTOR(labels)
                own_columns = True
            elif columns_depth > 1:
                # NOTE: we only support loading in IH if encoded in each header with a space delimiter
                constructor = cls._COLUMNS_HIERARCHY_CONSTRUCTOR.from_labels_delimited
                columns = constructor(labels, delimiter=' ')
                own_columns = True

                if columns_select:
                    iloc_sel = columns._loc_to_iloc(columns.isin(columns_select))
                    selector = itemgetter(*iloc_sel)
                    selector_reduces = len(iloc_sel) == 1
                    columns = columns.iloc[iloc_sel]

            index_constructor = None
            if index_depth > 0:
                # map dtypes in context of pre-index extraction
                get_col_dtype = None if dtypes is None else get_col_dtype_factory(
                        dtypes,
                        [col for (col, *_) in cursor.description],
                        )
                if index_depth == 1:
                    index = [] # lazily populate
                    if get_col_dtype:
                        index_constructor = partial(Index, dtype=get_col_dtype(0))
                    else:
                        index_constructor = Index
                    def row_gen() -> tp.Iterator[tp.Sequence[tp.Any]]:
                        for row in cursor:
                            index.append(row[0])
                            yield row[1:]
                else: # > 1
                    index = [list() for _ in range(index_depth)]
                    def index_constructor(iterables) -> IndexHierarchy: #pylint: disable=function-redefined
                        if get_col_dtype:
                            blocks = [iterable_to_array_1d(it, get_col_dtype(i))[0]
                                    for i, it in enumerate(iterables)]
                        else:
                            blocks = [iterable_to_array_1d(it)[0] for it in iterables]
                        return IndexHierarchy._from_type_blocks(
                                TypeBlocks.from_blocks(blocks),
                                own_blocks=True)

                    def row_gen() -> tp.Iterator[tp.Sequence[tp.Any]]:
                        for row in cursor:
                            for i, label in enumerate(row[:index_depth]):
                                index[i].append(label)
                            yield row[index_depth:]
            else:
                index = None
                row_gen = lambda: cursor

            if columns_select:
                row_gen_final = (filter_row(row) for row in row_gen())
            else:
                row_gen_final = row_gen()

            return cls.from_records(
                    row_gen_final,
                    columns=columns,
                    index=index,
                    dtypes=dtypes,
                    name=name,
                    own_columns=own_columns,
                    index_constructor=index_constructor,
                    consolidate_blocks=consolidate_blocks,
                    )
        finally:
            if cursor:
                cursor.close()

    @classmethod
    @doc_inject(selector='constructor_frame')
    def from_json(cls,
            json_data: str,
            *,
            dtypes: DtypesSpecifier = None,
            name: tp.Hashable = None,
            consolidate_blocks: bool = False
            ) -> 'Frame':
        '''Frame constructor from an in-memory JSON document.

        Args:
            json_data: a string of JSON, encoding a table as an array of JSON objects.
            {dtypes}
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`static_frame.Frame`
        '''
        data = json.loads(json_data)
        return cls.from_dict_records(data,
                name=name,
                dtypes=dtypes,
                consolidate_blocks=consolidate_blocks
                )

    @classmethod
    @doc_inject(selector='constructor_frame')
    def from_json_url(cls,
            url: str,
            *,
            dtypes: DtypesSpecifier = None,
            name: tp.Hashable = None,
            consolidate_blocks: bool = False
            ) -> 'Frame':
        '''Frame constructor from a JSON documenst provided via a URL.

        Args:
            url: URL to the JSON resource.
            {dtypes}
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`static_frame.Frame`
        '''
        return cls.from_json(_read_url(url), #pragma: no cover
                name=name,
                dtypes=dtypes,
                consolidate_blocks=consolidate_blocks
                )

    @classmethod
    @doc_inject(selector='constructor_frame')
    def from_delimited(cls,
            fp: PathSpecifierOrFileLikeOrIterator,
            *,
            delimiter: str,
            index_depth: int = 0,
            index_column_first: tp.Optional[tp.Union[int, str]] = None,
            index_name_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            index_continuation_token: tp.Union[tp.Hashable, None] = CONTINUATION_TOKEN_INACTIVE,
            columns_depth: int = 1,
            columns_name_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            columns_continuation_token: tp.Union[tp.Hashable, None] = CONTINUATION_TOKEN_INACTIVE,
            skip_header: int = 0,
            skip_footer: int = 0,
            quote_char: str = '"',
            encoding: tp.Optional[str] = None,
            dtypes: DtypesSpecifier = None,
            name: tp.Hashable = None,
            consolidate_blocks: bool = False,
            store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ) -> 'Frame':
        '''
        Create a Frame from a file path or a file-like object defining a delimited (CSV, TSV) data file.

        Args:
            fp: A file path or a file-like object.
            delimiter: The character used to seperate row elements.
            index_depth: Specify the number of columns used to create the index labels; a value greater than 1 will attempt to create a hierarchical index.
            index_column_first: Optionally specify a column, by position or name, to become the start of the index if index_depth is greater than 0. If not set and index_depth is greater than 0, the first column will be used.
            index_name_depth_level: If columns_depth is greater than 0, interpret values over index as the index name.
            columns_depth: Specify the number of rows after the skip_header used to create the column labels. A value of 0 will be no header; a value greater than 1 will attempt to create a hierarchical index.
            columns_name_depth_level: If index_depth is greater than 0, interpret values over index as the columns name.
            skip_header: Number of leading lines to skip.
            skip_footer: Number of trailing lines to skip.
            store_filter: A StoreFilter instance, defining translation between unrepresentable types. Presently nly the ``to_nan`` attributes is used.
            {dtypes}
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`static_frame.Frame`
        '''
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html

        # TODO: add columns_select as usecols styles selective loading

        if skip_header < 0:
            raise ErrorInitFrame('skip_header must be greater than or equal to 0')

        fp = path_filter(fp)
        delimiter_native = '\t'

        if delimiter != delimiter_native:
            # this is necessary if there are quoted cells that include the delimiter
            def file_like() -> tp.Iterator[str]:
                if isinstance(fp, str):
                    with open(fp, 'r') as f:
                        for row in csv.reader(f, delimiter=delimiter, quotechar=quote_char):
                            yield delimiter_native.join(row)
                else: # handling file like object works for stringio but not for bytesio
                    for row in csv.reader(fp, delimiter=delimiter, quotechar=quote_char):
                        yield delimiter_native.join(row)
        else:
            def file_like() -> tp.Iterator[str]: # = fp
                if isinstance(fp, str):
                    with open(fp, 'r') as f:
                        for row in f:
                            yield row
                else: # iterable of string lines, StringIO
                    for row in fp:
                        yield row

        # always accumulate columns rows, as np.genfromtxt will mutate the headers: adding enderscore, removing invalid characters, etc.
        apex_rows = []
        columns_rows = []

        def row_source() -> tp.Iterator[str]:
            # set equal to skip header unless column depth is > 1
            column_max = skip_header + columns_depth
            for i, row in enumerate(file_like()):
                if i < skip_header:
                    continue
                if i < column_max:
                    columns_rows.append(row)
                    continue
                yield row

        # genfromtxt takes missing_values, but this can only be a list, and does not work under some condition (i.e., a cell with no value). thus, this is deferred to from_sructured_array

        with warnings.catch_warnings():
            # silence: UserWarning: genfromtxt: Empty input file
            warnings.simplefilter('ignore', UserWarning)

            array = np.genfromtxt(
                    row_source(),
                    delimiter=delimiter_native,
                    skip_header=0, # done in row_source
                    skip_footer=skip_footer,
                    comments=None,
                    # strange NP convention for this parameter: False is not supported, must use None to not parase headers
                    names= None,
                    dtype=None,
                    encoding=encoding,
                    invalid_raise=False,
                    )
        array.flags.writeable = False

        # construct columns prior to preparing data from structured array, as need columns to map dtypes
        # columns_constructor = None
        if columns_depth == 0:
            columns = None
            own_columns = False
        else:
            # Process each row one at a time, as types align by row.
            columns_arrays = []
            for row in columns_rows:
                columns_array = np.genfromtxt(
                        (row,),
                        delimiter=delimiter_native,
                        comments=None,
                        names=None,
                        dtype=None,
                        encoding=encoding,
                        invalid_raise=False,
                        )
                # the array might be ndim=1, or ndim=0; must get a list before slicing
                # using the array directly for a string type might not hold the rights size after slicing
                columns_list = columns_array.tolist()
                apex_rows.append(columns_list[:index_depth])
                columns_arrays.append(columns_list[index_depth:])

            columns_name = None if index_depth == 0 else apex_to_name(
                    rows=apex_rows,
                    depth_level=columns_name_depth_level,
                    axis=1,
                    axis_depth=columns_depth)

            if columns_depth == 1:
                columns_constructor = cls._COLUMNS_CONSTRUCTOR
                columns = columns_constructor(columns_arrays[0], name=columns_name)
            else:
                columns_constructor = cls._COLUMNS_HIERARCHY_CONSTRUCTOR.from_labels
                if columns_continuation_token:
                    labels = zip_longest(
                            *(store_filter.to_type_filter_iterable(x) for x in columns_arrays),
                            fillvalue=columns_continuation_token,
                            )
                else:
                    labels = zip(*(store_filter.to_type_filter_iterable(x) for x in columns_arrays))
                columns = columns_constructor(
                        labels,
                        name=columns_name,
                        continuation_token=columns_continuation_token,
                        )
            own_columns = True

        if array.dtype.names is None: # not a structured array
            # genfromtxt may, in some situations, not return a structured array
            if array.ndim == 1:
                # got a single row
                array = array.reshape((1, len(array)))
            # NOTE: genfromtxt will return a one column input file as a 2D array with the vertical data as a horizontal row. There does not appear to be a way to distinguish this from a single row file

        if array.size > 0: # an empty, or column only table
            data, index_arrays, _ = cls._structured_array_to_d_ia_cl(
                    array=array,
                    index_depth=index_depth,
                    index_column_first=index_column_first,
                    dtypes=dtypes,
                    consolidate_blocks=consolidate_blocks,
                    store_filter=store_filter,
                    columns=columns
                    )
        else: # only column data in table
            if index_depth > 0:
                # no data is found an index depth was given; simulate empty index_arrays to create a empty index
                index_arrays = [EMPTY_TUPLE] * index_depth
            data = FRAME_INITIALIZER_DEFAULT

        kwargs = dict(
                data=data,
                own_data=True,
                columns=columns,
                own_columns=own_columns,
                name=name
                )

        if index_depth == 0:
            return cls(index=None, **kwargs)

        index_name = None if columns_depth == 0 else apex_to_name(
                rows=apex_rows,
                depth_level=index_name_depth_level,
                axis=0,
                axis_depth=index_depth)

        if index_depth == 1:
            index_constructor = partial(Index, name=index_name)
            return cls(
                index=index_arrays[0],
                index_constructor=index_constructor,
                **kwargs)

        index_constructor = partial(IndexHierarchy.from_labels,
                name=index_name,
                continuation_token=index_continuation_token,
                )
        return cls(
                index=zip(*index_arrays),
                index_constructor=index_constructor,
                **kwargs
                )

    @classmethod
    def from_csv(cls,
            fp: PathSpecifierOrFileLikeOrIterator,
            *,
            index_depth: int = 0,
            index_column_first: tp.Optional[tp.Union[int, str]] = None,
            index_name_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            index_continuation_token: tp.Union[tp.Hashable, None] = CONTINUATION_TOKEN_INACTIVE,
            columns_depth: int = 1,
            columns_name_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            columns_continuation_token: tp.Union[tp.Hashable, None] = CONTINUATION_TOKEN_INACTIVE,
            skip_header: int = 0,
            skip_footer: int = 0,
            quote_char: str = '"',
            encoding: tp.Optional[str] = None,
            dtypes: DtypesSpecifier = None,
            name: tp.Hashable = None,
            consolidate_blocks: bool = False,
            store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ) -> 'Frame':
        '''
        Specialized version of :obj:`Frame.from_delimited` for CSV files.

        Returns:
            :obj:`Frame`
        '''
        return cls.from_delimited(fp,
                delimiter=',',
                index_depth=index_depth,
                index_column_first=index_column_first,
                index_name_depth_level=index_name_depth_level,
                index_continuation_token=index_continuation_token,
                columns_depth=columns_depth,
                columns_name_depth_level=columns_name_depth_level,
                columns_continuation_token=columns_continuation_token,
                skip_header=skip_header,
                skip_footer=skip_footer,
                quote_char=quote_char,
                encoding=encoding,
                dtypes=dtypes,
                name=name,
                consolidate_blocks=consolidate_blocks,
                store_filter=store_filter,
                )

    @classmethod
    def from_tsv(cls,
            fp: PathSpecifierOrFileLikeOrIterator,
            *,
            index_depth: int = 0,
            index_column_first: tp.Optional[tp.Union[int, str]] = None,
            index_name_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            index_continuation_token: tp.Union[tp.Hashable, None] = CONTINUATION_TOKEN_INACTIVE,
            columns_depth: int = 1,
            columns_name_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            columns_continuation_token: tp.Union[tp.Hashable, None] = CONTINUATION_TOKEN_INACTIVE,
            skip_header: int = 0,
            skip_footer: int = 0,
            quote_char: str = '"',
            encoding: tp.Optional[str] = None,
            dtypes: DtypesSpecifier = None,
            name: tp.Hashable = None,
            consolidate_blocks: bool = False,
            store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ) -> 'Frame':
        '''
        Specialized version of :obj:`Frame.from_delimited` for TSV files.

        Returns:
            :obj:`static_frame.Frame`
        '''
        return cls.from_delimited(fp,
                delimiter='\t',
                index_depth=index_depth,
                index_column_first=index_column_first,
                index_name_depth_level=index_name_depth_level,
                index_continuation_token=index_continuation_token,
                columns_depth=columns_depth,
                columns_name_depth_level=columns_name_depth_level,
                columns_continuation_token=columns_continuation_token,
                skip_header=skip_header,
                skip_footer=skip_footer,
                quote_char=quote_char,
                encoding=encoding,
                dtypes=dtypes,
                name=name,
                consolidate_blocks=consolidate_blocks,
                store_filter=store_filter,
                )

    @classmethod
    def from_clipboard(cls,
            *,
            delimiter: str = '\t',
            index_depth: int = 0,
            index_column_first: tp.Optional[tp.Union[int, str]] = None,
            index_name_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            index_continuation_token: tp.Union[tp.Hashable, None] = CONTINUATION_TOKEN_INACTIVE,
            columns_depth: int = 1,
            columns_name_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            columns_continuation_token: tp.Union[tp.Hashable, None] = CONTINUATION_TOKEN_INACTIVE,
            skip_header: int = 0,
            skip_footer: int = 0,
            quote_char: str = '"',
            encoding: tp.Optional[str] = None,
            dtypes: DtypesSpecifier = None,
            name: NameType = None,
            consolidate_blocks: bool = False,
            store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ) -> 'Frame':
        '''
        Create a :obj:`Frame` from the contents of the clipboard (assuming a table is stored as delimited file).

        Returns:
            :obj:`static_frame.Frame`
        '''
        # HOTE: this uses tk for now, as this is simpler than pyperclip, as used by Pandas
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()

        # using a StringIO might handle platform newline conventions
        sio = StringIO()
        sio.write(root.clipboard_get())
        sio.seek(0)
        return cls.from_delimited(sio,
                delimiter=delimiter,
                index_depth=index_depth,
                index_column_first=index_column_first,
                index_name_depth_level=index_name_depth_level,
                index_continuation_token=index_continuation_token,
                columns_depth=columns_depth,
                columns_name_depth_level=columns_name_depth_level,
                columns_continuation_token=columns_continuation_token,
                skip_header=skip_header,
                skip_footer=skip_footer,
                quote_char=quote_char,
                encoding=encoding,
                dtypes=dtypes,
                name=name,
                consolidate_blocks=consolidate_blocks,
                store_filter=store_filter,
                )

    #---------------------------------------------------------------------------
    # Store-based constructors

    @classmethod
    def from_xlsx(cls,
            fp: PathSpecifier,
            *,
            label: tp.Hashable = STORE_LABEL_DEFAULT,
            index_depth: int = 0,
            index_name_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            columns_depth: int = 1,
            columns_name_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            dtypes: DtypesSpecifier = None,
            consolidate_blocks: bool = False,
            skip_header: int = 0,
            skip_footer: int = 0,
            trim_nadir: bool = False,
            store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT,
            ) -> 'Frame':
        '''
        Load Frame from the contents of a sheet in an XLSX workbook.

        Args:
            label: Optionally provide the sheet name from with to read. If not provided, the first sheet will be used.
        '''
        from static_frame.core.store import StoreConfig
        from static_frame.core.store_xlsx import StoreXLSX

        st = StoreXLSX(fp)
        config = StoreConfig(
                index_depth=index_depth,
                index_name_depth_level=index_name_depth_level,
                columns_depth=columns_depth,
                columns_name_depth_level=columns_name_depth_level,
                dtypes=dtypes,
                consolidate_blocks=consolidate_blocks,
                skip_header=skip_header,
                skip_footer=skip_footer,
                trim_nadir=trim_nadir,
                )
        return st.read(label,
                config=config,
                store_filter=store_filter,
                container_type=cls,
                )

    @classmethod
    def from_sqlite(cls,
            fp: PathSpecifier,
            *,
            label: tp.Hashable = STORE_LABEL_DEFAULT,
            index_depth: int = 0,
            columns_depth: int = 1,
            dtypes: DtypesSpecifier = None,
            consolidate_blocks: bool = False,
            # store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT,
            ) -> 'Frame':
        '''
        Load Frame from the contents of a table in an SQLite database file.
        '''
        from static_frame.core.store import StoreConfig
        from static_frame.core.store_sqlite import StoreSQLite

        st = StoreSQLite(fp)
        config = StoreConfig(
                index_depth=index_depth,
                columns_depth=columns_depth,
                dtypes=dtypes,
                consolidate_blocks=consolidate_blocks,
                )
        return st.read(label,
                config=config,
                container_type=cls,
                # store_filter=store_filter,
                )

    @classmethod
    def from_hdf5(cls,
            fp: PathSpecifier,
            *,
            label: tp.Hashable = STORE_LABEL_DEFAULT,
            index_depth: int = 0,
            columns_depth: int = 1,
            consolidate_blocks: bool = False,
            # store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT,
            ) -> 'Frame':
        '''
        Load Frame from the contents of a table in an HDF5 file.
        '''
        from static_frame.core.store import StoreConfig
        from static_frame.core.store_hdf5 import StoreHDF5

        st = StoreHDF5(fp)
        config = StoreConfig(
                index_depth=index_depth,
                columns_depth=columns_depth,
                consolidate_blocks=consolidate_blocks,
                )
        return st.read(label,
                config=config,
                container_type=cls,
                # store_filter=store_filter,
                )

    #---------------------------------------------------------------------------

    @classmethod
    @doc_inject()
    def from_pandas(cls,
            value: 'pandas.DataFrame',
            *,
            index_constructor: IndexConstructor = None,
            columns_constructor: IndexConstructor = None,
            name: NameType = NAME_DEFAULT,
            consolidate_blocks: bool = False,
            own_data: bool = False
            ) -> 'Frame':
        '''Given a Pandas DataFrame, return a Frame.

        Args:
            value: Pandas DataFrame.
            {index_constructor}
            {columns_constructor}
            {consolidate_blocks}
            {own_data}

        Returns:
            :obj:`Frame`
        '''
        import pandas
        if not isinstance(value, pandas.DataFrame):
            raise ErrorInitFrame(f'from_pandas must be called with a Pandas DataFrame object, not: {type(value)}')

        pdvu1 = pandas_version_under_1()

        def part_to_array(part: 'pandas.DataFrame') -> np.ndarray:
            if pdvu1:
                array = part.values
                if own_data:
                    array.flags.writeable = False
            else:
                array = pandas_to_numpy(part, own_data=own_data)
            return array

        # create generator of contiguous typed data
        # calling .values will force type unification accross all columns
        def blocks() -> tp.Iterator[np.ndarray]:
            pairs = enumerate(value.dtypes.values)
            column_start, dtype_current = next(pairs)
            column_last = column_start
            yield_block = False

            for column, dtype in pairs:
                try:
                    if dtype != dtype_current:
                        yield_block = True
                except TypeError:
                    # data type not understood, happens with pd datatypes to np dtypes in pd >= 1
                    yield_block = True

                if yield_block:
                    part = value.iloc[NULL_SLICE,
                            slice(column_start, column_last + 1)]
                    yield part_to_array(part)

                    column_start = column
                    dtype_current = dtype
                    yield_block = False

                column_last = column

            # always have left over
            part = value.iloc[NULL_SLICE, slice(column_start, None)]
            yield part_to_array(part)

        if consolidate_blocks:
            blocks = TypeBlocks.from_blocks(TypeBlocks.consolidate_blocks(blocks()))
        else:
            blocks = TypeBlocks.from_blocks(blocks())

        if name is not NAME_DEFAULT:
            pass # keep
        elif 'name' not in value.columns and hasattr(value, 'name'):
            # avoid getting a Series if a column
            name = value.name
        else:
            name = None # do not keep as NAME_DEFAULT

        own_index = True
        if index_constructor is IndexAutoFactory:
            index = None
            own_index = False
        elif index_constructor is not None:
            index = index_constructor(value.index)
        else:
            index = Index.from_pandas(value.index)

        own_columns = True
        if columns_constructor is IndexAutoFactory:
            columns = None
            own_columns = False
        elif columns_constructor is not None:
            columns = columns_constructor(value.columns)
        else:
            columns = cls._COLUMNS_CONSTRUCTOR.from_pandas(value.columns)

        return cls(blocks,
                index=index,
                columns=columns,
                name=name,
                own_data=True,
                own_index=own_index,
                own_columns=own_columns
                )

    @classmethod
    @doc_inject(selector='from_any')
    def from_arrow(cls,
            value: 'pyarrow.Table',
            *,
            index_depth: int = 0,
            index_name_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            columns_depth: int = 1,
            columns_name_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            dtypes: DtypesSpecifier = None,
            name: tp.Hashable = None,
            consolidate_blocks: bool = False,
            ) -> 'Frame':
        '''Realize a ``Frame`` from an Arrow Table.

        Args:
            value: A :obj:`pyarrow.Table` instance.
            {index_depth}
            {columns_depth}
            {dtypes}
            {name}
            {consolidate_blocks}
        '''

        # this is similar to from_structured_array
        index_start_pos = -1 # will be ignored
        index_end_pos = -1
        if index_depth > 0:
            index_start_pos = 0
            index_end_pos = index_start_pos + index_depth - 1
            apex_labels = []
            index_arrays = []
        else:
            apex_labels = None

        if columns_depth > 0:
            columns_labels = []


        # by using value.columns_names, we expose access to the index arrays, which is deemed desirable as that is what we do in from_delimited
        get_col_dtype = None if dtypes is None else get_col_dtype_factory(
                dtypes,
                value.column_names)

        pdvu1 = pandas_version_under_1()

        def blocks() -> tp.Iterator[np.ndarray]:
            for col_idx, (name, chunked_array) in enumerate(
                    zip(value.column_names, value.columns)):
                # NOTE: name will be the encoded columns representation, or auto increment integers; if an IndexHierarchy, will contain all depths: "['a' 1]"
                # This creates a Series with an index; better to find a way to go only to numpy, but does not seem available on ChunkedArray, even with pyarrow==0.16.0
                series = chunked_array.to_pandas(
                        date_as_object=False, # get an np array
                        self_destruct=True, # documented as "experimental"
                        ignore_metadata=True,
                        )
                if pdvu1:
                    array_final = series.values
                else:
                    array_final = pandas_to_numpy(series, own_data=True)

                if get_col_dtype:
                    # ordered values will include index positions
                    dtype = get_col_dtype(col_idx) #pylint: disable=E1102
                    if dtype is not None:
                        array_final = array_final.astype(dtype)

                array_final.flags.writeable = False

                is_index_col = (col_idx >= index_start_pos and col_idx <= index_end_pos)

                if is_index_col:
                    index_arrays.append(array_final)
                    apex_labels.append(name)
                    continue

                if not is_index_col and columns_depth > 0:
                    # only accumulate column names after index extraction
                    columns_labels.append(name)

                yield array_final

        if consolidate_blocks:
            data = TypeBlocks.from_blocks(TypeBlocks.consolidate_blocks(blocks()))
        else:
            data = TypeBlocks.from_blocks(blocks())

        # will be none if name_depth_level is None
        columns_name = None if not apex_labels else apex_to_name(rows=(apex_labels,),
                depth_level=columns_name_depth_level,
                axis=1,
                axis_depth=columns_depth,
                )

        if columns_depth == 0:
            columns = None
            own_columns = False
        elif columns_depth == 1:
            columns = cls._COLUMNS_CONSTRUCTOR(columns_labels, name=columns_name)
            own_columns = True
        elif columns_depth > 1:
            columns = cls._COLUMNS_HIERARCHY_CONSTRUCTOR.from_labels_delimited(
                    columns_labels,
                    delimiter=' ',
                    name=columns_name,
                    )
            own_columns = True

        index_name = None if not apex_labels else apex_to_name(rows=(apex_labels,),
                depth_level=index_name_depth_level,
                axis=0,
                axis_depth=index_depth,
                )

        if index_depth == 0:
            index = None
            own_index = False
        elif index_depth == 1:
            index = Index(index_arrays[0], name=index_name)
            own_index = True
        elif index_depth > 1:
            index = IndexHierarchy._from_type_blocks(
                    TypeBlocks.from_blocks(index_arrays),
                    name=index_name,
                    own_blocks=True,
                    )
            own_index = False

        return cls(
                data=data,
                columns=columns,
                index=index,
                name=name,
                own_data=True,
                own_columns=own_columns,
                own_index=own_index,
                )


    @classmethod
    @doc_inject(selector='from_any')
    def from_parquet(cls,
            fp: PathSpecifier,
            *,
            index_depth: int = 0,
            index_name_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            columns_depth: int = 1,
            columns_name_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            columns_select: tp.Optional[tp.Iterable[str]] = None,
            dtypes: DtypesSpecifier = None,
            name: tp.Hashable = None,
            consolidate_blocks: bool = False,
            ) -> 'Frame':
        '''
        Realize a ``Frame`` from a Parquet file.

        Args:
            {fp}
            {index_depth}
            {columns_depth}
            {columns_select}
            {dtypes}
            {name}
            {consolidate_blocks}
        '''
        import pyarrow.parquet as pq #type: ignore

        if columns_select and index_depth != 0:
            raise ErrorInitFrame(f'cannot load index_depth {index_depth} when columns_select is specified.')

        fp = path_filter(fp)

        if columns_select is not None and not isinstance(columns_select, list):
            columns_select = list(columns_select)

        # NOTE: the order of columns_select will determine their order
        table = pq.read_table(fp,
                columns=columns_select,
                use_pandas_metadata=False,
                )
        if columns_select:
            # pq.read_table will silently accept requested columns that are not found; this can be identified if we got back fewer columns than requested
            if len(table.column_names) < len(columns_select):
                missing = set(columns_select) - set(table.column_names)
                raise ErrorInitFrame(f'cannot load all columns in columns_select: missing {missing}')

        return cls.from_arrow(table,
                index_depth=index_depth,
                index_name_depth_level=index_name_depth_level,
                columns_depth=columns_depth,
                columns_name_depth_level=columns_name_depth_level,
                dtypes=dtypes,
                consolidate_blocks=consolidate_blocks,
                name=name
                )

    @staticmethod
    @doc_inject(selector='constructor_frame')
    def from_msgpack(
            msgpack_data: bin
            ) -> 'Frame':
        '''Frame constructor from an in-memory binary object formatted as a msgpack.

        Args:
            msgpack_data: A binary msgpack object, encoding a Frame as produced from to_msgpack()
        '''
        import msgpack
        import msgpack_numpy

        def decode(obj: dict, #dict produced by msgpack-python
                chain: tp.Callable[[tp.Any], str] = msgpack_numpy.decode,
                ) -> object:

            # NOTE: maybe this can be replaced by dictionary of SF containers provided by container_util
            globals_ref = globals()

            if b'sf' in obj:
                clsname = obj[b'sf']
                cls = globals_ref[clsname]

                if issubclass(cls, Frame):
                    blocks = unpackb(obj[b'blocks'])
                    return cls(
                            blocks,
                            name=obj[b'name'],
                            index=unpackb(obj[b'index']),
                            columns=unpackb(obj[b'columns']),
                            own_data=True,
                            )
                elif issubclass(cls, IndexHierarchy):
                    index_constructors=[
                            globals_ref[clsname] for clsname in unpackb(
                                   obj[b'index_constructors'])]
                    blocks = unpackb(obj[b'blocks'])
                    return cls._from_type_blocks(
                            blocks=blocks,
                            name=obj[b'name'],
                            index_constructors=index_constructors,
                            own_blocks=True)
                elif issubclass(cls, Index):
                    data = unpackb(obj[b'data'])
                    return cls(
                            data,
                            name=obj[b'name'])
                elif issubclass(cls, TypeBlocks):
                    blocks = unpackb(obj[b'blocks'])
                    return cls.from_blocks(blocks)

            elif b'np' in obj:
                #Overridden msgpack-numpy datatypes
                data = unpackb(obj[b'data'])
                typename = obj[b'dtype'].split('[', 1)[0]

                if typename in ['datetime64', 'timedelta64', '>m8', '>M8']:
                    array = np.array(data, dtype=obj[b'dtype'])

                elif typename == 'object_':
                    array = np.array(
                            list(map(element_decode, data)),
                            dtype=DTYPE_OBJECT)

                array.flags.writeable = False
                return array

            return chain(obj)

        unpackb = partial(msgpack.unpackb, object_hook=decode)
        element_decode = partial(MessagePackElement.decode, unpackb=unpackb)

        return unpackb(msgpack_data)

    #---------------------------------------------------------------------------
    @doc_inject(selector='container_init', class_name='Frame')
    def __init__(self,
            data: FrameInitializer = FRAME_INITIALIZER_DEFAULT,
            *,
            index: tp.Union[IndexInitializer, IndexAutoFactoryType] = None,
            columns: tp.Union[IndexInitializer, IndexAutoFactoryType] = None,
            name: NameType = NAME_DEFAULT,
            index_constructor: IndexConstructor = None,
            columns_constructor: IndexConstructor = None,
            own_data: bool = False,
            own_index: bool = False,
            own_columns: bool = False
            ) -> None:
        '''
        Initializer.

        Args:
            data: Default Frame initialization requires typed data such as a NumPy array. All other initialization should use specialized constructors.
            {index}
            {columns}
            {own_data}
            {own_index}
            {own_columns}
        '''
        # we can determine if columns or index are empty only if they are not iterators; those cases will have to use a deferred evaluation
        columns_empty = index_constructor_empty(columns)
        index_empty = index_constructor_empty(index)

        #-----------------------------------------------------------------------
        # blocks assignment

        blocks_constructor = None

        if data.__class__ is TypeBlocks: # PERF: no sublcasses supported
            if own_data:
                self._blocks = data
            else:
                # assume we need to create a new TB instance; this will not copy underlying arrays as all blocks are immutable
                self._blocks = TypeBlocks.from_blocks(data._blocks)
        elif data.__class__ is np.ndarray:
            if own_data:
                data.flags.writeable = False
            # from_blocks will apply immutable filter
            self._blocks = TypeBlocks.from_blocks(data)
        elif data is FRAME_INITIALIZER_DEFAULT:
            # NOTE: this will not catch all cases where index or columns is empty, as they might be iterators; those cases will be handled below.
            def blocks_constructor(shape: tp.Tuple[int, ...]) -> None: #pylint: disable=E0102
                if shape[0] > 0 and shape[1] > 0:
                    # if fillable and we still have default initializer, this is a problem
                    raise RuntimeError('must supply a non-default value for constructing a Frame with non-zero size.')
                self._blocks = TypeBlocks.from_zero_size_shape(shape)
        elif isinstance(data, Frame):
            self._blocks = data._blocks.copy()
            if index is None and index_constructor is None:
                # set up for direct assignment below; index is always immutable
                index = data.index
                own_index = True
            if columns is None and columns_constructor is None:
                # cannot own, but can let constructors handle potential mutability
                columns = data.columns
                columns_empty = index_constructor_empty(columns)
            if name is NAME_DEFAULT:
                name = data.name

        elif isinstance(data, dict):
            raise ErrorInitFrame('use Frame.from_dict to create a Frame from a mapping.')
        elif isinstance(data, Series):
            raise ErrorInitFrame('use Frame.from_series to create a Frame from a Series.')
        else:
            raise ErrorInitFrame('use Frame.from_element, Frame.from_elements, or Frame.from_records to create a Frame from 0, 1, or 2 dimensional untyped data (respectively).')

        # counts can be zero (not None) if _block was created but is empty
        row_count, col_count = (self._blocks._shape
                if not blocks_constructor else (None, None))

        self._name = None if name is NAME_DEFAULT else name_filter(name)

        #-----------------------------------------------------------------------
        # columns assignment

        if own_columns:
            self._columns = columns
            col_count = len(self._columns)
        elif columns_empty:
            col_count = 0 if col_count is None else col_count
            self._columns = IndexAutoFactory.from_optional_constructor(
                    col_count,
                    default_constructor=self._COLUMNS_CONSTRUCTOR,
                    explicit_constructor=columns_constructor
                    )
        else:
            self._columns = index_from_optional_constructor(columns,
                    default_constructor=self._COLUMNS_CONSTRUCTOR,
                    explicit_constructor=columns_constructor
                    )
            col_count = len(self._columns)
        # check after creation, as we cannot determine from the constructor (it might be a method on a class)
        if self._COLUMNS_CONSTRUCTOR.STATIC != self._columns.STATIC:
            raise ErrorInitFrame(f'supplied column constructor does not match required static attribute: {self._COLUMNS_CONSTRUCTOR.STATIC}')
        #-----------------------------------------------------------------------
        # index assignment

        if own_index:
            self._index = index
            row_count = len(self._index)
        elif index_empty:
            row_count = 0 if row_count is None else row_count
            self._index = IndexAutoFactory.from_optional_constructor(
                    row_count,
                    default_constructor=Index,
                    explicit_constructor=index_constructor,
                    )
        else:
            self._index = index_from_optional_constructor(index,
                    default_constructor=Index,
                    explicit_constructor=index_constructor,
                    )
            row_count = len(self._index)

        if not self._index.STATIC:
            raise ErrorInitFrame('non-static index cannot be assigned to Frame')

        #-----------------------------------------------------------------------
        # final evaluation

        # for indices that are created by generators, need to reevaluate if data has been given for an empty index or columns
        columns_empty = col_count == 0
        index_empty = row_count == 0

        if blocks_constructor:
            # if we have a blocks_constructor if is because data remained FRAME_INITIALIZER_DEFAULT
            blocks_constructor((row_count, col_count))

        # final check of block/index coherence
        if self._blocks.shape[0] != row_count:
            # row count might be 0 for an empty DF
            raise ErrorInitFrame(
                f'Index has incorrect size (got {self._blocks.shape[0]}, expected {row_count})'
                )
        if self._blocks.shape[1] != col_count:
            raise ErrorInitFrame(
                f'Columns has incorrect size (got {self._blocks.shape[1]}, expected {col_count})'
                )

    #---------------------------------------------------------------------------

    def __deepcopy__(self, memo: tp.Dict[int, tp.Any]) -> 'Frame':
        obj = self.__new__(self.__class__)
        obj._blocks = deepcopy(self._blocks, memo)
        obj._columns = deepcopy(self._columns, memo)
        obj._index = deepcopy(self._index, memo)
        obj._name = self._name # should be hashable/immutable

        memo[id(self)] = obj
        return obj #type: ignore

    # def __copy__(self) -> 'Frame':
    #     '''
    #     Return shallow copy of this Frame.
    #     '''

    # def copy(self)-> 'Frame':
    #     '''
    #     Return shallow copy of this Frame.
    #     '''
    #     return self.__copy__() #type: ignore

    #---------------------------------------------------------------------------
    # name interface

    @property
    @doc_inject()
    def name(self) -> NameType:
        '''{}'''
        return self._name

    def rename(self,
            name: NameType = NAME_DEFAULT,
            *,
            index: NameType = NAME_DEFAULT,
            columns: NameType = NAME_DEFAULT,
            ) -> 'Frame':
        '''
        Return a new Frame with an updated name attribute. Optionally update the name attribute of ``index`` and ``columns``.
        '''
        name = self.name if name is NAME_DEFAULT else name
        i = self._index if index is NAME_DEFAULT else self._index.rename(index)
        c = self._columns if columns is NAME_DEFAULT else self._columns.rename(columns)

        return self.__class__(self._blocks.copy(),
                index=i,
                columns=c, # let constructor handle if GO
                name=name,
                own_data=True,
                own_index=True)

    #---------------------------------------------------------------------------
    # interfaces

    @property
    def loc(self) -> InterfaceGetItem['Frame']:
        return InterfaceGetItem(self._extract_loc)

    @property
    def iloc(self) -> InterfaceGetItem['Frame']:
        return InterfaceGetItem(self._extract_iloc)

    @property
    def bloc(self) -> InterfaceGetItem['Frame']:
        return InterfaceGetItem(self._extract_bloc)

    @property
    def drop(self) -> InterfaceSelectTrio['Frame']:
        return InterfaceSelectTrio(
            func_iloc=self._drop_iloc,
            func_loc=self._drop_loc,
            func_getitem=self._drop_getitem)

    @property
    def mask(self) -> InterfaceSelectTrio['Frame']:
        return InterfaceSelectTrio(
            func_iloc=self._extract_iloc_mask,
            func_loc=self._extract_loc_mask,
            func_getitem=self._extract_getitem_mask)

    @property
    def masked_array(self) -> InterfaceSelectTrio['Frame']:
        return InterfaceSelectTrio(
            func_iloc=self._extract_iloc_masked_array,
            func_loc=self._extract_loc_masked_array,
            func_getitem=self._extract_getitem_masked_array)

    @property
    def assign(self) -> InterfaceAssignQuartet['Frame']:
        return InterfaceAssignQuartet(
            func_iloc=self._extract_iloc_assign,
            func_loc=self._extract_loc_assign,
            func_getitem=self._extract_getitem_assign,
            func_bloc=self._extract_bloc_assign,
            delegate=FrameAssign,
            )

    @property
    @doc_inject(select='astype')
    def astype(self) -> InterfaceAsType['Frame']:
        '''
        Retype one or more columns. When used as a function, can provide  retype the entire ``Frame``;  Alternatively, when used as a ``__getitem__`` interface, loc-style column selection can be used to type one or more coloumns.

        Args:
            {dtype}
        '''
        # NOTE: this uses the same function for __call__ and __getitem__; call simply uses the NULL_SLICE and applys the dtype argument immediately
        return InterfaceAsType(func_getitem=self._extract_getitem_astype)

    #---------------------------------------------------------------------------
    # via interfaces

    @property
    def via_str(self) -> InterfaceString['Frame']:
        '''
        Interface for applying string methods to elements in this container.
        '''
        def blocks_to_container(blocks: tp.Iterator[np.ndarray]) -> 'Frame':
            tb = TypeBlocks.from_blocks(blocks)
            return self.__class__(
                    tb,
                    index=self._index,
                    columns=self._columns,
                    name=self._name,
                    own_index=True,
                    own_data=True,
                    )

        return InterfaceString(
                blocks=self._blocks._blocks,
                blocks_to_container=blocks_to_container,
                )

    @property
    def via_dt(self) -> InterfaceDatetime['Frame']:
        '''
        Interface for applying datetime properties and methods to elements in this container.
        '''

        # NOTE: we only process object dt64 types; strings have to be converted explicitly

        def blocks_to_container(blocks: tp.Iterator[np.ndarray]) -> 'Frame':
            tb = TypeBlocks.from_blocks(blocks)
            return self.__class__(
                    tb,
                    index=self._index,
                    columns=self._columns,
                    name=self._name,
                    own_index=True,
                    own_data=True,
                    )

        return InterfaceDatetime(
                blocks=self._blocks._blocks,
                blocks_to_container=blocks_to_container,
                )

    @property
    def via_T(self) -> InterfaceTranspose['Frame']:
        '''
        Interface for using binary operators with one-dimensional sequences, where the opperand is applied column-wise.
        '''
        return InterfaceTranspose(
                container=self,
                )


    def via_fill_value(self,
            fill_value: object = np.nan,
            ) -> InterfaceFillValue['Frame']:
        '''
        Interface for using binary operators and methods with a pre-defined fill value.
        '''
        return InterfaceFillValue(
                container=self,
                fill_value=fill_value,
                )

    def via_re(self,
            pattern: str,
            flags: int = 0,
            ) -> InterfaceRe['Frame']:
        '''
        Interface for applying regular expressions to elements in this container.
        '''
        def blocks_to_container(blocks: tp.Iterator[np.ndarray]) -> 'Frame':
            tb = TypeBlocks.from_blocks(blocks)
            return self.__class__(
                    tb,
                    index=self._index,
                    columns=self._columns,
                    name=self._name,
                    own_index=True,
                    own_data=True,
                    )
        return InterfaceRe(
                blocks=self._blocks._blocks,
                blocks_to_container=blocks_to_container,
                pattern=pattern,
                flags=flags,
                )


    #---------------------------------------------------------------------------
    # iterators

    @property
    def iter_array(self) -> IterNodeAxis['Frame']:
        '''
        Iterator of :obj:`np.array`, where arrays are drawn from columns (axis=0) or rows (axis=1)
        '''
        return IterNodeAxis(
                container=self,
                function_values=self._axis_array,
                function_items=self._axis_array_items,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.SERIES_VALUES
                )

    @property
    def iter_array_items(self) -> IterNodeAxis['Frame']:
        '''
        Iterator of pairs of label, :obj:`np.array`, where arrays are drawn from columns (axis=0) or rows (axis=1)
        '''
        return IterNodeAxis(
                container=self,
                function_values=self._axis_array,
                function_items=self._axis_array_items,
                yield_type=IterNodeType.ITEMS,
                apply_type=IterNodeApplyType.SERIES_VALUES
                )

    @property
    def iter_tuple(self) -> IterNodeConstructorAxis['Frame']:
        '''
        Iterator of :obj:`NamedTuple`, where tuples are drawn from columns (axis=0) or rows (axis=1). An optional ``constructor`` callable can be used to provide a :obj:`NamedTuple` class (or any other constructor called with a single iterable) to be used to create each yielded axis value.
        '''
        return IterNodeConstructorAxis(
                container=self,
                function_values=self._axis_tuple,
                function_items=self._axis_tuple_items,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.SERIES_VALUES
                )

    @property
    def iter_tuple_items(self) -> IterNodeConstructorAxis['Frame']:
        '''
        Iterator of pairs of label, :obj:`NamedTuple`, where tuples are drawn from columns (axis=0) or rows (axis=1)
        '''
        return IterNodeConstructorAxis(
                container=self,
                function_values=self._axis_tuple,
                function_items=self._axis_tuple_items,
                yield_type=IterNodeType.ITEMS,
                apply_type=IterNodeApplyType.SERIES_VALUES
                )

    @property
    def iter_series(self) -> IterNodeAxis['Frame']:
        '''
        Iterator of :obj:`Series`, where :obj:`Series` are drawn from columns (axis=0) or rows (axis=1)
        '''
        return IterNodeAxis(
                container=self,
                function_values=self._axis_series,
                function_items=self._axis_series_items,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.SERIES_VALUES
                )

    @property
    def iter_series_items(self) -> IterNodeAxis['Frame']:
        '''
        Iterator of pairs of label, :obj:`Series`, where :obj:`Series` are drawn from columns (axis=0) or rows (axis=1)
        '''
        return IterNodeAxis(
                container=self,
                function_values=self._axis_series,
                function_items=self._axis_series_items,
                yield_type=IterNodeType.ITEMS,
                apply_type=IterNodeApplyType.SERIES_VALUES
                )

    #---------------------------------------------------------------------------
    @property
    def iter_group(self) -> IterNodeGroupAxis['Frame']:
        '''
        Iterator of :obj:`Frame` grouped by unique values found in one or more columns (axis=0) or rows (axis=1).
        '''
        return IterNodeGroupAxis(
                container=self,
                function_values=self._axis_group_loc,
                function_items=self._axis_group_loc_items,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_VALUES,
                )

    @property
    def iter_group_items(self) -> IterNodeGroupAxis['Frame']:
        '''
        Iterator of pairs of label, :obj:`Frame` grouped by unique values found in one or more columns (axis=0) or rows (axis=1).
        '''
        return IterNodeGroupAxis(
                container=self,
                function_values=self._axis_group_loc,
                function_items=self._axis_group_loc_items,
                yield_type=IterNodeType.ITEMS,
                apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_VALUES,
                )

    @property
    def iter_group_labels(self) -> IterNodeDepthLevelAxis['Frame']:
        '''
        Iterator of :obj:`Frame` grouped by unique labels found in one or more index depths (axis=0) or columns depths (axis=1).
        '''
        return IterNodeDepthLevelAxis(
                container=self,
                function_values=self._axis_group_labels,
                function_items=self._axis_group_labels_items,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_LABELS,
                )

    @property
    def iter_group_labels_items(self) -> IterNodeDepthLevelAxis['Frame']:
        '''
        Iterator of pairs of label, :obj:`Frame` grouped by unique labels found in one or more index depths (axis=0) or columns depths (axis=1).
        '''
        return IterNodeDepthLevelAxis(
                container=self,
                function_values=self._axis_group_labels,
                function_items=self._axis_group_labels_items,
                yield_type=IterNodeType.ITEMS,
                apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_LABELS,
                )

    #---------------------------------------------------------------------------

    @property
    @doc_inject(selector='window')
    def iter_window(self) -> IterNodeWindow['Frame']:
        '''
        Iterator of windowed values, where values are given as a :obj:`Frame`.

        {args}
        '''
        function_values = partial(self._axis_window, as_array=False)
        function_items = partial(self._axis_window_items, as_array=False)
        return IterNodeWindow(
                container=self,
                function_values=function_values,
                function_items=function_items,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.SERIES_ITEMS,
                )

    @property
    @doc_inject(selector='window')
    def iter_window_items(self) -> IterNodeWindow['Frame']:
        '''
        Iterator of pairs of label, windowed values, where values are given as a :obj:`Frame`.

        {args}
        '''
        function_values = partial(self._axis_window, as_array=False)
        function_items = partial(self._axis_window_items, as_array=False)
        return IterNodeWindow(
                container=self,
                function_values=function_values,
                function_items=function_items,
                yield_type=IterNodeType.ITEMS,
                apply_type=IterNodeApplyType.SERIES_ITEMS,
                )

    @property
    @doc_inject(selector='window')
    def iter_window_array(self) -> IterNodeWindow['Frame']:
        '''
        Iterator of windowed values, where values are given as a :obj:`np.array`.

        {args}
        '''
        function_values = partial(self._axis_window, as_array=True)
        function_items = partial(self._axis_window_items, as_array=True)
        return IterNodeWindow(
                container=self,
                function_values=function_values,
                function_items=function_items,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.SERIES_ITEMS,
                )

    @property
    @doc_inject(selector='window')
    def iter_window_array_items(self) -> IterNodeWindow['Frame']:
        '''
        Iterator of pairs of label, windowed values, where values are given as a :obj:`np.array`.

        {args}
        '''
        function_values = partial(self._axis_window, as_array=True)
        function_items = partial(self._axis_window_items, as_array=True)
        return IterNodeWindow(
                container=self,
                function_values=function_values,
                function_items=function_items,
                yield_type=IterNodeType.ITEMS,
                apply_type=IterNodeApplyType.SERIES_ITEMS,
                )

    #---------------------------------------------------------------------------
    @property
    def iter_element(self) -> IterNodeAxis['Frame']:
        '''Iterator of elements, ordered by row then column.
        '''
        return IterNodeAxis(
                container=self,
                function_values=self._iter_element_loc,
                function_items=self._iter_element_loc_items,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.FRAME_ELEMENTS
                )

    @property
    def iter_element_items(self) -> IterNodeAxis['Frame']:
        '''Iterator of pairs of label, element, where labels are pairs of index, columns labels, ordered by row then column.
        '''
        return IterNodeAxis(
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
            iloc_key: GetItemKeyTypeCompound,
            fill_value: object = np.nan
            ) -> tp.Union[Series, 'Frame']:
        '''Given a value that is a Series or Frame, reindex it to the index components, drawn from this Frame, that are specified by the iloc_key.
        '''
        if isinstance(iloc_key, tuple):
            row_key, column_key = iloc_key
        else:
            row_key, column_key = iloc_key, None

        # within this frame, get Index objects by extracting based on passed-in iloc keys
        nm_row, nm_column = self._extract_axis_not_multi(row_key, column_key)
        v = None

        if nm_row and not nm_column:
            # only column is multi selection, reindex by column
            if isinstance(value, Series):
                v = value.reindex(self._columns._extract_iloc(column_key),
                        fill_value=fill_value)
        elif not nm_row and nm_column:
            # only row is multi selection, reindex by index
            if isinstance(value, Series):
                v = value.reindex(self._index._extract_iloc(row_key),
                        fill_value=fill_value)
        elif not nm_row and not nm_column:
            # both multi, must be a Frame
            if isinstance(value, Frame):
                target_column_index = self._columns._extract_iloc(column_key)
                target_row_index = self._index._extract_iloc(row_key)
                # this will use the default fillna type, which may or may not be what is wanted
                v = value.reindex(
                        index=target_row_index,
                        columns=target_column_index,
                        fill_value=fill_value)
        if v is None:
            raise RuntimeError(('cannot assign '
                    + value.__class__.__name__
                    + ' with key configuration'), (nm_row, nm_column))
        return v

    @doc_inject(selector='reindex', class_name='Frame')
    def reindex(self,
            index: tp.Optional[IndexInitializer] = None,
            columns: tp.Optional[IndexInitializer] = None,
            *,
            fill_value: object = np.nan,
            own_index: bool = False,
            own_columns: bool = False,
            check_equals: bool = True,
            ) -> 'Frame':
        '''
        {doc}

        Args:
            index: {index_initializer}
            columns: {index_initializer}
            {fill_value}
            {own_index}
            {own_columns}
        '''
        if index is None and columns is None:
            raise RuntimeError('must specify one of index or columns')

        if index is not None:
            if not own_index:
                index = index_from_optional_constructor(index,
                        default_constructor=Index)

            if check_equals and self._index.equals(index):
                index_ic = None
            else:
                index_ic = IndexCorrespondence.from_correspondence(self._index, index)
        else:
            index = self._index
            index_ic = None
        # index can always be owned by this point, as self._index is STATIC, or  we have created a new Index, or we have bbeen given own_index
        own_index_frame = True

        if columns is not None:
            if not own_columns:
                columns = index_from_optional_constructor(columns,
                        default_constructor=self._COLUMNS_CONSTRUCTOR)

            if check_equals and self._columns.equals(columns):
                columns_ic = None
            else:
                columns_ic = IndexCorrespondence.from_correspondence(self._columns, columns)
            own_columns_frame = True
        else:
            columns = self._columns
            columns_ic = None
            own_columns_frame = self._COLUMNS_CONSTRUCTOR.STATIC

        return self.__class__(
                TypeBlocks.from_blocks(
                        self._blocks.resize_blocks(
                                index_ic=index_ic,
                                columns_ic=columns_ic,
                                fill_value=fill_value),
                        shape_reference=(len(index), len(columns))
                        ),
                index=index,
                columns=columns,
                name=self._name,
                own_data=True,
                own_index=own_index_frame,
                own_columns=own_columns_frame
                )

    @doc_inject(selector='relabel', class_name='Frame')
    def relabel(self,
            index: tp.Optional[RelabelInput] = None,
            columns: tp.Optional[RelabelInput] = None
            ) -> 'Frame':
        '''
        {doc}

        Args:
            index: {relabel_input}
            columns: {relabel_input}
        '''
        # create new index objects in both cases so as to call with own*
        if index is None and columns is None:
            raise RuntimeError('must specify one of index or columns')

        own_index = False
        if index is IndexAutoFactory:
            index = None
        elif is_callable_or_mapping(index):
            index = self._index.relabel(index)
            own_index = True
        elif index is None:
            index = self._index
        elif isinstance(index, Set):
            raise RelabelInvalid()

        own_columns = False
        if columns is IndexAutoFactory:
            columns = None
        elif is_callable_or_mapping(columns):
            columns = self._columns.relabel(columns)
            own_columns = True
        elif columns is None:
            columns = self._columns
        elif isinstance(columns, Set):
            raise RelabelInvalid()

        return self.__class__(
                self._blocks.copy(), # does not copy arrays
                index=index,
                columns=columns,
                name=self._name,
                own_data=True,
                own_index=own_index,
                own_columns=own_columns)

    @doc_inject(selector='relabel_flat', class_name='Frame')
    def relabel_flat(self,
            index: bool = False,
            columns: bool = False
            ) -> 'Frame':
        '''
        {doc}

        Args:
            index: Boolean to flag flatening on the index.
            columns: Boolean to flag flatening on the columns.
        '''
        if not index and not columns:
            raise RuntimeError('must specify one or both of columns, index')

        index = self._index.flat() if index else self._index.copy()
        columns = self._columns.flat() if columns else self._columns.copy()

        return self.__class__(
                self._blocks.copy(), # does not copy arrays
                index=index,
                columns=columns,
                name=self._name,
                own_data=True,
                own_index=True,
                own_columns=True)

    @doc_inject(selector='relabel_level_add', class_name='Frame')
    def relabel_level_add(self,
            index: tp.Hashable = None,
            columns: tp.Hashable = None
            ) -> 'Frame':
        '''
        {doc}

        Args:
            index: {level}
            columns: {level}
        '''

        index = self._index.level_add(index) if index else self._index
        columns = self._columns.level_add(columns) if columns else self._columns.copy()


        return self.__class__(
                self._blocks.copy(), # does not copy arrays
                index=index,
                columns=columns,
                name=self._name,
                own_data=True,
                own_index=True,
                own_columns=True)

    @doc_inject(selector='relabel_level_drop', class_name='Frame')
    def relabel_level_drop(self,
            index: int = 0,
            columns: int = 0
            ) -> 'Frame':
        '''
        {doc}

        Args:
            index: {count} Default is zero.
            columns: {count} Default is zero.
        '''

        index = self._index.level_drop(index) if index else self._index.copy()
        columns = self._columns.level_drop(columns) if columns else self._columns.copy()

        return self.__class__(
                self._blocks.copy(), # does not copy arrays
                index=index,
                columns=columns,
                name=self._name,
                own_data=True,
                own_index=True,
                own_columns=True)

    def relabel_shift_in(self,
            key: GetItemKeyType,
            *,
            axis: int = 0,
            ) -> 'Frame':
        '''
        Create, or augment, an :obj:`IndexHierarchy` by providing one or more selections via axis-appropriate ``loc`` selections.

        Args:
            key: a loc-style selection on the opposite axis.
            axis: 0 modifies the index by selecting columns with ``key``; 1 modifies the columns by selecting rows with ``key``.
        '''

        if axis == 0: # select from columns, add to index
            index_target = self._index
            index_opposite = self._columns
        else:
            index_target = self._columns
            index_opposite = self._index

        if index_target.depth == 1:
            ih_blocks = TypeBlocks.from_blocks((index_target.values,))
            name_prior = index_target.names if index_target.name is None else (index_target.name,)
        else:
            if index_target._recache:
                index_target._update_array_cache()
            ih_blocks = index_target._blocks.copy() # will mutate copied blocks
            # only use string form of labels if we are not storing a correctly sized tuple
            name_prior = index_target.name if index_target._name_is_names() else index_target.names

        iloc_key = index_opposite._loc_to_iloc(key)
        # NOTE: must do this before dropping
        if isinstance(iloc_key, INT_TYPES):
            ih_name = name_prior + (index_opposite[iloc_key],)
        else:
            ih_name = name_prior + tuple(index_opposite[iloc_key])

        index_opposite = index_opposite._drop_iloc(iloc_key)

        if axis == 0: # select from columns, add to index
            ih_blocks.extend(self._blocks._extract(column_key=iloc_key))
            frame_blocks = TypeBlocks.from_blocks(
                    self._blocks._drop_blocks(column_key=iloc_key),
                    shape_reference=(self.shape[0], len(index_opposite)),
                    )

            index = IndexHierarchy._from_type_blocks(ih_blocks, name=ih_name)
            columns = index_opposite
        else: # select from index, add to columns
            ih_blocks.extend(self._blocks._extract(row_key=iloc_key).transpose())
            frame_blocks = TypeBlocks.from_blocks(
                    self._blocks._drop_blocks(row_key=iloc_key),
                    shape_reference=(len(index_opposite), self.shape[1]),
                    )

            index = index_opposite
            columns = self._COLUMNS_HIERARCHY_CONSTRUCTOR._from_type_blocks(ih_blocks, name=ih_name)

        return self.__class__(
                frame_blocks, # does not copy arrays
                index=index,
                columns=columns,
                name=self._name,
                own_data=True,
                own_index=True,
                own_columns=True)


    def relabel_shift_out(self,
            depth_level: DepthLevelSpecifier,
            *,
            axis: int = 0,
            ) -> 'Frame':
        '''
        Shift values from an index on an axis to the Frame by providing one or more depth level selections.

        Args:
            key: an iloc-style selection on the axis.
            axis: 0 modifies the index by selecting columns with ``key``; 1 modifies the columns by selecting rows with ``key``.
        '''

        if axis == 0: # select from index, remove from index
            index_target = self._index
            index_opposite = self._columns
            target_ctor = Index
            target_hctor = IndexHierarchy
        elif axis == 1:
            index_target = self._columns
            index_opposite = self._index
            target_ctor = self._COLUMNS_CONSTRUCTOR
            target_hctor = self._COLUMNS_HIERARCHY_CONSTRUCTOR
        else:
            raise AxisInvalid(f'invalid axis {axis}')

        if index_target.depth == 1:
            index_target._depth_level_validate(depth_level) # will raise
            new_target = IndexAutoFactory
            add_blocks = (index_target.values,)
            new_labels = index_target.names if index_target.name is None else (index_target.name,)
        else:
            if index_target._recache:
                index_target._update_array_cache()

            label_src = index_target.name if index_target._name_is_names() else index_target.names
            if isinstance(depth_level, INT_TYPES):
                new_labels = (label_src[depth_level],)
                remain_labels = tuple(label for i, label in enumerate(label_src) if i != depth_level)
            else:
                new_labels = (label_src[i] for i in depth_level)
                remain_labels = tuple(label for i, label in enumerate(label_src) if i not in depth_level)

            target_tb = index_target._blocks
            add_blocks = target_tb._extract(column_key=depth_level)
            if not add_blocks.__class__ is np.ndarray:
                # get iterable off arrays
                add_blocks = add_blocks._blocks
            else:
                add_blocks = (add_blocks,)
            # this might fail if nothing left
            remain_blocks = TypeBlocks.from_blocks(
                    target_tb._drop_blocks(column_key=depth_level),
                    shape_reference=(len(index_target), 0))

            if remain_blocks.shape[1] == 0:
                new_target = IndexAutoFactory
            elif remain_blocks.shape[1] == 1:
                new_target = target_ctor(
                        column_1d_filter(remain_blocks._blocks[0]),
                        name=remain_labels[0])
            else:
                new_target = target_hctor._from_type_blocks(
                        remain_blocks,
                        name=remain_labels
                        )

        if axis == 0: # select from index, remove from index
            blocks = TypeBlocks.from_blocks(chain(add_blocks,
                    self._blocks._blocks))
            index = new_target
            # if we already have a hierarchical index here, there is no way to ensure that the new labels coming in are of appropriate depth; only option is to get a flat version of columns
            if self._columns.depth > 1:
                extend_labels = self._columns.flat().__iter__()
            else:
                extend_labels = self._columns.__iter__()
            columns = self._COLUMNS_CONSTRUCTOR.from_labels(
                    chain(new_labels, extend_labels),
                    name=self._columns.name,
                    )
        else:
            blocks = TypeBlocks.from_blocks(TypeBlocks.vstack_blocks_to_blocks(
                    (TypeBlocks.from_blocks(add_blocks).transpose(), self._blocks))
                    )
            if self._index.depth > 1:
                extend_labels = self._index.flat().__iter__()
            else:
                extend_labels = self._index.__iter__()
            index = Index.from_labels(chain(new_labels, extend_labels),
                    name=self._index.name)
            columns = new_target

        return self.__class__(
                blocks, # does not copy arrays
                index=index,
                columns=columns,
                name=self._name,
                own_data=True,
                own_index=index is not IndexAutoFactory,
                own_columns=columns is not IndexAutoFactory,
                )



    def rehierarch(self,
            index: tp.Optional[tp.Iterable[int]] = None,
            columns: tp.Optional[tp.Iterable[int]] = None,
            ) -> 'Frame':
        '''
        Produce a new `Frame` with index and/or columns constructed with a transformed hierarchy.
        '''
        if index and self.index.depth == 1:
            raise RuntimeError('cannot rehierarch on index when there is no hierarchy')
        if columns and self.columns.depth == 1:
            raise RuntimeError('cannot rehierarch on columns when there is no hierarchy')

        if index:
            index, index_iloc = rehierarch_from_index_hierarchy(
                    labels=self._index,
                    depth_map=index,
                    name=self._index.name
                    )
        else:
            index = self._index
            index_iloc = None

        if columns:
            columns, columns_iloc = rehierarch_from_index_hierarchy(
                    labels=self._columns,
                    depth_map=columns,
                    name=self._columns.name
                    )
            own_columns = True
        else:
            columns = self._columns
            own_columns = False # let constructor determine
            columns_iloc = None

        blocks = self._blocks._extract(index_iloc, columns_iloc)

        return self.__class__(
                blocks,
                index=index,
                columns=columns,
                name=self._name,
                own_data=True,
                own_index=True,
                own_columns=own_columns
                )



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
                own_index=True,
                own_data=True,
                )


    def notna(self) -> 'Frame':
        '''
        Return a same-indexed, Boolean Frame indicating True which values are not NaN or None.
        '''
        # always return a Frame, even if this is a FrameGO
        return Frame(self._blocks.notna(),
                index=self._index,
                columns=self._columns,
                own_index=True,
                own_data=True,
                )

    def dropna(self,
            axis: int = 0,
            condition: tp.Callable[[np.ndarray], bool] = np.all) -> 'Frame':
        '''
        Return a new Frame after removing rows (axis 0) or columns (axis 1) where any or all values are NA (NaN or None). The condition is determined by a NumPy ufunc that process the Boolean array returned by ``isna()``; the default is ``np.all``.

        Args:
            axis:
            condition:
        '''
        # returns Boolean areas that define axis to keep
        row_key, column_key = self._blocks.drop_missing_to_keep_locations(
                axis=axis,
                condition=condition,
                func=isna_array,
                )

        # NOTE: if no values to drop and this is a Frame (not a FrameGO) we can return self as it is immutable. only one of row_key, colum_Key will be an array
        if self.__class__ is Frame:
            if ((column_key is None and row_key.all()) or
                    (row_key is None and column_key.all())):
                return self
        return self._extract(row_key, column_key)


    #---------------------------------------------------------------------------
    # falsy handling

    def isfalsy(self) -> 'Frame':
        '''
        Return a same-indexed, Boolean Frame indicating True which values are falsy.
        '''
        # always return a Frame, even if this is a FrameGO
        return Frame(self._blocks.isfalsy(),
                index=self._index,
                columns=self._columns,
                own_index=True,
                own_data=True,
                )


    def notfalsy(self) -> 'Frame':
        '''
        Return a same-indexed, Boolean Frame indicating True which values are not falsy.
        '''
        # always return a Frame, even if this is a FrameGO
        return Frame(self._blocks.notfalsy(),
                index=self._index,
                columns=self._columns,
                own_index=True,
                own_data=True,
                )

    def dropfalsy(self,
            axis: int = 0,
            condition: tp.Callable[[np.ndarray], bool] = np.all) -> 'Frame':
        '''
        Return a new Frame after removing rows (axis 0) or columns (axis 1) where any or all values are falsy. The condition is determined by a NumPy ufunc that process the Boolean array returned by ``isfalsy()``; the default is ``np.all``.

        Args:
            axis:
            condition:
        '''
        # returns Boolean areas that define axis to keep
        row_key, column_key = self._blocks.drop_missing_to_keep_locations(
                axis=axis,
                condition=condition,
                func=isfalsy_array,
                )

        # NOTE: if no values to drop and this is a Frame (not a FrameGO) we can return self as it is immutable. only one of row_key, colum_Key will be an array
        if self.__class__ is Frame:
            if ((column_key is None and row_key.all()) or
                    (row_key is None and column_key.all())):
                return self
        return self._extract(row_key, column_key)

    #---------------------------------------------------------------------------
    def _fill_missing(self,
            value: tp.Any,
            func: tp.Callable[[np.ndarray], np.ndarray],
            ) -> 'Frame':
        '''
        Args:
            func: function to return True for missing values
        '''

        if hasattr(value, '__iter__') and not isinstance(value, str):
            if not isinstance(value, Frame):
                raise RuntimeError('unlabeled iterables cannot be used for fillna: use a Frame')
            # get a dummy fill_value to use during reindex and avoid undesirable type cooercions
            fill_value = dtype_to_fill_value(value._blocks._row_dtype)

            fill = value.reindex(
                    index=self.index,
                    columns=self.columns,
                    fill_value=fill_value
                    ).values
            # produce a Boolean array that shows True only for labels (index, columns) found in the original `value` argument (before reindexing) and also in the target; this will be used to not set a NA when the value to fill was produced by reindexing.
            fill_valid = self._blocks.extract_iloc_mask((
                    self.index.isin(value.index.values),
                    self.columns.isin(value.columns.values)
                    )).values
        else:
            fill = value
            fill_valid = None

        return self.__class__(
                self._blocks.fill_missing_by_unit(fill, fill_valid, func=func),
                index=self._index,
                columns=self._columns,
                name=self._name,
                own_data=True
                )

    @doc_inject(selector='fillna')
    def fillna(self, value: tp.Any) -> 'Frame':
        '''Return a new ``Frame`` after replacing null (NaN or None) values with the supplied value.

        Args:
            {value}
        '''
        return self._fill_missing(value, func=isna_array)

    @doc_inject(selector='fillna')
    def fillfalsy(self, value: tp.Any) -> 'Frame':
        '''Return a new ``Frame`` after replacing null (NaN or None) values with the supplied value.

        Args:
            {value}
        '''
        return self._fill_missing(value, func=isfalsy_array)

    #---------------------------------------------------------------------------
    @doc_inject(selector='fillna')
    def fillna_leading(self,
            value: tp.Any,
            *,
            axis: int = 0) -> 'Frame':
        '''
        Return a new ``Frame`` after filling leading (and only leading) null (NaN or None) with the first observed value.

        Args:
            {value}
            {axis}
        '''
        return self.__class__(self._blocks.fillna_leading(value, axis=axis),
                index=self._index,
                columns=self._columns,
                name=self._name,
                own_data=True)

    @doc_inject(selector='fillna')
    def fillna_trailing(self,
            value: tp.Any,
            *,
            axis: int = 0) -> 'Frame':
        '''
        Return a new ``Frame`` after filling trailing (and only trailing) null (NaN or None) with the last observed value.

        Args:
            {value}
            {axis}
        '''
        return self.__class__(self._blocks.fillna_trailing(value, axis=axis),
                index=self._index,
                columns=self._columns,
                name=self._name,
                own_data=True)

    @doc_inject(selector='fillna')
    def fillna_forward(self,
            limit: int = 0,
            *,
            axis: int = 0) -> 'Frame':
        '''
        Return a new ``Frame`` after filling forward null (NaN or None) with the last observed value.

        Args:
            {limit}
            {axis}
        '''
        return self.__class__(self._blocks.fillna_forward(limit=limit, axis=axis),
                index=self._index,
                columns=self._columns,
                name=self._name,
                own_data=True)

    @doc_inject(selector='fillna')
    def fillna_backward(self,
            limit: int = 0,
            *,
            axis: int = 0) -> 'Frame':
        '''
        Return a new ``Frame`` after filling backward null (NaN or None) with the first observed value.

        Args:
            {limit}
            {axis}
        '''
        return self.__class__(self._blocks.fillna_backward(limit=limit, axis=axis),
                index=self._index,
                columns=self._columns,
                name=self._name,
                own_data=True)

    #---------------------------------------------------------------------------

    def __len__(self) -> int:
        '''Length of rows in values.
        '''
        return self._blocks._shape[0]

    @doc_inject()
    def display(self,
            config: tp.Optional[DisplayConfig] = None,
            *,
            style_config: tp.Optional[StyleConfig] = None,
            ) -> Display:
        '''{doc}

        Args:
            {config}
        '''
        config = config or DisplayActive.get()
        index_depth = self._index.depth if config.include_index else 0

        # always get the index Display (even if we are not going to use it) to dettermine how many rows we need (which may include types, as well as truncation with elipsis).
        display_index = self._index.display(config=config)

        # header depth used for HTML and other foramtting; needs to be adjusted if removing types and/or columns and types, When showing types on a Frame, we need 2: one for the Frame type, the other for the index type.
        header_depth = (self._columns.depth * config.include_columns) + (2 * config.type_show)

        # create an empty display based on index display
        d = Display([list() for _ in range(len(display_index))],
                config=config,
                outermost=True,
                index_depth=index_depth,
                header_depth=header_depth,
                style_config=style_config,
                )


        if config.include_index:
            # this will add more rows to accomodate the index if it is bigger due to types
            d.extend_display(display_index)
            header_column = '' if config.type_show else None
        else:
            header_column = None

        if self._blocks._shape[1] > config.display_columns:
            # columns as they will look after application of truncation and insertion of ellipsis
            data_half_count = Display.truncate_half_count(
                    config.display_columns)
            column_gen = partial(_gen_skip_middle,
                    forward_iter=partial(self._blocks.axis_values, axis=0),
                    forward_count=data_half_count,
                    reverse_iter=partial(self._blocks.axis_values, axis=0, reverse=True),
                    reverse_count=data_half_count,
                    center_sentinel=Display.ELLIPSIS_CENTER_SENTINEL
                    )
        else:
            column_gen = partial(self._blocks.axis_values, axis=0)

        for column in column_gen():
            if column is Display.ELLIPSIS_CENTER_SENTINEL:
                d.extend_ellipsis()
            else:
                d.extend_iterable(column, header=header_column)

        #-----------------------------------------------------------------------
        config_transpose = config.to_transpose()

        #-----------------------------------------------------------------------
        # prepare header display of container class
        header_displays = []
        if config.type_show:
            display_cls = Display.from_values((),
                    header=DisplayHeader(self.__class__, self._name),
                    config=config_transpose)
            header_displays.append(display_cls.flatten())

        #-----------------------------------------------------------------------
        # prepare columns display
        if config.include_columns:
            # need to apply the config_transpose such that it truncates it based on the the max columns, not the max rows
            display_columns = self._columns.display(config=config_transpose)

            if config.type_show:
                index_depth_extend = self._index.depth - 1
                spacer_insert_index = 1 # after the first, the name
            elif not config.type_show and config.include_index:
                index_depth_extend = self._index.depth
                spacer_insert_index = 0
            elif not config.include_index: # type_show must be False
                index_depth_extend = 0
                spacer_insert_index = 0

            # add spacers to from of columns when we have a hierarchical index
            for _ in range(index_depth_extend):
                # will need a width equal to the column depth
                row = [Display.to_cell('', config=config)
                        for _ in range(self._columns.depth)]
                spacer = Display([row])
                display_columns.insert_displays(spacer,
                        insert_index=spacer_insert_index)

            if self._columns.depth > 1:
                display_columns_horizontal = display_columns.transform()
            else: # can just flatten a single column into one row
                display_columns_horizontal = display_columns.flatten()

            header_displays.append(display_columns_horizontal)

        if header_displays:
            d.insert_displays(*header_displays)

        return d

    #---------------------------------------------------------------------------
    # accessors

    @property
    @doc_inject(selector='values_2d', class_name='Frame')
    def values(self) -> np.ndarray:
        '''
        {}
        '''
        return self._blocks.values

    @property
    def index(self) -> Index:
        '''The ``IndexBase`` instance assigned for row labels.
        '''
        return self._index

    @property
    def columns(self) -> Index:
        '''The ``IndexBase`` instance assigned for column labels.
        '''
        return self._columns

    #---------------------------------------------------------------------------
    # common attributes from the numpy array

    @property
    def dtypes(self) -> Series:
        '''
        Return a Series of dytpes for each realizable column.

        Returns:
            :obj:`static_frame.Series`
        '''
        return Series(self._blocks.dtypes,
                index=immutable_index_filter(self._columns),
                name=self._name
                )

    @property
    @doc_inject()
    def mloc(self) -> np.ndarray:
        '''{doc_array}
        '''
        return self._blocks.mloc

    #---------------------------------------------------------------------------

    @property
    def shape(self) -> tp.Tuple[int, int]:
        '''
        Return a tuple describing the shape of the underlying NumPy array.

        Returns:
            :obj:`tp.Tuple[int]`
        '''
        return self._blocks._shape

    @property
    def ndim(self) -> int:
        '''
        Return the number of dimensions, which for a `Frame` is always 2.

        Returns:
            :obj:`int`
        '''
        return self._NDIM

    @property
    def size(self) -> int:
        '''
        Return the size of the underlying NumPy array.

        Returns:
            :obj:`int`
        '''

        return self._blocks.size

    @property
    def nbytes(self) -> int:
        '''
        Return the total bytes of the underlying NumPy array.

        Returns:
            :obj:`int`
        '''
        return self._blocks.nbytes


    #---------------------------------------------------------------------------
    def _extract_array(self,
            row_key: GetItemKeyType = None,
            column_key: GetItemKeyType = None,
            ) -> np.ndarray:
        '''
        Alternative extractor that returns just an ndarray
        '''
        return self._blocks._extract_array(row_key, column_key)

    @staticmethod
    def _extract_axis_not_multi(
                row_key: tp.Hashable,
                column_key: tp.Hashable,
                ) -> tp.Tuple[bool, bool]:
        '''
        If either row or column is given with a non-multiple type of selection (a single scalar), reduce dimensionality.
        '''
        row_nm = False
        column_nm = False
        # NOTE: can we just identify integer types?
        if row_key is not None and not isinstance(row_key, KEY_MULTIPLE_TYPES):
            row_nm = True # axis 0
        if column_key is not None and not isinstance(column_key, KEY_MULTIPLE_TYPES):
            column_nm = True # axis 1
        return row_nm, column_nm


    def _extract(self,
            row_key: GetItemKeyType = None,
            column_key: GetItemKeyType = None,
            ) -> tp.Union['Frame', Series]:
        '''
        Extract Container based on iloc selection (indices have already mapped)
        '''
        blocks = self._blocks._extract(row_key=row_key, column_key=column_key)

        if blocks.__class__ is not TypeBlocks:
            return blocks # reduced to an element

        own_index = True # the extracted Frame can always own this index
        row_key_is_slice = row_key.__class__ is slice
        if row_key is None or (row_key_is_slice and row_key == NULL_SLICE):
            index = self._index
        else:
            index = self._index._extract_iloc(row_key)
            if not row_key_is_slice:
                name_row = self._index.values[row_key]
                if self._index.depth > 1:
                    name_row = tuple(name_row)

        # can only own columns if _COLUMNS_CONSTRUCTOR is static
        column_key_is_slice = column_key.__class__ is slice
        if column_key is None or (column_key_is_slice and column_key == NULL_SLICE):
            columns = self._columns
            own_columns = self._COLUMNS_CONSTRUCTOR.STATIC
        else:
            columns = self._columns._extract_iloc(column_key)
            own_columns = True
            if not column_key_is_slice:
                name_column = self._columns.values[column_key]
                if self._columns.depth > 1:
                    name_column = tuple(name_column)

        # determine if an axis is not multi; if one axis is not multi, we return a Series instead of a Frame
        axis_nm = self._extract_axis_not_multi(row_key, column_key)
        blocks_shape = blocks._shape

        if blocks_shape[0] == 0 or blocks_shape[1] == 0:
            # return a 0-sized Series, `blocks` is already extracted
            array = column_1d_filter(blocks._blocks[0]) if blocks._blocks else EMPTY_ARRAY
            if axis_nm[0]: # if row not multi
                return Series(array,
                        index=immutable_index_filter(columns),
                        name=name_row)
            elif axis_nm[1]:
                return Series(array,
                        index=index,
                        name=name_column)
        elif blocks_shape == (1, 1):
            # if TypeBlocks did not return an element, need to determine which axis to use for Series index
            if axis_nm[0]: # if row not multi
                return Series(blocks.values[0],
                        index=immutable_index_filter(columns),
                        name=name_row)
            elif axis_nm[1]:
                return Series(blocks.values[0],
                        index=index,
                        name=name_column)
            # if both are multi, we return a Frame
        elif blocks_shape[0] == 1: # if one row
            if axis_nm[0]: # if row key not multi
                # best to use blocks.values, as will need to consolidate dtypes; will always return a 2D array
                return Series(blocks.values[0],
                        index=immutable_index_filter(columns),
                        name=name_row)
        elif blocks_shape[1] == 1: # if one column
            if axis_nm[1]: # if column key is not multi
                return Series(
                        column_1d_filter(blocks._blocks[0]),
                        index=index,
                        name=name_column)

        return self.__class__(blocks,
                index=index,
                columns=columns,
                name=self._name,
                own_data=True, # always get new TypeBlock instance above
                own_index=own_index,
                own_columns=own_columns
                )


    def _extract_iloc(self, key: GetItemKeyTypeCompound) -> 'Frame':
        '''
        Give a compound key, return a new Frame. This method simply handles the variabiliyt of single or compound selectors.
        '''
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
            iloc_column_key = self._columns._loc_to_iloc(loc_column_key)
        else:
            loc_row_key = key
            iloc_column_key = None

        iloc_row_key = self._index._loc_to_iloc(loc_row_key)
        return iloc_row_key, iloc_column_key


    def _extract_loc(self, key: GetItemKeyTypeCompound) -> 'Frame':
        return self._extract(*self._compound_loc_to_iloc(key))


    def _extract_bloc(self, key: Bloc2DKeyType) -> Series:
        '''
        2D Boolean selector, selected by either a Boolean 2D Frame or array.
        '''
        bloc_key = bloc_key_normalize(key=key, container=self)
        coords, values = self._blocks.extract_bloc(bloc_key) # immutable, 1D array
        index = Index(
                ((self._index[x], self._columns[y]) for x, y in coords),
                dtype=DTYPE_OBJECT)
        return Series(values, index=index, own_index=True)

    def _compound_loc_to_getitem_iloc(self,
            key: GetItemKeyTypeCompound) -> tp.Tuple[GetItemKeyType, GetItemKeyType]:
        '''Handle a potentially compound key in the style of __getitem__. This will raise an appropriate exception if a two argument loc-style call is attempted.
        '''
        iloc_column_key = self._columns._loc_to_iloc(key)
        return None, iloc_column_key

    @doc_inject(selector='selector')
    def __getitem__(self, key: GetItemKeyType) -> tp.Union['Frame', Series]:
        '''Selector of columns by label.

        Args:
            key: {key_loc}
        '''
        return self._extract(*self._compound_loc_to_getitem_iloc(key))


    #---------------------------------------------------------------------------

    def _drop_iloc(self, key: GetItemKeyTypeCompound) -> 'Frame':
        '''
        Args:
            key: If a Boolean Series was passed, it has been converted to Boolean NumPy array already in loc to iloc.
        '''

        blocks = self._blocks.drop(key)

        if isinstance(key, tuple):
            iloc_row_key, iloc_column_key = key

            index = self._index._drop_iloc(iloc_row_key)
            own_index = True

            columns = self._columns._drop_iloc(iloc_column_key)
            own_columns = True
        else:
            iloc_row_key = key # no column selection

            index = self._index._drop_iloc(iloc_row_key)
            own_index = True

            columns = self._columns
            own_columns = False

        return self.__class__(blocks,
                columns=columns,
                index=index,
                name=self._name,
                own_data=True,
                own_columns=own_columns,
                own_index=own_index
                )

    def _drop_loc(self, key: GetItemKeyTypeCompound) -> 'Frame':
        key = self._compound_loc_to_iloc(key)
        return self._drop_iloc(key=key)

    def _drop_getitem(self, key: GetItemKeyTypeCompound) -> 'Frame':
        key = self._compound_loc_to_getitem_iloc(key)
        return self._drop_iloc(key=key)


    #---------------------------------------------------------------------------
    def _extract_iloc_mask(self, key: GetItemKeyTypeCompound) -> 'Frame':
        masked_blocks = self._blocks.extract_iloc_mask(key)
        return self.__class__(masked_blocks,
                columns=self._columns,
                index=self._index,
                own_data=True)

    def _extract_loc_mask(self, key: GetItemKeyTypeCompound) -> 'Frame':
        key = self._compound_loc_to_iloc(key)
        return self._extract_iloc_mask(key=key)

    def _extract_getitem_mask(self, key: GetItemKeyTypeCompound) -> 'Frame':
        key = self._compound_loc_to_getitem_iloc(key)
        return self._extract_iloc_mask(key=key)

    #---------------------------------------------------------------------------
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
    def _extract_iloc_assign(self, key: GetItemKeyTypeCompound) -> 'FrameAssignILoc':
        return FrameAssignILoc(self, key=key)

    def _extract_loc_assign(self, key: GetItemKeyTypeCompound) -> 'FrameAssignILoc':
        # extract if tuple, then pack back again
        key = self._compound_loc_to_iloc(key)
        return self._extract_iloc_assign(key=key)

    def _extract_getitem_assign(self, key: GetItemKeyTypeCompound) -> 'FrameAssignILoc':
        # extract if tuple, then pack back again
        key = self._compound_loc_to_getitem_iloc(key)
        return self._extract_iloc_assign(key=key)

    def _extract_bloc_assign(self, key: Bloc2DKeyType) -> 'FrameAssignBLoc':
        '''Assignment based on a Boolean Frame or array.'''
        return FrameAssignBLoc(self, key=key)

    #---------------------------------------------------------------------------

    def _extract_getitem_astype(self, key: GetItemKeyType) -> 'FrameAsType':
        # extract if tuple, then pack back again
        _, key = self._compound_loc_to_getitem_iloc(key)
        return FrameAsType(self, column_key=key)

    #---------------------------------------------------------------------------
    # dictionary-like interface

    def keys(self) -> tp.Iterable[tp.Hashable]:
        '''Iterator of column labels.
        '''
        return self._columns

    def __iter__(self) -> tp.Iterable[tp.Hashable]:
        '''
        Iterator of column labels, same as :py:meth:`Frame.keys`.
        '''
        return self._columns.__iter__()

    def __contains__(self, value: tp.Hashable) -> bool:
        '''
        Inclusion of value in column labels.
        '''
        return self._columns.__contains__(value)

    def items(self) -> tp.Iterator[tp.Tuple[tp.Hashable, Series]]:
        '''Iterator of pairs of column label and corresponding column :obj:`Series`.
        '''
        for label, array in zip(self._columns.values, self._blocks.axis_values(0)):
            # array is assumed to be immutable
            yield label, Series(array, index=self._index, name=label)

    def get(self,
            key: tp.Hashable,
            default: tp.Optional[Series] = None,
            ) -> Series:
        '''
        Return the value found at the columns key, else the default if the key is not found. This method is implemented to complete the dictionary-like interface.
        '''
        if key not in self._columns:
            return default
        return self.__getitem__(key)


    #---------------------------------------------------------------------------
    # operator functions

    def _ufunc_unary_operator(self,
            operator: tp.Callable[[np.ndarray], np.ndarray],
            ) -> 'Frame':
        # call the unary operator on _blocks
        return self.__class__(
                self._blocks._ufunc_unary_operator(operator=operator),
                index=self._index,
                columns=self._columns,
                name=self._name,
                )

    def _ufunc_binary_operator(self, *,
            operator: UFunc,
            other: tp.Any,
            axis: int = 0,
            fill_value: object = np.nan,
            ) -> 'Frame':

        if operator.__name__ == 'matmul':
            return matmul(self, other)
        elif operator.__name__ == 'rmatmul':
            return matmul(other, self)

        if isinstance(other, Frame):
            name = None
            # reindex both dimensions to union indices
            # NOTE: union and reindexing check equals first
            columns = self._columns.union(other._columns)
            index = self._index.union(other._index)

            # NOTE: always own column, index, as we will just extract Typeblocks
            self_tb = self.reindex(
                    columns=columns,
                    index=index,
                    own_index=True,
                    own_columns=True,
                    fill_value=fill_value,
                    )._blocks
            # NOTE: we create columns from self._columns, and thus other can only own it if STATIC matches
            own_columns = other.STATIC == self.STATIC
            other_tb = other.reindex(
                    columns=columns,
                    index=index,
                    own_index=True,
                    own_columns=own_columns,
                    fill_value=fill_value,
                    )._blocks
            return self.__class__(self_tb._ufunc_binary_operator(
                            operator=operator,
                            other=other_tb),
                    index=index,
                    columns=columns,
                    own_data=True,
                    own_index=True,
                    )
        elif isinstance(other, Series):
            name = None
            if axis == 0:
                # when operating on a Series, we treat axis 0 as a row-wise operation, and thus take the union of the Series.index and Frame.columns
                columns = self._columns.union(other._index)
                self_tb = self.reindex(
                        columns=columns,
                        own_columns=True,
                        fill_value=fill_value,
                        )._blocks
                other_array = other.reindex(
                        columns,
                        own_index=True,
                        fill_value=fill_value,
                        ).values
                blocks = self_tb._ufunc_binary_operator(
                        operator=operator,
                        other=other_array,
                        axis=axis,
                        )
                return self.__class__(blocks,
                        index=self._index,
                        columns=columns,
                        own_data=True,
                        own_index=True,
                        )
            elif axis == 1:
                # column-wise operation, take union of Series.index and Frame.index
                index = self._index.union(other._index)
                self_tb = self.reindex(
                        index=index,
                        own_index=True,
                        fill_value=fill_value,
                        )._blocks
                other_array = other.reindex(
                        index,
                        own_index=True,
                        fill_value=fill_value,
                        ).values
                blocks = self_tb._ufunc_binary_operator(
                        operator=operator,
                        other=other_array,
                        axis=axis,
                        )
                return self.__class__(blocks,
                        index=index,
                        columns=self._columns,
                        own_data=True,
                        own_index=True,
                        )
            else:
                raise AxisInvalid(f'invalid axis: {axis}')
        elif other.__class__ is np.ndarray:
            name = None
        elif other.__class__ is InterfaceFillValue:
            raise RuntimeError('via_fill_value interfaces can only be used on the left-hand side of binary expressions.')
        else:
            other = iterable_to_array_nd(other)
            if other.ndim == 0:# only for elements should we keep name
                name = self._name
            else:
                name = None

        # assume we will keep dimensionality
        blocks = self._blocks._ufunc_binary_operator(
                operator=operator,
                other=other,
                axis=axis,
                )
        return self.__class__(blocks,
                index=self._index,
                columns=self._columns,
                own_data=True,
                own_index=True,
                name=name,
                )

    #---------------------------------------------------------------------------
    # axis functions

    def _ufunc_axis_skipna(self, *,
            axis: int,
            skipna: bool,
            ufunc: UFunc,
            ufunc_skipna: UFunc,
            composable: bool,
            dtypes: tp.Tuple[np.dtype, ...],
            size_one_unity: bool
            ) -> 'Series':
        # axis 0 processes ros, deliveres column index
        # axis 1 processes cols, delivers row index
        assert axis < 2

        post = self._blocks.ufunc_axis_skipna(
                skipna=skipna,
                axis=axis,
                ufunc=ufunc,
                ufunc_skipna=ufunc_skipna,
                composable=composable,
                dtypes=dtypes,
                size_one_unity=size_one_unity
                )

        # post has been made immutable so Series will own
        if axis == 0:
            return Series(
                    post,
                    index=immutable_index_filter(self._columns)
                    )
        return Series(post, index=self._index)

    def _ufunc_shape_skipna(self, *,
            axis: int,
            skipna: bool,
            ufunc: UFunc,
            ufunc_skipna: UFunc,
            composable: bool,
            dtypes: tp.Tuple[np.dtype, ...],
            size_one_unity: bool
            ) -> 'Frame':
        # axis 0 processes ros, deliveres column index
        # axis 1 processes cols, delivers row index
        assert axis < 2

        dtype = None if not dtypes else dtypes[0] # only a tuple

        # assumed not composable for axis 1, full-shape processing requires processing contiguous values
        v = self.values
        if skipna:
            post = ufunc_skipna(v, axis=axis, dtype=dtype)
        else:
            post = ufunc(v, axis=axis, dtype=dtype)

        post.flags.writeable = False

        return self.__class__(
                TypeBlocks.from_blocks(post),
                index=self._index,
                columns=self._columns,
                own_data=True,
                own_index=True
                )

    #---------------------------------------------------------------------------
    # axis iterators
    # NOTE: if there is more than one argument, the axis argument needs to be key-word only

    def _axis_array(self, axis: int) -> tp.Iterator[np.ndarray]:
        '''Generator of arrays across an axis
        '''
        yield from self._blocks.axis_values(axis)

    def _axis_array_items(self, axis: int) -> tp.Iterator[tp.Tuple[tp.Hashable, np.ndarray]]:
        keys = self._index if axis == 1 else self._columns
        yield from zip(keys, self._blocks.axis_values(axis))


    def _axis_tuple(self, *,
            axis: int,
            constructor: tp.Type[tuple] = None,
            ) -> tp.Iterator[tp.NamedTuple]:
        '''Generator of named tuples across an axis.

        Args:
            axis: 0 iterates over columns (index axis), 1 iterates over rows (column axis)
        '''
        if constructor is None:
            if axis == 1:
                labels = self._columns.values
            elif axis == 0:
                labels = self._index.values
            else:
                raise AxisInvalid(f'no support for axis {axis}')
            # uses _make method to call with iterable
            constructor = get_tuple_constructor(labels)
        elif (isinstance(constructor, type) and
                issubclass(constructor, tuple) and
                hasattr(constructor, '_make')):
            constructor = constructor._make

        for axis_values in self._blocks.axis_values(axis):
            yield constructor(axis_values)

    def _axis_tuple_items(self, *,
            axis: int,
            constructor: tp.Type[tuple] = None,
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, np.ndarray]]:
        keys = self._index if axis == 1 else self._columns
        yield from zip(keys, self._axis_tuple(axis=axis, constructor=constructor))


    def _axis_series(self, axis: int) -> tp.Iterator[Series]:
        '''Generator of Series across an axis
        '''
        # reference the indices and let the constructor reuse what is reusable
        if axis == 1:
            index = (self._columns if self._columns.STATIC
                    else self._columns._IMMUTABLE_CONSTRUCTOR(self._columns))
            labels = self._index
        elif axis == 0:
            index = self._index
            labels = self._columns

        for label, axis_values in zip(labels, self._blocks.axis_values(axis)):
            yield Series(axis_values, index=index, name=label, own_index=True)

    def _axis_series_items(self, axis: int) -> tp.Iterator[tp.Tuple[tp.Hashable, np.ndarray]]:
        keys = self._index if axis == 1 else self._columns
        yield from zip(keys, self._axis_series(axis=axis))


    #---------------------------------------------------------------------------
    # grouping methods naturally return their "index" as the group element

    def _axis_group_iloc_items(self,
            key: GetItemKeyType,
            *,
            axis: int,
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, 'Frame']]:

        for group, selection, tb in self._blocks.group(axis=axis, key=key):
            if axis == 0:
                # axis 0 is a row iter, so need to slice index, keep columns
                yield group, self.__class__(tb,
                        index=self._index[selection],
                        columns=self._columns,
                        own_columns=self.STATIC, # own if static
                        own_index=True,
                        own_data=True)
            elif axis == 1:
                # axis 1 is a column iterators, so need to slice columns, keep index
                yield group, self.__class__(tb,
                        index=self._index,
                        columns=self._columns[selection],
                        own_index=True,
                        own_columns=True,
                        own_data=True)
            else:
                raise AxisInvalid(f'invalid axis: {axis}') #pragma: no cover (already caught above)

    def _axis_group_sort_items(self,
            key: GetItemKeyType,
            iloc_key: GetItemKeyType,
            axis: int
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, 'Frame']]:
        '''
        Optimized grouping when key is an element.
        '''
        # Create a sorted copy since we do not want to change the underlying data
        frame_sorted: Frame = self.sort_values(key, axis=not axis)

        def extract_frame(
                key: GetItemKeyType,
                index: IndexBase,
                ) -> 'Frame':
            if axis == 0:
                return Frame(frame_sorted._blocks._extract(row_key=key),
                        columns=self._columns,
                        index=index,
                        own_columns=self.STATIC, # own if static
                        own_index=True,
                        own_data=True,
                        )
            return Frame(frame_sorted._blocks._extract(column_key=key),
                    columns=index,
                    index=self._index,
                    own_columns=True,
                    own_index=True,
                    own_data=True,
                    )

        if not self._blocks.size:
            return

        if axis == 0:
            index: Index = frame_sorted.index
            group_values = frame_sorted._blocks._extract_array(column_key=iloc_key)
        else:
            index = frame_sorted.columns
            group_values = frame_sorted._blocks._extract_array(row_key=iloc_key)

        # find where new value is not equal to previous; drop the first as roll wraps
        transitions = np.flatnonzero(group_values != np.roll(group_values, 1))[1:]
        start = 0
        for t in transitions:
            slc = slice(start, t)
            yield group_values[start], extract_frame(slc, index[slc])
            start = t
        yield group_values[start], extract_frame(slice(start, None), index[start:])


    def _axis_group_loc_items(self,
            key: GetItemKeyType,
            *,
            axis: int = 0
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, 'Frame']]:
        '''
        Args:
            key: We accept any thing that can do loc to iloc. Note that a tuple is permitted as key, where it would be interpreted as a single label for an IndexHierarchy.
        '''
        if axis == 0: # row iterator, selecting columns for group by
            iloc_key = self._columns._loc_to_iloc(key)
        elif axis == 1: # column iterator, selecting rows for group by
            iloc_key = self._index._loc_to_iloc(key)
        else:
            raise AxisInvalid(f'invalid axis: {axis}')

        # NOTE: might identify when key is a list of one item

        # Optimized sorting approach is only supported in a limited number of cases
        if (self.columns.depth == 1 and
                self.index.depth == 1 and
                not isinstance(key, KEY_MULTIPLE_TYPES)
                ):
            if axis == 0:
                has_object = self._blocks.dtypes[iloc_key] == DTYPE_OBJECT
            else:
                has_object = self._blocks._row_dtype == DTYPE_OBJECT
            if not has_object:
                yield from self._axis_group_sort_items(key=key,
                        iloc_key=iloc_key,
                        axis=axis)
            else:
                yield from self._axis_group_iloc_items(key=iloc_key, axis=axis)
        else:
            yield from self._axis_group_iloc_items(key=iloc_key, axis=axis)


    def _axis_group_loc(self,
            key: GetItemKeyType,
            *,
            axis: int = 0
            ) -> tp.Iterator['Frame']:
        yield from (x for _, x in self._axis_group_loc_items(key=key, axis=axis))


    def _axis_group_labels_items(self,
            depth_level: DepthLevelSpecifier = 0,
            *,
            axis: int = 0,
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, 'Frame']]:

        if axis == 0: # maintain columns, group by index
            ref_index = self._index
        elif axis == 1: # maintain index, group by columns
            ref_index = self._columns
        else:
            raise AxisInvalid(f'invalid axis: {axis}')

        values = ref_index.values_at_depth(depth_level)
        group_to_tuple = values.ndim > 1

        groups, locations = array_to_groups_and_locations(values)

        for idx, group in enumerate(groups):
            selection = locations == idx

            if axis == 0:
                # axis 0 is a row iter, so need to slice index, keep columns
                tb = self._blocks._extract(row_key=selection)
                yield group, self.__class__(tb,
                        index=self._index[selection],
                        columns=self._columns, # let constructor determine ownership
                        own_index=True,
                        own_data=True)

            elif axis == 1:
                # axis 1 is a column iterators, so need to slice columns, keep index
                tb = self._blocks._extract(column_key=selection)
                yield group, self.__class__(tb,
                        index=self._index,
                        columns=self._columns[selection],
                        own_index=True,
                        own_columns=True,
                        own_data=True)

    def _axis_group_labels(self,
            depth_level: DepthLevelSpecifier = 0,
            *,
            axis: int = 0,
            ) -> tp.Iterator['Frame']:
        yield from (x for _, x in self._axis_group_labels_items(
                depth_level=depth_level, axis=axis))

    #---------------------------------------------------------------------------
    def _axis_window_items(self, *,
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
        '''Generator of index, processed-window pairs.
        '''
        yield from axis_window_items(
                source=self,
                size=size,
                axis=axis,
                step=step,
                window_sized=window_sized,
                window_func=window_func,
                window_valid=window_valid,
                label_shift=label_shift,
                start_shift=start_shift,
                size_increment=size_increment,
                as_array=as_array
                )


    def _axis_window(self, *,
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
            ) -> tp.Iterator['Frame']:
        yield from (x for _, x in self._axis_window_items(
                size=size,
                axis=axis,
                step=step,
                window_sized=window_sized,
                window_func=window_func,
                window_valid=window_valid,
                label_shift=label_shift,
                start_shift=start_shift,
                size_increment=size_increment,
                as_array=as_array
                ))


    #---------------------------------------------------------------------------

    def _iter_element_iloc_items(self,
            axis: int = 0,
            ) -> tp.Iterator[tp.Tuple[tp.Tuple[int, int], tp.Any]]:
        yield from self._blocks.element_items(axis=axis)

    # def _iter_element_iloc(self):
    #     yield from (x for _, x in self._iter_element_iloc_items())

    def _iter_element_loc_items(self,
            axis: int = 0,
            ) -> tp.Iterator[tp.Tuple[tp.Tuple[tp.Hashable, tp.Hashable], tp.Any]]:
        '''
        Generator of pairs of (index, column), value. This is driven by ``np.ndindex``, and thus orders by row.
        '''
        yield from (
                ((self._index[k[0]], self._columns[k[1]]), v)
                for k, v in self._blocks.element_items(axis=axis)
                )

    def _iter_element_loc(self,
            axis: int = 0,
            ) -> tp.Iterator[tp.Any]:
        yield from (x for _, x in
                self._iter_element_loc_items(axis=axis))


    #---------------------------------------------------------------------------
    # transformations resulting in the same dimensionality

    def __reversed__(self) -> tp.Iterator[tp.Hashable]:
        '''
        Returns a reverse iterator on the frame's columns.
        '''
        return reversed(self._columns)

    @doc_inject(selector='sort')
    def sort_index(self,
            *,
            ascending: BoolOrBools = True,
            kind: str = DEFAULT_SORT_KIND,
            key: tp.Optional[tp.Callable[[IndexBase], tp.Union[np.ndarray, IndexBase]]] = None,
            ) -> 'Frame':
        '''
        Return a new :obj:`Frame` ordered by the sorted Index.

        Args:
            {ascendings}
            {kind}
            {key}
        '''
        order = sort_index_for_order(self._index, kind=kind, ascending=ascending, key=key)

        index = self._index[order]

        blocks = self._blocks.iloc[order]
        return self.__class__(blocks,
                index=index,
                columns=self._columns,
                name=self._name,
                own_data=True,
                own_index=True,
                )

    @doc_inject(selector='sort')
    def sort_columns(self,
            *,
            ascending: BoolOrBools = True,
            kind: str = DEFAULT_SORT_KIND,
            key: tp.Optional[tp.Callable[[IndexBase], tp.Union[np.ndarray, IndexBase]]] = None,
            ) -> 'Frame':
        '''
        Return a new :obj:`Frame` ordered by the sorted ``columns``.

        Args:
            {ascendings}
            {kind}
            {key}
        '''
        order = sort_index_for_order(self._columns, kind=kind, ascending=ascending, key=key)

        columns = self._columns[order]

        blocks = self._blocks[order]
        return self.__class__(blocks,
                index=self._index,
                columns=columns,
                name=self._name,
                own_data=True,
                own_columns=True,
                )

    @doc_inject(selector='sort')
    def sort_values(self,
            label: KeyOrKeys, # elsewhere this is called 'key'
            *,
            ascending: BoolOrBools = True,
            axis: int = 1,
            kind: str = DEFAULT_SORT_KIND,
            key: tp.Optional[tp.Callable[[tp.Union['Frame', Series]], tp.Union[np.ndarray, 'Series', 'Frame']]] = None,
            ) -> 'Frame':
        '''
        Return a new :obj:`Frame` ordered by the sorted values, where values are given by single column or iterable of columns.

        Args:
            label: A label or iterable of labels to select the columns (for axis 1) or rows (for axis 0) to sort.
            *
            {ascendings}
            axis: Axis upon which to sort; 0 orders columns based on one or more rows; 1 orders rows based on one or more columns.
            {kind}
            {key}
        '''
        values_for_sort: tp.Optional[np.ndarray] = None
        values_for_lex: tp.Optional[tp.List[np.ndarray]] = None

        if axis == 0: # get a column ordering based on one or more rows
            iloc_key = self._index._loc_to_iloc(label)
            if key:
                cfs = key(self._extract(row_key=iloc_key))
                cfs_is_array = cfs.__class__ is np.ndarray
                if (cfs.ndim == 1 and len(cfs) != self.shape[1]) or (cfs.ndim == 2 and cfs.shape[1] != self.shape[1]):
                    raise RuntimeError('key function returned a container of invalid length')
            else: # go straigt to array as, since this is row-wise, have to find a consolidated
                cfs = self._blocks._extract_array(row_key=iloc_key)
                cfs_is_array = True

            if cfs_is_array:
                if cfs.ndim == 1:
                    values_for_sort = cfs
                elif cfs.ndim == 2 and cfs.shape[0] == 1:
                    values_for_sort = cfs[0]
                else:
                    values_for_lex = [cfs[i] for i in range(cfs.shape[0]-1, -1, -1)]
            elif cfs.ndim == 1: # Series
                values_for_sort = cfs.values
            elif isinstance(cfs, Frame):
                cfs = cfs._blocks
                if cfs.shape[0] == 1:
                    values_for_sort = cfs._extract_array(row_key=0)
                else:
                    values_for_lex = [cfs._extract_array(row_key=i)
                            for i in range(cfs.shape[0]-1, -1, -1)]

        elif axis == 1: # get a row ordering based on one or more columns
            iloc_key = self._columns._loc_to_iloc(label)
            if key:
                cfs = key(self._extract(column_key=iloc_key))
                cfs_is_array = cfs.__class__ is np.ndarray
                if (cfs.ndim == 1 and len(cfs) != self.shape[0]) or (cfs.ndim == 2 and cfs.shape[0] != self.shape[0]):
                    raise RuntimeError('key function returned a container of invalid length')
            else: # get array from blocks
                cfs = self._blocks._extract(column_key=iloc_key) # get TypeBlocks
                cfs_is_array = False

            if cfs_is_array:
                if cfs.ndim == 1:
                    values_for_sort = cfs
                elif cfs.ndim == 2 and cfs.shape[1] == 1:
                    values_for_sort = cfs[:, 0]
                else:
                    values_for_lex = [cfs[:, i] for i in range(cfs.shape[1]-1, -1, -1)]
            elif cfs.ndim == 1: # Series
                values_for_sort = cfs.values
            else: #Frame/TypeBlocks from here
                if isinstance(cfs, Frame):
                    cfs = cfs._blocks
                if cfs.shape[1] == 1:
                    values_for_sort = cfs._extract_array(column_key=0)
                else:
                    values_for_lex = [cfs._extract_array(column_key=i)
                            for i in range(cfs.shape[1]-1, -1, -1)]
        else:
            raise AxisInvalid(f'invalid axis: {axis}')

        asc_is_element, values_for_lex = prepare_values_for_lex(
                ascending=ascending,
                values_for_lex=values_for_lex,
                )
        # asc_is_element = isinstance(ascending, BOOL_TYPES)
        # if not asc_is_element:
        #     ascending = tuple(ascending)
        #     if values_for_lex is None or len(ascending) != len(values_for_lex):
        #         raise RuntimeError(f'Multiple ascending values must match number of arrays selected.')

        if values_for_lex is not None:
            # if not asc_is_element:
            #     # values for lex are in reversed order; thus take ascending reversed
            #     values_for_lex_post = []
            #     for asc, a in zip(reversed(ascending), values_for_lex):
            #         # if not ascending, replace with an inverted dense rank
            #         if not asc:
            #             values_for_lex_post.append(
            #                     rank_1d(a, method=RankMethod.DENSE, ascending=False))
            #         else:
            #             values_for_lex_post.append(a)
            #     values_for_lex = values_for_lex_post
            order = np.lexsort(values_for_lex)
        elif values_for_sort is not None:
            order = np.argsort(values_for_sort, kind=kind)
        else:
            raise RuntimeError('unable to resovle sort type')

        if asc_is_element and not ascending:
            # NOTE: if asc is not an element, then ascending Booleans have already been applied to values_for_lex
            # NOTE: putting the order in reverse, not invetering the selection, produces the descending sort
            order = order[::-1]

        if axis == 0:
            columns = self._columns[order]
            blocks = self._blocks[order] # order columns
            return self.__class__(blocks,
                    index=self._index,
                    columns=columns,
                    name=self._name,
                    own_data=True,
                    own_columns=True,
                    own_index=True,
                    )

        index = self._index[order]
        blocks = self._blocks.iloc[order]
        return self.__class__(blocks,
                index=index,
                columns=self._columns,
                name=self._name,
                own_data=True,
                own_index=True
                )

    def isin(self, other: tp.Any) -> 'Frame':
        '''
        Return a same-sized Boolean :obj:`Frame` that shows if the same-positioned element is in the passed iterable.
        '''
        return self.__class__(
                self._blocks.isin(other),
                index=self._index,
                columns=self._columns,
                own_data=True,
                name=self._name,
                )


    @doc_inject(class_name='Frame')
    def clip(self, *,
            lower: tp.Optional[tp.Union[float, Series, 'Frame']] = None,
            upper: tp.Optional[tp.Union[float, Series, 'Frame']] = None,
            axis: tp.Optional[int] = None
            ) -> 'Frame':
        '''{}

        Args:
            lower: value, :obj:`Series`, :obj:`Frame`
            upper: value, :obj:`Series`, :obj:`Frame`
            axis: required if ``lower`` or ``upper`` are given as a :obj:`Series`.
        '''
        if lower is None and upper is None:
            return self.__class__(self._blocks.copy(),
                    index=self._index,
                    columns=self._columns,
                    own_data=True,
                    name=self._name
                    )

        args = [lower, upper]
        for idx, arg in enumerate(args):
            if arg is None:
                continue
            bound = -np.inf if idx == 0 else np.inf

            if isinstance(arg, Series):
                if axis is None:
                    raise RuntimeError('cannot use a Series argument without specifying an axis')
                target = self._index if axis == 0 else self._columns
                values = arg.reindex(target).fillna(bound).values
                if axis == 0: # duplicate the same column over the width
                    # NOTE: extracting array, then scaling in a list, assuming we are just multiply references, not creating copies
                    args[idx] = [values] * self.shape[1]
                else:
                    # create a list of row-length arrays for maximal type preservation
                    args[idx] = [np.full(self.shape[0], v) for v in values]

            elif isinstance(arg, Frame):
                args[idx] = arg.reindex(
                        index=self._index,
                        columns=self._columns).fillna(bound)._blocks._blocks

            elif hasattr(arg, '__iter__'):
                raise RuntimeError('only Series or Frame are supported as iterable lower/upper arguments')
            # assume single value otherwise, no change necessary

        blocks = self._blocks.clip(*args)

        return self.__class__(blocks,
                columns=self._columns,
                index=self._index,
                name=self._name,
                own_data=True,
                own_index=True,
                )


    def transpose(self) -> 'Frame':
        '''Transpose. Return a :obj:`Frame` with ``index`` as ``columns`` and vice versa.
        '''
        return self.__class__(self._blocks.transpose(),
                index=self._columns,
                columns=self._index,
                own_data=True,
                name=self._name)

    @property
    def T(self) -> 'Frame':
        '''Transpose. Return a :obj:`Frame` with ``index`` as ``columns`` and vice versa.
        '''
        return self.transpose()

    @doc_inject(selector='duplicated')
    def duplicated(self, *,
            axis: int = 0,
            exclude_first: bool = False,
            exclude_last: bool = False) -> 'Series':
        '''
        Return an axis-sized Boolean :obj:`Series` that shows True for all rows (axis 0) or columns (axis 1) duplicated.

        Args:
            {axis}
            {exclude_first}
            {exclude_last}
        '''
        # TODO: might be able to do this witnout calling .values and passing in TypeBlocks, but TB needs to support roll
        duplicates = array_to_duplicated(self.values,
                axis=axis,
                exclude_first=exclude_first,
                exclude_last=exclude_last)
        duplicates.flags.writeable = False
        if axis == 0: # index is index
            return Series(duplicates, index=self._index)
        return Series(duplicates, index=self._columns)

    @doc_inject(selector='duplicated')
    def drop_duplicated(self, *,
            axis: int = 0,
            exclude_first: bool = False,
            exclude_last: bool = False
            ) -> 'Frame':
        '''
        Return a :obj:`Frame` with duplicated rows (axis 0) or columns (axis 1) removed. All values in the row or column are compared to determine duplication.

        Args:
            {axis}
            {exclude_first}
            {exclude_last}
        '''
        # NOTE: full row or column comparison is necessary, so passing .values is likely the only option.
        duplicates = array_to_duplicated(self.values,
                axis=axis,
                exclude_first=exclude_first,
                exclude_last=exclude_last,
                )

        if not duplicates.any():
            return self.__class__(
                    self._blocks.copy(),
                    index=self._index,
                    columns=self._columns,
                    own_data=True,
                    own_index=True,
                    name=self._name,
                    )

        keep = ~duplicates

        if axis == 0: # return rows with index indexed
            return self.__class__(
                    self._blocks._extract(row_key=keep),
                    index=self._index[keep],
                    columns=self._columns,
                    own_index=True,
                    name=self._name,
                    own_data=True,
                    )
        return self.__class__(
                self._blocks._extract(column_key=keep),
                index=self._index,
                columns=self._columns[keep],
                own_index=True,
                name=self._name,
                own_data=True,
                )
        # invalid axis will raise in array_to_duplicated

    def set_index(self,
            column: tp.Hashable,
            *,
            drop: bool = False,
            index_constructor: IndexConstructor = Index,
            ) -> 'Frame':
        '''
        Return a new frame produced by setting the given column as the index, optionally removing that column from the new Frame.
        '''
        column_iloc = self._columns._loc_to_iloc(column)

        if drop:
            blocks = TypeBlocks.from_blocks(
                    self._blocks._drop_blocks(column_key=column_iloc))
            columns = self._columns._drop_iloc(column_iloc)
            own_data = True
            own_columns = True
        else:
            blocks = self._blocks
            columns = self._columns
            own_data = False
            own_columns = False

        if isinstance(column_iloc, INT_TYPES):
            index_values = self._blocks._extract_array(column_key=column_iloc)
            name = column
        else:
            index_values = array2d_to_array1d(
                    self._blocks._extract_array(column_key=column_iloc))
            name = tuple(self._columns[column_iloc])

        index = index_constructor(index_values, name=name)

        return self.__class__(blocks,
                columns=columns,
                index=index,
                own_data=own_data,
                own_columns=own_columns,
                own_index=True,
                name=self._name
                )

    def set_index_hierarchy(self,
            columns: GetItemKeyType,
            *,
            drop: bool = False,
            index_constructors: tp.Optional[IndexConstructors] = None,
            reorder_for_hierarchy: bool = False,
            ) -> 'Frame':
        '''
        Given an iterable of column labels, return a new ``Frame`` with those columns as an ``IndexHierarchy`` on the index.

        Args:
            columns: Iterable of column labels.
            drop: Boolean to determine if selected columns should be removed from the data.
            index_constructors: Optionally provide a sequence of ``Index`` constructors, of length equal to depth, to be used in converting columns Index components in the ``IndexHierarchy``.
            reorder_for_hierarchy: reorder the rows to produce a hierarchible Index from the selected columns, assuming hierarchability is possible.

        Returns:
            :obj:`Frame`
        '''
        if isinstance(columns, tuple):
            column_loc = list(columns)
            column_name = columns
        else:
            column_loc = columns
            column_name = None # could be a slice, must get post iloc conversion

        column_iloc = self._columns._loc_to_iloc(column_loc)

        if column_name is None:
            column_name = tuple(self._columns.values[column_iloc])

        # index_labels = self._blocks._extract_array(column_key=column_iloc)
        index_labels = self._blocks._extract(column_key=column_iloc)

        if reorder_for_hierarchy:
            index, order_lex = rehierarch_from_type_blocks(
                    labels=index_labels,
                    depth_map=range(index_labels.shape[1]), # keep order
                    index_cls=IndexHierarchy,
                    index_constructors=index_constructors,
                    name=column_name,
                    )
            blocks_src = self._blocks._extract(row_key=order_lex)
        else:
            index = IndexHierarchy._from_type_blocks(
                    index_labels,
                    index_constructors=index_constructors,
                    name=column_name,
                    own_blocks=True,
                    )
            blocks_src = self._blocks


        if drop:
            blocks = TypeBlocks.from_blocks(
                    blocks_src._drop_blocks(column_key=column_iloc))
            columns = self._columns._drop_iloc(column_iloc)
            own_data = True
            own_columns = True
        else:
            blocks = blocks_src
            columns = self._columns
            own_data = False
            own_columns = False


        return self.__class__(blocks,
                columns=columns,
                index=index,
                own_data=own_data,
                own_columns=own_columns,
                own_index=True,
                name=self._name
                )

    def unset_index(self, *,
            names: tp.Iterable[tp.Hashable] = EMPTY_TUPLE,
            # index_column_first: tp.Optional[tp.Union[int, str]] = 0,
            consolidate_blocks: bool = False
            ) -> 'Frame':
        '''
        Return a new :obj:`Frame where the index is added to the front of the data, and an :obj:`IndexAutoFactory` is used to populate a new index. If the :obj:`Index` has a ``name``, that name will be used for the column name, otherwise a suitable default will be used. As underlying NumPy arrays are immutable, data is not copied.

        Args:
            names: An iterable of hashables to be used to name the unset index. If an ``Index``, a single hashable should be provided; if an ``IndexHierarchy``, as many hashables as the depth must be provided.
        '''
        from static_frame.core.index_level import IndexLevel

        def blocks() -> tp.Iterator[np.ndarray]:
            if self._index.ndim == 1:
                yield self._index.values
            else:
                if self._index._recache:
                    self._index._update_array_cache()
                yield from self._index._blocks._blocks
            for b in self._blocks._blocks:
                yield b

        if consolidate_blocks:
            block_gen = lambda: TypeBlocks.consolidate_blocks(blocks())
        else:
            block_gen = blocks

        if not names:
            names = self._index.names
        names_t = zip(*names)

        # self._columns._blocks may be None until array cache is updated.
        if self._columns._recache:
            self._columns._update_array_cache()

        if self._columns.depth > 1:
            column_blocks = self._columns._blocks._blocks
            column_blocks_new = tuple(
                    concat_resolved((np.array([name]), block[np.newaxis]), axis=1).T
                    for name, block in zip(names_t, column_blocks)
                    )
            column_type_blocks = TypeBlocks.from_blocks(column_blocks_new)
            columns = self._COLUMNS_HIERARCHY_CONSTRUCTOR._from_type_blocks(
                    column_type_blocks,
                    #index_constructors=index_constructors,
                    own_blocks=True,
                    )
        else:
            columns = chain(names, self._columns.values)

        return self.__class__(
                TypeBlocks.from_blocks(block_gen()),
                columns=columns,
                index=None,
                own_data=True,
                )

    def __round__(self, decimals: int = 0) -> 'Frame':
        '''
        Return a :obj:`Frame` rounded to the given decimals. Negative decimals round to the left of the decimal point.

        Args:
            decimals: number of decimals to round to.

        Returns:
            :obj:`Frame`
        '''
        return self.__class__(
                self._blocks.__round__(decimals=decimals),
                index=self._index,
                columns=self._columns,
                name=self._name,
                own_data=True,
                own_index=True,
                )

    def roll(self,
            index: int = 0,
            columns: int = 0,
            *,
            include_index: bool = False,
            include_columns: bool = False) -> 'Frame':
        '''
        Roll columns and/or rows by positive or negative integer counts, where columns and/or rows roll around the axis.

        Args:
            include_index: Determine if index is included in index-wise rotation.
            include_columns: Determine if column index is included in index-wise rotation.
        '''
        shift_index = index
        shift_column = columns

        blocks = TypeBlocks.from_blocks(
                self._blocks._shift_blocks(
                row_shift=shift_index,
                column_shift=shift_column,
                wrap=True
                ))

        if include_index:
            index = self._index.roll(shift_index)
            own_index = True
        else:
            index = self._index
            own_index = False

        if include_columns:
            columns = self._columns.roll(shift_column)
            own_columns = True
        else:
            columns = self._columns
            own_columns = False

        return self.__class__(blocks,
                columns=columns,
                index=index,
                name=self._name,
                own_data=True,
                own_columns=own_columns,
                own_index=own_index,
                )

    def shift(self,
            index: int = 0,
            columns: int = 0,
            *,
            fill_value: tp.Any = np.nan) -> 'Frame':
        '''
        Shift columns and/or rows by positive or negative integer counts, where columns and/or rows fall of the axis and introduce missing values, filled by `fill_value`.
        '''

        shift_index = index
        shift_column = columns

        blocks = TypeBlocks.from_blocks(
                self._blocks._shift_blocks(
                row_shift=shift_index,
                column_shift=shift_column,
                wrap=False,
                fill_value=fill_value
                ))

        return self.__class__(blocks,
                columns=self._columns,
                index=self._index,
                name=self._name,
                own_data=True,
                )

    #---------------------------------------------------------------------------
    # ranking transformations resulting in the same dimensionality
    # NOTE: this could be implemented on TypeBlocks, but handling missing values requires using indices, and is thus better handled at the Frame level

    def _rank(self, *,
            method: RankMethod,
            axis: int = 0, # 0 ranks columns, 1 ranks rows
            skipna: bool = True,
            ascending: BoolOrBools = True,
            start: int = 0,
            fill_value: tp.Any = np.nan,
    ) -> 'Frame':

        shape = self._blocks._shape
        asc_is_element = isinstance(ascending, BOOL_TYPES)

        if not asc_is_element:
            ascending = tuple(ascending)
            opposite_axis = int(not axis)
            if len(ascending) != shape[opposite_axis]:
                raise RuntimeError(f'Multiple ascending values must match length of axis {opposite_axis}.')

        def array_iter() -> tp.Iterator[np.ndarray]:
            for idx, array in enumerate(self._blocks.axis_values(axis=axis)):
                asc = ascending if asc_is_element else ascending[idx]
                if not skipna or array.dtype.kind not in DTYPE_NA_KINDS:
                    yield rank_1d(array,
                            method=method,
                            ascending=asc,
                            start=start,
                            )
                else:
                    index = self._index if axis == 0 else self._columns
                    s = Series(array, index=index, own_index=True)
                    # skipna is True
                    yield s._rank(method=method,
                            skipna=skipna,
                            ascending=asc,
                            start=start,
                            fill_value=fill_value,
                            ).values
        if axis == 0:
            # array_iter returns blocks
            blocks = TypeBlocks.from_blocks(array_iter())
        elif axis == 1:
            # create one array of type int or float
            arrays = list(array_iter())
            dtype = resolve_dtype_iter(a.dtype for a in arrays)
            block = np.empty(shape, dtype=dtype)
            for i, a in enumerate(arrays):
                block[i] = a
            block.flags.writeable = False
            blocks = TypeBlocks.from_blocks(block)
        else:
            raise AxisInvalid()

        return self.__class__(blocks,
                columns=self._columns,
                index=self._index,
                name=self._name,
                own_data=True,
                own_index=True,
                own_columns=self.STATIC,
                )

    @doc_inject(selector='rank')
    def rank_ordinal(self, *,
            axis: int = 0,
            skipna: bool = True,
            ascending: BoolOrBools = True,
            start: int = 0,
            fill_value: tp.Any = np.nan,
            ) -> 'Frame':
        '''Rank values distinctly, where ties get distinct values that maintain their ordering, and ranks are contiguous unique integers.

        Args:
            {axis}
            {skipna}
            {ascendings}
            {start}
            {fill_value}

        Returns:
            :obj:`Series`
        '''
        return self._rank(
                method=RankMethod.ORDINAL,
                axis=axis,
                skipna=skipna,
                ascending=ascending,
                start=start,
                fill_value=fill_value,
                )

    @doc_inject(selector='rank')
    def rank_dense(self, *,
            axis: int = 0,
            skipna: bool = True,
            ascending: BoolOrBools = True,
            start: int = 0,
            fill_value: tp.Any = np.nan,
            ) -> 'Frame':
        '''Rank values as compactly as possible, where ties get the same value, and ranks are contiguous (potentially non-unique) integers.

        Args:
            {axis}
            {skipna}
            {ascendings}
            {start}
            {fill_value}

        Returns:
            :obj:`Frame`
        '''
        return self._rank(
                method=RankMethod.DENSE,
                axis=axis,
                skipna=skipna,
                ascending=ascending,
                start=start,
                fill_value=fill_value,
                )

    @doc_inject(selector='rank')
    def rank_min(self, *,
            axis: int = 0,
            skipna: bool = True,
            ascending: BoolOrBools = True,
            start: int = 0,
            fill_value: tp.Any = np.nan,
            ) -> 'Frame':
        '''Rank values where tied values are assigned the minimum ordinal rank; ranks are potentially non-contiguous and non-unique integers.

        Args:
            {axis}
            {skipna}
            {ascendings}
            {start}
            {fill_value}

        Returns:
            :obj:`Frame`
        '''
        return self._rank(
                method=RankMethod.MIN,
                axis=axis,
                skipna=skipna,
                ascending=ascending,
                start=start,
                fill_value=fill_value,
                )

    @doc_inject(selector='rank')
    def rank_max(self, *,
            axis: int = 0,
            skipna: bool = True,
            ascending: BoolOrBools = True,
            start: int = 0,
            fill_value: tp.Any = np.nan,
            ) -> 'Frame':
        '''Rank values where tied values are assigned the maximum ordinal rank; ranks are potentially non-contiguous and non-unique integers.

        Args:
            {axis}
            {skipna}
            {ascendings}
            {start}
            {fill_value}

        Returns:
            :obj:`Frame`
        '''
        return self._rank(
                method=RankMethod.MAX,
                axis=axis,
                skipna=skipna,
                ascending=ascending,
                start=start,
                fill_value=fill_value,
                )

    @doc_inject(selector='rank')
    def rank_mean(self, *,
            axis: int = 0,
            skipna: bool = True,
            ascending: BoolOrBools = True,
            start: int = 0,
            fill_value: tp.Any = np.nan,
            ) -> 'Frame':
        '''Rank values where tied values are assigned the mean of the ordinal ranks; ranks are potentially non-contiguous and non-unique floats.

        Args:
            {axis}
            {skipna}
            {ascendings}
            {start}
            {fill_value}

        Returns:
            :obj:`Frame`
        '''
        return self._rank(
                method=RankMethod.MEAN,
                axis=axis,
                skipna=skipna,
                ascending=ascending,
                start=start,
                fill_value=fill_value,
                )



    #---------------------------------------------------------------------------
    # transformations resulting in changed dimensionality

    @doc_inject(selector='head', class_name='Frame')
    def head(self, count: int = 5) -> 'Frame':
        '''{doc}

        Args:
            {count}
        '''
        return self.iloc[:count]

    @doc_inject(selector='tail', class_name='Frame')
    def tail(self, count: int = 5) -> 'Frame':
        '''{doc}

        Args:
            {count}
        '''
        return self.iloc[-count:]


    def count(self, *,
            skipna: bool = True,
            skipfalsy: bool = False,
            unique: bool = False,
            axis: int = 0,
            ) -> Series:
        '''
        Return the count of non-NA values along the provided ``axis``, where 0 provides counts per column, 1 provides counts per row.

        Args:
            axis
        '''
        if not skipna and skipfalsy:
            raise RuntimeError('Cannot skipfalsy and not skipna.')

        labels = self._columns if axis == 0 else self._index

        if not skipna and not skipfalsy and not unique:
            array = np.full(len(labels),
                    self._blocks._shape[axis],
                    dtype=DTYPE_INT_DEFAULT,
                    )
        else:
            array = np.empty(len(labels), dtype=DTYPE_INT_DEFAULT)
            for i, values in enumerate(self._blocks.axis_values(axis=axis)):
                valid: tp.Optional[np.ndarray] = None

                if skipfalsy: # always includes skipna
                    valid = ~isfalsy_array(values)
                elif skipna: # NOTE: elif, as skipfalsy incldues skipna
                    valid = ~isna_array(values)

                if unique and valid is None:
                    array[i] = len(ufunc_unique(values))
                elif unique and valid is not None: # valid is a Boolean array
                    array[i] = len(ufunc_unique(values[valid]))
                elif not unique and valid is not None:
                    array[i] = valid.sum()
                else: # not unique, valid is None, means no removals, handled above
                    raise NotImplementedError()

        array.flags.writeable = False
        return Series(array, index=labels)

    @doc_inject(selector='sample')
    def sample(self,
            index: tp.Optional[int] = None,
            columns: tp.Optional[int] = None,
            *,
            seed: tp.Optional[int] = None,
            ) -> 'Frame':
        '''
        {doc}

        Args:
            {index}
            {columns}
            {seed}
        '''
        if index is not None:
            index, index_key = self._index._sample_and_key(count=index, seed=seed)
            own_index = True
        else:
            index = self._index
            index_key = None
            own_index = True

        if columns is not None:
            columns, columns_key = self._columns._sample_and_key(count=columns, seed=seed)
            own_columns = True
        else:
            columns = self._columns
            columns_key = None
            own_columns = False # might be GO

        if index_key is not None or columns_key is not None:
            blocks = self._blocks._extract(row_key=index_key, column_key=columns_key)
        else:
            blocks = self._blocks.copy()

        return self.__class__(blocks,
                columns=columns,
                index=index,
                name=self._name,
                own_data=True,
                own_index=own_index,
                own_columns=own_columns,
                )

    @doc_inject(selector='argminmax')
    def loc_min(self, *,
            skipna: bool = True,
            axis: int = 0
            ) -> Series:
        '''
        Return the labels corresponding to the minimum value found.

        Args:
            {skipna}
            {axis}
        '''
        # this operation is not composable for axis 1; cannot use _ufunc_axis_skipna interface as do not have out argument, and need to determine returned dtype in advance

        # if this has NaN we cannot get a loc
        post = argmin_2d(self.values, skipna=skipna, axis=axis)
        if post.dtype == DTYPE_FLOAT_DEFAULT:
            raise RuntimeError('cannot produce loc representation from NaNs')

        # post has been made immutable so Series will own
        if axis == 0:
            return Series(
                    self.index.values[post],
                    index=immutable_index_filter(self._columns)
                    )
        return Series(self.columns.values[post], index=self._index)

    @doc_inject(selector='argminmax')
    def iloc_min(self, *,
            skipna: bool = True,
            axis: int = 0
            ) -> Series:
        '''
        Return the integer indices corresponding to the minimum values found.

        Args:
            {skipna}
            {axis}
        '''
        # if this has NaN can continue
        post = argmin_2d(self.values, skipna=skipna, axis=axis)
        post.flags.writeable = False
        if axis == 0:
            return Series(post, index=immutable_index_filter(self._columns))
        return Series(post, index=self._index)

    @doc_inject(selector='argminmax')
    def loc_max(self, *,
            skipna: bool = True,
            axis: int = 0
            ) -> Series:
        '''
        Return the labels corresponding to the maximum values found.

        Args:
            {skipna}
            {axis}
        '''
        # if this has NaN we cannot get a loc
        post = argmax_2d(self.values, skipna=skipna, axis=axis)
        if post.dtype == DTYPE_FLOAT_DEFAULT:
            raise RuntimeError('cannot produce loc representation from NaNs')

        if axis == 0:
            return Series(
                    self.index.values[post],
                    index=immutable_index_filter(self._columns)
                    )
        return Series(self.columns.values[post], index=self._index)

    @doc_inject(selector='argminmax')
    def iloc_max(self, *,
            skipna: bool = True,
            axis: int = 0
            ) -> Series:
        '''
        Return the integer indices corresponding to the maximum values found.

        Args:
            {skipna}
            {axis}
        '''
        # if this has NaN can continue
        post = argmax_2d(self.values, skipna=skipna, axis=axis)
        post.flags.writeable = False
        if axis == 0:
            return Series(post, index=immutable_index_filter(self._columns))
        return Series(post, index=self._index)

    def cov(self, *,
            axis: int = 1,
            ddof: int = 1,
            ) -> 'Frame':
        '''Compute a covariance matrix.

        Args:
            axis: if 0, each row represents a variable, with observations as columns; if 1, each column represents a variable, with observations as rows. Defaults to 1.
            ddof: Delta degrees of freedom, defaults to 1.
        '''
        if axis == 0:
            rowvar = True
            labels = self._index
            own_index = True
            own_columns = self.STATIC
        else:
            rowvar = False
            labels = self._columns
            # can own columns if static
            own_index = self.STATIC
            own_columns = self.STATIC

        values = np.cov(self.values, rowvar=rowvar, ddof=ddof)
        values.flags.writeable = False

        return self.__class__(values,
                index=labels,
                columns=labels,
                own_index=own_index,
                own_columns=own_columns,
                name=self._name,
                )



    #---------------------------------------------------------------------------
    # pivot family

    def pivot(self,
            index_fields: KeyOrKeys,
            columns_fields: KeyOrKeys = EMPTY_TUPLE,
            data_fields: KeyOrKeys = EMPTY_TUPLE,
            *,
            func: CallableOrCallableMap = None,
            fill_value: object = np.nan,
            ) -> 'Frame':
        '''
        Produce a pivot table, where one or more columns is selected for each of index_fields, columns_fields, and data_fields. Unique values from the provided ``index_fields`` will be used to create a new index; unique values from the provided ``columns_fields`` will be used to create a new columns; if one ``data_fields`` value is selected, that is the value that will be displayed; if more than one values is given, those values will be presented with a hierarchical index on the columns; if ``data_fields`` is not provided, all unused fields will be displayed.

        Args:
            index_fields
            columns_fields
            data_fields
            fill_value: If the index expansion produces coordinates that have no existing data value, fill that position with this value.
            func: function to apply to ``data_fields``, or a dictionary of labelled functions to apply to data fields, producing an additional hierarchical level.
        '''
        func = np.nansum if func is None else func
        if callable(func):
            func_map = (('', func),) # store iterable of pairs
        else:
            func_map = tuple(func.items())

        func_single = func_map[0][1] if len(func_map) == 1 else None

        func_fields = EMPTY_TUPLE if func_single else tuple(label for label, _ in func_map)
        index_fields = key_normalize(index_fields)
        columns_fields = key_normalize(columns_fields)
        data_fields = key_normalize(data_fields)

        for field in chain(index_fields, columns_fields):
            if field not in self._columns:
                raise ErrorInitFrame(f'cannot create a pivot Frame from a field ({field}) that is not a column')
        if not data_fields:
            used = set(chain(index_fields, columns_fields))
            data_fields = [x for x in self.columns if x not in used]
            if not data_fields:
                raise ErrorInitFrame('no fields remain to populate data.')

        index_depth = len(index_fields)
        index_loc = index_fields if index_depth > 1 else index_fields[0]
        index_values = ufunc_unique(
                self._blocks._extract_array(
                        column_key=self._columns._loc_to_iloc(index_loc)),
                axis=0)

        # index_inner is used for avoiding dealing with IndexHierarchy
        if index_depth == 1:
            index = Index(index_values, name=index_fields[0])
            index_inner = index
        else:
            index = IndexHierarchy.from_labels(index_values, name=tuple(index_fields))
            index_inner = index.flat() # insure we have the right order of tuples

        # For data fields, we add the field name, not the field values, to the columns.
        columns_name = tuple(columns_fields)
        if len(data_fields) > 1 or not columns_fields: # if no columns_fields, have to add values label
            columns_name = tuple(chain(*columns_fields, ('values',)))
        if len(func_map) > 1:
            columns_name = columns_name + ('func',)

        columns_depth = len(columns_name)
        if columns_depth == 1:
            columns_name = columns_name[0]
            columns_constructor = partial(self._COLUMNS_CONSTRUCTOR, name=columns_name)
        else:
            columns_constructor = partial(self._COLUMNS_HIERARCHY_CONSTRUCTOR.from_labels,
                    depth_reference=columns_depth,
                    name=columns_name)

        if not columns_fields: # group by is index_fields
            group_fields = index_fields if index_depth > 1 else index_fields[0]
            columns = data_fields if func_single else tuple(product(data_fields, func_fields))
            index_constructor = None if index_depth > 1 else partial(Index, name=index_fields[0])

            if len(columns) == 1:
                f = self.from_series(
                        Series.from_items(
                                pivot_items(frame=self,
                                        group_fields=group_fields,
                                        group_depth=np.inf,
                                        data_fields=data_fields,
                                        func_single=func_single,
                                        ),
                                name=columns[0],
                                index_constructor=index_constructor),
                        columns_constructor=columns_constructor)
            else:
                dtypes = tuple(pivot_records_dtypes(
                                frame=self,
                                data_fields=data_fields,
                                func_single=func_single,
                                func_map=func_map,
                                ))
                f = self.from_records_items(
                        pivot_records_items(
                                frame=self,
                                group_fields=group_fields,
                                group_depth=np.inf, # avoid reducing labels
                                data_fields=data_fields,
                                func_single=func_single,
                                func_map=func_map,
                        ),
                        columns_constructor=columns_constructor,
                        columns=columns,
                        index_constructor=index_constructor,
                        dtypes=dtypes,
                        )
            # if we have an IH, we will relabel with that IH, and might have a different order than the order here; thus, reindex. This is not observed with the present implementation of iter_group_items, but that might change.
            if index_depth > 1 and not f.index.equals(index_inner):
                f = f.reindex(index_inner, own_index=True, check_equals=False) #pragma: no cover
        else:
            # collect subframes based on an index of tuples and columns of tuples (if depth > 1)
            index_fields_len = len(index_fields)
            sub_frames = []
            sub_columns_collected = []

            # import ipdb; ipdb.set_trace()
            # avoid doing a multi-column-style selection if not needed
            if len(columns_fields) == 1:
                columns_group = columns_fields[0]
                retuple_group_label = True
            else:
                columns_group = columns_fields
                retuple_group_label = False

            for group, sub in self.iter_group_items(columns_group):
                if index_fields_len == 1:
                    sub_index_labels = sub._blocks._extract_array(row_key=None,
                            column_key=sub.columns._loc_to_iloc(index_fields[0]))
                else: # match to an index of tuples; the order might not be the same as IH
                    sub_index_labels = tuple(zip(*(
                            sub._blocks._extract_array(row_key=None,
                                    column_key=sub.columns._loc_to_iloc(f))
                            for f in index_fields)))

                sub_columns = extrapolate_column_fields(
                        columns_fields,
                        group if not retuple_group_label else (group,),
                        data_fields,
                        func_fields)
                sub_columns_collected.extend(sub_columns)

                # if sub_index_labels are not unique we need to aggregate
                if len(set(sub_index_labels)) != len(sub_index_labels):
                    if len(sub_columns) == 1:
                        sub_frame = Frame.from_series(
                                Series.from_items(
                                        pivot_items(frame=sub,
                                                group_fields=index_fields,
                                                group_depth=index_depth,
                                                data_fields=data_fields,
                                                func_single=func_single,
                                                ),
                                        ))
                    else:
                        dtypes = tuple(pivot_records_dtypes(
                                frame=self,
                                data_fields=data_fields,
                                func_single=func_single,
                                func_map=func_map,
                                ))
                        sub_frame = Frame.from_records_items(
                                pivot_records_items(
                                        frame=sub,
                                        group_fields=index_fields,
                                        group_depth=index_depth,
                                        data_fields=data_fields,
                                        func_single=func_single,
                                        func_map=func_map),
                                dtypes=dtypes,
                                )
                else:
                    if func_single: # assume no aggregation necessary
                        if len(data_fields) == 1:
                            data_fields_iloc = sub.columns._loc_to_iloc(data_fields[0])
                        else:
                            data_fields_iloc = sub.columns._loc_to_iloc(data_fields)
                        sub_frame = Frame(
                                sub._blocks._extract(row_key=None,
                                        column_key=data_fields_iloc),
                                index=sub_index_labels,
                                own_data=True)
                    else:
                        def blocks() -> tp.Iterator[np.ndarray]:
                            for field in data_fields:
                                for _, func in func_map:
                                    yield sub._blocks._extract_array(row_key=None,
                                            column_key=sub.columns._loc_to_iloc(field))
                        sub_frame = Frame(
                                TypeBlocks.from_blocks(blocks()),
                                index=sub_index_labels,
                                own_data=True,
                                )
                sub_frames.append(sub_frame)

            f = self.__class__.from_concat(sub_frames,
                    index=index_inner,
                    columns=sub_columns_collected,
                    axis=1,
                    fill_value=fill_value)

        index_final = None if index_depth == 1 else index

        # have to rename columns if derived in from_concat
        columns_final = (f.columns.rename(columns_name) if columns_depth == 1
                else columns_constructor(f.columns))

        f = f.relabel(index=index_final, columns=columns_final)

        return f


    #---------------------------------------------------------------------------
    # pivot stack, unstack

    def pivot_stack(self,
            depth_level: DepthLevelSpecifier = -1,
            *,
            fill_value: object = np.nan,
            ) -> 'Frame':
        '''
        Move labels from the columns to the index, creating or extending an :obj:`IndexHierarchy` on the index.

        Args:
            depth_level: selection of columns depth or depth to move onto the index.
        '''
        values_src = self._blocks
        index_src = self.index
        columns_src = self.columns
        dtype_fill = np.array(fill_value).dtype
        dtypes_src = self.dtypes.values

        pim = pivot_index_map(
                index_src=columns_src,
                depth_level=depth_level,
                dtypes_src=dtypes_src,
                )
        targets_unique = pim.targets_unique
        group_to_target_map = pim.group_to_target_map
        group_to_dtype = pim.group_to_dtype

        pdc = pivot_derive_constructors(
                contract_src=columns_src,
                expand_src=index_src,
                group_select=pim.group_select,
                group_depth=pim.group_depth,
                target_select=pim.target_select,
                group_to_target_map=group_to_target_map,
                expand_is_columns=False,
                frame_cls=self.__class__,
                )

        group_has_fill = {g: False for g in group_to_target_map}

        # We produce the resultant frame by iterating over the source index labels (providing outer-most hierarchical levels), we then extend each label of that index with each unique "target", or new labels coming from the columns.
        def records_items() -> tp.Iterator[tp.Tuple[tp.Hashable, tp.Sequence[tp.Any]]]:
            for row_idx, outer in enumerate(index_src): # iter tuple or label
                for target in targets_unique: # target is always a tuple
                    # derive the new index
                    if index_src.depth == 1:
                        key = (outer,) + target
                    else:
                        key = outer + target
                    record = []
                    # this is equivalent to iterating over the new columns to get a row of data
                    for group, target_map in group_to_target_map.items():
                        if target in target_map:
                            col_idx = target_map[target]
                            record.append(values_src._extract(row_idx, col_idx))
                        else:
                            record.append(fill_value)
                            group_has_fill[group] = True
                    yield key, record

        # NOTE: this is a generator to defer evaluation until after records_items() is run, whereby group_has_fill is populated
        def dtypes() -> tp.Iterator[np.dtype]:
            for g, dtype in group_to_dtype.items():
                if group_has_fill[g]:
                    yield resolve_dtype(dtype, dtype_fill)
                else:
                    yield dtype

        return self.from_records_items(
                records_items(),
                index_constructor=pdc.expand_constructor,
                columns=pdc.contract_dst,
                columns_constructor=pdc.contract_constructor,
                name=self.name,
                dtypes=dtypes(),
                )


    def pivot_unstack(self,
            depth_level: DepthLevelSpecifier = -1,
            *,
            fill_value: object = np.nan,
            ) -> 'Frame':
        '''
        Move labels from the index to the columns, creating or extending an :obj:`IndexHierarchy` on the columns.

        Args:
            depth_level: selection of index depth or depth to move onto the columns.
        '''
        values_src = self._blocks
        index_src = self.index
        columns_src = self.columns

        dtype_fill = np.array(fill_value).dtype
        dtypes_src = self.dtypes.values # dtypes need to be "exploded" into new columns

        # We produce the resultant frame by iterating over the source index labels (providing outer-most hierarchical levels), we then extend each label of that index with each unique "target", or new labels coming from the columns.

        pim = pivot_index_map(
                index_src=index_src,
                depth_level=depth_level,
                dtypes_src=None,
                )
        targets_unique = pim.targets_unique
        group_to_target_map = pim.group_to_target_map

        pdc = pivot_derive_constructors(
                contract_src=index_src,
                expand_src=columns_src,
                group_select=pim.group_select,
                group_depth=pim.group_depth,
                target_select=pim.target_select,
                group_to_target_map=group_to_target_map,
                expand_is_columns=True,
                frame_cls=self.__class__,
                )

        def items() -> tp.Iterator[tp.Tuple[tp.Hashable, np.ndarray]]:
            for col_idx, outer in enumerate(columns_src): # iter tuple or label
                dtype_src_col = dtypes_src[col_idx]

                for target in targets_unique: # target is always a tuple
                    if columns_src.depth == 1:
                        key = (outer,) + target
                    else:
                        key = outer + target
                    # cannot allocate array as do not know dtype until after fill_value
                    values = []
                    for group, target_map in group_to_target_map.items():
                        if target in target_map:
                            row_idx = target_map[target]
                            dtype = dtype_src_col
                            values.append(values_src._extract(row_idx, col_idx))
                        else:
                            values.append(fill_value)
                            dtype = resolve_dtype(dtype_src_col, dtype_fill)

                    array = np.array(values, dtype=dtype)
                    array.flags.writeable = False
                    yield key, array

        return self.from_items(
                items(),
                index=pdc.contract_dst,
                index_constructor=pdc.contract_constructor,
                columns_constructor=pdc.expand_constructor,
                name=self.name,
                )

    #---------------------------------------------------------------------------
    def _join(self,
            other: 'Frame', # support a named Series as a 1D frame?
            *,
            join_type: Join, # intersect, left, right, union,
            left_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            left_columns: GetItemKeyType = None,
            right_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            right_columns: GetItemKeyType = None,
            left_template: str = '{}',
            right_template: str = '{}',
            fill_value: tp.Any = np.nan,
            composite_index: bool = True,
            composite_index_fill_value: tp.Hashable = None,
            ) -> 'Frame':

        left_index = self.index
        right_index = other.index

        #-----------------------------------------------------------------------
        # find matches

        if left_depth_level is None and left_columns is None:
            raise RuntimeError('Must specify one or both of left_depth_level and left_columns.')
        if right_depth_level is None and right_columns is None:
            raise RuntimeError('Must specify one or both of right_depth_level and right_columns.')

        # for now we reduce the targets to arrays; possible coercion in some cases, but seems inevitable as we will be doing row-wise comparisons
        target_left = TypeBlocks.from_blocks(
                arrays_from_index_frame(self, left_depth_level, left_columns)).values
        target_right = TypeBlocks.from_blocks(
                arrays_from_index_frame(other, right_depth_level, right_columns)).values

        if target_left.shape[1] != target_right.shape[1]:
            raise RuntimeError('left and right selections must be the same width.')

        # Find matching pairs. Get iloc of left to iloc of right.
        # If composite_index is True, is_many is True, if False, need to check if it is possible to not havea composite index.
        is_many = composite_index # one to many or many to many

        map_iloc = {}
        seen = set()

        # NOTE: this could be optimized by always iterating over the shorter target
        for idx_left, row_left in enumerate(target_left):
            # Get 1D vector showing matches along right's full heigh
            matched = row_left == target_right
            if matched is False:
                continue
            matched = matched.all(axis=1)
            if not matched.any():
                continue
            matched_idx = np.flatnonzero(matched)
            if not is_many:
                if len(matched_idx) > 1:
                    is_many = True
                elif len(matched_idx) == 1:
                    if matched_idx[0] in seen:
                        is_many = True
                    seen.add(matched_idx[0])

            map_iloc[idx_left] = matched_idx

        if not composite_index and is_many:
            raise RuntimeError('A composite index is required in this join.')

        #-----------------------------------------------------------------------
        # store collections of matches, derive final index

        left_loc_set = set()
        right_loc_set = set()
        many_loc = []
        many_iloc = []

        cifv = composite_index_fill_value

        # NOTE: doing selection and using iteration (from set, and with zip, below) reduces chances for type coercion in IndexHierarchy
        left_loc = left_index[list(map_iloc.keys())]

        for (k, v), left_loc_element in zip(map_iloc.items(), left_loc):
            left_loc_set.add(left_loc_element)
            # right at v is an array
            right_loc_part = right_index[v] # iloc to loc
            right_loc_set.update(right_loc_part)

            if is_many:
                many_loc.extend(Pair(p) for p in product((left_loc_element,), right_loc_part))
                many_iloc.extend(Pair(p) for p in product((k,), v))

        if join_type is Join.INNER:
            if is_many:
                final_index = Index(many_loc)
            else: # just those matched from the left, which are also on right
                final_index = Index(left_loc)
        elif join_type is Join.LEFT:
            if is_many:
                extend = (PairLeft((x, cifv))
                        for x in left_index if x not in left_loc_set)
                # What if we are extending an index that already has a tuple
                final_index = Index(chain(many_loc, extend))
            else:
                final_index = left_index
        elif join_type is Join.RIGHT:
            if is_many:
                extend = (PairRight((cifv, x))
                        for x in right_index if x not in right_loc_set)
                final_index = Index(chain(many_loc, extend))
            else:
                final_index = right_index
        elif join_type is Join.OUTER:
            extend_left = (PairLeft((x, cifv))
                    for x in left_index if x not in left_loc_set)
            extend_right = (PairRight((cifv, x))
                    for x in right_index if x not in right_loc_set)
            if is_many:
                # must revese the many_loc so as to preserent right id first
                final_index = Index(chain(many_loc, extend_left, extend_right))
            else:
                final_index = left_index.union(right_index)
        else:
            raise NotImplementedError(f'index source must be one of {tuple(Join)}')

        #-----------------------------------------------------------------------
        # construct final frame

        if not is_many:
            final = FrameGO(index=final_index)
            left_columns = (left_template.format(c) for c in self.columns)
            final.extend(self.relabel(columns=left_columns), fill_value=fill_value)
            # build up a Series for each new column
            for idx_col, col in enumerate(other.columns):
                values = []
                for loc in final_index:
                    # what if loc is in both left and rihgt?
                    if loc in left_index and left_index._loc_to_iloc(loc) in map_iloc:
                        iloc = map_iloc[left_index._loc_to_iloc(loc)]
                        assert len(iloc) == 1 # not is_many, so all have to be length 1
                        values.append(other.iloc[iloc[0], idx_col])
                    elif loc in right_index:
                        values.append(other.loc[loc, col])
                    else:
                        values.append(fill_value)
                final[right_template.format(col)] = values
            return final.to_frame()

        # From here, is_many is True
        row_key = []
        final_index_left = []
        for p in final_index:
            if p.__class__ is Pair: # in both
                iloc = left_index._loc_to_iloc(p[0])
                row_key.append(iloc)
                final_index_left.append(p)
            elif p.__class__ is PairLeft:
                row_key.append(left_index._loc_to_iloc(p[0]))
                final_index_left.append(p)

        # extract potentially repeated rows
        tb = self._blocks._extract(row_key=row_key)
        left_columns = (left_template.format(c) for c in self.columns)

        final = FrameGO(tb,
                index=Index(final_index_left),
                columns=left_columns,
                own_data=True)

        # only do this if we have PairRight above
        if len(final_index_left) < len(final_index):
            final = final.reindex(final_index, fill_value=fill_value)

        # populate from right columns
        for idx_col, col in enumerate(other.columns):
            values = []
            for pair in final_index:
                if isinstance(pair, Pair):
                    loc_left, loc_right = pair
                    if pair.__class__ is PairRight: # get from right
                        values.append(other.loc[loc_right, col])
                    elif pair.__class__ is PairLeft:
                        # get from left, but we do not have col, so fill value
                        values.append(fill_value)
                    else: # is this case needed?
                        values.append(other.loc[loc_right, col])
                else:
                    values.append(fill_value)
            final[right_template.format(col)] = values
        return final.to_frame()


    @doc_inject(selector='join')
    def join_inner(self,
            other: 'Frame', # support a named Series as a 1D frame?
            *,
            left_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            left_columns: GetItemKeyType = None,
            right_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            right_columns: GetItemKeyType = None,
            left_template: str = '{}',
            right_template: str = '{}',
            fill_value: tp.Any = np.nan,
            composite_index: bool = True,
            composite_index_fill_value: tp.Hashable = None,
            ) -> 'Frame':
        '''
        Perform an inner join.

        Args:
            {left_depth_level}
            {left_columns}
            {right_depth_level}
            {right_columns}
            {left_template}
            {right_template}
            {fill_value}
            {composite_index}
            {composite_index_fill_value}

        Returns:
            :obj:`Frame`
        '''
        return self._join(other=other,
                join_type=Join.INNER,
                left_depth_level=left_depth_level,
                left_columns=left_columns,
                right_depth_level=right_depth_level,
                right_columns=right_columns,
                left_template=left_template,
                right_template=right_template,
                fill_value=fill_value,
                composite_index=composite_index,
                composite_index_fill_value=composite_index_fill_value,
                )

    @doc_inject(selector='join')
    def join_left(self,
            other: 'Frame', # support a named Series as a 1D frame?
            *,
            left_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            left_columns: GetItemKeyType = None,
            right_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            right_columns: GetItemKeyType = None,
            left_template: str = '{}',
            right_template: str = '{}',
            fill_value: tp.Any = np.nan,
            composite_index: bool = True,
            composite_index_fill_value: tp.Hashable = None,
            ) -> 'Frame':
        '''
        Perform a left outer join.

        Args:
            {left_depth_level}
            {left_columns}
            {right_depth_level}
            {right_columns}
            {left_template}
            {right_template}
            {fill_value}
            {composite_index}
            {composite_index_fill_value}

        Returns:
            :obj:`Frame`
        '''
        return self._join(other=other,
                join_type=Join.LEFT,
                left_depth_level=left_depth_level,
                left_columns=left_columns,
                right_depth_level=right_depth_level,
                right_columns=right_columns,
                left_template=left_template,
                right_template=right_template,
                fill_value=fill_value,
                composite_index=composite_index,
                composite_index_fill_value=composite_index_fill_value,
                )

    @doc_inject(selector='join')
    def join_right(self,
            other: 'Frame', # support a named Series as a 1D frame?
            *,
            left_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            left_columns: GetItemKeyType = None,
            right_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            right_columns: GetItemKeyType = None,
            left_template: str = '{}',
            right_template: str = '{}',
            fill_value: tp.Any = np.nan,
            composite_index: bool = True,
            composite_index_fill_value: tp.Hashable = None,
            ) -> 'Frame':
        '''
        Perform a right outer join.

        Args:
            {left_depth_level}
            {left_columns}
            {right_depth_level}
            {right_columns}
            {left_template}
            {right_template}
            {fill_value}
            {composite_index}
            {composite_index_fill_value}

        Returns:
            :obj:`Frame`
        '''
        return self._join(other=other,
                join_type=Join.RIGHT,
                left_depth_level=left_depth_level,
                left_columns=left_columns,
                right_depth_level=right_depth_level,
                right_columns=right_columns,
                left_template=left_template,
                right_template=right_template,
                fill_value=fill_value,
                composite_index=composite_index,
                composite_index_fill_value=composite_index_fill_value,
                )

    @doc_inject(selector='join')
    def join_outer(self,
            other: 'Frame', # support a named Series as a 1D frame?
            *,
            left_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            left_columns: GetItemKeyType = None,
            right_depth_level: tp.Optional[DepthLevelSpecifier] = None,
            right_columns: GetItemKeyType = None,
            left_template: str = '{}',
            right_template: str = '{}',
            fill_value: tp.Any = np.nan,
            composite_index: bool = True,
            composite_index_fill_value: tp.Hashable = None,
            ) -> 'Frame':
        '''
        Perform an outer join.

        Args:
            {left_depth_level}
            {left_columns}
            {right_depth_level}
            {right_columns}
            {left_template}
            {right_template}
            {fill_value}
            {composite_index}
            {composite_index_fill_value}

        Returns:
            :obj:`Frame`
        '''
        return self._join(other=other,
                join_type=Join.OUTER,
                left_depth_level=left_depth_level,
                left_columns=left_columns,
                right_depth_level=right_depth_level,
                right_columns=right_columns,
                left_template=left_template,
                right_template=right_template,
                fill_value=fill_value,
                composite_index=composite_index,
                composite_index_fill_value=composite_index_fill_value,
                )

    #---------------------------------------------------------------------------
    def _insert(self,
            key: int, # iloc positions
            container: tp.Union['Frame', Series],
            *,
            fill_value: tp.Any = np.nan,
            ) -> 'Frame':
        '''
        Return a new Frame with the provided container inserted at the position determined by the column key; values existing at that key come after the inserted container.

        NOTE: At this time we do not accept elements or unlabelled iterables, as our interface does not permit supplying the required new column names with those arguments.
        '''
        if not isinstance(container, (Series, Frame)):
            raise NotImplementedError(
                    f'No support for inserting with {type(container)}')

        if not len(container.index): # must be empty data, empty index container
            return self if self.STATIC else self.__class__(self)

        # self's index will never change; we only take what aligns in the passed container
        if not self._index.equals(container._index):
            container = container.reindex(self._index,
                    fill_value=fill_value,
                    check_equals=False,
                    )

        # NOTE: might introduce coercions in IndexHierarchy
        labels_prior = self._columns.values

        if isinstance(container, Frame):
            if not len(container.columns):
                return self if self.STATIC else self.__class__(self)

            labels_insert = container.columns.__iter__()
            blocks_insert = container._blocks._blocks

        elif isinstance(container, Series):
            labels_insert = (container.name,)
            blocks_insert = (container.values,)

        columns = self._columns.__class__.from_labels(chain(
                labels_prior[:key],
                labels_insert,
                labels_prior[key:],
                ))

        blocks = TypeBlocks.from_blocks(chain(
                self._blocks._slice_blocks(column_key=slice(0, key)),
                blocks_insert,
                self._blocks._slice_blocks(column_key=slice(key, None)),
                ))

        return self.__class__(blocks,
                index=self._index,
                columns=columns,
                name=self._name,
                own_data=True,
                own_columns=True,
                own_index=True,
                )

    @doc_inject(selector='insert')
    def insert_before(self,
            key: tp.Hashable,
            container: tp.Union['Frame', Series],
            *,
            fill_value: tp.Any = np.nan,
            ) -> 'Frame':
        '''
        Create a new :obj:`Frame` by inserting a named :obj:`Series` or :obj:`Frame` at the position before the label specified by ``key``.

        Args:
            {key_before}
            {container}
            {fill_value}

        Returns:
            :obj:`Frame`
        '''
        iloc_key = self._columns._loc_to_iloc(key)
        if not isinstance(iloc_key, INT_TYPES):
            raise RuntimeError(f'Unsupported key type: {key}')
        return self._insert(iloc_key, container, fill_value=fill_value)

    @doc_inject(selector='insert')
    def insert_after(self,
            key: tp.Hashable,
            container: tp.Union['Frame', Series],
            *,
            fill_value: tp.Any = np.nan,
            ) -> 'Frame':
        '''
        Create a new :obj:`Frame` by inserting a named :obj:`Series` or :obj:`Frame` at the position after the label specified by ``key``.

        Args:
            {key_after}
            {container}
            {fill_value}

        Returns:
            :obj:`Frame`
        '''
        iloc_key = self._columns._loc_to_iloc(key)
        if not isinstance(iloc_key, INT_TYPES):
            raise RuntimeError(f'Unsupported key type: {key}')
        return self._insert(iloc_key + 1, container, fill_value=fill_value)

    #---------------------------------------------------------------------------
    # utility function to numpy array or other types

    @doc_inject()
    def equals(self,
            other: tp.Any,
            *,
            compare_name: bool = False,
            compare_dtype: bool = False,
            compare_class: bool = False,
            skipna: bool = True,
            ) -> bool:
        '''
        {doc}

        Args:
            {compare_name}
            {compare_dtype}
            {compare_class}
            {skipna}
        '''
        if id(other) == id(self):
            return True

        if compare_class and self.__class__ != other.__class__:
            return False
        elif not isinstance(other, Frame):
            return False

        if self._blocks.shape != other._blocks.shape:
            return False
        if compare_name and self._name != other._name:
            return False

        # dtype check will happen in TypeBlocks
        if not self._blocks.equals(other._blocks,
                compare_dtype=compare_dtype,
                compare_class=compare_class,
                skipna=skipna,
                ):
            return False

        if not self._index.equals(other._index,
                compare_name=compare_name,
                compare_dtype=compare_dtype,
                compare_class=compare_class,
                skipna=skipna,
                ):
            return False

        if not self._columns.equals(other._columns,
                compare_name=compare_name,
                compare_dtype=compare_dtype,
                compare_class=compare_class,
                skipna=skipna,
                ):
            return False

        return True


    def unique(self, *,
            axis: tp.Optional[int] = None,
            ) -> np.ndarray:
        '''
        Return a NumPy array of unqiue values. If the axis argument is provied, uniqueness is determined by columns or row.
        '''
        return ufunc_unique(self.values, axis=axis)

    #---------------------------------------------------------------------------
    # exporters

    def to_pairs(self, axis: int = 0) -> tp.Iterable[
            tp.Tuple[tp.Hashable, tp.Iterable[tp.Tuple[tp.Hashable, tp.Any]]]]:
        '''
        Return a tuple of major axis key, minor axis key vlaue pairs, where major axis is determined by the axis argument.
        '''
        index_values = tuple(self._index)
        columns_values = tuple(self._columns)

        if axis == 1:
            major = index_values
            minor = columns_values
        elif axis == 0:
            major = columns_values
            minor = index_values
        else:
            raise AxisInvalid(f'invalid axis: {axis}')

        return tuple(
                zip(major, (tuple(zip(minor, v))
                for v in self._blocks.axis_values(axis))))

    def to_pandas(self) -> 'pandas.DataFrame':
        '''
        Return a Pandas DataFrame.
        '''
        import pandas

        if self._blocks.unified:
            # make copy to get writeable
            array = self._blocks._blocks[0].copy()
            df = pandas.DataFrame(array,
                    index=self._index.to_pandas(),
                    columns=self._columns.to_pandas()
                    )
        else:
            df = pandas.DataFrame(index=self._index.to_pandas())
            # use integer columns for initial loading, then replace
            # NOTE: alternative approach of trying to assign blocks (wrapped in a DF) is not faster than single column assignment
            for i, array in enumerate(self._blocks.axis_values(0)):
                df[i] = array

            df.columns = self._columns.to_pandas()

        if 'name' not in df.columns and self._name is not None:
            df.name = self._name
        return df

    def to_arrow(self,
            *,
            include_index: bool = True,
            include_index_name: bool = True,
            include_columns: bool = True,
            include_columns_name: bool = False,
            ) -> 'pyarrow.Table':
        '''
        Return a ``pyarrow.Table`` from this :obj:`Frame`.
        '''
        import pyarrow
        from static_frame.core.store import Store

        field_names, dtypes = Store.get_field_names_and_dtypes(
                frame=self,
                include_index=include_index,
                include_index_name=include_index_name,
                include_columns=include_columns,
                include_columns_name=include_columns_name,
                force_str_names=True,
                )

        def arrays() -> tp.Iterator[np.ndarray]:
            for array, dtype in zip(
                    Store.get_column_iterator(frame=self, include_index=include_index),
                    dtypes,
                    ):
                if (dtype.kind == DTYPE_DATETIME_KIND
                        and np.datetime_data(dtype)[0] not in DTU_PYARROW):
                    yield array.astype(DT64_NS)
                else:
                    yield array

        # field_names have to be strings
        return pyarrow.Table.from_arrays(tuple(arrays()), names=field_names)


    def to_parquet(self,
            fp: tp.Union[PathSpecifier, BytesIO],
            *,
            include_index: bool = True,
            include_index_name: bool = True,
            include_columns: bool = True,
            include_columns_name: bool = False,
            ) -> None:
        '''
        Write an Arrow Parquet binary file.
        '''
        import pyarrow.parquet as pq

        table = self.to_arrow(
                include_index=include_index,
                include_index_name=include_index_name,
                include_columns=include_columns,
                include_columns_name=include_columns_name,
                )
        fp = path_filter(fp)
        pq.write_table(table, fp)


    def to_msgpack(self) -> 'bin':
        '''
        Return a msgpack.
        '''
        import msgpack
        import msgpack_numpy

        def encode(obj: object,
                chain: tp.Callable[[np.ndarray], str] = msgpack_numpy.encode,
                ) -> dict: #returns dict that msgpack-python consumes
            cls = obj.__class__
            clsname = cls.__name__
            package = cls.__module__.split('.', 1)[0]

            if package == 'static_frame':
                if isinstance(obj, Frame):
                    return {b'sf':clsname,
                            b'name':obj.name,
                            b'blocks':packb(obj._blocks),
                            b'index':packb(obj.index),
                            b'columns':packb(obj.columns)}
                elif isinstance(obj, IndexHierarchy):
                    if obj._recache:
                        obj._update_array_cache()
                    return {b'sf':clsname,
                            b'name':obj.name,
                            b'index_constructors': packb([
                                    a.__name__ for a in obj.index_types.values.tolist()]),
                            b'blocks':packb(obj._blocks)}
                elif isinstance(obj, Index):
                    return {b'sf':clsname,
                            b'name':obj.name,
                            b'data':packb(obj.values)}
                elif isinstance(obj, TypeBlocks):
                    return {b'sf':clsname,
                            b'blocks':packb(obj._blocks)}

            elif package == 'numpy':
                #msgpack-numpy is breaking with these data types, overriding here
                if obj.__class__ is np.ndarray:
                    if obj.dtype.type == np.object_:
                        data = list(map(element_encode, obj))
                        return {b'np': True,
                                b'dtype': 'object_',
                                b'data': packb(data)}
                    elif obj.dtype.type == np.datetime64:
                        data = obj.astype(str)
                        return {b'np': True,
                                b'dtype': str(obj.dtype),
                                b'data': packb(data)}
                    elif obj.dtype.type == np.timedelta64:
                        data = obj.astype(np.float64)
                        return {b'np': True,
                                b'dtype': str(obj.dtype),
                                b'data': packb(data)}
            return chain(obj) #let msgpack_numpy.encode take over

        packb = partial(msgpack.packb, default=encode)
        element_encode = partial(MessagePackElement.encode, packb=packb)

        return packb(self)

    def to_xarray(self) -> 'Dataset':
        '''
        Return an xarray Dataset.

        In order to preserve columnar types, and following the precedent of Pandas, the :obj:`Frame`, with a 1D index, is translated as a Dataset of 1D arrays, where each DataArray is a 1D array. If the index is an :obj:`IndexHierarhcy`, each column is mapped into an ND array of shape equal to the unique values found at each depth of the index.
        '''
        import xarray

        columns = self.columns
        index = self.index

        if index.depth == 1:
            index_name = index.names[0]
            coords = {index_name: index.values}
        else:
            index_name = index.names
            # index values are reduced to unique values for 2d presentation
            coords = {index_name[d]: ufunc_unique(index.values_at_depth(d))
                    for d in range(index.depth)}
            # create dictionary version
            coords_index = {k: Index(v) for k, v in coords.items()}

        # columns form the keys in data_vars dict
        if columns.depth == 1:
            columns_values = columns.values
            # needs to be called with axis argument
            columns_arrays = partial(self._blocks.axis_values, axis=0)
        else: # must be hashable
            columns_values = array2d_to_tuples(columns.values)

            def columns_arrays() -> tp.Iterator[np.ndarray]:
                for c in self.iter_series(axis=0): #type: ignore
                    # dtype must be able to accomodate a float NaN
                    resolved = resolve_dtype(c.dtype, DTYPE_FLOAT_DEFAULT)
                    # create multidimensional arsdfray of all axis for each
                    array = np.full(
                            shape=[len(coords[v]) for v in coords],
                            fill_value=np.nan,
                            dtype=resolved)

                    for index_labels, value in c.items():
                        # translate to index positions
                        insert_pos = [coords_index[k]._loc_to_iloc(label)
                                for k, label in zip(coords, index_labels)]
                        # must convert to tuple to give position per dimension
                        array[tuple(insert_pos)] = value

                    yield array

        data_vars = {k: (index_name, v)
                for k, v in zip(columns_values, columns_arrays())}

        return xarray.Dataset(data_vars, coords=coords) #type: ignore

    def to_frame(self) -> 'Frame':
        '''
        Return Frame version of this Frame, which (as the Frame is immutable) is self.
        '''
        return self

    def _to_frame(self,
            constructor: tp.Type['Frame']
            ) -> 'Frame':
        return constructor(
                self._blocks.copy(),
                index=self.index,
                columns=self._columns,
                name=self._name,
                own_data=True,
                own_index=True,
                own_columns=constructor is FrameHE,
                )

    def to_frame_he(self) -> 'FrameHE':
        '''
        Return a ``FrameHE`` version of this Frame.
        '''
        return self._to_frame(FrameHE) #type: ignore

    def to_frame_go(self) -> 'FrameGO':
        '''
        Return a ``FrameGO`` version of this Frame.
        '''
        return self._to_frame(FrameGO) #type: ignore

    def to_series(self,
            *,
            index_constructor: IndexConstructor = Index,
            ) -> Series:
        '''
        Return a ``Series`` representation of this ``Frame``, where the index is extended with columns to from tuple labels for each element in the ``Frame``.

        Args:
            index_constructor: Index constructor of the tuples produced by combining index and columns into one label. Providing ``IndexHierarchy.from_labels`` will produce a hierarchical index.
        '''
        # NOTE: do not force block consolidation to avoid type coercion

        if self._index.ndim > 1: # force creation of a tuple of tuples
            index_tuples = tuple(self._index.__iter__())
        else:
            index_tuples = tuple((l,) for l in self._index.values)

        if self._columns.ndim > 1:
            columns_tuples = tuple(self._columns.__iter__())
        else:
            columns_tuples = tuple((l,) for l in self._columns.values)

        # immutability should be prserved
        array = self._blocks.values.reshape(self._blocks.size)

        def labels() -> tp.Iterator[tp.Hashable]:
            for row, col in np.ndindex(self._blocks._shape):
                yield index_tuples[row] + columns_tuples[col]

        index = index_constructor(labels())
        return Series(array, index=index, own_index=True, name=self._name)

    #---------------------------------------------------------------------------
    def _to_str_records(self,
            *,
            include_index: bool = True,
            include_index_name: bool = True,
            include_columns: bool = True,
            include_columns_name: bool = False,
            store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ) -> None:
        '''
        Iterator of records with values converted to strings.
        '''
        if sum((include_columns_name, include_index_name)) > 1:
            raise RuntimeError('cannot set both `include_columns_name` and `include_index_name`')

        index = self._index
        columns = self._columns

        if include_index:
            index_values = index.values # get once for caching
            index_names = index.names # normalized presentation
            index_depth = index.depth

        if include_columns:
            columns_names = columns.names

        if store_filter:
            filter_func = store_filter.from_type_filter_element

        if include_columns:
            if columns.depth == 1:
                columns_rows = (columns,)
            else:
                columns_rows = columns.values.T

            for row_idx, columns_row in enumerate(columns_rows):
                row = [] # column depth is a row

                if include_index:
                    # only have apex space if include columns and index
                    if include_index_name:
                        # index_names serves as a proxy for the index_depth
                        for name in index_names:
                            # we always write index name labels on the top-most
                            row.append(f'{name}' if row_idx == 0 else '')
                    elif include_columns_name:
                        for col_idx in range(index_depth):
                            row.append(f'{columns_names[row_idx]}' if col_idx == 0 else '')
                    else:
                        row.extend(('' for _ in range(index_depth)))
                # write the rest of the line
                if store_filter:
                    row.extend(f'{filter_func(x)}' for x in columns_row)
                else:
                    row.extend(f'{x}' for x in columns_row)
                yield row

        col_idx_last = self._blocks._shape[1] - 1
        # avoid row creation to avoid joining types; avoide creating a list for each row
        row_current_idx: tp.Optional[int] = None

        for (row_idx, col_idx), element in self._iter_element_iloc_items():
            if row_idx != row_current_idx: # each new row
                if row_current_idx is not None: # if not the first
                    yield row
                row = []
                if include_index:
                    if index_depth == 1:
                        index_value = index_values[row_idx]
                        if store_filter:
                            row.append(f'{filter_func(index_value)}')
                        else:
                            row.append(f'{index_value}')
                    else:
                        for index_value in index_values[row_idx]:
                            if store_filter:
                                row.append(f'{filter_func(index_value)}')
                            else:
                                row.append(f'{index_value}')
                row_current_idx = row_idx
            if store_filter:
                row.append(f'{filter_func(element)}')
            else:
                row.append(f'{element}')

        if row_current_idx is not None:
            yield row

    @doc_inject(selector='delimited')
    def to_delimited(self,
            fp: PathSpecifierOrFileLike,
            *,
            delimiter: str,
            include_index: bool = True,
            include_index_name: bool = True,
            include_columns: bool = True,
            include_columns_name: bool = False,
            encoding: tp.Optional[str] = None,
            line_terminator: str = '\n',
            quote_char: str = '"',
            quote_double: bool = True,
            escape_char: tp.Optional[str] = None,
            quoting: int = csv.QUOTE_MINIMAL,
            store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ) -> None:
        '''
        {doc} A ``delimiter`` character must be specified.

        Args:
            {fp}
            *
            {delimiter}
            {include_index}
            {include_index_name}
            {include_columns}
            {include_columns_name}
            {encoding}
            {line_terminator}
            {quote_char}
            {quote_double}
            {escape_char}
            {quoting}
            {store_filter}
        '''
        with file_like_manager(fp, encoding=encoding, mode='w') as fl:
            csvw = csv.writer(fl,
                    delimiter=delimiter,
                    escapechar=escape_char,
                    quotechar=quote_char,
                    lineterminator=line_terminator,
                    quoting=quoting,
                    doublequote=quote_double,
                    )
            for row in self._to_str_records(
                    include_index=include_index,
                    include_index_name=include_index_name,
                    include_columns=include_columns,
                    include_columns_name=include_columns_name,
                    store_filter=store_filter,
                    ):
                csvw.writerow(row)

    @doc_inject(selector='delimited')
    def to_csv(self,
            fp: PathSpecifierOrFileLike,
            *,
            include_index: bool = True,
            include_index_name: bool = True,
            include_columns: bool = True,
            include_columns_name: bool = False,
            encoding: tp.Optional[str] = None,
            line_terminator: str = '\n',
            quote_char: str = '"',
            quote_double: bool = True,
            escape_char: tp.Optional[str] = None,
            quoting: int = csv.QUOTE_MINIMAL,
            store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ) -> None:
        '''
        {doc} The delimiter is set to a comma.

        Args:
            {fp}
            *
            {include_index}
            {include_index_name}
            {include_columns}
            {include_columns_name}
            {encoding}
            {line_terminator}
            {quote_char}
            {quote_double}
            {escape_char}
            {quoting}
            {store_filter}
        '''
        return self.to_delimited(fp=fp,
                delimiter=',',
                include_index=include_index,
                include_index_name=include_index_name,
                include_columns=include_columns,
                include_columns_name=include_columns_name,
                encoding=encoding,
                line_terminator=line_terminator,
                quote_char=quote_char,
                quote_double=quote_double,
                escape_char=escape_char,
                quoting=quoting,
                store_filter=store_filter
                )

    @doc_inject(selector='delimited')
    def to_tsv(self,
            fp: PathSpecifierOrFileLike,
            *,
            include_index: bool = True,
            include_index_name: bool = True,
            include_columns: bool = True,
            include_columns_name: bool = False,
            encoding: tp.Optional[str] = None,
            line_terminator: str = '\n',
            quote_char: str = '"',
            quote_double: bool = True,
            escape_char: tp.Optional[str] = None,
            quoting: int = csv.QUOTE_MINIMAL,
            store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ) -> None:
        '''
        {doc} The delimiter is set to a tab.

        Args:
            {fp}
            *
            {include_index}
            {include_index_name}
            {include_columns}
            {include_columns_name}
            {encoding}
            {line_terminator}
            {quote_char}
            {quote_double}
            {escape_char}
            {quoting}
            {store_filter}
        '''
        return self.to_delimited(fp=fp,
                delimiter='\t',
                include_index=include_index,
                include_index_name=include_index_name,
                include_columns=include_columns,
                include_columns_name=include_columns_name,
                encoding=encoding,
                line_terminator=line_terminator,
                quote_char=quote_char,
                quote_double=quote_double,
                escape_char=escape_char,
                quoting=quoting,
                store_filter=store_filter
                )

    @doc_inject(selector='delimited')
    def to_clipboard(self,
            *,
            delimiter: str = '\t',
            include_index: bool = True,
            include_index_name: bool = True,
            include_columns: bool = True,
            include_columns_name: bool = False,
            encoding: tp.Optional[str] = None,
            line_terminator: str = '\n',
            quote_char: str = '"',
            quote_double: bool = True,
            escape_char: tp.Optional[str] = None,
            quoting: int = csv.QUOTE_MINIMAL,
            store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ) -> None:
        '''
        {doc} The ``delimiter`` defaults to a tab.

        Args:
            {fp}
            *
            {delimiter}
            {include_index}
            {include_index_name}
            {include_columns}
            {include_columns_name}
            {encoding}
            {line_terminator}
            {quote_char}
            {quote_double}
            {escape_char}
            {quoting}
            {store_filter}
        '''
        sio = StringIO()
        self.to_delimited(fp=sio,
                delimiter=delimiter,
                include_index=include_index,
                include_index_name=include_index_name,
                include_columns=include_columns,
                include_columns_name=include_columns_name,
                encoding=encoding,
                line_terminator=line_terminator,
                quote_char=quote_char,
                quote_double=quote_double,
                escape_char=escape_char,
                quoting=quoting,
                store_filter=store_filter
                )
        sio.seek(0)

        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.clipboard_clear() #type: ignore
        root.clipboard_append(sio.read()) #type: ignore

    #---------------------------------------------------------------------------
    # Store based output

    def to_xlsx(self,
            fp: PathSpecifier, # not sure I can take a file like yet
            *,
            label: tp.Hashable = STORE_LABEL_DEFAULT,
            include_index: bool = True,
            include_index_name: bool = True,
            include_columns: bool = True,
            include_columns_name: bool = False,
            merge_hierarchical_labels: bool = True,
            store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT,
            ) -> None:
        '''
        Write the Frame as single-sheet XLSX file.
        '''
        from static_frame.core.store_xlsx import StoreXLSX
        from static_frame.core.store import StoreConfig

        config = StoreConfig(
                include_index=include_index,
                include_index_name=include_index_name,
                include_columns=include_columns,
                include_columns_name=include_columns_name,
                merge_hierarchical_labels=merge_hierarchical_labels
                )
        st = StoreXLSX(fp)
        st.write(((label, self),),
                config=config,
                store_filter=store_filter,
                )

    def to_sqlite(self,
            fp: PathSpecifier, # not sure file-like StringIO works
            *,
            label: tp.Hashable = STORE_LABEL_DEFAULT,
            include_index: bool = True,
            include_columns: bool = True,
            # store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT,
            ) -> None:
        '''
        Write the Frame as single-table SQLite file.
        '''
        from static_frame.core.store_sqlite import StoreSQLite
        from static_frame.core.store import StoreConfig

        config = StoreConfig(
                include_index=include_index,
                include_columns=include_columns,
                )

        st = StoreSQLite(fp)
        st.write(((label, self),),
                config=config,
                # store_filter=store_filter,
                )

    def to_hdf5(self,
            fp: PathSpecifier, # not sure file-like StringIO works
            *,
            label: tp.Hashable = STORE_LABEL_DEFAULT,
            include_index: bool = True,
            include_columns: bool = True,
            # store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT,
            ) -> None:
        '''
        Write the Frame as single-table SQLite file.
        '''
        from static_frame.core.store_hdf5 import StoreHDF5
        from static_frame.core.store import StoreConfig

        config = StoreConfig(
                include_index=include_index,
                include_columns=include_columns,
                )

        if label is STORE_LABEL_DEFAULT:
            if not self.name:
                raise RuntimeError('must provide a label or define Frame name.')
            label = self.name

        st = StoreHDF5(fp)
        st.write(((label, self),),
                config=config,
                # store_filter=store_filter,
                )

    #---------------------------------------------------------------------------

    @doc_inject(class_name='Frame')
    def to_html(self,
            config: tp.Optional[DisplayConfig] = None,
            style_config: tp.Optional[StyleConfig] = STYLE_CONFIG_DEFAULT,
            ) -> str:
        '''
        {}
        '''
        # if a config is given, try to use all settings; if using active, hide types
        config = config or DisplayActive.get(type_show=False)
        config = config.to_display_config(
                display_format=DisplayFormats.HTML_TABLE,
                )

        style_config = style_config_css_factory(style_config, self)
        return repr(self.display(config, style_config=style_config))

    @doc_inject(class_name='Frame')
    def to_html_datatables(self,
            fp: tp.Optional[PathSpecifierOrFileLike] = None,
            show: bool = True,
            config: tp.Optional[DisplayConfig] = None
            ) -> tp.Optional[str]:
        '''
        {}
        '''
        config = config or DisplayActive.get(type_show=False)
        config = config.to_display_config(
                display_format=DisplayFormats.HTML_DATATABLES,
                )
        content = repr(self.display(config))

        # path_filter called internally
        fp = write_optional_file(content=content, fp=fp)

        if show:
            import webbrowser #pragma: no cover
            webbrowser.open_new_tab(fp) #pragma: no cover
        return fp

    def to_rst(self,
            config: tp.Optional[DisplayConfig] = None
            ) -> str:
        '''
        Display the Frame as an RST formatted table.
        '''
        # if a config is given, try to use all settings; if using active, hide types
        config = config or DisplayActive.get(type_show=False)
        config = config.to_display_config(
                display_format=DisplayFormats.RST,
                )
        return repr(self.display(config))

    def to_markdown(self,
            config: tp.Optional[DisplayConfig] = None
            ) -> str:
        '''
        Display the Frame as a Markdown formatted table.
        '''
        # if a config is given, try to use all settings; if using active, hide types
        config = config or DisplayActive.get(type_show=False)
        config = config.to_display_config(
                display_format=DisplayFormats.MARKDOWN,
                )
        return repr(self.display(config))


    def to_latex(self,
            config: tp.Optional[DisplayConfig] = None
            ) -> str:
        '''
        Display the Frame as a LaTeX formatted table.
        '''
        # if a config is given, try to use all settings; if using active, hide types
        config = config or DisplayActive.get(type_show=False)
        config = config.to_display_config(
                display_format=DisplayFormats.LATEX,
                type_delimiter_left='$<$', # escape
                type_delimiter_right='$>$',
                )
        return repr(self.display(config))

#-------------------------------------------------------------------------------

class FrameGO(Frame):
    '''A grow-only Frame, providing a two-dimensional, ordered, labelled container, immutable with grow-only columns.
    '''

    __slots__ = (
            '_blocks',
            '_columns',
            '_index',
            '_name'
            )

    STATIC = False
    _COLUMNS_CONSTRUCTOR = IndexGO
    _COLUMNS_HIERARCHY_CONSTRUCTOR = IndexHierarchyGO

    _columns: IndexGO


    def __setitem__(self,
            key: tp.Hashable,
            value: tp.Any,
            fill_value: tp.Any = np.nan
            ) -> None:
        '''For adding a single column, one column at a time.
        '''
        if key in self._columns:
            raise RuntimeError(f'The provided key ({key}) is already defined in columns; if you want to change or replace this column, use .assign to get new Frame')

        row_count = len(self._index)

        if isinstance(value, Series):
            # select only the values matching our index
            block = value.reindex(self.index, fill_value=fill_value).values
        elif isinstance(value, Frame):
            raise RuntimeError(
                    f'cannot use setitem with a Frame; use {self.__class__.__name__}.extend()')
        elif value.__class__ is np.ndarray:
            # this permits unaligned assignment as no index is used, possibly remove
            if value.ndim != 1:
                raise RuntimeError(
                        f'can only use setitem with 1D containers')
            if len(value) != row_count:
                # block may have zero shape if created without columns
                raise RuntimeError(f'incorrectly sized unindexed value: {len(value)} != {row_count}')
            block = value # NOTE: could own_data here with additional argument

        else:
            if not hasattr(value, '__iter__') or isinstance(value, str):
                block = np.full(row_count, value)
                block.flags.writeable = False
            else:
                block, _ = iterable_to_array_1d(value) # returns immutable

            if block.ndim != 1 or len(block) != row_count:
                raise RuntimeError('incorrectly sized, unindexed value')

        # Wait until after extracting block from value before updating _columns, as value evaluation might fail.
        self._columns.append(key)
        self._blocks.append(block)


    def extend_items(self,
            pairs: tp.Iterable[tp.Tuple[tp.Hashable, Series]],
            fill_value: tp.Any = np.nan,
            ) -> None:
        '''
        Given an iterable of pairs of column name, column value, extend this FrameGO. Columns values can be any iterable suitable for usage in __setitem__.
        '''
        for k, v in pairs:
            self.__setitem__(k, v, fill_value)


    def extend(self,
            container: tp.Union['Frame', Series],
            fill_value: tp.Any = np.nan
            ) -> None:
        '''Extend this FrameGO (in-place) with another Frame's blocks or Series array; as blocks are immutable, this is a no-copy operation when indices align. If indices do not align, the passed-in Frame or Series will be reindexed (as happens when adding a column to a FrameGO).

        If a Series is passed in, the column name will be taken from the Series ``name`` attribute.

        This method differs from FrameGO.extend_items() by permitting contiguous underlying blocks to be extended from another Frame into this Frame.
        '''
        if not isinstance(container, (Series, Frame)):
            raise NotImplementedError(
                    f'no support for extending with {type(container)}')

        # self's index will never change; we only take what aligns in the passed container
        if not self._index.equals(container._index):
            container = container.reindex(self._index,
                    fill_value=fill_value,
                    check_equals=False,
                    )

        if isinstance(container, Frame):
            if not len(container.columns):
                return
            self._columns.extend(container.keys())
            self._blocks.extend(container._blocks)
        elif isinstance(container, Series):
            self._columns.append(container.name)
            self._blocks.append(container.values)

        # this should never happen, and is hard to test!
        assert len(self._columns) == self._blocks._shape[1] #pragma: no cover

    #---------------------------------------------------------------------------
    def via_fill_value(self,
            fill_value: object = np.nan,
            ) -> InterfaceFillValueGO:
        '''
        Interface for using binary operators and methods with a pre-defined fill value.
        '''
        return InterfaceFillValueGO(
                container=self,
                fill_value=fill_value,
                )

    #---------------------------------------------------------------------------
    def _to_frame(self,
            constructor: tp.Type[Frame]
            ) -> Frame:
        return constructor(
                self._blocks.copy(),
                index=self.index,
                columns=self._columns,
                name=self._name,
                own_data=True,
                own_index=True,
                own_columns=False, # all cases need new columns
                )

    def to_frame(self) -> Frame:
        '''
        Return :obj:`Frame` version of this :obj:`FrameGO`.
        '''
        return self._to_frame(Frame)

    def to_frame_he(self) -> 'FrameHE':
        '''
        Return a :obj:`FrameGO` version of this :obj:`FrameGO`.
        '''
        return self._to_frame(FrameHE) #type: ignore

    def to_frame_go(self) -> 'FrameGO':
        '''
        Return a :obj:`FrameGO` version of this :obj:`FrameGO`.
        '''
        return self._to_frame(FrameGO) #type: ignore


#-------------------------------------------------------------------------------
# utility delegates returned from selection routines and exposing the __call__ interface.

class FrameAssign(Assign):
    __slots__ = (
        'container',
        'key',
        )

   # common base classe for supplying delegate; need to define interface for docs
    def __call__(self,
            value: tp.Any,
            *,
            fill_value: tp.Any = np.nan,
            ) -> 'Frame':
        '''
        Assign the ``value`` in the position specified by the selector. The `name` attribute is propagated to the returned container.

        Args:
            value: Value to assign, which can be a :obj:`Series`, :obj:`Frame`, np.ndarray, or element.
            *.
            fill_value: If the ``value`` parameter has to be reindexed, this element will be used to fill newly created elements.
        '''
        raise NotImplementedError()

    def apply(self,
            func: AnyCallable,
            *,
            fill_value: tp.Any = np.nan,
            ) -> 'Frame':
        '''
        Provide a function to apply to the assignment target, and use that as the assignment value.

        Args:
            func: A function to apply to the assignment target.
            *.
            fill_value: If the function does not produce a container with a matching index, the element will be used to fill newly created elements.
        '''
        raise NotImplementedError()



class FrameAssignILoc(FrameAssign):
    __slots__ = (
        'container',
        'key',
        )

    def __init__(self,
            container: Frame,
            key: GetItemKeyTypeCompound = None,
            ) -> None:
        '''
        Args:
            key: an iloc key.
        '''
        self.container = container
        self.key = key

    def __call__(self,
            value: tp.Any,
            *,
            fill_value: tp.Any = np.nan,
            ) -> 'Frame':
        is_frame = isinstance(value, Frame)
        is_series = isinstance(value, Series)

        if isinstance(self.key, tuple):
            # NOTE: the iloc key's order is not relevant in assignment, and block assignment requires that column keys are ascending
            key = (self.key[0], #type: ignore [index]
                    key_to_ascending_key(
                            self.key[1],
                            self.container.shape[1] #type: ignore [index]
                    ))
        else:
            key = (self.key, None)

        if is_series:
            assigned = self.container._reindex_other_like_iloc(value,
                    key,
                    fill_value=fill_value).values
            blocks = self.container._blocks.extract_iloc_assign_by_unit(
                    key,
                    assigned,
                    )
        elif is_frame:
            assigned = self.container._reindex_other_like_iloc(value, #type: ignore [union-attr]
                    key,
                    fill_value=fill_value)._blocks._blocks
            blocks = self.container._blocks.extract_iloc_assign_by_blocks(
                    key,
                    assigned,
                    )
        else: # could be array or single element
            assigned = value
            blocks = self.container._blocks.extract_iloc_assign_by_unit(
                    key,
                    assigned,
                    )

        return self.container.__class__(
                data=blocks,
                columns=self.container._columns,
                index=self.container._index,
                name=self.container._name,
                own_data=True
                )

    def apply(self,
            func: AnyCallable,
            *,
            fill_value: tp.Any = np.nan,
            ) -> 'Frame':
        value = func(self.container.iloc[self.key])
        return self.__call__(value, fill_value=fill_value)


class FrameAssignBLoc(FrameAssign):
    __slots__ = (
        'container',
        'key',
        )

    def __init__(self,
            container: Frame,
            key: tp.Optional[Bloc2DKeyType] = None,
            ) -> None:
        '''
        Args:
            key: a bloc-style key.
        '''
        self.container = container
        self.key = key

    def __call__(self,
            value: tp.Any,
            *,
            fill_value: tp.Any = np.nan,
            ) -> 'Frame':
        is_frame = isinstance(value, Frame)
        is_series = isinstance(value, Series)

        # get Bollean key of normalized shape; in most cases this will be a new, mutable array
        key = bloc_key_normalize(
                key=self.key,
                container=self.container
                )
        if is_series:
            # assumes a Series from a bloc selection, i.e., tuples of index/col loc labels
            index = self.container._index
            columns = self.container._columns

            # cannot assume order of coordinates, so create a mapping for lookup by coordinate
            values_map = {}
            for (i, c), e in value.items():
                values_map[index._loc_to_iloc(i), columns._loc_to_iloc(c)] = e

            # NOTE: should we pass dtype here, or re-evaluate dtype from observed values for each block?
            blocks = self.container._blocks.extract_bloc_assign_by_coordinate(
                    key,
                    values_map,
                    value.values.dtype,
                    )

        elif is_frame:
            # NOTE: the type of FILL_VALUE_DEFAULT might coerce other blocks
            value = value.reindex(
                    index=self.container._index,
                    columns=self.container._columns,
                    fill_value=FILL_VALUE_DEFAULT)
            values = value._blocks._blocks

            # if we produced any invalid entries, cannot select them
            invalid_found = (value == FILL_VALUE_DEFAULT).values
            if invalid_found.any():
                if not key.flags.writeable:
                    key = key.copy() # mutate a copy
                key[invalid_found] = False

            blocks = self.container._blocks.extract_bloc_assign_by_blocks(key, values)

        else: # an array or an element
            if value.__class__ is np.ndarray and value.shape != self.container.shape:
                raise RuntimeError(f'value must match shape {self.container.shape}')
            blocks = self.container._blocks.extract_bloc_assign_by_unit(key, value)

        return self.container.__class__(
                data=blocks,
                columns=self.container._columns,
                index=self.container._index,
                name=self.container._name,
                own_data=True
                )

    def apply(self,
            func: AnyCallable,
            *,
            fill_value: tp.Any = np.nan,
            ) -> 'Frame':
        # use the Boolean key for a bloc selection, which always returns a Series
        value = func(self.container.bloc[self.key])
        return self.__call__(value, fill_value=fill_value)

#-------------------------------------------------------------------------------
class FrameAsType:
    '''
    The object returned from the getitem selector, exposing the functional (__call__) interface to pass in the dtype, as well as (optionally) whether blocks are consolidated.
    '''
    __slots__ = ('container', 'column_key',)

    def __init__(self,
            container: Frame,
            column_key: GetItemKeyType
            ) -> None:
        self.container = container
        self.column_key = column_key

    def __call__(self,
            dtypes: DtypesSpecifier,
            *,
            consolidate_blocks: bool = True,
            ) -> 'Frame':

        if self.column_key.__class__ is slice and self.column_key == NULL_SLICE:
            if is_mapping(dtypes):
                # translate keys loc to iloc
                dtypes = {self.container._columns._loc_to_iloc(k): v
                        for k, v in dtypes.items()} #type: ignore [union-attr]
            gen = self.container._blocks._astype_blocks_from_dtypes(dtypes)
        else:
            if not is_dtype_specifier(dtypes):
                raise RuntimeError('must supply a single dtype specifier if using a column selection other than the NULL slice')
            gen = self.container._blocks._astype_blocks(self.column_key, dtypes)

        if consolidate_blocks:
            gen = TypeBlocks.consolidate_blocks(gen)

        blocks = TypeBlocks.from_blocks(gen)

        return self.container.__class__(
                data=blocks,
                columns=self.container.columns,
                index=self.container.index,
                name=self.container._name,
                own_data=True,
                )


#-------------------------------------------------------------------------------
class FrameHE(Frame):
    '''
    A hash/equals subclass of :obj:`Frame`, permiting usage in a Python set, dictionary, or other contexts where a hashable container is needed. To support hashability, ``__eq__`` is implemented to return a Boolean rather than a Boolean :obj:`Frame`
    '''

    __slots__ = (
            '_blocks',
            '_columns',
            '_index',
            '_name',
            '_hash',
            )

    _hash: int

    def __eq__(self, other: tp.Any) -> bool:
        '''
        Return True if other is a ``Frame`` with the same labels, values, and name. Container class and underlying dtypes are not independently compared.
        '''
        return self.equals(other, #type: ignore [no-any-return]
                compare_name=True,
                compare_dtype=False,
                compare_class=False,
                skipna=True,
                )

    def __ne__(self, other: tp.Any) -> bool:
        '''
        Return False if other is a ``Frame`` with the different labels, values, or name. Container class and underlying dtypes are not independently compared.
        '''
        return not self.__eq__(other)

    def __hash__(self) -> int:
        if not hasattr(self, '_hash'):
            self._hash = hash((
                    tuple(self.index.values),
                    tuple(self.columns.values),
                    # tuple(dt.str for dt in self._blocks.dtypes)
                    ))
        return self._hash

    def to_frame_he(self) -> 'FrameHE':
        '''
        Return :obj:`FrameHE` version of this :obj:`FrameHE`, which (as the :obj:`FrameHE` is immutable) is self.
        '''
        return self

    def _to_frame(self,
            constructor: tp.Type[Frame]
            ) -> Frame:
        return constructor(
                self._blocks.copy(),
                index=self.index,
                columns=self._columns,
                name=self._name,
                own_data=True,
                own_index=True,
                own_columns=constructor is Frame,
                )

    def to_frame(self) -> Frame:
        '''
        Return obj:`Frame` version of this obj:`FrameHE`.
        '''
        return self._to_frame(Frame)

    def to_frame_go(self) -> FrameGO:
        '''
        Return a obj:`FrameGO` version of this obj:`FrameHE`.
        '''
        return self._to_frame(FrameGO) #type: ignore [return-value]
