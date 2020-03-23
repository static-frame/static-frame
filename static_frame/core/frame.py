import typing as tp
import sqlite3
import csv
import json
from functools import partial
from itertools import chain
from itertools import repeat

import numpy as np

from numpy.ma import MaskedArray

from static_frame.core.util import UFunc
from static_frame.core.util import DEFAULT_SORT_KIND
from static_frame.core.util import DTYPE_FLOAT_DEFAULT
from static_frame.core.util import EMPTY_TUPLE
from static_frame.core.util import DTYPE_OBJECT

from static_frame.core.util import NULL_SLICE
from static_frame.core.util import KEY_MULTIPLE_TYPES
from static_frame.core.util import INT_TYPES
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import GetItemKeyTypeCompound
from static_frame.core.util import KeyOrKeys
from static_frame.core.util import PathSpecifier
from static_frame.core.util import PathSpecifierOrFileLike
from static_frame.core.util import PathSpecifierOrFileLikeOrIterator

from static_frame.core.util import DtypesSpecifier
from static_frame.core.util import DtypeSpecifier

from static_frame.core.util import FILL_VALUE_DEFAULT
from static_frame.core.util import path_filter
from static_frame.core.util import Bloc2DKeyType

from static_frame.core.util import IndexSpecifier
from static_frame.core.util import IndexInitializer
from static_frame.core.util import IndexConstructor
from static_frame.core.util import IndexConstructors

from static_frame.core.util import FrameInitializer
from static_frame.core.util import FRAME_INITIALIZER_DEFAULT
from static_frame.core.util import column_2d_filter
from static_frame.core.util import column_1d_filter

from static_frame.core.util import name_filter
from static_frame.core.util import _gen_skip_middle
from static_frame.core.util import iterable_to_array_1d
from static_frame.core.util import isin
# from static_frame.core.util import _dict_to_sorted_items
from static_frame.core.util import array_to_duplicated
from static_frame.core.util import ufunc_set_iter
from static_frame.core.util import array2d_to_tuples
from static_frame.core.util import _read_url
from static_frame.core.util import write_optional_file
from static_frame.core.util import ufunc_unique
# from static_frame.core.util import STATIC_ATTR
from static_frame.core.util import concat_resolved
from static_frame.core.util import DepthLevelSpecifier
from static_frame.core.util import array_to_groups_and_locations
from static_frame.core.util import is_callable_or_mapping
from static_frame.core.util import CallableOrCallableMap
from static_frame.core.util import ufunc_axis_skipna
from static_frame.core.util import AnyCallable

from static_frame.core.util import argmin_2d
from static_frame.core.util import argmax_2d
from static_frame.core.util import resolve_dtype
from static_frame.core.util import key_normalize
from static_frame.core.util import get_tuple_constructor
from static_frame.core.util import dtype_to_na
from static_frame.core.util import is_hashable
from static_frame.core.util import reversed_iter

from static_frame.core.selector_node import InterfaceGetItem
from static_frame.core.selector_node import InterfaceSelection2D
from static_frame.core.selector_node import InterfaceAssign2D
from static_frame.core.selector_node import InterfaceAsType

from static_frame.core.index_correspondence import IndexCorrespondence
from static_frame.core.container import ContainerOperand

from static_frame.core.container_util import matmul
from static_frame.core.container_util import index_from_optional_constructor
from static_frame.core.container_util import axis_window_items
from static_frame.core.container_util import bloc_key_normalize
from static_frame.core.container_util import rehierarch_and_map
from static_frame.core.container_util import array_from_value_iter
from static_frame.core.container_util import dtypes_mappable
from static_frame.core.container_util import key_to_ascending_key
from static_frame.core.container_util import index_constructor_empty
from static_frame.core.container_util import pandas_version_under_1
from static_frame.core.container_util import pandas_to_numpy

from static_frame.core.iter_node import IterNodeApplyType
from static_frame.core.iter_node import IterNodeType
# from static_frame.core.iter_node import IterNode

from static_frame.core.iter_node import IterNodeAxis
from static_frame.core.iter_node import IterNodeDepthLevelAxis
from static_frame.core.iter_node import IterNodeWindow
from static_frame.core.iter_node import IterNodeGroupAxis
from static_frame.core.iter_node import IterNodeNoArg


from static_frame.core.display import DisplayConfig
from static_frame.core.display import DisplayActive
from static_frame.core.display import Display
from static_frame.core.display import DisplayFormats
from static_frame.core.display import DisplayHeader

from static_frame.core.type_blocks import TypeBlocks

from static_frame.core.series import Series
from static_frame.core.series import RelabelInput

from static_frame.core.index_base import IndexBase

from static_frame.core.index import Index
from static_frame.core.index import IndexGO
from static_frame.core.index import _requires_reindex
from static_frame.core.index import _index_initializer_needs_init
from static_frame.core.index import immutable_index_filter

from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.index_hierarchy import IndexHierarchyGO

from static_frame.core.index_auto import IndexAutoFactory
from static_frame.core.index_auto import IndexAutoFactoryType

from static_frame.core.store_filter import StoreFilter
from static_frame.core.store_filter import STORE_FILTER_DEFAULT


from static_frame.core.exception import ErrorInitFrame
from static_frame.core.exception import AxisInvalid

from static_frame.core.doc_str import doc_inject

if tp.TYPE_CHECKING:
    import pandas #pylint: disable=W0611 #pragma: no cover
    from xarray import Dataset #pylint: disable=W0611 #pragma: no cover
    import pyarrow #pylint: disable=W0611 #pragma: no cover



@doc_inject(selector='container_init', class_name='Frame')
class Frame(ContainerOperand):
    '''
    A two-dimensional ordered, labelled collection, immutable and of fixed size.

    Args:
        data: A Frame initializer, given as either a NumPy array, a single value (to be used to fill a shape defined by ``index`` and ``columns``), or an iterable suitable to given to the NumPy array constructor.
        {index}
        {columns}
        {own_data}
        {own_index}
        {own_columns}
    '''

    __slots__ = (
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
            ):
        '''
        Frame constructor from a Series:

        Args:
            series: A Series instance, to be realized as single column, with the column label taken from the `name` attribute.
        '''
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
            ):
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

        array = np.full(
                (len(index_final), len(columns_final)),
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
            ):
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
            name: tp.Hashable = None,
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
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`static_frame.Frame`
        '''

        # when doing axis 1 concat (growin horizontally) Series need to be presented as rows (axis 0)
        # axis_series = (0 if axis is 1 else 1)
        frames = [f if isinstance(f, Frame) else f.to_frame(axis) for f in frames]

        own_columns = False
        own_index = False

        # End quickly if empty iterable
        if not frames:
            return cls(
                    index=index,
                    columns=columns,
                    name=name,
                    own_columns=own_columns,
                    own_index=own_index)

        # switch if we have reduced the columns argument to an array
        from_array_columns = False
        from_array_index = False

        if axis == 1: # stacks columns (extends rows horizontally)
            # index can be the same, columns must be redefined if not unique
            if columns is IndexAutoFactory:
                columns = None # let default creation happen
            elif columns is None:
                # returns immutable array
                columns = concat_resolved([frame._columns.values for frame in frames])
                from_array_columns = True
                # avoid sort for performance; always want rows if ndim is 2
                if len(ufunc_unique(columns, axis=0)) != len(columns):
                    raise ErrorInitFrame('Column names after horizontal concatenation are not unique; supply a columns argument or IndexAutoFactory.')

            if index is IndexAutoFactory:
                raise ErrorInitFrame('for axis 1 concatenation, index must be used for reindexing row alignment: IndexAutoFactory is not permitted')
            elif index is None:
                # get the union index, or the common index if identical
                index = ufunc_set_iter(
                        (frame._index.values for frame in frames),
                        union=union,
                        assume_unique=True # all from indices
                        )
                index.flags.writeable = False
                from_array_index = True

            def blocks():
                for frame in frames:
                    if len(frame.index) != len(index) or (frame.index != index).any():
                        frame = frame.reindex(index=index, fill_value=fill_value)
                    for block in frame._blocks._blocks:
                        yield block

        elif axis == 0: # stacks rows (extends columns vertically)
            if index is IndexAutoFactory:
                index = None # let default creationn happen
            elif index is None:
                # returns immutable array
                index = concat_resolved([frame._index.values for frame in frames])
                from_array_index = True
                # avoid sort for performance; always want rows if ndim is 2
                if len(ufunc_unique(index, axis=0)) != len(index):
                    raise ErrorInitFrame('Index names after vertical concatenation are not unique; supply an index argument or IndexAutoFactory.')

            if columns is IndexAutoFactory:
                raise ErrorInitFrame('for axis 0 concatenation, columns must be used for reindexing and column alignment: IndexAutoFactory is not permitted')
            elif columns is None:
                columns = ufunc_set_iter(
                        (frame._columns.values for frame in frames),
                        union=union,
                        assume_unique=True
                        )
                columns.flags.writeable = False
                from_array_columns = True

            def blocks():
                aligned_frames = []
                previous_frame = None
                block_compatible = True
                reblock_compatible = True

                for frame in frames:
                    if len(frame.columns) != len(columns) or (frame.columns != columns).any():
                        frame = frame.reindex(columns=columns, fill_value=fill_value)

                    aligned_frames.append(frame)
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

                if block_compatible or reblock_compatible:
                    if not block_compatible and reblock_compatible:
                        # after reblocking, will be compatible
                        type_blocks = [f._blocks.consolidate() for f in aligned_frames]
                    else: # blocks by column are compatible
                        type_blocks = [f._blocks for f in aligned_frames]

                    # all TypeBlocks have the same number of blocks by here
                    for block_idx in range(len(type_blocks[0]._blocks)):
                        block_parts = []
                        for frame_idx in range(len(type_blocks)):
                            b = column_2d_filter(
                                    type_blocks[frame_idx]._blocks[block_idx])
                            block_parts.append(b)
                        # returns immutable array
                        yield concat_resolved(block_parts)
                else: # blocks not alignable
                    # break into single column arrays for maximum type integrity; there might be an alternative reblocking that could be more efficient, but determining that shape might be complex
                    for i in range(len(columns)):
                        block_parts = [
                            f._blocks._extract_array(column_key=i) for f in aligned_frames]
                        yield concat_resolved(block_parts)
        else:
            raise NotImplementedError(f'no support for {axis}')

        if from_array_columns:
            if columns.ndim == 2: # we have a hierarchical index
                columns = cls._COLUMNS_HIERARCHY_CONSTRUCTOR.from_labels(columns)
                own_columns = True

        if from_array_index:
            if index.ndim == 2: # we have a hierarchical index
                # NOTE: could pass index_constructors here
                index = IndexHierarchy.from_labels(index)
                own_index = True

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
            items: tp.Iterable[tp.Tuple[tp.Hashable, 'Frame']],
            *,
            axis: int = 0,
            union: bool = True,
            name: tp.Hashable = None,
            fill_value: object = np.nan,
            consolidate_blocks: bool = False
            ) -> 'Frame':
        '''
        Produce a :obj:`Frame` with a hierarchical index from an iterable of pairs of labels, :obj:`Frame`. The :obj:`IndexHierarchy` is formed from the provided labels and the :obj:`Index` if each :obj:`Frame`.

        Args:
            items: Iterable of pairs of label, :obj:`Series`
        '''
        frames = []

        def gen():
            for label, frame in items:
                # must normalize Series here to avoid down-stream confusion
                if isinstance(frame, Series):
                    frame = frame.to_frame(axis)

                frames.append(frame)
                if axis == 0:
                    yield label, frame._index
                elif axis == 1:
                    yield label, frame._columns
                # we have already evaluated AxisInvalid


        # populates array_values as side effect
        if axis == 0:
            ih = IndexHierarchy.from_index_items(gen())
            kwargs = dict(index=ih)
        elif axis == 1:
            ih = cls._COLUMNS_HIERARCHY_CONSTRUCTOR.from_index_items(gen())
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
        '''Frame constructor from an iterable of rows, where rows are defined as iterables, including tuples, lists, and arrays. If each row is a NamedTuple, and ``columns`` is not provided, column names will be derived from the NamedTuple fields.

        For records defined as ``Series``, use ``Frame.from_concat``; for records defined as dictionary, use ``Frame.from_dict_records``; for creating a ``Frame`` from a single dictionary, where keys are column labels and values are columns, use ``Frame.from_dict``.

        Args:
            records: Iterable of row values, where row values are arrays, tuples, lists, or namedtuples. For dictionary records, use ``Frame.from_dict_records``.
            index: Optionally provide an iterable of index labels, equal in length to the number of records. If a generator, this value will not be evaluated until after records are loaded.
            columns: Optionally provide an iterable of column labels, equal in length to the number of elements in a row.
            {dtypes}
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`static_frame.Frame`
        '''
        # if records is np; we can just pass it to constructor, as is alrady a consolidate type
        if isinstance(records, np.ndarray):
            if dtypes is not None:
                raise ErrorInitFrame('specifying dtypes when using NP records is not permitted')
            return cls(records,
                    index=index,
                    columns=columns,
                    index_constructor=index_constructor,
                    columns_constructor=columns_constructor,
                    own_index=own_index,
                    own_columns=own_columns,
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

        column_name_getter = None
        if columns is None and hasattr(row_reference, '_fields'): # NamedTuple
            column_name_getter = row_reference._fields.__getitem__
            columns = []

        if dtypes:
            dtypes_is_map = dtypes_mappable(dtypes)
            def get_col_dtype(col_idx):
                if dtypes_is_map:
                    return dtypes.get(columns[col_idx], None)
                return dtypes[col_idx]
        else:
            get_col_dtype = None

        col_count = len(row_reference)

        def get_value_iter(col_key):
            rows_iter = rows if not rows_to_iter else iter(rows)
            # this is possible to support ragged lists, but it noticeably reduces performance
            return (row[col_key] for row in rows_iter)

        def blocks():
            # iterate over final column order, yielding 1D arrays
            for col_idx in range(col_count):
                if column_name_getter: # append as side effect of generator!
                    columns.append(column_name_getter(col_idx))

                values = array_from_value_iter(
                        key=col_idx,
                        idx=col_idx,
                        get_value_iter=get_value_iter, get_col_dtype=get_col_dtype,
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
        '''Frame constructor from an iterable of dictionaries; column names will be derived from the union of all keys.

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

        if dtypes:
            dtypes_is_map = dtypes_mappable(dtypes)
            def get_col_dtype(col_idx):
                if dtypes_is_map:
                    return dtypes.get(columns[col_idx], None)
                return dtypes[col_idx]
        else:
            get_col_dtype = None

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
        def get_value_iter(col_key):
            rows_iter = rows if not rows_to_iter else iter(rows)
            return (row.get(col_key, fill_value) for row in rows_iter)

        def blocks():
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
            consolidate_blocks: bool = False) -> 'Frame':
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

        def gen():
            for label, values in items:
                index.append(label)
                yield values

        return cls.from_records(gen(),
                index=index,
                columns=columns,
                dtypes=dtypes,
                name=name,
                consolidate_blocks=consolidate_blocks
                )

    @classmethod
    @doc_inject(selector='constructor_frame')
    def from_dict_records_items(cls,
            items: tp.Iterator[tp.Tuple[tp.Hashable, tp.Iterable[tp.Any]]],
            *,
            dtypes: DtypesSpecifier = None,
            name: tp.Hashable = None,
            consolidate_blocks: bool = False) -> 'Frame':
        '''Frame constructor from iterable of pairs of index value, row (where row is an iterable).

        Args:
            items: Iterable of pairs of index label, row values, where row values are arrays, tuples, lists, dictionaries, or namedtuples.
            {dtypes}
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`static_frame.Frame`

        '''
        index = []

        def gen():
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
            ):
        '''Frame constructor from an iterator or generator of pairs, where the first value is the column label and the second value is an iterable of a column's values.

        Args:
            pairs: Iterable of pairs of column name, column values.
            index: Iterable of values to create an Index.
            fill_value: If pairs include Series, they will be reindexed with the provided index; reindexing will use this fill value.
            {dtypes}
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`static_frame.Frame`
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

        dtypes_is_map = dtypes_mappable(dtypes)
        def get_col_dtype(col_idx):
            if dtypes_is_map:
                return dtypes.get(columns[col_idx], None)
            return dtypes[col_idx]

        def blocks():
            for col_idx, (k, v) in enumerate(pairs):
                columns.append(k) # side effet of generator!

                if dtypes:
                    column_type = get_col_dtype(col_idx)
                else:
                    column_type = None

                if isinstance(v, np.ndarray):
                    # NOTE: we rely on TypeBlocks constructor to check that these are same sized
                    if column_type is not None:
                        yield v.astype(column_type)
                    else:
                        yield v
                elif isinstance(v, Series):
                    if index is None:
                        raise ErrorInitFrame('can only consume Series in Frame.from_items if an Index is provided.')

                    if _requires_reindex(v.index, index):
                        v = v.reindex(index, fill_value=fill_value)
                    # return values array post reindexing
                    if column_type is not None:
                        yield v.values.astype(column_type)
                    else:
                        yield v.values

                elif isinstance(v, Frame):
                    raise ErrorInitFrame('Frames are not supported in from_items constructor.')
                else:
                    values = np.array(v, dtype=column_type)
                    values.flags.writeable = False
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
            index: IndexInitializer = None,
            fill_value: object = np.nan,
            dtypes: DtypesSpecifier = None,
            name: tp.Hashable = None,
            index_constructor: IndexConstructor = None,
            columns_constructor: IndexConstructor = None,
            consolidate_blocks: bool = False
            ) -> 'Frame':
        '''
        Create a Frame from a dictionary, or any object that has an items() method.

        Args:
            mapping: a dictionary or similar mapping interface.
            {dtypes}
            {name}
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
            # construct columns_by_col_idx from columns, adding Nones for index columns
            for i in range(len(names)):
                if i >= index_start_pos and i <= index_end_pos:
                    columns_by_col_idx.append(index_field_placeholder)
                    continue
                columns_by_col_idx.append(columns[columns_idx])
                columns_idx += 1

        dtypes_is_map = dtypes_mappable(dtypes)

        def get_col_dtype(col_idx: int):
            if dtypes_is_map:
                # columns_by_col_idx may have a index_field_placeholder: will return None
                return dtypes.get(columns_by_col_idx[col_idx], None)
            # assume dytpes is an ordered sequences
            return dtypes[col_idx]

        def blocks():
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

                if dtypes:
                    dtype = get_col_dtype(col_idx)
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
    # iloc/loc pairs constructors: these are not public, not sure if they should be

    @classmethod
    def from_element_iloc_items(cls,
            items,
            *,
            index,
            columns,
            dtype,
            name: tp.Hashable = None
            ) -> 'Frame':
        '''
        Given an iterable of pairs of iloc coordinates and values, populate a Frame as defined by the given index and columns. The dtype must be specified, and must be the same for all values.

        Returns:
            :obj:`static_frame.Frame`
        '''
        index = Index(index)
        columns = cls._COLUMNS_CONSTRUCTOR(columns)

        tb = TypeBlocks.from_element_items(items,
                shape=(len(index), len(columns)),
                dtype=dtype)
        return cls(tb,
                index=index,
                columns=columns,
                name=name,
                own_data=True,
                own_index=True,
                own_columns=True)

    @classmethod
    def from_element_loc_items(cls,
            items: tp.Iterable[tp.Tuple[
                    tp.Tuple[tp.Hashable, tp.Hashable], tp.Any]],
            *,
            index: IndexInitializer,
            columns: IndexInitializer,
            dtype=None,
            name: tp.Hashable = None,
            fill_value: object = FILL_VALUE_DEFAULT,
            index_constructor: IndexConstructor = None,
            columns_constructor: IndexConstructor = None,
            own_index: bool = False,
            own_columns: bool = False,
            ) -> 'Frame':
        '''
        This function is partialed (setting the index and columns) and used by ``IterNodeDelegate`` as the apply constructor for doing application on element iteration.

        Args:s
            items: an iterable of pairs of 2-tuples of row, column loc labels and values.


        Returns:
            :obj:`static_frame.Frame`
        '''
        if not own_index:
            index = index_from_optional_constructor(index,
                    default_constructor=Index,
                    explicit_constructor=index_constructor
                    )

        if not own_columns:
            columns = index_from_optional_constructor(columns,
                    default_constructor=cls._COLUMNS_CONSTRUCTOR,
                    explicit_constructor=columns_constructor
                    )

        items = (((index.loc_to_iloc(k[0]), columns.loc_to_iloc(k[1])), v)
                for k, v in items)

        dtype = dtype if dtype is not None else object

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
                own_index=True, # always true as either provided or created new
                own_columns=True
                )

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
            dtypes: DtypesSpecifier = None,
            name: tp.Hashable = None,
            consolidate_blocks: bool = False,
            ) -> 'Frame':
        '''
        Frame constructor from an SQL query and a database connection object.

        Args:
            query: A query string.
            connection: A DBAPI2 (PEP 249) Connection object, such as those returned from SQLite (via the sqlite3 module) or PyODBC.
            {dtypes}
            {name}
            {consolidate_blocks}
        '''
        row_gen = connection.execute(query)

        own_columns = False
        columns = None
        if columns_depth == 1:
            columns = cls._COLUMNS_CONSTRUCTOR(b[0] for b in row_gen.description[index_depth:])
            own_columns = True
        elif columns_depth > 1:
            # use IH: get via static attr of columns const
            constructor = cls._COLUMNS_HIERARCHY_CONSTRUCTOR.from_labels_delimited
            labels = (b[0] for b in row_gen.description[index_depth:])
            columns = constructor(labels, delimiter=' ')
            own_columns = True

        index_constructor = None

        if index_depth > 0:
            index = [] # lazily populate
            if index_depth == 1:
                index_constructor = Index

                def row_gen_final() -> tp.Iterator[tp.Sequence[tp.Any]]:
                    for row in row_gen:
                        index.append(row[0])
                        yield row[1:]

            else: # > 1
                index_constructor = IndexHierarchy.from_labels

                def row_gen_final() -> tp.Iterator[tp.Sequence[tp.Any]]:
                    for row in row_gen:
                        index.append(row[:index_depth])
                        yield row[index_depth:]
        else:
            index = None
            row_gen_final = lambda: row_gen

        # let default type induction do its work
        return cls.from_records(
                row_gen_final(),
                columns=columns,
                index=index,
                dtypes=dtypes,
                name=name,
                own_columns=own_columns,
                index_constructor=index_constructor,
                consolidate_blocks=consolidate_blocks,
                )

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
            columns_depth: int = 1,
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
            columns_depth: Specify the number of rows after the skip_header used to create the column labels. A value of 0 will be no header; a value greater than 1 will attempt to create a hierarchical index.
            skip_header: Number of leading lines to skip.
            skip_footer: Number of trailing lines to skip.
            store_filter: A StoreFilter instance, defining translation between unrepresentable types. Presently only the ``to_nan`` attributes is used.
            {dtypes}
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`static_frame.Frame`
        '''
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html


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
                columns_arrays.append(columns_array.tolist()[index_depth:])

            if columns_depth == 1:
                columns_constructor = cls._COLUMNS_CONSTRUCTOR
                columns = columns_constructor(columns_arrays[0])
                own_columns = True
            else:
                columns_constructor = cls._COLUMNS_HIERARCHY_CONSTRUCTOR.from_labels
                columns = columns_constructor(
                        zip(*(store_filter.to_type_filter_iterable(x) for x in columns_arrays))
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
                    columns = columns
                    )
        else: # only column data in table
            if index_depth > 0:
                raise ErrorInitFrame(f'no data from which to extract index_depth {index_depth}')
            data = FRAME_INITIALIZER_DEFAULT

        kwargs = dict(
                data=data,
                own_data=True,
                columns=columns,
                own_columns=own_columns,
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
    def from_csv(cls,
            fp: PathSpecifierOrFileLikeOrIterator,
            *,
            index_depth: int = 0,
            index_column_first: tp.Optional[tp.Union[int, str]] = None,
            columns_depth: int = 1,
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
        Specialized version of :py:meth:`Frame.from_delimited` for CSV files.

        Returns:
            :obj:`static_frame.Frame`
        '''
        return cls.from_delimited(fp,
                delimiter=',',
                index_depth=index_depth,
                index_column_first=index_column_first,
                columns_depth=columns_depth,
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
            columns_depth: int = 1,
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
        Specialized version of :py:meth:`Frame.from_delimited` for TSV files.

        Returns:
            :obj:`static_frame.Frame`
        '''
        return cls.from_delimited(fp,
                delimiter='\t',
                index_depth=index_depth,
                index_column_first=index_column_first,
                columns_depth=columns_depth,
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
    def from_xlsx(cls,
            fp: PathSpecifier,
            *,
            label: tp.Optional[str] = None,
            index_depth: int = 0,
            columns_depth: int = 1,
            dtypes: DtypesSpecifier = None,
            consolidate_blocks: bool = False,
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
                columns_depth=columns_depth,
                dtypes=dtypes,
                consolidate_blocks=consolidate_blocks,
                )
        return st.read(label, config=config, container_type=cls)

    @classmethod
    def from_sqlite(cls,
            fp: PathSpecifier,
            *,
            label: tp.Optional[str] = None,
            index_depth: int = 0,
            columns_depth: int = 1,
            dtypes: DtypesSpecifier = None,
            consolidate_blocks: bool = False,
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
        return st.read(label, config=config, container_type=cls)

    @classmethod
    def from_hdf5(cls,
            fp: PathSpecifier,
            *,
            label: str,
            index_depth: int = 0,
            columns_depth: int = 1,
            consolidate_blocks: bool = False,
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
        return st.read(label, config=config, container_type=cls)

    @classmethod
    @doc_inject()
    def from_pandas(cls,
            value: 'pandas.DataFrame',
            *,
            index_constructor: IndexConstructor = None,
            columns_constructor: IndexConstructor = None,
            consolidate_blocks: bool = False,
            own_data: bool = False
            ) -> 'Frame':
        '''Given a Pandas DataFrame, return a Frame.

        Args:
            value: Pandas DataFrame.
            {own_data}

        Returns:
            :obj:`static_frame.Frame`
        '''
        pdvu1 = pandas_version_under_1()

        def part_to_array(part):
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
            pairs = value.dtypes.items()
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
                    # use loc to select before calling .values
                    part = value.loc[NULL_SLICE, slice(column_start, column_last)]
                    yield part_to_array(part)

                    column_start = column
                    dtype_current = dtype
                    yield_block = False

                column_last = column

            # always have left over
            part = value.loc[NULL_SLICE, slice(column_start, None)]
            yield part_to_array(part)

        if consolidate_blocks:
            blocks = TypeBlocks.from_blocks(TypeBlocks.consolidate_blocks(blocks()))
        else:
            blocks = TypeBlocks.from_blocks(blocks())

        # avoid getting a Series if a column
        if 'name' not in value.columns and hasattr(value, 'name'):
            name = value.name
        else:
            name = None

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
    def from_arrow(cls,
            value: 'pyarrow.Table',
            *,
            index_depth: int = 0,
            columns_depth: int = 1,
            consolidate_blocks: bool = False,
            name: tp.Hashable = None
            ) -> 'Frame':
        '''Convert an Arrow Table into a Frame.
        '''
        # this is similar to from_structured_array
        index_start_pos = -1 # will be ignored
        index_end_pos = -1
        if index_depth > 0:
            index_start_pos = 0
            index_end_pos = index_start_pos + index_depth - 1

        index_arrays = []
        if columns_depth > 0:
            columns = []

        def blocks():
            for col_idx, (name, chunked_array) in enumerate(
                    zip(value.column_names, value.columns)):
                # This creates a Series with an index; better to find a way to go only to numpy, but does not seem available on ChunkedArray
                array_final = chunked_array.to_pandas(
                        ignore_metadata=True).values
                if col_idx >= index_start_pos and col_idx <= index_end_pos:
                    index_arrays.append(array_final)
                    continue
                if columns_depth > 0:
                    columns.append(name)
                yield array_final

        if consolidate_blocks:
            block_gen = lambda: TypeBlocks.consolidate_blocks(blocks())
        else:
            block_gen = blocks

        columns_constructor = None
        if columns_depth == 0:
            columns = None
        elif columns_depth > 1:
            columns_constructor = partial(
                    cls._COLUMNS_HIERARCHY_CONSTRUCTOR.from_labels_delimited,
                    delimiter=' ')

        kwargs = dict(
                data=TypeBlocks.from_blocks(block_gen()),
                own_data=True,
                columns=columns,
                columns_constructor=columns_constructor,
                name=name
                )

        if index_depth == 0:
            return cls(index=None, **kwargs)
        if index_depth == 1:
            return cls(index=index_arrays[0], **kwargs)
        return cls(
                index=zip(*index_arrays),
                index_constructor=IndexHierarchy.from_labels,
                **kwargs
                )

    @classmethod
    def from_parquet(cls,
            fp: PathSpecifier,
            *,
            index_depth: int = 0,
            columns_depth: int = 1,
            consolidate_blocks: bool = False,
            name: tp.Hashable = None,
            ) -> 'Frame':
        '''
        Realize a ``Frame`` from a Parquet file.
        '''
        import pyarrow.parquet as pq

        table = pq.read_table(fp)
        return cls.from_arrow(table,
                index_depth=index_depth,
                columns_depth=columns_depth,
                consolidate_blocks=consolidate_blocks,
                name=name
                )

    #---------------------------------------------------------------------------
    def __init__(self,
            data: FrameInitializer = FRAME_INITIALIZER_DEFAULT,
            *,
            index: tp.Union[IndexInitializer, IndexAutoFactoryType] = None,
            columns: tp.Union[IndexInitializer, IndexAutoFactoryType] = None,
            name: tp.Hashable = None,
            index_constructor: IndexConstructor = None,
            columns_constructor: IndexConstructor = None,
            own_data: bool = False,
            own_index: bool = False,
            own_columns: bool = False
            ) -> None:
        # doc string at class def

        self._name = name if name is None else name_filter(name)

        # we can determine if columns or index are empty only if they are not iterators; those cases will have to use a deferred evaluation
        columns_empty = index_constructor_empty(columns)
        index_empty = index_constructor_empty(index)

        #-----------------------------------------------------------------------
        # blocks assignment

        blocks_constructor = None

        if isinstance(data, TypeBlocks):
            if own_data:
                self._blocks = data
            else:
                # assume we need to create a new TB instance; this will not copy underlying arrays as all blocks are immutable
                self._blocks = TypeBlocks.from_blocks(data._blocks)

        elif isinstance(data, np.ndarray):
            if own_data:
                data.flags.writeable = False
            # from_blocks will apply immutable filter
            self._blocks = TypeBlocks.from_blocks(data)

        elif data is FRAME_INITIALIZER_DEFAULT:
            # NOTE: this will not catch all cases where index or columns is empty, as they might be iterators; those cases will be handled below.
            def blocks_constructor(shape): #pylint: disable=E0102
                if shape[0] > 0 and shape[1] > 0:
                    # if fillable and we still have default initializer, this is a problem
                    raise RuntimeError('must supply a non-default value for constructing a Frame with non-zero size.')
                self._blocks = TypeBlocks.from_zero_size_shape(shape)

        elif isinstance(data, dict):
            raise ErrorInitFrame('use Frame.from_dict to create a Frame from a mapping.')
        elif isinstance(data, Series):
            raise ErrorInitFrame('use Frame.from_series to create a Frame from a Series.')
        else:
            raise ErrorInitFrame('use Frame.from_element, Frame.from_elements, or Frame.from_records to create a Frame from 0, 1, or 2 dimensional untyped data (respectively).')

        # counts can be zero (not None) if _block was created but is empty
        row_count, col_count = (self._blocks._shape
                if not blocks_constructor else (None, None))

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
                    explicit_constructor=index_constructor
                    )
        else:
            self._index = index_from_optional_constructor(index,
                    default_constructor=Index,
                    explicit_constructor=index_constructor
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
            # if we have a blocks_constructor, we are determining final size from index and/or columns; we might have a legitamate single value for data, but it cannot be FRAME_INITIALIZER_DEFAULT
            if data is not FRAME_INITIALIZER_DEFAULT and (
                    columns_empty or index_empty):
                raise ErrorInitFrame('cannot supply a data argument to Frame constructor when index or columns is empty')
            # must update the row/col counts, sets self._blocks
            blocks_constructor((row_count, col_count))

        # final check of block/index coherence

        if self._blocks.ndim != self._NDIM:
            raise ErrorInitFrame('dimensionality of final values not supported')

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
    # name interface

    @property
    @doc_inject()
    def name(self) -> tp.Hashable:
        '''{}'''
        return self._name

    def rename(self, name: tp.Hashable) -> 'Frame':
        '''
        Return a new Frame with an updated name attribute.
        '''
        # copying blocks does not copy underlying data
        return self.__class__(self._blocks.copy(),
                index=self._index,
                columns=self._columns, # let constructor handle if GO
                name=name,
                own_data=True,
                own_index=True)

    #---------------------------------------------------------------------------
    # interfaces

    @property
    def loc(self) -> InterfaceGetItem:
        return InterfaceGetItem(self._extract_loc)

    @property
    def iloc(self) -> InterfaceGetItem:
        return InterfaceGetItem(self._extract_iloc)

    @property
    def bloc(self) -> InterfaceGetItem:
        return InterfaceGetItem(self._extract_bloc)

    @property
    def drop(self) -> InterfaceSelection2D:
        return InterfaceSelection2D(
            func_iloc=self._drop_iloc,
            func_loc=self._drop_loc,
            func_getitem=self._drop_getitem)

    @property
    def mask(self) -> InterfaceSelection2D:
        return InterfaceSelection2D(
            func_iloc=self._extract_iloc_mask,
            func_loc=self._extract_loc_mask,
            func_getitem=self._extract_getitem_mask)

    @property
    def masked_array(self) -> InterfaceSelection2D:
        return InterfaceSelection2D(
            func_iloc=self._extract_iloc_masked_array,
            func_loc=self._extract_loc_masked_array,
            func_getitem=self._extract_getitem_masked_array)

    @property
    def assign(self) -> InterfaceAssign2D:
        # all functions that return a FrameAssign
        return InterfaceAssign2D(
            func_iloc=self._extract_iloc_assign,
            func_loc=self._extract_loc_assign,
            func_getitem=self._extract_getitem_assign,
            func_bloc=self._extract_bloc_assign
            )

    @property
    @doc_inject(select='astype')
    def astype(self) -> InterfaceAsType:
        '''
        Retype one or more columns. Can be used as as function to retype the entire ``Frame``; alternatively, a ``__getitem__`` interface permits retyping selected columns.

        Args:
            {dtype}
        '''
        return InterfaceAsType(func_getitem=self._extract_getitem_astype)

    # generators
    @property
    def iter_array(self) -> IterNodeAxis:
        '''
        Iterator of 1D NumPy array, where arrays are drawn from columns (axis=0) or rows (axis=1)
        '''
        return IterNodeAxis(
            container=self,
            function_values=self._axis_array,
            function_items=self._axis_array_items,
            yield_type=IterNodeType.VALUES
            )

    @property
    def iter_array_items(self) -> IterNodeAxis:
        '''
        Iterator of pairs of label, 1D NumPy array, where arrays are drawn from columns (axis=0) or rows (axis=1)
        '''
        return IterNodeAxis(
            container=self,
            function_values=self._axis_array,
            function_items=self._axis_array_items,
            yield_type=IterNodeType.ITEMS
            )

    @property
    def iter_tuple(self) -> IterNodeAxis:
        return IterNodeAxis(
            container=self,
            function_values=self._axis_tuple,
            function_items=self._axis_tuple_items,
            yield_type=IterNodeType.VALUES
            )

    @property
    def iter_tuple_items(self) -> IterNodeAxis:
        return IterNodeAxis(
            container=self,
            function_values=self._axis_tuple,
            function_items=self._axis_tuple_items,
            yield_type=IterNodeType.ITEMS
            )

    @property
    def iter_series(self) -> IterNodeAxis:
        return IterNodeAxis(
            container=self,
            function_values=self._axis_series,
            function_items=self._axis_series_items,
            yield_type=IterNodeType.VALUES
            )

    @property
    def iter_series_items(self) -> IterNodeAxis:
        return IterNodeAxis(
            container=self,
            function_values=self._axis_series,
            function_items=self._axis_series_items,
            yield_type=IterNodeType.ITEMS
            )

    #---------------------------------------------------------------------------
    @property
    def iter_group(self) -> IterNodeGroupAxis:
        '''
        Iterate over Frames grouped by unique values in one or more rows or columns.
        '''
        return IterNodeGroupAxis(
            container=self,
            function_values=self._axis_group_loc,
            function_items=self._axis_group_loc_items,
            yield_type=IterNodeType.VALUES,
            apply_type=IterNodeApplyType.SERIES_ITEMS_FLAT,
            )

    @property
    def iter_group_items(self) -> IterNodeGroupAxis:
        return IterNodeGroupAxis(
            container=self,
            function_values=self._axis_group_loc,
            function_items=self._axis_group_loc_items,
            yield_type=IterNodeType.ITEMS,
            apply_type=IterNodeApplyType.SERIES_ITEMS_FLAT,
            )

    @property
    def iter_group_labels(self) -> IterNodeDepthLevelAxis:
        return IterNodeDepthLevelAxis(
            container=self,
            function_values=self._axis_group_labels,
            function_items=self._axis_group_labels_items,
            yield_type=IterNodeType.VALUES,
            apply_type=IterNodeApplyType.SERIES_ITEMS_FLAT,
            )

    @property
    def iter_group_labels_items(self) -> IterNodeDepthLevelAxis:
        return IterNodeDepthLevelAxis(
            container=self,
            function_values=self._axis_group_labels,
            function_items=self._axis_group_labels_items,
            yield_type=IterNodeType.ITEMS,
            apply_type=IterNodeApplyType.SERIES_ITEMS_FLAT,
            )

    #---------------------------------------------------------------------------

    @property
    def iter_window(self) -> IterNodeWindow:
        function_values = partial(self._axis_window, as_array=False)
        function_items = partial(self._axis_window_items, as_array=False)
        return IterNodeWindow(
            container=self,
            function_values=function_values,
            function_items=function_items,
            yield_type=IterNodeType.VALUES
            )

    @property
    def iter_window_items(self) -> IterNodeWindow:
        function_values = partial(self._axis_window, as_array=False)
        function_items = partial(self._axis_window_items, as_array=False)
        return IterNodeWindow(
            container=self,
            function_values=function_values,
            function_items=function_items,
            yield_type=IterNodeType.ITEMS
            )

    @property
    def iter_window_array(self) -> IterNodeWindow:
        function_values = partial(self._axis_window, as_array=True)
        function_items = partial(self._axis_window_items, as_array=True)
        return IterNodeWindow(
            container=self,
            function_values=function_values,
            function_items=function_items,
            yield_type=IterNodeType.VALUES
            )

    @property
    def iter_window_array_items(self) -> IterNodeWindow:
        function_values = partial(self._axis_window, as_array=True)
        function_items = partial(self._axis_window_items, as_array=True)
        return IterNodeWindow(
            container=self,
            function_values=function_values,
            function_items=function_items,
            yield_type=IterNodeType.ITEMS
            )

    #---------------------------------------------------------------------------
    @property
    def iter_element(self) -> IterNodeNoArg:
        return IterNodeNoArg(
            container=self,
            function_values=self._iter_element_loc,
            function_items=self._iter_element_loc_items,
            yield_type=IterNodeType.VALUES,
            apply_type=IterNodeApplyType.FRAME_ELEMENTS
            )

    @property
    def iter_element_items(self) -> IterNodeNoArg:
        return IterNodeNoArg(
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
            fill_value=np.nan
            ) -> 'Frame':
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
            fill_value=np.nan,
            own_index: bool = False,
            own_columns: bool = False
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
            if isinstance(index, IndexBase):
                if not own_index:
                    # always use the Index constructor for safe reuse when poss[ible
                    if not index.STATIC:
                        index = index._IMMUTABLE_CONSTRUCTOR(index)
                    else:
                        index = index.__class__(index)
            else: # create the Index if not already an index, assume 1D
                index = Index(index)
            index_ic = IndexCorrespondence.from_correspondence(self._index, index)
            own_index_frame = True
        else:
            index = self._index
            index_ic = None
            # cannot own self._index, need a new index on Frame construction
            own_index_frame = False

        if columns is not None:
            if isinstance(columns, IndexBase):
                # always use the Index constructor for safe reuse when possible
                if not own_columns:
                    if columns.STATIC != self._COLUMNS_CONSTRUCTOR.STATIC:
                        columns_constructor = columns._IMMUTABLE_CONSTRUCTOR
                    else:
                        columns_constructor = columns.__class__
                    columns = columns_constructor(columns)
            else: # create the Index if not already an columns, assume 1D
                columns = self._COLUMNS_CONSTRUCTOR(columns)
            columns_ic = IndexCorrespondence.from_correspondence(self._columns, columns)
            own_columns_frame = True
        else:
            columns = self._columns
            columns_ic = None
            # if static, can own
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

        own_columns = False
        if columns is IndexAutoFactory:
            columns = None
        elif is_callable_or_mapping(columns):
            columns = self._columns.relabel(columns)
            own_columns = True
        elif columns is None:
            columns = self._columns

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

    @doc_inject(selector='relabel_add_level', class_name='Frame')
    def relabel_add_level(self,
            index: tp.Hashable = None,
            columns: tp.Hashable = None
            ) -> 'Frame':
        '''
        {doc}

        Args:
            index: {level}
            columns: {level}
        '''

        index = self._index.add_level(index) if index else self._index.copy()
        columns = self._columns.add_level(columns) if columns else self._columns.copy()

        return self.__class__(
                self._blocks.copy(), # does not copy arrays
                index=index,
                columns=columns,
                name=self._name,
                own_data=True,
                own_index=True,
                own_columns=True)

    @doc_inject(selector='relabel_drop_level', class_name='Frame')
    def relabel_drop_level(self,
            index: int = 0,
            columns: int = 0
            ) -> 'Frame':
        '''
        {doc}

        Args:
            index: {count} Default is zero.
            columns: {count} Default is zero.
        '''

        index = self._index.drop_level(index) if index else self._index.copy()
        columns = self._columns.drop_level(columns) if columns else self._columns.copy()

        return self.__class__(
                self._blocks.copy(), # does not copy arrays
                index=index,
                columns=columns,
                name=self._name,
                own_data=True,
                own_index=True,
                own_columns=True)

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
            index, index_iloc = rehierarch_and_map(
                    labels=self._index.values,
                    depth_map=index,
                    index_constructor=self._index.from_labels,
                    name=self._index.name
                    )
        else:
            index = self._index
            index_iloc = None

        if columns:
            columns, columns_iloc = rehierarch_and_map(
                    labels=self._columns.values,
                    depth_map=columns,
                    index_constructor=self._columns.from_labels,
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
            axis: int = 0,
            condition: tp.Callable[[np.ndarray], bool] = np.all) -> 'Frame':
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

    @doc_inject(selector='fillna')
    def fillna(self, value: tp.Any) -> 'Frame':
        '''Return a new ``Frame`` after replacing null (NaN or None) with the supplied value.

        Args:
            {value}
        '''
        if hasattr(value, '__iter__') and not isinstance(value, str):
            if not isinstance(value, Frame):
                raise RuntimeError('unlabeled iterables cannot be used for fillna: use a Frame')
            # not sure what fill_value is best here, as value Frame might have hetergenous types; this might result in some undesirable type coercion
            fill_value = dtype_to_na(value._blocks._row_dtype)

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
                self._blocks.fillna(fill, fill_valid),
                index=self._index,
                columns=self._columns,
                name=self._name,
                own_data=True
                )

    @doc_inject(selector='fillna')
    def fillna_leading(self,
            value: tp.Any,
            *,
            axis: int = 0) -> 'Frame':
        '''
        Return a new ``Frame`` after filling leading (and only leading) null (NaN or None) with the supplied value.

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
        Return a new ``Frame`` after filling trailing (and only trailing) null (NaN or None) with the supplied value.

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
        Return a new ``Frame`` after filling forward null (NaN or None) with the supplied value.

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
        Return a new ``Frame`` after filling backward null (NaN or None) with the supplied value.

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
            config: tp.Optional[DisplayConfig] = None
            ) -> Display:
        '''{doc}

        Args:
            {config}
        '''
        config = config or DisplayActive.get()

        # create an empty display, then populate with index
        d = Display([[]],
                config=config,
                outermost=True,
                index_depth=self._index.depth,
                header_depth=self._columns.depth + 2)

        display_index = self._index.display(config=config)
        d.extend_display(display_index)

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
                d.extend_iterable(column, header='')

        #-----------------------------------------------------------------------
        # prepare columns display
        config_transpose = config.to_transpose()
        display_cls = Display.from_values((),
                header=DisplayHeader(self.__class__, self._name),
                config=config_transpose)

        # need to apply the column config such that it truncates it based on the the max columns, not the max rows
        display_columns = self._columns.display(
                config=config_transpose)

        # add spacers to from of columns when we have a hierarchical index
        for _ in range(self._index.depth - 1):
            # will need a width equal to the column depth
            row = [Display.to_cell('', config=config)
                    for _ in range(self._columns.depth)]
            spacer = Display([row])
            display_columns.insert_displays(spacer,
                    insert_index=1) # after the first, the name

        if self._columns.depth > 1:
            display_columns_horizontal = display_columns.transform()
        else: # can just flatten a single column into one row
            display_columns_horizontal = display_columns.flatten()

        #-----------------------------------------------------------------------
        d.insert_displays(
                display_cls.flatten(),
                display_columns_horizontal,
                )
        return d

    def _repr_html_(self):
        '''
        Provide HTML representation for Jupyter Notebooks.
        '''
        # modify the active display to be for HTML
        config = DisplayActive.get(
                display_format=DisplayFormats.HTML_TABLE,
                type_show=False
                )
        return repr(self.display(config))

    #---------------------------------------------------------------------------
    # accessors

    @property
    def values(self) -> np.ndarray:
        '''A 2D array of values. Note: type coercion might be necessary.
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

    def __bool__(self) -> bool:
        '''
        True if this container has size.
        '''
        return bool(self._blocks.size)



    #---------------------------------------------------------------------------
    @staticmethod
    def _extract_axis_not_multi(row_key, column_key) -> tp.Tuple[bool, bool]:
        '''
        If either row or column is given with a non-multiple type of selection (a single scalar), reduce dimensionality.
        '''
        row_nm = False
        column_nm = False
        if row_key is not None and not isinstance(row_key, KEY_MULTIPLE_TYPES):
            row_nm = True # axis 0
        if column_key is not None and not isinstance(column_key, KEY_MULTIPLE_TYPES):
            column_nm = True # axis 1
        return row_nm, column_nm


    def _extract(self,
            row_key: GetItemKeyType = None,
            column_key: GetItemKeyType = None) -> tp.Union['Frame', Series]:
        '''
        Extract based on iloc selection (indices have already mapped)
        '''
        blocks = self._blocks._extract(row_key=row_key, column_key=column_key)

        if not isinstance(blocks, TypeBlocks):
            return blocks # reduced to an element

        own_index = True # the extracted Frame can always own this index
        row_key_is_slice = isinstance(row_key, slice)
        if row_key is None or (row_key_is_slice and row_key == NULL_SLICE):
            index = self._index
        else:
            index = self._index._extract_iloc(row_key)
            if not row_key_is_slice:
                name_row = self._index.values[row_key]
                if self._index.depth > 1:
                    name_row = tuple(name_row)

        # can only own columns if _COLUMNS_CONSTRUCTOR is static
        column_key_is_slice = isinstance(column_key, slice)
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
            # return a 0-sized Series
            if axis_nm[0]: # if row not multi
                return Series(EMPTY_TUPLE,
                        index=immutable_index_filter(columns),
                        name=name_row)
            elif axis_nm[1]:
                return Series(EMPTY_TUPLE,
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
            iloc_column_key = self._columns.loc_to_iloc(loc_column_key)
        else:
            loc_row_key = key
            iloc_column_key = None

        iloc_row_key = self._index.loc_to_iloc(loc_row_key)
        return iloc_row_key, iloc_column_key

    def _compound_loc_to_getitem_iloc(self,
            key: GetItemKeyTypeCompound) -> tp.Tuple[GetItemKeyType, GetItemKeyType]:
        '''Handle a potentially compound key in the style of __getitem__. This will raise an appropriate exception if a two argument loc-style call is attempted.
        '''
        # if isinstance(key, tuple) and self._columns.depth == 1:
        #     raise KeyError('__getitem__ does not support multiple indexers on a 1D Index')
        iloc_column_key = self._columns.loc_to_iloc(key)
        return None, iloc_column_key


    def _extract_loc(self, key: GetItemKeyTypeCompound) -> 'Frame':
        iloc_row_key, iloc_column_key = self._compound_loc_to_iloc(key)
        return self._extract(
                row_key=iloc_row_key,
                column_key=iloc_column_key
                )


    def _extract_bloc(self, key: Bloc2DKeyType) -> Series:
        '''
        2D Boolean selector, selected by either a Boolean 2D Frame or array.
        '''
        bloc_key = bloc_key_normalize(key=key, container=self)
        values = self.values[bloc_key]
        values.flags.writeable = False

        index = Index(
                (self._index[x], self._columns[y])
                for x, y in zip(*np.nonzero(bloc_key))
                )
        return Series(values, index=index, own_index=True)


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
    def _extract_iloc_assign(self, key: GetItemKeyTypeCompound) -> 'FrameAssign':
        return FrameAssign(self, iloc_key=key)

    def _extract_loc_assign(self, key: GetItemKeyTypeCompound) -> 'FrameAssign':
        # extract if tuple, then pack back again
        key = self._compound_loc_to_iloc(key)
        return self._extract_iloc_assign(key=key)

    def _extract_getitem_assign(self, key: GetItemKeyTypeCompound) -> 'FrameAssign':
        # extract if tuple, then pack back again
        key = self._compound_loc_to_getitem_iloc(key)
        return self._extract_iloc_assign(key=key)

    def _extract_bloc_assign(self, key: Bloc2DKeyType) -> 'FrameAssign':
        '''Assignment based on a Boolean Frame or array.'''
        return FrameAssign(self, bloc_key=key)

    #---------------------------------------------------------------------------

    def _extract_getitem_astype(self, key: GetItemKeyType) -> 'FrameAsType':
        # extract if tuple, then pack back again
        _, key = self._compound_loc_to_getitem_iloc(key)
        return FrameAsType(self, column_key=key)



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

    def items(self) -> tp.Iterator[tp.Tuple[tp.Any, Series]]:
        '''Iterator of pairs of column label and corresponding column :obj:`Series`.
        '''
        for label, array in zip(self._columns.values, self._blocks.axis_values(0)):
            # array is assumed to be immutable
            yield label, Series(array, index=self._index, name=label)

    def get(self, key, default=None):
        '''
        Return the value found at the columns key, else the default if the key is not found. This method is implemented to complete the dictionary-like interface.
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

    def _ufunc_binary_operator(self, *,
            operator,
            other
            ) -> 'Frame':

        if operator.__name__ == 'matmul':
            return matmul(self, other)
        elif operator.__name__ == 'rmatmul':
            return matmul(other, self)

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
            axis,
            skipna,
            ufunc,
            ufunc_skipna,
            composable: bool,
            dtypes: tp.Tuple[np.dtype, ...],
            size_one_unity: bool
            ) -> 'Frame':
        # axis 0 processes ros, deliveres column index
        # axis 1 processes cols, delivers row index
        assert axis < 2

        dtype = None if not dtypes else dtypes[0]

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
            Tuple = get_tuple_constructor(self._columns.values)
            # Tuple = namedtuple('Axis', self._columns.values)
        elif axis == 0:
            Tuple = get_tuple_constructor(self._index.values)
            # Tuple = namedtuple('Axis', self._index.values)
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
        # reference the indices and let the constructor reuse what is reusable
        if axis == 1:
            index = self._columns
            labels = self._index
        elif axis == 0:
            index = self._index
            labels = self._columns
        for label, axis_values in zip(labels, self._blocks.axis_values(axis)):
            yield Series(axis_values, index=index, name=label)

    def _axis_series_items(self, axis):
        keys = self._index if axis == 1 else self._columns
        yield from zip(keys, self._axis_series(axis=axis))


    #---------------------------------------------------------------------------
    # grouping methods naturally return their "index" as the group element

    def _axis_group_iloc_items(self,
            key,
            *,
            axis: int):

        for group, selection, tb in self._blocks.group(axis=axis, key=key):

            if axis == 0:
                # axis 0 is a row iter, so need to slice index, keep columns
                yield group, self.__class__(tb,
                        index=self._index[selection],
                        columns=self._columns, # let constructor determine ownership
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
            key,
            iloc_key,
            axis: int
            ) -> tp.Generator[tp.Tuple[tp.Hashable, 'Frame'], None, None]:
        # Create a sorted copy since we do not want to change the underlying data
        frame_sorted: Frame = self.sort_values(key, axis=not axis)

        def build_frame(key, index):
            if axis == 0:
                return Frame(frame_sorted._blocks._extract(row_key=key),
                        columns=self.columns,
                        index=index,
                        own_data=True)
            else:
                return Frame(frame_sorted._blocks._extract(column_key=key),
                        columns=index,
                        index=self.index,
                        own_data=True)

        if axis == 0:
            max_iloc: int = len(self._index)
            index: Index = frame_sorted.index
            def get_group(i: int) -> tp.Hashable:
                return frame_sorted.iloc[i, iloc_key]
        else:
            max_iloc = len(self._columns)
            index = frame_sorted.columns
            def get_group(i: int) -> tp.Hashable:
                return frame_sorted.iloc[iloc_key, i]

        group: tp.Hashable = get_group(0)
        start = 0
        i = 0

        while i < max_iloc:
            next_group: tp.Hashable = get_group(i)

            if group != next_group:
                slc: slice = slice(start, i)
                sliced_index: Index = index[slc]
                yield group, build_frame(slc, sliced_index)

                start = i
                group = next_group
            i += 1

        yield group, build_frame(slice(start, None), index[start:])


    def _axis_group_loc_items(self,
            key: GetItemKeyType,
            *,
            axis: int = 0
            ):
        '''
        Args:
            key: We accept any thing that can do loc to iloc. Note that a tuple is permitted as key, where it would be interpreted as a single label for an IndexHierarchy.
        '''
        if axis == 0: # row iterator, selecting columns for group by
            iloc_key = self._columns.loc_to_iloc(key)
        elif axis == 1: # column iterator, selecting rows for group by
            iloc_key = self._index.loc_to_iloc(key)
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
            ):
        yield from (x for _, x in self._axis_group_loc_items(key=key, axis=axis))


    def _axis_group_labels_items(self,
            depth_level: DepthLevelSpecifier = 0,
            *,
            axis=0):

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
            axis=0):
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
            ):
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

    def _iter_element_iloc_items(self):
        yield from self._blocks.element_items()

    # def _iter_element_iloc(self):
    #     yield from (x for _, x in self._iter_element_iloc_items())

    def _iter_element_loc_items(self) -> tp.Iterator[
            tp.Tuple[tp.Tuple[tp.Hashable, tp.Hashable], tp.Any]]:
        '''
        Generator of pairs of (index, column), value.
        '''
        yield from (
                ((self._index[k[0]], self._columns[k[1]]), v)
                for k, v in self._blocks.element_items()
                )

    def _iter_element_loc(self):
        yield from (x for _, x in self._iter_element_loc_items())


    #---------------------------------------------------------------------------
    # transformations resulting in the same dimensionality

    def __reversed__(self) -> tp.Iterator[tp.Hashable]:
        '''
        Returns a reverse iterator on the frame's columns.
        '''
        return reversed(self._columns)

    def sort_index(self,
            *,
            ascending: bool = True,
            kind: str = DEFAULT_SORT_KIND
            ) -> 'Frame':
        '''
        Return a new Frame ordered by the sorted Index.
        '''
        if self._index.depth > 1:
            v = self._index.values
            order = np.lexsort([v[:, i] for i in range(v.shape[1]-1, -1, -1)])
        else:
            # argsort lets us do the sort once and reuse the results
            order = np.argsort(self._index.values, kind=kind)

        if not ascending:
            order = order[::-1]

        index_values = self._index.values[order]
        index_values.flags.writeable = False
        index = self._index.from_labels(index_values, name=self._index.name)

        blocks = self._blocks.iloc[order]
        return self.__class__(blocks,
                index=index,
                columns=self._columns,
                name=self._name,
                own_data=True,
                own_index=True,
                )

    def sort_columns(self,
            *,
            ascending: bool = True,
            kind: str = DEFAULT_SORT_KIND) -> 'Frame':
        '''
        Return a new Frame ordered by the sorted Columns.
        '''
        if self._columns.depth > 1:
            v = self._columns.values
            order = np.lexsort([v[:, i] for i in range(v.shape[1]-1, -1, -1)])
        else:
            # argsort lets us do the sort once and reuse the results
            order = np.argsort(self._columns.values, kind=kind)

        if not ascending:
            order = order[::-1]

        columns_values = self._columns.values[order]
        columns_values.flags.writeable = False
        columns = self._columns.from_labels(columns_values,  name=self._columns.name)

        blocks = self._blocks[order]
        return self.__class__(blocks,
                index=self._index,
                columns=columns,
                name=self._name,
                own_data=True,
                own_columns=True,
                )

    def sort_values(self,
            key: KeyOrKeys,
            *,
            ascending: bool = True,
            axis: int = 1,
            kind=DEFAULT_SORT_KIND) -> 'Frame':
        '''
        Return a new Frame ordered by the sorted values, where values is given by single column or iterable of columns.

        Args:
            key: a key or iterable of keys.
        '''
        # argsort lets us do the sort once and reuse the results
        if axis == 0: # get a column ordering based on one or more rows
            col_count = self._columns.__len__()
            if is_hashable(key) and key in self._index:
                iloc_key = self._index.loc_to_iloc(key)
                sort_array = self._blocks._extract_array(row_key=iloc_key)
                order = np.argsort(sort_array, kind=kind)
            else: # assume an iterable of keys
                # order so that highest priority is last
                iloc_keys = (self._index.loc_to_iloc(k) for k in reversed_iter(key))
                sort_array = [self._blocks._extract_array(row_key=key)
                        for key in iloc_keys]
                order = np.lexsort(sort_array)
        elif axis == 1: # get a row ordering based on one or more columns
            if is_hashable(key) and key in self._columns:
                iloc_key = self._columns.loc_to_iloc(key)
                sort_array = self._blocks._extract_array(column_key=iloc_key)
                order = np.argsort(sort_array, kind=kind)
            else: # assume an iterable of keys
                # order so that highest priority is last
                iloc_keys = (self._columns.loc_to_iloc(k) for k in reversed_iter(key))
                sort_array = [self._blocks._extract_array(column_key=key)
                        for key in iloc_keys]
                order = np.lexsort(sort_array)
        else:
            raise AxisInvalid(f'invalid axis: {axis}')

        if not ascending:
            order = order[::-1]

        if axis == 0:
            column_values = self._columns.values[order]
            column_values.flags.writeable = False
            columns = self._columns.from_labels(
                    column_values,
                    name=self._columns._name
                    )
            blocks = self._blocks[order]
            return self.__class__(blocks,
                    index=self._index,
                    columns=columns,
                    name=self._name,
                    own_data=True,
                    own_columns=True
                    )

        index_values = self._index.values[order]
        index_values.flags.writeable = False
        index = self._index.from_labels(
                index_values,
                name=self._index._name
                )
        blocks = self._blocks.iloc[order]
        return self.__class__(blocks,
                index=index,
                columns=self._columns,
                name=self._name,
                own_data=True,
                own_index=True
                )

    def isin(self, other) -> 'Frame':
        '''
        Return a same-sized Boolean Frame that shows if the same-positioned element is in the iterable passed to the function.
        '''
        array = isin(self.values, other)
        return self.__class__(array, columns=self._columns, index=self._index)

    @doc_inject(class_name='Frame')
    def clip(self, *,
            lower=None,
            upper=None,
            axis: tp.Optional[int] = None):
        '''{}

        Args:
            lower: value, :obj:`static_frame.Series`, :obj:`static_frame.Frame`
            upper: value, :obj:`static_frame.Series`, :obj:`static_frame.Frame`
            axis: required if ``lower`` or ``upper`` are given as a :obj:`static_frame.Series`.
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
                    args[idx] = np.vstack([values] * self.shape[1]).T
                else:
                    args[idx] = np.vstack([values] * self.shape[0])
            elif isinstance(arg, Frame):
                args[idx] = arg.reindex(
                        index=self._index,
                        columns=self._columns).fillna(bound).values
            elif hasattr(arg, '__iter__'):
                raise RuntimeError('only Series or Frame are supported as iterable lower/upper arguments')
            # assume single value otherwise, no change necessary

        array = np.clip(self.values, *args)
        array.flags.writeable = False

        return self.__class__(array,
                columns=self._columns,
                index=self._index,
                name=self._name
                )


    def transpose(self) -> 'Frame':
        '''Transpose. Return a :obj:`static_frame.Frame` with ``index`` as ``columns`` and vice versa.
        '''
        return self.__class__(self._blocks.transpose(),
                index=self._columns,
                columns=self._index,
                own_data=True,
                name=self._name)

    @property
    def T(self) -> 'Frame':
        '''Transpose. Return a :obj:`static_frame.Frame` with ``index`` as ``columns`` and vice versa.
        '''
        return self.transpose()

    @doc_inject(selector='duplicated')
    def duplicated(self, *,
            axis=0,
            exclude_first=False,
            exclude_last=False) -> 'Series':
        '''
        Return an axis-sized Boolean Series that shows True for all rows (axis 0) or columns (axis 1) duplicated.

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
            axis=0,
            exclude_first: bool = False,
            exclude_last: bool = False
            ) -> 'Frame':
        '''
        Return a Frame with duplicated rows (axis 0) or columns (axis 1) removed. All values in the row or column are compared to determine duplication.

        Args:
            {axis}
            {exclude_first}
            {exclude_last}
        '''
        # NOTE: can avoid calling .vaalues with extensions to TypeBlocks
        duplicates = array_to_duplicated(self.values,
                axis=axis,
                exclude_first=exclude_first,
                exclude_last=exclude_last)

        if not duplicates.any():
            return self.__class__(
                    self._blocks.copy(),
                    index=self._index,
                    columns=self._columns,
                    own_data=True,
                    own_index=True,
                    name=self._name)

        keep = ~duplicates

        if axis == 0: # return rows with index indexed
            return self.__class__(
                    self.values[keep],
                    index=self._index[keep],
                    columns=self._columns,
                    own_index=True,
                    name=self._name
                    )
        elif axis == 1:
            return self.__class__(
                    self.values[:, keep],
                    index=self._index,
                    columns=self._columns[keep],
                    own_index=True,
                    name=self._name
                    )
        # invalid axis will raise in array_to_duplicated

    def set_index(self,
            column: GetItemKeyType,
            *,
            drop: bool = False,
            index_constructor=Index) -> 'Frame':
        '''
        Return a new frame produced by setting the given column as the index, optionally removing that column from the new Frame.
        '''
        column_iloc = self._columns.loc_to_iloc(column)

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

        index_values = self._blocks._extract_array(column_key=column_iloc)
        index = index_constructor(index_values, name=column)

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

        column_iloc = self._columns.loc_to_iloc(column_loc)

        if column_name is None:
            column_name = tuple(self._columns.values[column_iloc])

        index_labels = self._blocks._extract_array(column_key=column_iloc)

        if reorder_for_hierarchy:
            index, order_lex = rehierarch_and_map(
                    labels=index_labels,
                    depth_map=range(index_labels.shape[1]), # keep order
                    index_constructor=IndexHierarchy.from_labels,
                    index_constructors=index_constructors,
                    name=column_name,
                    )
            blocks_src = self._blocks._extract(row_key=order_lex)
        else:
            index = IndexHierarchy.from_labels(index_labels,
                    index_constructors=index_constructors,
                    name=column_name,
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
        Return a new ``Frame`` where the index is added to the front of the data, and an ``IndexAutoFactory`` is used to populate a new index. If the ``Index`` has a ``name``, that name will be used for the column name, otherwise a suitable default will be used. As underlying NumPy arrays are immutable, data is not copied.

        Args:
            names: An iterable of hashables to be used to name the unset index. If an ``Index``, a single hashable should be provided; if an ``IndexHierarchy``, as many hashables as the depth must be provided.
        '''

        def blocks():
            yield self.index.values # 2D immutable array
            for b in self._blocks._blocks:
                yield b

        if consolidate_blocks:
            block_gen = lambda: TypeBlocks.consolidate_blocks(blocks())
        else:
            block_gen = blocks

        if self._columns.depth > 1:
            raise ErrorInitFrame('cannot unset index with a columns with depth greater than 1')

        if names:
            columns = chain(names, self._columns.values)
        else:
            columns = chain(self._index.names, self._columns.values)

        return self.__class__(
                TypeBlocks.from_blocks(block_gen()),
                columns=columns,
                index=None,
                own_data=True,
                )


    def roll(self,
            index: int = 0,
            columns: int = 0,
            include_index: bool = False,
            include_columns: bool = False) -> 'Frame':
        '''
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
            fill_value=np.nan) -> 'Frame':

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

    def pivot(self,
            index_fields: KeyOrKeys,
            columns_fields: KeyOrKeys = EMPTY_TUPLE,
            data_fields: KeyOrKeys = EMPTY_TUPLE,
            *,
            func: CallableOrCallableMap = None,
            fill_value: object = FILL_VALUE_DEFAULT,
            ) -> 'Frame':
        '''
        Produce a pivot table, where one or more columns is selected for each of index_fields, columns_fields, and data_fields. Unique values from the provided ``index_fields`` will be used to create a new index; unique values from the provided ``columns_fields`` will be used to create a new columns; if one ``data_fields`` value is selected, that is the value that will be displayed; if more than one values is given, those values will be presented with a hierarchical index on the columns; if not ``data_fields`` ar provided, all unused fields will be displayed.

        Args:
            index_fields
            columns_fields
            data_fields
            fill_value: If the index expansion produces coordinates that have no existing data value, fill that position with this value.
            func: function to apply to ``data_fields``, or a dictionary of labelled functions to apply to data fields, producing an additional hierarchical level.
        '''
        if func is None:
            # form the equivalent Series function for summing
            func = partial(ufunc_axis_skipna,
                    skipna=True,
                    axis=0,
                    ufunc=np.sum,
                    ufunc_skipna=np.nansum
                    )
            func_map = (('', func),)
        elif callable(func):
            func_map = (('', func),) # store iterable of pairs
        else:
            func_map = tuple(func.items())

        index_fields = key_normalize(index_fields)
        columns_fields = key_normalize(columns_fields)
        data_fields = key_normalize(data_fields)

        if not data_fields:
            used = set(chain(index_fields, columns_fields))
            data_fields = [x for x in self.columns if x not in used]
            if not data_fields:
                raise ErrorInitFrame('no fields remain to populate data.')

        idx_start_columns = len(index_fields)

        # Take fields_group before extending columns with values
        fields_group = index_fields + columns_fields
        for field in fields_group:
            if field not in self._columns:
                raise ErrorInitFrame(f'cannot create a pivot Frame from a field ({field}) that is not a column')

        if idx_start_columns == 1:
            # reduce loc to single itme to get 1D array
            index_loc = index_fields[0]
        else:
            index_loc = index_fields
        # will return np.array or frozen set
        index_values = ufunc_unique(
                self._blocks._extract_array(
                        column_key=self._columns.loc_to_iloc(index_loc)),
                axis=0)

        if idx_start_columns == 1:
            index = Index(index_values, name=index_fields[0])
        else:
            index = IndexHierarchy.from_labels(index_values, name=tuple(index_fields))

        # Colect bundle of values for from_product constrcution if columns.
        columns_product = []
        for field in columns_fields:
            # Take one at a time
            columns_values = ufunc_unique(
                    self._blocks._extract_array(column_key=self._columns.loc_to_iloc(field)))
            columns_product.append(columns_values)

        # For data fields, we add the field name, not the field values, to the columns.
        columns_name = tuple(columns_fields)
        if len(data_fields) > 1 or not columns_fields:
            # if no columns fields, have to add values fields
            columns_product.append(data_fields)
            columns_name = tuple(chain(*columns_fields, ('values',)))

        if len(func_map) > 1:
            # add the labels as another product level
            labels = tuple(x for x, _ in func_map)
            columns_product.append(labels)
            columns_name = columns_name + ('func',)

        if len(columns_product) > 1:
            columns = self._COLUMNS_HIERARCHY_CONSTRUCTOR.from_product(
                    *columns_product,
                    name=columns_name)
        else:
            columns = self._COLUMNS_CONSTRUCTOR(
                    columns_product[0],
                    name=columns_name[0])

        def items():
            func_single = func_map[0][1] if len(func_map) == 1 else None

            for group, sub in self.iter_group_items(fields_group):
                index_label = group[:idx_start_columns]
                index_label = tuple(index_label) if len(index_label) > 1 else index_label[0]

                # get the found parts of the columns labels from the group; this will never have data_fields
                columns_label_raw = group[idx_start_columns:]

                if len(columns_label_raw) == 0:
                    # if none, it means we just have data fields
                    columns_labels = data_fields
                elif len(columns_label_raw) == 1 and len(data_fields) == 1:
                    # only one data field, do not need to display it
                    columns_labels = repeat(columns_label_raw[0])
                elif len(columns_label_raw) > 1 and len(data_fields) == 1:
                    # only one data field
                    columns_labels = repeat(tuple(columns_label_raw))
                elif len(columns_label_raw) >= 1 and len(data_fields) > 1:
                    # create column labels that has an entry for each data field
                    if len(columns_label_raw) == 1:
                        columns_labels = ((columns_label_raw[0], v) for v in data_fields)
                    else:
                        columns_labels = (tuple(chain(columns_label_raw, (v,))) for v in data_fields)

                for field, column_label in zip(data_fields, columns_labels):
                    # NOTE: sub[field] produces a Series, which is not needed; better to go to blocks and extract array
                    if func_single:
                        yield (index_label, column_label), func_single(sub[field].values)
                    else:
                        for label, func in func_map:
                            if isinstance(column_label, tuple):
                                column_label_final = column_label + (label,)
                            else: # a single hashable
                                column_label_final = (column_label, label)
                            yield (index_label, column_label_final), func(sub[field].values)

        return self.__class__.from_element_loc_items(
                items(),
                index=index,
                columns=columns,
                dtype=None, # supply if possible,
                fill_value=fill_value,
                own_index=True,
                own_columns=True
                )



    #---------------------------------------------------------------------------
    # utility function to numpy array

    def unique(self, axis: tp.Optional[int] = None) -> np.ndarray:
        '''
        Return a NumPy array of unqiue values. If the axis argument is provied, uniqueness is determined by columns or row.
        '''
        return ufunc_unique(self.values, axis=axis)

    #---------------------------------------------------------------------------
    # exporters

    def to_pairs(self, axis) -> tp.Iterable[
            tp.Tuple[tp.Hashable, tp.Iterable[tp.Tuple[tp.Hashable, tp.Any]]]]:
        '''
        Return a tuple of major axis key, minor axis key vlaue pairs, where major axis is determined by the axis argument.
        '''
        # TODO: find a common interfave on IndexHierarchy that cna give hashables
        if isinstance(self._index, IndexHierarchy):
            index_values = list(array2d_to_tuples(self._index.values))
        else:
            index_values = self._index.values

        if isinstance(self._columns, IndexHierarchy):
            columns_values = list(array2d_to_tuples(self._columns.values))
        else:
            columns_values = self._columns.values

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

        df = pandas.DataFrame(index=self._index.to_pandas())

        # iter columns to preserve types
        # use integer columns for initial loading
        for i, array in enumerate(self._blocks.axis_values(0)):
            df[i] = array

        df.columns = self._columns.to_pandas()

        if 'name' not in df.columns and self._name is not None:
            df.name = self._name
        return df

    def to_arrow(self,
            *,
            include_index: bool = True,
            include_columns: bool = True,
            ) -> 'pyarrow.Table':
        '''
        Return a ``pyarrow.Table`` from this :obj:`Frame`.
        '''
        import pyarrow
        from static_frame.core.store import Store

        field_names, _ = Store.get_field_names_and_dtypes(
                frame=self,
                include_index=include_index,
                include_columns=include_columns,
                force_str_names=True
                )
        arrays = tuple(Store.get_column_iterator(
                frame=self,
                include_index=include_index)
                )
        # field_names have to be strings
        return pyarrow.Table.from_arrays(arrays, names=field_names)


    def to_parquet(self,
            fp: PathSpecifier,
            *,
            include_index: bool = True,
            include_columns: bool = True,
            ) -> None:
        '''
        Write an Arrow Parquet binary file.
        '''
        import pyarrow.parquet as pq

        table = self.to_arrow()
        fp = path_filter(fp)
        pq.write_table(table, fp)

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
                for c in self.iter_series(axis=0):
                    # dtype must be able to accomodate a float NaN
                    resolved = resolve_dtype(c.dtype, DTYPE_FLOAT_DEFAULT)
                    # create multidimensional arsdfray of all axis for each
                    array = np.full(
                            shape=[len(coords[v]) for v in coords],
                            fill_value=np.nan,
                            dtype=resolved)

                    for index_labels, value in c.items():
                        # translate to index positions
                        insert_pos = [coords_index[k].loc_to_iloc(label)
                                for k, label in zip(coords, index_labels)]
                        # must convert to tuple to give position per dimension
                        array[tuple(insert_pos)] = value

                    yield array

        data_vars = {k: (index_name, v)
                for k, v in zip(columns_values, columns_arrays())}

        return xarray.Dataset(data_vars, coords=coords)

    def to_frame_go(self) -> 'FrameGO':
        '''
        Return a FrameGO view of this Frame. As underlying data is immutable, this is a no-copy operation.
        '''
        # copying blocks does not copy underlying data
        return FrameGO(
                self._blocks.copy(),
                index=self.index, # can reuse
                columns=self.columns,
                columns_constructor=self.columns._MUTABLE_CONSTRUCTOR,
                name=self._name,
                own_data=True,
                own_index=True,
                own_columns=False # need to make grow only
                )

    def to_delimited(self,
            fp: PathSpecifierOrFileLike,
            *,
            delimiter: str,
            include_index: bool = True,
            include_columns: bool = True,
            encoding: tp.Optional[str] = None,
            line_terminator: str = '\n',
            store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ):
        '''
        Given a file path or file-like object, write the Frame as delimited text.

        Args:
            delimiter: character to be used for delimiterarating elements.
        '''
        fp = path_filter(fp)

        if isinstance(fp, str):
            f = open(fp, 'w', encoding=encoding)
            is_file = True
        else:
            f = fp # assume an open file like
            is_file = False

        index = self._index
        columns = self._columns

        if include_index:
            index_values = index.values # get once for caching
            index_names = index.names # normalized presentation

        if store_filter:
            filter_func = store_filter.from_type_filter_element

        try: # manage finally closing of file
            if include_columns:
                if columns.depth == 1:
                    columns_rows = (columns,)
                else:
                    columns_rows = columns.values.T
                for row_idx, columns_row in enumerate(columns_rows):
                    if include_index:
                        for name in index_names:
                            if row_idx == 0:
                                f.write(f'{name}{delimiter}')
                            else:
                                f.write(f'{delimiter}')
                    if store_filter:
                        f.write(delimiter.join(f'{filter_func(x)}' for x in columns_row))
                    else:
                        f.write(delimiter.join(f'{x}' for x in columns_row))
                    f.write(line_terminator)

            col_idx_last = self._blocks._shape[1] - 1
            # avoid row creation to avoid joining types; avoide creating a list for each row
            row_current_idx = None
            for (row_idx, col_idx), element in self._iter_element_iloc_items():
                if row_idx != row_current_idx:
                    if row_current_idx is not None:
                        f.write(line_terminator)
                    if include_index:
                        if index.depth == 1:
                            index_value = index_values[row_idx]
                            if store_filter:
                                f.write(f'{filter_func(index_value)}{delimiter}')
                            else:
                                f.write(f'{index_value}{delimiter}')
                        else:
                            for index_value in index_values[row_idx]:
                                if store_filter:
                                    f.write(f'{filter_func(index_value)}{delimiter}')
                                else:
                                    f.write(f'{index_value}{delimiter}')

                    row_current_idx = row_idx
                if store_filter:
                    f.write(f'{filter_func(element)}')
                else:
                    f.write(f'{element}')
                if col_idx != col_idx_last:
                    f.write(delimiter)
        except: #pragma: no cover
            raise #pragma: no cover
        finally:
            if is_file:
                f.close()
        if is_file:
            f.close()


    def to_csv(self,
            fp: PathSpecifierOrFileLike,
            *,
            include_index: bool = True,
            include_columns: bool = True,
            encoding: tp.Optional[str] = None,
            line_terminator: str = '\n',
            store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ):
        '''
        Given a file path or file-like object, write the Frame as tab-delimited text.
        '''
        return self.to_delimited(fp=fp,
                delimiter=',',
                include_index=include_index,
                include_columns=include_columns,
                encoding=encoding,
                line_terminator=line_terminator,
                store_filter=store_filter
                )

    def to_tsv(self,
            fp: PathSpecifierOrFileLike,
            *,
            include_index: bool = True,
            include_columns: bool = True,
            encoding: tp.Optional[str] = None,
            line_terminator: str = '\n',
            store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ):
        '''
        Given a file path or file-like object, write the Frame as tab-delimited text.
        '''
        return self.to_delimited(fp=fp,
                delimiter='\t',
                include_index=include_index,
                include_columns=include_columns,
                encoding=encoding,
                line_terminator=line_terminator,
                store_filter=store_filter
                )


    def to_xlsx(self,
            fp: PathSpecifier, # not sure I can take a file like yet
            *,
            label: tp.Optional[str] = None,
            include_index: bool = True,
            include_columns: bool = True,
            merge_hierarchical_labels: bool = True
            ) -> None:
        '''
        Write the Frame as single-sheet XLSX file.
        '''
        from static_frame.core.store_xlsx import StoreXLSX
        from static_frame.core.store import StoreConfig

        config = StoreConfig(
                include_index=include_index,
                include_columns=include_columns,
                merge_hierarchical_labels=merge_hierarchical_labels
                )
        st = StoreXLSX(fp)
        st.write(((label, self),), config=config)


    def to_sqlite(self,
            fp: PathSpecifier, # not sure file-like StringIO works
            *,
            label: tp.Optional[str] = None,
            include_index: bool = True,
            include_columns: bool = True,
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
        st.write(((label, self),), config=config)

    def to_hdf5(self,
            fp: PathSpecifier, # not sure file-like StringIO works
            *,
            label: tp.Optional[str] = None,
            include_index: bool = True,
            include_columns: bool = True,
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

        if not label:
            if not self.name:
                raise RuntimeError('must provide a label or define Frame name.')
            label = self.name

        st = StoreHDF5(fp)
        st.write(((label, self),), config=config)


    @doc_inject(class_name='Frame')
    def to_html(self,
            config: tp.Optional[DisplayConfig] = None
            ) -> str:
        '''
        {}
        '''
        # if a config is given, try to use all settings; if using active, hide types
        config = config or DisplayActive.get(type_show=False)
        config = config.to_display_config(
                display_format=DisplayFormats.HTML_TABLE,
                )
        return repr(self.display(config))

    @doc_inject(class_name='Frame')
    def to_html_datatables(self,
            fp: tp.Optional[PathSpecifierOrFileLike] = None,
            show: bool = True,
            config: tp.Optional[DisplayConfig] = None
            ) -> str:
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
    '''A two-dimensional, ordered, labelled collection, immutable with grow-only columns. Initialization arguments are the same as for :obj:`Frame`.
    '''

    __slots__ = (
            '_blocks',
            '_columns',
            '_index',
            '_name'
            )

    _COLUMNS_CONSTRUCTOR = IndexGO
    _COLUMNS_HIERARCHY_CONSTRUCTOR = IndexHierarchyGO


    def __setitem__(self,
            key: tp.Hashable,
            value: tp.Any,
            fill_value=np.nan
            ) -> None:
        '''For adding a single column, one column at a time.
        '''
        if key in self._columns:
            raise RuntimeError(f'The provided key ({key}) is already defined in columns; if you want to change or replace this column, use .assign to get new Frame')

        row_count = len(self._index)

        if isinstance(value, Series):
            # select only the values matching our index
            block = value.reindex(self.index, fill_value=fill_value).values

        elif isinstance(value, np.ndarray): # is numpy array
            # this permits unaligned assignment as no index is used, possibly remove
            if value.ndim != 1 or len(value) != row_count:
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
            fill_value=np.nan):
        '''
        Given an iterable of pairs of column name, column value, extend this FrameGO.
        '''
        for k, v in pairs:
            self.__setitem__(k, v, fill_value)


    def extend(self,
            container: tp.Union['Frame', Series],
            fill_value=np.nan
            ):
        '''Extend this FrameGO (in-place) with another Frame's blocks or Series array; as blocks are immutable, this is a no-copy operation when indices align. If indices do not align, the passed-in Frame or Series will be reindexed (as happens when adding a column to a FrameGO).

        If a Series is passed in, the column name will be taken from the Series ``name`` attribute.

        This method differs from FrameGO.extend_items() by permitting contiguous underlying blocks to be extended from another Frame into this Frame.
        '''
        if not isinstance(container, (Series, Frame)):
            raise NotImplementedError(
                    f'no support for extending with {type(container)}')

        if not len(container.index): # must be empty data, empty index container
            return

        # self's index will never change; we only take what aligns in the passed container
        if _requires_reindex(self._index, container._index):
            container = container.reindex(self._index, fill_value=fill_value)

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
    def to_frame(self) -> Frame:
        '''
        Return Frame version of this Frame.
        '''
        # copying blocks does not copy underlying data
        return Frame(self._blocks.copy(),
                index=self.index,
                columns=self.columns.values,
                name=self._name,
                own_data=True,
                own_index=True,
                own_columns=False # need to make static only
                )

    def to_frame_go(self) -> 'FrameGO':
        '''
        Return a FrameGO version of this Frame.
        '''
        raise ErrorInitFrame('This Frame is already a FrameGO')


#-------------------------------------------------------------------------------
# utility delegates returned from selection routines and exposing the __call__ interface.

class FrameAssign:
    __slots__ = (
        'container',
        'iloc_key',
        'bloc_key',
        )

    def __init__(self,
            container: Frame,
            iloc_key: GetItemKeyTypeCompound = None,
            bloc_key: tp.Optional[Bloc2DKeyType] = None,
            ) -> None:
        '''Store a reference to ``Frame``, as well as a key to be used for assignment with ``__call__``
        '''
        self.container = container
        self.iloc_key = iloc_key
        self.bloc_key = bloc_key

        if not (self.iloc_key is not None) ^ (self.bloc_key is not None):
            raise RuntimeError('must set only one of ``iloc_key``, ``bloc_key``')

    def __call__(self,
            value,
            fill_value: tp.Any = np.nan
            ) -> 'Frame':
        '''
        Called with ``file_value`` to execute assignment configured fromthe init key.
        '''
        if self.iloc_key is not None:
            # NOTE: the iloc key's order is not relevant in assignment
            if isinstance(value, (Series, Frame)):
                if isinstance(value, Series):
                    iloc_key = self.iloc_key
                elif isinstance(value, Frame):
                    # block assignment requires that column keys are ascending
                    iloc_key = (self.iloc_key[0],
                            key_to_ascending_key(self.iloc_key[1], self.container.shape[1]))
                # conform the passed in value to the targets given by self.iloc_key
                assigned_container = self.container._reindex_other_like_iloc(value,
                        iloc_key,
                        fill_value=fill_value)
                # NOTE: taking .values here forces a single-type array from Frame
                assigned = assigned_container.values
            else: # could be array or single element
                iloc_key = self.iloc_key
                assigned = value

            blocks = self.container._blocks.extract_iloc_assign(iloc_key, assigned)

        else: # use bloc
            bloc_key = bloc_key_normalize(
                    key=self.bloc_key,
                    container=self.container
                    )

            if isinstance(value, Frame):
                invalid = object()
                value = value.reindex(
                        index=self.container._index,
                        columns=self.container._columns,
                        fill_value=invalid
                        ).values

                # if we produced any invalid entries, cannot select them
                invalid_found = value == invalid
                if invalid_found.any():
                    bloc_key = bloc_key.copy() # mutate a copy
                    bloc_key[invalid_found] = False

            elif isinstance(value, np.ndarray):
                if value.shape != self.container.shape:
                    raise RuntimeError(f'value must match shape {self.container.shape}')

            blocks = self.container._blocks.extract_bloc_assign(bloc_key, value)


        return self.container.__class__(
                data=blocks,
                columns=self.container._columns,
                index=self.container._index,
                name=self.container._name,
                own_data=True
                )


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

    def __call__(self, dtype, consolidate_blocks: bool = True) -> 'Frame':

        blocks = self.container._blocks._astype_blocks(self.column_key, dtype)

        if consolidate_blocks:
            blocks = TypeBlocks.consolidate_blocks(blocks)

        blocks = TypeBlocks.from_blocks(blocks)

        return self.container.__class__(
                data=blocks,
                columns=self.container.columns,
                index=self.container.index,
                name=self.container._name,
                own_data=True)
