from __future__ import annotations

import csv
import json
import pickle
import sqlite3
from collections import deque
from collections.abc import Set
from copy import deepcopy
from dataclasses import is_dataclass
from functools import partial
from io import BytesIO
from io import StringIO
from itertools import chain
from itertools import product
from itertools import zip_longest
from operator import itemgetter

import numpy as np
import typing_extensions as tp
from arraykit import array_to_tuple_array
from arraykit import array_to_tuple_iter
from arraykit import column_1d_filter
from arraykit import delimited_to_arrays
from arraykit import first_true_2d
from arraykit import name_filter
from arraykit import resolve_dtype
from arraykit import resolve_dtype_iter
from arraykit import split_after_count
from numpy.ma import MaskedArray

from static_frame.core.archive_npy import NPYFrameConverter
from static_frame.core.archive_npy import NPZFrameConverter
from static_frame.core.assign import Assign
from static_frame.core.container import ContainerOperand
# from static_frame.core.container_util import pandas_version_under_1
from static_frame.core.container_util import ContainerMap
from static_frame.core.container_util import MessagePackElement
from static_frame.core.container_util import apex_to_name
from static_frame.core.container_util import array_from_value_iter
from static_frame.core.container_util import axis_window_items
from static_frame.core.container_util import bloc_key_normalize
from static_frame.core.container_util import constructor_from_optional_constructors
from static_frame.core.container_util import df_slice_to_arrays
from static_frame.core.container_util import frame_to_frame
from static_frame.core.container_util import get_col_dtype_factory
from static_frame.core.container_util import get_col_fill_value_factory
from static_frame.core.container_util import index_constructor_empty
from static_frame.core.container_util import index_from_optional_constructor
from static_frame.core.container_util import index_from_optional_constructors
from static_frame.core.container_util import index_many_concat
from static_frame.core.container_util import index_many_to_one
from static_frame.core.container_util import is_fill_value_factory_initializer
from static_frame.core.container_util import iter_component_signature_bytes
from static_frame.core.container_util import key_to_ascending_key
from static_frame.core.container_util import matmul
from static_frame.core.container_util import pandas_to_numpy
from static_frame.core.container_util import prepare_values_for_lex
from static_frame.core.container_util import rehierarch_from_index_hierarchy
from static_frame.core.container_util import rehierarch_from_type_blocks
from static_frame.core.container_util import sort_index_for_order
from static_frame.core.display import Display
from static_frame.core.display import DisplayActive
from static_frame.core.display import DisplayHeader
from static_frame.core.display_config import DisplayConfig
from static_frame.core.display_config import DisplayFormats
from static_frame.core.doc_str import doc_inject
from static_frame.core.doc_str import doc_update
from static_frame.core.exception import AxisInvalid
from static_frame.core.exception import ErrorInitColumns
from static_frame.core.exception import ErrorInitFrame
from static_frame.core.exception import ErrorInitIndex
from static_frame.core.exception import ErrorInitIndexNonUnique
from static_frame.core.exception import GrowOnlyInvalid
from static_frame.core.exception import InvalidFillValue
from static_frame.core.exception import RelabelInvalid
from static_frame.core.index import Index
from static_frame.core.index import IndexGO
from static_frame.core.index import _index_initializer_needs_init
from static_frame.core.index import immutable_index_filter
from static_frame.core.index_auto import IndexAutoFactory
from static_frame.core.index_auto import IndexDefaultConstructorFactory
from static_frame.core.index_auto import TIndexInitOrAuto
from static_frame.core.index_auto import TRelabelInput
from static_frame.core.index_base import IndexBase
from static_frame.core.index_correspondence import IndexCorrespondence
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.index_hierarchy import IndexHierarchyGO
from static_frame.core.join import join
from static_frame.core.metadata import JSONMeta
from static_frame.core.node_dt import InterfaceDatetime
from static_frame.core.node_fill_value import InterfaceFillValue
from static_frame.core.node_fill_value import InterfaceFillValueGO
from static_frame.core.node_iter import IterNodeApplyType
from static_frame.core.node_iter import IterNodeAxis
from static_frame.core.node_iter import IterNodeAxisElement
from static_frame.core.node_iter import IterNodeConstructorAxis
from static_frame.core.node_iter import IterNodeDepthLevelAxis
from static_frame.core.node_iter import IterNodeGroupAxis
from static_frame.core.node_iter import IterNodeGroupOtherReducible
from static_frame.core.node_iter import IterNodeWindowReducible
from static_frame.core.node_re import InterfaceRe
from static_frame.core.node_selector import InterfaceAssignQuartet
from static_frame.core.node_selector import InterfaceConsolidate
from static_frame.core.node_selector import InterfaceFrameAsType
from static_frame.core.node_selector import InterfaceGetItemBLoc
from static_frame.core.node_selector import InterfaceSelectTrio
from static_frame.core.node_selector import InterGetItemILocCompoundReduces
from static_frame.core.node_selector import InterGetItemLocCompoundReduces
from static_frame.core.node_selector import TFrameOrSeries
from static_frame.core.node_str import InterfaceString
from static_frame.core.node_transpose import InterfaceTranspose
from static_frame.core.node_values import InterfaceValues
from static_frame.core.pivot import pivot_derive_constructors
from static_frame.core.pivot import pivot_index_map
from static_frame.core.protocol_dfi import DFIDataFrame
from static_frame.core.rank import RankMethod
from static_frame.core.rank import rank_1d
from static_frame.core.series import Series
from static_frame.core.store_filter import STORE_FILTER_DEFAULT
from static_frame.core.store_filter import StoreFilter
from static_frame.core.style_config import STYLE_CONFIG_DEFAULT
from static_frame.core.style_config import StyleConfig
from static_frame.core.style_config import style_config_css_factory
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.type_blocks import group_match
from static_frame.core.type_blocks import group_sorted
from static_frame.core.util import BOOL_TYPES
from static_frame.core.util import CONTINUATION_TOKEN_INACTIVE
from static_frame.core.util import DEFAULT_FAST_SORT_KIND
from static_frame.core.util import DEFAULT_SORT_KIND
from static_frame.core.util import DEFAULT_STABLE_SORT_KIND
from static_frame.core.util import DT64_NS
from static_frame.core.util import DTU_PYARROW
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import DTYPE_DATETIME_KIND
from static_frame.core.util import DTYPE_FLOAT_DEFAULT
from static_frame.core.util import DTYPE_INT_DEFAULT
from static_frame.core.util import DTYPE_NA_KINDS
from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import DTYPE_OBJECT_KIND
from static_frame.core.util import DTYPE_TIMEDELTA_KIND
from static_frame.core.util import EMPTY_ARRAY
from static_frame.core.util import FILL_VALUE_DEFAULT
from static_frame.core.util import FRAME_INITIALIZER_DEFAULT
from static_frame.core.util import INT_TYPES
from static_frame.core.util import KEY_MULTIPLE_TYPES
from static_frame.core.util import NAME_DEFAULT
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import STORE_LABEL_DEFAULT
from static_frame.core.util import STRING_TYPES
from static_frame.core.util import IterNodeType
from static_frame.core.util import Join
from static_frame.core.util import JSONFilter
from static_frame.core.util import ManyToOneType
from static_frame.core.util import TBlocKey
from static_frame.core.util import TBoolOrBools
from static_frame.core.util import TCallableAny
from static_frame.core.util import TCallableOrCallableMap
from static_frame.core.util import TDepthLevel
from static_frame.core.util import TDtypeSpecifier
from static_frame.core.util import TDtypesSpecifier
from static_frame.core.util import TFrameInitializer
from static_frame.core.util import TILocSelector
from static_frame.core.util import TILocSelectorCompound
from static_frame.core.util import TILocSelectorMany
from static_frame.core.util import TILocSelectorOne
from static_frame.core.util import TIndexCtor
from static_frame.core.util import TIndexCtorSpecifier
from static_frame.core.util import TIndexCtorSpecifiers
from static_frame.core.util import TIndexHierarchyCtor
from static_frame.core.util import TIndexInitializer
from static_frame.core.util import TIndexSpecifier
from static_frame.core.util import TKeyOrKeys
from static_frame.core.util import TLabel
from static_frame.core.util import TLocSelector
from static_frame.core.util import TLocSelectorCompound
from static_frame.core.util import TLocSelectorMany
from static_frame.core.util import TName
from static_frame.core.util import TPathSpecifier
from static_frame.core.util import TPathSpecifierOrBinaryIO
from static_frame.core.util import TPathSpecifierOrTextIO
from static_frame.core.util import TPathSpecifierOrTextIOOrIterator
from static_frame.core.util import TSortKinds
from static_frame.core.util import TTupleCtor
from static_frame.core.util import TUFunc
from static_frame.core.util import WarningsSilent
from static_frame.core.util import argmax_2d
from static_frame.core.util import argmin_2d
from static_frame.core.util import array_to_duplicated
from static_frame.core.util import blocks_to_array_2d
from static_frame.core.util import concat_resolved
from static_frame.core.util import dtype_from_element
from static_frame.core.util import dtype_kind_to_na
from static_frame.core.util import dtype_to_fill_value
from static_frame.core.util import file_like_manager
from static_frame.core.util import full_for_fill
from static_frame.core.util import get_tuple_constructor
from static_frame.core.util import iloc_to_insertion_iloc
from static_frame.core.util import is_callable_or_mapping
from static_frame.core.util import is_dtype_specifier
from static_frame.core.util import isfalsy_array
from static_frame.core.util import isna_array
from static_frame.core.util import iterable_to_array_1d
from static_frame.core.util import iterable_to_array_nd
from static_frame.core.util import key_normalize
from static_frame.core.util import path_filter
from static_frame.core.util import ufunc_unique
from static_frame.core.util import ufunc_unique1d
from static_frame.core.util import ufunc_unique_enumerated
from static_frame.core.util import write_optional_file

if tp.TYPE_CHECKING:
    import pandas  # pragma: no cover
    import pyarrow  # pragma: no cover
    from xarray import Dataset  # pragma: no cover

    from static_frame.core.reduce import ReduceDispatchAligned  # pylint: disable=W0611,C0412 #pragma: no cover

    TNDArrayAny = np.ndarray[tp.Any, tp.Any] #pragma: no cover
    TDtypeAny = np.dtype[tp.Any] #pragma: no cover
    TOptionalArrayList = tp.Optional[tp.List[TNDArrayAny]] #pragma: no cover
    TIndexAny = Index[tp.Any] #pragma: no cover

TSeriesAny = Series[tp.Any, tp.Any]

def _NA_BLOCKS_CONSTRCTOR(shape: tp.Tuple[int, int]) -> None: ...

TVIndex = tp.TypeVar('TVIndex', bound=IndexBase, default=tp.Any) # pylint: disable=E1123
TVColumns = tp.TypeVar('TVColumns', bound=IndexBase, default=tp.Any) # pylint: disable=E1123
TVDtypes = tp.TypeVarTuple('TVDtypes', # pylint: disable=E1123
        default=tp.Unpack[tp.Tuple[tp.Any, ...]])

class Frame(ContainerOperand, tp.Generic[TVIndex, TVColumns, tp.Unpack[TVDtypes]]):
    '''A two-dimensional ordered, labelled collection, immutable and of fixed size.
    '''
    __slots__ = (
            '__weakref__',
            '_blocks',
            '_columns',
            '_index',
            '_name',
            )

    _blocks: TypeBlocks
    _columns: IndexBase
    _index: IndexBase
    _name: TLabel

    _COLUMNS_CONSTRUCTOR = Index
    _COLUMNS_HIERARCHY_CONSTRUCTOR = IndexHierarchy

    _NDIM: int = 2

    #---------------------------------------------------------------------------
    # constructors

    @classmethod
    def from_series(cls,
            series: TSeriesAny,
            *,
            name: TLabel = None,
            columns_constructor: TIndexCtorSpecifier = None,
            ) -> tp.Self:
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
    def _from_zero_size_shape(cls,
            *,
            index: tp.Optional[TIndexInitializer] = None,
            columns: tp.Optional[TIndexInitializer] = None,
            dtypes: TDtypesSpecifier = None,
            name: TLabel = None,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            own_index: bool = False,
            own_columns: bool = False
            ) -> tp.Self:
        '''
        Create a zero-sized Frame based on ``index`` or ``columns`` (though not both of size).
        '''
        if own_columns:
            columns_final = columns
        else:
            columns_final = index_from_optional_constructor(
                    columns if columns is not None else (),
                    default_constructor=cls._COLUMNS_CONSTRUCTOR,
                    explicit_constructor=columns_constructor
                    )
        if own_index:
            index_final = index
        else:
            index_final = index_from_optional_constructor(
                    index if index is not None else (),
                    default_constructor=Index,
                    explicit_constructor=index_constructor
                    )

        shape = (len(index_final), len(columns_final)) # type: ignore
        if shape[0] > 0 and shape[1] > 0:
            raise ErrorInitFrame('Cannot create zero-sized Frame from sized index and columns.')

        get_col_dtype = ((lambda x: None) if dtypes is None
                else get_col_dtype_factory(dtypes, columns)) #type: ignore

        return cls(TypeBlocks.from_zero_size_shape(shape, get_col_dtype),
                index=index_final,
                columns=columns_final,
                name=name,
                own_data=True,
                own_index=True,
                own_columns=True,
                )

    @classmethod
    def from_element(cls,
            element: tp.Any,
            *,
            index: TIndexInitializer,
            columns: TIndexInitializer,
            dtype: TDtypeSpecifier = None,
            name: TLabel = None,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            own_index: bool = False,
            own_columns: bool = False
            ) -> tp.Self:
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

        shape = (len(index_final), len(columns_final)) #type: ignore
        dtype = None if dtype is None else np.dtype(dtype)
        array = full_for_fill(
                dtype,
                shape,
                element,
                resolve_fill_value_dtype=dtype is None, # True means derive from fill value
                )
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
            index: TIndexInitOrAuto = None,
            columns: TIndexInitOrAuto = None,
            dtype: TDtypeSpecifier = None,
            name: TLabel = None,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            own_index: bool = False,
            own_columns: bool = False
            ) -> tp.Self:
        '''
        Create a Frame from an iterable of elements, to be formed into a ``Frame`` with a single column.
        '''
        # will be immutable
        array, _ = iterable_to_array_1d(elements, dtype=dtype)

        #-----------------------------------------------------------------------
        if own_columns:
            columns_final = columns
            col_count = len(columns_final) #type: ignore
        elif index_constructor_empty(columns):
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
        elif index_constructor_empty(index):
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
    def from_concat(cls: tp.Type[tp.Self],
            frames: tp.Iterable[tp.Union[TFrameAny, TSeriesAny]],
            *,
            axis: int = 0,
            union: bool = True,
            index: TIndexInitOrAuto = None,
            columns: TIndexInitOrAuto = None,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            name: TName = None,
            fill_value: tp.Any = np.nan,
            consolidate_blocks: bool = False,
            ) -> tp.Self:
        '''
        Concatenate multiple :obj:`Frame` or :obj:`Series` into a new :obj:`Frame`. If index or columns are provided and appropriately sized, the resulting :obj:`Frame` will use those indices. If the axis along concatenation (index for axis 0, columns for axis 1) is unique after concatenation, it will be preserved; otherwise, a new index or an :obj:`IndexAutoFactory` must be supplied.

        Args:
            frames: Iterable of Frames.
            axis: Integer specifying 0 to concatenate supplied Frames vertically (aligning on columns), 1 to concatenate horizontally (aligning on rows).
            union: If True, the union of the aligned indices is used; if False, the intersection is used.
            index: Optionally specify a new index.
            columns: Optionally specify new columns.
            index_constructor: Optionally apply a constructor to the derived or passed labels.
            columns_constructor: Optionally apply a constructor to the derived or passed labels.
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`static_frame.Frame`
        '''

        frame_seq: tp.List[TFrameAny] = []
        for f in frames:
            if isinstance(f, Frame):
                frame_seq.append(f)
            else:
                # NOTE: we need to determine if the name attr of the Series is to be used as a label; providing IndexAutoFactory will forbid the usage of the name attr; the name attr is assigned to index if axis is 0, to columns if axis is 1. If index/columns is provided, force Series.to_frame() to not try to use the name attr.
                index_to_frame = None
                columns_to_frame = None
                index_constructor_to_frame = index_constructor
                columns_constructor_to_frame = columns_constructor

                # vstack, Series will be row
                if axis == 0:
                    if index is not None: # if we have an index, do not use name
                        index_to_frame = IndexAutoFactory
                        index_constructor_to_frame = None
                # hstack, Series will be col
                if axis == 1:
                    if columns is not None: # if we have columns, do not use name
                        columns_to_frame = IndexAutoFactory
                        columns_constructor_to_frame = None
                frame_seq.append(
                        f.to_frame(axis,
                        index = index_to_frame,
                        index_constructor=index_constructor_to_frame,
                        columns=columns_to_frame,
                        columns_constructor=columns_constructor_to_frame,
                        ))
        own_index = False
        own_columns = False
        if not frame_seq:
            return cls(
                    index=index,
                    columns=columns,
                    name=name,
                    own_columns=own_columns,
                    own_index=own_index,
                    index_constructor=index_constructor,
                    columns_constructor=columns_constructor,
                    )

        if axis == 1: # stacks columns (extends rows horizontally)
            # index can be the same, columns must be redefined if not unique
            if columns is IndexAutoFactory:
                columns = None # let default creation happen
            elif columns is None:
                try:
                    columns = index_many_concat(
                            (f._columns for f in frame_seq),
                            cls._COLUMNS_CONSTRUCTOR,
                            columns_constructor,
                            )
                except ErrorInitIndexNonUnique as e:
                    raise ErrorInitFrame('Column names after horizontal concatenation are not unique; supply a columns argument or IndexAutoFactory.') from e
                own_columns = True

            if index is IndexAutoFactory:
                raise ErrorInitFrame('for axis 1 concatenation, index must be used for reindexing row alignment: IndexAutoFactory is not permitted')
            elif index is None:
                index = index_many_to_one(
                        (f._index for f in frame_seq),
                        Index,
                        ManyToOneType.UNION if union else ManyToOneType.INTERSECT,
                        index_constructor,
                        )
                own_index = True

            def blocks() -> tp.Iterator[TNDArrayAny]:
                for frame in frame_seq:
                    if not frame.index.equals(index):
                        frame = frame.reindex(index=index, # type: ignore
                                fill_value=fill_value,
                                check_equals=False,
                                )
                    for block in frame._blocks._blocks:
                        yield block

        elif axis == 0: # stacks rows (extends columns vertically)
            if index is IndexAutoFactory:
                index = None # let default creation happen
            elif index is None:
                try:
                    index = index_many_concat(
                            (f._index for f in frame_seq),
                            Index,
                            index_constructor,
                            )
                except ErrorInitIndexNonUnique as e:
                    raise ErrorInitFrame('Index names after vertical concatenation are not unique; supply an index argument or IndexAutoFactory.') from e
                own_index = True

            if columns is IndexAutoFactory:
                raise ErrorInitFrame('for axis 0 concatenation, columns must be used for reindexing and column alignment: IndexAutoFactory is not permitted')
            elif columns is None:
                columns = index_many_to_one(
                        (f._columns for f in frame_seq),
                        cls._COLUMNS_CONSTRUCTOR,
                        ManyToOneType.UNION if union else ManyToOneType.INTERSECT,
                        columns_constructor,
                        )
                own_columns = True

            def blocks() -> tp.Iterator[TNDArrayAny]:
                type_blocks = []
                previous_frame: tp.Optional[TFrameAny] = None
                block_compatible = True
                reblock_compatible = True

                for frame in frame_seq:
                    if not frame.columns.equals(columns):
                        frame = frame.reindex(columns=columns, # type: ignore
                                fill_value=fill_value,
                                check_equals=False,
                                )
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

        block_gen: tp.Callable[..., tp.Iterator[TNDArrayAny]]
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
                own_index=own_index,
                index_constructor=index_constructor,
                columns_constructor=columns_constructor,
                )

    @classmethod
    def from_concat_items(cls,
            items: tp.Iterable[tp.Tuple[TLabel, tp.Union[TFrameAny, TSeriesAny]]],
            *,
            axis: int = 0,
            union: bool = True,
            name: TName = None,
            fill_value: tp.Any = np.nan,
            index_constructor: tp.Optional[TIndexCtorSpecifier] = None,
            columns_constructor: tp.Optional[TIndexCtorSpecifier] = None,
            consolidate_blocks: bool = False,
            ) -> tp.Self:
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

        def gen() -> tp.Iterator[tp.Tuple[TLabel, IndexBase]]:
            # default index construction does not yield elements, but instead yield Index objects for more efficient IndexHierarchy construction
            yield_elements = True
            if axis == 0 and (index_constructor is None or isinstance(index_constructor, IndexDefaultConstructorFactory)):
                yield_elements = False
            elif axis == 1 and (columns_constructor is None or isinstance(columns_constructor, IndexDefaultConstructorFactory)):
                yield_elements = False

            for label, frame in items:
                # must normalize Series here to avoid down-stream confusion
                if isinstance(frame, Series):
                    frame = frame.to_frame(axis)

                frames.append(frame)
                if axis == 0:
                    if yield_elements:
                        yield from product((label,), frame._index) # pyright: ignore
                    else:
                        yield label, frame._index
                elif axis == 1:
                    if yield_elements:
                        yield from product((label,), frame._columns) # pyright: ignore
                    else:
                        yield label, frame._columns

                # we have already evaluated AxisInvalid

        if axis == 0:
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
                **kwargs # type: ignore
                )

    @classmethod
    def from_overlay(cls,
            containers: tp.Iterable[TFrameAny],
            *,
            index: tp.Optional[TIndexInitializer] = None,
            columns: tp.Optional[TIndexInitializer] = None,
            union: bool = True,
            name: TName = None,
            func: tp.Callable[[TNDArrayAny], TNDArrayAny] = isna_array,
            fill_value: tp.Any = FILL_VALUE_DEFAULT,
            ) -> tp.Self:
        '''
        Return a new :obj:`Frame` made by overlaying containers, filling in values with aligned values from subsequent containers. Values are filled based on a passed function that must return a Boolean array. By default, that function is `isna_array`, returning True for missing values (NaN and None).

        Args:
            containers: Iterable of :obj:`Frame`.
            index: An optional :obj:`Index`, :obj:`IndexHierarchy`, or index initializer, to be used as the index upon which all containers are aligned. :obj:`IndexAutoFactory` is not supported.
            columns: An optional :obj:`Index`, :obj:`IndexHierarchy`, or columns initializer, to be used as the columns upon which all containers are aligned. :obj:`IndexAutoFactory` is not supported.
            union: If True, and no ``index`` or ``columns`` argument is supplied, a union index or columns from ``containers`` will be used; if False, the intersection index or columns will be used.
            name:
            func: A function that takes an array and returns a same-sized Boolean array, where True indicates availability for insertion.
        '''
        if not hasattr(containers, '__len__'):
            containers = tuple(containers) # exhaust a generator

        if index is None:
            index = index_many_to_one(
                    (c.index for c in containers),
                    cls_default=Index,
                    many_to_one_type=ManyToOneType.UNION if union else ManyToOneType.INTERSECT,
                    )
        else:
            index = index_from_optional_constructor(index,
                    default_constructor=Index
                    )
        if columns is None:
            columns = index_many_to_one(
                    (c.columns for c in containers),
                    cls_default=cls._COLUMNS_CONSTRUCTOR,
                    many_to_one_type=ManyToOneType.UNION if union else ManyToOneType.INTERSECT,
                    )
        else:
            columns = index_from_optional_constructor(columns,
                    default_constructor=cls._COLUMNS_CONSTRUCTOR)

        fill_arrays = {} # NOTE: we will hash to NaN and NaT, but can assume we are using the same instance

        containers_iter = iter(containers)
        container = next(containers_iter)

        if fill_value is FILL_VALUE_DEFAULT:
            fill_value_reindex = dtype_kind_to_na(container._blocks._index.dtype.kind)
        else:
            fill_value_reindex = fill_value # just pass along even if FillValueAuto

        # get the first container
        post = frame_to_frame(container, cls).reindex(
                index=index,
                columns=columns,
                fill_value=fill_value_reindex,
                own_index=True,
                own_columns=True,
                )

        # we need a fill value that will be identified as a missing value by ``func`` on subsequent iterations, otherwise this fill value will not be identified as fillable
        get_col_fill_value: tp.Callable[..., tp.Any]
        if fill_value is FILL_VALUE_DEFAULT:
            get_col_fill_value = lambda _, dtype: dtype_kind_to_na(dtype.kind)
        else:
            get_col_fill_value = get_col_fill_value_factory(fill_value, columns)

        # dtype column mapping will not change
        dtypes = post.dtypes
        post_blocks = post._blocks

        for container in containers_iter:
            values = []
            index_match = container._index.equals(index)
            # iterate over reindexed, full dtypes; some containers will not have columns
            for col_count, (col, dtype_at_col) in enumerate(dtypes.items()):
                if col not in container:
                    # get fill value based on previous container
                    fill_value = get_col_fill_value(col_count, dtype_at_col)
                    # store fill_arrays for re-use
                    if fill_value not in fill_arrays:
                        array = np.full(len(index), fill_value)
                        array.flags.writeable = False
                        fill_arrays[fill_value] = array
                    array = fill_arrays[fill_value]
                elif index_match:
                    iloc_column_key = container._columns._loc_to_iloc(col)
                    array = container._blocks._extract_array_column(iloc_column_key) # type: ignore
                else: # need to reindex
                    col_series = container[col]
                    fill_value = get_col_fill_value(col_count, col_series.dtype)
                    array = col_series.reindex(index, fill_value=fill_value).values
                    array.flags.writeable = False
                values.append(array)

            # apply values only where missing
            post_blocks = post_blocks.fill_missing_by_values(values, func=func)
            if not post_blocks.boolean_apply_any(func):
                break

        return cls(post_blocks,
                    index=index,
                    columns=columns,
                    name=name,
                    own_data=True,
                    own_index=True,
                    own_columns=True,
                    )


    @classmethod
    @doc_inject(selector='constructor_frame')
    def from_records(cls,
            records: tp.Iterable[tp.Any],
            *,
            index: tp.Optional[TIndexInitializer] = None,
            columns: tp.Optional[TIndexInitializer] = None,
            dtypes: TDtypesSpecifier = None,
            name: TLabel = None,
            consolidate_blocks: bool = False,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            own_index: bool = False,
            own_columns: bool = False
            ) -> tp.Self:
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

        rows: tp.Sequence[tp.Any]
        if not hasattr(records, '__len__'):
            # might be a generator; must convert to sequence
            rows = list(records)
        else: # could be a sequence, or something like a dict view
            rows = records # type: ignore
        row_count = len(rows)

        if not row_count:
            if columns is not None: # we can create a zero-record Frame
                return cls._from_zero_size_shape(
                        columns=columns,
                        columns_constructor=columns_constructor,
                        own_columns=own_columns,
                        name=name,
                        dtypes=dtypes,
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

        is_dc_inst = hasattr(row_reference, '__dataclass_fields__')
        if is_dc_inst:
            fields_dc = tuple(row_reference.__dataclass_fields__.keys())


        column_name_getter = None
        # NOTE: even if getter is defined, columns list is needed to be available to get_col_dtype after it is populated
        if columns is None and hasattr(row_reference, '_fields'): # NamedTuple
            column_name_getter = row_reference._fields.__getitem__
            columns = []
        elif columns is None and is_dc_inst:
            column_name_getter = fields_dc.__getitem__
            columns = []

        get_col_dtype = None if dtypes is None else get_col_dtype_factory(dtypes, columns) # type: ignore

        # NOTE: row data by definition does not have Index data, so col count is length of row
        if hasattr(row_reference, '__len__'):
            col_count = len(row_reference)
        elif is_dc_inst:
            col_count = len(fields_dc) # defined in branch above
        else:
            raise NotImplementedError(f'cannot get col_count from {row_reference}')

        if not is_dc_inst:
            def get_value_iter(col_key: TLabel, col_idx: int) -> tp.Iterator[tp.Any]:
                rows_iter = rows if not rows_to_iter else iter(rows)
                return (row[col_key] for row in rows_iter)
        else:
            def get_value_iter(col_key: TLabel, col_idx: int) -> tp.Iterator[tp.Any]:
                rows_iter = rows if not rows_to_iter else iter(rows)
                return (getattr(row, fields_dc[col_key]) for row in rows_iter) #type: ignore

        def blocks() -> tp.Iterator[TNDArrayAny]:
            # iterate over final column order, yielding 1D arrays
            for col_idx in range(col_count):
                if column_name_getter: # append as side effect of generator!
                    columns.append(column_name_getter(col_idx)) # type: ignore
                values = array_from_value_iter(
                        key=col_idx,
                        idx=col_idx, # integer used
                        get_value_iter=get_value_iter,
                        get_col_dtype=get_col_dtype,
                        row_count=row_count
                        )
                yield values

        block_gen: tp.Callable[..., tp.Iterator[TNDArrayAny]]
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
            records: tp.Iterable[tp.Mapping[tp.Any, tp.Any]],
            *,
            index: tp.Optional[TIndexInitializer] = None,
            dtypes: TDtypesSpecifier = None,
            name: TLabel = None,
            fill_value: tp.Any = np.nan,
            consolidate_blocks: bool = False,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            own_index: bool = False,
            ) -> tp.Self:
        '''Frame constructor from an iterable of dictionaries, where each dictionary represents a row; column names will be derived from the union of all row dictionary keys.

        Args:
            records: Iterable of row values, where row values are dictionaries.
            index: Optionally provide an iterable of index labels, equal in length to the number of records. If a generator, this value will not be evaluated until after records are loaded.
            index:
            {dtypes}
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`Frame`
        '''
        columns: tp.List[TLabel] = []
        get_col_dtype = None if dtypes is None else get_col_dtype_factory(dtypes, columns)
        get_col_fill_value = (None if not is_fill_value_factory_initializer(fill_value)
                else get_col_fill_value_factory(fill_value, columns))

        rows: tp.Sequence[tp.Mapping[TLabel, tp.Any]]
        if not hasattr(records, '__len__'):
            # might be a generator; must convert to sequence
            rows = list(records)
        else: # could be a sequence, or something like a dict view
            rows = records # type: ignore
        row_count = len(rows)

        if not row_count:
            raise ErrorInitFrame('no rows available in records.')

        if hasattr(rows, '__getitem__'):
            rows_to_iter = False
        else: # dict view, or other sized iterable that does not support getitem
            rows_to_iter = True

        # derive union columns
        row_reference: tp.Dict[TLabel, tp.Any] = {}
        for row in rows: # produce a row that has a value for all observed keys
            row_reference.update(row)

        # get value for a column accross all rows
        def get_value_iter(col_key: TLabel, col_idx: int) -> tp.Iterator[tp.Any]:
            rows_iter = rows if not rows_to_iter else iter(rows)

            if get_col_fill_value is not None and get_col_dtype is not None:
                return (row.get(col_key, get_col_fill_value(
                                col_idx,
                                np.dtype(get_col_dtype(col_idx)))) # might be dtype specifier
                        for row in rows_iter)

            if get_col_fill_value is not None:
                return (row.get(col_key, get_col_fill_value(col_idx, None))
                        for row in rows_iter)

            return (row.get(col_key, fill_value) for row in rows_iter)

        def blocks() -> tp.Iterator[TNDArrayAny]:
            # iterate over final column order, yielding 1D arrays
            for col_idx, col_key in enumerate(row_reference.keys()):
                columns.append(col_key)
                yield array_from_value_iter(
                        key=col_key,
                        idx=col_idx,
                        get_value_iter=get_value_iter,
                        get_col_dtype=get_col_dtype,
                        row_count=row_count
                        )

        block_gen: tp.Callable[..., tp.Iterator[TNDArrayAny]]
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
            items: tp.Iterable[tp.Tuple[TLabel, tp.Iterable[tp.Any]]],
            *,
            columns: tp.Optional[TIndexInitializer] = None,
            dtypes: TDtypesSpecifier = None,
            name: TLabel = None,
            consolidate_blocks: bool = False,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            own_columns: bool = False,
            ) -> tp.Self:
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

        def gen() -> tp.Iterator[tp.Iterable[tp.Any]]:
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
            items: tp.Iterable[tp.Tuple[TLabel, tp.Mapping[tp.Any, tp.Any]]],
            *,
            dtypes: TDtypesSpecifier = None,
            name: TLabel = None,
            consolidate_blocks: bool = False) -> tp.Self:
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

        def gen() -> tp.Iterator[tp.Mapping[tp.Any, tp.Any]]:
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
            pairs: tp.Iterable[tp.Tuple[TLabel, tp.Iterable[tp.Any]]],
            *,
            index: tp.Optional[TIndexInitializer] = None,
            fill_value: tp.Any = np.nan,
            dtypes: TDtypesSpecifier = None,
            name: TLabel = None,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            consolidate_blocks: bool = False
            ) -> tp.Self:
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
        columns: tp.List[TLabel] = []

        # if an index initializer is passed, and we expect to get Series, we need to create the index in advance of iterating blocks
        # NOTE: could add own_index argument in signature, see implementation in from_fields()
        own_index = False
        if _index_initializer_needs_init(index):
            index = index_from_optional_constructor(index,
                    default_constructor=Index,
                    explicit_constructor=index_constructor
                    )
            own_index = True

        get_col_dtype = None if dtypes is None else get_col_dtype_factory(dtypes, columns)
        get_col_fill_value = get_col_fill_value_factory(fill_value, columns=columns)

        def blocks() -> tp.Iterator[TNDArrayAny]:
            for col_idx, (k, v) in enumerate(pairs):
                columns.append(k) # side effect of generator!
                column_type = None if get_col_dtype is None else get_col_dtype(col_idx) #pylint: disable=E1102

                if v.__class__ is np.ndarray:
                    # NOTE: we rely on TypeBlocks constructor to check that these are same sized
                    if column_type is not None:
                        yield v.astype(column_type) # type: ignore
                    else:
                        yield v # pyright: ignore
                elif isinstance(v, Series):
                    if index is None:
                        raise ErrorInitFrame('can only consume Series in Frame.from_items if an Index is provided.')

                    if not v.index.equals(index):
                        # NOTE: we assume we should use column_type if it is specified
                        dtype_for_fv = (np.dtype(column_type) if column_type is not None
                                else v.dtype)
                        v = v.reindex(index,
                                fill_value=get_col_fill_value(col_idx, dtype_for_fv),
                                check_equals=False,
                                )
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

        block_gen: tp.Callable[[], tp.Iterator[TNDArrayAny]]
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

    # NOTE: mapping keys must be tp.Any; anything else requires uses TLabel

    @classmethod
    @doc_inject(selector='constructor_frame')
    def from_dict(cls,
            mapping: tp.Mapping[tp.Any, tp.Iterable[tp.Any]],
            *,
            index: tp.Optional[TIndexInitializer] = None,
            fill_value: object = np.nan,
            dtypes: TDtypesSpecifier = None,
            name: TLabel = None,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            consolidate_blocks: bool = False
            ) -> tp.Self:
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
            index: tp.Optional[TIndexInitializer] = None,
            columns: tp.Optional[TIndexInitializer] = None,
            fill_value: tp.Any = np.nan,
            dtypes: TDtypesSpecifier = None,
            name: TLabel = None,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            own_index: bool = False,
            own_columns: bool = False,
            consolidate_blocks: bool = False
            ) -> tp.Self:
        '''Frame constructor from an iterator of columns, where columns are iterables. :obj:`Series` can be provided as values if an ``index`` argument is supplied. This constructor is similar to ``from_items()``, though here columns are provided through an independent ``columns`` argument.

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

        get_col_dtype = None if dtypes is None else get_col_dtype_factory(dtypes, columns) #type: ignore
        get_col_fill_value = get_col_fill_value_factory(fill_value, columns=columns) # type: ignore

        def blocks() -> tp.Iterator[TNDArrayAny]:
            for col_idx, v in enumerate(fields):
                column_type = None if get_col_dtype is None else get_col_dtype(col_idx) #pylint: disable=E1102

                if v.__class__ is np.ndarray:
                    if column_type is not None:
                        yield v.astype(column_type) # type: ignore
                    else:
                        yield v # pyright: ignore
                elif isinstance(v, Series):
                    if index is None:
                        raise ErrorInitFrame('can only consume Series in Frame.from_fields if an Index is provided.')

                    if not v.index.equals(index):
                        dtype_for_fv = (np.dtype(column_type) if column_type is not None
                                else v.dtype)
                        v = v.reindex(index,
                                fill_value=get_col_fill_value(col_idx, dtype_for_fv),
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

        block_gen: tp.Callable[..., tp.Iterator[TNDArrayAny]]
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
                columns_constructor=columns_constructor,
                index_constructor=None if own_index else index_constructor,
                )


    @classmethod
    @doc_inject(selector='constructor_frame')
    def from_dict_fields(cls,
            fields: tp.Iterable[tp.Mapping[tp.Any, tp.Any]],
            *,
            columns: tp.Optional[TIndexInitializer] = None,
            dtypes: TDtypesSpecifier = None,
            name: TLabel = None,
            fill_value: tp.Any = np.nan,
            consolidate_blocks: bool = False,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            own_index: bool = False,
            ) -> tp.Self:
        '''Frame constructor from an iterable of dictionaries, where each dictionary represents a column; index labels will be derived from the union of all column dictionary keys.

        Args:
            fields: Iterable of column values, where column values are dictionaries.
            index: Optionally provide an iterable of index labels, equal in length to the number of fields. If a generator, this value will not be evaluated until after fields are loaded.
            columns:
            {dtypes}
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`Frame`
        '''
        get_col_dtype = None if dtypes is None else get_col_dtype_factory(dtypes, columns) # type: ignore
        get_col_fill_value = (None if not is_fill_value_factory_initializer(fill_value)
                else get_col_fill_value_factory(fill_value, columns)) # type: ignore

        cols: tp.Sequence[tp.Mapping[tp.Any, tp.Any]]
        if not hasattr(fields, '__len__'):
            # might be a generator; must convert to sequence
            cols = list(fields)
        else: # could be a sequence, or something like a dict view
            cols = fields # type: ignore
        cols_count = len(cols)

        if not cols_count:
            raise ErrorInitFrame('No columns available in `fields`.')

        # derive union index
        col_reference: tp.Dict[TLabel, tp.Any] = {}
        for col in cols: # produce a column that has a value for all observed keys
            col_reference.update(col)

        def blocks() -> tp.Iterator[TNDArrayAny]:
            cols_iter = cols if hasattr(cols, '__getitem__') else iter(cols)
            for col_idx, col_dict in enumerate(cols_iter):

                dtype = None
                if get_col_fill_value is not None and get_col_dtype is not None:
                    dts = get_col_dtype(col_idx)
                    dtype = None if dts is None else np.dtype(dts)
                    fv = get_col_fill_value(col_idx, dtype) # might be dtype specifier
                if get_col_fill_value is not None:
                    fv = get_col_fill_value(col_idx, None)
                else:
                    fv = fill_value

                values = []
                for key in col_reference:
                    values.append(col_dict.get(key, fv))

                if dtype is None:
                    array, _ = iterable_to_array_1d(values, count=len(values))
                else:
                    array = np.array(values, dtype=dtype)

                array.flags.writeable = False
                yield array

        block_gen: tp.Callable[..., tp.Iterator[TNDArrayAny]]
        if consolidate_blocks:
            block_gen = lambda: TypeBlocks.consolidate_blocks(blocks())
        else:
            block_gen = blocks

        return cls(TypeBlocks.from_blocks(block_gen()),
                index=col_reference.keys(),
                columns=columns,
                name=name,
                own_data=True,
                index_constructor=index_constructor,
                columns_constructor=columns_constructor,
                own_index=own_index,
                )

    @staticmethod
    def _structured_array_to_d_ia_cl(
            array: TNDArrayAny,
            *,
            index_depth: int = 0,
            index_column_first: tp.Optional[TIndexSpecifier] = None,
            dtypes: TDtypesSpecifier = None,
            consolidate_blocks: bool = False,
            store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT,
            ) -> tp.Tuple[TypeBlocks, tp.Sequence[TNDArrayAny], tp.Sequence[TLabel]]:
        '''
        Expanded function name: _structure_array_to_data_index_arrays_columns_labels

        Utility function for creating TypeBlocks from structure array (or a 2D array that np.genfromtxt might have returned) while extracting index and columns labels. Does not form Index objects for columns or index, allowing down-stream processes to do so.

        Args:
            index_column_first: optionally name the column that will start the block of index columns.
        '''
        names = array.dtype.names # using names instead of fields, as this is NP convention
        is_structured_array = True
        if names is None:
            is_structured_array = False
            # raise ErrorInitFrame('array is not a structured array')
            # could use np.rec.fromarrays, but that makes a copy; better to use the passed in array
            # must be a 2D array
            names = tuple(range(array.shape[1]))

        index_start_pos: int | np.integer[tp.Any] = -1 # will be ignored
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
        columns_by_col_idx: tp.List[TLabel] = []

        get_col_dtype = None if dtypes is None else get_col_dtype_factory(
                dtypes,
                columns_by_col_idx)

        def blocks() -> tp.Iterator[TNDArrayAny]:
            # iterate over column names and yield one at a time for block construction; collect index arrays and column labels as we go
            for col_idx, name in enumerate(names):
                # append here as we iterate for usage in get_col_dtype
                columns_by_col_idx.append(name)

                if is_structured_array:
                    # expect a 1D array with selection, not a copy
                    array_final = array[name]
                    if array_final.ndim == 0:
                        # NOTE: observed with some version of NumPy some structured arrays give 0 ndim arrays when selected by name, but cannot reproduce with newer NumPy
                        array_final = np.reshape(array_final, (1,)) #pragma: no cover
                else: # alyways a 2D array, name is integer for column, slice a 1D array
                    array_final = array[NULL_SLICE, name]

                # do StoreFilter conversions before dtype
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
            index_arrays: tp.Sequence[TNDArrayAny],
            index_constructors: TIndexCtorSpecifiers,
            columns_depth: int,
            columns_labels: tp.Sequence[TLabel],
            columns_constructors: TIndexCtorSpecifiers,
            name: TLabel,
            ) -> tp.Self:
        '''
        Private constructor used for specialized construction from NP Structured array, as well as StoreHDF5.
        '''
        columns_default_constructor: TIndexCtorSpecifier
        if columns_depth <= 1:
            columns_default_constructor = cls._COLUMNS_CONSTRUCTOR
        else:
            columns_default_constructor = partial(
                    cls._COLUMNS_HIERARCHY_CONSTRUCTOR.from_labels_delimited,
                    delimiter=' ')

        columns, own_columns = index_from_optional_constructors(
                columns_labels,
                depth=columns_depth,
                default_constructor=columns_default_constructor,
                explicit_constructors=columns_constructors, # cannot supply name
                )

        index_values: tp.Iterable[tp.Any]
        if index_depth == 1:
            index_values = index_arrays[0]
            index_default_constructor = Index
        else: # > 1
            # might use _from_type_blocks, but would not be able to use continuation token
            index_values = zip(*index_arrays)
            index_default_constructor = IndexHierarchy.from_labels # type: ignore

        index, own_index = index_from_optional_constructors(
                index_values,
                depth=index_depth,
                default_constructor=index_default_constructor,
                explicit_constructors=index_constructors, # cannot supply name
                )

        return cls(data=data,
                own_data=True,
                columns=columns,
                own_columns=own_columns,
                index=index,
                own_index=own_index,
                name=name,
                )


    @classmethod
    @doc_inject(selector='constructor_frame')
    def from_structured_array(cls,
            array: TNDArrayAny,
            *,
            index_depth: int = 0,
            index_column_first: tp.Optional[TIndexSpecifier] = None,
            index_constructors: TIndexCtorSpecifiers = None,
            columns_depth: int = 1,
            columns_constructors: TIndexCtorSpecifiers = None,
            dtypes: TDtypesSpecifier = None,
            name: TLabel = None,
            consolidate_blocks: bool = False,
            store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ) -> tp.Self:
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
                index_constructors=index_constructors,
                columns_depth=columns_depth,
                columns_labels=columns_labels,
                columns_constructors=columns_constructors,
                name=name
                )

    #---------------------------------------------------------------------------
    @classmethod
    def from_element_items(cls,
            items: tp.Iterable[tp.Tuple[
                    tp.Tuple[TLabel, TLabel], tp.Any]],
            *,
            index: TIndexInitializer,
            columns: TIndexInitializer,
            dtype: TDtypesSpecifier = None,
            axis: tp.Optional[int] = None,
            name: TName = None,
            fill_value: tp.Any = FILL_VALUE_DEFAULT,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            own_index: bool = False,
            own_columns: bool = False,
            ) -> tp.Self:
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

        if axis is None:
            if not is_dtype_specifier(dtype):
                raise ErrorInitFrame('cannot provide multiple dtypes when creating a Frame from element items and axis is None')
            if is_fill_value_factory_initializer(fill_value):
                raise InvalidFillValue(fill_value, 'axis==None')

            items_iloc: tp.Iterator[tp.Tuple[tp.Tuple[int, int], tp.Any]] = (
                    ((index._loc_to_iloc(k[0]), columns._loc_to_iloc(k[1])), v) # type: ignore
                    for k, v in items)

            dt: TDtypeSpecifier = dtype if dtype is not None else DTYPE_OBJECT # type: ignore
            tb = TypeBlocks.from_element_items(
                    items_iloc,
                    shape=(len(index), len(columns)), #type: ignore
                    dtype=dt,
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

        elif axis == 1: # column wise, use from_fields
            def fields() -> tp.Iterator[tp.List[tp.Any]]:
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
            index_constructors: TIndexCtorSpecifiers = None,
            columns_depth: int = 1,
            columns_select: tp.Iterable[str | tp.Tuple[str, ...]] | None = None,
            columns_constructors: TIndexCtorSpecifiers = None,
            dtypes: TDtypesSpecifier = None,
            name: TLabel = None,
            consolidate_blocks: bool = False,
            parameters: tp.Any = (),
            ) -> tp.Self:
        '''
        Frame constructor from an SQL query and a database connection object.

        Args:
            query: A query string.
            connection: A DBAPI2 (PEP 249) Connection object, such as those returned from SQLite (via the sqlite3 module) or PyODBC.
            {dtypes}
            index_depth:
            index_constructors:
            columns_depth:
            columns_select: An optional iterable of field names to extract from the results of the query.
            columns_constructors:
            {name}
            {consolidate_blocks}
            parameters: Provide a list of values for an SQL query expecting parameter substitution.
        '''
        columns: tp.Optional[IndexBase] = None
        own_columns = False

        # We cannot assume the cursor object returned by DBAPI Connection to have a context manager, thus all cursor usage needs to be wrapped in a try/finally to insure that the cursor is closed.
        cursor: sqlite3.Cursor | None = None
        try:
            cursor = connection.cursor()
            cursor.execute(query, parameters)

            if columns_select:
                columns_select = set(columns_select)
                # selector function defined below
                def filter_row(row: tp.Sequence[tp.Any]) -> tp.Sequence[tp.Any]:
                    post = selector(row)
                    return post if not selector_reduces else (post,) # type: ignore

            if columns_depth > 0 or columns_select:
                # always need to derive labels if using columns_select
                labels = (col for (col, *_) in cursor.description[index_depth:])

            if columns_depth <= 1 and columns_select:
                iloc_sel, labels = zip(*(
                        pair for pair in enumerate(labels) if pair[1] in columns_select
                        ))
                selector = itemgetter(*iloc_sel)
                selector_reduces = len(iloc_sel) == 1

            if columns_depth == 1:
                columns, own_columns = index_from_optional_constructors(
                        labels,
                        depth=columns_depth,
                        default_constructor=cls._COLUMNS_CONSTRUCTOR,
                        explicit_constructors=columns_constructors, # cannot supply name
                        )
            elif columns_depth > 1:
                # NOTE: we only support loading in IH if encoded in each header with a space delimiter
                columns_constructor: TIndexHierarchyCtor = partial(
                        cls._COLUMNS_HIERARCHY_CONSTRUCTOR.from_labels_delimited,
                        delimiter=' ',
                        )
                columns, own_columns = index_from_optional_constructors(
                        labels,
                        depth=columns_depth,
                        default_constructor=columns_constructor,
                        explicit_constructors=columns_constructors,
                        )

                if columns_select:
                    iloc_sel = columns._loc_to_iloc(columns.isin(columns_select)) # type: ignore
                    selector = itemgetter(*iloc_sel)
                    selector_reduces = len(iloc_sel) == 1 # pyright: ignore
                    columns = columns.iloc[iloc_sel] # type: ignore

            # NOTE: cannot own_index as we defer calling the constructor until after call Frame
            # map dtypes in context of pre-index extraction
            if index_depth > 0:
                get_col_dtype = None if dtypes is None else get_col_dtype_factory(
                        dtypes,
                        [col for (col, *_) in cursor.description],
                        )

            index_constructor: TIndexCtorSpecifier
            row_gen: tp.Callable[..., tp.Iterator[tp.Sequence[tp.Any]]] # pyright: ignore

            if index_depth == 0:
                index = None
                row_gen = lambda: cursor
                index_constructor = None
            elif index_depth == 1:
                index = [] # lazily populate
                default_constructor: tp.Type[Index] = partial(Index, dtype=get_col_dtype(0)) if get_col_dtype else Index # type: ignore
                # parital to include everything but values
                index_constructor = constructor_from_optional_constructors(
                        depth=index_depth,
                        default_constructor=default_constructor,
                        explicit_constructors=index_constructors,
                        )
                def row_gen() -> tp.Iterator[tp.Sequence[tp.Any]]:
                    for row in cursor:
                        index.append(row[0])
                        yield row[1:]
            else: # > 1
                index = [list() for _ in range(index_depth)]

                def default_constructor(
                        iterables: tp.Iterable[tp.Iterable[TLabel]],
                        index_constructors: TIndexCtorSpecifiers,
                        ) -> IndexHierarchy: #pylint: disable=function-redefined
                    if get_col_dtype:
                        blocks = [iterable_to_array_1d(it, get_col_dtype(i))[0]
                                for i, it in enumerate(iterables)]
                    else:
                        blocks = [iterable_to_array_1d(it)[0] for it in iterables]
                    return IndexHierarchy._from_type_blocks(
                            TypeBlocks.from_blocks(blocks),
                            index_constructors=index_constructors,
                            own_blocks=True,
                            )
                # parital to include everything but values
                index_constructor = constructor_from_optional_constructors(
                        depth=index_depth,
                        default_constructor=default_constructor,
                        explicit_constructors=index_constructors,
                        )

                def row_gen() -> tp.Iterator[tp.Sequence[tp.Any]]:
                    for row in cursor:
                        for i, label in enumerate(row[:index_depth]):
                            index[i].append(label)
                        yield row[index_depth:]

            if columns_select:
                row_gen_final = (filter_row(row) for row in row_gen())
            else:
                row_gen_final = row_gen() # type: ignore

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

    #---------------------------------------------------------------------------
    @classmethod
    @doc_inject(selector='json')
    def from_json_index(cls,
            json_data: tp.Union[str, StringIO],
            *,
            dtypes: TDtypesSpecifier = None,
            name: TLabel = None,
            consolidate_blocks: bool = False,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            ) -> tp.Self:
        '''Frame constructor from an in-memory JSON document in the following format: {json_index}

        Args:
            json_data: a string or StringIO of JSON data
            {dtypes}
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`Frame`
        '''
        if isinstance(json_data, STRING_TYPES):
            data = json.loads(json_data)
        else: # StringIO or open file
            data = json.load(json_data)

        index = []

        def gen() -> tp.Iterator[tp.Iterable[tp.Any]]:
            for k, v in data.items():
                index.append(k)
                yield v

        return cls.from_dict_records(gen(), # type: ignore
                index=index,
                dtypes=dtypes,
                name=name,
                consolidate_blocks=consolidate_blocks,
                index_constructor=index_constructor,
                columns_constructor=columns_constructor,
                )

    @classmethod
    @doc_inject(selector='json')
    def from_json_columns(cls,
            json_data: tp.Union[str, StringIO],
            *,
            dtypes: TDtypesSpecifier = None,
            name: TLabel = None,
            consolidate_blocks: bool = False,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            ) -> tp.Self:
        '''Frame constructor from an in-memory JSON document in the following format: {json_columns}

        Args:
            json_data: a string or StringIO of JSON data
            {dtypes}
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`Frame`
        '''
        if isinstance(json_data, STRING_TYPES):
            data = json.loads(json_data)
        else: # StringIO or open file
            data = json.load(json_data)

        columns = []

        def gen() -> tp.Iterator[tp.Iterable[tp.Any]]:
            for k, v in data.items():
                columns.append(k)
                yield v

        return cls.from_dict_fields(gen(), # type: ignore
                columns=columns,
                dtypes=dtypes,
                name=name,
                consolidate_blocks=consolidate_blocks,
                index_constructor=index_constructor,
                columns_constructor=columns_constructor,
                )

    @classmethod
    @doc_inject(selector='json')
    def from_json_split(cls,
            json_data: tp.Union[str, StringIO],
            *,
            dtypes: TDtypesSpecifier = None,
            name: TLabel = None,
            consolidate_blocks: bool = False,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            ) -> tp.Self:
        '''Frame constructor from an in-memory JSON document in the following format: {json_split}

        Args:
            json_data: a string or StringIO of JSON data
            {dtypes}
            {name}
            {consolidate_blocks}
        Returns:
            :obj:`Frame`
        '''
        if isinstance(json_data, STRING_TYPES):
            data = json.loads(json_data)
        else: # StringIO or open file
            data = json.load(json_data)

        return cls.from_records(data['data'],
                index=data['index'],
                columns=data['columns'],
                dtypes=dtypes,
                name=name,
                consolidate_blocks=consolidate_blocks,
                index_constructor=index_constructor,
                columns_constructor=columns_constructor,
                )

    @classmethod
    @doc_inject(selector='json')
    def from_json_records(cls,
            json_data: tp.Union[str, StringIO],
            *,
            index: tp.Optional[TIndexInitializer] = None,
            dtypes: TDtypesSpecifier = None,
            name: TLabel = None,
            consolidate_blocks: bool = False,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            ) -> tp.Self:
        '''Frame constructor from an in-memory JSON document in the following format: {json_records}

        Args:
            json_data: a string or StringIO of JSON data
            {dtypes}
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`Frame`
        '''
        if isinstance(json_data, STRING_TYPES):
            data = json.loads(json_data)
        else: # StringIO or open file
            data = json.load(json_data)

        return cls.from_dict_records(data,
                index=index,
                dtypes=dtypes,
                name=name,
                consolidate_blocks=consolidate_blocks,
                index_constructor=index_constructor,
                columns_constructor=columns_constructor,
                )

    @classmethod
    @doc_inject(selector='json')
    def from_json_values(cls,
            json_data: tp.Union[str, StringIO],
            *,
            index: tp.Optional[TIndexInitializer] = None,
            columns: tp.Optional[TIndexInitializer] = None,
            dtypes: TDtypesSpecifier = None,
            name: TLabel = None,
            consolidate_blocks: bool = False,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            ) -> tp.Self:
        '''Frame constructor from an in-memory JSON document in the following format: {json_values}

        Args:
            json_data: a string or StringIO of JSON data
            {dtypes}
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`Frame`
        '''
        if isinstance(json_data, STRING_TYPES):
            data = json.loads(json_data)
        else: # StringIO or open file
            data = json.load(json_data)

        return cls.from_records(data,
                index=index,
                columns=columns,
                dtypes=dtypes,
                name=name,
                consolidate_blocks=consolidate_blocks,
                index_constructor=index_constructor,
                columns_constructor=columns_constructor,
                )


    @classmethod
    @doc_inject(selector='json')
    def from_json_typed(cls,
            json_data: tp.Union[str, StringIO],
            *,
            consolidate_blocks: bool = False,
            ) -> tp.Self:
        '''Frame constructor from an in-memory JSON document in the following format: {json_typed}

        Args:
            json_data: a string or StringIO of JSON data
        Returns:
            :obj:`Frame`
        '''
        if isinstance(json_data, STRING_TYPES):
            data = json.loads(json_data)
        else: # StringIO or open file
            data = json.load(json_data)

        md = data['__meta__']
        name = md[JSONMeta.KEY_NAMES][0] # first is for Frame
        dtypes = md[JSONMeta.KEY_DTYPES]
        index_constructor, columns_constructor = JSONMeta.from_dict_to_ctors(
                md,
                cls.STATIC,
                )
        return cls.from_fields(data['data'],
                index=data['index'],
                columns=data['columns'],
                dtypes=dtypes,
                name=name,
                consolidate_blocks=consolidate_blocks,
                index_constructor=index_constructor,
                columns_constructor=columns_constructor,
                )


    #---------------------------------------------------------------------------
    @classmethod
    @doc_inject(selector='constructor_frame')
    def from_delimited(cls,
            fp: TPathSpecifierOrTextIOOrIterator,
            *,
            delimiter: str,
            index_depth: int = 0,
            index_column_first: int = 0,
            index_name_depth_level: tp.Optional[TDepthLevel] = None,
            index_constructors: TIndexCtorSpecifiers = None,
            index_continuation_token: tp.Optional[TLabel] = CONTINUATION_TOKEN_INACTIVE,
            columns_depth: int = 1,
            columns_name_depth_level: tp.Optional[TDepthLevel] = None,
            columns_constructors: TIndexCtorSpecifiers = None,
            columns_continuation_token: tp.Optional[TLabel] = CONTINUATION_TOKEN_INACTIVE,
            columns_select: tp.Optional[tp.Iterable[TLabel]] = None,
            skip_header: int = 0,
            skip_footer: int = 0,
            skip_initial_space: bool = False,
            quoting: int = csv.QUOTE_MINIMAL,
            quote_char: str = '"',
            quote_double: bool = True,
            escape_char: tp.Optional[str] = None,
            thousands_char: str = '',
            decimal_char: str = '.',
            encoding: tp.Optional[str] = None,
            dtypes: TDtypesSpecifier = None,
            name: TLabel = None,
            consolidate_blocks: bool = False,
            store_filter: tp.Optional[StoreFilter] = None,
            ) -> tp.Self:
        '''
        Create a :obj:`Frame` from a file path or a file-like object defining a delimited (CSV, TSV) data file.

        Args:
            fp: A file path or a file-like object.
            delimiter: The character used to seperate row elements.
            index_depth: Specify the number of columns used to create the index labels; a value greater than 1 will attempt to create a hierarchical index.
            index_column_first: Optionally specify a column, by position in the realized columns, to become the start of the index if index_depth is greater than 0 and columns_depth is 0.
            index_name_depth_level: If columns_depth is greater than 0, interpret values over index as the index name.
            index_constructors:
            index_continuation_token:
            columns_depth: Specify the number of rows after the skip_header used to create the column labels. A value of 0 will be no header; a value greater than 1 will attempt to create a hierarchical index.
            columns_name_depth_level: If index_depth is greater than 0, interpret values over index as the columns name.
            columns_constructors:
            columns_continuation_token:
            columns_select: an iterable of columns to select by label or position; can only be used if index_depth is 0.
            skip_header: Number of leading lines to skip.
            skip_footer: Number of trailing lines to skip.
            store_filter: A StoreFilter instance, defining translation between unrepresentable strings and types. By default it is disabled, and only empty fields or "NAN" are intepreted as NaN. To force usage, set the type of the column to string.
            {dtypes}
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`static_frame.Frame`
        '''
        if skip_header < 0:
            raise ErrorInitFrame('skip_header must be greater than or equal to 0')

        fpf = path_filter(fp) # normalize Path to strings

        if not skip_footer:
            def file_like() -> tp.Iterator[str]:
                if isinstance(fpf, str):
                    with open(fpf, 'r', encoding=encoding) as f:
                        yield from f
                else: # iterable of string lines, StringIO
                    yield from fpf
        else:
            def file_like() -> tp.Iterator[str]:
                row_buffer: tp.Deque[str] = deque(maxlen=skip_footer)

                if isinstance(fpf, str):
                    with open(fpf, 'r', encoding=encoding) as f:
                        for i, row in enumerate(f):
                            if i >= skip_footer:
                                yield row_buffer.popleft()
                            row_buffer.append(row)
                else:
                    for i, row in enumerate(fpf):
                        if i >= skip_footer:
                            yield row_buffer.popleft()
                        row_buffer.append(row)

        row_iter = file_like()
        if skip_header:
            for _ in range(skip_header):
                next(row_iter)

        apex_rows = []
        if columns_depth:
            columns_arrays = []
            for _ in range(columns_depth):
                row = next(row_iter)
                if index_depth == 0:
                    row_left = ''
                    row_right = row
                else:
                    row_left, row_right = split_after_count(
                            row,
                            delimiter=delimiter,
                            count=index_depth,
                            quoting=quoting,
                            quotechar=quote_char,
                            doublequote=quote_double,
                            escapechar=escape_char,
                            )

                [array_right] = delimited_to_arrays(
                        (row_right,),
                        axis=0, # process type per row
                        delimiter=delimiter,
                        quoting=quoting,
                        quotechar=quote_char,
                        doublequote=quote_double,
                        escapechar=escape_char,
                        thousandschar=thousands_char,
                        decimalchar=decimal_char,
                        skipinitialspace=skip_initial_space,
                        )
                columns_arrays.append(array_right)

                if row_left:
                    [array_left] = delimited_to_arrays(
                            (row_left,),
                            axis=0, # process type per row
                            delimiter=delimiter,
                            quoting=quoting,
                            quotechar=quote_char,
                            doublequote=quote_double,
                            escapechar=escape_char,
                            thousandschar=thousands_char,
                            decimalchar=decimal_char,
                            skipinitialspace=skip_initial_space,
                            )
                    apex_rows.append(array_left)

        if columns_depth == 0:
            columns = None
            own_columns = False
        else:
            columns_name = None if index_depth == 0 else apex_to_name(
                    rows=apex_rows,
                    depth_level=columns_name_depth_level,
                    axis=1,
                    axis_depth=columns_depth)

            columns_constructor: TIndexHierarchyCtor

            if columns_depth == 1:
                columns, own_columns = index_from_optional_constructors(
                        columns_arrays[0],
                        depth=columns_depth,
                        default_constructor=partial(cls._COLUMNS_CONSTRUCTOR, name=columns_name),
                        explicit_constructors=columns_constructors, # cannot supply name
                        )
            elif columns_continuation_token is not CONTINUATION_TOKEN_INACTIVE:
                if store_filter is not None:
                    labels = zip_longest(
                            *(store_filter.to_type_filter_array(x) for x in columns_arrays), # pyright: ignore
                            fillvalue=columns_continuation_token,
                            )
                else:
                    labels = zip_longest(
                            *columns_arrays,
                            fillvalue=columns_continuation_token,
                            )
                columns_constructor = partial(
                        cls._COLUMNS_HIERARCHY_CONSTRUCTOR.from_labels,
                        name=columns_name,
                        continuation_token=columns_continuation_token,
                        )
                columns, own_columns = index_from_optional_constructors(
                        labels,
                        depth=columns_depth,
                        default_constructor=columns_constructor,
                        explicit_constructors=columns_constructors,
                        )
            else:
                if store_filter is not None:
                    columns_arrays = [store_filter.to_type_filter_array(x) for x in columns_arrays] # pyright: ignore
                columns_constructor = partial(
                        cls._COLUMNS_HIERARCHY_CONSTRUCTOR.from_values_per_depth,
                        name=columns_name,
                        )
                columns, own_columns = index_from_optional_constructors(
                        columns_arrays, # pyright: ignore
                        depth=columns_depth,
                        default_constructor=columns_constructor,
                        explicit_constructors=columns_constructors,
                        )

        line_select: tp.Optional[tp.Callable[[int], bool]]
        if columns_select:
            if index_depth:
                raise ErrorInitFrame('Cannot use columns_select if index_depth is greater than zero.')
                # NOTE: this is because the final columns labels might be different than those provided via input due to line_select and index_depth
            if columns is not None:
                columns_included = list(columns.loc_to_iloc(l) for l in columns_select)
                columns = columns.iloc[columns_included]
            else: # assume columns_select are integers
                columns_included = list(columns_select) # type: ignore
            # order of columns_included maters
            line_select = set(columns_included).__contains__
        else:
            line_select = None

        get_col_dtype = (None if dtypes is None
                else get_col_dtype_factory(dtypes, columns, index_depth))
        values_arrays: tp.Sequence[TNDArrayAny] = delimited_to_arrays(
                row_iter,
                axis=1, # process type per column
                line_select=line_select,
                delimiter=delimiter,
                quoting=quoting,
                quotechar=quote_char,
                doublequote=quote_double,
                escapechar=escape_char,
                thousandschar=thousands_char,
                decimalchar=decimal_char,
                skipinitialspace=skip_initial_space,
                dtypes=get_col_dtype,
                )
        if store_filter is not None:
            values_arrays = [store_filter.to_type_filter_array(a)
                    for a in values_arrays]
        if index_depth:
            if index_column_first:
                # NOTE: we cannot use index_columns_first with labels in columns, as columns has to be truncated for index_depth before the index can be created
                if columns is not None:
                    raise ErrorInitFrame('Cannot use index_column_first if columns_depth is greater than 0.')
                elif isinstance(index_column_first, INT_TYPES):
                    index_start = index_column_first
                else:
                    raise ErrorInitFrame('index_column_first must be an integer.')
                index_end = index_start + index_depth
                index_arrays = values_arrays[index_start: index_end]
                values_arrays = chain( #type: ignore
                        values_arrays[:index_start],
                        values_arrays[index_end:],
                        )
            else:
                index_arrays = values_arrays[:index_depth]
                values_arrays = values_arrays[index_depth:]
        else:
            if index_column_first:
                raise ErrorInitFrame('Cannot set index_column_first without setting nonzero index_depth.')

        if values_arrays:
            if consolidate_blocks:
                blocks = TypeBlocks.from_blocks(
                        TypeBlocks.consolidate_blocks(values_arrays))
            else:
                blocks = TypeBlocks.from_blocks(values_arrays)
        else:
            blocks = FRAME_INITIALIZER_DEFAULT # type: ignore

        kwargs = dict(
                data=blocks,
                own_data=True,
                columns=columns,
                own_columns=own_columns,
                name=name
                )

        if index_depth == 0:
            return cls(index=None, **kwargs) # type: ignore

        index_name = None if columns_depth == 0 else apex_to_name(
                rows=apex_rows,
                depth_level=index_name_depth_level,
                axis=0,
                axis_depth=index_depth)

        index_values: tp.Iterable[tp.Any]
        index_constructor: TIndexCtor

        if index_depth == 1:
            if not index_arrays:
                index_values = () # assume an empty Frame
                assert blocks is FRAME_INITIALIZER_DEFAULT
            else:
                index_values = index_arrays[0]
            index_constructor = partial(Index, name=index_name)
            index, own_index = index_from_optional_constructors(
                    index_values,
                    depth=index_depth,
                    default_constructor=index_constructor,
                    explicit_constructors=index_constructors, # cannot supply name
                    )
        elif index_continuation_token is not CONTINUATION_TOKEN_INACTIVE:
            # expect all index_arrays to have the same length
            index_values = zip(*index_arrays)
            index_constructor = partial(IndexHierarchy.from_labels,
                    name=index_name,
                    continuation_token=index_continuation_token,
                    )
            index, own_index = index_from_optional_constructors(
                    index_values,
                    depth=index_depth,
                    default_constructor=index_constructor,
                    explicit_constructors=index_constructors, # cannot supply name
                    )
        else: # index_depth > 1, no continuation toke`n
            index_constructor = partial(
                    IndexHierarchy.from_values_per_depth,
                    name=index_name,
                    )
            index, own_index = index_from_optional_constructors(
                    index_arrays, # type: ignore
                    depth=index_depth,
                    default_constructor=index_constructor,
                    explicit_constructors=index_constructors, # cannot supply name
                    )
        return cls(
                index=index,
                own_index=own_index,
                **kwargs # type: ignore
                )

    @classmethod
    def from_csv(cls,
            fp: TPathSpecifierOrTextIOOrIterator,
            *,
            index_depth: int = 0,
            index_column_first: int = 0,
            index_name_depth_level: tp.Optional[TDepthLevel] = None,
            index_constructors: TIndexCtorSpecifiers = None,
            index_continuation_token: tp.Union[TLabel, None] = CONTINUATION_TOKEN_INACTIVE,
            columns_depth: int = 1,
            columns_name_depth_level: tp.Optional[TDepthLevel] = None,
            columns_constructors: TIndexCtorSpecifiers = None,
            columns_continuation_token: tp.Union[TLabel, None] = CONTINUATION_TOKEN_INACTIVE,
            columns_select: tp.Optional[tp.Iterable[TLabel]] = None,
            skip_header: int = 0,
            skip_footer: int = 0,
            skip_initial_space: bool = False,
            quoting: int = csv.QUOTE_MINIMAL,
            quote_char: str = '"',
            quote_double: bool = True,
            escape_char: tp.Optional[str] = None,
            thousands_char: str = '',
            decimal_char: str = '.',
            encoding: tp.Optional[str] = None,
            dtypes: TDtypesSpecifier = None,
            name: TLabel = None,
            consolidate_blocks: bool = False,
            store_filter: tp.Optional[StoreFilter] = None,
            ) -> tp.Self:
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
                index_constructors=index_constructors,
                index_continuation_token=index_continuation_token,
                columns_depth=columns_depth,
                columns_name_depth_level=columns_name_depth_level,
                columns_constructors=columns_constructors,
                columns_continuation_token=columns_continuation_token,columns_select=columns_select,
                skip_header=skip_header,
                skip_footer=skip_footer,
                skip_initial_space=skip_initial_space,
                quoting=quoting,
                quote_char=quote_char,
                quote_double=quote_double,
                escape_char=escape_char,
                thousands_char=thousands_char,
                decimal_char=decimal_char,
                encoding=encoding,
                dtypes=dtypes,
                name=name,
                consolidate_blocks=consolidate_blocks,
                store_filter=store_filter,
                )

    @classmethod
    def from_tsv(cls,
            fp: TPathSpecifierOrTextIOOrIterator,
            *,
            index_depth: int = 0,
            index_column_first: int = 0,
            index_name_depth_level: tp.Optional[TDepthLevel] = None,
            index_constructors: TIndexCtorSpecifiers = None,
            index_continuation_token: tp.Union[TLabel, None] = CONTINUATION_TOKEN_INACTIVE,
            columns_depth: int = 1,
            columns_name_depth_level: tp.Optional[TDepthLevel] = None,
            columns_constructors: TIndexCtorSpecifiers = None,
            columns_continuation_token: tp.Union[TLabel, None] = CONTINUATION_TOKEN_INACTIVE,
            columns_select: tp.Optional[tp.Iterable[TLabel]] = None,
            skip_header: int = 0,
            skip_footer: int = 0,
            skip_initial_space: bool = False,
            quoting: int = csv.QUOTE_MINIMAL,
            quote_char: str = '"',
            quote_double: bool = True,
            escape_char: tp.Optional[str] = None,
            thousands_char: str = '',
            decimal_char: str = '.',
            encoding: tp.Optional[str] = None,
            dtypes: TDtypesSpecifier = None,
            name: TLabel = None,
            consolidate_blocks: bool = False,
            store_filter: tp.Optional[StoreFilter] = None,
            ) -> tp.Self:
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
                index_constructors=index_constructors,
                index_continuation_token=index_continuation_token,
                columns_depth=columns_depth,
                columns_name_depth_level=columns_name_depth_level,
                columns_constructors=columns_constructors,
                columns_continuation_token=columns_continuation_token,
                columns_select=columns_select,
                skip_header=skip_header,
                skip_footer=skip_footer,
                skip_initial_space=skip_initial_space,
                quoting=quoting,
                quote_char=quote_char,
                quote_double=quote_double,
                escape_char=escape_char,
                thousands_char=thousands_char,
                decimal_char=decimal_char,
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
            index_column_first: int = 0,
            index_name_depth_level: tp.Optional[TDepthLevel] = None,
            index_constructors: TIndexCtorSpecifiers = None,
            index_continuation_token: tp.Union[TLabel, None] = CONTINUATION_TOKEN_INACTIVE,
            columns_depth: int = 1,
            columns_name_depth_level: tp.Optional[TDepthLevel] = None,
            columns_constructors: TIndexCtorSpecifiers = None,
            columns_continuation_token: tp.Union[TLabel, None] = CONTINUATION_TOKEN_INACTIVE,
            columns_select: tp.Optional[tp.Iterable[TLabel]] = None,
            skip_header: int = 0,
            skip_footer: int = 0,
            skip_initial_space: bool = False,
            quoting: int = csv.QUOTE_MINIMAL,
            quote_char: str = '"',
            quote_double: bool = True,
            escape_char: tp.Optional[str] = None,
            thousands_char: str = '',
            decimal_char: str = '.',
            encoding: tp.Optional[str] = None,
            dtypes: TDtypesSpecifier = None,
            name: TName = None,
            consolidate_blocks: bool = False,
            store_filter: tp.Optional[StoreFilter] = None,
            ) -> tp.Self:
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
                index_constructors=index_constructors,
                index_continuation_token=index_continuation_token,
                columns_depth=columns_depth,
                columns_name_depth_level=columns_name_depth_level,
                columns_constructors=columns_constructors,
                columns_continuation_token=columns_continuation_token,
                columns_select=columns_select,
                skip_header=skip_header,
                skip_footer=skip_footer,
                skip_initial_space=skip_initial_space,
                quoting=quoting,
                quote_char=quote_char,
                quote_double=quote_double,
                escape_char=escape_char,
                thousands_char=thousands_char,
                decimal_char=decimal_char,
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
            fp: TPathSpecifier,
            *,
            label: TLabel = STORE_LABEL_DEFAULT,
            index_depth: int = 0,
            index_name_depth_level: tp.Optional[TDepthLevel] = None,
            index_constructors: TIndexCtorSpecifiers = None,
            columns_depth: int = 1,
            columns_name_depth_level: tp.Optional[TDepthLevel] = None,
            columns_constructors: TIndexCtorSpecifiers = None,
            dtypes: TDtypesSpecifier = None,
            name: TName = NAME_DEFAULT,
            consolidate_blocks: bool = False,
            skip_header: int = 0,
            skip_footer: int = 0,
            trim_nadir: bool = False,
            store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT,
            ) -> tp.Self:
        '''
        Load Frame from the contents of a sheet in an XLSX workbook.

        Args:
            label: Optionally provide the sheet name from which to read. If not provided, the first sheet will be used.
        '''
        from static_frame.core.store_config import StoreConfig
        from static_frame.core.store_xlsx import StoreXLSX

        st = StoreXLSX(fp)
        config = StoreConfig(
                index_depth=index_depth,
                index_name_depth_level=index_name_depth_level,
                index_constructors=index_constructors,
                columns_depth=columns_depth,
                columns_name_depth_level=columns_name_depth_level,
                columns_constructors=columns_constructors,
                dtypes=dtypes,
                consolidate_blocks=consolidate_blocks,
                skip_header=skip_header,
                skip_footer=skip_footer,
                trim_nadir=trim_nadir,
                )
        f: tp.Self = st.read(label,
                config=config,
                store_filter=store_filter,
                container_type=cls,
                )
        return f if name is NAME_DEFAULT else f.rename(name)

    @classmethod
    def from_sqlite(cls,
            fp: TPathSpecifier,
            *,
            label: TLabel,
            index_depth: int = 0,
            index_constructors: TIndexCtorSpecifiers = None,
            columns_depth: int = 1,
            columns_constructors: TIndexCtorSpecifiers = None,
            dtypes: TDtypesSpecifier = None,
            name: TName = NAME_DEFAULT,
            consolidate_blocks: bool = False,
            # store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT,
            ) -> tp.Self:
        '''
        Load Frame from the contents of a table in an SQLite database file.
        '''
        from static_frame.core.store_config import StoreConfig
        from static_frame.core.store_sqlite import StoreSQLite

        st = StoreSQLite(fp)
        config = StoreConfig(
                index_depth=index_depth,
                index_constructors=index_constructors,
                columns_depth=columns_depth,
                columns_constructors=columns_constructors,
                dtypes=dtypes,
                consolidate_blocks=consolidate_blocks,
                )
        f: tp.Self = st.read(label,
                config=config,
                container_type=cls,
                # store_filter=store_filter,
                )
        return f if name is NAME_DEFAULT else f.rename(name)

    @classmethod
    def from_duckdb(cls,
            fp: TPathSpecifier,
            *,
            label: TLabel,
            index_depth: int = 0,
            index_constructors: TIndexCtorSpecifiers = None,
            columns_depth: int = 1,
            columns_constructors: TIndexCtorSpecifiers = None,
            consolidate_blocks: bool = False,
            ) -> tp.Self:
        '''
        Load Frame from the contents of a table in an SQLite database file.
        '''
        from static_frame.core.store_config import StoreConfig
        from static_frame.core.store_duckdb import StoreDuckDB

        st = StoreDuckDB(fp)
        config = StoreConfig(
                index_depth=index_depth,
                index_constructors=index_constructors,
                columns_depth=columns_depth,
                columns_constructors=columns_constructors,
                consolidate_blocks=consolidate_blocks,
                )
        return st.read(label, # type: ignore
                config=config,
                container_type=cls,
                )

    @classmethod
    def from_hdf5(cls,
            fp: TPathSpecifier,
            *,
            label: TLabel,
            index_depth: int = 0,
            index_constructors: TIndexCtorSpecifiers = None,
            columns_depth: int = 1,
            columns_constructors: TIndexCtorSpecifiers = None,
            name: TName = NAME_DEFAULT,
            consolidate_blocks: bool = False,
            # store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT,
            ) -> tp.Self:
        '''
        Load Frame from the contents of a table in an HDF5 file.
        '''
        from static_frame.core.store_config import StoreConfig
        from static_frame.core.store_hdf5 import StoreHDF5

        st = StoreHDF5(fp)
        config = StoreConfig(
                index_depth=index_depth,
                index_constructors=index_constructors,
                columns_depth=columns_depth,
                columns_constructors=columns_constructors,
                consolidate_blocks=consolidate_blocks,
                )
        f: tp.Self = st.read(label,
                config=config,
                container_type=cls,
                # store_filter=store_filter,
                )
        return f if name is NAME_DEFAULT else f.rename(name)

    @classmethod
    def from_npz(cls,
            fp: TPathSpecifierOrBinaryIO,
            ) -> TFrameAny:
        '''
        Create a :obj:`Frame` from an npz file.
        '''
        # NOTE: `fp`` can be a bytes object
        return NPZFrameConverter.from_archive(
                constructor=cls,
                fp=fp,
                )

    @classmethod
    def from_npy(cls,
            fp: TPathSpecifier,
            ) -> TFrameAny:
        '''
        Create a :obj:`Frame` from an directory of npy files.

        Args:
            fp: The path to the NPY directory.
        '''
        return NPYFrameConverter.from_archive(
                constructor=cls,
                fp=fp,
                )

    @classmethod
    def from_npy_mmap(cls,
            fp: TPathSpecifier,
            ) -> tp.Tuple[TFrameAny, tp.Callable[[], None]]:
        '''
        Create a :obj:`Frame` from an directory of npy files using memory maps.

        Args:
            fp: The path to the NPY directory.

        Returns:
            A tuple of :obj:`Frame` and the callable needed to close the open memory map objects. On some platforms this must be called before the process exits.
        '''
        return NPYFrameConverter.from_archive_mmap(
                constructor=cls,
                fp=fp,
                )

    @classmethod
    def from_pickle(cls,
            fp: TPathSpecifier,
            ) -> TFrameAny:
        '''
        Create a :obj:`Frame` from a pickle file.

        The pickle module is not secure. Only unpickle data you trust.

        Args:
            fp: The path to the pickle file.
        '''
        with open(fp, 'rb')as file:
            f = pickle.load(file)
        return frame_to_frame(f, cls)


    #---------------------------------------------------------------------------

    @classmethod
    @doc_inject()
    def from_pandas(cls,
            value: 'pandas.DataFrame',
            *,
            index: TIndexInitOrAuto = None,
            index_constructor: TIndexCtorSpecifier = None,
            columns: TIndexInitOrAuto = None,
            columns_constructor: TIndexCtorSpecifier = None,
            dtypes: TDtypesSpecifier = None,
            name: TName = NAME_DEFAULT,
            consolidate_blocks: bool = False,
            own_data: bool = False
            ) -> tp.Self:
        '''Given a Pandas DataFrame, return a Frame.

        Args:
            value: Pandas DataFrame.
            {index_constructor}
            {columns_constructor}
            dtypes:
            {consolidate_blocks}
            {own_data}

        Returns:
            :obj:`Frame`
        '''
        # NOTE: for specifying intra index types within IndexHierarchy, a partialed constructor must be used
        import pandas
        if not isinstance(value, pandas.DataFrame):
            raise ErrorInitFrame(f'from_pandas must be called with a Pandas DataFrame object, not: {type(value)}')

        get_col_dtype = None if dtypes is None else get_col_dtype_factory(
                dtypes,
                value.columns.values, # pyright: ignore # should be an array
                )
        # create generator of contiguous typed data
        # calling .values will force type unification across all columns
        def gen() -> tp.Iterator[TNDArrayAny]:
            pairs = enumerate(value.dtypes.values)
            column_start, dtype_current = next(pairs)
            column_last = column_start
            yield_block = False
            for column, dtype in pairs: # iloc column values
                try:
                    if dtype != dtype_current:
                        yield_block = True
                except TypeError: #pragma: no cover
                    # NOTE: raises data type not understood, happens with pd datatypes to np dtypes in pd >= 1, but fixed in later versions of pd and presently not reproducible
                    yield_block = True #pragma: no cover

                if yield_block:
                    column_end = column_last + 1
                    part = value.iloc[NULL_SLICE,
                            slice(column_start, column_end)]
                    yield from df_slice_to_arrays(part=part,
                            column_ilocs=range(column_start, column_end),
                            get_col_dtype=get_col_dtype,
                            own_data=own_data,
                            )
                    column_start = column
                    dtype_current = dtype
                    yield_block = False

                column_last = column

            # always have left over
            column_end = column_last + 1
            part = value.iloc[NULL_SLICE, slice(column_start, column_end)]
            yield from df_slice_to_arrays(part=part,
                    column_ilocs=range(column_start, column_end),
                    get_col_dtype=get_col_dtype,
                    own_data=own_data,
                    )

        if value.size == 0:
            blocks = TypeBlocks.from_zero_size_shape(value.shape, get_col_dtype)
        elif consolidate_blocks:
            blocks = TypeBlocks.from_blocks(TypeBlocks.consolidate_blocks(gen()))
        else:
            blocks = TypeBlocks.from_blocks(gen())

        if name is not NAME_DEFAULT:
            pass # keep
        elif 'name' not in value.columns and hasattr(value, 'name'):
            # avoid getting a Series if a column
            name = value.name
        else:
            name = None # do not keep as NAME_DEFAULT

        own_index = False
        if index is IndexAutoFactory:
            index = None
        elif index is not None:
            pass
        elif isinstance(value.index, pandas.MultiIndex):
            index = IndexHierarchy.from_pandas(value.index)
            own_index = True
        else:
            index = Index.from_pandas(value.index)
            own_index = index_constructor is None

        own_columns = False
        if columns is IndexAutoFactory:
            columns = None
        elif columns is not None:
            pass
        elif isinstance(value.columns, pandas.MultiIndex):
            columns = cls._COLUMNS_HIERARCHY_CONSTRUCTOR.from_pandas(value.columns)
            own_columns = True
        else:
            columns = cls._COLUMNS_CONSTRUCTOR.from_pandas(value.columns)
            own_columns = columns_constructor is None

        return cls(blocks,
                index=index,
                columns=columns,
                name=name,
                index_constructor=index_constructor,
                columns_constructor=columns_constructor,
                own_data=True,
                own_index=own_index,
                own_columns=own_columns,
                )

    @classmethod
    @doc_inject(selector='from_any')
    def from_arrow(cls,
            value: 'pyarrow.Table',
            *,
            index_depth: int = 0,
            index_name_depth_level: tp.Optional[TDepthLevel] = None,
            index_constructors: TIndexCtorSpecifiers = None,
            columns_depth: int = 1,
            columns_name_depth_level: tp.Optional[TDepthLevel] = None,
            columns_constructors: TIndexCtorSpecifiers = None,
            dtypes: TDtypesSpecifier = None,
            name: TLabel = None,
            consolidate_blocks: bool = False,
            ) -> tp.Self:
        '''Realize a ``Frame`` from an Arrow Table.

        Args:
            value: A :obj:`pyarrow.Table` instance.
            {index_depth}
            {columns_depth}
            {dtypes}
            {name}
            {consolidate_blocks}

        Returns:
            :obj:`Frame`
        '''

        # this is similar to from_structured_array
        index_start_pos = -1 # will be ignored
        index_end_pos = -1
        if index_depth > 0:
            index_start_pos = 0
            index_end_pos = index_start_pos + index_depth - 1
            apex_labels: tp.Optional[tp.Sequence[str]] = []
            index_arrays: tp.Optional[tp.Sequence[TNDArrayAny]] = []
        else:
            apex_labels = None
            index_arrays = None

        columns_labels: tp.List[TLabel] = []

        # by using value.columns_names, we expose access to the index arrays, which is deemed desirable as that is what we do in from_delimited
        get_col_dtype = None if dtypes is None else get_col_dtype_factory(
                dtypes,
                value.column_names)

        def blocks() -> tp.Iterator[TNDArrayAny]:
            for col_idx, (name, chunked_array) in enumerate(
                    zip(value.column_names, value.columns)):
                # NOTE: name will be the encoded columns representation, or auto increment integers; if an IndexHierarchy, will contain all depths: "['a' 1]"
                # This creates a Series with an index; better to find a way to go only to numpy, but does not seem available on ChunkedArray, even with pyarrow==0.16.0
                series = chunked_array.to_pandas(
                        date_as_object=False, # get an np array
                        self_destruct=True, # documented as "experimental"
                        ignore_metadata=True,
                        )

                array_final = pandas_to_numpy(series, own_data=True)

                if get_col_dtype:
                    # ordered values will include index positions
                    dtype = get_col_dtype(col_idx) #pylint: disable=E1102
                    if dtype is not None:
                        array_final = array_final.astype(dtype)

                array_final.flags.writeable = False

                is_index_col = (col_idx >= index_start_pos and col_idx <= index_end_pos)

                if is_index_col:
                    index_arrays.append(array_final) # type: ignore
                    apex_labels.append(name) # type: ignore
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

        columns_default_constructor: TIndexCtor
        if columns_depth <= 1:
            columns_default_constructor = partial(
                    cls._COLUMNS_CONSTRUCTOR,
                    name=columns_name)
        else:
            columns_default_constructor = partial(
                    cls._COLUMNS_HIERARCHY_CONSTRUCTOR.from_labels_delimited,
                    delimiter=' ',
                    name=columns_name)

        columns, own_columns = index_from_optional_constructors(
                columns_labels,
                depth=columns_depth,
                default_constructor=columns_default_constructor,
                explicit_constructors=columns_constructors, # cannot supply name
                )

        index_name = None if not apex_labels else apex_to_name(rows=(apex_labels,),
                depth_level=index_name_depth_level,
                axis=0,
                axis_depth=index_depth,
                )

        index_default_constructor: TIndexCtor # pyright: ignore
        if index_depth == 1:
            index_values = index_arrays[0] # type: ignore
            index_default_constructor = partial(Index, name=index_name) # pyright: ignore
        else: # > 1
            index_values = index_arrays

            def index_default_constructor(values: tp.Iterable[TNDArrayAny],
                    *,
                    index_constructors: TIndexCtorSpecifiers = None,
                    ) -> IndexBase:
                return IndexHierarchy._from_type_blocks(
                    TypeBlocks.from_blocks(values),
                    name=index_name,
                    index_constructors=index_constructors,
                    own_blocks=True,
                    )

        index, own_index = index_from_optional_constructors(
                index_values, # pyright: ignore
                depth=index_depth,
                default_constructor=index_default_constructor,
                explicit_constructors=index_constructors, # cannot supply name
                )

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
            fp: TPathSpecifier,
            *,
            index_depth: int = 0,
            index_name_depth_level: tp.Optional[TDepthLevel] = None,
            index_constructors: TIndexCtorSpecifiers = None,
            columns_depth: int = 1,
            columns_name_depth_level: tp.Optional[TDepthLevel] = None,
            columns_constructors: TIndexCtorSpecifiers = None,
            columns_select: tp.Optional[tp.Iterable[str]] = None,
            dtypes: TDtypesSpecifier = None,
            name: TLabel = None,
            consolidate_blocks: bool = False,
            ) -> tp.Self:
        '''
        Realize a ``Frame`` from a Parquet file.

        Args:
            {fp}
            {index_depth}
            index_name_depth_level:
            index_constructors:
            {columns_depth}
            columns_name_depth_level:
            columns_constructors:
            {columns_select}
            {dtypes}
            {name}
            {consolidate_blocks}
        '''
        import pyarrow.parquet as pq
        from pyarrow.lib import ArrowInvalid  # pylint: disable=E0611

        if columns_select and index_depth != 0:
            raise ErrorInitFrame(f'cannot load index_depth {index_depth} when columns_select is specified.')

        fpf: str = path_filter(fp) # type: ignore

        if columns_select is not None and not isinstance(columns_select, list):
            columns_select = list(columns_select)

        # NOTE: the order of columns_select will determine their order
        try:
            table = pq.read_table(fp,
                    columns=columns_select,
                    use_pandas_metadata=False,
                    )
        except ArrowInvalid:  # pragma: no cover
            # support loading parquet files saved with pyarrow<1.0
            # https://github.com/apache/arrow/issues/32660
            table = pq.read_table(fp,  # pragma: no cover
                    columns=columns_select,
                    use_pandas_metadata=False,
                    use_legacy_dataset=True,
                    )
        if columns_select:
            # pq.read_table will silently accept requested columns that are not found; this can be identified if we got back fewer columns than requested
            if len(table.column_names) < len(columns_select):
                missing = set(columns_select) - set(table.column_names)
                raise ErrorInitFrame(f'cannot load all columns in columns_select: missing {missing}')

        return cls.from_arrow(table,
                index_depth=index_depth,
                index_name_depth_level=index_name_depth_level,
                index_constructors=index_constructors,
                columns_depth=columns_depth,
                columns_name_depth_level=columns_name_depth_level,
                columns_constructors=columns_constructors,
                dtypes=dtypes,
                consolidate_blocks=consolidate_blocks,
                name=name
                )

    @staticmethod
    @doc_inject(selector='constructor_frame')
    def from_msgpack(
            msgpack_data: bytes
            ) -> TFrameAny:
        '''Frame constructor from an in-memory binary object formatted as a msgpack.

        Args:
            msgpack_data: A binary msgpack object, encoding a Frame as produced from to_msgpack()
        '''
        import msgpack  # type: ignore
        import msgpack_numpy  # type: ignore

        def decode(obj: tp.Dict[bytes, tp.Any], #dict produced by msgpack-python
                chain: tp.Callable[[tp.Any], str] = msgpack_numpy.decode,
                ) -> object:

            if b'sf' in obj:
                cls_name = obj[b'sf']
                cls = ContainerMap.get(cls_name)

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
                    index_constructors: tp.List[tp.Type[TIndexAny]] = [ # pyright: ignore
                            ContainerMap.get(cls_name) for cls_name in unpackb(
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

        return unpackb(msgpack_data) # type: ignore

    #---------------------------------------------------------------------------
    def __init__(self,
            data: TFrameInitializer = FRAME_INITIALIZER_DEFAULT, # type: ignore
            *,
            index: TIndexInitOrAuto = None,
            columns: TIndexInitOrAuto = None,
            name: TName = NAME_DEFAULT,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            own_data: bool = False,
            own_index: bool = False,
            own_columns: bool = False,
            ) -> None:
        '''
        Initializer.

        Args:
            data: Default Frame initialization requires typed data such as a NumPy array. All other initialization should use specialized constructors.
            {index}
            {columns}
            index_constructor:
            columns_constructor:
            {own_data}
            {own_index}
            {own_columns}
        '''
        #-----------------------------------------------------------------------
        # blocks assignment

        blocks_constructor = _NA_BLOCKS_CONSTRCTOR

        if data.__class__ is TypeBlocks:
            if own_data:
                self._blocks = data # type: ignore
            else:
                # assume we need to create a new TB instance; this will not copy underlying arrays as all blocks are immutable
                self._blocks = TypeBlocks.from_blocks(data._blocks) # type: ignore
        elif data.__class__ is np.ndarray:
            if own_data:
                data.flags.writeable = False # type: ignore
            # from_blocks will apply immutable filter
            self._blocks = TypeBlocks.from_blocks(data) # type: ignore
        elif data is FRAME_INITIALIZER_DEFAULT:
            # NOTE: this will not catch all cases where index or columns is empty, as they might be iterators; those cases will be handled below.
            def blocks_constructor(shape: tp.Tuple[int, int]) -> None: #pylint: disable=E0102
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
            if name is NAME_DEFAULT:
                name = data.name

        elif isinstance(data, dict):
            raise ErrorInitFrame('use Frame.from_dict to create a Frame from a mapping.')
        elif isinstance(data, Series):
            raise ErrorInitFrame('use Frame.from_series to create a Frame from a Series.')
        else:
            raise ErrorInitFrame('use Frame.from_element, Frame.from_elements, or Frame.from_records to create a Frame from 0, 1, or 2 dimensional untyped data (respectively).')

        # counts can be zero (not None) if _block was created but is empty
        row_count, col_count = (self._blocks.shape # pyright: ignore
                if blocks_constructor is _NA_BLOCKS_CONSTRCTOR else (None, None))

        self._name = None if name is NAME_DEFAULT else name_filter(name) # pyright: ignore

        #-----------------------------------------------------------------------
        # columns assignment

        if own_columns:
            self._columns = columns # type: ignore
            col_count = len(self._columns) # pyright: ignore
        elif index_constructor_empty(columns):
            col_count = 0 if col_count is None else col_count
            self._columns = IndexAutoFactory.from_optional_constructor(
                    col_count,
                    default_constructor=self._COLUMNS_CONSTRUCTOR,
                    explicit_constructor=columns_constructor
                    )
        else:
            try:
                self._columns = index_from_optional_constructor(columns,
                        default_constructor=self._COLUMNS_CONSTRUCTOR,
                        explicit_constructor=columns_constructor
                        )
            except ErrorInitIndex as e: # show this as a column exception
                raise ErrorInitColumns(str(e)) from None

            col_count = len(self._columns)
        # check after creation, as we cannot determine from the constructor (it might be a method on a class)
        if self._COLUMNS_CONSTRUCTOR.STATIC != self._columns.STATIC: # pyright: ignore
            raise ErrorInitFrame(f'Supplied `columns_constructor` does not match required static attribute: {self._COLUMNS_CONSTRUCTOR.STATIC}')

        #-----------------------------------------------------------------------
        # index assignment

        if own_index:
            self._index = index # type: ignore
            row_count = len(self._index) # pyright: ignore
        elif index_constructor_empty(index):
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

        if not self._index.STATIC: # pyright: ignore
            raise ErrorInitFrame('non-static index cannot be assigned to Frame')

        #-----------------------------------------------------------------------
        # final evaluation

        if blocks_constructor is not _NA_BLOCKS_CONSTRCTOR:
            # if we have a blocks_constructor if is because data remained FRAME_INITIALIZER_DEFAULT
            blocks_constructor((row_count, col_count))

        # final check of block/index coherence
        block_row, block_col = self._blocks.shape
        if block_row != row_count: # pyright: ignore
            # row count might be 0 for an empty DF
            raise ErrorInitFrame(
                f'Index has incorrect size (got {block_row}, expected {row_count})' # pyright: ignore
                )
        if block_col != col_count: # pyright: ignore
            raise ErrorInitFrame(
                f'Columns has incorrect size (got {block_col}, expected {col_count})' # pyright: ignore
                )

    #---------------------------------------------------------------------------

    def __deepcopy__(self, memo: tp.Dict[int, tp.Any]) -> tp.Self:
        obj = self.__class__.__new__(self.__class__)
        obj._blocks = deepcopy(self._blocks, memo)
        obj._columns = deepcopy(self._columns, memo)
        obj._index = deepcopy(self._index, memo)
        obj._name = self._name # should be hashable/immutable

        memo[id(self)] = obj
        return obj

    # def __copy__(self) -> TFrameAny:
    #     '''
    #     Return shallow copy of this Frame.
    #     '''

    # def copy(self)-> TFrameAny:
    #     '''
    #     Return shallow copy of this Frame.
    #     '''
    #     return self.__copy__() #type: ignore

    def _memory_label_component_pairs(self,
            ) -> tp.Iterable[tp.Tuple[str, tp.Any]]:
        return (('Name', self._name),
                ('Index', self._index),
                ('Columns', self._columns),
                ('Blocks', self._blocks),
                )

    #---------------------------------------------------------------------------
    # external protocols

    def __dataframe__(self,
            nan_as_null: bool = False,
            allow_copy: bool = True,
            ) -> DFIDataFrame:
        '''Return a data-frame interchange protocol compliant object. See https://data-apis.org/dataframe-protocol/latest for more information.
        '''
        return DFIDataFrame(self,
                nan_as_null=nan_as_null,
                allow_copy=allow_copy,
                recast_blocks=True,
                )

    #---------------------------------------------------------------------------
    # name interface

    @property
    @doc_inject()
    def name(self) -> TName:
        '''{}'''
        return self._name

    def rename(self,
            name: TName = NAME_DEFAULT,
            *,
            index: TName = NAME_DEFAULT,
            columns: TName = NAME_DEFAULT,
            ) -> tp.Self:
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
    def loc(self) -> InterGetItemLocCompoundReduces[TFrameAny]:
        return InterGetItemLocCompoundReduces(self._extract_loc)

    @property
    def iloc(self) -> InterGetItemILocCompoundReduces[TFrameAny]:
        return InterGetItemILocCompoundReduces(self._extract_iloc)

    @property
    def bloc(self) -> InterfaceGetItemBLoc[TSeriesAny]:
        return InterfaceGetItemBLoc(self._extract_bloc)

    @property
    def drop(self) -> InterfaceSelectTrio[TFrameAny]:
        return InterfaceSelectTrio( # type: ignore # NOTE: does not reuturn Frame, but a delegate
            func_iloc=self._drop_iloc,
            func_loc=self._drop_loc,
            func_getitem=self._drop_getitem)

    @property
    def mask(self) -> InterfaceSelectTrio[TFrameAny]:
        return InterfaceSelectTrio( # type: ignore # NOTE: does not return Frame, but a delegate
            func_iloc=self._extract_iloc_mask,
            func_loc=self._extract_loc_mask,
            func_getitem=self._extract_getitem_mask)

    @property
    def masked_array(self) -> InterfaceSelectTrio[TFrameAny]:
        return InterfaceSelectTrio( # type: ignore
            func_iloc=self._extract_iloc_masked_array,
            func_loc=self._extract_loc_masked_array,
            func_getitem=self._extract_getitem_masked_array)

    # NOTE: the typing needs work as it does not return `Frame`, but FrameAssignILoc
    @property
    def assign(self) -> InterfaceAssignQuartet[FrameAssignILoc]:
        return InterfaceAssignQuartet( # type: ignore
            func_iloc=self._extract_iloc_assign,
            func_loc=self._extract_loc_assign,
            func_getitem=self._extract_getitem_assign,
            func_bloc=self._extract_bloc_assign,
            delegate=FrameAssign,
            )

    @property
    @doc_inject(select='astype')
    def astype(self) -> InterfaceFrameAsType[TFrameAny]:
        '''
        Retype one or more columns. When used as a function, can be used to retype the entire ``Frame``. Alternatively, when used as a ``__getitem__`` interface, loc-style column selection can be used to type one or more coloumns.

        Args:
            {dtype}
        '''
        # NOTE: this uses the same function for __call__ and __getitem__; call simply uses the NULL_SLICE and applys the dtype argument immediately
        return InterfaceFrameAsType(func_getitem=self._extract_getitem_astype)

    @property
    def consolidate(self) -> InterfaceConsolidate[TFrameAny]:
        '''
        Consolidate one or more columns. When used as a function, can be used to retype the entire ``Frame``. Alternatively, when used as a ``__getitem__`` interface, loc-style column selection can be used to consolidate one or more coloumns.

        '''
        return InterfaceConsolidate(
                container=self,
                func_getitem=self._extract_getitem_consolidate,
                )

    #---------------------------------------------------------------------------
    # via interfaces

    @property
    def via_values(self) -> InterfaceValues[TFrameAny]:
        '''
        Interface for applying functions to values (as arrays) in this container.

        Args:
            consolidate_blocks: Group adjacent same-typed arrays into 2D arrays.
            unify_blocks: Group all arrays into single array, re-typing to an appropriate dtype.
            dtype: specify a dtype to be used in conversion before consolidation or unification, and before function application.
        '''
        return InterfaceValues(self)

    @property
    def via_str(self) -> InterfaceString[TFrameAny]:
        '''
        Interface for applying string methods to elements in this container.
        '''
        def blocks_to_container(blocks: tp.Iterator[TNDArrayAny]) -> TFrameAny:
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
                ndim=self._NDIM,
                labels=self._columns,
                )

    @property
    def via_dt(self) -> InterfaceDatetime[TFrameAny]:
        '''
        Interface for applying datetime properties and methods to elements in this container.
        '''

        # NOTE: we only process object dt64 types; strings have to be converted explicitly

        def blocks_to_container(blocks: tp.Iterator[TNDArrayAny]) -> TFrameAny:
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
    def via_T(self) -> InterfaceTranspose[TFrameAny]:
        '''
        Interface for using binary operators with one-dimensional sequences, where the opperand is applied column-wise.
        '''
        return InterfaceTranspose(
                container=self,
                )


    def via_fill_value(self,
            fill_value: tp.Any = np.nan,
            ) -> InterfaceFillValue[TFrameAny]:
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
            ) -> InterfaceRe[TFrameAny]:
        '''
        Interface for applying regular expressions to elements in this container.
        '''
        def blocks_to_container(blocks: tp.Iterator[TNDArrayAny]) -> TFrameAny:
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
    def iter_array(self) -> IterNodeAxis[TFrameAny]:
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
    def iter_array_items(self) -> IterNodeAxis[TFrameAny]:
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
    def iter_tuple(self) -> IterNodeConstructorAxis[TFrameAny]:
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
    def iter_tuple_items(self) -> IterNodeConstructorAxis[TFrameAny]:
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
    def iter_series(self) -> IterNodeAxis[TFrameAny]:
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
    def iter_series_items(self) -> IterNodeAxis[TFrameAny]:
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
    def iter_group(self) -> IterNodeGroupAxis[TFrameAny]:
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
    def iter_group_items(self) -> IterNodeGroupAxis[TFrameAny]:
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

    #---------------------------------------------------------------------------
    @property
    def iter_group_array(self) -> IterNodeGroupAxis[TFrameAny]:
        '''
        Iterator of ``np.ndarray`` grouped by unique values found in one or more columns (axis=0) or rows (axis=1).
        '''
        return IterNodeGroupAxis(
                container=self,
                function_values=partial(self._axis_group_loc, as_array=True),
                function_items=partial(self._axis_group_loc_items, as_array=True),
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_VALUES,
                )

    @property
    def iter_group_array_items(self) -> IterNodeGroupAxis[TFrameAny]:
        '''
        Iterator of pairs of label, ``np.ndarray`` grouped by unique values found in one or more columns (axis=0) or rows (axis=1).
        '''
        return IterNodeGroupAxis(
                container=self,
                function_values=partial(self._axis_group_loc, as_array=True),
                function_items=partial(self._axis_group_loc_items, as_array=True),
                yield_type=IterNodeType.ITEMS,
                apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_VALUES,
                )

    #---------------------------------------------------------------------------
    @property
    def iter_group_labels(self) -> IterNodeDepthLevelAxis[TFrameAny]:
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
    def iter_group_labels_items(self) -> IterNodeDepthLevelAxis[TFrameAny]:
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
    def iter_group_labels_array(self) -> IterNodeDepthLevelAxis[TFrameAny]:
        '''
        Iterator of ``np.ndarray`` grouped by unique labels found in one or more index depths (axis=0) or columns depths (axis=1).
        '''
        return IterNodeDepthLevelAxis(
                container=self,
                function_values=partial(self._axis_group_labels, as_array=True),
                function_items=partial(self._axis_group_labels_items, as_array=True),
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_LABELS,
                )

    @property
    def iter_group_labels_array_items(self) -> IterNodeDepthLevelAxis[TFrameAny]:
        '''
        Iterator of pairs of label, ``np.ndarray`` grouped by unique labels found in one or more index depths (axis=0) or columns depths (axis=1).
        '''
        return IterNodeDepthLevelAxis(
                container=self,
                function_values=partial(self._axis_group_labels, as_array=True),
                function_items=partial(self._axis_group_labels_items, as_array=True),
                yield_type=IterNodeType.ITEMS,
                apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_LABELS,
                )


    #---------------------------------------------------------------------------
    @property
    def iter_group_other(self) -> IterNodeGroupOtherReducible[TFrameAny]:
        '''
        Iterator of :obj:`Frame` grouped by unique values found in a supplied container.
        '''
        return IterNodeGroupOtherReducible(
                container=self,
                function_values=self._axis_group_other,
                function_items=self._axis_group_other_items,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_VALUES,
                )

    @property
    def iter_group_other_items(self) -> IterNodeGroupOtherReducible[TFrameAny]:
        '''
        Iterator of :obj:`Frame` grouped by unique values found in a supplied container.
        '''
        return IterNodeGroupOtherReducible(
                container=self,
                function_values=self._axis_group_other,
                function_items=self._axis_group_other_items,
                yield_type=IterNodeType.ITEMS,
                apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_VALUES,
                )

    #---------------------------------------------------------------------------
    @property
    def iter_group_other_array(self) -> IterNodeGroupOtherReducible[TFrameAny]:
        '''
        Iterator of :obj:`Frame` grouped by unique values found in a supplied container.
        '''
        return IterNodeGroupOtherReducible(
                container=self,
                function_values=partial(self._axis_group_other,
                        as_array=True),
                function_items=partial(self._axis_group_other_items,
                        as_array=True),
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_VALUES,
                )

    @property
    def iter_group_other_array_items(self) -> IterNodeGroupOtherReducible[TFrameAny]:
        '''
        Iterator of :obj:`Frame` grouped by unique values found in a supplied container.
        '''
        return IterNodeGroupOtherReducible(
                container=self,
                function_values=partial(self._axis_group_other,
                        as_array=True),
                function_items=partial(self._axis_group_other_items,
                        as_array=True),
                yield_type=IterNodeType.ITEMS,
                apply_type=IterNodeApplyType.SERIES_ITEMS_GROUP_VALUES,
                )

    #---------------------------------------------------------------------------
    @property
    @doc_inject(selector='window')
    def iter_window(self) -> IterNodeWindowReducible[TFrameAny]:
        '''
        Iterator of windowed values, where values are given as a :obj:`Frame`.

        {args}
        '''
        function_values = partial(self._axis_window, as_array=False)
        function_items = partial(self._axis_window_items, as_array=False)
        return IterNodeWindowReducible(
                container=self,
                function_values=function_values,
                function_items=function_items,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.SERIES_ITEMS,
                )

    @property
    @doc_inject(selector='window')
    def iter_window_items(self) -> IterNodeWindowReducible[TFrameAny]:
        '''
        Iterator of pairs of label, windowed values, where values are given as a :obj:`Frame`.

        {args}
        '''
        function_values = partial(self._axis_window, as_array=False)
        function_items = partial(self._axis_window_items, as_array=False)
        return IterNodeWindowReducible(
                container=self,
                function_values=function_values,
                function_items=function_items,
                yield_type=IterNodeType.ITEMS,
                apply_type=IterNodeApplyType.SERIES_ITEMS,
                )

    @property
    @doc_inject(selector='window')
    def iter_window_array(self) -> IterNodeWindowReducible[TFrameAny]:
        '''
        Iterator of windowed values, where values are given as a :obj:`np.array`.

        {args}
        '''
        function_values = partial(self._axis_window, as_array=True)
        function_items = partial(self._axis_window_items, as_array=True)
        return IterNodeWindowReducible(
                container=self,
                function_values=function_values,
                function_items=function_items,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.SERIES_ITEMS,
                )

    @property
    @doc_inject(selector='window')
    def iter_window_array_items(self) -> IterNodeWindowReducible[TFrameAny]:
        '''
        Iterator of pairs of label, windowed values, where values are given as a :obj:`np.array`.

        {args}
        '''
        function_values = partial(self._axis_window, as_array=True)
        function_items = partial(self._axis_window_items, as_array=True)
        return IterNodeWindowReducible(
                container=self,
                function_values=function_values,
                function_items=function_items,
                yield_type=IterNodeType.ITEMS,
                apply_type=IterNodeApplyType.SERIES_ITEMS,
                )

    #---------------------------------------------------------------------------
    @property
    def iter_element(self) -> IterNodeAxisElement[TFrameAny]:
        '''Iterator of elements, ordered by row then column.
        '''
        return IterNodeAxisElement(
                container=self,
                function_values=self._iter_element_loc,
                function_items=self._iter_element_loc_items,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.FRAME_ELEMENTS
                )

    @property
    def iter_element_items(self) -> IterNodeAxisElement[TFrameAny]:
        '''Iterator of pairs of label, element, where labels are pairs of index, columns labels, ordered by row then column.
        '''
        return IterNodeAxisElement(
                container=self,
                function_values=self._iter_element_loc,
                function_items=self._iter_element_loc_items,
                yield_type=IterNodeType.ITEMS,
                apply_type=IterNodeApplyType.FRAME_ELEMENTS
                )

    #---------------------------------------------------------------------------
    @property
    def reduce(self) -> ReduceDispatchAligned:
        '''Return a ``ReduceAligned`` interface, permitting function application per column or on entire containers.
        '''
        from static_frame.core.reduce import ReduceDispatchAligned
        return ReduceDispatchAligned(
                ((self._name, self),),
                self._columns,
                yield_type=IterNodeType.VALUES,
                )

    #---------------------------------------------------------------------------
    # index manipulation

    def _reindex_other_like_iloc(self,
            value: tp.Union[TSeriesAny, TFrameAny],
            iloc_key: TILocSelectorCompound,
            is_series: bool,
            is_frame: bool,
            fill_value: tp.Any = np.nan,
            ) -> tp.Union[TSeriesAny, TFrameAny]:
        '''Given a value that is a Series or Frame, reindex it to the index components, drawn from this Frame, that are specified by the iloc_key.
        '''
        # assert iloc_key.__class__ is tuple # must already be normalized
        assert is_series ^ is_frame # one must be True

        row_key: TILocSelector
        col_key: TILocSelector
        row_key, col_key = iloc_key # type: ignore

        # within this frame, get Index objects by extracting based on passed-in iloc keys
        # NOTE: NM (not many) means an integer or label
        nm_row, nm_column = self._extract_axis_not_multi(row_key, col_key)
        v: None | TSeriesAny | TFrameAny = None
        col_key_many: TILocSelectorMany
        row_key_many: TILocSelectorMany

        if nm_row and not nm_column:
            # only column is multi selection, reindex by column
            if is_series:
                col_key_many = col_key # type: ignore[assignment]
                v = value.reindex(self._columns._extract_iloc(col_key_many),
                        fill_value=fill_value)
        elif not nm_row and nm_column:
            # only row is multi selection, reindex by index
            if is_series:
                row_key_many = row_key # type: ignore[assignment]
                v = value.reindex(self._index._extract_iloc(row_key_many),
                        fill_value=fill_value)
        elif not nm_row and not nm_column:
            # both multi, must be a Frame
            if is_frame:
                col_key_many = col_key # type: ignore[assignment]
                row_key_many = row_key # type: ignore[assignment]
                target_column_index = self._columns._extract_iloc(col_key_many)
                target_row_index = self._index._extract_iloc(row_key_many)
                # this will use the default fillna type, which may or may not be what is wanted
                v = value.reindex( # type: ignore
                        index=target_row_index,
                        columns=target_column_index, # pyright: ignore
                        fill_value=fill_value)
        if v is None:
            raise RuntimeError(f'cannot assign {value.__class__.__name__} with key configuration: {nm_row}, {nm_column}')
        return v

    @doc_inject(selector='reindex', class_name='Frame')
    def reindex(self,
            index: tp.Optional[TIndexInitializer] = None,
            columns: tp.Optional[TIndexInitializer] = None,
            *,
            fill_value: tp.Any = np.nan,
            own_index: bool = False,
            own_columns: bool = False,
            check_equals: bool = True,
            ) -> tp.Self:
        '''
        {doc}

        Args:
            index: {index_initializer}
            columns: {index_initializer}
            {fill_value}
            {own_index}
            {own_columns}
            check_equals:
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
                index_ic = IndexCorrespondence.from_correspondence(self._index, index) # type: ignore
        else:
            index = self._index
            index_ic = None
        # index can always be owned by this point, as self._index is STATIC, or  we have created a new Index, or we have bbeen given own_index
        own_index_frame = True

        columns_owned: IndexBase
        if columns is not None:
            if not own_columns:
                columns_owned = index_from_optional_constructor(columns,
                        default_constructor=self._COLUMNS_CONSTRUCTOR)
            else:
                columns_owned = columns # type: ignore

            if check_equals and self._columns.equals(columns):
                columns_ic = None
            else:
                columns_ic = IndexCorrespondence.from_correspondence(self._columns, columns_owned)
            own_columns_frame = True
        else:
            columns_owned = self._columns
            columns_ic = None
            own_columns_frame = self._COLUMNS_CONSTRUCTOR.STATIC

        # if fill_value is a non-element, call get_col_fill_value_factory with the new index/columns, not the old
        if is_fill_value_factory_initializer(fill_value):
            get_col_fill_value = get_col_fill_value_factory(fill_value, columns=columns_owned)
            return self.__class__(
                    TypeBlocks.from_blocks(
                            self._blocks.resize_blocks_by_callable(
                                    index_ic=index_ic,
                                    columns_ic=columns_ic,
                                    fill_value=get_col_fill_value),
                            shape_reference=(len(index), len(columns_owned)) #type: ignore
                            ),
                    index=index,
                    columns=columns_owned,
                    name=self._name,
                    own_data=True,
                    own_index=own_index_frame,
                    own_columns=own_columns_frame
                    )

        return self.__class__(
                TypeBlocks.from_blocks(
                        self._blocks.resize_blocks_by_element(
                                index_ic=index_ic,
                                columns_ic=columns_ic,
                                fill_value=fill_value),
                        shape_reference=(len(index), len(columns_owned)) #type: ignore
                        ),
                index=index,
                columns=columns_owned,
                name=self._name,
                own_data=True,
                own_index=own_index_frame,
                own_columns=own_columns_frame
                )

    @doc_inject(selector='relabel', class_name='Frame')
    def relabel(self,
            index: tp.Optional[TRelabelInput] = None,
            columns: tp.Optional[TRelabelInput] = None,
            *,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            ) -> tp.Self:
        '''
        {doc}

        Args:
            index: {relabel_input_index}
            columns: {relabel_input_columns}
        '''
        own_index = False
        if index is IndexAutoFactory:
            index = None
        elif is_callable_or_mapping(index):
            index = self._index.relabel(index) # type: ignore
            # can only own if index_constructor is None
            own_index = index_constructor is None
        elif index is None:
            index = self._index
            own_index = index_constructor is None
        elif isinstance(index, Set):
            raise RelabelInvalid()

        own_columns = False
        if columns is IndexAutoFactory:
            columns = None
        elif is_callable_or_mapping(columns):
            columns = self._columns.relabel(columns) # type: ignore
            # can only own if columns_constructor is None
            own_columns = columns_constructor is None
        elif columns is None:
            columns = self._columns
            own_columns = columns_constructor is None and self.STATIC
        elif isinstance(columns, Set):
            raise RelabelInvalid()

        return self.__class__(
                self._blocks.copy(), # does not copy arrays
                index=index, # type: ignore
                columns=columns, # type: ignore
                name=self._name,
                index_constructor=index_constructor,
                columns_constructor=columns_constructor,
                own_data=True,
                own_index=own_index,
                own_columns=own_columns,
                )

    @doc_inject(selector='relabel_flat', class_name='Frame')
    def relabel_flat(self,
            index: bool = False,
            columns: bool = False
            ) -> tp.Self:
        '''
        {doc}

        Args:
            index: Boolean to flag flatening on the index.
            columns: Boolean to flag flatening on the columns.
        '''
        if not index and not columns:
            raise RuntimeError('must specify one or both of columns, index')

        index_owned = self._index.flat() if index else self._index.copy() # type: ignore
        columns_owned = self._columns.flat() if columns else self._columns.copy() # type: ignore

        return self.__class__(
                self._blocks.copy(), # does not copy arrays
                index=index_owned,
                columns=columns_owned,
                name=self._name,
                own_data=True,
                own_index=True,
                own_columns=True)

    @doc_inject(selector='relabel_level_add', class_name='Frame')
    def relabel_level_add(self,
            index: TLabel = None,
            columns: TLabel = None,
            *,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            ) -> tp.Self:
        '''
        {doc}

        Args:
            index: {level}
            columns: {level}
            *
            index_constructor:
            columns_constructor:
        '''
        index_final = (self._index.level_add(
                index, index_constructor=index_constructor)
                if index is not None else self._index
                )
        columns_final = (self._columns.level_add(
                columns, index_constructor=columns_constructor)
                if columns is not None else self._columns
                )

        return self.__class__(
                self._blocks.copy(), # does not copy arrays
                index=index_final,
                columns=columns_final,
                name=self._name,
                own_data=True,
                own_index=True,
                own_columns=self.STATIC)

    @doc_inject(selector='relabel_level_drop', class_name='Frame')
    def relabel_level_drop(self,
            index: int = 0,
            columns: int = 0
            ) -> tp.Self:
        '''
        {doc}

        Args:
            index: {count} Default is zero.
            columns: {count} Default is zero.
        '''

        index_owned = self._index.level_drop(index) if index else self._index.copy() # type: ignore
        columns_owned = self._columns.level_drop(columns) if columns else self._columns.copy() # type: ignore

        return self.__class__(
                self._blocks.copy(), # does not copy arrays
                index=index_owned,
                columns=columns_owned,
                name=self._name,
                own_data=True,
                own_index=True,
                own_columns=True)

    def relabel_shift_in(self,
            key: TLocSelector,
            *,
            axis: int = 0,
            index_constructors: TIndexCtorSpecifiers = None,
            ) -> tp.Self:
        '''
        Create, or augment, an :obj:`IndexHierarchy` by providing one or more selections from the Frame (via axis-appropriate ``loc`` selections) to move into the :obj:`Index`.

        Args:
            key: a loc-style selection on the opposite axis.
            axis: 0 modifies the index by selecting columns with ``key``; 1 modifies the columns by selecting rows with ``key``.
        '''

        if axis == 0: # select from columns, add to index
            index_target = self._index
            index_opposite = self._columns
            target_default_ctr = Index
        else:
            index_target = self._columns
            index_opposite = self._index
            target_default_ctr = self._COLUMNS_CONSTRUCTOR

        name_prior: tp.Tuple[TName, ...]
        ih_index_constructors: tp.List[TIndexCtorSpecifier]

        if index_target.depth == 1:
            ih_blocks = TypeBlocks.from_blocks((index_target.values,))
            name_prior = index_target.names if index_target.name is None else (index_target.name,)
            ih_index_constructors = [index_target.__class__]
        else:
            # No recache is needed as it's not possible for an index to be GO
            ih_blocks = index_target._blocks.copy() # type: ignore # will mutate copied blocks
            # only use string form of labels if we are not storing a correctly sized tuple
            name_prior = index_target.name if index_target._name_is_names() else index_target.names # type: ignore
            ih_index_constructors = index_target.index_types.values.tolist()

        iloc_key = index_opposite._loc_to_iloc(key)
        # NOTE: must do this before dropping
        name_posterior: tp.Tuple[TLabel, ...]
        if isinstance(iloc_key, INT_TYPES):
            name_posterior = (index_opposite[iloc_key],)
        else:
            name_posterior = tuple(index_opposite[iloc_key])

        ih_name = name_prior + name_posterior

        if index_constructors is None:
            ih_index_constructors.extend(target_default_ctr for _ in name_posterior)
        elif callable(index_constructors): # one constructor
            ih_index_constructors.extend(index_constructors for _ in name_posterior) # pyright: ignore
        else: # assume properly sized iterable
            ih_index_constructors.extend(index_constructors)
            if len(ih_index_constructors) != len(ih_name):
                raise RuntimeError('Incorrect number of values in index_constructors.')

        index_opposite = index_opposite._drop_iloc(iloc_key)

        if axis == 0: # select from columns, add to index
            ih_blocks.extend(self._blocks._extract(column_key=iloc_key))
            frame_blocks = TypeBlocks.from_blocks(
                    self._blocks._drop_blocks(column_key=iloc_key),
                    shape_reference=(self.shape[0], len(index_opposite)),
                    )

            index = IndexHierarchy._from_type_blocks(
                    ih_blocks,
                    name=ih_name,
                    index_constructors=ih_index_constructors,
                    )
            columns = index_opposite
        else: # select from index, add to columns
            ih_blocks.extend(self._blocks._extract(row_key=iloc_key).transpose())
            frame_blocks = TypeBlocks.from_blocks(
                    self._blocks._drop_blocks(row_key=iloc_key),
                    shape_reference=(len(index_opposite), self.shape[1]),
                    )
            index = index_opposite # type: ignore
            columns = self._COLUMNS_HIERARCHY_CONSTRUCTOR._from_type_blocks(
                    ih_blocks,
                    name=ih_name,
                    index_constructors=ih_index_constructors,
                    )

        return self.__class__(
                frame_blocks, # does not copy arrays
                index=index,
                columns=columns,
                name=self._name,
                own_data=True,
                own_index=True,
                own_columns=True)


    def relabel_shift_out(self,
            depth_level: TDepthLevel,
            *,
            axis: int = 0,
            ) -> tp.Self:
        '''
        Shift values from an index on an axis to the Frame by providing one or more depth level selections.

        Args:
            dpeth_level: an iloc-style selection on the :obj:`Index` of the specified axis.
            axis: 0 modifies the index by selecting columns with ``depth_level``; 1 modifies the columns by selecting rows with ``depth_level``.
        '''

        if axis == 0: # select from index, remove from index
            index_target = self._index
            target_ctors = self._index.index_types # Series
            target_hctor = IndexHierarchy
        elif axis == 1:
            index_target = self._columns
            target_ctors = self._columns.index_types # Series
            target_hctor = self._COLUMNS_HIERARCHY_CONSTRUCTOR
        else:
            raise AxisInvalid(f'invalid axis {axis}')

        new_labels: tp.Iterable[TLabel]

        if index_target.depth == 1:
            index_target._depth_level_validate(depth_level) # type: ignore # will raise
            new_target = IndexAutoFactory
            add_blocks = (index_target.values,)
            new_labels = index_target.names if index_target.name is None else (index_target.name,)
        else:
            if index_target._recache:
                index_target._update_array_cache()

            label_src: tp.Tuple[TName] = (index_target.name if index_target._name_is_names() # type: ignore
                    else index_target.names)

            if isinstance(depth_level, INT_TYPES):
                new_labels = (label_src[depth_level],)
                remain_labels = tuple(label for i, label
                        in enumerate(label_src) if i != depth_level)
            else:
                new_labels = (label_src[i] for i in depth_level)
                remain_labels = tuple(label for i, label
                        in enumerate(label_src) if i not in depth_level)

            target_tb = index_target._blocks # type: ignore
            add_blocks = target_tb._slice_blocks(None,
                    depth_level,
                    False,
                    True)

            # this might fail if nothing left
            remain_blocks = TypeBlocks.from_blocks(
                    target_tb._drop_blocks(column_key=depth_level),
                    shape_reference=(len(index_target), 0))

            remain_columns = remain_blocks.shape[1]
            if remain_columns == 0:
                new_target = IndexAutoFactory
            elif remain_columns == 1:
                target_ctor = target_ctors.drop.iloc[depth_level].iloc[0]
                new_target = target_ctor(
                        column_1d_filter(remain_blocks._blocks[0]),
                        name=remain_labels[0])
            else:
                index_constructors = target_ctors.drop.iloc[depth_level].values
                new_target = target_hctor._from_type_blocks( # type: ignore
                        remain_blocks,
                        name=remain_labels,
                        index_constructors=index_constructors,
                        )

        if axis == 0: # select from index, remove from index
            blocks = TypeBlocks.from_blocks(chain(add_blocks,
                    self._blocks._blocks))
            index = new_target
            # if we already have a hierarchical index here, there is no way to ensure that the new labels coming in are of appropriate depth; only option is to get a flat version of columns
            if self._columns.depth > 1:
                extend_labels = self._columns.flat().__iter__() # type: ignore
            else:
                extend_labels = self._columns.__iter__()
            columns = self._COLUMNS_CONSTRUCTOR.from_labels(
                    chain(new_labels, extend_labels), # type: ignore
                    name=self._columns.name,
                    )
        else:
            blocks = TypeBlocks.from_blocks(TypeBlocks.vstack_blocks_to_blocks(
                    (TypeBlocks.from_blocks(add_blocks).transpose(), self._blocks))
                    )
            if self._index.depth > 1:
                extend_labels = self._index.flat().__iter__() # type: ignore
            else:
                extend_labels = self._index.__iter__()
            index = Index.from_labels(
                    chain(new_labels, extend_labels), # type: ignore
                    name=self._index.name)
            columns = new_target # type: ignore

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
            index: tp.Optional[tp.Sequence[int]] = None,
            columns: tp.Optional[tp.Sequence[int]] = None,
            *,
            index_constructors: TIndexCtorSpecifiers = None,
            columns_constructors: TIndexCtorSpecifiers = None,
            ) -> tp.Self:
        '''
        Produce a new `Frame` with index and/or columns constructed with a transformed hierarchy.

        Args:
            index: Depth level specifier
            columns: Depth level specifier
        '''
        if index and self.index.depth == 1:
            raise RuntimeError('cannot rehierarch on index when there is no hierarchy')
        if columns and self.columns.depth == 1:
            raise RuntimeError('cannot rehierarch on columns when there is no hierarchy')

        if index:
            index_idx, index_iloc = rehierarch_from_index_hierarchy(
                    labels=self._index, # type: ignore
                    depth_map=index,
                    index_constructors=index_constructors,
                    name=self._index.name
                    )
        else:
            index_idx = self._index
            index_iloc = None

        if columns:
            columns_idx, columns_iloc = rehierarch_from_index_hierarchy(
                    labels=self._columns, # type: ignore
                    depth_map=columns,
                    index_constructors=columns_constructors,
                    name=self._columns.name
                    )
            own_columns = True
        else:
            columns_idx = self._columns
            own_columns = False # let constructor determine
            columns_iloc = None

        blocks = self._blocks._extract(index_iloc, columns_iloc)

        return self.__class__(
                blocks,
                index=index_idx,
                columns=columns_idx,
                name=self._name,
                own_data=True,
                own_index=True,
                own_columns=own_columns
                )



    #---------------------------------------------------------------------------
    # na handling

    def isna(self) -> tp.Self:
        '''
        Return a same-indexed, Boolean Frame indicating True which values are NaN or None.
        '''
        return self.__class__(self._blocks.isna(),
                index=self._index,
                columns=self._columns,
                own_index=True,
                own_columns=self.STATIC,
                own_data=True,
                )


    def notna(self) -> tp.Self:
        '''
        Return a same-indexed, Boolean Frame indicating True which values are not NaN or None.
        '''
        return self.__class__(self._blocks.notna(),
                index=self._index,
                columns=self._columns,
                own_index=True,
                own_columns=self.STATIC,
                own_data=True,
                )

    def dropna(self,
            axis: int = 0,
            condition: tp.Callable[[TNDArrayAny], TNDArrayAny] = np.all) -> tp.Self:
        '''
        Return a new :obj:`Frame` after removing rows (axis 0) or columns (axis 1) where any or all values are NA (NaN or None). The condition is determined by a NumPy ufunc that process the Boolean array returned by ``isna()``; the default is ``np.all``.

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
        if self.STATIC:
            if ((column_key is None and row_key.all()) or # type: ignore
                    (row_key is None and column_key.all())): # type: ignore
                return self
        return self._extract(row_key, column_key)


    #---------------------------------------------------------------------------
    # falsy handling

    def isfalsy(self) -> tp.Self:
        '''
        Return a same-indexed, Boolean Frame indicating True which values are falsy.
        '''
        # always return a Frame, even if this is a FrameGO
        return self.__class__(self._blocks.isfalsy(),
                index=self._index,
                columns=self._columns,
                own_index=True,
                own_columns=self.STATIC,
                own_data=True,
                )


    def notfalsy(self) -> tp.Self:
        '''
        Return a same-indexed, Boolean Frame indicating True which values are not falsy.
        '''
        # always return a Frame, even if this is a FrameGO
        return self.__class__(self._blocks.notfalsy(),
                index=self._index,
                columns=self._columns,
                own_index=True,
                own_columns=self.STATIC,
                own_data=True,
                )

    def dropfalsy(self,
            axis: int = 0,
            condition: tp.Callable[[TNDArrayAny], TNDArrayAny] = np.all) -> tp.Self:
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
            if ((column_key is None and row_key.all()) or # type: ignore
                    (row_key is None and column_key.all())): # type: ignore
                return self
        return self._extract(row_key, column_key)

    #---------------------------------------------------------------------------
    def _fill_missing(self,
            value: tp.Any,
            func: tp.Callable[[TNDArrayAny], TNDArrayAny],
            ) -> tp.Self:
        '''
        Args:
            func: function to return True for missing values
        '''
        kwargs = dict(
                index=self._index,
                columns=self._columns,
                name=self._name,
                own_index=True,
                own_columns=self.STATIC,
                own_data=True,
                )
        # NOTE: we branch based on value type to use more efficient TypeBlock methods when we know we have an element or a 2D array
        if isinstance(value, Frame):
            fill_value = dtype_to_fill_value(value._blocks._index.dtype)
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
            return self.__class__(
                    self._blocks.fill_missing_by_unit(fill, fill_valid, func=func),
                    **kwargs, # type: ignore
                    )
        elif is_fill_value_factory_initializer(value):
            # we have a iterable or a mapping, or FillValueAuto
            get_col_fill_value = get_col_fill_value_factory(value, columns=self._columns)
            return self.__class__(
                    self._blocks.fill_missing_by_callable(
                            func_missing=func,
                            get_col_fill_value=get_col_fill_value,
                            ),
                    **kwargs, # type: ignore
                    )
        # if not an iterable or if a string
        return self.__class__(
                self._blocks.fill_missing_by_unit(value, None, func=func),
                **kwargs, # type: ignore
                )



    @doc_inject(selector='fillna')
    def fillna(self, value: tp.Any) -> tp.Self:
        '''Return a new ``Frame`` after replacing null (NaN or None) values with the supplied value.

        Args:
            {value}
        '''
        return self._fill_missing(value, func=isna_array)

    @doc_inject(selector='fillna')
    def fillfalsy(self, value: tp.Any) -> tp.Self:
        '''Return a new ``Frame`` after replacing falsy values with the supplied value.

        Args:
            {value}
        '''
        return self._fill_missing(value, func=isfalsy_array)

    #---------------------------------------------------------------------------
    @doc_inject(selector='fillna')
    def fillna_leading(self,
            value: tp.Any,
            *,
            axis: int = 0) -> tp.Self:
        '''
        Return a new ``Frame`` after filling leading (and only leading) null (NaN or None) with the provided ``value``.

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
            axis: int = 0) -> tp.Self:
        '''
        Return a new ``Frame`` after filling trailing (and only trailing) null (NaN or None) with the provided ``value``.

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
    def fillfalsy_leading(self,
            value: tp.Any,
            *,
            axis: int = 0) -> tp.Self:
        '''
        Return a new ``Frame`` after filling leading (and only leading) falsy values with the provided ``value``.

        Args:
            {value}
            {axis}
        '''
        return self.__class__(self._blocks.fillfalsy_leading(value, axis=axis),
                index=self._index,
                columns=self._columns,
                name=self._name,
                own_data=True)

    @doc_inject(selector='fillna')
    def fillfalsy_trailing(self,
            value: tp.Any,
            *,
            axis: int = 0) -> tp.Self:
        '''
        Return a new ``Frame`` after filling trailing (and only trailing) falsy values with the provided ``value``.

        Args:
            {value}
            {axis}
        '''
        return self.__class__(self._blocks.fillfalsy_trailing(value, axis=axis),
                index=self._index,
                columns=self._columns,
                name=self._name,
                own_data=True)


    @doc_inject(selector='fillna')
    def fillna_forward(self,
            limit: int = 0,
            *,
            axis: int = 0) -> tp.Self:
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
            axis: int = 0) -> tp.Self:
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


    @doc_inject(selector='fillna')
    def fillfalsy_forward(self,
            limit: int = 0,
            *,
            axis: int = 0) -> tp.Self:
        '''
        Return a new ``Frame`` after filling forward falsy values with the last observed value.

        Args:
            {limit}
            {axis}
        '''
        return self.__class__(self._blocks.fillfalsy_forward(limit=limit, axis=axis),
                index=self._index,
                columns=self._columns,
                name=self._name,
                own_data=True)

    @doc_inject(selector='fillna')
    def fillfalsy_backward(self,
            limit: int = 0,
            *,
            axis: int = 0) -> tp.Self:
        '''
        Return a new ``Frame`` after filling backward falsy values with the first observed value.

        Args:
            {limit}
            {axis}
        '''
        return self.__class__(self._blocks.fillfalsy_backward(limit=limit, axis=axis),
                index=self._index,
                columns=self._columns,
                name=self._name,
                own_data=True)

    #---------------------------------------------------------------------------

    def __len__(self) -> int:
        '''Length of rows in values.
        '''
        return self._blocks._index.rows

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
        return Display.from_params(
                index=self._index,
                columns=self._columns,
                header=DisplayHeader(self.__class__, self._name),
                column_forward_iter=partial(self._blocks.axis_values, axis=0),
                column_reverse_iter=partial(self._blocks.axis_values, axis=0, reverse=True),
                column_default_iter=partial(self._blocks.axis_values, axis=0),
                config=config,
                style_config=style_config,
                )

    #---------------------------------------------------------------------------
    # accessors

    @property
    @doc_inject(selector='values_2d', class_name='Frame')
    def values(self) -> TNDArrayAny:
        '''
        {}
        '''
        return self._blocks.values

    @property
    def index(self) -> IndexBase:
        '''The ``IndexBase`` instance assigned for row labels.
        '''
        return self._index

    @property
    def columns(self) -> IndexBase:
        '''The ``IndexBase`` instance assigned for column labels.
        '''
        return self._columns

    #---------------------------------------------------------------------------
    # common attributes from the numpy array

    @property
    def dtypes(self) -> TSeriesAny:
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
    def mloc(self) -> TNDArrayAny:
        '''{doc_array}
        '''
        return self._blocks.mloc

    #---------------------------------------------------------------------------

    @property
    def shape(self) -> tp.Tuple[int, int]:
        '''
        Return a tuple describing the shape of the underlying NumPy array.

        Returns:
            :obj:`tp.Tuple[int, int]`
        '''
        return self._blocks._index.shape

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
            row_key: TILocSelector = None,
            column_key: TILocSelector = None,
            ) -> TNDArrayAny:
        '''
        Alternative extractor that returns just an ndarray. Keys are iloc keys.
        '''
        return self._blocks._extract_array(row_key, column_key)

    @staticmethod
    def _extract_axis_not_multi(
                row_key: tp.Any,
                column_key: tp.Any,
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

    @tp.overload
    def _extract(self, row_key: TILocSelectorOne) -> TSeriesAny: ...

    @tp.overload
    def _extract(self, row_key: TILocSelectorMany) -> tp.Self: ...

    @tp.overload
    def _extract(self, column_key: TILocSelectorOne) -> TSeriesAny: ...

    @tp.overload
    def _extract(self, column_key: TILocSelectorMany) -> tp.Self: ...

    @tp.overload
    def _extract(self, row_key: TILocSelectorMany, column_key: TILocSelectorOne) -> TSeriesAny: ...

    @tp.overload
    def _extract(self, row_key: TILocSelectorOne, column_key: TILocSelectorMany) -> TSeriesAny: ...

    @tp.overload
    def _extract(self, row_key: TILocSelectorMany, column_key: TILocSelectorMany) -> tp.Self: ...

    @tp.overload
    def _extract(self, row_key: TILocSelectorOne, column_key: TILocSelectorOne) -> tp.Any: ...

    @tp.overload
    def _extract(self, row_key: TILocSelector) -> tp.Any: ...

    def _extract(self, # pyright: ignore
            row_key: TILocSelector = None,
            column_key: TILocSelector = None,
            ) -> tp.Any:
        '''
        Extract Container based on iloc selection (indices have already mapped)
        '''
        blocks = self._blocks._extract(row_key=row_key, column_key=column_key)
        if blocks.__class__ is not TypeBlocks:
            return blocks

        index: IndexBase
        own_index = True # the extracted Frame can always own this index
        row_key_is_slice = row_key.__class__ is slice
        if row_key is None or (row_key_is_slice and row_key == NULL_SLICE):
            index = self._index
        elif not row_key_is_slice and isinstance(row_key, INT_TYPES):
            name_row = self._index._extract_iloc_by_int(row_key)
        else:
            index = self._index._extract_iloc(row_key) # type: ignore

        columns: IndexBase
        # can only own columns if _COLUMNS_CONSTRUCTOR is static
        column_key_is_slice = column_key.__class__ is slice
        if column_key is None or (column_key_is_slice and column_key == NULL_SLICE):
            columns = self._columns
            own_columns = self._COLUMNS_CONSTRUCTOR.STATIC
        elif not column_key_is_slice and isinstance(column_key, INT_TYPES):
            name_column = self._columns._extract_iloc_by_int(column_key)
        else:
            columns = self._columns._extract_iloc(column_key) # type: ignore
            own_columns = True

        # determine if an axis is not multi; if one axis is not multi, we return a Series instead of a Frame
        axis_nm = self._extract_axis_not_multi(row_key, column_key)
        blocks_shape = blocks.shape

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


    @tp.overload
    def _extract_iloc(self, key: TILocSelectorOne) -> TSeriesAny: ...

    @tp.overload
    def _extract_iloc(self, key: TILocSelectorMany) -> tp.Self: ...

    @tp.overload
    def _extract_iloc(self, key: tp.Tuple[TILocSelectorOne, TILocSelectorMany]) -> TSeriesAny: ...

    @tp.overload
    def _extract_iloc(self, key: tp.Tuple[TILocSelectorMany, TILocSelectorOne]) -> TSeriesAny: ...

    @tp.overload
    def _extract_iloc(self, key: tp.Tuple[TILocSelectorMany, TILocSelectorMany]) -> tp.Self: ...

    @tp.overload
    def _extract_iloc(self, key: tp.Tuple[TILocSelectorOne, TILocSelectorOne]) -> tp.Any: ...

    @tp.overload
    def _extract_iloc(self, key: TILocSelectorCompound) -> tp.Any: ...

    def _extract_iloc(self, key: TILocSelectorCompound) -> tp.Any: # pyright: ignore
        '''
        Give a compound key, return a new Frame. This method simply handles the variability of single or compound selectors.
        '''
        if isinstance(key, tuple):
            r, c = key
            return self._extract(r, c)
        return self._extract(key)

    def _compound_loc_to_iloc(self,
            key: TLocSelectorCompound,
            # ) -> TILocSelectorCompound:
            ) -> tp.Tuple[TILocSelector, TILocSelector]:
        '''
        Given a compound iloc key, return a tuple of row, column keys. Assumes the first argument is always a row extractor.
        '''
        if isinstance(key, tuple):
            loc_row_key, loc_column_key = key # pyright: ignore
            iloc_column_key = self._columns._loc_to_iloc(loc_column_key)
        else:
            loc_row_key = key
            iloc_column_key = None

        iloc_row_key = self._index._loc_to_iloc(loc_row_key)
        return iloc_row_key, iloc_column_key

    def _extract_loc(self, key: TLocSelectorCompound) -> tp.Any:
        r, c = self._compound_loc_to_iloc(key)
        return self._extract(r, c)

    def _extract_loc_columns(self, key: TLocSelector) -> TFrameOrSeries:
        '''Alternate extract of a columns only selection.
        '''
        return self._extract(None,
                self._columns._loc_to_iloc(key),
                )

    def _extract_bloc(self, key: TBlocKey) -> TSeriesAny:
        '''
        2D Boolean selector, selected by either a Boolean 2D Frame or array.
        '''
        bloc_key = bloc_key_normalize(key=key, container=self)
        coords, values = self._blocks.extract_bloc(bloc_key)
        index: Index[np.object_] = Index(
                ((self._index[x], self._columns[y]) for x, y in coords),
                dtype=DTYPE_OBJECT)
        return Series(values, index=index, own_index=True)

    def _compound_loc_to_getitem_iloc(self,
            key: TLocSelectorCompound) -> tp.Tuple[None, TILocSelector]:
        '''Handle a potentially compound key in the style of __getitem__. This will raise an appropriate exception if a two argument loc-style call is attempted.
        '''
        iloc_column_key = self._columns._loc_to_iloc(key)
        return None, iloc_column_key

    @tp.overload
    def __getitem__(self, key: TLabel) -> TSeriesAny: ...

    @tp.overload
    def __getitem__(self, key: tp.List[int]) -> tp.Self: ...

    @tp.overload
    def __getitem__(self, key: tp.List[str]) -> tp.Self: ...

    @tp.overload
    def __getitem__(self, key: TLocSelectorMany) -> tp.Self: ...

    @tp.overload
    def __getitem__(self, key: TLocSelector) -> tp.Self | TSeriesAny: ...

    @doc_inject(selector='selector')
    def __getitem__(self, key: TLocSelector) -> tp.Self | TSeriesAny: # pyright: ignore
        '''Selector of columns by label.

        Args:
            key: {key_loc}
        '''
        r, c = self._compound_loc_to_getitem_iloc(key)
        return self._extract(r, c)


    #---------------------------------------------------------------------------

    def _drop_iloc(self, key: TILocSelectorCompound) -> tp.Self:
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

    def _drop_loc(self, key: TLocSelectorCompound) -> tp.Self:
        key_iloc = self._compound_loc_to_iloc(key)
        return self._drop_iloc(key=key_iloc)

    def _drop_getitem(self, key: TLocSelectorCompound) -> tp.Self:
        key_iloc = self._compound_loc_to_getitem_iloc(key)
        return self._drop_iloc(key=key_iloc)


    #---------------------------------------------------------------------------
    def _extract_iloc_mask(self, key: TILocSelectorCompound) -> TFrameAny:
        masked_blocks = self._blocks.extract_iloc_mask(key)
        return self.__class__(masked_blocks,
                columns=self._columns,
                index=self._index,
                own_data=True)

    def _extract_loc_mask(self, key: TLocSelectorCompound) -> TFrameAny:
        key_iloc = self._compound_loc_to_iloc(key)
        return self._extract_iloc_mask(key=key_iloc)

    def _extract_getitem_mask(self, key: TLocSelectorCompound) -> TFrameAny:
        key_iloc = self._compound_loc_to_getitem_iloc(key)
        return self._extract_iloc_mask(key=key_iloc)

    #---------------------------------------------------------------------------
    def _extract_iloc_masked_array(self,
            key: TILocSelectorCompound,
            ) -> MaskedArray[tp.Any, tp.Any]:
        masked_blocks = self._blocks.extract_iloc_mask(key)
        return MaskedArray(data=self.values, mask=masked_blocks.values) # type: ignore

    def _extract_loc_masked_array(self, key: TLocSelectorCompound) -> MaskedArray[tp.Any, tp.Any]:
        key_iloc = self._compound_loc_to_iloc(key)
        return self._extract_iloc_masked_array(key=key_iloc)

    def _extract_getitem_masked_array(self, key: TLocSelectorCompound) -> MaskedArray[tp.Any, tp.Any]:
        key_iloc = self._compound_loc_to_getitem_iloc(key)
        return self._extract_iloc_masked_array(key=key_iloc)

    #---------------------------------------------------------------------------
    def _extract_iloc_assign(self, key: TILocSelectorCompound) -> 'FrameAssignILoc':
        return FrameAssignILoc(self, key=key)

    def _extract_loc_assign(self, key: TLocSelectorCompound) -> 'FrameAssignILoc':
        # extract if tuple, then pack back again
        key_iloc = self._compound_loc_to_iloc(key)
        return self._extract_iloc_assign(key=key_iloc)

    def _extract_getitem_assign(self, key: TLocSelectorCompound) -> 'FrameAssignILoc':
        # extract if tuple, then pack back again
        key_iloc = self._compound_loc_to_getitem_iloc(key)
        return self._extract_iloc_assign(key=key_iloc)

    def _extract_bloc_assign(self, key: TBlocKey) -> 'FrameAssignBLoc':
        '''Assignment based on a Boolean Frame or array.'''
        return FrameAssignBLoc(self, key=key)

    #---------------------------------------------------------------------------

    def _extract_getitem_astype(self, key: TLocSelector) -> 'FrameAsType':
        # extract if tuple, then pack back again
        _, key_iloc = self._compound_loc_to_getitem_iloc(key)
        return FrameAsType(self, column_key=key_iloc)

    def _extract_getitem_consolidate(self, key: TLocSelector) -> TFrameAny:
        _, key_iloc = self._compound_loc_to_getitem_iloc(key)
        blocks = TypeBlocks.from_blocks(
                self._blocks._consolidate_select_blocks(key_iloc))
        return self.__class__(blocks,
                index=self._index,
                columns=self._columns,
                own_index=True,
                own_data=True,
                )

    #---------------------------------------------------------------------------
    # dictionary-like interface

    def keys(self) -> tp.Iterable[TLabel]:
        '''Iterator of column labels.
        '''
        return self._columns

    def __iter__(self) -> tp.Iterable[TLabel]:
        '''
        Iterator of column labels, same as :py:meth:`Frame.keys`.
        '''
        return self._columns.__iter__()

    def __contains__(self, value: TLabel) -> bool:
        '''
        Inclusion of value in column labels.
        '''
        return self._columns.__contains__(value)

    def items(self) -> tp.Iterator[tp.Tuple[TLabel, TSeriesAny]]:
        '''Iterator of pairs of column label and corresponding column :obj:`Series`.
        '''
        for label, array in zip(self._columns.values, self._blocks.iter_columns_arrays()):
            # array is assumed to be immutable
            yield label, Series(array, index=self._index, name=label)

    def get(self,
            key: TLabel,
            default: tp.Optional[TSeriesAny] = None,
            ) -> TSeriesAny:
        '''
        Return the value found at the columns key, else the default if the key is not found. This method is implemented to complete the dictionary-like interface.
        '''
        if key not in self._columns:
            return default # type: ignore
        return self.__getitem__(key)


    #---------------------------------------------------------------------------
    # operator functions

    def _ufunc_unary_operator(self,
            operator: tp.Callable[[TNDArrayAny], TNDArrayAny],
            ) -> TFrameAny:
        # call the unary operator on _blocks
        return self.__class__(
                self._blocks._ufunc_unary_operator(operator=operator),
                index=self._index,
                columns=self._columns,
                name=self._name,
                )

    def _ufunc_binary_operator(self, *,
            operator: TUFunc,
            other: tp.Any,
            axis: int = 0,
            fill_value: tp.Any = np.nan,
            ) -> TFrameAny:

        if operator.__name__ == 'matmul':
            return matmul(self, other) # type: ignore
        elif operator.__name__ == 'rmatmul':
            return matmul(other, self) # type: ignore

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
                # if self is a FrameGO, columns will be a GO, and we can own columns
                self_tb = self.reindex(
                        columns=columns,
                        own_columns=True,
                        fill_value=fill_value,
                        )._blocks
                # we can only own this index if other is immutable
                other_array = other.reindex(
                        columns,
                        own_index=self.STATIC,
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
                        own_columns=self.STATIC,
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
                # NOTE: axis always internally supplied
                raise AxisInvalid(f'invalid axis: {axis}') #pragma: no cover

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
            ufunc: TUFunc,
            ufunc_skipna: TUFunc,
            composable: bool,
            dtypes: tp.Tuple[TDtypeAny, ...],
            size_one_unity: bool
            ) -> TSeriesAny:
        # axis 0 processes ros, deliveres column index
        # axis 1 processes cols, delivers row index
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
            ufunc: TUFunc,
            ufunc_skipna: TUFunc,
            composable: bool,
            dtypes: tp.Tuple[TDtypeAny, ...],
            size_one_unity: bool
            ) -> TFrameAny:
        # axis 0 processes ros, deliveres column index
        # axis 1 processes cols, delivers row index
        dtype = None if not dtypes else dtypes[0] # only a tuple
        if skipna:
            post = ufunc_skipna(self.values, axis=axis, dtype=dtype)
        else:
            post = ufunc(self.values, axis=axis, dtype=dtype)
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

    def _axis_array(self, axis: int) -> tp.Iterator[TNDArrayAny]:
        '''Generator of arrays across an axis
        '''
        yield from self._blocks.axis_values(axis)

    def _axis_array_items(self,
            axis: int,
            ) -> tp.Iterator[tp.Tuple[TLabel, TNDArrayAny]]:
        keys = self._index if axis == 1 else self._columns
        yield from zip(keys, self._blocks.axis_values(axis))

    def _axis_tuple(self, *,
            axis: int,
            constructor: tp.Optional[TTupleCtor] = None,
            ) -> tp.Iterator[tp.Sequence[tp.Any]]:
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
            ctor = get_tuple_constructor(labels)
        elif isinstance(constructor, type):
            if (issubclass(constructor, tuple) and
                    hasattr(constructor, '_make')):
                # discover named tuples, use _make method for single-value calling
                ctor = constructor._make # pyright: ignore
            elif is_dataclass(constructor):
                # this will fail if kw_only is true in python 3.10
                ctor = lambda args: constructor(*args) # type: ignore
            else: # assume it can take a single arguments
                ctor = constructor
        else:
            ctor = constructor

        # NOTE: if all types are the same, it will be faster to use axis_values
        if axis == 1 and not self._blocks.unified_dtypes:
            yield from self._blocks.iter_row_tuples(key=None, constructor=ctor) # pyright: ignore
        else: # for columns, slicing arrays from blocks should be cheap
            for axis_values in self._blocks.axis_values(axis):
                yield ctor(axis_values) # pyright: ignore

    def _axis_tuple_items(self, *,
            axis: int,
            constructor: tp.Optional[TTupleCtor] = None,
            ) -> tp.Iterator[tp.Tuple[TLabel, tp.Sequence[tp.Any]]]:
        keys = self._index if axis == 1 else self._columns
        yield from zip(keys, self._axis_tuple(axis=axis, constructor=constructor))


    def _axis_series(self, axis: int) -> tp.Iterator[TSeriesAny]:
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
            # NOTE: axis_values here are already immutable
            yield Series(axis_values, index=index, name=label, own_index=True)

    def _axis_series_items(self, axis: int) -> tp.Iterator[tp.Tuple[TLabel, TSeriesAny]]:
        keys = self._index if axis == 1 else self._columns
        yield from zip(keys, self._axis_series(axis=axis))


    #---------------------------------------------------------------------------
    # grouping methods

    def _axis_group_final_iter(self, *,
            axis: int,
            as_array: bool,
            group_iter: tp.Iterator[tp.Tuple[TLabel, slice | TNDArrayAny, TypeBlocks | TNDArrayAny]],
            index: IndexBase,
            columns: IndexBase,
            ordering: tp.Optional[TNDArrayAny],
            ) -> tp.Iterator[tp.Tuple[TLabel, TFrameAny | TNDArrayAny]]:
        '''Utility for final iteration of the group_iter, shared by three methods.
        '''
        if as_array:
            yield from ((group, array) for group, _, array in group_iter) # pyright: ignore
        else:
            for group, selection, tb in group_iter:
                # NOTE: selection can be a Boolean array or a slice
                if axis == 0:
                    # axis 0 is a row iter, so need to slice index, keep columns
                    index_group = (index._extract_iloc(selection) if ordering is None
                            else index._extract_iloc(ordering[selection])
                            )
                    yield group, self.__class__(tb,
                            index=index_group,
                            columns=columns,
                            own_columns=self.STATIC, # own if static
                            own_index=True,
                            own_data=True)
                else:
                    # axis 1 is a column iterators, so need to slice columns, keep index
                    columns_group = (columns._extract_iloc(selection) if ordering is None
                            else columns._extract_iloc(ordering[selection])
                            )
                    yield group, self.__class__(tb,
                            index=index,
                            columns=columns_group,
                            own_index=True,
                            own_columns=True,
                            own_data=True)


    def _axis_group_iloc_items(self,
            key: TILocSelector,
            *,
            axis: int,
            drop: bool = False,
            stable: bool = True,
            as_array: bool = False,
            ) -> tp.Iterator[tp.Tuple[TLabel, TFrameAny | TNDArrayAny]]:
        '''
        Core group implementation.

        Args:
            as_array: if True, return arrays instead of ``Frame``
        '''
        blocks = self._blocks

        if drop:
            shape = blocks._index.columns if axis == 0 else blocks._index.rows
            drop_mask = np.full(shape, True, dtype=DTYPE_BOOL)
            drop_mask[key] = False

        # NOTE: in limited studies using stable does not show significant overhead
        kind: TSortKinds = DEFAULT_STABLE_SORT_KIND if stable else DEFAULT_FAST_SORT_KIND
        try:
            blocks, ordering = blocks.sort(key=key, axis=not axis, kind=kind)
            use_sorted = True
        except TypeError:
            use_sorted = False
            ordering = None

        columns: IndexBase
        index: IndexBase

        group_iter: tp.Iterator[tp.Tuple[TLabel, slice | TNDArrayAny, tp.Union[TypeBlocks, TNDArrayAny]]]
        if use_sorted:
            group_iter = group_sorted(
                    blocks=blocks,
                    axis=axis,
                    key=key,
                    drop=drop,
                    as_array=as_array,
                    )

        else:
            group_iter = group_match(
                    blocks=blocks,
                    axis=axis,
                    key=key,
                    drop=drop,
                    as_array=as_array,
                    )

        if axis == 0:
            index = self._index
            columns = self._columns if not drop else self._columns[drop_mask]
        else:
            index = self._index if not drop else self._index[drop_mask]
            columns = self._columns

        yield from self._axis_group_final_iter(
                axis=axis,
                as_array=as_array,
                group_iter=group_iter,
                index=index,
                columns=columns,
                ordering=ordering,
                )

    def _axis_group_loc_items(self,
            key: TLocSelector,
            *,
            axis: int = 0,
            drop: bool = False,
            stable: bool = True,
            as_array: bool = False,
            ) -> tp.Iterator[tp.Tuple[TLabel, TFrameAny | TNDArrayAny]]:
        '''
        Args:
            key: We accept any thing that can do loc to iloc. Note that a tuple is permitted as key, where it would be interpreted as a single label for an IndexHierarchy.
            axis:
            drop: exclude the target of the group in the returned results.
        '''
        if axis == 0: # row iterator, selecting columns for group by
            iloc_key = self._columns._loc_to_iloc(key)
        elif axis == 1: # column iterator, selecting rows for group by
            iloc_key = self._index._loc_to_iloc(key)
        else:
            raise AxisInvalid(f'invalid axis: {axis}')
        yield from self._axis_group_iloc_items(key=iloc_key,
                axis=axis,
                drop=drop,
                stable=stable,
                as_array=as_array,
                )

    def _axis_group_loc(self,
            key: TLocSelector,
            *,
            axis: int = 0,
            drop: bool = False,
            as_array: bool = False,
            ) -> tp.Iterator[TFrameAny | TNDArrayAny]:
        yield from (x for _, x in self._axis_group_loc_items(
                key=key,
                axis=axis,
                drop=drop,
                as_array=as_array,
                ))

    #-----------------------------------------------------------------------
    def _axis_group_labels_items(self,
            depth_level: TDepthLevel = 0,
            *,
            axis: int = 0,
            as_array: bool = False,
            ) -> tp.Iterator[tp.Tuple[TLabel, TFrameAny | TNDArrayAny]]:
        # NOTE: simlar to _axis_group_iloc_items

        blocks = self._blocks
        index = self._index
        columns = self._columns

        if axis == 0: # maintain columns, group by index
            ref_index = index
        elif axis == 1: # maintain index, group by columns
            ref_index = columns
        else:
            raise AxisInvalid(f'invalid axis: {axis}')

        if isinstance(depth_level, INT_TYPES):
            labels = [ref_index.values_at_depth(depth_level)]
        else:
            labels = [ref_index.values_at_depth(i) for i in depth_level]

        ordering = None
        try:
            if len(labels) > 1:
                ordering = np.lexsort(list(reversed(labels)))
            else:
                ordering = np.argsort(labels[0], kind=DEFAULT_STABLE_SORT_KIND)
            use_sorted = True
        except TypeError:
            use_sorted = False

        if len(labels) > 1:
            # NOTE: this will do an h-strack style concatenation; this is ultimately what is needed in group_source
            group_source = blocks_to_array_2d(labels)
            if use_sorted:
                group_source = group_source[ordering]
        else:
            # group_source = column_2d_filter(labels[0])
            group_source = labels[0]
            if use_sorted:
                group_source = group_source[ordering]

        group_iter: tp.Iterator[tp.Tuple[TLabel, slice | TNDArrayAny, TypeBlocks | TNDArrayAny]]
        if use_sorted:
            if axis == 0:
                blocks = self._blocks._extract(row_key=ordering)
            else:
                blocks = self._blocks._extract(column_key=ordering)
            group_iter = group_sorted(
                    blocks=blocks,
                    axis=axis,
                    key=None, # assume this is not used
                    drop=False,
                    as_array=as_array,
                    group_source=group_source,
                    )
        else:
            group_iter = group_match(
                    blocks=blocks,
                    axis=axis,
                    key=None,
                    drop=False,
                    as_array=as_array,
                    group_source=group_source,
                    )

        yield from self._axis_group_final_iter(
                axis=axis,
                as_array=as_array,
                group_iter=group_iter,
                index=index,
                columns=columns,
                ordering=ordering,
                )


    def _axis_group_labels(self,
            depth_level: TDepthLevel = 0,
            *,
            axis: int = 0,
            as_array: bool = False,
            ) -> tp.Iterator[TFrameAny | TNDArrayAny]:
        yield from (x for _, x in self._axis_group_labels_items(
                depth_level=depth_level,
                axis=axis,
                as_array=as_array,
                ))

    #-----------------------------------------------------------------------
    def _axis_group_other_items(self,
            *,
            axis: int = 0,
            as_array: bool = False,
            group_source: TNDArrayAny,
            ) -> tp.Iterator[tp.Tuple[TLabel, TFrameAny | TNDArrayAny]]:

        blocks = self._blocks
        index = self._index
        columns = self._columns

        group_source_ndim = group_source.ndim
        ordering = None
        if group_source_ndim > 1:
            # normalize group_source for lex sorting
            group_source_cols = [group_source[NULL_SLICE, i]
                    for i in range(group_source.shape[1])]
        try:
            if group_source_ndim > 1:
                ordering = np.lexsort(list(reversed(group_source_cols)))
            else:
                ordering = np.argsort(group_source, kind=DEFAULT_STABLE_SORT_KIND)
            use_sorted = True
        except TypeError:
            use_sorted = False

        if use_sorted:
            group_source = group_source[ordering]

        group_iter: tp.Iterator[tp.Tuple[TLabel, slice | TNDArrayAny, TypeBlocks | TNDArrayAny]]
        if use_sorted:
            if axis == 0:
                blocks = self._blocks._extract(row_key=ordering)
            else:
                blocks = self._blocks._extract(column_key=ordering)

            group_iter = group_sorted(
                    blocks=blocks,
                    axis=axis,
                    key=None, # assume this is not used
                    drop=False,
                    as_array=as_array,
                    group_source=group_source,
                    )
        else:
            group_iter = group_match(
                    blocks=blocks,
                    axis=axis,
                    key=None,
                    drop=False,
                    as_array=as_array,
                    group_source=group_source,
                    )

        yield from self._axis_group_final_iter(
                axis=axis,
                as_array=as_array,
                group_iter=group_iter,
                index=index,
                columns=columns,
                ordering=ordering,
                )

    def _axis_group_other(self,
            *,
            axis: int = 0,
            as_array: bool = False,
            group_source: TNDArrayAny,
            ) -> tp.Iterator[TFrameAny | TNDArrayAny]:
        yield from (x for _, x in self._axis_group_other_items(
                axis=axis,
                as_array=as_array,
                group_source=group_source,
                ))

    #---------------------------------------------------------------------------
    def _axis_window_items(self, *,
            size: int,
            axis: int = 0,
            step: int = 1,
            window_sized: bool = True,
            window_func: tp.Optional[TCallableAny] = None,
            window_valid: tp.Optional[TCallableAny] = None,
            label_shift: int = 0,
            label_missing_skips: bool = True,
            label_missing_raises: bool = False,
            start_shift: int = 0,
            size_increment: int = 0,
            as_array: bool = False,
            ) -> tp.Iterator[tp.Tuple[TLabel, tp.Any]]:
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
                label_missing_skips=label_missing_skips,
                label_missing_raises=label_missing_raises,
                start_shift=start_shift,
                size_increment=size_increment,
                as_array=as_array,
                derive_label=True,
                )


    def _axis_window(self, *,
            size: int,
            axis: int = 0,
            step: int = 1,
            window_sized: bool = True,
            window_func: tp.Optional[TCallableAny] = None,
            window_valid: tp.Optional[TCallableAny] = None,
            label_shift: int = 0,
            label_missing_skips: bool = True,
            label_missing_raises: bool = False,
            start_shift: int = 0,
            size_increment: int = 0,
            as_array: bool = False,
            ) -> tp.Iterator[TFrameAny]:
        yield from (x for _, x in axis_window_items(
                source=self,
                size=size,
                axis=axis,
                step=step,
                window_sized=window_sized,
                window_func=window_func,
                window_valid=window_valid,
                label_shift=label_shift,
                label_missing_skips=label_missing_skips,
                label_missing_raises=label_missing_raises,
                start_shift=start_shift,
                size_increment=size_increment,
                as_array=as_array,
                derive_label=False,
                ))


    #---------------------------------------------------------------------------

    def _iter_element_iloc_items(self,
            axis: int = 0,
            ) -> tp.Iterator[tp.Tuple[tp.Tuple[int, ...], tp.Any]]:
        yield from self._blocks.element_items(axis=axis)

    # def _iter_element_iloc(self):
    #     yield from (x for _, x in self._iter_element_iloc_items())

    def _iter_element_loc_items(self,
            axis: int = 0,
            ) -> tp.Iterator[tp.Tuple[tp.Tuple[TLabel, TLabel], tp.Any]]:
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

    def __reversed__(self) -> tp.Iterator[TLabel]:
        '''
        Returns a reverse iterator on the frame's columns.
        '''
        return reversed(self._columns)

    @doc_inject(selector='sort')
    def sort_index(self,
            *,
            ascending: TBoolOrBools = True,
            kind: TSortKinds = DEFAULT_SORT_KIND,
            key: tp.Optional[tp.Callable[[IndexBase], tp.Union[TNDArrayAny, IndexBase]]] = None,
            ) -> TFrameAny:
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
            ascending: TBoolOrBools = True,
            kind: TSortKinds = DEFAULT_SORT_KIND,
            key: tp.Optional[tp.Callable[[IndexBase], tp.Union[TNDArrayAny, IndexBase]]] = None,
            ) -> TFrameAny:
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
            label: TKeyOrKeys, # elsewhere this is called 'key'
            *,
            ascending: TBoolOrBools = True,
            axis: int = 1,
            kind: TSortKinds = DEFAULT_SORT_KIND,
            key: tp.Optional[tp.Callable[[tp.Union[TFrameAny, TSeriesAny]], tp.Union[TNDArrayAny, TSeriesAny, TFrameAny]]] = None,
            ) -> tp.Self:
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
        values_for_sort: TNDArrayAny | tp.List[TNDArrayAny] | None = None
        values_for_lex: TOptionalArrayList = None
        cfs: TNDArrayAny | TSeriesAny | TFrameAny | TypeBlocks

        if axis == 0: # get a column ordering based on one or more rows
            iloc_key = self._index._loc_to_iloc(label) # type: ignore
            if key:
                cfs = key(self._extract(row_key=iloc_key))
                cfs_is_array = cfs.__class__ is np.ndarray
                if (cfs.ndim == 1 and len(cfs) != self.shape[1]) or (cfs.ndim == 2 and cfs.shape[1] != self.shape[1]): # pyright: ignore
                    raise RuntimeError('key function returned a container of invalid length')
            else: # go straight to array as, since this is row-wise, have to find a consolidated
                cfs = self._blocks._extract_array(row_key=iloc_key)
                cfs_is_array = True

            if cfs_is_array:
                if cfs.ndim == 1:
                    values_for_sort = cfs # type: ignore
                elif cfs.ndim == 2 and cfs.shape[0] == 1:
                    values_for_sort = cfs[0] # type: ignore
                else:
                    values_for_lex = [cfs[i] for i in range(cfs.shape[0]-1, -1, -1)] # pyright: ignore
            elif cfs.ndim == 1: # Series
                values_for_sort = cfs.values # type: ignore
            elif isinstance(cfs, Frame):
                cfs = cfs._blocks
                if cfs.shape[0] == 1:
                    values_for_sort = cfs._extract_array(row_key=0)
                else:
                    values_for_lex = [cfs._extract_array(row_key=i)
                            for i in range(cfs.shape[0]-1, -1, -1)]

        elif axis == 1: # get a row ordering based on one or more columns
            iloc_key = self._columns._loc_to_iloc(label) # type: ignore
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
                    values_for_sort = cfs # type: ignore
                elif cfs.ndim == 2 and cfs.shape[1] == 1: # pyright: ignore
                    values_for_sort = cfs[:, 0] # type: ignore
                else:
                    values_for_lex = [cfs[:, i] for i in range(cfs.shape[1]-1, -1, -1)] #type: ignore
            elif cfs.ndim == 1: # Series
                values_for_sort = cfs.values # type: ignore
            else: #Frame/TypeBlocks from here
                if isinstance(cfs, Frame):
                    cfs = cfs._blocks
                if cfs.shape[1] == 1: # pyright: ignore
                    values_for_sort = cfs._extract_array_column(0) # type: ignore
                else:
                    values_for_lex = [cfs._extract_array_column(i) # type: ignore
                            for i in range(cfs.shape[1]-1, -1, -1)] # pyright: ignore
        else:
            raise AxisInvalid(f'invalid axis: {axis}')

        asc_is_element, values_for_lex = prepare_values_for_lex( # type: ignore
                ascending=ascending,
                values_for_lex=values_for_lex,
                )

        if values_for_lex is not None:
            order = np.lexsort(values_for_lex)
        elif values_for_sort is not None:
            order = np.argsort(values_for_sort, kind=kind)

        if asc_is_element and not ascending:
            # NOTE: if asc is not an element, then ascending Booleans have already been applied to values_for_lex
            # NOTE: putting the order in reverse, not invetering the selection, produces the descending sort
            order = order[::-1]

        if axis == 0:
            columns = self._columns[order]
            blocks = self._blocks._extract(column_key=order) # order columns
            return self.__class__(blocks,
                    index=self._index,
                    columns=columns,
                    name=self._name,
                    own_data=True,
                    own_columns=True,
                    own_index=True,
                    )

        index = self._index[order]
        blocks = self._blocks._extract(row_key=order)
        return self.__class__(blocks,
                index=index,
                columns=self._columns,
                name=self._name,
                own_data=True,
                own_index=True
                )

    def isin(self, other: tp.Any) -> TFrameAny:
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
            lower: tp.Optional[tp.Union[float, TSeriesAny, TFrameAny]] = None,
            upper: tp.Optional[tp.Union[float, TSeriesAny, TFrameAny]] = None,
            axis: tp.Optional[int] = None
            ) -> TFrameAny:
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

        args: tp.List[float | TNDArrayAny | ContainerOperand | None] = [lower, upper]
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
                    args[idx] = [values] * self.shape[1] # type: ignore
                else:
                    # create a list of row-length arrays for maximal type preservation
                    args[idx] = [np.full(self.shape[0], v) for v in values] # type: ignore

            elif isinstance(arg, Frame):
                args[idx] = arg.reindex( # type: ignore
                        index=self._index,
                        columns=self._columns).fillna(bound)._blocks._blocks

            elif hasattr(arg, '__iter__'):
                raise RuntimeError('only Series or Frame are supported as iterable lower/upper arguments')
            # assume single value otherwise, no change necessary

        blocks = self._blocks.clip(*args) # type: ignore

        return self.__class__(blocks,
                columns=self._columns,
                index=self._index,
                name=self._name,
                own_data=True,
                own_index=True,
                )

    def transpose(self) -> TFrameAny:
        '''Transpose. Return a :obj:`Frame` with ``index`` as ``columns`` and vice versa.
        '''
        return self.__class__(self._blocks.transpose(),
                index=self._columns,
                columns=self._index,
                own_data=True,
                own_index=self.STATIC,
                own_columns=self.STATIC,
                name=self._name)

    @property
    def T(self) -> TFrameAny:
        '''Transpose. Return a :obj:`Frame` with ``index`` as ``columns`` and vice versa.
        '''
        return self.transpose()

    @doc_inject(selector='duplicated')
    def duplicated(self, *,
            axis: int = 0,
            exclude_first: bool = False,
            exclude_last: bool = False) -> TSeriesAny:
        '''
        Return an axis-sized Boolean :obj:`Series` that shows True for all rows (axis 0) or columns (axis 1) duplicated.

        Args:
            {axis}
            {exclude_first}
            {exclude_last}
        '''
        # might be able to do this witnout calling .values and passing in TypeBlocks, but TB needs to support roll
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
            ) -> TFrameAny:
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
            column: TLabel,
            *,
            drop: bool = False,
            index_constructor: TIndexCtorSpecifier = None,
            ) -> TFrameAny:
        '''
        Return a new :obj:`Frame` produced by setting the given column as the index, optionally removing that column from the new :obj:`Frame`.

        Args:
            column:
            *
            drop:
            index_constructor:
        '''
        column_iloc = self._columns._loc_to_iloc(column)
        if column_iloc is None: # if None was a key it would have an iloc
            return self if self.STATIC else self.__class__(self)

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

        index_values: tp.Iterable[TLabel]
        if isinstance(column_iloc, INT_TYPES):
            index_values = self._blocks._extract_array_column(column_iloc)
            name = column
        else:
            index_values = self._blocks.iter_row_tuples(column_iloc)
            name = tuple(self._columns[column_iloc])

        index = index_from_optional_constructor(index_values,
                default_constructor=Index,
                explicit_constructor=index_constructor,
                )
        if index.name is None:
            # NOTE: if a constructor has not set a name, we set the name as expected
            index = index.rename(name)

        return self.__class__(blocks,
                columns=columns,
                index=index,
                own_data=own_data,
                own_columns=own_columns,
                own_index=True,
                name=self._name,
                )

    def set_index_hierarchy(self,
            columns: TLocSelector,
            *,
            drop: bool = False,
            index_constructors: TIndexCtorSpecifiers = None,
            reorder_for_hierarchy: bool = False,
            ) -> TFrameAny:
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
        column_loc: TLocSelector
        if isinstance(columns, tuple):
            # NOTE: this prohibits selecting a single tuple label, which might be fine given context
            column_loc = list(columns)
            name = columns
        else:
            column_loc = columns
            name = None # could be a slice, must get post iloc conversion

        column_iloc = self._columns._loc_to_iloc(column_loc)

        if name is None:
            # NOTE: is this the best approach if columns is IndexHierarchy?
            name = tuple(self._columns[column_iloc])

        index_labels = self._blocks._extract(column_key=column_iloc)

        if reorder_for_hierarchy:
            rehierarched_blocks, order_lex = rehierarch_from_type_blocks(
                    labels=index_labels,
                    depth_map=range(index_labels.shape[1]), # keep order
                    )
            index = IndexHierarchy._from_type_blocks(
                    blocks=rehierarched_blocks,
                    index_constructors=index_constructors,
                    name=name,
                    own_blocks=True,
                    name_interleave=True,
                    )
            blocks_src = self._blocks._extract(row_key=order_lex)
        else:
            index = IndexHierarchy._from_type_blocks(
                    index_labels,
                    index_constructors=index_constructors,
                    name=name,
                    own_blocks=True,
                    name_interleave=True,
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
            names: tp.Sequence[TLabel] = (),
            drop: bool = False,
            consolidate_blocks: bool = False,
            columns_constructors: TIndexCtorSpecifiers = None,
            ) -> TFrameAny:
        '''
        Return a new :obj:`Frame` where the index is added to the front of the data, and an :obj:`IndexAutoFactory` is used to populate a new index. If the :obj:`Index` has a ``name``, that name will be used for the column name, otherwise a suitable default will be used. As underlying NumPy arrays are immutable, data is not copied.

        Args:
            names: An iterable of hashables to be used to name the unset index. If an ``Index``, a single hashable should be provided; if an ``IndexHierarchy``, as many hashables as the depth must be provided.
            consolidate_blocks:
            columns_constructors:
        '''
        # disallows specifying names with 'drop=True'
        if drop is True and names:
            raise RuntimeError("Cannot specify `names` when `drop=True`, as the index will not be added back as columns.")

        def blocks() -> tp.Iterator[TNDArrayAny]:
            # yield index as columns, then remaining blocks currently in Frame
            if not drop:
                if self._index.ndim == 1:
                    yield self._index.values
                else:
                    # No recache is needed as it's not possible for an index to be GO
                    yield from self._index._blocks._blocks # type: ignore
            yield from self._blocks._blocks

        block_gen: tp.Callable[[], tp.Iterator[TNDArrayAny]]
        if consolidate_blocks:
            block_gen = lambda: TypeBlocks.consolidate_blocks(blocks())
        else:
            block_gen = blocks

        columns: None | IndexBase
        if drop:
            # When dropping the index, keep the existing columns without adding index names
            columns, own_columns = self._columns, self.STATIC
        else:
            if not names:
                names = self._index.names

            columns_depth = self._columns.depth
            index_depth = self._index.depth

            if len(names) != index_depth:
                raise RuntimeError('Passed `names` must have a label (or sequence of labels) per depth of index.')

            if columns_depth > 1:
                if isinstance(names[0], str) or not hasattr(names[0], '__len__'):
                    raise RuntimeError(f'Invalid name labels ({names[0]!r}); provide a sequence with a label per columns depth.')

                if index_depth == 1:
                    # assume that names[0] is an iterable of labels per columns depth level (one column of labels)
                    columns_labels = TypeBlocks.from_blocks( # type: ignore
                            concat_resolved((np.array([name]), self._columns.values_at_depth(i)))
                            for i, name in enumerate(names[0]) #type: ignore
                            )
                else:
                    # assume that names is an iterable of columns, each column with a label per columns depth
                    labels_per_depth = []
                    for labels in zip(*names):
                        a, _ = iterable_to_array_1d(labels)
                        labels_per_depth.append(a)

                    # assert len(labels_per_depth) == columns_depth
                    columns_labels = TypeBlocks.from_blocks(
                            concat_resolved((labels, self._columns.values_at_depth(i)))
                            for i, labels in enumerate(labels_per_depth)
                            )

                columns_default_constructor: TIndexHierarchyCtor = partial(
                        self._COLUMNS_HIERARCHY_CONSTRUCTOR._from_type_blocks,
                        own_blocks=True)
            else:
                # columns depth is 1, label per index depth is correct
                columns_labels = chain(names, self._columns.values) # type: ignore
                columns_default_constructor = self._COLUMNS_CONSTRUCTOR # type: ignore

            columns, own_columns = index_from_optional_constructors(
                    columns_labels, # pyright: ignore
                    depth=columns_depth,
                    default_constructor=columns_default_constructor,
                    explicit_constructors=columns_constructors, # cannot supply name
                    )

        return self.__class__(
                TypeBlocks.from_blocks(block_gen()),
                columns=columns,
                own_columns=own_columns,
                index=None,
                own_data=True,
                name=self._name,
                )

    def set_columns(self,
            index: TLabel,
            *,
            drop: bool = False,
            columns_constructor: TIndexCtorSpecifier = None,
            ) -> TFrameAny:
        '''
        Return a new :obj:`Frame` produced by setting the given row as the columns, optionally removing that row from the new :obj:`Frame`.

        Args:
            index:
            *
            drop:
            columns_constructor:
        '''
        index_iloc = self._index._loc_to_iloc(index)
        if index_iloc is None or (index_iloc.__class__ is np.ndarray and len(index_iloc) == 0): # type: ignore
            # if None was a key it would have an iloc
            return self if self.STATIC else self.__class__(self)

        if drop:
            blocks = TypeBlocks.from_blocks(
                    self._blocks._drop_blocks(row_key=index_iloc))
            index_final = self._index._drop_iloc(index_iloc)
            own_data = True
        else:
            blocks = self._blocks
            index_final = self._index
            own_data = False

        if isinstance(index_iloc, INT_TYPES):
            columns_values = self._blocks.iter_row_elements(index_iloc)
            name = index
        else:
            # given a multiple row selection, yield a tuple accross rows (column values) as tuples; this acvoids going through arrays
            columns_values = self._blocks.iter_columns_tuples(index_iloc)
            name = tuple(self._index[index_iloc])

        columns = index_from_optional_constructor(columns_values,
                default_constructor=self._COLUMNS_CONSTRUCTOR,
                explicit_constructor=columns_constructor,
                )
        if columns.name is None:
            # NOTE: if a constructor has not set a name, we set the name as expected
            columns = columns.rename(name)

        return self.__class__(blocks,
                columns=columns,
                index=index_final,
                own_data=own_data,
                own_columns=True,
                own_index=True,
                name=self._name,
                )

    def set_columns_hierarchy(self,
            index: TLocSelector,
            *,
            drop: bool = False,
            columns_constructors: TIndexCtorSpecifiers = None,
            reorder_for_hierarchy: bool = False,
            ) -> TFrameAny:
        '''
        Given an iterable of index labels, return a new ``Frame`` with those rows as an ``IndexHierarchy`` on the columns.

        Args:
            index: Iterable of index labels.
            drop: Boolean to determine if selected rows should be removed from the data.
            columns_constructors: Optionally provide a sequence of ``Index`` constructors, of length equal to depth, to be used in converting row Index components in the ``IndexHierarchy``.
            reorder_for_hierarchy: reorder the columns to produce a hierarchible Index from the selected columns.

        Returns:
            :obj:`Frame`
        '''
        index_loc: TLocSelector
        if isinstance(index, tuple):
            # NOTE: this prohibits selecting a single tuple label, which might be fine given context
            index_loc = list(index)
            name = index
        else:
            index_loc = index
            name = None # could be a slice, must get post iloc conversion

        index_iloc = self._index._loc_to_iloc(index_loc)

        if name is None:
            # NOTE: is this the best approach if index is IndexHierarchy?
            name = tuple(self._index[index_iloc])

        # NOTE: must transpose so that blocks are organized by what was each row
        columns_labels = self._blocks._extract(row_key=index_iloc).transpose()

        if reorder_for_hierarchy:
            rehierarched_blocks, order_lex = rehierarch_from_type_blocks(
                    labels=columns_labels,
                    depth_map=range(columns_labels.shape[1]), # keep order
                    )
            columns = self._COLUMNS_HIERARCHY_CONSTRUCTOR._from_type_blocks(
                    blocks=rehierarched_blocks,
                    index_constructors=columns_constructors,
                    name=name,
                    own_blocks=True,
                    name_interleave=True,
                    )
            blocks_src = self._blocks._extract(column_key=order_lex)
        else:
            columns = self._COLUMNS_HIERARCHY_CONSTRUCTOR._from_type_blocks(
                    columns_labels,
                    index_constructors=columns_constructors,
                    name=name,
                    own_blocks=True,
                    name_interleave=True,
                    )
            blocks_src = self._blocks

        if drop:
            blocks = TypeBlocks.from_blocks(
                    blocks_src._drop_blocks(row_key=index_iloc))
            index = self._index._drop_iloc(index_iloc)
            own_data = True
            own_index = True
        else:
            blocks = blocks_src
            index = self._index
            own_data = False
            own_index = False

        return self.__class__(blocks,
                columns=columns,
                index=index,
                own_data=own_data,
                own_columns=True,
                own_index=own_index,
                name=self._name
                )

    def unset_columns(self, *,
            names: tp.Sequence[TLabel] = (),
            drop: bool = False,
            index_constructors: TIndexCtorSpecifiers = None,
            ) -> TFrameAny:
        '''
        Return a new :obj:`Frame` where columns are added to the top of the data, and an :obj:`IndexAutoFactory` is used to populate new columns. This operation potentially forces a complete copy of all data.

        Args:
            names: An sequence of hashables to be used to name the unset columns. If an ``Index``, a single hashable should be provided; if an ``IndexHierarchy``, as many hashables as the depth must be provided.
            index_constructors:
        '''
        if drop is True and names:
            raise RuntimeError("The `names` parameter cannot be used with `drop=True` because the column labels will not be included in the resulting Frame.")

        if not names and drop is False:
            names = self._columns.names

        index: None | IndexBase
        if drop is True:
            index = self._index
            own_index = True
            blocks = self._blocks.copy() # permit owning
        else:
            # columns blocks are oriented as "rows" here, and might have different types per row; when moved on to the frame, types will have to be consolidated "vertically", meaning there is little chance of consolidation. A maximal decomposition might give a chance, but each ultimate column would have to be re-evaluated, and that would be expense.
            blocks = TypeBlocks.from_blocks(
                TypeBlocks.vstack_blocks_to_blocks((
                        TypeBlocks.from_blocks(self.columns.values).transpose(),
                        self._blocks
                        ))
                )
            columns_depth = self._columns.depth
            index_depth = self._index.depth

            if len(names) != columns_depth:
                raise RuntimeError('Passed `names` must have a label (or sequence of labels) per depth of columns.')

            index_default_constructor: TIndexCtorSpecifier

            if index_depth > 1:
                if isinstance(names[0], str) or not hasattr(names[0], '__len__'):
                    raise RuntimeError(f'Invalid name labels ({names[0]!r}); provide a sequence with a label per index depth.')

                if columns_depth == 1:
                    # assume that names[0] is an iterable of labels per index depth level (one row of labels)
                    index_labels = TypeBlocks.from_blocks( # type: ignore
                            concat_resolved((np.array([name]), self._index.values_at_depth(i)))
                            for i, name in enumerate(names[0]) # type: ignore
                            )
                else:
                    # assume that names is an iterable of rows, each row with a label per index depth
                    labels_per_depth = []
                    for labels in zip(*names):
                        a, _ = iterable_to_array_1d(labels)
                        labels_per_depth.append(a)

                    # assert len(labels_per_depth) == index_depth
                    index_labels = TypeBlocks.from_blocks(
                            concat_resolved((labels, self._index.values_at_depth(i)))
                            for i, labels in enumerate(labels_per_depth)
                            )

                index_default_constructor = partial(
                        IndexHierarchy._from_type_blocks,
                        own_blocks=True)
            else:
                # index depth is 1, label per columns depth is correct
                index_labels = chain(names, self._index.values) # type: ignore
                index_default_constructor = Index

            index, own_index = index_from_optional_constructors(
                    index_labels, # pyright: ignore
                    depth=index_depth,
                    default_constructor=index_default_constructor,
                    explicit_constructors=index_constructors, # cannot supply name
                    )
        return self.__class__(
                blocks,
                columns=None,
                own_index=own_index,
                index=index,
                own_data=True,
                name=self._name,
                )

    def __round__(self, decimals: int = 0) -> TFrameAny:
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
            include_columns: bool = False) -> TFrameAny:
        '''
        Roll columns and/or rows by positive or negative integer counts, where columns and/or rows roll around the axis.

        Args:
            include_index: Determine if index is included in index-wise rotation.
            include_columns: Determine if column index is included in index-wise rotation.
        '''
        shift_index = index
        shift_column = columns

        blocks = TypeBlocks.from_blocks(
                self._blocks._shift_blocks_fill_by_element(
                row_shift=shift_index,
                column_shift=shift_column,
                wrap=True
                ))

        if include_index:
            index_idx = self._index.roll(shift_index)
            own_index = True
        else:
            index_idx = self._index
            own_index = False

        if include_columns:
            columns_idx = self._columns.roll(shift_column)
            own_columns = True
        else:
            columns_idx = self._columns
            own_columns = False

        return self.__class__(blocks,
                columns=columns_idx,
                index=index_idx,
                name=self._name,
                own_data=True,
                own_columns=own_columns,
                own_index=own_index,
                )

    def shift(self,
            index: int = 0,
            columns: int = 0,
            *,
            fill_value: tp.Any = np.nan,
            ) -> TFrameAny:
        '''
        Shift columns and/or rows by positive or negative integer counts, where columns and/or rows fall of the axis and introduce missing values, filled by `fill_value`.
        '''

        shift_index = index
        shift_column = columns

        if is_fill_value_factory_initializer(fill_value):
            get_col_fill_value = get_col_fill_value_factory(
                    fill_value,
                    tuple(self._columns), # tuple better for IH
                    )
            blocks = TypeBlocks.from_blocks(
                    self._blocks._shift_blocks_fill_by_callable(
                    row_shift=shift_index,
                    column_shift=shift_column,
                    wrap=False,
                    get_col_fill_value=get_col_fill_value
                    ))
        else:
            blocks = TypeBlocks.from_blocks(
                    self._blocks._shift_blocks_fill_by_element(
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
            ascending: TBoolOrBools = True,
            start: int = 0,
            fill_value: tp.Any = np.nan,
    ) -> TFrameAny:

        if axis == 1 and is_fill_value_factory_initializer(fill_value):
            raise InvalidFillValue(fill_value, 'axis==1')

        shape = self._blocks.shape
        asc_is_element = isinstance(ascending, BOOL_TYPES)

        if not asc_is_element:
            ascending = tuple(ascending) # type: ignore
            opposite_axis = int(not axis)
            if len(ascending) != shape[opposite_axis]:
                raise RuntimeError(f'Multiple ascending values must match length of axis {opposite_axis}.')

        if axis == 0:
            fill_value_factory = get_col_fill_value_factory(fill_value, self.columns)

        def array_iter() -> tp.Iterator[TNDArrayAny]:
            for idx, array in enumerate(self._blocks.axis_values(axis=axis)):
                asc = ascending if asc_is_element else ascending[idx] # type: ignore
                if not skipna or array.dtype.kind not in DTYPE_NA_KINDS:
                    yield rank_1d(array,
                            method=method,
                            ascending=asc, # pyright: ignore
                            start=start,
                            )
                else:
                    index = self._index if axis == 0 else self._columns
                    s: TSeriesAny = Series(array, index=index, own_index=True)
                    # if iterating rows, all arrays will have the same dtype
                    fv = fill_value if axis == 1 else fill_value_factory(idx, array.dtype)
                    yield s._rank(method=method,
                            skipna=skipna,
                            ascending=asc, # pyright: ignore
                            start=start,
                            fill_value=fv,
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
            ascending: TBoolOrBools = True,
            start: int = 0,
            fill_value: tp.Any = np.nan,
            ) -> TFrameAny:
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
            ascending: TBoolOrBools = True,
            start: int = 0,
            fill_value: tp.Any = np.nan,
            ) -> TFrameAny:
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
            ascending: TBoolOrBools = True,
            start: int = 0,
            fill_value: tp.Any = np.nan,
            ) -> TFrameAny:
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
            ascending: TBoolOrBools = True,
            start: int = 0,
            fill_value: tp.Any = np.nan,
            ) -> TFrameAny:
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
            ascending: TBoolOrBools = True,
            start: int = 0,
            fill_value: tp.Any = np.nan,
            ) -> TFrameAny:
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
    def head(self, count: int = 5) -> TFrameAny:
        '''{doc}

        Args:
            {count}
        '''
        return self.iloc[:count]

    @doc_inject(selector='tail', class_name='Frame')
    def tail(self, count: int = 5) -> TFrameAny:
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
            ) -> TSeriesAny:
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
                    self._blocks.shape[axis],
                    dtype=DTYPE_INT_DEFAULT,
                    )
        else:
            array = np.empty(len(labels), dtype=DTYPE_INT_DEFAULT)
            for i, values in enumerate(self._blocks.axis_values(axis=axis)):
                valid: tp.Optional[TNDArrayAny] = None

                if skipfalsy: # always includes skipna
                    valid = ~isfalsy_array(values)
                elif skipna: # NOTE: elif, as skipfalsy incldues skipna
                    valid = ~isna_array(values)

                if unique and valid is None:
                    array[i] = len(ufunc_unique1d(values))
                elif unique and valid is not None: # valid is a Boolean array
                    array[i] = len(ufunc_unique1d(values[valid]))
                elif not unique and valid is not None:
                    array[i] = valid.sum()
                else: # not unique, valid is None, means no removals, handled above
                    raise NotImplementedError() #pragma: no cover

        array.flags.writeable = False
        return Series(array, index=labels)

    @doc_inject(selector='sample')
    def sample(self,
            index: tp.Optional[int] = None,
            columns: tp.Optional[int] = None,
            *,
            seed: tp.Optional[int] = None,
            ) -> TFrameAny:
        '''
        {doc}

        Args:
            {index}
            {columns}
            {seed}
        '''
        if index is not None:
            index_idx, index_key = self._index._sample_and_key(count=index, seed=seed)
            own_index = True
        else:
            index_idx = self._index
            index_key = None
            own_index = True

        if columns is not None:
            columns_idx, columns_key = self._columns._sample_and_key(count=columns, seed=seed)
            own_columns = True
        else:
            columns_idx = self._columns
            columns_key = None
            own_columns = False # might be GO

        if index_key is not None or columns_key is not None:
            blocks = self._blocks._extract(row_key=index_key, column_key=columns_key)
        else:
            blocks = self._blocks.copy()

        return self.__class__(blocks,
                columns=columns_idx,
                index=index_idx,
                name=self._name,
                own_data=True,
                own_index=own_index,
                own_columns=own_columns,
                )

    #---------------------------------------------------------------------------

    def _container_from_index_values(
            self,
            values: np.ndarray[tp.Any, tp.Any],
            axis: int,
            ) -> TSeriesAny:

        if values.ndim == 2:
            values = array_to_tuple_array(values)
            values.flags.writeable = False

        if axis == 0:
            return Series(
                    values,
                    index=immutable_index_filter(self._columns),
                    name=self.index.name,
                    )

        return Series(
                values,
                index=immutable_index_filter(self._index),
                name=self.columns.name,
                )

    @doc_inject(selector='argminmax')
    def loc_min(self, *,
            skipna: bool = True,
            axis: int = 0
            ) -> TSeriesAny:
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
            values = self.index.values[post]
        else:
            values = self.columns.values[post]

        return self._container_from_index_values(values, axis)

    @doc_inject(selector='argminmax')
    def iloc_min(self, *,
            skipna: bool = True,
            axis: int = 0
            ) -> TSeriesAny:
        '''
        Return the integer indices corresponding to the minimum values found.

        Args:
            {skipna}
            {axis}
        '''
        # if this has NaN can continue
        values = argmin_2d(self.values, skipna=skipna, axis=axis)
        values.flags.writeable = False

        return self._container_from_index_values(values, axis)

    @doc_inject(selector='argminmax')
    def loc_max(self, *,
            skipna: bool = True,
            axis: int = 0
            ) -> TSeriesAny:
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
            values = self.index.values[post]
        else:
            values = self.columns.values[post]

        return self._container_from_index_values(values, axis)

    @doc_inject(selector='argminmax')
    def iloc_max(self, *,
            skipna: bool = True,
            axis: int = 0
            ) -> TSeriesAny:
        '''
        Return the integer indices corresponding to the maximum values found.

        Args:
            {skipna}
            {axis}
        '''
        # if this has NaN can continue
        values = argmax_2d(self.values, skipna=skipna, axis=axis)
        values.flags.writeable = False

        return self._container_from_index_values(values, axis)


    #---------------------------------------------------------------------------
    def _label_not_missing(self,
            *,
            axis: int,
            return_label: bool,
            forward: bool,
            fill_value: TLabel = np.nan,
            func: tp.Callable[[TNDArrayAny], TNDArrayAny],
            ) -> TSeriesAny:
        '''
        For a given axis, return the first or last label observed that is not missing.
        Args:
            return_label: If True, return the label, else the iloc position.
            func: Array processor such as `isna_array`.
        '''

        def blocks() -> tp.Iterator[TNDArrayAny]:
            for b in self._blocks._blocks:
                # func returns True for missing, invert for not missing
                bool_block = ~func(b)
                bool_block.flags.writeable = False
                yield bool_block

        target = blocks_to_array_2d(blocks(),
                shape=self.shape,
                dtype=DTYPE_BOOL,
                )

        if axis == 0:
            labels_returned = self._columns
            labels_opposite = self._index
        else:
            labels_returned = self._index
            labels_opposite = self._columns

        pos = first_true_2d(target, axis=axis, forward=forward)
        fill_target = pos == -1 # test to expected missing
        fill_all = fill_target.all()

        if fill_all:
            array = np.full(shape=len(labels_returned), fill_value=fill_value)
        elif return_label:
            # do an expanding selection as labels might be found in multiple positions, if missing values (-1) are found, they select the last value
            if labels_opposite._NDIM == 1:
                array = labels_opposite.values[pos]
            else:
                array = labels_opposite.flat().values[pos] # type: ignore
        else:
            array = pos # will contain -1 in fill positions

        if not fill_all and fill_target.any():
            if return_label or fill_value != -1:
                if labels_opposite._NDIM == 1:
                    labels_dtype = labels_opposite.dtype # type: ignore
                else: # get resolved row dtype from IH
                    labels_dtype = labels_opposite._blocks._index.dtype # type: ignore
                dtype = resolve_dtype(labels_dtype, dtype_from_element(fill_value))
                if dtype != array.dtype:
                    array = array.astype(dtype)
                array[fill_target] = fill_value

        array.flags.writeable = False
        index = immutable_index_filter(labels_returned)
        return Series(array, index=index, own_index=True)


    def iloc_notna_first(self, *,
            fill_value: int = -1,
            axis: int = 0
            ) -> TSeriesAny:
        '''
        Return the position corresponding to the first non-missing values along the selected axis.

        Args:
            {skipna}
            {axis}
        '''
        return self._label_not_missing(
                axis=axis,
                forward=True,
                return_label=False,
                fill_value=fill_value,
                func=isna_array,
                )

    def iloc_notna_last(self, *,
            fill_value: int = -1,
            axis: int = 0
            ) -> TSeriesAny:
        '''
        Return the position corresponding to the last non-missing values along the selected axis.

        Args:
            {skipna}
            {axis}
        '''
        return self._label_not_missing(
                axis=axis,
                forward=False,
                return_label=False,
                fill_value=fill_value,
                func=isna_array,
                )

    def loc_notna_first(self, *,
            fill_value: TLabel = np.nan,
            axis: int = 0
            ) -> TSeriesAny:
        '''
        Return the labels corresponding to the first non-missing values along the selected axis.

        Args:
            {skipna}
            {axis}
        '''
        return self._label_not_missing(
                axis=axis,
                forward=True,
                return_label=True,
                fill_value=fill_value,
                func=isna_array,
                )

    def loc_notna_last(self, *,
            fill_value: TLabel = np.nan,
            axis: int = 0
            ) -> TSeriesAny:
        '''
        Return the labels corresponding to the last non-missing values along the selected axis.

        Args:
            {skipna}
            {axis}
        '''
        return self._label_not_missing(
                axis=axis,
                forward=False,
                return_label=True,
                fill_value=fill_value,
                func=isna_array,
                )

    #---------------------------------------------------------------------------
    def iloc_notfalsy_first(self, *,
            fill_value: int = -1,
            axis: int = 0
            ) -> TSeriesAny:
        '''
        Return the position corresponding to the first non-falsy (including nan) values along the selected axis.

        Args:
            {skipna}
            {axis}
        '''
        return self._label_not_missing(
                axis=axis,
                forward=True,
                return_label=False,
                fill_value=fill_value,
                func=isfalsy_array,
                )

    def iloc_notfalsy_last(self, *,
            fill_value: int = -1,
            axis: int = 0
            ) -> TSeriesAny:
        '''
        Return the position corresponding to the last non-falsy (including nan) values along the selected axis.

        Args:
            {skipna}
            {axis}
        '''
        return self._label_not_missing(
                axis=axis,
                forward=False,
                return_label=False,
                fill_value=fill_value,
                func=isfalsy_array,
                )

    def loc_notfalsy_first(self, *,
            fill_value: TLabel = np.nan,
            axis: int = 0
            ) -> TSeriesAny:
        '''
        Return the labels corresponding to the first non-falsy (including nan) values along the selected axis.

        Args:
            {skipna}
            {axis}
        '''
        return self._label_not_missing(
                axis=axis,
                forward=True,
                return_label=True,
                fill_value=fill_value,
                func=isfalsy_array,
                )

    def loc_notfalsy_last(self, *,
            fill_value: TLabel = np.nan,
            axis: int = 0
            ) -> TSeriesAny:
        '''
        Return the labels corresponding to the last non-falsy (including nan) values along the selected axis.

        Args:
            {skipna}
            {axis}
        '''
        return self._label_not_missing(
                axis=axis,
                forward=False,
                return_label=True,
                fill_value=fill_value,
                func=isfalsy_array,
                )

    #---------------------------------------------------------------------------

    def cov(self,
            *,
            axis: int = 1,
            ddof: int = 1,
            ) -> tp.Self:
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

    def corr(self,
            *,
            axis: int = 1,
            ) -> tp.Self:
        '''Compute a correlation matrix.

        Args:
            axis: if 0, each row represents a variable, with observations as columns; if 1, each column represents a variable, with observations as rows. Defaults to 1.
        '''
        if axis == 0:
            rowvar = True
            labels = self._index
            own_index = True
            own_columns = self.STATIC
        else:
            rowvar = False
            labels = self._columns
            own_index = self.STATIC
            own_columns = self.STATIC

        values = np.corrcoef(self.values, rowvar=rowvar)
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
            index_fields: TKeyOrKeys,
            columns_fields: TKeyOrKeys = (),
            data_fields: TKeyOrKeys = (),
            *,
            func: tp.Optional[TCallableOrCallableMap] = np.nansum,
            fill_value: tp.Any = np.nan,
            index_constructor: TIndexCtorSpecifier = None,
            ) -> TFrameAny:
        '''
        Produce a pivot table, where one or more columns is selected for each of index_fields, columns_fields, and data_fields. Unique values from the provided ``index_fields`` will be used to create a new index; unique values from the provided ``columns_fields`` will be used to create a new columns; if one ``data_fields`` value is selected, that is the value that will be displayed; if more than one values is given, those values will be presented with a hierarchical index on the columns; if ``data_fields`` is not provided, all unused fields will be displayed.

        Args:
            index_fields
            columns_fields
            data_fields
            *
            fill_value: If the index expansion produces coordinates that have no existing data value, fill that position with this value.
            func: function to apply to ``data_fields``, or a dictionary of labelled functions to apply to data fields, producing an additional hierarchical level.
            index_constructor:
        '''
        if is_fill_value_factory_initializer(fill_value):
            raise InvalidFillValue(fill_value, 'pivot')

        # NOTE: default in Pandas pivot_table is a mean
        if func is None:
            func_map = ()
            func_single = None
            func_fields = ()
        elif callable(func):
            func_map = (('', func),) # type: ignore  # store iterable of pairs
            func_single = func
            func_fields = ()
        else: # assume func has an items method
            func_map = tuple(func.items()) # type: ignore
            func_single = func_map[0][1] if len(func_map) == 1 else None # type: ignore
            func_fields = () if func_single else tuple(label for label, _ in func_map) # type: ignore

        # normalize all keys to lists of values
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

        #-----------------------------------------------------------------------
        # have final fields and normalized function representation
        all_fields = list(chain(index_fields, columns_fields, data_fields))
        frame: TFrameAny
        if len(all_fields) < len(self.columns):
            frame = self._extract_loc_columns(all_fields) # type: ignore
        else:
            frame = self
        from static_frame.core.pivot import pivot_core
        return pivot_core(frame=frame,
                index_fields=index_fields,
                columns_fields=columns_fields,
                data_fields=data_fields,
                func_fields=func_fields,
                func_single=func_single,
                func_map=func_map,
                fill_value=fill_value,
                index_constructor=index_constructor,
                )

    #---------------------------------------------------------------------------
    # pivot stack, unstack

    def pivot_stack(self,
            depth_level: TDepthLevel = -1,
            *,
            fill_value: object = np.nan,
            ) -> tp.Self:
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

        if is_fill_value_factory_initializer(fill_value):
            raise InvalidFillValue(fill_value, 'pivot_stack')

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
        def records_items() -> tp.Iterator[tp.Tuple[TLabel, tp.Sequence[tp.Any]]]:
            for row_idx, outer in enumerate(index_src): # iter tuple or label
                for target in targets_unique: # target is always a tuple
                    # derive the new index
                    if index_src.depth == 1:
                        key = (outer,) + target # type: ignore
                    else:
                        key = outer + target # type: ignore
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
        def dtypes() -> tp.Iterator[TDtypeAny]:
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
            depth_level: TDepthLevel = -1,
            *,
            fill_value: object = np.nan,
            ) -> tp.Self:
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

        if is_fill_value_factory_initializer(fill_value):
            raise InvalidFillValue(fill_value, 'pivot_unstack')

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

        def items() -> tp.Iterator[tp.Tuple[TLabel, TNDArrayAny]]:
            for col_idx, outer in enumerate(columns_src): # iter tuple or label
                dtype_src_col = dtypes_src[col_idx]

                for target in targets_unique: # target is always a tuple
                    if columns_src.depth == 1:
                        key = (outer,) + target # type: ignore
                    else:
                        key = outer + target # type: ignore
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
    @doc_inject(selector='join')
    def merge_inner(self,
            other: TFrameAny,
            *,
            left_depth_level: tp.Optional[TDepthLevel] = None,
            left_columns: TLocSelector = None,
            right_depth_level: tp.Optional[TDepthLevel] = None,
            right_columns: TLocSelector = None,
            merge_labels: tp.Sequence[TLabel] | None = None,
            left_template: str = '{}',
            right_template: str = '{}',
            fill_value: tp.Any = np.nan,
            include_index: bool = False,
            ) -> TFrameAny:
        '''
        Perform an inner merge, an inner join where matched columns are coalesced.

        Args:
            {left_depth_level}
            {left_columns}
            {right_depth_level}
            {right_columns}
            {merge_labels}
            {left_template}
            {right_template}
            {fill_value}
            {include_index}
        Returns:
            :obj:`Frame`
        '''
        return join(frame=self,
                other=other,
                join_type=Join.INNER,
                left_depth_level=left_depth_level,
                left_columns=left_columns,
                right_depth_level=right_depth_level,
                right_columns=right_columns,
                left_template=left_template,
                right_template=right_template,
                fill_value=fill_value,
                include_index=include_index,
                merge_labels=merge_labels,
                merge=True,
                )

    @doc_inject(selector='join')
    def merge_left(self,
            other: TFrameAny,
            *,
            left_depth_level: tp.Optional[TDepthLevel] = None,
            left_columns: TLocSelector = None,
            right_depth_level: tp.Optional[TDepthLevel] = None,
            right_columns: TLocSelector = None,
            merge_labels: tp.Sequence[TLabel] | None = None,
            left_template: str = '{}',
            right_template: str = '{}',
            fill_value: tp.Any = np.nan,
            include_index: bool = False,
            ) -> TFrameAny:
        '''
        Perform a left merge, a left join where matched columns are coalesced.

        Args:
            {left_depth_level}
            {left_columns}
            {right_depth_level}
            {right_columns}
            {merge_labels}
            {left_template}
            {right_template}
            {fill_value}
            {include_index}
        Returns:
            :obj:`Frame`
        '''
        return join(frame=self,
                other=other,
                join_type=Join.LEFT,
                left_depth_level=left_depth_level,
                left_columns=left_columns,
                right_depth_level=right_depth_level,
                right_columns=right_columns,
                left_template=left_template,
                right_template=right_template,
                fill_value=fill_value,
                include_index=include_index,
                merge_labels=merge_labels,
                merge=True,
                )

    @doc_inject(selector='join')
    def merge_right(self,
            other: TFrameAny,
            *,
            left_depth_level: tp.Optional[TDepthLevel] = None,
            left_columns: TLocSelector = None,
            right_depth_level: tp.Optional[TDepthLevel] = None,
            right_columns: TLocSelector = None,
            merge_labels: tp.Sequence[TLabel] | None = None,
            left_template: str = '{}',
            right_template: str = '{}',
            fill_value: tp.Any = np.nan,
            include_index: bool = False,
            ) -> TFrameAny:
        '''
        Perform a right merge, a right join where matched columns are coalesced.

        Args:
            {left_depth_level}
            {left_columns}
            {right_depth_level}
            {right_columns}
            {merge_labels}
            {left_template}
            {right_template}
            {fill_value}
            {include_index}
        Returns:
            :obj:`Frame`
        '''
        return join(frame=self,
                other=other,
                join_type=Join.RIGHT,
                left_depth_level=left_depth_level,
                left_columns=left_columns,
                right_depth_level=right_depth_level,
                right_columns=right_columns,
                left_template=left_template,
                right_template=right_template,
                fill_value=fill_value,
                include_index=include_index,
                merge_labels=merge_labels,
                merge=True,
                )

    @doc_inject(selector='join')
    def merge_outer(self,
            other: TFrameAny,
            *,
            left_depth_level: tp.Optional[TDepthLevel] = None,
            left_columns: TLocSelector = None,
            right_depth_level: tp.Optional[TDepthLevel] = None,
            right_columns: TLocSelector = None,
            merge_labels: tp.Sequence[TLabel] | None = None,
            left_template: str = '{}',
            right_template: str = '{}',
            fill_value: tp.Any = np.nan,
            include_index: bool = False,
            ) -> TFrameAny:
        '''
        Perform an outer merge, an outer join where matched columns are coalesced.

        Args:
            {left_depth_level}
            {left_columns}
            {right_depth_level}
            {right_columns}
            {merge_labels}
            {left_template}
            {right_template}
            {fill_value}
            {include_index}
        Returns:
            :obj:`Frame`
        '''
        return join(frame=self,
                other=other,
                join_type=Join.OUTER,
                left_depth_level=left_depth_level,
                left_columns=left_columns,
                right_depth_level=right_depth_level,
                right_columns=right_columns,
                left_template=left_template,
                right_template=right_template,
                fill_value=fill_value,
                include_index=include_index,
                merge_labels=merge_labels,
                merge=True,
                )

    @doc_inject(selector='join')
    def join_inner(self,
            other: TFrameAny, # support a named Series as a 1D frame?
            *,
            left_depth_level: tp.Optional[TDepthLevel] = None,
            left_columns: TLocSelector = None,
            right_depth_level: tp.Optional[TDepthLevel] = None,
            right_columns: TLocSelector = None,
            left_template: str = '{}',
            right_template: str = '{}',
            fill_value: tp.Any = np.nan,
            include_index: bool = False,
            # composite_index: bool = True,
            # composite_index_fill_value: TLabel = None,
            ) -> TFrameAny:
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
            {include_index}

        Returns:
            :obj:`Frame`
        '''
        return join(frame=self,
                other=other,
                join_type=Join.INNER,
                left_depth_level=left_depth_level,
                left_columns=left_columns,
                right_depth_level=right_depth_level,
                right_columns=right_columns,
                left_template=left_template,
                right_template=right_template,
                fill_value=fill_value,
                include_index=include_index,
                # composite_index=composite_index,
                # composite_index_fill_value=composite_index_fill_value,
                )

    @doc_inject(selector='join')
    def join_left(self,
            other: TFrameAny, # support a named Series as a 1D frame?
            *,
            left_depth_level: tp.Optional[TDepthLevel] = None,
            left_columns: TLocSelector = None,
            right_depth_level: tp.Optional[TDepthLevel] = None,
            right_columns: TLocSelector = None,
            left_template: str = '{}',
            right_template: str = '{}',
            fill_value: tp.Any = np.nan,
            include_index: bool = False,
            # composite_index: bool = True,
            # composite_index_fill_value: TLabel = None,
            ) -> TFrameAny:
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
            {include_index}

        Returns:
            :obj:`Frame`
        '''
        return join(frame=self,
                other=other,
                join_type=Join.LEFT,
                left_depth_level=left_depth_level,
                left_columns=left_columns,
                right_depth_level=right_depth_level,
                right_columns=right_columns,
                left_template=left_template,
                right_template=right_template,
                fill_value=fill_value,
                include_index=include_index,
                # composite_index=composite_index,
                # composite_index_fill_value=composite_index_fill_value,
                )

    @doc_inject(selector='join')
    def join_right(self,
            other: TFrameAny, # support a named Series as a 1D frame?
            *,
            left_depth_level: tp.Optional[TDepthLevel] = None,
            left_columns: TLocSelector = None,
            right_depth_level: tp.Optional[TDepthLevel] = None,
            right_columns: TLocSelector = None,
            left_template: str = '{}',
            right_template: str = '{}',
            fill_value: tp.Any = np.nan,
            include_index: bool = False,
            # composite_index: bool = True,
            # composite_index_fill_value: TLabel = None,
            ) -> TFrameAny:
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
            {include_index}

        Returns:
            :obj:`Frame`
        '''
        return join(frame=self,
                other=other,
                join_type=Join.RIGHT,
                left_depth_level=left_depth_level,
                left_columns=left_columns,
                right_depth_level=right_depth_level,
                right_columns=right_columns,
                left_template=left_template,
                right_template=right_template,
                fill_value=fill_value,
                include_index=include_index,
                # composite_index=composite_index,
                # composite_index_fill_value=composite_index_fill_value,
                )

    @doc_inject(selector='join')
    def join_outer(self,
            other: TFrameAny, # support a named Series as a 1D frame?
            *,
            left_depth_level: tp.Optional[TDepthLevel] = None,
            left_columns: TLocSelector = None,
            right_depth_level: tp.Optional[TDepthLevel] = None,
            right_columns: TLocSelector = None,
            left_template: str = '{}',
            right_template: str = '{}',
            fill_value: tp.Any = np.nan,
            include_index: bool = False,
            # composite_index: bool = True,
            # composite_index_fill_value: TLabel = None,
            ) -> TFrameAny:
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
            {include_index}

        Returns:
            :obj:`Frame`
        '''
        return join(frame=self,
                other=other,
                join_type=Join.OUTER,
                left_depth_level=left_depth_level,
                left_columns=left_columns,
                right_depth_level=right_depth_level,
                right_columns=right_columns,
                left_template=left_template,
                right_template=right_template,
                fill_value=fill_value,
                include_index=include_index,
                # composite_index=composite_index,
                # composite_index_fill_value=composite_index_fill_value,
                )

    #---------------------------------------------------------------------------
    def _insert(self,
            key: int | np.integer[tp.Any], # iloc positions
            container: tp.Union[TFrameAny, TSeriesAny],
            *,
            after: bool,
            fill_value: tp.Any = np.nan,
            ) -> tp.Self:
        '''
        Return a new Frame with the provided container inserted at the position (or after the position) determined by the column key.

        NOTE: At this time we do not accept elements or unlabelled iterables, as our interface does not permit supplying the required new column names with those arguments.
        '''
        if not isinstance(container, (Series, Frame)):
            raise NotImplementedError(
                    f'No support for inserting with {type(container)}')

        if container.ndim == 2 and not len(container.columns): # type: ignore
            return self if self.STATIC else self.__class__(self)

        # this filter is needed to handle possible invalid ILoc values passed through
        key = iloc_to_insertion_iloc(key, self.shape[1]) + after

        # self's index will never change; we only take what aligns in the passed container
        if not self._index.equals(container._index):
            container = container.reindex(self._index,
                    fill_value=fill_value,
                    check_equals=False,
                    )

        # NOTE: might introduce coercions in IndexHierarchy
        labels_prior = self._columns.values
        labels_insert: tp.Iterable[TLabel]

        if isinstance(container, Frame):
            labels_insert = container.columns.__iter__()
            blocks_insert = container._blocks._blocks

        elif isinstance(container, Series):
            labels_insert = (container.name,)
            blocks_insert = (container.values,) # type: ignore

        columns = self._columns.__class__.from_labels(chain(
                labels_prior[:key],
                labels_insert, # type: ignore
                labels_prior[key:],
                ))

        blocks = TypeBlocks.from_blocks(chain(
                        self._blocks._slice_blocks(None,
                                slice(0, key),
                                False,
                                True),
                        blocks_insert,
                        self._blocks._slice_blocks(None,
                                slice(key, None),
                                False,
                                True),
                        ),
                own_data=True,
                )

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
            key: TLabel,
            container: tp.Union[TFrameAny, TSeriesAny],
            *,
            fill_value: tp.Any = np.nan,
            ) -> tp.Self:
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
            raise RuntimeError(f'Unsupported key type: {key!r}')
        return self._insert(iloc_key, container, after=False, fill_value=fill_value)

    @doc_inject(selector='insert')
    def insert_after(self,
            key: TLabel,
            container: tp.Union[TFrameAny, TSeriesAny],
            *,
            fill_value: tp.Any = np.nan,
            ) -> tp.Self:
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
            raise RuntimeError(f'Unsupported key type: {key!r}')
        return self._insert(iloc_key, container, after=True, fill_value=fill_value)

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
            ) -> TNDArrayAny:
        '''
        Return a NumPy array of unqiue values. If the axis argument is provided, uniqueness is determined by columns or row.
        '''
        return ufunc_unique(self.values, axis=axis)

    def unique_enumerated(self, *,
            retain_order: bool = False,
            func: tp.Optional[tp.Callable[[tp.Any], bool]] = None,
            ) -> tp.Tuple[TNDArrayAny, TNDArrayAny]:
        '''
        {doc}
        {args}
        '''
        return ufunc_unique_enumerated(self.values,
                retain_order=retain_order,
                func=func,
                )


    #---------------------------------------------------------------------------
    # exporters

    def to_pairs(self, axis: int = 0) -> tp.Iterable[
            tp.Tuple[TLabel, tp.Iterable[tp.Tuple[TLabel, tp.Any]]]]:
        '''
        Return a tuple of major axis key, minor axis key vlaue pairs, where major axis is determined by the axis argument. Note that the returned object is eagerly constructed; use an iterator interface for lazy iteration.
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


    def _to_signature_bytes(self,
            include_name: bool = True,
            include_class: bool = True,
            encoding: str = 'utf-8',
            ) -> bytes:

        v = []
        for a in self._blocks._blocks:
            if a.dtype == DTYPE_OBJECT:
                raise TypeError('Object dtypes do not have stable hashes')
            # NOTE: use Fortran ordering to ensure uniform result regardless of block consolidation
            v.append(a.tobytes('F'))

        return b''.join(chain(
                iter_component_signature_bytes(self,
                        include_name=include_name,
                        include_class=include_class,
                        encoding=encoding),
                (self._index._to_signature_bytes(
                        include_name=include_name,
                        include_class=include_class,
                        encoding=encoding),
                self._columns._to_signature_bytes(
                        include_name=include_name,
                        include_class=include_class,
                        encoding=encoding)),
                v))


    #---------------------------------------------------------------------------
    # exporters: alternate libraries

    def to_pandas(self) -> 'pandas.DataFrame':
        '''
        Return a Pandas DataFrame.
        '''
        import pandas

        if self._blocks.unified and self._blocks._blocks:
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
            with WarningsSilent():
                # Pandas issues: PerformanceWarning: DataFrame is highly fragmented.
                for i, array in enumerate(self._blocks.iter_columns_arrays()):
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

        def arrays() -> tp.Iterator[TNDArrayAny]:
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
            fp: tp.Union[TPathSpecifier, BytesIO],
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
        fpf = path_filter(fp) # type: ignore
        # NOTE:  compression='none' shown to not provide a clear performance improvement over the assumed default, 'snappy'
        pq.write_table(table, fpf)


    def to_msgpack(self) -> bytes:
        '''
        Return msgpack bytes.
        '''
        import msgpack
        import msgpack_numpy

        def encode(obj: tp.Union[ContainerOperand, IndexBase, TNDArrayAny],
                chain: tp.Callable[[tp.Any], tp.Dict[bytes, tp.Any]] = msgpack_numpy.encode,
                ) -> tp.Dict[bytes, tp.Any]: #returns dict that msgpack-python consumes
            cls = obj.__class__
            cls_name = cls.__name__
            package = cls.__module__.split('.', 1)[0]

            if package == 'static_frame':
                if isinstance(obj, Frame):
                    return {b'sf':cls_name,
                            b'name':obj.name,
                            b'blocks':packb(obj._blocks),
                            b'index':packb(obj.index),
                            b'columns':packb(obj.columns)}
                elif isinstance(obj, IndexHierarchy):
                    if obj._recache:
                        obj._update_array_cache()
                    return {b'sf':cls_name,
                            b'name':obj.name,
                            b'index_constructors': packb([
                                    a.__name__ for a in obj.index_types.values.tolist()]),
                            b'blocks':packb(obj._blocks)}
                elif isinstance(obj, Index):
                    return {b'sf':cls_name,
                            b'name':obj.name,
                            b'data':packb(obj.values)}
                elif isinstance(obj, TypeBlocks):
                    return {b'sf':cls_name,
                            b'blocks':packb(obj._blocks)}

            elif package == 'numpy':
                #msgpack-numpy is breaking with these data types, overriding here
                if obj.__class__ is np.ndarray:
                    if obj.dtype.kind == DTYPE_OBJECT_KIND: # type: ignore
                        data = list(map(element_encode, obj)) # type: ignore
                        return {b'np': True,
                                b'dtype': 'object_',
                                b'data': packb(data)}
                    elif obj.dtype.kind == DTYPE_DATETIME_KIND: # type: ignore
                        data = obj.astype(str) # type: ignore
                        return {b'np': True,
                                b'dtype': str(obj.dtype), # type: ignore
                                b'data': packb(data)}
                    elif obj.dtype.kind == DTYPE_TIMEDELTA_KIND: # type: ignore
                        data = obj.astype(DTYPE_FLOAT_DEFAULT) # type: ignore
                        return {b'np': True,
                                b'dtype': str(obj.dtype), # type: ignore
                                b'data': packb(data)}
            return chain(obj) #let msgpack_numpy.encode take over

        packb = partial(msgpack.packb, default=encode)
        # NOTE: element_encode used in closure above
        element_encode = partial(MessagePackElement.encode, packb=packb)
        return packb(self) # type: ignore

    def to_xarray(self) -> 'Dataset':
        '''
        Return an xarray Dataset.

        In order to preserve columnar types, and following the precedent of Pandas, the :obj:`Frame`, with a 1D index, is translated as a Dataset of 1D arrays, where each DataArray is a 1D array. If the index is an :obj:`IndexHierarchy`, each column is mapped into an ND array of shape equal to the unique values found at each depth of the index.
        '''
        import xarray

        columns = self.columns
        index = self.index

        index_name: tp.Union[TLabel, tp.Tuple[TLabel, ...]]
        if index.depth == 1:
            index_name = index.names[0]
            coords = {index_name: index.values}
        else:
            index_name = index.names
            # index values are reduced to unique values for 2d presentation
            coords = {index_name[d]: ufunc_unique1d(index.values_at_depth(d))
                    for d in range(index.depth)}
            # create dictionary version
            coords_index: tp.Dict[TLabel, Index[tp.Any]] = {
                    k: Index(v) for k, v in coords.items()
                    }

        # columns form the keys in data_vars dict
        columns_values: tp.Iterable[tp.Any]
        if columns.depth == 1:
            columns_values = columns.values
            # needs to be called with axis argument
            columns_arrays = partial(self._blocks.axis_values, axis=0)
        else: # must be hashable
            columns_values = array_to_tuple_iter(columns.values)

            def columns_arrays() -> tp.Iterator[TNDArrayAny]:
                c: TSeriesAny
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

        return xarray.Dataset(data_vars, coords=coords)

    def _to_frame(self,
            constructor: tp.Type[TFrameAny],
            *,
            name: TName = NAME_DEFAULT,
            ) -> TFrameAny:

        if self.__class__ is constructor and constructor in (Frame, FrameHE):
            if name is not NAME_DEFAULT:
                return self.rename(name)
            return self

        own_columns = constructor is not FrameGO and self.__class__ is not FrameGO # type: ignore

        return constructor(
                self._blocks.copy(),
                index=self.index,
                columns=self._columns,
                name=name if name is not NAME_DEFAULT else self._name,
                own_data=True,
                own_index=True,
                own_columns=own_columns,
                )

    def to_frame(self,
            *,
            name: TName = NAME_DEFAULT,
            ) -> TFrameAny:
        '''
        Return ``Frame`` instance from this ``Frame``. If this ``Frame`` is immutable the same instance will be returned.
        '''
        return self._to_frame(Frame, name=name)

    def to_frame_he(self,
            *,
            name: TName = NAME_DEFAULT,
            ) -> TFrameHEAny:
        '''
        Return a ``FrameHE`` instance from this ``Frame``. If this ``Frame`` is immutable the same instance will be returned.
        '''
        return self._to_frame(FrameHE, name=name) #type: ignore

    def to_frame_go(self,
            *,
            name: TName = NAME_DEFAULT,
            ) -> TFrameGOAny:
        '''
        Return a ``FrameGO`` instance from this ``Frame``.
        '''
        return self._to_frame(FrameGO, name=name) #type: ignore

    def to_series(self,
            *,
            index_constructor: TIndexCtor = Index,
            name: TName = NAME_DEFAULT,
            ) -> TSeriesAny:
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

        # immutability should be preserved
        array = self._blocks.values.reshape(self._blocks.size)

        def labels() -> tp.Iterator[TLabel]:
            for row, col in np.ndindex(self._blocks.shape):
                yield index_tuples[row] + columns_tuples[col] # type: ignore

        index = index_constructor(labels())
        name = name if name is not NAME_DEFAULT else self._name

        return Series(array, index=index, own_index=True, name=name)

    #---------------------------------------------------------------------------
    # exporters: json

    @doc_inject(selector='json')
    def to_json_index(self, indent: tp.Optional[int] = None) -> str:
        '''
        Export a :obj:`Frame` as a JSON string constructed as follows: {json_index}

        Args:
            {indent}
        '''
        d = ((k, dict(zip(self._columns, v)))
                for k, v in self.iter_tuple_items(constructor=tuple, axis=1))
        return json.dumps(JSONFilter.encode_items(d), indent=indent)

    @doc_inject(selector='json')
    def to_json_columns(self, indent: tp.Optional[int] = None) -> str:
        '''
        Export a :obj:`Frame` as a JSON string constructed as follows: {json_columns}

        Args:
            {indent}
        '''
        d = ((k, dict(zip(self._index, v))) for k, v in self.iter_array_items(axis=0))
        return json.dumps(JSONFilter.encode_items(d), indent=indent)

    @doc_inject(selector='json')
    def to_json_split(self,
            indent: tp.Optional[int] = None,
            ) -> str:
        '''
        Export a :obj:`Frame` as a JSON string constructed as follows: {json_split}

        Args:
            {indent}
        '''
        d = dict(columns=JSONFilter.encode_iterable(self._columns),
                index=JSONFilter.encode_iterable(self._index),
                data=JSONFilter.encode_iterable(
                        self._blocks.iter_row_tuples(key=None, constructor=tuple)
                        ),
                )
        return json.dumps(d, indent=indent)

    @doc_inject(selector='json')
    def to_json_records(self, indent: tp.Optional[int] = None) -> str:
        '''
        Export a :obj:`Frame` as a JSON string constructed as follows: {json_records}

        Args:
            {indent}
        '''
        d = (dict(zip(self._columns, v))
                for v in self.iter_tuple(constructor=tuple, axis=1))
        return json.dumps(JSONFilter.encode_iterable(d), indent=indent)

    @doc_inject(selector='json')
    def to_json_values(self, indent: tp.Optional[int] = None) -> str:
        '''
        Export a :obj:`Frame` as a JSON string constructed as follows: {json_values}

        Args:
            {indent}
        '''
        d = self._blocks.iter_row_tuples(key=None, constructor=tuple)
        return json.dumps(JSONFilter.encode_iterable(d), indent=indent)

    @doc_inject(selector='json')
    def to_json_typed(self,
            indent: tp.Optional[int] = None,
            ) -> str:
        '''
        Export a :obj:`Frame` as a JSON string constructed as follows: {json_typed}

        Args:
            {indent}
        '''
        index = (None if self._index._map is None # type: ignore
                else JSONFilter.encode_iterable(self._index))
        columns = (None if self._columns._map is None # type: ignore
                else JSONFilter.encode_iterable(self._columns))

        d = dict(columns=columns,
                index=index,
                data=JSONFilter.encode_iterable(self._blocks.iter_columns_arrays()),
                __meta__=JSONMeta.to_dict(self),
                )
        return json.dumps(d, indent=indent)

    #---------------------------------------------------------------------------
    # exporters: delimited

    def _to_str_records(self,
            *,
            include_index: bool = True,
            include_index_name: bool = True,
            include_columns: bool = True,
            include_columns_name: bool = False,
            store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT
            ) -> tp.Iterator[tp.Sequence[str]]:
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

        columns_rows: tp.Iterable[tp.Iterable[tp.Any]]
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
                        for name in index_names: # pyright: ignore
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

        col_idx_last = self._blocks._index.columns - 1
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
            fp: TPathSpecifierOrTextIO,
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
            csvw = csv.writer(fl, # type: ignore
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
            fp: TPathSpecifierOrTextIO,
            *,
            include_index: bool = True,
            include_index_name: bool = True,
            include_columns: bool = True,
            include_columns_name: bool = False,
            encoding: tp.Optional[str] = None,
            line_terminator: str = '\n',
            quoting: int = csv.QUOTE_MINIMAL,
            quote_char: str = '"',
            quote_double: bool = True,
            escape_char: tp.Optional[str] = None,
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
            fp: TPathSpecifierOrTextIO,
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
        root.clipboard_clear()
        root.clipboard_append(sio.read())

    #---------------------------------------------------------------------------
    # Store based output

    def to_xlsx(self,
            fp: TPathSpecifier,
            *,
            label: TLabel = STORE_LABEL_DEFAULT,
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
        from static_frame.core.store_config import StoreConfig
        from static_frame.core.store_xlsx import StoreXLSX

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
            fp: TPathSpecifier, # not sure file-like StringIO works
            *,
            label: TLabel = STORE_LABEL_DEFAULT,
            include_index: bool = True,
            include_columns: bool = True,
            # store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT,
            ) -> None:
        '''
        Write the Frame as single-table SQLite file.
        '''
        from static_frame.core.store_config import StoreConfig
        from static_frame.core.store_sqlite import StoreSQLite

        config = StoreConfig(
                include_index=include_index,
                include_columns=include_columns,
                )

        if label is STORE_LABEL_DEFAULT:
            if not self.name:
                raise RuntimeError('must provide a label or define `Frame` name.')
            label = self.name

        st = StoreSQLite(fp)
        st.write(((label, self),),
                config=config,
                # store_filter=store_filter,
                )

    def to_duckdb(self,
            fp: TPathSpecifier,
            *,
            label: TLabel = STORE_LABEL_DEFAULT,
            include_index: bool = True,
            include_columns: bool = True,
            ) -> None:
        '''
        Write the Frame as single-table DuckDB file.
        '''
        from static_frame.core.store_config import StoreConfig
        from static_frame.core.store_duckdb import StoreDuckDB

        config = StoreConfig(
                include_index=include_index,
                include_columns=include_columns,
                )

        if label is STORE_LABEL_DEFAULT:
            if not self.name:
                raise RuntimeError('must provide a label or define `Frame` name.')
            label = self.name

        st = StoreDuckDB(fp)
        st.write(((label, self),),
                config=config,
                )

    def to_hdf5(self,
            fp: TPathSpecifier, # not sure file-like StringIO works
            *,
            label: TLabel = STORE_LABEL_DEFAULT,
            include_index: bool = True,
            include_columns: bool = True,
            # store_filter: tp.Optional[StoreFilter] = STORE_FILTER_DEFAULT,
            ) -> None:
        '''
        Write the Frame as single-table SQLite file.
        '''
        from static_frame.core.store_config import StoreConfig
        from static_frame.core.store_hdf5 import StoreHDF5

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
    def to_npz(self,
            fp: TPathSpecifierOrBinaryIO,
            *,
            include_index: bool = True,
            include_columns: bool = True,
            consolidate_blocks: bool = False,
            ) -> None:
        '''
        Write a :obj:`Frame` as an npz file.
        '''
        NPZFrameConverter.to_archive(
                frame=self,
                fp=fp,
                include_index=include_index,
                include_columns=include_columns,
                consolidate_blocks=consolidate_blocks,
                )

    def to_npy(self,
            fp: TPathSpecifier, # not sure file-like StringIO works
            *,
            include_index: bool = True,
            include_columns: bool = True,
            consolidate_blocks: bool = False,
            ) -> None:
        '''
        Write a :obj:`Frame` as a directory of npy file.
        '''
        NPYFrameConverter.to_archive(
                frame=self,
                fp=fp,
                include_index=include_index,
                include_columns=include_columns,
                consolidate_blocks=consolidate_blocks,
                )

    def to_pickle(self,
            fp: TPathSpecifier,
            *,
            protocol: tp.Optional[int] = None,
            ) -> None:
        '''
        Write a :obj:`Frame` as a Python pickle.

        The pickle module is not secure. Only unpickle data you trust.

        Args:
            fp: file path to write.
            protocol: Pickle protocol to use.
        '''
        with open(fp, 'wb') as file:
            pickle.dump(self, file, protocol=protocol)

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
            fp: tp.Optional[TPathSpecifierOrTextIO] = None,
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
            import webbrowser  # pragma: no cover
            webbrowser.open_new_tab(fp) #type: ignore #pragma: no cover
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

# NOTE: must not use decorator to avoid mypy confusion on interface
doc_update(Frame.__init__, selector='container_init', class_name='Frame')

#-------------------------------------------------------------------------------

class FrameGO(Frame[TVIndex, TVColumns]):
    '''A grow-only Frame, providing a two-dimensional, ordered, labelled container, immutable with grow-only columns.
    '''

    __slots__ = ()

    STATIC = False
    _COLUMNS_CONSTRUCTOR = IndexGO
    _COLUMNS_HIERARCHY_CONSTRUCTOR = IndexHierarchyGO
    _columns: IndexGO[tp.Any]

    def __setitem__(self,
            key: TLabel,
            value: tp.Any,
            fill_value: tp.Any = np.nan
            ) -> None:
        '''For adding a single column, one column at a time.
        '''
        if key in self._columns:
            raise RuntimeError(f'The provided key ({key!r}) is already defined in columns; if you want to change or replace this column, use `Frame.assign` to get new `Frame`.')

        row_count = len(self._index)

        if isinstance(value, Series):
            # select only the values matching our index
            block = value.reindex(self.index, fill_value=fill_value).values
        elif isinstance(value, Index):
            if len(value) != row_count:
                raise RuntimeError(f'incorrectly sized unindexed value: {len(value)} != {row_count}')
            block = value.values
        elif isinstance(value, Frame):
            raise RuntimeError(
                    f'cannot use setitem with a Frame; use {self.__class__.__name__}.extend()')
        elif value.__class__ is np.ndarray:
            # this permits unaligned assignment as no index is used, possibly remove
            if value.ndim != 1:
                raise RuntimeError('can only use setitem with 1D containers')
            if len(value) != row_count:
                # block may have zero shape if created without columns
                raise RuntimeError(f'incorrectly sized unindexed value: {len(value)} != {row_count}')
            block = value

        else:
            if not hasattr(value, '__iter__') or isinstance(value, STRING_TYPES):
                block = np.full(row_count, value)
                block.flags.writeable = False
            else:
                block, _ = iterable_to_array_1d(value) # returns immutable

            if block.ndim != 1 or len(block) != row_count:
                raise RuntimeError('incorrectly sized, unindexed value')

        # Wait until after extracting block from value before updating _columns, as value evaluation might fail.
        try:
            self._columns.append(key)
        except GrowOnlyInvalid:
            # if GO is invalid, re-evaluate the type in a new Index
            self._columns = IndexGO(chain(self._columns, (key,)))
        self._blocks.append(block)

    def extend_items(self,
            pairs: tp.Iterable[tp.Tuple[TLabel, TSeriesAny]],
            fill_value: tp.Any = np.nan,
            ) -> None:
        '''
        Given an iterable of pairs of column name, column value, extend this FrameGO. Columns values can be any iterable suitable for usage in __setitem__.
        '''
        for k, v in pairs:
            self.__setitem__(k, v, fill_value)

    def extend(self,
            container: tp.Union[TFrameAny, TSeriesAny],
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
            try:
                self._columns.extend(container._columns)
            except GrowOnlyInvalid:
                self._columns = IndexGO(chain(self._columns, container._columns))
            self._blocks.extend(container._blocks)
        elif isinstance(container, Series):
            try:
                self._columns.append(container.name)
            except GrowOnlyInvalid:
                self._columns = IndexGO(chain(self._columns, (container.name,)))

            self._blocks.append(container.values)

        # this should never happen, and is hard to test!
        assert len(self._columns) == self._blocks._index.columns #pragma: no cover

    #---------------------------------------------------------------------------
    def via_fill_value(self,
            fill_value: object = np.nan,
            ) -> InterfaceFillValueGO[TFrameAny]:
        '''
        Interface for using binary operators and methods with a pre-defined fill value.
        '''
        return InterfaceFillValueGO(
                container=self,
                fill_value=fill_value,
                )

    #---------------------------------------------------------------------------
    # interfaces are redefined to show type returned type

    @property
    def loc(self) -> InterGetItemLocCompoundReduces[TFrameGOAny]:
        return InterGetItemLocCompoundReduces(self._extract_loc)

    @property
    def iloc(self) -> InterGetItemILocCompoundReduces[TFrameGOAny]:
        return InterGetItemILocCompoundReduces(self._extract_iloc)

#-------------------------------------------------------------------------------
class FrameHE(Frame[TVIndex, TVColumns, tp.Unpack[TVDtypes]]):
    '''
    A hash/equals subclass of :obj:`Frame`, permiting usage in a Python set, dictionary, or other contexts where a hashable container is needed. To support hashability, ``__eq__`` is implemented to return a Boolean rather than a Boolean :obj:`Frame`
    '''

    __slots__ = (
            '_hash',
            )

    _hash: int

    def __eq__(self, other: tp.Any) -> bool:
        '''
        Return True if other is a ``Frame`` with the same labels, values, and name. Container class and underlying dtypes are not independently compared.
        '''
        return self.equals(other,
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
            # NOTE: we hash based on labels, which we use as a faster-than full identity check
            self._hash = hash((
                    tuple(self.index),
                    tuple(self.columns),
                    ))
        return self._hash


    #---------------------------------------------------------------------------
    # interfaces are redefined to show type returned type

    @property
    def loc(self) -> InterGetItemLocCompoundReduces[TFrameHEAny]:
        return InterGetItemLocCompoundReduces(self._extract_loc)

    @property
    def iloc(self) -> InterGetItemILocCompoundReduces[TFrameHEAny]:
        return InterGetItemILocCompoundReduces(self._extract_iloc)


#-------------------------------------------------------------------------------
# utility delegates returned from selection routines and exposing the __call__ interface.

class FrameAssign(Assign):
    __slots__ = (
        'container',
        'key',
        )

    _INTERFACE = (
        '__call__',
        'apply',
        'apply_element',
        'apply_element_items',
        )

   # common base classe for supplying delegate; need to define interface for docs
    def __call__(self,
            value: tp.Any,
            *,
            fill_value: tp.Any = np.nan,
            ) -> TFrameAny:
        '''
        Assign the ``value`` in the position specified by the selector. The `name` attribute is propagated to the returned container.

        Args:
            value: Value to assign, which can be a :obj:`Series`, :obj:`Frame`, np.ndarray, or element.
            *.
            fill_value: If the ``value`` parameter has to be reindexed, this element will be used to fill newly created elements.
        '''
        raise NotImplementedError() #pragma: no cover

    def apply(self,
            func: TCallableAny,
            *,
            fill_value: tp.Any = np.nan,
            ) -> TFrameAny:
        '''
        Provide a function to apply to the assignment target, and use that as the assignment value.

        Args:
            func: A function to apply to the assignment target.
            *
            fill_value: If the function does not produce a container with a matching index, the element will be used to fill newly created elements.
        '''
        raise NotImplementedError() #pragma: no cover

    def apply_element(self,
            func: TCallableAny,
            *,
            dtype: TDtypeSpecifier = None,
            fill_value: tp.Any = np.nan,
            ) -> TFrameAny:
        '''
        Provide a function to apply to each element in the assignment target, and use that as the assignment value.

        Args:
            func: A function to apply to the assignment target.
            *
            fill_value: If the function does not produce a container with a matching index, the element will be used to fill newly created elements.
        '''
        return self.apply(
                lambda c: c.iter_element().apply(func, dtype=dtype),
                fill_value=fill_value,
                )

    def apply_element_items(self,
            func: TCallableAny,
            *,
            dtype: TDtypeSpecifier = None,
            fill_value: tp.Any = np.nan,
            ) -> TFrameAny:
        '''
        Provide a function, taking pairs of label, element, to apply to each element in the assignment target, and use that as the assignment value.

        Args:
            func: A function, taking pairs of label, element, to apply to the assignment target.
            *
            fill_value: If the function does not produce a container with a matching index, the element will be used to fill newly created elements.
        '''
        return self.apply(
                lambda c: c.iter_element_items().apply(func, dtype=dtype),
                fill_value=fill_value,
                )

class FrameAssignILoc(FrameAssign):
    __slots__ = ()

    def __init__(self,
            container: TFrameAny,
            key: TILocSelectorCompound = None,
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
            ) -> TFrameAny:
        is_frame = isinstance(value, Frame)
        is_series = isinstance(value, Series)

        key: tp.Tuple[TILocSelector, TILocSelector]
        if isinstance(self.key, tuple):
            # NOTE: the iloc key's order is not relevant in assignment, and block assignment requires that column keys are ascending
            key = (self.key[0], # type: ignore
                    key_to_ascending_key(
                            self.key[1], # type: ignore
                            self.container.shape[1]
                    ))
        else:
            key = (self.key, None)

        column_only = key[0] is None
        column_is_multiple = key[1] is None or isinstance(key[1], KEY_MULTIPLE_TYPES)

        assigned: TNDArrayAny | tp.Iterable[TNDArrayAny]
        if is_series:
            assigned = self.container._reindex_other_like_iloc(value,
                    key,
                    is_series=is_series,
                    is_frame=is_frame,
                    fill_value=fill_value).values
            blocks = self.container._blocks.extract_iloc_assign_by_unit(
                    key,
                    assigned,
                    )
        elif is_frame:
            assigned = self.container._reindex_other_like_iloc(value, # type: ignore
                    key,
                    is_series=is_series,
                    is_frame=is_frame,
                    fill_value=fill_value)._blocks._blocks # pyright: ignore
            blocks = self.container._blocks.extract_iloc_assign_by_blocks( # pyright: ignore
                    key,
                    assigned,
                    )
        elif (column_is_multiple
                and not column_only
                and not value.__class__ is np.ndarray
                and hasattr(value, '__len__')
                and not isinstance(value, tuple)
                and not isinstance(value, STRING_TYPES)
                ):
            # if column_only, we are expecting a "vertical" assignment, and use the by_unit interface
            blocks = self.container._blocks.extract_iloc_assign_by_sequence(
                    key,
                    value,
                    )
        else: # could be array or single element, or an NP array, or an iterable to be used for a column
            blocks = self.container._blocks.extract_iloc_assign_by_unit(
                    key,
                    value,
                    )

        return self.container.__class__(
                data=blocks,
                columns=self.container._columns,
                index=self.container._index,
                name=self.container._name,
                own_data=True
                )

    def apply(self,
            func: TCallableAny,
            *,
            fill_value: tp.Any = np.nan,
            ) -> TFrameAny:
        value = func(self.container.iloc[self.key])
        return self.__call__(value, fill_value=fill_value)


class FrameAssignBLoc(FrameAssign):
    __slots__ = ()

    def __init__(self,
            container: TFrameAny,
            key: TBlocKey = None,
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
            ) -> TFrameAny:
        is_frame = isinstance(value, Frame)
        is_series = isinstance(value, Series)

        # get Boolean key of normalized shape; in most cases this will be a new, mutable array
        key = bloc_key_normalize(
                key=self.key,
                container=self.container
                )
        if is_series:
            # assumes a Series from a bloc selection, i.e., tuples of index/col loc labels
            index = self.container._index
            columns = self.container._columns

            # cannot assume order of coordinates, so create a mapping for lookup by coordinate
            values_map: tp.Dict[tp.Tuple[int, int], tp.Any] = {}
            for (i, c), e in value.items():
                values_map[index._loc_to_iloc(i), columns._loc_to_iloc(c)] = e # type: ignore

            # NOTE: should we pass dtype here, or re-evaluate dtype from observed values for each block?
            blocks = self.container._blocks.extract_bloc_assign_by_coordinate(
                    key,
                    values_map,
                    value.values.dtype,
                    )

        elif is_frame:
            # NOTE: the object type of FILL_VALUE_DEFAULT might coerce other blocks
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
            func: TCallableAny,
            *,
            fill_value: tp.Any = np.nan,
            ) -> TFrameAny:
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
            container: TFrameAny,
            column_key: TILocSelector
            ) -> None:
        self.container = container
        self.column_key = column_key

    def __call__(self,
            dtypes: TDtypesSpecifier,
            *,
            consolidate_blocks: bool = False,
            ) -> TFrameAny:
        '''This method is only called after a __getitem__() selection has been made; this instance is created and returned from that __getitem__() call; this instance then exposes __call__() for the final provisioning of dtypes. When a root node gets __call__() direclty, an instance if this object is created and called.
        '''
        if self.column_key.__class__ is slice and self.column_key == NULL_SLICE:
            dtype_factory = get_col_dtype_factory(dtypes, self.container._columns)
            gen = self.container._blocks._astype_blocks_from_dtypes(dtype_factory)
        else:
            if not is_dtype_specifier(dtypes):
                raise RuntimeError('must supply a single dtype specifier if using a column selection other than the NULL slice')
            gen = self.container._blocks._astype_blocks(self.column_key, dtypes) # type: ignore

        if consolidate_blocks:
            gen = TypeBlocks.consolidate_blocks(gen)

        blocks = TypeBlocks.from_blocks(gen, shape_reference=self.container.shape)

        return self.container.__class__(
                data=blocks,
                columns=self.container.columns,
                index=self.container.index,
                name=self.container._name,
                own_data=True,
                )



TFrameAny = Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]]
TFrameHEAny = FrameHE[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]]
TFrameGOAny = FrameGO[tp.Any, tp.Any]
