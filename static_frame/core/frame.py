


from types import GeneratorType
import typing as tp

import csv
import json

from collections import namedtuple

from itertools import zip_longest
from functools import partial


import numpy as np
from numpy.ma import MaskedArray


from static_frame.core.util import _DEFAULT_SORT_KIND
from static_frame.core.util import _NULL_SLICE
from static_frame.core.util import _KEY_MULTIPLE_TYPES

from static_frame.core.util import GetItemKeyType
from static_frame.core.util import GetItemKeyTypeCompound
from static_frame.core.util import CallableOrMapping
from static_frame.core.util import KeyOrKeys
from static_frame.core.util import FilePathOrFileLike
from static_frame.core.util import DtypeSpecifier
from static_frame.core.util import IndexSpecifier
from static_frame.core.util import IndexInitializer
from static_frame.core.util import FrameInitializer
from static_frame.core.util import immutable_filter

from static_frame.core.util import _gen_skip_middle
from static_frame.core.util import _iterable_to_array
from static_frame.core.util import _dict_to_sorted_items
from static_frame.core.util import _array_to_duplicated
from static_frame.core.util import _array_set_ufunc_many
from static_frame.core.util import _array2d_to_tuples

from static_frame.core.util import _read_url

from static_frame.core.util import GetItem
from static_frame.core.util import ExtractInterface
from static_frame.core.util import IndexCorrespondence

from static_frame.core.operator_delegate import MetaOperatorDelegate

from static_frame.core.iter_node import IterNodeApplyType
from static_frame.core.iter_node import IterNodeType
from static_frame.core.iter_node import IterNode

from static_frame.core.display import DisplayConfig
from static_frame.core.display import DisplayActive
from static_frame.core.display import Display


from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.series import Series
from static_frame.core.index import Index
from static_frame.core.index import IndexGO
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.index_hierarchy import IndexHierarchyGO

from static_frame.core.index import Index



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

        # TODO: expand to handle hierarchical indices

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
            index: tp.Optional[IndexInitializer]=None,
            columns: tp.Optional[IndexInitializer]=None
            ):
        '''Frame constructor from an iterable of rows.

        Args:
            records: Iterable of row values, provided either as arrays, tuples, lists, or namedtuples.
            index: Iterable of index labels, equal in length to the number of records.
            columns: Iterable of column labels, equal in length to the length of each row.
        '''
        derive_columns = False
        if columns is None:
            derive_columns = True
            # leave columns list in outer scope for blocks() to use
            columns = []

        # if records is np; we can just pass it to constructor, as is alrady a consolidate type
        if isinstance(records, np.ndarray):
            return cls(records, index=index, columns=columns)

        def blocks():

            if not hasattr(records, '__len__'):
                rows = list(records)
            else:
                rows = records

            row_reference = rows[0]
            row_count = len(rows)
            col_count = len(row_reference)

            column_getter = None
            if isinstance(row_reference, dict):
                col_idx_iter = (k for k, _ in _dict_to_sorted_items(row_reference))
                if derive_columns:
                    # just pass the key back
                    column_getter = lambda key: key
            else:
                # all other iterables
                col_idx_iter = range(col_count)
                if hasattr(row_reference, '_fields'):
                    if derive_columns:
                        column_getter = row_reference._fields.__getitem__

            # derive types from first rows
            # string, datetime64 types requires size, so cannot use np.fromiter, as we do not know the size of all columns
            for col_idx in col_idx_iter:
                if column_getter:
                    # side effect of generator!
                    columns.append(column_getter(col_idx))

                field_ref = row_reference[col_idx]
                column_type = (type(field_ref)
                        if not isinstance(field_ref, (str, np.datetime64))
                        else None)
                if column_type is None: # let array constructor determine type
                    values = np.array([row[col_idx] for row in rows])
                else:
                    values = np.fromiter(
                            (row[col_idx] for row in rows),
                            count=row_count,
                            dtype=column_type)
                values.flags.writeable = False
                yield values

        return cls(TypeBlocks.from_blocks(TypeBlocks.consolidate_blocks(blocks())),
                index=index,
                columns=columns,
                own_data=True)

    @classmethod
    def from_json(cls, json_data):
        '''Frame constructor from an in-memory JSON document.
        '''
        data = json.loads(json_data)
        return cls.from_records(data)

    @classmethod
    def from_json_url(cls, url):
        '''Frame constructor from a JSON document provided via a URL.
        '''
        return cls.from_json(_read_url(url))



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


    @classmethod
    def from_pandas(cls,
            value,
            *,
            own_data: bool=False,
            own_index: bool=False,
            own_columns: bool=False) -> 'Frame':
        '''Given a Pandas DataFrame, return a Frame.

        Args:
            value: Pandas DataFrame.
            own_data: If True, the underlying NumPy data array will be made immutable and used without a copy.
            own_index: If True, the underlying NumPy index label array will be made immutable and used without a copy.
            own_columns: If True, the underlying NumPy index label array will be made immutable and used without a copy.
        '''
        if own_index:
            index = value.index.values
            index.flags.writeable = False
        else:
            index = immutable_filter(value.index.values)

        if own_columns:
            columns = value.columns.values
            columns.flags.writeable = False
        else:
            columns = immutable_filter(value.columns.values)

        # create generator of contiguous typed data
        # calling .values will force type unification accross all columns
        def blocks():
            pairs = value.dtypes.items()
            column_start, dtype_current = next(pairs)
            column_last = None
            for column, dtype in pairs:

                if dtype != dtype_current:
                    # use loc to select before calling .values
                    array = value.loc[_NULL_SLICE, slice(column_start, column_last)].values
                    if own_data:
                        array.flags.writeable = False
                    yield array
                    column_start = column
                    dtype_current = dtype

                column_last = column

            # always have left over
            array = value.loc[_NULL_SLICE, slice(column_start, None)].values
            if own_data:
                array.flags.writeable = False
            yield array

        blocks = TypeBlocks.from_blocks(blocks())
        return cls(blocks,
                index=index,
                columns=columns,
                own_data=True)



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
            own_index: if True, the index is taken as is and is not passed to an Index initializer.
        '''
        # TODO: support construction from Series?

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
            self._blocks = TypeBlocks.from_blocks(data)

        elif isinstance(data, dict):
            # if a dictionary is given, it is treated as a dictionary of columns
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

        elif data is None and columns is None:
            # will have shape of 0,0
            self._blocks = TypeBlocks.from_none()

        elif not hasattr(data, '__len__') and not isinstance(data, str):
            # data is not None, single element to scale to size of index and columns
            def blocks_constructor(shape):
                a = np.full(shape, data)
                a.flags.writeable = False
                self._blocks = TypeBlocks.from_blocks(a)

        else:
            # could be list of lists to be made into an array
            a = np.array(data)
            a.flags.writeable = False
            self._blocks = TypeBlocks.from_blocks(a)


        # counts can be zero (not None) if _block was created but is empty
        row_count, col_count = self._blocks._shape if not blocks_constructor else (None, None)

        #-----------------------------------------------------------------------
        # index assignment

        # columns could be an np array, or an Index instance, thus cannot be truthy
        if columns is None or (hasattr(columns, '__len__') and len(columns) == 0):
            if col_count is None:
                raise Exception('cannot create columns when no data given')
            self._columns = self._COLUMN_CONSTRUCTOR(
                    range(col_count),
                    loc_is_iloc=True)
        elif own_columns or (hasattr(columns, 'STATIC') and columns.STATIC):
            # if it is a STATIC index we can assign directly
            self._columns = columns
        else:
            self._columns = self._COLUMN_CONSTRUCTOR(columns)


        if index is None or (hasattr(index, '__len__') and len(index) == 0):
            if row_count is None:
                raise Exception('cannot create rows when no data given')
            self._index = Index(range(row_count), loc_is_iloc=True)
        elif own_index or (hasattr(index, 'STATIC') and index.STATIC):
            self._index = index
        else:
            self._index = Index(index)

        # permit bypassing this check if the

        if blocks_constructor:
            row_count = self._index.__len__()
            col_count = self._columns.__len__()
            blocks_constructor((row_count, col_count))

        if row_count and len(self._index) != row_count:
            # row count might be 0 for an empty DF
            raise Exception('index provided do not have correct size')
        if len(self._columns) != col_count:
            import ipdb; ipdb.set_trace()
            raise Exception('columns provided do not have correct size')


        #-----------------------------------------------------------------------
        # attributes

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
            if isinstance(index, (Index, IndexHierarchy)):
                # always use the Index constructor for safe reuse when possible
                index = index.__class__(index)
            else: # create the Index if not already an index, assume 1D
                index = Index(index)
            index_ic = IndexCorrespondence.from_correspondence(self._index, index)
        else:
            index = self._index
            index_ic = None


        if columns is not None:
            if isinstance(columns, (Index, IndexHierarchy)):
                # always use the Index constructor for safe reuse when possible
                if columns.STATIC != self._COLUMN_CONSTRUCTOR.STATIC:
                    raise Exception('static status of index does not match expected column static status')
                columns = columns.__class__(columns)
            else: # create the Index if not already an columns, assume 1D
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
        Return a new Frame based on a mapping (or callable) from old to new index values.
        '''
        # create new index objects in both cases so as to call with own*
        index = self._index.relabel(index) if index else self._index.copy()
        columns = self._columns.relabel(columns) if columns else self._columns.copy()

        return self.__class__(
                self._blocks.copy(), # does not copy arrays
                index=index,
                columns=columns,
                own_data=True,
                own_index=True,
                own_columns=True)



    def reindex_flat(self,
            index: bool=False,
            columns: bool=False) -> 'Frame':
        '''
        Return a new Frame, where a hierarhical index or column (if deifined) is replaced with a flat, one-dimension index of tuples.
        '''

        index = self._index.flat() if index else self._index.copy()
        columns = self._columns.flat() if columns else self._columns.copy()

        return self.__class__(
                self._blocks.copy(), # does not copy arrays
                index=index,
                columns=columns,
                own_data=True,
                own_index=True,
                own_columns=True)

    def reindex_add_level(self,
            index: tp.Hashable=None,
            columns: tp.Hashable=None) -> 'Frame':
        '''
        Return a new Frame, adding a new root level to the index and/or columns.
        '''

        index = self._index.add_level(index) if index else self._index.copy()
        columns = self._columns.add_level(columns) if columns else self._columns.copy()

        return self.__class__(
                self._blocks.copy(), # does not copy arrays
                index=index,
                columns=columns,
                own_data=True,
                own_index=True,
                own_columns=True)

    def reindex_drop_level(self,
            index: int=0,
            columns: int=0) -> 'Frmae':
        '''
        Return a new Frame, dropping one or more leaf levels from the index and/or columns.
        '''

        index = self._index.drop_level(index) if index else self._index.copy()
        columns = self._columns.drop_level(columns) if columns else self._columns.copy()

        return self.__class__(
                self._blocks.copy(), # does not copy arrays
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
        return self._blocks._shape[0]

    def display(self, config: DisplayConfig=None) -> Display:
        config = config or DisplayActive.get()

        d = self._index.display(config=config)

        if self._blocks._shape[1] > config.display_columns:
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

    @property
    def columns(self):
        return self._columns

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
        return self._blocks._shape

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
        Extract based on iloc selection (indices have already mapped)
        '''
        blocks = self._blocks._extract(row_key=row_key, column_key=column_key)

        if not isinstance(blocks, TypeBlocks):
            return blocks # reduced to element

        own_index = True # the extracted Frame can always own this index
        if row_key is None:
            index = self._index
        elif isinstance(row_key, slice) and row_key == _NULL_SLICE:
            index = self._index
        else:
            index = self._index._extract_iloc(row_key)

        # can only own columns if _COLUMN_CONSTRUCTOR is static
        if column_key is None:
            columns = self._columns
            own_columns = self._COLUMN_CONSTRUCTOR.STATIC
        elif isinstance(column_key, slice) and column_key == _NULL_SLICE:
            columns = self._columns
            own_columns = self._COLUMN_CONSTRUCTOR.STATIC
        else:
            columns = self._columns._extract_iloc(column_key)
            own_columns = True

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
                # 2d block, get teh first row
                return Series(block[0], index=columns)
        elif blocks._shape[1] == 1: # if one column
            if axis_nm[1]: # if column key is not multi
                return Series(blocks.values, index=index)

        return self.__class__(blocks,
                index=index,
                columns=columns,
                own_data=True, # always get new TypeBlock instance above
                own_index=own_index,
                own_columns=own_columns
                )


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
                    (x != column_iloc for x in range(self._blocks._shape[1])),
                    count=self._blocks._shape[1],
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
        if isinstance(self._index, IndexHierarchy):
            index_values = list(_array2d_to_tuples(self._index.values))
        else:
            index_values = self._index.values

        if isinstance(self._columns, IndexHierarchy):
            columns_values = list(_array2d_to_tuples(self._columns.values))
        else:
            columns_values = self._columns.values

        if axis == 1:
            major = index_values
            minor = columns_values
        elif axis == 0:
            major = columns_values
            minor = index_values
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

            col_idx_last = self._blocks._shape[1] - 1
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
    # _COLUMN_HIERARCHY_CONSTRUCTOR = IndexHierarchyGO.from_any


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
                    value = np.full(self._blocks._shape[0], value)
                else:
                    # for now, we assume all values make sense to covnert to NP array
                    value = np.array(value)
                value.flags.writeable = False

            if value.ndim != 1 or (
                    self._blocks._shape[0] > 0 and
                    len(value) != self._blocks._shape[0]):
                # block may have zero shape if created without columns
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

        if len(self._columns) != self._blocks._shape[1]:
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
        # pass Index objects unchanged, so as to let types be handled elsewhere
        return self.data.__class__(
                data=blocks,
                columns=self.data.columns,
                index=self.data.index,
                own_data=True)





