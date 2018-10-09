

import typing as tp

from itertools import zip_longest

import numpy as np


from static_frame.core.util import _NULL_SLICE

from static_frame.core.util import _INT_TYPES
from static_frame.core.util import _KEY_ITERABLE_TYPES
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import GetItemKeyTypeCompound

from static_frame.core.util import mloc
from static_frame.core.util import _full_for_fill
from static_frame.core.util import _resolve_dtype
from static_frame.core.util import _resolve_dtype_iter
from static_frame.core.util import _dtype_to_na
from static_frame.core.util import _array_to_groups_and_locations
from static_frame.core.util import _isna

from static_frame.core.util import GetItem
from static_frame.core.util import IndexCorrespondence
from static_frame.core.util import immutable_filter

from static_frame.core.display import DisplayConfig
from static_frame.core.display import DisplayActive
from static_frame.core.display import Display

from static_frame.core.operator_delegate import MetaOperatorDelegate



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
            blocks.append(immutable_filter(raw_blocks))
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

                blocks.append(immutable_filter(block))

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
        Return a new TypeBlocks. Underlying arrays are not copied.
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
        # offsets = np.cumsum([b.shape[1] if b.ndim == 2 else 1 for b in self._blocks])
        # gens = [np.ndenumerate(self.single_column_filter(b)) for b in self._blocks]

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
        if stop_idx > start_idx: # ascending indices
            return slice(start_idx, stop_idx + 1)

        if stop_idx == 0:
            return slice(start_idx, None, -1)
        # stop is less than start, need to reduce by 1 to cover range
        return slice(start_idx, stop_idx - 1, -1)


    @classmethod
    def _indices_to_contiguous_pairs(cls, indices) -> tp.Generator:
        '''Indices are pairs of (block_idx, value); convert these to pairs of (block_idx, slice) when we identify contiguous indices within a block (these are block slices)

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

    def _all_block_slices(self):
        '''
        Alternaitve to _indices_to_contiguous_pairs when we need all indices per block in a slice.
        '''
        for idx, b in enumerate(self._blocks):
            if b.ndim == 1:
                yield (idx, slice(0, 1)) # cannot give an integer here instead of a slice
            else:
                yield (idx, slice(0, b.shape[1]))

    def _key_to_block_slices(self, key) -> tp.Generator[
                tp.Tuple[int, tp.Union[slice, int]], None, None]:
        '''
        For a column key (an integer, slice, or iterable), generate pairs of (block_idx, slice or integer) to cover all extractions. First, get the relevant index values (pairs of block id, column id), then convert those to contiguous slices.

        Returns:
            A generator iterable of pairs, where values are block index, slice or column index
        '''
        if key is None or (isinstance(key, slice) and slice == _NULL_SLICE):
            yield from self._all_block_slices()
        else:
            # do type checking on slice v others, as with others we need to sort once iterable of keys
            if isinstance(key, _INT_TYPES):
                # the index has the pair block, column integer
                yield self._index[key]
            else:
                if isinstance(key, slice):
                    # we have already handled the null slice
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
        The result is suitable for passing to TypeBlocks constructor.
        '''
        row_key_null = (row_key is None or
                (isinstance(row_key, slice) and row_key == _NULL_SLICE))

        single_row = False
        if row_key_null:
            if self._shape[0] == 1:
                # this codition used to only hold if the arg is a null slice; now if None too and shape has one row
                single_row = True
        elif isinstance(row_key, int):
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
                if row_key_null:
                    block_sliced = b
                else:
                    block_sliced = b[row_key]
            else: # given 2D, use row key and column slice
                if row_key_null:
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
            row_key_null = (row_key is None or
                    (isinstance(row_key, slice) and row_key == _NULL_SLICE))
            if b.ndim == 1:
                if row_key_null: # return a column
                    return TypeBlocks.from_blocks(b)
                elif isinstance(row_key, int):
                    return b[row_key] # return single item
                return TypeBlocks.from_blocks(b[row_key])

            if row_key_null:
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
                self_operands = self._blocks
                other_operands = other._blocks
            elif self._shape == other._shape:
                # if the result of reblock does not result in compatible shapes, we have to use .values as operands; the dtypes can be different so we only have to check that they columns sizes, the second element of the signature, all match.
                if not self.reblock_compatible(other):
                    self_operands = (self.values,)
                    other_operands = (other.values,)
                else:
                    self_operands = self._reblock()
                    other_operands = other._reblock()
            else: # raise same error as NP
                raise NotImplementedError('cannot apply binary operators to arbitrary TypeBlocks')

            def operation():
                for a, b in zip_longest(
                        (self.single_column_filter(op) for op in self_operands),
                        (self.single_column_filter(op) for op in other_operands)
                        ):
                    result = operator(a, b)
                    result.flags.writeable = False # own the data
                    yield result
        else:
            # process other as an array
            self_operands = self._blocks
            if not isinstance(other, np.ndarray):
                # this maybe expensive for a single scalar
                other = np.array(other) # this will work with a single scalar too

            # handle dimensions
            if other.ndim == 0 or (other.ndim == 1 and len(other) == 1):
                # a scalar: reference same value for each block position
                other_operands = (other for _ in range(len(self._blocks)))
            elif other.ndim == 1 and len(other) == self._shape[1]:
                # if given a 1d array
                # one dimensional array of same size: chop to block width
                other_operands = (other[s] for s in self._block_shape_slices())
            else:
                raise NotImplementedError('cannot apply binary operators to arbitrary np arrays.')

            def operation():
                for a, b in zip_longest(self_operands, other_operands):
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
        self._blocks.append(immutable_filter(block))

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

