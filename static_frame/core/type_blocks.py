

import typing as tp
from itertools import zip_longest
from itertools import chain
from functools import partial
from copy import deepcopy

import numpy as np


from static_frame.core.container import ContainerOperand
from static_frame.core.container_util import apply_binary_operator_blocks
from static_frame.core.container_util import apply_binary_operator_blocks_columnar
from static_frame.core.container_util import get_col_dtype_factory
from static_frame.core.display import Display
from static_frame.core.display import DisplayActive
from static_frame.core.display_config import DisplayConfig
from static_frame.core.doc_str import doc_inject
from static_frame.core.exception import AxisInvalid
from static_frame.core.exception import ErrorInitTypeBlocks
from static_frame.core.index_correspondence import IndexCorrespondence
from static_frame.core.node_selector import InterfaceGetItem
from static_frame.core.util import array_shift
from static_frame.core.util import array_to_groups_and_locations
from static_frame.core.util import array2d_to_tuples
from static_frame.core.util import binary_transition
from static_frame.core.util import column_1d_filter
from static_frame.core.util import column_2d_filter
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import DTYPE_INEXACT_KINDS
from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import dtype_to_fill_value
from static_frame.core.util import DtypeSpecifier
from static_frame.core.util import DtypesSpecifier
from static_frame.core.util import dtype_from_element
from static_frame.core.util import FILL_VALUE_DEFAULT
from static_frame.core.util import full_for_fill
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import GetItemKeyTypeCompound
from static_frame.core.util import immutable_filter
from static_frame.core.util import INT_TYPES
from static_frame.core.util import isna_array
from static_frame.core.util import iterable_to_array_nd
from static_frame.core.util import KEY_ITERABLE_TYPES
from static_frame.core.util import KEY_MULTIPLE_TYPES
from static_frame.core.util import mloc
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import resolve_dtype
from static_frame.core.util import resolve_dtype_iter
from static_frame.core.util import row_1d_filter
from static_frame.core.util import shape_filter
from static_frame.core.util import slice_to_ascending_slice
from static_frame.core.util import slices_from_targets
from static_frame.core.util import UFunc
from static_frame.core.util import ufunc_axis_skipna
from static_frame.core.util import UNIT_SLICE
from static_frame.core.util import EMPTY_ARRAY
from static_frame.core.util import isin_array
from static_frame.core.util import iterable_to_array_1d
from static_frame.core.util import concat_resolved
from static_frame.core.util import array_deepcopy

#-------------------------------------------------------------------------------
class TypeBlocks(ContainerOperand):
    '''An ordered collection of type-heterogenous, immutable NumPy arrays, providing an external array-like interface of a single, 2D array. Used by :obj:`Frame` for core, unindexed array management.

    A TypeBlocks instance can have a zero size shape (where the length of one axis is zero). Internally, when axis 0 (rows) is of size 0, we store similarly sized arrays. When axis 1 (columns) is of size 0, we do not store arrays, as such arrays do not define a type (as types are defined by columns).
    '''

    __slots__ = (
            '_blocks',
            '_dtypes',
            '_index',
            '_shape',
            '_row_dtype',
            )

    STATIC = False

    #---------------------------------------------------------------------------
    # constructors

    @classmethod
    def from_blocks(cls,
            raw_blocks: tp.Iterable[np.ndarray],
            shape_reference: tp.Optional[tp.Tuple[int, int]] = None
            ) -> 'TypeBlocks':
        '''
        Main constructor using iterator (or generator) of TypeBlocks; the order of the blocks defines the order of the columns contained.

        It is acceptable to construct blocks with a 0-sided shape.

        Args:
            raw_blocks: iterable (generator compatible) of NDArrays, or a single NDArray.
            shape_reference: optional argument to support cases where no blocks are found in the ``raw_blocks`` iterable, but the outer context is one with rows but no columns, or columns and no rows.

        '''
        blocks: tp.List[np.ndarray] = [] # ordered blocks
        dtypes: tp.List[np.dtype] = [] # column position to dtype
        index: tp.List[tp.Tuple[int, int]] = [] # columns position to blocks key
        block_count = 0

        row_count: tp.Optional[int]

        # if a single block, no need to loop
        if isinstance(raw_blocks, np.ndarray):
            if raw_blocks.ndim > 2:
                raise ErrorInitTypeBlocks('arrays of dimensionality greater than 2 cannot be used to create TypeBlocks')

            row_count, column_count = shape_filter(raw_blocks)
            if column_count == 0:
                # set shape but do not store array
                return cls(blocks=blocks,
                        dtypes=dtypes,
                        index=index,
                        shape=(row_count, column_count)
                        )
            blocks.append(immutable_filter(raw_blocks))
            for i in range(column_count):
                index.append((block_count, i))
                dtypes.append(raw_blocks.dtype)

        else: # an iterable of blocks
            row_count = None
            column_count = 0

            for block in raw_blocks:
                if not isinstance(block, np.ndarray):
                    raise ErrorInitTypeBlocks(f'found non array block: {block}')

                if block.ndim > 2:
                    raise ErrorInitTypeBlocks(f'cannot include array with {block.ndim} dimensions')

                r, c = shape_filter(block)

                # check number of rows is the same for all blocks
                if row_count is not None and r != row_count:
                    raise ErrorInitTypeBlocks(f'mismatched row count: {r}: {row_count}')
                else: # assign on first
                    row_count = r

                # we keep array with 0 rows but > 0 columns, as they take type spce in the TypeBlocks object; arrays with 0 columns do not take type space and thus can be skipped entirely
                if c == 0:
                    continue

                blocks.append(immutable_filter(block))

                # store position to key of block, block columns
                for i in range(c):
                    index.append((block_count, i))
                    dtypes.append(block.dtype)

                column_count += c
                block_count += 1

        # blocks cam be empty
        if row_count is None:
            if shape_reference is not None:
                # if columns have gone to zero, and this was created from a TB that had rows, continue to represent those rows
                row_count = shape_reference[0]
            else:
                raise ErrorInitTypeBlocks('cannot derive a row_count from blocks; provide a shape reference')

        return cls(
                blocks=blocks,
                dtypes=dtypes,
                index=index,
                shape=(row_count, column_count),
                )

    @classmethod
    def from_element_items(cls,
            items: tp.Iterable[tp.Tuple[tp.Tuple[int, ...], object]],
            shape: tp.Tuple[int, ...],
            dtype: np.dtype,
            fill_value: object = FILL_VALUE_DEFAULT
            ) -> 'TypeBlocks':
        '''Given a generator of pairs of iloc coords and values, return a TypeBlock of the desired shape and dtype. This permits only uniformly typed data, as we have to create a single empty array first, then populate it.
        '''
        fill_value = (fill_value if fill_value is not FILL_VALUE_DEFAULT
                else dtype_to_fill_value(dtype))

        a = np.full(shape, fill_value=fill_value, dtype=dtype)
        for iloc, v in items:
            a[iloc] = v
        a.flags.writeable = False
        return cls.from_blocks(a)

    @classmethod
    def from_zero_size_shape(cls,
            shape: tp.Tuple[int, int] = (0, 0)
            ) -> 'TypeBlocks':
        '''
        Given a shape where one or both axis is 0 (a zero sized array), return a TypeBlocks instance.
        '''
        #NOTE: might want to take dtypes here, so as we can create a zero row Frame with properly defined dtypes. The challenge is that DtypesSpecifier includes column name maps, and we do not have access to an index-like map in this context.

        rows, columns = shape

        if not (rows == 0 or columns == 0):
            raise RuntimeError(f'invalid shape for empty TypeBlocks: {shape}')

        # as types are organized vertically, storing an array with 0 rows but > 0 columns is appropriate as it takes type space
        if rows == 0 and columns > 0:
            a = np.empty(shape)
            a.flags.writeable = False
            return cls.from_blocks(a)

        # for arrays with no width, favor storing shape alone and not creating an array object; the shape will be binding for future appending
        return cls(blocks=list(), dtypes=list(), index=list(), shape=shape)

    #---------------------------------------------------------------------------

    def __init__(self, *,
            blocks: tp.List[np.ndarray],
            dtypes: tp.List[np.dtype],
            index: tp.List[tp.Tuple[int, int]],
            shape: tp.Tuple[int, int]
            ) -> None:
        '''
        Default constructor. We own all lists passed in to this constructor.

        Args:
            blocks: A list of one or two-dimensional NumPy arrays
            dtypes: list of dtypes per external column
            index: list of pairs, where the first element is the block index, the second elemetns is the intra-block column
            shape: two-element tuple defining row and column count. A (0, 0) shape is permitted for empty TypeBlocks.
        '''
        self._blocks = blocks
        self._dtypes = dtypes
        self._index = index # list where index, as column, gets block, offset
        self._shape = shape

        if self._blocks:
            self._row_dtype = resolve_dtype_iter(b.dtype for b in self._blocks)
        else:
            # NOTE: this violates the type; however, this is desirable when appending such that this value does not force an undesirable type resolution
            self._row_dtype = None

    #---------------------------------------------------------------------------
    def __setstate__(self, state: tp.Tuple[object, tp.Mapping[str, tp.Any]]) -> None:
        '''
        Ensure that reanimated NP arrays are set not writeable.
        '''
        for key, value in state[1].items():
            setattr(self, key, value)

        for b in self._blocks:
            b.flags.writeable = False

    def __deepcopy__(self, memo: tp.Dict[int, tp.Any]) -> 'TypeBlocks':
        obj = self.__new__(self.__class__)
        obj._blocks = [array_deepcopy(b, memo) for b in self._blocks]
        obj._dtypes = deepcopy(self._dtypes, memo)
        obj._index = self._index.copy() # list of tuples of ints
        obj._shape = self._shape # immutable, no copy necessary
        obj._row_dtype = deepcopy(self._row_dtype, memo)

        memo[id(self)] = obj
        return obj #type: ignore

    def __copy__(self) -> 'TypeBlocks':
        '''
        Return shallow copy of this TypeBlocks. Underlying arrays are not copied.
        '''
        return self.__class__(
                blocks=[b for b in self._blocks],
                dtypes=self._dtypes.copy(), # list
                index=self._index.copy(),
                shape=self._shape,
                )

    def copy(self) -> 'TypeBlocks':
        '''
        Return shallow copy of this TypeBlocks. Underlying arrays are not copied.
        '''
        return self.__copy__()

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

    @property
    def shapes(self) -> np.ndarray:
        '''
        Return an immutable array that, for each block, reports the shape as a tuple.
        '''
        a = np.empty(len(self._blocks), dtype=object)
        a[:] = [b.shape for b in self._blocks]
        a.flags.writeable = False
        return a

    @property #type: ignore
    @doc_inject()
    def mloc(self) -> np.ndarray:
        '''{doc_array}
        '''
        a = np.fromiter(
                (mloc(b) for b in self._blocks),
                count=len(self._blocks),
                dtype=np.int64)
        a.flags.writeable = False
        return a

    @property
    def unified(self) -> bool:
        return len(self._blocks) <= 1

    #---------------------------------------------------------------------------
    # interfaces

    @property
    def iloc(self) -> InterfaceGetItem: #type: ignore
        return InterfaceGetItem(self._extract_iloc)

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
    def _blocks_to_array(*,
            blocks: tp.Sequence[np.ndarray],
            shape: tp.Tuple[int, int],
            row_dtype: tp.Optional[np.dtype],
            row_multiple: bool
            ) -> np.ndarray:
        '''
        Given blocks and a combined shape, return a consolidated 2D or 1D array.

        Args:
            shape: used in construting returned array; not ussed as a constraint.
            row_multiple: if False, a single row reduces to a 1D
        '''
        # assume column_multiple is True, as this routine is called after handling extraction of single columns
        if len(blocks) == 1:
            if not row_multiple:
                return row_1d_filter(blocks[0])
            else:
                return column_2d_filter(blocks[0])

        # get empty array and fill parts
        # NOTE: row_dtype may be None if an unfillable array; defaults to NP default
        if not row_multiple:
            # return 1 row TypeBlock as a 1D array with length equal to the number of columns
            array = np.empty(shape[1], dtype=row_dtype)
        else: # get ndim 2 shape array
            array = np.empty(shape, dtype=row_dtype)

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
        # always return a 2D array
        return self._blocks_to_array(
                blocks=self._blocks,
                shape=self._shape,
                row_dtype=self._row_dtype,
                row_multiple=True)

    def axis_values(self,
            axis: int = 0,
            reverse: bool = False
            ) -> tp.Iterator[np.ndarray]:
        '''Generator of arrays produced along an axis. Clients can expect to get an immutable array.

        Args:
            axis: 0 iterates over columns, 1 iterates over rows
        '''

        if axis == 1: # iterate over rows
            zero_size = not bool(self._blocks)
            unified = self.unified
            # iterate over rows; might be faster to create entire values
            if not reverse:
                row_idx_iter = range(self._shape[0])
            else:
                row_idx_iter = range(self._shape[0] - 1, -1, -1)

            for i in row_idx_iter:
                if zero_size:
                    yield EMPTY_ARRAY
                elif unified:
                    b = self._blocks[0]
                    if b.ndim == 1:
                        # single element slice to force array creation (not an element)
                        yield b[i: i+1]
                    else:
                        # if a 2d array, we can yield rows through simple indexing
                        yield b[i]
                else:
                    # cannot use a generator w/ np concat
                    # use == for type comparisons
                    parts = []
                    for b in self._blocks:
                        if b.ndim == 1:
                            # get a slice to permit concatenation
                            key: tp.Union[int, slice] = slice(i, i+1)
                        else:
                            key = i
                        if b.dtype == self._row_dtype:
                            parts.append(b[key])
                        else:
                            parts.append(b[key].astype(self._row_dtype))
                    part = np.concatenate(parts)
                    part.flags.writeable = False
                    yield part

        elif axis == 0: # iterate over columns
            if not reverse:
                block_column_iter: tp.Iterable[tp.Tuple[int, int]] = self._index
            else:
                block_column_iter = reversed(self._index)

            for block_idx, column in block_column_iter:
                b = self._blocks[block_idx]
                if b.ndim == 1:
                    yield b
                else:
                    yield b[:, column] # excpeted to be immutable
        else:
            raise AxisInvalid(f'no support for axis: {axis}')


    def element_items(self,
            axis: int = 0,
            ) -> tp.Iterator[tp.Tuple[tp.Tuple[int, int], tp.Any]]:
        '''
        Generator of pairs of iloc locations, values across entire TypeBlock. Used in creating a IndexHierarchy instance from a TypeBlocks.

        Args:
            axis: if 0, use row major iteration,  vary fastest along row.
        '''

        shape = self._shape if axis == 0 else (self._shape[1], self._shape[0])

        for iloc in np.ndindex(shape):
            if axis != 0: # invert
                iloc = (iloc[1], iloc[0])

            block_idx, column = self._index[iloc[1]]
            b = self._blocks[block_idx]
            if b.ndim == 1:
                yield iloc, b[iloc[0]]
            else:
                yield iloc, b[iloc[0], column]

    #---------------------------------------------------------------------------
    # methods for evaluating compatibility with other blocks, and reblocking
    def _reblock_signature(self) -> tp.Iterator[tp.Tuple[np.dtype, int]]:
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

    def block_compatible(self,
            other: 'TypeBlocks',
            axis: tp.Optional[int] = None) -> bool:
        '''Block compatible means that the blocks are the same shape. Type is not yet included in this evaluation.

        Args:
            axis: If True, the full shape is compared; if False, only the columns width is compared.
        '''
        # if shape characteristics do not match, blocks cannot be compatible
        if axis is None and self.shape != other.shape:
            return False
        elif axis is not None and self.shape[axis] != other.shape[axis]:
            return False

        for a, b in zip_longest(self._blocks, other._blocks, fillvalue=None):
            if a is None or b is None:
                return False
            if axis is None:
                if shape_filter(a) != shape_filter(b):
                    return False
            else:
                if shape_filter(a)[axis] != shape_filter(b)[axis]:
                    return False
        return True

    def reblock_compatible(self, other: 'TypeBlocks') -> bool:
        '''
        Return True if post reblocking these TypeBlocks are compatible. This only compares columns in blocks, not the entire shape.
        '''
        if self.shape[1] != other.shape[1]:
            return False
        # we only compare size, not the type
        return not any(a is None or b is None or a[1] != b[1]
                for a, b in zip_longest(
                self._reblock_signature(),
                other._reblock_signature()))

    @classmethod
    def _concatenate_blocks(cls,
            group: tp.Iterable[np.ndarray],
            dtype: DtypeSpecifier = None,
            ) -> np.array:
        '''Join blocks on axis 1, assuming the they have an appropriate dtype. This will always return a 2D array.
        '''
        # NOTE: if len(group) is 1, can return
        post = np.concatenate([column_2d_filter(x) for x in group], axis=1)
        # NOTE: if give non-native byteorder dtypes, will convert them to native
        if dtype is not None and post.dtype != dtype:
            # could use `out` arguement of np.concatenate to avoid copy, but would have to calculate resultant size first
            return post.astype(dtype)
        return post

    @classmethod
    def consolidate_blocks(cls,
            raw_blocks: tp.Iterable[np.ndarray]) -> tp.Iterator[np.ndarray]:
        '''
        Generator consumer, generator producer of np.ndarray, consolidating if types are exact matches.
        '''
        group_dtype = None # store type found along contiguous blocks
        group = []

        for block in raw_blocks:
            if group_dtype is None: # first block of a type
                group_dtype = block.dtype
                group.append(block)
                continue

            # NOTE: could be less strict and look for compatibility within dtype kind (or other compatible types)
            if block.dtype != group_dtype:
                # new group found, return stored
                if len(group) == 1: # return reference without copy
                    # NOTE: using pop() here not shown to be faster
                    yield group[0]
                else: # combine groups
                    # could pre allocating and assing as necessary for large groups
                    yield cls._concatenate_blocks(group, group_dtype)
                group_dtype = block.dtype
                group = [block]
            else: # new block has same group dtype
                group.append(block)

        # always have one or more leftover
        if group:
            if len(group) == 1:
                yield group[0]
            else:
                yield cls._concatenate_blocks(group, group_dtype)


    def _reblock(self) -> tp.Iterator[np.ndarray]:
        '''Generator of new block that consolidate adjacent types that are the same.
        '''
        yield from self.consolidate_blocks(raw_blocks=self._blocks)

    def consolidate(self) -> 'TypeBlocks':
        '''Return a new TypeBlocks that unifies all adjacent types.
        '''
        # note: not sure if we have a single block if we should return a new TypeBlocks instance (as done presently), or simply return self; either way, no new np arrays will be created
        return self.from_blocks(self.consolidate_blocks(raw_blocks=self._blocks))


    def resize_blocks(self, *,
            index_ic: tp.Optional[IndexCorrespondence],
            columns_ic: tp.Optional[IndexCorrespondence],
            fill_value: tp.Any
            ) -> tp.Iterator[np.ndarray]:
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
                    shape: tp.Union[int, tp.Tuple[int, int]] = index_ic.size if b.ndim == 1 else (index_ic.size, b.shape[1])
                    values = full_for_fill(b.dtype, shape, fill_value)
                    if index_ic.has_common:
                        values[index_ic.iloc_dst] = b[index_ic.iloc_src]
                    values.flags.writeable = False
                    yield values

        elif columns_ic is not None and index_ic is None:
            if not columns_ic.has_common:
                # just return an empty frame; what type it shold be is not clear
                shape = self.shape[0], columns_ic.size
                values = full_for_fill(self._row_dtype, shape, fill_value)
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
                    dst_to_src = dict(
                            zip(
                                    tp.cast(tp.Iterable[int], columns_ic.iloc_dst),
                                    tp.cast(tp.Iterable[int], columns_ic.iloc_src),
                            )
                    )
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
                            values = full_for_fill(self._row_dtype,
                                    self.shape[0],
                                    fill_value)
                            values.flags.writeable = False
                            yield values

        else: # both defined
            assert columns_ic is not None and index_ic is not None
            if not columns_ic.has_common and not index_ic.has_common:
                # just return an empty frame; what type it shold be is not clear
                shape = index_ic.size, columns_ic.size
                values = full_for_fill(self._row_dtype, shape, fill_value)
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
                    columns_dst_to_src = dict(
                            zip(
                                    tp.cast(tp.Iterable[int], columns_ic.iloc_dst),
                                    tp.cast(tp.Iterable[int], columns_ic.iloc_src),
                            )
                    )

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
                                values = full_for_fill(self._row_dtype,
                                        index_ic.size,
                                        fill_value)
                                if b.ndim == 1:
                                    values[index_ic.iloc_dst] = b[index_ic.iloc_src]
                                else:
                                    values[index_ic.iloc_dst] = b[index_ic.iloc_src, block_col]
                                values.flags.writeable = False
                                yield values
                        else:
                            values = full_for_fill(self._row_dtype,
                                        index_ic.size,
                                        fill_value)
                            values.flags.writeable = False
                            yield values


    def group(self,
            axis: int,
            key: GetItemKeyTypeCompound,
            # drop: bool = False,
            ) -> tp.Iterator[tp.Tuple[np.ndarray, np.ndarray, 'TypeBlocks']]:
        '''
        Args:
            key: iloc selector on opposite axis

        Returns:
            Generator of group, selection pairs, where selection is an np.ndarray. Returned is as an np.ndarray if key is more than one column.
        '''
        # in worse case this will make a copy of the values extracted; this is probably still cheaper than iterating manually through rows/columns
        unique_axis = None

        # NOTE: in axis_values we determine zero size by looking for empty _blocks; not sure if that is appropriate here.
        if self._shape[0] == 0 or self._shape[1] == 0: # zero sized
            return

        if axis == 0:
            # axis 0 means we return row groups; key is a column key
            group_source = self._extract_array(column_key=key)
            if group_source.ndim > 1:
                unique_axis = 0
        elif axis == 1:
            # axis 1 means we return column groups; key is a row key
            group_source = self._extract_array(row_key=key)
            if group_source.ndim > 1 and group_source.shape[0] > 1:
                unique_axis = 1
        else:
            raise AxisInvalid(f'invalid axis: {axis}')

        groups, locations = array_to_groups_and_locations(
                group_source,
                unique_axis)

        if unique_axis is not None:
            # NOTE: this is expensive!
            # make the groups hashable for usage in index construction
            if axis == 0:
                groups = array2d_to_tuples(groups)
            elif axis == 1:
                groups = array2d_to_tuples(groups.T)

        for idx, g in enumerate(groups):
            selection = locations == idx
            if axis == 0: # return row extractions
                yield g, selection, self._extract(row_key=selection)
            elif axis == 1: # return columns extractions
                yield g, selection, self._extract(column_key=selection)


    #---------------------------------------------------------------------------
    # transformations resulting in reduced dimensionality

    def ufunc_axis_skipna(self, *,
            skipna: bool,
            axis: int,
            ufunc: UFunc,
            ufunc_skipna: UFunc,
            composable: bool,
            dtypes: tp.Tuple[np.dtype, ...],
            size_one_unity: bool
            ) -> np.ndarray:
        '''Apply a function that reduces blocks to a single axis. Note that this only works in axis 1 if the operation can be applied more than once, first by block, then by reduced blocks. This will not work for a ufunc like argmin, argmax, where the result of the function cannot be compared to the result of the function applied on a different block.

        Args:
            composable: when True, the function application will return a correct result by applying the function to blocks first, and then the result of the blocks (i.e., add, prod); where observation count is relevant (i.e., mean, var, std), this must be False.
            dtypes: if we know the return type of func, we can provide it here to avoid having to use the row dtype.

        Returns:
            As this is a reduction of axis where the caller (a Frame) is likely to return a Series, this function is not a generator of blocks, but instead just returns a consolidated 1d array.
        '''
        if axis < 0 or axis > 1:
            raise RuntimeError(f'invalid axis: {axis}')

        func = partial(ufunc_axis_skipna,
                skipna=skipna,
                ufunc=ufunc,
                ufunc_skipna=ufunc_skipna,
                )

        if self.unified:
            result = func(array=column_2d_filter(self._blocks[0]), axis=axis)
            result.flags.writeable = False
            return result
        else:
            if axis == 0:
                # reduce all rows to 1d with column width
                shape: tp.Union[int, tp.Tuple[int, int]] = self._shape[1]
                pos = 0
            elif composable: # axis 1
                # reduce all columns to 2d blocks with 1 column
                shape = (self._shape[0], len(self._blocks))
            else: # axis 1, not block composable
                # Cannot do block-wise processing, must resolve to single array and return
                array = self._blocks_to_array(
                        blocks=self._blocks,
                        shape=self._shape,
                        row_dtype=self._row_dtype,
                        row_multiple=True)
                result = func(array=array, axis=axis)
                result.flags.writeable = False
                return result

            # this will be uninitialzied and thus, if a value is not assigned, will have garbage
            if dtypes:
                # Favor self._row_dtype's kind if it is in dtypes, else take first of passed dtypes
                for dt in dtypes:
                    if self._row_dtype.kind == dt.kind:
                        dtype = self._row_dtype
                        break
                else: # no break encountered
                    dtype = dtypes[0]
                astype_pre = dtype.kind in DTYPE_INEXACT_KINDS
            else:
                dtype = self._row_dtype
                astype_pre = True # if no dtypes given (like bool) we can coerce

            # If dtypes were specified, we know we have specific targets in mind for output
            out = np.empty(shape, dtype=dtype)
            # print('out', out, out.dtype, self._row_dtype)
            for idx, b in enumerate(self._blocks):

                if astype_pre and b.dtype != dtype:
                    b = b.astype(dtype)

                if axis == 0: # Combine rows, end with columns shape.
                    if b.size == 1 and size_one_unity and not skipna:
                        # No function call is necessary; if skipna could turn NaN to zero.
                        end = pos + 1
                        # Can assign an array, even 2D, as an element if size is 1
                        out[pos] = b
                    elif b.ndim == 1:
                        end = pos + 1
                        out[pos] = func(array=b, axis=axis)
                    else:
                        end = pos + b.shape[1]
                        func(array=b, axis=axis, out=out[pos: end])
                    pos = end
                else:
                    # Combine columns, end with block length shape and then call func again, for final result
                    if b.size == 1 and size_one_unity and not skipna:
                        out[:, idx] = b
                    elif b.ndim == 1:
                        # if this is a composable, numeric single columns we just copy it and process it later; but if this is a logical application (and, or) then it is already Boolean
                        if out.dtype == DTYPE_BOOL and b.dtype != DTYPE_BOOL:
                            # making 2D with axis 0 func will result in element-wise operation
                            out[:, idx] = func(array=column_2d_filter(b), axis=1)
                        else: # otherwise, keep as is
                            out[:, idx] = b
                    else:
                        func(array=b, axis=axis, out=out[:, idx])

        if axis == 0: # nothing more to do
            out.flags.writeable = False
            return out
        # If axis 1 and composable, can call function one more time on remaining components. Note that composability is problematic in cases where overflow is possible
        result = func(array=out, axis=1)
        result.flags.writeable = False
        return result


    #---------------------------------------------------------------------------
    def __round__(self, decimals: int = 0) -> 'TypeBlocks':
        '''
        Return a TypeBlocks rounded to the given decimals. Negative decimals round to the left of the decimal point.
        '''
        func = partial(np.round, decimals=decimals)
        # for now, we do not expose application of rounding on a subset of blocks, but is doable by setting the column_key
        return self.__class__(
                blocks=list(self._ufunc_blocks(column_key=NULL_SLICE, func=func)),
                dtypes=self._dtypes.copy(), # list
                index=self._index.copy(),
                shape=self._shape
                )

    def __len__(self) -> int:
        '''Length, as with NumPy and Pandas, is the number of rows. Note that A shape of (3, 0) will return a length of 3, even though there is no data.
        '''
        return self._shape[0]

    @doc_inject()
    def display(self,
            config: tp.Optional[DisplayConfig] = None
            ) -> Display:
        '''{doc}

        Args:
            {config}
        '''
        # NOTE: the TypeBlocks Display is not composed into other Displays

        config = config or DisplayActive.get()
        d = None
        outermost = True # only for the first
        idx = 0
        for block in self._blocks:
            block = column_2d_filter(block)
            if block.shape[1] == 0:
                continue

            h = '' if idx > 0 else self.__class__

            display = Display.from_values(block,
                    h,
                    config=config,
                    outermost=outermost)
            if not d: # assign first
                d = display
                outermost = False
            else:
                d.extend_display(display)
            # explicitly enumerate so as to not count no-width blocks
            idx += 1

        assert d is not None # for mypy
        return d


    #---------------------------------------------------------------------------
    # extraction utilities

    @staticmethod
    def _cols_to_slice(indices: tp.Sequence[int]) -> slice:
        '''Translate an iterable of contiguous integers into a slice. Integers are assumed to be intentionally ordered and contiguous.
        '''
        start_idx = indices[0]
        # single column as a single slice
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
    def _indices_to_contiguous_pairs(cls, indices: tp.Iterable[tp.Tuple[int, int]]
        ) -> tp.Iterator[tp.Tuple[int, slice]]:
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
                # do not need to store all col, only the last, however probably easier to just accumulate all
                bundle.append(col)
                continue
            # either new block, or not contiguous on same block
            yield (last[0], cls._cols_to_slice(bundle))
            # start a new bundle
            bundle = [col]
            last = (block_idx, col)

        # last can be None
        if last and bundle:
            yield (last[0], cls._cols_to_slice(bundle))

    def _all_block_slices(self) -> tp.Iterator[tp.Tuple[int, slice]]:
        '''
        Alternaitve to _indices_to_contiguous_pairs when we need all indices per block in a slice.
        '''
        for idx, b in enumerate(self._blocks):
            if b.ndim == 1:
                yield (idx, UNIT_SLICE) # cannot give an integer here instead of a slice
            else:
                yield (idx, slice(0, b.shape[1]))

    def _key_to_block_slices(self,
            key: GetItemKeyTypeCompound,
            retain_key_order: bool = True
            ) -> tp.Iterator[tp.Tuple[int, tp.Union[slice, int]]]:
        '''
        For a column key (an integer, slice, or iterable), generate pairs of (block_idx, slice or integer) to cover all extractions. First, get the relevant index values (pairs of block id, column id), then convert those to contiguous slices.

        Args:
            retain_key_order: if False, returned slices will be in ascending order.

        Returns:
            A generator iterable of pairs, where values are block index, slice or column index
        '''
        if key is None or (isinstance(key, slice) and key == NULL_SLICE):
            yield from self._all_block_slices() # slow from line profiler, 80% of this function call
        else:
            if isinstance(key, INT_TYPES):
                # the index has the pair block, column integer
                yield self._index[key]
            else: # all cases where we try to get contiguous slices
                if isinstance(key, slice):
                    #  slice the index; null slice already handled
                    if not retain_key_order:
                        key = slice_to_ascending_slice(key, self._shape[1])
                    indices: tp.Iterable[tp.Tuple[int, int]] = self._index[key]
                elif isinstance(key, np.ndarray) and key.dtype == bool:
                    # NOTE: if self._index was an array we could use Boolean selection directly
                    indices = (self._index[idx] for idx, v in enumerate(key) if v)
                elif isinstance(key, KEY_ITERABLE_TYPES):
                    # an iterable of keys, may not have contiguous regions; provide in the order given; set as a generator; self._index is a list, not an np.array, so cannot slice self._index; requires iteration in passed generator so probably this is as fast as it can be.
                    if retain_key_order:
                        indices = (self._index[x] for x in key)
                    else:
                        indices = (self._index[x] for x in sorted(key))
                elif key is None: # get all
                    indices = self._index
                else:
                    raise NotImplementedError('Cannot handle key', key)
                yield from self._indices_to_contiguous_pairs(indices)

    #---------------------------------------------------------------------------
    def _mask_blocks(self,
            row_key: tp.Optional[GetItemKeyTypeCompound] = None,
            column_key: tp.Optional[GetItemKeyTypeCompound] = None) -> tp.Iterator[np.ndarray]:
        '''Return Boolean blocks of the same size and shape, where key selection sets values to True.
        '''

        # this selects the columns; but need to return all blocks

        # block slices must be in ascending order, not key order
        block_slices = iter(self._key_to_block_slices(
                column_key,
                retain_key_order=False))
        target_block_idx = target_slice = None
        targets_remain = True

        for block_idx, b in enumerate(self._blocks):
            mask = np.full(b.shape, False, dtype=bool)

            while targets_remain:
                # get target block and slice
                if target_block_idx is None: # can be zero
                    try:
                        target_block_idx, target_slice = next(block_slices)
                    except StopIteration:
                        targets_remain = False
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


    def _astype_blocks(self,
            column_key: GetItemKeyType,
            dtype: DtypeSpecifier
            ) -> tp.Iterator[np.ndarray]:
        '''
        Give any column selection, apply a single dtype.
        Generator producer of np.ndarray.
        '''
        # block slices must be in ascending order, not key order
        block_slices = iter(self._key_to_block_slices(
                column_key,
                retain_key_order=False))

        target_slice: tp.Optional[tp.Union[slice, int]]

        target_block_idx = target_slice = None
        targets_remain = True

        for block_idx, b in enumerate(self._blocks):
            parts = []
            part_start_last = 0

            while targets_remain:
                # get target block and slice
                if target_block_idx is None: # can be zero
                    try:
                        target_block_idx, target_slice = next(block_slices)
                    except StopIteration:
                        targets_remain = False
                        break

                if block_idx != target_block_idx:
                    break # need to advance blocks

                if dtype == b.dtype:
                    target_block_idx = target_slice = None
                    continue # there may be more slices for this block

                if b.ndim == 1: # given 1D array, our row key is all we need
                    parts.append(b.astype(dtype))
                    part_start_last = 1
                    target_block_idx = target_slice = None
                    break

                assert target_slice is not None
                # target_slice can be a slice or an integer
                if isinstance(target_slice, slice):
                    target_start = target_slice.start
                    target_stop = target_slice.stop
                else: # it is an integer
                    target_start = target_slice
                    target_stop = target_slice + 1

                assert target_start is not None and target_stop is not None
                if target_start > part_start_last:
                    # yield un changed components before and after
                    parts.append(b[NULL_SLICE, slice(part_start_last, target_start)])

                parts.append(b[NULL_SLICE, target_slice].astype(dtype))
                part_start_last = target_stop

                target_block_idx = target_slice = None

            # if this is a 1D block, we either convert it or do not, and thus either have parts or not, and do not need to get other part pieces of the block
            if b.ndim != 1 and part_start_last < b.shape[1]:
                parts.append(b[NULL_SLICE, slice(part_start_last, None)])

            if not parts:
                yield b # no change for this block
            else:
                yield from parts



    def _astype_blocks_from_dtypes(self,
            dtypes: DtypesSpecifier
            ) -> tp.Iterator[np.ndarray]:
        '''
        Generator producer of np.ndarray.

        Args:
            dtypes: specify dtypes as single item, iterable, or mapping.
        '''
        # use a range() of integers as columns labels
        get_col_dtype = get_col_dtype_factory(dtypes, range(self._shape[1]))

        iloc = 0
        for b in self._blocks:
            if b.ndim == 1:
                dtype = get_col_dtype(iloc)
                if dtype is not None:
                    yield b.astype(dtype)
                else:
                    yield b
                iloc += 1
            else:
                group_start = 0
                for pos in range(b.shape[1]):
                    dtype = get_col_dtype(iloc)
                    if pos == 0:
                        dtype_last = dtype
                    elif dtype != dtype_last:
                        # this dtype is different, so need to cast all up to (but not including) this one
                        if dtype_last is not None:
                            yield b[NULL_SLICE, slice(group_start, pos)].astype(dtype_last)
                        else:
                            yield b[NULL_SLICE, slice(group_start, pos)]
                        group_start = pos # this is the start of a new group
                        dtype_last = dtype
                    # else: dtype is the same
                    iloc += 1
                # there is always one more to yield
                if dtype_last is not None:
                    yield b[NULL_SLICE, slice(group_start, None)].astype(dtype_last)
                else:
                    yield b[NULL_SLICE, slice(group_start, None)]

    def _ufunc_blocks(self,
            column_key: GetItemKeyType,
            func: UFunc
            ) -> tp.Iterator[np.ndarray]:
        '''
        Return a new blocks after processing each columnar block with the passed ufunc. It is assumed the ufunc will retain the shape of the input 1D or 2D array. All blocks must be processed, which is different than _astype_blocks, which can check the type and skip processing some blocks.

        Generator producer of np.ndarray.
        '''
        # block slices must be in ascending order, not key order
        block_slices = iter(self._key_to_block_slices(
                column_key,
                retain_key_order=False))

        target_slice: tp.Optional[tp.Union[slice, int]]

        target_block_idx = target_slice = None
        targets_remain = True

        for block_idx, b in enumerate(self._blocks):
            parts = []
            part_start_last = 0

            while targets_remain:
                # get target block and slice
                if target_block_idx is None: # can be zero
                    try:
                        target_block_idx, target_slice = next(block_slices)
                    except StopIteration:
                        targets_remain = False
                        break

                if block_idx != target_block_idx:
                    break # need to advance blocks

                if b.ndim == 1: # given 1D array, our row key is all we need
                    parts.append(func(b))
                    part_start_last = 1
                    target_block_idx = target_slice = None
                    break

                assert target_slice is not None
                # target_slice can be a slice or an integer
                if isinstance(target_slice, slice):
                    target_start = target_slice.start
                    target_stop = target_slice.stop
                else: # it is an integer
                    target_start = target_slice
                    target_stop = target_slice + 1

                assert target_start is not None and target_stop is not None
                if target_start > part_start_last:
                    # yield un changed components before and after
                    parts.append(b[:, slice(part_start_last, target_start)])

                # apply func
                parts.append(func(b[:, target_slice]))
                part_start_last = target_stop

                target_block_idx = target_slice = None

            # if this is a 1D block, we either convert it or do not, and thus either have parts or not, and do not need to get other part pieces of the block
            if b.ndim != 1 and part_start_last < b.shape[1]:
                parts.append(b[:, slice(part_start_last, None)])

            if not parts:
                yield b # no change for this block
            else:
                yield from parts

    def _drop_blocks(self,
            row_key: GetItemKeyType = None,
            column_key: GetItemKeyType = None,
            ) -> tp.Iterator[np.ndarray]:
        '''
        Generator producer of np.ndarray. Note that this appraoch should be more efficient than using selection/extraction, as here we are only concerned with columns.

        Args:
            column_key: Selection of columns to leave out of blocks.
        '''
        if column_key is None:
            # the default should not be the null slice, which would drop all
            block_slices: tp.Iterator[tp.Tuple[int, tp.Union[slice, int]]] = iter(())
        else:
            if not self._blocks:
                raise IndexError('cannot drop columns from zero-blocks')
            # block slices must be in ascending order, not key order
            block_slices = iter(self._key_to_block_slices(
                    column_key,
                    retain_key_order=False))

        if isinstance(row_key, np.ndarray) and row_key.dtype == bool:
            # row_key is used with np.delete, which does not support Boolean arrays; instead, convert to an array of integers
            row_key = np.arange(len(row_key))[row_key]

        target_block_idx = target_slice = None
        targets_remain = True

        for block_idx, b in enumerate(self._blocks):
            # for each block, we evaluate if we have any targets in that block and update the block accordingly; otherwise, we yield the block unchanged

            parts = []
            drop_block = False # indicate entire block is dropped
            part_start_last = 0 # within this block, keep track of where our last change was started

            while targets_remain:
                # get target block and slice; this is what we want to remove
                if target_block_idx is None: # can be zero
                    try:
                        target_block_idx, target_slice = next(block_slices)
                    except StopIteration:
                        targets_remain = False
                        break

                if block_idx != target_block_idx:
                    break # need to advance blocks

                if b.ndim == 1 or b.shape[1] == 1: # given 1D array or 2D, 1 col array
                    part_start_last = 1
                    target_block_idx = target_slice = None
                    drop_block = True
                    break

                # target_slice can be a slice or an integer
                if isinstance(target_slice, slice):
                    target_start = target_slice.start
                    target_stop = target_slice.stop
                else: # it is an integer
                    target_start = target_slice # can be zero
                    target_stop = target_slice + 1

                assert target_start is not None and target_stop is not None
                # if the target start (what we want to remove) is greater than 0 or our last starting point, then we need to slice off everything that came before, so as to keep it
                if target_start == 0 and target_stop == b.shape[1]:
                    drop_block = True
                elif target_start > part_start_last:
                    # yield retained components before and after
                    parts.append(b[:, slice(part_start_last, target_start)])
                part_start_last = target_stop
                # reset target block index, forcing fetching next target info
                target_block_idx = target_slice = None

            # if this is a 1D block we can rely on drop_block Boolean and parts list to determine action
            if b.ndim != 1 and 0 < part_start_last < b.shape[1]:
                # if a 2D block, and part_start_last is less than the shape, collect the remaining slice
                parts.append(b[:, slice(part_start_last, None)])

            # for row deletions, we use np.delete, which handles finding the inverse of a slice correctly; the returned array requires writeability re-set; np.delete does not work correctly with Boolean selectors
            if not drop_block and not parts:
                if row_key is not None:
                    b = np.delete(b, row_key, axis=0)
                    b.flags.writeable = False
                yield b
            elif parts:
                if row_key is not None:
                    for part in parts:
                        part = np.delete(part, row_key, axis=0)
                        part.flags.writeable = False
                        yield part
                else:
                    yield from parts


    def _shift_blocks(self,
            row_shift: int = 0,
            column_shift: int = 0,
            wrap: bool = True,
            fill_value: object = np.nan
            ) -> tp.Iterator[np.ndarray]:
        '''
        Shift type blocks independently on rows or columns. When ``wrap`` is True, the operation is a roll-style shift; when ``wrap`` is False, shifted-out values are not replaced and are filled with ``fill_value``.
        '''
        row_count, column_count = self._shape

        # new start index is the opposite of the shift; if shifting by 2, the new start is the second from the end
        index_start_pos = -(column_shift % column_count)
        row_start_pos = -(row_shift % row_count)

        # possibly be truthy
        # index is columns here
        if wrap and index_start_pos == 0 and row_start_pos == 0:
            yield from self._blocks
        elif not wrap and column_shift == 0 and row_shift == 0:
            yield from self._blocks
        else:
            block_start_idx, block_start_column = self._index[index_start_pos]
            block_start = self._blocks[block_start_idx]

            if block_start_column == 0:
                # we are starting at the block, no tail, always yield;  captures all 1 dim block cases
                block_head_iter: tp.Iterable[np.ndarray] = chain(
                        (block_start,),
                        self._blocks[block_start_idx + 1:])
                block_tail_iter: tp.Iterable[np.ndarray] = self._blocks[:block_start_idx]
            else:
                block_head_iter = chain(
                        (block_start[:, block_start_column:],),
                        self._blocks[block_start_idx + 1:])
                block_tail_iter = chain(
                        self._blocks[:block_start_idx],
                        (block_start[:, :block_start_column],)
                        )

            if not wrap:
                shape = (self._shape[0], min(self._shape[1], abs(column_shift)))
                empty = np.full(shape, fill_value)
                if column_shift > 0:
                    block_head_iter = (empty,)
                elif column_shift < 0:
                    block_tail_iter = (empty,)

            # NOTE: might consider not rolling when yielding an empty array
            for b in chain(block_head_iter, block_tail_iter):
                if (wrap and row_start_pos == 0) or (not wrap and row_shift == 0):
                    yield b
                else:
                    b = array_shift(
                            array=b,
                            shift=row_shift,
                            axis=0,
                            wrap=wrap,
                            fill_value=fill_value)
                    b.flags.writeable = False
                    yield b

    #---------------------------------------------------------------------------
    def _assign_blocks_from_keys_by_blocks(self,
            values: tp.Iterable[np.ndarray],
            row_key: tp.Optional[GetItemKeyTypeCompound] = None,
            column_key: tp.Optional[GetItemKeyTypeCompound] = None,
            ) -> tp.Iterator[np.ndarray]:
        '''
        Given row, column key selections, assign from an iterable of 1D or 2D block arrays.
        '''
        # see clip().get_block_match() for one example of drawing values from another sequence of blocks, where we take blocks and slices from blocks using a list as a stack
        target_block_slices = self._key_to_block_slices(
                column_key,
                retain_key_order=True)
        target_key: tp.Optional[tp.Union[int, slice]] = None
        target_block_idx: tp.Optional[int] = None
        targets_remain: bool = True
        target_is_slice: bool
        row_key_is_null_slice = row_key is None or (
                isinstance(row_key, slice) and row_key == NULL_SLICE)

        # get a mutable list in reverse order for pop/pushing
        values_source = list(reversed(values)) #type: ignore

        def get_block_match(
                width: int
                ) -> tp.Iterator[np.ndarray]:
            '''Draw from values to provide as many columns as specified by width.
            '''
            if width == 1: # no loop necessary
                v = values_source.pop()
                if v.ndim == 1:
                    yield v
                else: # ndim == 2
                    if v.shape[1] > 1: # more than one column
                        # restore remained to values source
                        values_source.append(v[NULL_SLICE, 1:])
                    yield v[NULL_SLICE, 0]
            else:
                width_found = 0
                while width_found < width:
                    v = values_source.pop()
                    if v.ndim == 1:
                        yield v
                        width_found += 1
                        continue
                    # ndim == 2
                    width_v = v.shape[1]
                    width_needed = width - width_found
                    if width_v <= width_needed:
                        yield v
                        width_found += width_v
                        continue
                    # width_v > width_needed
                    values_source.append(v[NULL_SLICE, width_needed:])
                    yield v[NULL_SLICE, :width_needed]
                    break

        for block_idx, b in enumerate(self._blocks):
            assigned_stop = 0 # exclusive maximum

            while targets_remain:
                if target_block_idx is None:
                    try:
                        target_block_idx, target_key = next(target_block_slices)
                    except StopIteration:
                        targets_remain = False # stop entering while loop
                        break
                    target_is_slice = isinstance(target_key, slice)

                if block_idx != target_block_idx:
                    break # need to advance blocks, keep targets

                if target_is_slice:
                    t_start = target_key.start #type: ignore
                    t_stop = target_key.stop #type: ignore
                    t_width = t_stop - t_start
                else:
                    t_start = target_key
                    t_stop = t_start + 1
                    t_width = 1

                # yield all block slices up to the target, then then the target; remain components of this block will be yielded on next iteration (if there is another target) or out of while by looking at assigned stop
                if t_start != 0:
                    yield b[NULL_SLICE, assigned_stop: t_start]
                if row_key_is_null_slice:
                    yield from get_block_match(t_width)
                else:
                    assigned_blocks = tuple(
                            column_2d_filter(a) for a in get_block_match(t_width))
                    assigned_dtype = resolve_dtype_iter(
                            chain((a.dtype for a in assigned_blocks), (b.dtype,)))
                    if b.ndim == 2:
                        assigned = b[NULL_SLICE, t_start: t_stop].astype(assigned_dtype)
                        assigned[row_key, NULL_SLICE] = concat_resolved(assigned_blocks, axis=1)
                    else:
                        assigned = b.astype(assigned_dtype)
                        assigned[row_key] = column_1d_filter(assigned_blocks[0])
                    assigned.flags.writeable = False
                    yield assigned

                assigned_stop = t_stop
                target_block_idx = target_key = None # get a new target

            if assigned_stop == 0:
                yield b # no targets were found for this block; or no targets remain
            elif b.ndim == 1 and assigned_stop == 1:
                pass
            elif b.ndim == 2 and assigned_stop < b.shape[1]:
                yield b[NULL_SLICE, assigned_stop:]



    def _assign_blocks_from_keys(self,
            row_key: tp.Optional[GetItemKeyTypeCompound] = None,
            column_key: tp.Optional[GetItemKeyTypeCompound] = None,
            value: object = None
            ) -> tp.Iterator[np.ndarray]:
        '''Assign a single value (a tuple, array, or element) into all blocks, returning blocks of the same size and shape.

        Args:
            column_key: must be sorted in ascending order.
        '''
        value_dtype = dtype_from_element(value)

        # NOTE: this requires column_key to be ordered to work; we cannot use retain_key_order=False, as the passed `value` is ordered by that key
        target_block_slices = self._key_to_block_slices(
                column_key,
                retain_key_order=True)
        target_key: tp.Optional[tp.Union[int, slice]] = None
        target_block_idx: tp.Optional[int] = None
        targets_remain: bool = True
        target_is_slice: bool
        row_key_is_null_slice = row_key is None or (
                isinstance(row_key, slice) and row_key == NULL_SLICE)

        for block_idx, b in enumerate(self._blocks):
            assigned_stop = 0 # exclusive maximum

            while targets_remain:
                if target_block_idx is None: # can be zero
                    try:
                        target_block_idx, target_key = next(target_block_slices)
                    except StopIteration:
                        targets_remain = False # stop entering while loop
                        break
                    target_is_slice = isinstance(target_key, slice)

                if block_idx != target_block_idx:
                    break # need to advance blocks, keep targets

                # at least one target we need to apply in the current block.
                block_is_column = b.ndim == 1 or (b.ndim > 1 and b.shape[1] == 1)
                start: int = target_key if not target_is_slice else target_key.start # type: ignore

                if start > assigned_stop: # yield component from the last assigned position
                    b_component = b[NULL_SLICE, slice(assigned_stop, start)] # keeps writeable=False
                    yield b_component

                # add empty components for the assignment region
                if target_is_slice and not block_is_column:
                    # can assume this slice has no strides
                    t_width = target_key.stop - target_key.start # type: ignore
                    t_shape = (b.shape[0], t_width)
                else: # b.ndim == 1 or target is an integer: get a 1d array
                    t_width = 1
                    t_shape = b.shape[0]

                if row_key_is_null_slice: #will replace entire sub block, can be empty
                    assigned_target = np.empty(t_shape, dtype=value_dtype)
                else: # will need to mix types
                    assigned_dtype = resolve_dtype(value_dtype, b.dtype)
                    if block_is_column:
                        assigned_target_pre = b if b.ndim == 1 else b.reshape(b.shape[0]) # make 1D
                    else:
                        assigned_target_pre = b[NULL_SLICE, target_key]
                    if b.dtype == assigned_dtype:
                        assigned_target = assigned_target_pre.copy()
                    else:
                        assigned_target = assigned_target_pre.astype(assigned_dtype)

                assigned_stop = start + t_width

                # match sliceable, when target_key is a slice (can be an element)
                if (target_is_slice and
                        not isinstance(value, str)
                        and hasattr(value, '__len__')):
                    if block_is_column:
                        v_width = 1
                        # if block is 1D, then we can only take 1 column if we have a 2d value
                        value_piece_column_key: tp.Union[slice, int] = 0
                    else:
                        v_width = len(range(*target_key.indices(b.shape[1]))) # type: ignore
                        # if block id 2D, can take up to v_width from value
                        value_piece_column_key = slice(0, v_width)

                    if isinstance(value, np.ndarray) and value.ndim > 1:
                        value_piece = value[NULL_SLICE, value_piece_column_key]
                        value = value[NULL_SLICE, slice(v_width, None)] # restore for next iter
                    else: # value is 1D array or tuple, assume assigning into a horizontal position
                        value_piece = value[value_piece_column_key] #type: ignore
                        value = value[slice(v_width, None)] #type: ignore
                else: # not sliceable; this can be a single column
                    value_piece = value

                # write `value` into assigned
                row_target = NULL_SLICE if row_key_is_null_slice else row_key
                if assigned_target.ndim == 1:
                    assigned_target[row_target] = value_piece
                else: # we are editing the entire assigned target sub block
                    assigned_target[row_target, NULL_SLICE] = value_piece

                assigned_target.flags.writeable = False
                yield assigned_target
                target_block_idx = target_key = None # get a new target

            if assigned_stop == 0:
                yield b # no targets were found for this block; or no targets remain
            elif b.ndim == 1 and assigned_stop == 1:
                pass
            elif b.ndim == 2 and assigned_stop < b.shape[1]:
                yield b[NULL_SLICE, assigned_stop:]


    #---------------------------------------------------------------------------
    # There are three approaches to setting values from Boolean indicators; the difference between the first two is if the Booleans are given in a single array, or in block-aligned arrays. The third approach uses block-aligned arrays, but values are provided as an iterable of arrays.

    def _assign_blocks_from_boolean_blocks(self,
            targets: tp.Iterable[np.ndarray],
            value: object,
            value_valid: tp.Optional[np.ndarray]
            ) -> tp.Iterator[np.ndarray]:
        '''Assign value (a single element or a matching array) into all blocks based on a Boolean arrays of shape equal to each block in these blocks, yielding blocks of the same size and shape. Value is set where the Boolean is True.

        Args:
            value: Must be a single value or an array
            value_valid: same size Boolean area to be combined with targets
        '''
        if isinstance(value, np.ndarray):
            value_dtype = value.dtype
            is_element = False
            assert value.shape == self.shape
            if value_valid is not None:
                assert value_valid.shape == self.shape
        else: # assumed to be non-string, non-iterable
            value_dtype = dtype_from_element(value)
            # value_dtype = np.array(value).dtype
            is_element = True

        start = 0
        value_slice: tp.Union[int, slice]

        for block, target in zip_longest(self._blocks, targets):
            if block is None or target is None:
                raise RuntimeError('blocks or targets do not align')

            if not is_element:
                if block.ndim == 1:
                    end = start + 1
                    value_slice = start
                else:
                    end = start + block.shape[1]
                    value_slice = slice(start, end)

                # update target to valid values
                if value_valid is not None:
                    value_valid_part = value_valid[NULL_SLICE, value_slice]
                    target &= value_valid_part

                value_part = value[NULL_SLICE, value_slice][target] #type: ignore
                start = end # always update start
            else:
                value_part = value

            # evaluate after updating target
            if not target.any(): # works for ndim 1 and 2
                yield block

            else:
                assigned_dtype = resolve_dtype(value_dtype, block.dtype)
                if block.dtype == assigned_dtype:
                    assigned = block.copy()
                else:
                    assigned = block.astype(assigned_dtype)

                assigned[target] = value_part
                assigned.flags.writeable = False
                yield assigned


    def _assign_blocks_from_boolean_blocks_and_value_arrays(self,
            targets: tp.Iterable[np.ndarray],
            values: tp.Sequence[np.ndarray],
            ) -> tp.Iterator[np.ndarray]:
        '''Assign values (derived from an iterable of arrays) into all blocks based on a Boolean arrays of shape equal to each block in these blocks. This yields blocks of the same size and shape. Value is set where the Boolean is True.

        This approach minimizes type coercion by reducing assigned values to columnar types.

        Args:
            value: Must be a single value or an array
        '''
        start = 0
        for block, target in zip_longest(self._blocks, targets):
            if block is None or target is None:
                raise RuntimeError('blocks or targets do not align')

            if block.ndim == 1:
                end = start + 1
            else:
                end = start + block.shape[1]

            if not target.any(): # works for ndim 1 and 2
                yield block
            else:
                values_for_block = values[start: end] # get 1D array from tuple
                # target and block must be ndim=2
                for i in range(end - start):
                    if block.ndim == 1: # will only do one iteration
                        assert len(values_for_block) == 1
                        target_sub = target
                        block_sub = block
                    else:
                        target_sub = target[:, i]
                        block_sub = block[:, i]

                    if not target_sub.any():
                        yield block_sub
                    else:
                        values_to_assign = values_for_block[i]
                        if target_sub.all():
                            # will be made immutable of not already
                            yield values_to_assign
                        else:
                            assigned_dtype = resolve_dtype(values_to_assign.dtype, block.dtype)
                            if block.dtype == assigned_dtype:
                                assigned = block_sub.copy()
                            else:
                                assigned = block_sub.astype(assigned_dtype)
                            assigned[target_sub] = values_to_assign[target_sub]
                            assigned.flags.writeable = False
                            yield assigned
            start = end # always update start


    def _assign_blocks_from_bloc_key(self,
            bloc_key: np.ndarray,
            value: tp.Any # an array, or element for single assigment
            ) -> tp.Iterator[np.ndarray]:
        '''
        Given an Boolean array of targets, fill targets from value, where value is either a single value or an array. Unlike with _assign_blocks_from_boolean_blocks, this method takes a single block_key.
        '''
        if isinstance(value, np.ndarray):
            value_dtype = value.dtype
            is_element = False
            assert value.shape == self.shape
        else:
            # value_dtype = np.array(value).dtype
            value_dtype = dtype_from_element(value)
            is_element = True

        start = 0
        target_slice: tp.Union[int, slice]

        for block in self._blocks:

            if block.ndim == 1:
                end = start + 1
                target_slice = start
            else:
                end = start + block.shape[1]
                target_slice = slice(start, end)

            target = bloc_key[NULL_SLICE, target_slice]

            if not target.any():
                yield block
            else:
                assigned_dtype = resolve_dtype(value_dtype, block.dtype)
                if block.dtype == assigned_dtype:
                    assigned = block.copy()
                else:
                    assigned = block.astype(assigned_dtype)

                if is_element:
                    assigned[target] = value
                else:
                    assigned[target] = value[NULL_SLICE, target_slice][target]

                assigned.flags.writeable = False
                yield assigned

            start = end # always update start

    #---------------------------------------------------------------------------
    def _slice_blocks(self,
            row_key: tp.Optional[GetItemKeyTypeCompound] = None,
            column_key: tp.Optional[GetItemKeyTypeCompound] = None
            ) -> tp.Iterator[np.ndarray]:
        '''
        Generator of sliced blocks, given row and column key selectors.
        The result is suitable for passing to TypeBlocks constructor.
        '''
        row_key_null = (row_key is None or
                (isinstance(row_key, slice) and row_key == NULL_SLICE))

        single_row = False
        if row_key_null:
            if self._shape[0] == 1:
                # this codition used to only hold if the arg is a null slice; now if None too and shape has one row
                single_row = True
        elif isinstance(row_key, INT_TYPES):
            single_row = True
        elif isinstance(row_key, slice):
            # need to determine if there is only one index returned by range (after getting indices from the slice); do this without creating a list/tuple, or walking through the entire range; get constant time look-up of range length after uses slice.indicies
            if len(range(*row_key.indices(self._shape[0]))) == 1:
                single_row = True
        elif isinstance(row_key, np.ndarray) and row_key.dtype == bool:
            # must check this case before general iterables, below
            if row_key.sum() == 1:
                single_row = True
        elif isinstance(row_key, KEY_ITERABLE_TYPES) and len(row_key) == 1:
            # an iterable of index integers is expected here
            single_row = True

        # convert column_key into a series of block slices; we have to do this as we stride blocks; do not have to convert row_key as can use directly per block slice
        for block_idx, slc in self._key_to_block_slices(column_key): # slow from line profiler
            b = self._blocks[block_idx]
            if b.ndim == 1: # given 1D array, our row key is all we need
                if row_key_null:
                    block_sliced = b
                else:
                    block_sliced = b[row_key] # slow from line profiler
            else: # given 2D, use row key and column slice
                if row_key_null:
                    block_sliced = b[NULL_SLICE, slc]
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
                # NOTE: this is faster than using np.full(1, block_sliced, dtype=dtype)
                block_sliced = np.array((block_sliced,), dtype=b.dtype)

            yield block_sliced


    def _extract_array(self,
            row_key: tp.Optional[GetItemKeyTypeCompound] = None,
            column_key: tp.Optional[GetItemKeyTypeCompound] = None
            ) -> np.ndarray:
        '''Alternative extractor that returns just an ndarray, concatenating blocks as necessary. Used by internal clients that need to process row/column with an array.

        This will be consistent with NumPy as to the dimensionality returned: if a non-multi selection is made, 1D array will be returned.
        '''
        # identifying column_key as integer, then we only access one block, and can return directly without iterating over blocks
        if isinstance(column_key, INT_TYPES):
            block_idx, column = self._index[column_key]
            b = self._blocks[block_idx]
            if b.ndim == 1:
                if row_key is None:
                    return b
                return b[row_key]
            if row_key is None:
                return b[:, column]
            return b[row_key, column]

        # figure out shape from keys so as to not accumulate?
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

        row_dtype = resolve_dtype_iter(b.dtype for b in blocks)
        row_multiple = row_key is None or isinstance(row_key, KEY_MULTIPLE_TYPES)

        return self._blocks_to_array(
                blocks=blocks,
                shape=(rows, columns),
                row_dtype=row_dtype,
                row_multiple=row_multiple)

    def _extract(self,
            row_key: GetItemKeyType = None,
            column_key: GetItemKeyType = None
            ) -> tp.Union['TypeBlocks', np.ndarray]: # but sometimes an element
        '''
        Return a TypeBlocks after performing row and column selection using iloc selection.

        Row and column keys can be:
            integer: single row/column selection
            slices: one or more contiguous selections
            iterable of integers: one or more non-contiguous and/or repeated selections

        Returns:
            TypeBlocks, or a single element if both are coordinates
        '''
        # identifying column_key as integer, then we only access one block, and can return directly without iterating over blocks
        if isinstance(column_key, INT_TYPES):
            block_idx, column = self._index[column_key]
            b = self._blocks[block_idx]
            row_key_null = (row_key is None or
                    (isinstance(row_key, slice)
                    and row_key == NULL_SLICE))
            if b.ndim == 1:
                if row_key_null: # return a column
                    return TypeBlocks.from_blocks(b)
                elif isinstance(row_key, INT_TYPES):
                    return b[row_key] # return single item
                return TypeBlocks.from_blocks(b[row_key])

            if row_key_null:
                return TypeBlocks.from_blocks(b[:, column])
            elif isinstance(row_key, INT_TYPES):
                return b[row_key, column]
            return TypeBlocks.from_blocks(b[row_key, column])

        # pass a generator to from_block; will return a TypeBlocks or a single element
        return self.from_blocks(
                self._slice_blocks(
                        row_key=row_key,
                        column_key=column_key),
                shape_reference=self._shape
                )


    def _extract_iloc(self,
            key: GetItemKeyTypeCompound
            ) -> 'TypeBlocks':
        if isinstance(key, tuple):
            return self._extract(*key)
        return self._extract(row_key=key)

    def extract_iloc_mask(self,
            key: GetItemKeyTypeCompound
            ) -> 'TypeBlocks':
        if isinstance(key, tuple):
            return TypeBlocks.from_blocks(self._mask_blocks(*key))
        return TypeBlocks.from_blocks(self._mask_blocks(row_key=key))

    def extract_iloc_assign(self,
            key: GetItemKeyTypeCompound,
            value: object,
            value_is_blocks: bool = False,
            ) -> 'TypeBlocks':
        if isinstance(key, tuple):
            row_key, column_key = key
        else:
            row_key = key
            column_key = None

        if value_is_blocks:
            return TypeBlocks.from_blocks(self._assign_blocks_from_keys_by_blocks(
                    row_key=row_key,
                    column_key=column_key,
                    values=value, #type: ignore
                    ))
        return TypeBlocks.from_blocks(self._assign_blocks_from_keys(
                row_key=row_key,
                column_key=column_key,
                value=value))

    def extract_bloc_assign(self,
            key: np.ndarray,
            value: tp.Any
            ) -> 'TypeBlocks':
        return TypeBlocks.from_blocks(self._assign_blocks_from_bloc_key(
                bloc_key=key,
                value=value
                ))


    def drop(self, key: GetItemKeyTypeCompound) -> 'TypeBlocks':
        '''
        Drop rows or columns from a TyepBlocks instance.

        Args:
            key: if a single value, treated as a row key; if a tuple, treated as a pair of row, column keys.
        '''
        if isinstance(key, tuple):
            # column dropping can leed to a TB with generator that yields nothing;
            return TypeBlocks.from_blocks(
                    self._drop_blocks(*key),
                    shape_reference=self._shape
                    )
        return TypeBlocks.from_blocks(
                self._drop_blocks(row_key=key),
                shape_reference=self._shape
                )


    def __getitem__(self, key: GetItemKeyTypeCompound) -> 'TypeBlocks':
        '''
        Returns a column, or a column slice.
        '''
        # NOTE: if key is a tuple it means that multiple indices are being provided
        if isinstance(key, tuple):
            raise KeyError('__getitem__ does not support multiple indexers')
        return self._extract(row_key=None, column_key=key)

    #---------------------------------------------------------------------------
    # operators

    def _ufunc_unary_operator(self, operator: tp.Callable[[np.ndarray], np.ndarray]) -> 'TypeBlocks':
        # for now, do no reblocking; though, in many cases, operating on a unified block will be faster
        def operation() -> tp.Iterator[np.ndarray]:
            for b in self._blocks:
                result = operator(b)
                result.flags.writeable = False
                yield result

        return self.from_blocks(operation())

    #---------------------------------------------------------------------------

    def _block_shape_slices(self) -> tp.Iterator[slice]:
        '''Generator of slices necessary to slice a 1d array of length equal to the number of columns into a length suitable for each block.
        '''
        start = 0
        for b in self._blocks:
            end = start + (1 if b.ndim == 1 else b.shape[1])
            yield slice(start, end)
            start = end

    def _ufunc_binary_operator(self, *,
            operator: tp.Callable[[np.ndarray, np.ndarray], np.ndarray],
            other: tp.Iterable[tp.Any],
            axis: int = 0,
            ) -> 'TypeBlocks':
        '''Axis is only relevant in the application of a 1D array to a 2D TypeBlocks, where axis 0 (the default) will apply the array per row, while axis 1 will apply the array per column.
        '''
        self_operands: tp.Iterable[np.ndarray]
        other_operands: tp.Iterable[np.ndarray]

        if operator.__name__ == 'matmul' or operator.__name__ == 'rmatmul':
            # this could be implemented but would force block consolidation
            raise NotImplementedError('matrix multiplication not supported')

        columnar = False

        if isinstance(other, TypeBlocks):
            apply_column_2d_filter = True

            if self.block_compatible(other, axis=None):
                # this means that the blocks are the same shape; we do not check types
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
        else: # process other as an array
            self_operands = self._blocks
            if not isinstance(other, np.ndarray):
                other = iterable_to_array_nd(other)

            # handle dimensions
            if other.ndim == 0 or (other.ndim == 1 and len(other) == 1):
                # a scalar: reference same value for each block position
                apply_column_2d_filter = False
                other_operands = (other for _ in range(len(self._blocks)))
            elif other.ndim == 1:
                if axis == 0 and len(other) == self._shape[1]:
                    # 1d array applied to the rows: chop to block width
                    apply_column_2d_filter = False
                    other_operands = (other[s] for s in self._block_shape_slices())
                elif axis == 1 and len(other) == self._shape[0]:
                    columnar = True
                else:
                    raise NotImplementedError(f'cannot apply binary operators with a 1D array along axis {axis}: {self._shape}, {other.shape}.')
            elif other.ndim == 2 and other.shape == self._shape:
                apply_column_2d_filter = True
                other_operands = (other[NULL_SLICE, s] for s in self._block_shape_slices())
            else:
                raise NotImplementedError(f'cannot apply binary operators to arrays without alignable shapes: {self._shape}, {other.shape}.')

        if columnar:
            return self.from_blocks(apply_binary_operator_blocks_columnar(
                    values=self_operands,
                    other=other,
                    operator=operator,
                    ))

        return self.from_blocks(apply_binary_operator_blocks(
                values=self_operands,
                other=other_operands,
                operator=operator,
                apply_column_2d_filter=apply_column_2d_filter,
                ))

    #---------------------------------------------------------------------------
    # transformations resulting in the same dimensionality

    def isin(self, other: tp.Any) -> 'TypeBlocks':
        '''Return a new Boolean TypeBlocks that returns True if an element is in `other`.
        '''
        if hasattr(other, '__len__') and len(other) == 0:
            array = np.full(self._shape, False, dtype=bool)
            array.flags.writeable = False
            return self.from_blocks(array)

        other, other_is_unique = iterable_to_array_1d(other)

        def blocks() -> tp.Iterator[np.ndarray]:
            for b in self._blocks:
                # yields immutable arrays
                yield isin_array(array=b,
                        array_is_unique=False, # expensive to determine
                        other=other,
                        other_is_unique=other_is_unique,
                        )

        return self.from_blocks(blocks())


    def transpose(self) -> 'TypeBlocks':
        '''Return a new TypeBlocks that transposes and concatenates all blocks.
        '''
        blocks = []
        for b in self._blocks:
            b = column_2d_filter(b).transpose()
            if b.dtype != self._row_dtype:
                b = b.astype(self._row_dtype)
            blocks.append(b)

        array = np.concatenate(blocks)
        array.flags.writeable = False # keep this array
        return self.from_blocks(array)


    def isna(self, include_none: bool = True) -> 'TypeBlocks':
        '''Return a Boolean TypeBlocks where True is NaN or None.
        '''
        def blocks() -> tp.Iterator[np.ndarray]:
            for b in self._blocks:
                bool_block = isna_array(b, include_none)
                bool_block.flags.writeable = False
                yield bool_block

        return self.from_blocks(blocks())


    def notna(self, include_none: bool = True) -> 'TypeBlocks':
        '''Return a Boolean TypeBlocks where True is not NaN or None.
        '''
        def blocks() -> tp.Iterator[np.ndarray]:
            for b in self._blocks:
                bool_block = np.logical_not(isna_array(b, include_none))
                bool_block.flags.writeable = False
                yield bool_block

        return self.from_blocks(blocks())


    def clip(self,
            lower: tp.Union[None, float, tp.Iterable[np.ndarray]],
            upper: tp.Union[None, float, tp.Iterable[np.ndarray]],
            ) -> 'TypeBlocks':
        '''
        Apply clip to blocks. If clipping is not supported for a dtype, we will raise instead of silently returning the block unchanged.

        Args:
            lower, upper: a float, or iterable of array of aggregate shape equal to that of this TypeBlocks
        '''
        lower_is_element = not hasattr(lower, '__len__')
        upper_is_element = not hasattr(upper, '__len__')

        lower_is_array = isinstance(lower, np.ndarray)
        upper_is_array = isinstance(upper, np.ndarray)

        # get a mutable list in reverse order for pop/pushing
        if lower_is_element or lower_is_array:
            lower_source = lower
        else:
            lower_source = list(reversed(lower)) #type: ignore

        if upper_is_element or upper_is_array:
            upper_source = upper
        else:
            upper_source = list(reversed(upper)) #type: ignore

        def get_block_match(
                start: int, # relative to total size
                end: int, # exclusive
                ndim: int,
                source: tp.Union[None, float, np.ndarray, tp.List[np.ndarray]],
                is_element: bool,
                is_array: bool,
                ) -> np.ndarray:
            '''
            Handle extraction of clip boundaries from multiple different types of sources. NOTE: ndim is the target ndim, and is only relevant when width is 1
            '''
            if is_element:
                return source

            width_target = end - start # 1 is lowest value

            if is_array: # if we have a homogenous 2D array
                block = source[NULL_SLICE, start:end] # type: ignore
                func = column_1d_filter if ndim == 1 else column_2d_filter
                return func(block)

            assert isinstance(source, list)
            block = source.pop()
            width = shape_filter(block)[1]

            if width_target == 1:
                if width > 1: # 2d array with more than one column
                    source.append(block[NULL_SLICE, 1:])
                    block = block[NULL_SLICE, 0]
                func = column_1d_filter if ndim == 1 else column_2d_filter
                return func(block)

            # width_target is > 1
            if width == width_target:
                return block
            elif width > width_target:
                source.append(block[NULL_SLICE, width_target:])
                return block[NULL_SLICE, :width_target]

            # width < width_target, accumulate multiple blocks
            parts = [column_2d_filter(block)]
            while width < width_target:
                block = column_2d_filter(source.pop())
                width += block.shape[1]
                if width == width_target:
                    parts.append(block)
                    return concat_resolved(parts, axis=1)
                if width > width_target:
                    diff = width - width_target
                    trim = block.shape[1] - diff
                    parts.append(block[NULL_SLICE, :trim])
                    source.append(block[NULL_SLICE, trim:])
                    return concat_resolved(parts, axis=1)
                # width < width_target
                parts.append(block)

        def blocks() -> tp.Iterator[np.ndarray]:
            start = end = 0
            for b in self._blocks:
                end += shape_filter(b)[1]
                lb = get_block_match(
                        start,
                        end,
                        b.ndim,
                        lower_source,
                        is_element=lower_is_element,
                        is_array=lower_is_array,
                        )
                ub = get_block_match(
                        start,
                        end,
                        b.ndim,
                        upper_source,
                        is_element=upper_is_element,
                        is_array=upper_is_array,
                        )
                yield np.clip(b, lb, ub)
                start = end

        return self.from_blocks(blocks())

    #---------------------------------------------------------------------------
    # fillna sided

    @staticmethod
    def _fillna_sided_axis_0(
            blocks: tp.Iterable[np.ndarray],
            value: tp.Any,
            sided_leading: bool) -> tp.Iterator[np.ndarray]:
        '''Return a TypeBlocks where NaN or None are replaced in sided (leading or trailing) segments along axis 0, meaning vertically.

        Args:
            sided_leading: True sets the side to fill is the leading side; False sets the side to fill to the trailiing side.

        '''
        if isinstance(value, np.ndarray):
            raise RuntimeError('cannot assign an array to fillna')

        sided_index = 0 if sided_leading else -1

        # store flag for when non longer need to check blocks, yield immediately

        for b in blocks:
            sel = isna_array(b) # True for is NaN
            ndim = sel.ndim

            if ndim == 1 and not sel[sided_index]:
                # if last value (bottom row) is not NaN, we can return block
                yield b
            elif ndim > 1 and ~sel[sided_index].any(): # if not any are NaN
                # can use this last-row observation below
                yield b
            else:
                assignable_dtype = resolve_dtype(
                        dtype_from_element(value),
                        b.dtype)
                if b.dtype == assignable_dtype:
                    assigned = b.copy()
                else:
                    assigned = b.astype(assignable_dtype)

                # because np.nonzero is easier / faster to parse if applied on a 1D array, w can make 2d look like 1D here
                if ndim == 1:
                    sel_nonzeros = ((0, sel),)
                else:
                    # only collect columns for sided NaNs
                    sel_nonzeros = ((i, sel[:, i]) for i, j in enumerate(sel[sided_index]) if j) #type: ignore

                for idx, sel_nonzero in sel_nonzeros:
                    # indices of not-nan values, per column
                    targets = np.nonzero(~sel_nonzero)[0]
                    if len(targets):
                        if sided_leading:
                            sel_slice = slice(0, targets[0])
                        else: # trailings
                            sel_slice = slice(targets[-1]+1, None)
                    else: # all are NaN
                        sel_slice = NULL_SLICE

                    if ndim == 1:
                        assigned[sel_slice] = value
                    else:
                        assigned[sel_slice, idx] = value

                # done writing
                assigned.flags.writeable = False
                yield assigned


    @staticmethod
    def _fillna_sided_axis_1(
            blocks: tp.Iterable[np.ndarray],
            value: tp.Any,
            sided_leading: bool) -> tp.Iterator[np.ndarray]:
        '''Return a TypeBlocks where NaN or None are replaced in sided (leading or trailing) segments along axis 1.

        NOTE: blocks are generated in reverse order when sided_leading is False.

        Args:
            sided_leading: True sets the side to fill is the leading side; False sets the side to fill to the trailing side.

        '''
        if isinstance(value, np.ndarray):
            raise RuntimeError('cannot assign an array to fillna')

        sided_index = 0 if sided_leading else -1

        # will need to re-reverse blocks coming out of this
        block_iter = blocks if sided_leading else reversed(blocks) #type: ignore

        isna_exit_previous = None

        # iterate over blocks to observe NaNs contiguous horizontally
        for b in block_iter:
            sel = isna_array(b) # True for is NaN
            ndim = sel.ndim

            if isna_exit_previous is None:
                # for first block, model as all True
                isna_exit_previous = np.full(sel.shape[0], True, dtype=bool)

            # to continue nan propagation, the exit previous must be NaN, as well as this start
            if ndim == 1:
                isna_entry = sel & isna_exit_previous
            else:
                isna_entry = sel[:, sided_index] & isna_exit_previous

            if not isna_entry.any():
                yield b
            else:
                assignable_dtype = resolve_dtype(
                        dtype_from_element(value),
                        b.dtype)
                if b.dtype == assignable_dtype:
                    assigned = b.copy()
                else:
                    assigned = b.astype(assignable_dtype)

                if ndim == 1:
                    # if one dim, we simply fill nan values
                    assigned[isna_entry] = value
                else:
                    # only collect rows that have a sided NaN
                    # could use np.nonzero()
                    candidates = (i for i, j in enumerate(isna_entry) if j)
                    sels_nonzero = ((i, sel[i]) for i in candidates)

                    for idx, sel_nonzero in sels_nonzero:
                        # indices of not-nan values, per row
                        targets = np.nonzero(~sel_nonzero)[0]
                        if len(targets):
                            if sided_leading:
                                sel_slice = slice(0, targets[0])
                            else: # trailing
                                sel_slice = slice(targets[-1]+1, None)
                        else: # all are NaN
                            sel_slice = NULL_SLICE

                        if ndim == 1:
                            assigned[sel_slice] = value
                        else:
                            assigned[idx, sel_slice] = value

                assigned.flags.writeable = False
                yield assigned

            # always execute these lines after each yield
            # return True for next block only if all values are NaN in the row
            if ndim == 1:
                isna_exit_previous = isna_entry
            else:
                isna_exit_previous = sel.all(axis=1) & isna_exit_previous


    def fillna_leading(self,
            value: tp.Any,
            *,
            axis: int = 0) -> 'TypeBlocks':
        '''Return a TypeBlocks instance replacing leading values with the passed `value`. Leading, axis 0 fills columns, going from top to bottom. Leading axis 1 fills rows, going from left to right.
        '''
        if axis == 0:
            return self.from_blocks(self._fillna_sided_axis_0(
                    blocks=self._blocks,
                    value=value,
                    sided_leading=True))
        elif axis == 1:
            return self.from_blocks(self._fillna_sided_axis_1(
                    blocks=self._blocks,
                    value=value,
                    sided_leading=True))
        raise NotImplementedError(f'no support for axis {axis}')

    def fillna_trailing(self,
            value: tp.Any,
            *,
            axis: int = 0) -> 'TypeBlocks':
        '''Return a TypeBlocks instance replacing trailing NaNs with the passed `value`. Trailing, axis 0 fills columns, going from bottom to top. Trailing axis 1 fills rows, going from right to left.
        '''
        if axis == 0:
            return self.from_blocks(self._fillna_sided_axis_0(
                    blocks=self._blocks,
                    value=value,
                    sided_leading=False))
        elif axis == 1:
            # must reverse when not leading
            blocks = reversed(tuple(self._fillna_sided_axis_1(
                    blocks=self._blocks,
                    value=value,
                    sided_leading=False)))
            return self.from_blocks(blocks)

        raise NotImplementedError(f'no support for axis {axis}')

    #---------------------------------------------------------------------------
    # fillna directional

    @staticmethod
    def _fillna_directional_axis_0(
            blocks: tp.Iterable[np.ndarray],
            directional_forward: bool,
            limit: int = 0
            ) -> tp.Iterator[np.ndarray]:
        '''
        Do a directional fill along axis 0, meaning filling vertically, going top/bottom or bottom/top.

        Args:
            directional_forward: if True, start from the forward (top or left) side.
        '''

        for b in blocks:
            sel = isna_array(b) # True for is NaN
            ndim = sel.ndim

            if ndim == 1 and not np.any(sel):
                yield b
            elif ndim == 2 and not np.any(sel).any():
                yield b
            else:
                target_indexes = binary_transition(sel)

                if ndim == 1:
                    # make single array look like iterable of tuples
                    slots = 1
                    length = len(sel)

                elif ndim == 2:
                    slots = b.shape[1] # axis 0 has column width
                    length = b.shape[0]

                # type is already compatible, no need for check
                assigned = b.copy()

                for i in range(slots):

                    if ndim == 1:
                        target_index = target_indexes
                        if not len(target_index):
                            continue
                        target_values = b[target_index]

                        def slice_condition(target_slice: slice) -> bool:
                            # NOTE: start is never None
                            return sel[target_slice.start] # type: ignore

                    else: # 2D blocks
                        target_index = target_indexes[i]
                        if not target_index:
                            continue
                        target_values = b[target_index, i]

                        def slice_condition(target_slice: slice) -> bool:
                            # NOTE: start is never None
                            return sel[target_slice.start, i] # type: ignore

                    for target_slice, value in slices_from_targets(
                            target_index=target_index,
                            target_values=target_values,
                            length=length,
                            directional_forward=directional_forward,
                            limit=limit,
                            slice_condition=slice_condition
                            ):

                        if ndim == 1:
                            assigned[target_slice] = value
                        else:
                            assigned[target_slice, i] = value

                assigned.flags.writeable = False
                yield assigned



    @staticmethod
    def _fillna_directional_axis_1(
            blocks: tp.Iterable[np.ndarray],
            directional_forward: bool,
            limit: int = 0
            ) -> tp.Iterator[np.ndarray]:
        '''
        Do a directional fill along axis 1, or horizontally, going left to right or right to left.

        NOTE: blocks are generated in reverse order when directional_forward is False.

        '''
        bridge_src_index = -1 if directional_forward else 0
        bridge_dst_index = 0 if directional_forward else -1

        # will need to re-reverse blocks coming out of this
        block_iter = blocks if directional_forward else reversed(blocks) # type: ignore

        bridging_values: tp.Optional[np.ndarray] = None
        bridging_count: tp.Optional[np.ndarray] = None
        bridging_isna: tp.Optional[np.ndarray] = None # Boolean array describing isna of bridging values

        for b in block_iter:
            sel = isna_array(b) # True for is NaN
            ndim = sel.ndim

            if ndim == 1 and not np.any(sel):
                bridging_values = b
                bridging_isna = sel
                bridging_count = np.full(b.shape[0], 0)
                yield b
            elif ndim == 2 and not np.any(sel).any():
                bridging_values = b[:, bridge_src_index]
                bridging_isna = sel[:, bridge_src_index]
                bridging_count = np.full(b.shape[0], 0)
                yield b
            else: # some NA in this block
                if bridging_values is None:
                    assigned = b.copy()
                    bridging_count = np.full(b.shape[0], 0)
                else:
                    assignable_dtype = resolve_dtype(bridging_values.dtype, b.dtype)
                    assigned = b.astype(assignable_dtype)

                if ndim == 1:
                    # a single array has either NaN or non-NaN values; will only fill in NaN if we have a caried value from the previous block
                    if bridging_values is not None: # sel has at least one NaN
                        bridging_isnotna = ~bridging_isna # type: ignore #pylint: disable=E1130

                        sel_sided = sel & bridging_isnotna
                        if limit:
                            # set to false those values where bridging already at limit
                            sel_sided[bridging_count >= limit] = False # type: ignore

                        # set values in assigned if there is a NaN here (sel_sided) and we are not beyond the count
                        assigned[sel_sided] = bridging_values[sel_sided] #pylint: disable=E1136
                        # only increment positions that are NaN here and have not-nan bridging values
                        sel_count_increment = sel & bridging_isnotna
                        bridging_count[sel_count_increment] += 1 # type: ignore
                        # set unassigned to zero
                        bridging_count[~sel_count_increment] = 0 # type: ignore
                    else:
                        bridging_count = np.full(b.shape[0], 0)

                    bridging_values = assigned
                    bridging_isna = isna_array(bridging_values) # must reevaluate if assigned

                elif ndim == 2:

                    slots = b.shape[0] # axis 0 has column width
                    length = b.shape[1]

                    # set to True when can reset count to zero; this is always the case if the bridge src value is not NaN (before we do any filling)
                    bridging_count_reset = ~sel[:, bridge_src_index]

                    if bridging_values is not None:
                        bridging_isnotna = ~bridging_isna #type: ignore #pylint: disable=E1130

                        # find leading NaNs segments if they exist, and if there is as corrresponding non-nan value to bridge
                        isna_entry = sel[:, bridge_dst_index] & bridging_isnotna
                        # get a row of Booleans for plausible candidates
                        candidates = (i for i, j in enumerate(isna_entry) if j)
                        sels_nonzero = ((i, sel[i]) for i in candidates)

                        # get appropriate leading slice to cover nan region
                        for idx, sel_nonzero in sels_nonzero:
                            # indices of not-nan values, per row
                            targets = np.nonzero(~sel_nonzero)[0]
                            if len(targets):
                                if directional_forward:
                                    sel_slice = slice(0, targets[0])
                                else: # backward
                                    sel_slice = slice(targets[-1]+1, length)
                            else: # all are NaN
                                sel_slice = slice(0, length)

                            # truncate sel_slice by limit-
                            sided_len = len(range(*sel_slice.indices(length)))

                            if limit and bridging_count[idx] >= limit: # type: ignore #pylint: disable=R1724
                                # if already at limit, do not assign
                                bridging_count[idx] += sided_len # type: ignore
                                continue
                            elif limit and (bridging_count[idx] + sided_len) >= limit: # type: ignore
                                # trim slice to fit
                                shift = bridging_count[idx] + sided_len - limit # type: ignore
                                # shift should only be positive only here
                                if directional_forward:
                                    sel_slice = slice(
                                            sel_slice.start,
                                            sel_slice.stop - shift)
                                else:
                                    sel_slice = slice(
                                            sel_slice.start + shift,
                                            sel_slice.stop)

                            # update with full length or limited length?
                            bridging_count[idx] += sided_len # type: ignore
                            assigned[idx, sel_slice] = bridging_values[idx] #pylint: disable=E1136

                    # handle each row (going horizontally) in isolation
                    target_indexes = binary_transition(sel, axis=1)
                    for i in range(slots):

                        target_index = target_indexes[i]
                        if target_index is None:
                            # found no transitions, so either all NaN or all not NaN; if all NaN, might have been filled in bridging; if had values, will aready identify as bridging_count_reset[i] == True
                            continue

                        target_values = b[i, target_index]

                        def slice_condition(target_slice: slice) -> bool:
                            # NOTE: start is never None
                            return sel[i, target_slice.start] # type: ignore

                        target_slice = None
                        for target_slice, value in slices_from_targets(
                                target_index=target_index,
                                target_values=target_values,
                                length=length,
                                directional_forward=directional_forward,
                                limit=limit,
                                slice_condition=slice_condition
                                ):
                            assigned[i, target_slice] = value

                        # update counts from the last slice; this will have already been limited if necessary, but need to reflext contiguous values going into the next block; if slices does not go to edge; will identify as needing as reset
                        if target_slice is not None:
                            bridging_count[i] = len(range(*target_slice.indices(length))) # type: ignore

                    bridging_values = assigned[:, bridge_src_index]
                    bridging_isna = isna_array(bridging_values) # must reevaluate if assigned

                    # if the birdging values is NaN now, it could not be filled, or was not filled enough, and thus does not continue a count; can set to zero
                    bridging_count_reset |= bridging_isna
                    bridging_count[bridging_count_reset] = 0 # type: ignore

                assigned.flags.writeable = False
                yield assigned


    def fillna_forward(self,
            limit: int = 0,
            *,
            axis: int = 0) -> 'TypeBlocks':
        '''Return a new ``TypeBlocks`` after feeding forward the last non-null (NaN or None) observation across contiguous nulls. Forward axis 0 fills columns, going from top to bottom. Forward axis 1 fills rows, going from left to right.
        '''
        if axis == 0:
            return self.from_blocks(self._fillna_directional_axis_0(
                    blocks=self._blocks,
                    directional_forward=True,
                    limit=limit
                    ))
        elif axis == 1:
            return self.from_blocks(self._fillna_directional_axis_1(
                    blocks=self._blocks,
                    directional_forward=True,
                    limit=limit
                    ))

        raise AxisInvalid(f'no support for axis {axis}')


    def fillna_backward(self,
            limit: int = 0,
            *,
            axis: int = 0) -> 'TypeBlocks':
        '''Return a new ``TypeBlocks`` after feeding backward the last non-null (NaN or None) observation across contiguous nulls. Backward, axis 0 fills columns, going from bottom to top. Backward axis 1 fills rows, going from right to left.
        '''
        if axis == 0:
            return self.from_blocks(self._fillna_directional_axis_0(
                    blocks=self._blocks,
                    directional_forward=False,
                    limit=limit
                    ))
        elif axis == 1:
            blocks = reversed(tuple(self._fillna_directional_axis_1(
                    blocks=self._blocks,
                    directional_forward=False,
                    limit=limit
                    )))
            return self.from_blocks(blocks)

        raise AxisInvalid(f'no support for axis {axis}')



    #---------------------------------------------------------------------------

    def dropna_to_keep_locations(self,
            axis: int = 0,
            condition: tp.Callable[..., bool] = np.all,
            ) -> tp.Tuple[tp.Optional[np.ndarray], tp.Optional[np.ndarray]]:
        '''
        Return the row and column slices to extract the new TypeBlock. This is to be used by Frame, where the slices will be needed on the indices as well.

        Args:
            axis: Dimension to drop, where 0 will drop rows and 1 will drop columns based on the condition function applied to a Boolean array.
        '''
        # get a unified boolean array; as iisna will always return a Boolean, we can simply take the firtst block out of consolidation
        unified = next(self.consolidate_blocks(isna_array(b) for b in self._blocks))

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


    def fillna(self,
            value: object,
            value_valid: tp.Optional[np.ndarray] = None,
            ) -> 'TypeBlocks':
        '''
        Return a new TypeBlocks instance that fills missing values with the passed value.

        Args:
            value: value to fill missing with; can be an element or a same-sized array.
            value_valid: Optionally provide a same-size array mask of the value setting (useful for carrying forward information from labels).
        '''
        return self.from_blocks(
                self._assign_blocks_from_boolean_blocks(
                        targets=(isna_array(b) for b in self._blocks),
                        value=value,
                        value_valid=value_valid
                        )
                )

    def fillna_by_values(self,
            values: tp.Sequence[np.ndarray],
            ) -> 'TypeBlocks':
        '''
        Return a new TypeBlocks instance that fills missing values with the aligned columnar arrays.

        Args:
            values: iterable of arrays to be aligned as columns.
        '''
        return self.from_blocks(
                self._assign_blocks_from_boolean_blocks_and_value_arrays(
                        targets=(isna_array(b) for b in self._blocks),
                        values=values,
                        )
                )

    @doc_inject()
    def equals(self,
            other: tp.Any,
            *,
            compare_dtype: bool = False,
            compare_class: bool = False,
            skipna: bool = True,
            ) -> bool:
        '''
        {doc} Underlying block structure is not considered in determining equality.

        Args:
            {compare_dtype}
            {compare_class}
            {skipna}
        '''
        if id(other) == id(self):
            return True

        # NOTE: there is only one TypeBlocks class, but better to be consistent
        if compare_class and self.__class__ != other.__class__:
            return False
        elif not isinstance(other, TypeBlocks):
            return False

        # same type from here
        if self._shape != other._shape:
            return False
        if compare_dtype and self._dtypes != other._dtypes: # these are lists
            return False

        # NOTE: TypeBlocks handles array operations that return Boolean
        try:
            eq = self == other # returns a Boolean TypeBlocks instance
        except ValueError:
            # this can happen due to NP returning singel Booleans instaed of arrays
            return False

        if skipna:
            isna_self = self.isna(include_none=False) # returns type blocks
            isna_other = other.isna(include_none=False)
            isna_both = isna_self & isna_self

        start = 0
        end = 0
        for block in eq._blocks:
            # permit short circuiting on iteration
            if skipna:
                end = start + block.shape[1]
                target = isna_both._extract_array(column_key=slice(start, end))
                start = end
                # fill-in NaN values with True
                block.flags.writeable = True
                block[target] = True

            if not block.all():
                return False
        return True

    #---------------------------------------------------------------------------
    # mutate

    def append(self, block: np.ndarray) -> None:
        '''Add a block; an array copy will not be made unless the passed in block is not immutable'''
        # NOTE: shape can be 0, 0 if empty, or any one dimension can be 0. if columns is 0 and rows is non-zero, that row count is binding for appending (though the array need no tbe appended); if columns is > 0 and rows is zero, that row is binding for appending (and the array should be appended).

        row_count = self._shape[0]

        # update shape
        if block.shape[0] != row_count:
            raise RuntimeError(f'appended block shape {block.shape} does not align with shape {self._shape}')

        if block.ndim == 1:
            # length already confirmed to match row count; even if this is a zero length 1D array, we keep it as it (by definition) defines a column (if the existing row_count is zero). said another way, a zero length, 1D array always has a shape of (0, 1)
            block_columns = 1
        else:
            block_columns = block.shape[1]
            if block_columns == 0:
                # do not append 0 width arrays
                return

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
            # we do not use resolve_dtype here as we want to preserve types, not safely cooerce them (i.e., int to float)
            self._row_dtype = DTYPE_OBJECT

    def extend(self,
            other: tp.Union['TypeBlocks', tp.Iterable[np.ndarray]]
            ) -> None:
        '''Extend this TypeBlock with the contents of another TypeBlocks instance, or an iterable of arrays. Note that an iterable of TypeBlocks is not currently supported.
        '''
        if isinstance(other, TypeBlocks):
            if self._shape[0]:
                if self._shape[0] != other._shape[0]:
                    raise RuntimeError('cannot extend unaligned shapes')
            blocks: tp.Iterable[np.ndarray] = other._blocks
        else: # accept iterables of np.arrays
            blocks = other
        # row count must be the same
        for block in blocks:
            self.append(block)
