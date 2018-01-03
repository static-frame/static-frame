import typing as tp

from itertools import chain
from itertools import zip_longest
from itertools import product

from functools import wraps

import operator as operator_mod

import numpy as np
import pandas as pd


from collections import OrderedDict
from collections import defaultdict


_UNARY_OPERATORS = (
        '__pos__',
        '__neg__',
        '__abs__',
        '__invert__')

_BINARY_OPERATORS = (
        '__add__',
        '__sub__',
        '__mul__',
        '__matmul__',
        '__truediv__',
        '__floordiv__',
        '__mod__',
        #'__divmod__', this returns two np.arrays when called on an np array
        '__pow__',
        '__lshift__',
        '__rshift__',
        '__and__',
        '__xor__',
        '__or__',
        '__lt__',
        '__le__',
        '__eq__',
        '__ne__',
        '__gt__',
        '__ge__',
        )

# all reverse are binary
_REVERSE_OPERATOR_MAP = {
        '__radd__': '__add__',
        '__rsub__': '__sub__',
        '__rmul__': '__mul__',
        '__rtruediv__': '__truediv__',
        '__rfloordiv__': '__floordiv__',
        }

class MetaOperatorDelegate(type):
    '''Auto-populate binary and unary methods based on instance methods named `_unary_operator` and `_binary_operator`.
    '''

    @staticmethod
    def create_func(func_name, opperand_count=1, reverse=False):
        # operator module defines alias to funcs with names like __add__, etc
        if not reverse:
            operator_func = getattr(operator_mod, func_name)
            func_wrapper = operator_func
        else:
            unreversed_operator_func = getattr(operator_mod, _REVERSE_OPERATOR_MAP[func_name])
            # flip the order of the arguments
            operator_func = lambda rhs, lhs: unreversed_operator_func(lhs, rhs)
            func_wrapper = unreversed_operator_func

        if opperand_count == 1:
            assert not reverse # cannot reverse a single opperand
            def func(self):
                return self._unary_operator(operator_func)
        elif opperand_count == 2:
            def func(self, other):
                return self._binary_operator(operator=operator_func, other=other)
        else:
            raise NotImplementedError()

        f = wraps(func_wrapper)(func)
        f.__name__ = func_name
        return f

    def __new__(mcs, name, bases, attrs):
        for opperand_count, func_name in chain(
                product((1,), _UNARY_OPERATORS),
                product((2,), _BINARY_OPERATORS)):
            attrs[func_name] = mcs.create_func(func_name, opperand_count=opperand_count)
        for func_name in _REVERSE_OPERATOR_MAP.keys():
            attrs[func_name] = mcs.create_func(func_name, opperand_count=2, reverse=True)

        return type.__new__(mcs, name, bases, attrs)


# for getitem / loc selection
_KEY_ITERABLE_TYPES = (tuple, list, np.ndarray) # TODO: remove tuple
GetItemKeyType = tp.Union[int, slice, tp.Iterable[int]] # how can I use the constant above?

class GetItem:
    __slots__ = ('callback',)

    def __init__(self, callback):
        self.callback = callback

    def __getitem__(self, key: GetItemKeyType):
        return self.callback(key)


#-------------------------------------------------------------------------------
class TypeBlocks(metaclass=MetaOperatorDelegate):
    '''Store data in chunks based on type; reconsolidate contiguous types when necessary; perform as single NP 2D array.
    '''
    # related to Pandas BlockManager
    __slots__ = (
            '_blocks',
            '_dtypes',
            '_index',
            '_shape',
            '_row_dtype',
            'iloc')

    @staticmethod
    def immutable_filter(src_array: np.ndarray) -> np.ndarray:
        '''Pass an immutable array; otherwise, return an immutable copy of the provided array.
        '''
        if src_array.flags.writeable:
            dst_array = src_array.copy()
            dst_array.flags.writeable = False
            return dst_array
        return src_array # keep it as is

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
    # constructor

    @classmethod
    def from_blocks(cls, raw_blocks) -> 'TypeBlocks':
        '''
        The order of the blocks defines the order of the columns contained.

        Args:
            raw_blocks: can be a Generator
        '''
        blocks = [] # ordered blocks
        index = [] # columns position to blocks key
        dtypes = [] # column position to dtype
        block_count = 0

        # if a single block, no need to loop
        if isinstance(raw_blocks, np.ndarray):
            row_count, column_count = cls.shape_filter(raw_blocks)
            blocks.append(cls.immutable_filter(raw_blocks))
            for i in range(column_count):
                index.append((block_count, i))
                dtypes.append(raw_blocks.dtype)

        else: # an iterable of blocks
            row_count = None
            column_count = 0
            for block in raw_blocks:
                assert isinstance(block, np.ndarray), 'found non array block: %s' % block

                r, c = cls.shape_filter(block)
                # check number of rows is the same for all blocks
                if row_count is None:
                    row_count = r
                else:
                    assert r == row_count, 'mismatched row count: %s: %s' % (r, row_count)

                blocks.append(cls.immutable_filter(block))

                # store position to key of block, block columns
                for i in range(c):
                    index.append((block_count, i))
                    dtypes.append(block.dtype)
                column_count += c
                block_count += 1

        assert blocks, 'no blocks defined'

        return cls(blocks=blocks,
                dtypes=dtypes,
                index=index,
                shape=(row_count, column_count),
                )

    #---------------------------------------------------------------------------
    @staticmethod
    def union_dtype(blocks: tp.Iterable[np.ndarray]) -> type:
        '''For an iterable of dtypes, return the dtype common to all, or if not common, then object dtype.

        Returns:
            type
        '''
        dtypes = iter(b.dtype for b in blocks)
        first = next(dtypes)
        for dtype in dtypes:
            if not dtype is first:
                return object
        return first # they are all the same

    def __init__(self, *,
            blocks: tp.Iterable[np.ndarray],
            dtypes: tp.Iterable[np.dtype],
            index: tp.Iterable[tp.Tuple[int, int]],
            shape: tp.Tuple[int, int] # could be derived
            ):
        '''
        Args:
            dtypes: list of dtypes
            index: list of tuple coords to block, column
        '''
        self._blocks = blocks
        self._dtypes = dtypes
        self._index = index # list where index, as column, gets block, offset
        self._shape = shape

        self._row_dtype = self.union_dtype(self._blocks)

        assert len(self._dtypes) == len(self._index) == self._shape[1]
        # store integers for slicing; store in same format as _index
        #self._row_index = tuple((0, x) for x in range(self._shape[0]))

        # set up callbacks
        self.iloc = GetItem(self._extract_iloc)

    #---------------------------------------------------------------------------
    # new propertyies
    @property
    def dtypes(self) -> np.ndarray:
        # this creates a new array every time it is called
        a = np.array(self._dtypes, dtype=self._row_dtype)
        a.flags.writeable = False
        return a

    # consider renaming pointers
    @property
    def mloc(self) -> np.ndarray:
        '''Return an ndarray of NP array memory location integers.
        '''
        a = np.fromiter(
                (b.__array_interface__['data'][0] for b in self._blocks),
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

    @property
    def values(self) -> np.ndarray:
        '''Returns a consolidated NP array of the all blocks.
        '''
        if len(self._blocks) == 1:
            return self._blocks[0]

        # get empty array and fill parts
        # cam we use a structured array?
        if self._shape[0] == 1:
            # get a 1D arrayr based on number of columns
            array = np.empty(self._shape[1], dtype=self._row_dtype)
        else: # get ndim 2 shape array
            array = np.empty(self._shape, dtype=self._row_dtype)

        # can we use a np.concatenate to do this

        pos = 0
        for block in self._blocks:
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

    #---------------------------------------------------------------------------
    # methods for evaluating compatibility with other blocks, and reblocking

    def block_compatible(self, other: 'TypeBlocks') -> bool:
        '''Block compatible means that the blocks are the same shape and the same dtype.
        '''
        for a, b in zip_longest(self._blocks, other._blocks, fillvalue=None):
            if a is None or b is None:
                return False
            if not self.shape_filter(a) == self.shape_filter(b):
                return False
            # this does not show us if the types can be operated on;
            # similarly, np.can_cast, np.result_type do not telll us if an operation will succeede
            # if not a.dtype is b.dtype:
            #     return False
        return True

    # def dtype_compatible(self, other: 'TypeBlocks') -> bool:
    #     '''dtype compatible means that columns are compatible, and thus, if not block compatible, will be after consolidation.
    #     '''
    #     # need to handle cases where types are easily expaned, such as <U1, <U2
    #     return self._dtypes == other._dtypes


    @classmethod
    def _concatenate_blocks(cls, group: tp.Iterable[np.ndarray]):
        return np.concatenate([cls.single_column_filter(x) for x in group], axis=1)

    def _reblock(self) -> tp.Generator[np.ndarray, None, None]:
        '''Generator of new block that consolidate adjacent types that are the same.
        '''
        group_dtype = None # store type found along contiguous blocks
        group = []

        if len(self._blocks) == 1:
            yield self._blocks[0]
        else:
            for block in self._blocks:
                if group_dtype is None: # first block of a type
                    # TODO: can we use can_cast here to determine what we can concatenate? NP will concatenate a single character string with an Boolean, for example, and that is not what we want
                    group_dtype = block.dtype
                    group.append(block)
                    continue

                if block.dtype != group_dtype:
                    # new group found, return stored
                    if len(group) == 1: # return reference without copy
                        yield group[0]
                    else: # combine groups
                        # could pre allocating and assing as necessary for large groups
                        yield self._concatenate_blocks(group)
                    group_dtype = block.dtype
                    group = [block]
                else: # new block has same group dtype
                    group.append(block)

            # get anything leftover
            if group:
                if len(group) == 1:
                    yield group[0]
                else:
                    yield self._concatenate_blocks(group)


    def consolidate(self) -> 'TypeBlocks':
        '''Return a new TypeBlocks that unifies all adjacent types.
        '''
        # note: not sure if we have a single block if we should return a new TypeBlocks instance (as done presently), or simply return self; either way, no new np arrays will be created
        return self.from_blocks(self._reblock())


    #---------------------------------------------------------------------------
    def __len__(self):
        '''Length, as with NumPy and Pandas, is the number of rows.
        '''
        return self._shape[0]

    # def __hash__(self):
    #     '''Not a collision free hash, but so as to be keys in dicts.
    #     '''
    #     return hash(tuple(hash(b) for b in self._blocks))

    def __repr__(self) -> str:

        parts = defaultdict(list)
        block_width = []
        for block in self._blocks:
            w = 0
            # configure array_str as needed
            block = self.single_column_filter(block)
            #if block.ndim == 1:
                #block = np.reshape(block, (block.shape[0], 1))
            line_parts = chain(np.array_str(block).split('\n'), (str(block.dtype),))

            for row, line in enumerate(line_parts):
                w = w if len(line) < w else len(line)
                parts[row].append(line)

            block_width.append(w + 1) # padding of 1

        # order here does not matter
        just = {k: ''.join(
                (part.ljust(block_width[pos]) for pos, part in enumerate(v)))
                for k, v in parts.items()}
        h = '<' + self.__class__.__name__ + '>\n'
        return h + '\n'.join(just[x] for x in range(len(parts)))


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
        if stop_idx > start_idx:
            # ascending indices
            return slice(start_idx, stop_idx + 1)

        if stop_idx == 0:
            return slice(start_idx, None, -1)
        # stop is less than start, need to reduce by 1 to cover range
        return slice(start_idx, stop_idx - 1, -1)


    @classmethod
    def _indices_to_contiguous_pairs(cls, indices) -> tp.Generator:
        '''Indices are pairs of (block_idx, value); convert these to pairs of (block_idx, slice) when we identify contiguous indices.

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


    def _key_to_block_slices(self, key) -> tp.Generator[
                tp.Tuple[int, tp.Union[slice, int]], None, None]:
        '''
        For a column key (an integer, slice, or iterable), generate pairs of (block_idx, slice or integer) to cover all extractions. First, get the relevant index values (pairs of block id, column id), then convert those to contiguous slices.

        Returns:
            A generator iterable of pairs, where values are block index, slice or column index
        '''

        # do type checking on slice v others, as with others we need to sort once iterable of keys
        if isinstance(key, int):
            # the index has the pair block, column integer
            yield self._index[key]
        else:
            if isinstance(key, slice):
                indices = self._index[key] # slice the index
                # already sorted
            elif isinstance(key, _KEY_ITERABLE_TYPES):
                # an iterable of keys, may not have contiguous regions; provide in the order given; set as a generator; this is a list not an np.array, so cannot slice self._index; requires iteration in passed generator anyways so probably this is as fast as it can be.
                indices = (self._index[x] for x in key)
            elif key is None: # get all
                indices = self._index
            else:
                raise NotImplementedError()
            yield from self._indices_to_contiguous_pairs(indices)


    def _slice_blocks(self,
            row_key=None,
            column_key=None) -> tp.Generator[np.ndarray, None, None]:
        '''
        Generator of sliced blocks, given row and column key selectors.
        The result is suitable for pass to TypeBlocks constructor.
        '''
        single_row = False
        if isinstance(row_key, int):
            single_row = True
        elif isinstance(row_key, _KEY_ITERABLE_TYPES) and len(row_key) == 1:
            # an iterable of index integers is expected here
            single_row = True
        elif isinstance(row_key, slice):
            # need to determine if there is only one index returned by range (after getting indices from the slice); do this without creating a list/tuple, or walking through the entire range; get constant time look-up of range length after uses slice.indicies
            if len(range(*row_key.indices(self._shape[0]))) == 1:
                single_row = True

        #print('_slice_blocks: row_key', row_key, 'column_key', column_key, 'single_row', single_row)

        # convert column_key into a series of block slices; we have to do this as we stride blocks; do not have to convert row_key as can use directly per block slice
        for block_idx, slc in self._key_to_block_slices(column_key):
            b = self._blocks[block_idx]
            if b.ndim == 1: # given 1D array, our row key is all we need
                block_sliced = b[row_key]
            else: # given 2D, use row key and column slice
                block_sliced = b[row_key, slc]

            # optionally, apply additoinal selection, reshaping, or adjustments to what we got out of the block
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


    def _extract(self, row_key=None, column_key=None) -> 'TypeBlocks': # but sometimes an element
        '''
        Return a TypeBlocks after performing row and column selection using iloc-style selection.

        Row and column keys can be:
            integer: single row/column selection
            slices: one or more contiguous selections
            iterable of integers: one or more non-contiguous and/or repeated selections

        Note: Boolean-based selection is not (yet?) implemented here, but instead will be implemented at the `loc` level. This might imply that Boolean selection is only available with `loc`. This avoids the stra

        Returns:
            TypeBlocks, or single elemtn if both are coordinats
        '''
        # identifying column_key as integer, then we only access one block, and can return directly without iterating over blocks
        if isinstance(column_key, int):
            block_idx, column = self._index[column_key]
            b = self._blocks[block_idx]
            if b.ndim == 1:
                assert column == 0
                if row_key is None: # return a column
                    return TypeBlocks.from_blocks(b)
                elif isinstance(row_key, int):
                    return b[row_key] # return single item
                else: # extraneous!
                    return TypeBlocks.from_blocks(b[row_key])
            else: # extraneous?
                if row_key is None: # return a column
                    return TypeBlocks.from_blocks(b[:, column])
                elif isinstance(row_key, int):
                    return b[row_key, column] # return single item
                else: # extraneous!
                    return TypeBlocks.from_blocks(b[row_key, column])

        # pass a generator to from_block; will return a TypeBlocks or a single element
        return self.from_blocks(self._slice_blocks(
                row_key=row_key,
                column_key=column_key))


    def _extract_iloc(self, key) -> 'TypeBlocks':
        if self.unified:
            # perform slicing directly on block if possible
            return self.from_blocks(self._blocks[0][key])
        if isinstance(key, tuple):
            return self._extract(*key)
        return self._extract(row_key=key)

    def __getitem__(self, key) -> 'TypeBlocks':
        '''
        Returns a column, or a column slice.
        '''
        # NOTE: if key is a tuple it means that multiple indices are being provided; this should probably raise an error
        if isinstance(key, tuple):
            raise KeyError('__getitem__ does not support multiple indexers')
        return self._extract(row_key=slice(None), column_key=key)


    #---------------------------------------------------------------------------
    # slice replace permits defining a slice and new values for that slice, and returnina new TypeBlocks

    def slice_replace(self, row_key=None, column_key=None, value=None):
        print('slice replace', row_key, column_key, value)



    #---------------------------------------------------------------------------
    # mutate

    def append(self, block: np.ndarray):
        '''Add a block; an array copy will not be made unless the passed in block is not immutable'''
        # handle if shape is empty
        row_count = self._shape[0]

        # update shape
        if block.ndim == 1:
            assert len(block) == row_count, 'mismatched row count'
            block_columns = 1
        else:
            assert block.shape[0] == row_count, 'mismatched row count'
            block_columns = block.shape[1]

        # extend shape
        self._shape = (row_count, self._shape[1] + block_columns)

        # add block, dtypes, index
        block_idx = len(self._blocks) # next block
        for i in range(block_columns):
            self._index.append((block_idx, i))
            self._dtypes.append(block.dtype)

        # make immutable copy if necessary before appending
        self._blocks.append(self.immutable_filter(block))

        # if already aligned, nothing to do
        if block.dtype != self._row_dtype:
            self._row_dtype = object



    def extend(self, other: 'TypeBlocks'):
        '''Extend this TypeBlock with the contents of another.
        '''
        # accept iterables of np.arrays?
        assert isinstance(other, TypeBlocks)
        # row count must be the same
        assert self._shape[0] == other._shape[0]
        for block in other._blocks:
            self.append(block)

    #---------------------------------------------------------------------------
    # operators

    def _unary_operator(self, operator: tp.Callable) -> 'TypeBlocks':
        # for now, do no reblocking; though, in many cases, operating on a unified block will be faster
        return self.from_blocks(operator(b) for b in self._blocks)


    #---------------------------------------------------------------------------

    def _block_shape_slices(self) -> tp.Generator[slice, None, None]:
        '''Generator of slices necessary to slice a 1d array of length equal to the number of columns into a lenght suitable for each block.
        '''
        start = 0
        for b in self._blocks:
            end = start + (1 if b.ndim == 1 else b.shape[1])
            yield slice(start, end)
            start = end

    def _binary_operator(self, *, operator: tp.Callable, other) -> 'TypeBlocks':
        if isinstance(other, TypeBlocks):
            if self.block_compatible(other):
                self_opperands = self._blocks
                other_opperands = other._blocks
            elif self._shape == other._shape:
                # for now, we do something less optimal, and just consolidate both, knowing that they will then be compatible
                self_opperands = self._reblock()
                other_opperands = other._reblock()
            else: # raise same error as NP
                raise NotImplementedError('cannot apply binary operators to arbitrary TypeBlocks')
        else: # process other as an array
            self_opperands = self._blocks
            if not isinstance(other, np.ndarray):
                # this maybe expensive for a single scalar
                other = np.array(other) # this will work with a single scalar too
            # handle dimensions
            if other.ndim == 0 or (other.ndim == 1 and len(other) == 1):
                # a scalar: reference same value for each block position
                other_opperands = (other for _ in range(len(self._blocks)))
            elif other.ndim == 1 and len(other) == self._shape[1]:
                # extract single row from a 2d, one row array
                if other.ndim == 2 and other.shape == (1, self._shape[1]):
                    other = other[0]
                # one dimensional array of same size: chop to block width
                other_opperands = (other[s] for s in self._block_shape_slices())
            else:
                raise NotImplementedError('cannot apply binary operators to arbitrary np arrays yet')

        # this means if operators are not compatible we will get an error with the first block, rather than after converting all operands
        return self.from_blocks(
                operator(a, b) for a, b in zip_longest(
                    (self.single_column_filter(op) for op in self_opperands),
                    (self.single_column_filter(op) for op in other_opperands)
                    )
                )



class Display:

    @staticmethod
    def from_values(values: np.ndarray, header: str) -> tp.Dict[int, tp.Iterable[str]]:
        display_rows = len(values) + 2  # first is header, last for types
        display = OrderedDict((k, list()) for k in range(display_rows))

        # lables per row
        labels = [header]
        for v in values:
            labels.append(str(v))

        if isinstance(values, np.ndarray):
            labels.append(str(values.dtype))
        else:
            labels.append('')

        width = max(len(x) for x in labels) + 1

        for row, value in enumerate(labels):
            display[row].append(str(value).ljust(width))

        return display



class LocMap:

    @staticmethod
    def map_slice_args(label_to_pos, key: slice):
        '''Given a slice and a label to position mapping, yield each argument necessary to create a new slice.

        Args:
            label_to_pos: mapping, no order dependency
        '''
        # TODO: just iter over (key.start) etc.
        for field in ('start', 'stop', 'step'):
            attr = getattr(key, field)
            if attr is None:
                yield None
            else:
                if field == 'stop':
                    # loc selections are inclusive, so iloc gets one more
                    yield label_to_pos[attr] + 1
                else:
                    yield label_to_pos[attr]

    @classmethod
    def loc_to_iloc(cls,
            label_to_pos: tp.Dict,
            positions: np.ndarray,
            key: GetItemKeyType) -> GetItemKeyType:
        '''
        Returns:
            A integer mapped slice, or GetItemKey type that is based on integers, compatible with TypeBlocks
        '''

        if isinstance(key, slice):
            return slice(*cls.map_slice_args(label_to_pos, key))

        elif isinstance(key, _KEY_ITERABLE_TYPES):

            # can be an iterable of labels (keys) or an iterable of Booleans
            # if len(key) == len(label_to_pos) and isinstance(key[0], (bool, np.bool_)):
            if isinstance(key, np.ndarray) and key.dtype == bool:
                return positions[key]

            # map labels to integer positions
            # NOTE: we miss the opportunity to get a reference from values when we have contiguous keys
            return [label_to_pos[x] for x in key]

        # if a single element (an integer, string, or date, we just get the integer out of the map
        return label_to_pos[key]



#-------------------------------------------------------------------------------

class Index(metaclass=MetaOperatorDelegate):
    '''Index defines an ordered mapping of labels to index positions. Indices are not stored as np.array as need to be able to grow. Indices (via __getitem__ and similar) do not hand out immutable references (like TypeBlocks), but rather copies (like taking a slice of a list).
    '''

    __slots__ = (
            '_map',
            '_labels',
            '_positions',
            '_loc_is_iloc',
            'loc',
            'iloc',
            )

    @staticmethod
    def _extract_labels(mapping, labels) -> tp.Tuple[tp.Iterable[int], tp.Iterable[tp.Any]]:
        '''Derive labels, a cache of the mapping keys in a sequence type (either an ndarray or a list).

        If the labels passed at instantiation are an ndarray, they are used after immutable filtering. Otherwise, the mapping keys are used to create an ndarray.

        This method is verridden in the derived class.

        Args:
            labels: might be an expired Generator, but if it is an immutable npdarry, we can use it without a copy
        '''
        # pre-fetching labels for faster get_item construction
        if isinstance(labels, np.ndarray): # if an np array can handle directly
            labels = TypeBlocks.immutable_filter(labels)
        elif hasattr(labels, '__len__'): # not a generator, not an array
            labels = np.array(labels)
            labels.flags.writeable = False
        else:
            # since we do not know the data type, cannot use np.fromiter
            labels = np.array(tuple(mapping.keys()))
            labels.flags.writeable = False

        return labels

    @staticmethod
    def _extract_positions(mapping, positions):
        # positions is either None or an ndarray
        if isinstance(positions, np.ndarray): # if an np array can handle directly
            return TypeBlocks.immutable_filter(positions)
        positions = np.arange(len(mapping))
        positions.flags.writeable = False
        return positions


    # @staticmethod
    # def _compare_labels_positions(labels, positions) -> bool:
    #     if isinstance(labels, np.ndarray) and isinstance(positions, np.ndarray):
    #         if labels.dtype == int:
    #             return (labels == positions).all()
    #     return False


    def __init__(self, labels: tp.Generator[tp.Hashable, None, None]):
        '''
        Args:
            labels: the ordered keys to use as the index; can be a generator
        '''
        positions = None

        if isinstance(labels, Index):
            # get a reference to the immutable arrays
            positions = labels._positions
            labels = labels._labels

        elif isinstance(labels, IndexGrowOnly):
            raise Exception('cannot construct an IndexGrowOnly from an IndexGrowOnly yet')

        # map provided values to integer positions; do only one iteration of labels to support generators
        # collections.abs.sized
        if hasattr(labels, '__len__'):
            # NOTE: dict() function shown to be faster then gen expression
            self._map = dict(zip(labels, range(len(labels))))
            #self._map = {k: v for k, v in zip(labels, range(len(labels)))}
        else:
            # NOTE: dict() function shown slower in this case
            # self._map = dict((v, k) for k, v in enumerate(labels))
            self._map = {v: k for k, v in enumerate(labels)}

        # this might be NP array, or a list, depending on if static or grow only
        self._labels = self._extract_labels(self._map, labels)
        self._positions = self._extract_positions(self._map, positions)

        self._loc_is_iloc = False #self._compare_labels_positions(labels, positions)

        if len(self._map) != len(self._labels):
            raise KeyError('labels have non-unique values')

        self.loc = GetItem(self._extract_loc)
        self.iloc = GetItem(self._extract_iloc)


    def display(self) -> tp.Dict[int, tp.Iterable[str]]:
        '''Return a list of lists of strings for creating a complete display
        '''
        return Display.from_values(self._labels, '<' + self.__class__.__name__ + '>')

    def __repr__(self):
        return '\n'.join(''.join(row) for row in self.display().values())


    def loc_to_iloc(self, key: GetItemKeyType) -> GetItemKeyType:
        '''
        Returns:
            Return GetItemKey type that is based on integers, compatible with TypeBlocks
        '''
        if self._loc_is_iloc:
            return key
        return LocMap.loc_to_iloc(self._map, self._positions, key)

    def __len__(self):
        return len(self._labels)

    @property
    def values(self) -> np.ndarray:
        '''Return the immutable labels array
        '''
        return self._labels

    @property
    def mloc(self):
        '''Memory location
        '''
        return self._labels.__array_interface__['data'][0]

    def copy(self) -> 'Index':
        # this is not a complete deepcopy, as _labels here is an immutable np array (a new map will be created)
        return self.__class__(labels=self._labels)

    #---------------------------------------------------------------------------
    # set operations

    def intersection(self, other) -> 'Index':
        if isinstance(other, np.ndarray):
            opperand = other
        else: # assume we can get it from a .values attribute
            opperand = other.values

        return self.__class__(labels=np.intersect1d(self._labels, opperand))

    def union(self, other) -> 'Index':
        if isinstance(other, np.ndarray):
            opperand = other
        else: # assume we can get it from a .values attribute
            opperand = other.values

        return self.__class__(labels=np.union1d(self._labels, opperand))

    #---------------------------------------------------------------------------
    # extraction and selection

    def _extract_iloc(self, key) -> 'Index':
        '''Extract a new index given an iloc key
        '''
        if isinstance(key, slice):
            # if labels is an np array, this will be a view; if a list, a copy
            labels = self._labels[key]
        elif isinstance(key, _KEY_ITERABLE_TYPES):
            # we assume Booleans have been normalized to integers here
            # can select directly from _labels[key] if if key is a list
            labels = self._labels[key]
        else: # select a single label value
            labels = (self._labels[key],)
        return self.__class__(labels=labels)

    def _extract_loc(self, key: GetItemKeyType) -> 'Index':
        return self._extract_iloc(self.loc_to_iloc(key))

    def __getitem__(self, key: GetItemKeyType) -> 'Index':
        '''Extract a new index given an iloc key (this is the same as Pandas).
        '''
        return self._extract_iloc(key)

    #---------------------------------------------------------------------------
    # operators

    def _unary_operator(self, operator: tp.Callable) -> np.ndarray:
        '''Always return an NP array. Deviates form Pandas.
        '''
        array = operator(self._labels)
        array.flags.writeable = False
        return array

    def _binary_operator(self, *, operator: tp.Callable, other) -> np.ndarray:
        '''
        Binary operators applied to an index always return an NP array. This deviates from Pandas, where some operations (multipling an int index by an int) result in a new Index, while other opertations result in a np.array (using == on two Index).
        '''
        if isinstance(other, Index):
            other = other.values # operate on labels to labels
        array = operator(self._labels, other)
        array.flags.writeable = False
        return array


class IndexGrowOnly(Index):
    __slots__ = (
            '_map',
            '_labels',
            'iloc',
            )

    @staticmethod
    def _extract_labels(mapping, labels) -> tp.Iterable[tp.Any]:
        '''For a mutable Index, labels is stored as a list.
        '''
        if isinstance(labels, np.ndarray):
            return labels.tolist()
        if hasattr(labels, '__len__'):
            return list(labels)
        return list(mapping.keys())

    @property
    def values(self) -> np.ndarray:
        '''Convert the mutable labels to an immutable array for consistency
        '''
        labels = np.array(self._labels)
        labels.flags.writeable = False
        return labels

    @property
    def mloc(self):
        '''Memory location is None when a mutable index.
        '''
        return None

    def copy(self) -> 'Index':
        # keys is ordered; using generator creation
        return self.__class__(labels=self._map.keys())

    #---------------------------------------------------------------------------
    # grow only mutation

    def append(self, value):
        '''Add a value
        '''
        if value in self._map:
            raise KeyError('duplicate key append attempted', value)
        position = len(self._map) + 1
        self._map[value] = position
        self._labels.append(value)

    def extend(self, values: _KEY_ITERABLE_TYPES):
        '''Add multiple values
        Args:
            values: can be a generator.
        '''
        for value in values:
            if value in self._map:
                raise KeyError('duplicate key append attempted', value)
            # might bet better performance by calling extend() on _positions and _labels
            self.append(value)


    #---------------------------------------------------------------------------
    # operators
    # might be faster to do element wise operations in a generator, but would have to handle differenet dimensional other args; for now, let NP hanle

    def _unary_operator(self, operator: tp.Callable) -> np.ndarray:
        array = operator(np.array(self._labels))
        array.flags.writeable = False
        return array

    def _binary_operator(self, *, operator: tp.Callable, other) -> np.ndarray:
        array = operator(np.array(self._labels), other)
        array.flags.writeable = False
        return array


#-------------------------------------------------------------------------------
class Series(metaclass=MetaOperatorDelegate):

    __slots__ = (
        'values',
        'dtype',
        'shape',
        'ndim',
        'size',
        'bytes',
        '_index',
        'iloc',
        'loc',
        )

    def __init__(self, values, *, index=None, dtype=None):

        #-----------------------------------------------------------------------
        # values assignment
        # expose .values directly as it is immutable
        if not isinstance(values, np.ndarray):
            if dtype and hasattr(values, '__iter__'):
                self.values = np.fromiter(values, dtype=dtype)
            elif hasattr(values, '__len__'):
                self.values = np.array(values, dtype=dtype)
            elif hasattr(values, '__next__'): # a generator-like
                self.values = np.array(tuple(values), dtype=dtype)
            else: # it must be a single item
                self.values = np.array((values,), dtype=dtype)
            self.values.flags.writeable = False
        else: # is numpy
            if dtype is not None and dtype != values.dtype:
                raise Exception() # what to do here?
            self.values = TypeBlocks.immutable_filter(values)

        #-----------------------------------------------------------------------
        # index assignment

        if not index: # create an integer index
            self._index = Index(range(len(self.values)))
        elif isinstance(index, Index):
            # do not make a copy of it is an immutable index
            self._index = index
        elif isinstance(index, IndexGrowOnly):
            # if a grow only index need to make a copy; perhaps pass a parameter when we pass a newly-created index that can be held by the Series
            self._index = index.copy()
        else: # let index handle instantiation
            self._index = Index(index)

        assert len(self.values) == len(self._index), 'values and index do not match length'

        #-----------------------------------------------------------------------
        # attributes

        # populate attrbiutes from self.values
        # should these be properties to make assigment impossible
        self.dtype = self.values.dtype
        self.shape = self.values.shape
        self.ndim = self.values.ndim
        self.size = self.values.size
        self.bytes = self.values.nbytes

        self.loc = GetItem(self._extract_loc)
        self.iloc = GetItem(self._extract_iloc)

    #---------------------------------------------------------------------------
    # index manipulation

    def reindex(self,
            index: tp.Union[Index, tp.Sequence[tp.Any]],
            fill_value=np.nan) -> 'Series':
        '''
        Args:
            fill_value: attempted to be used, but may be coerced by the dtype of this Series. `
        '''
        # TODO: implement `method` argument with bfill, ffill options

        # always use the Index constructor for safe aliasing when possible
        index = Index(index)

        # manually do intersection for best performance
        common_labels = np.intersect1d(self._index._labels, index._labels)

        # if we are just reordering or selecting a subset
        if len(common_labels) == len(index):
            # same as calling .loc, which creates a new index, but cannot resuse self's labels array
            # this approach uses the labels created above
            values = self.values[self._index.loc_to_iloc(index.values)]
            # values already be immutable
            return self.__class__(values, index=index)

        try:
            fill_can_cast = np.can_cast(fill_value, self.values.dtype)
        except TypeError: # happens when fill is None and dtype is float
            fill_can_cast = False

        # only use value's dtype if fill_value casn cast
        dtype = object if not fill_can_cast else self.values.dtype
        values = np.full(len(index), fill_value, dtype=dtype)

        # if some intersection of values
        if len(common_labels) > 0:
            # need to walk through new to old slices for most efficient assignment when contiguous
            # NOTE: loc_to_iloc might return a slice, but not an iterable of slices
            src_iloc = self._index.loc_to_iloc(common_labels)
            dst_iloc = index.loc_to_iloc(common_labels)

            for src, dst in zip_longest(src_iloc, dst_iloc):
                values[dst] = self.values[src]

        # make immutable so a copy is not made
        values.flags.writeable = False
        return self.__class__(values, index=index)


    @staticmethod
    def _isnull_array(array) -> np.ndarray:
        '''Utility function that, given an np.ndarray, returns a bolean arrea setting True nulls (
        '''
        # need == comparison, not `is` comparison
        if array.dtype == float:
            return np.isnan(array)
        # match everything that is not an object; options are: biufcmMOSUV
        elif array.dtype.kind != 'O':
            return np.full(len(array), False, dtype=bool)
        # only check for None if we have an object type
        # np.equal matches None, no match to nan
        # astype: None gets converted to nan
        try: # this will only work for arrays that do not have strings
            # cannot use can_cast to reliabily identify arrays with non-float-castable elements
            return np.isnan(array.astype(float))
        except ValueError:
            # this means there was a character or something not castable to float; have to prceed slowly
            # TODO: this is a big perforamnce hit; problem is cannot find np.nan in numpy object array
            return np.fromiter((x is None or x is np.nan for x in array),
                    count=len(array),
                    dtype=bool)


    def isnull(self) -> 'Series':
        '''
        Return a same-indexed, Boolean Series indicating which values are NaN or None.
        '''
        # consider returning self if not values.any()?
        values = self._isnull_array(self.values)
        values.flags.writeable = False
        return self.__class__(values, index=self._index)

    def notnull(self):
        '''
        Return a same-indexed, Boolean Series indicating which values are NaN or None.
        '''
        values = np.logical_not(self._isnull_array(self.values))
        values.flags.writeable = False
        return self.__class__(values, index=self._index)

    def dropna(self):
        sel = np.logical_not(self._isnull_array(self.values))
        if not np.any(sel):
            return self

        values = self.values[sel]
        values.flags.writeable = False
        return self.__class__(values, index=self._index.loc[sel])


    def fillna(self, value):
        '''
        If no missing values, self is returned.
        '''
        sel = self._isnull_array(self.values)
        if not np.any(sel):
            return self

        try:
            fill_can_cast = np.can_cast(value, self.values.dtype)
        except TypeError: # happens when fill is None and dtype is float
            fill_can_cast = False

        if fill_can_cast:
            values = self.values.copy() # copy makes mutable again
        else:
            values = self.values.astype(object)

        values[sel] = value
        values.flags.writeable = False
        return self.__class__(values, index=self._index)

    #---------------------------------------------------------------------------
    # operators
    def _unary_operator(self, operator: tp.Callable) -> 'Series':
        return self.__class__(operator(self.values), index=self._index, dtype=self.dtype)

    def _binary_operator(self, *, operator: tp.Callable, other) -> 'Series':

        values = self.values
        index = self._index

        if isinstance(other, Series):
            # if indices are the same, we can simply set other to values and fallback on NP
            if (self.index == other.index).all(): # this is an array
                other = other.values
            else:
                index = self.index.union(other.index)
                # now need to reindex the Series
                values = self.reindex(index).values
                other = other.reindex(index).values

        # if its an np array, we simply fall back on np behavior
        elif isinstance(other, np.ndarray):
            if other.ndim > 1:
                raise NotImplementedError('Operator application to greater dimensionalities will result in an array with more than 1 dimension; it is not clear how such an array should be indexed.')
        # permit single value constants; not sure about filtering other types

        # we want the dtype to be the result of applying the operator; this happends by default
        return self.__class__(operator(values, other), index=index)

    #---------------------------------------------------------------------------
    def __len__(self):
        return len(self.values)

    def display(self) -> tp.Dict[int, tp.Iterable[str]]:
        d = self._index.display()
        dv = Display.from_values(self.values, '<' + self.__class__.__name__ + '>')
        for row_idx, row in dv.items():
            #print('\n', row_idx, row)
            d[row_idx].extend(row)
        return d

    def __repr__(self):
        return '\n'.join(''.join(row) for row in self.display().values())

    #---------------------------------------------------------------------------
    # common attributes from the numpy array

    @property
    def mloc(self):
        '''Memory location
        '''
        return self.values.__array_interface__['data'][0]

    #---------------------------------------------------------------------------
    # extraction

    def _extract_iloc(self, key: GetItemKeyType) -> 'Series':
        # iterable selection should be handled by NP (but maybe not if a tuple)
        return self.__class__(
                self.values[key],
                index=self._index.iloc[key])

    def _extract_loc(self, key: GetItemKeyType) -> 'Series':
        if isinstance(key, Series):
            key = key.values
        iloc_key = self._index.loc_to_iloc(key)
        values = self.values[iloc_key]

        if not isinstance(values, np.ndarray): # if we have a single element
            return values
        # this might create new index from iloc, and then createa another Index on Index __init__
        return self.__class__(values, index=self._index.iloc[iloc_key])

    def __getitem__(self, key: GetItemKeyType) -> 'Series':
        '''A Loc selection (by index labels).

        Compatibility:
            Pandas supports using both loc and iloc style selections with the __getitem__ interface on Series. This is undesirable, so here we only expose the loc interface (making the Series dictionary like, but unlike the Index, where __getitem__ is an iloc).
        '''
        return self._extract_loc(key)


    # TODO: special setter for .index that converts raw values into Index objects

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        if len(value) != len(self._index):
            raise Exception('new index must match length of old index')
        self._index = Index(value)

    #---------------------------------------------------------------------------
    # dictionary like interface

    def items(self) -> tp.Generator[tp.Tuple[tp.Any, tp.Any], None, None]:
        '''Provide a generator over pairs of index label and values.
        '''
        return zip(self._index.values, self.values)





class Frame:
    '''
    Store columns in uniform typed chunks?
    '''

    def __init__(self, ):
        pass

    @property
    def values(self):
        '''
        Pandas returns an np array as a list of rows
        np.matrix similarly is a list of rows; getting an array returns the same results
        '''