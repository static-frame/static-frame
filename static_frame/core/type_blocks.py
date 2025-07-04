from __future__ import annotations

from functools import partial
from itertools import chain, repeat, zip_longest

import numpy as np
import typing_extensions as tp
from arraykit import (
    BlockIndex,
    ErrorInitTypeBlocks,
    array_deepcopy,
    array_to_tuple_iter,
    astype_array,
    column_1d_filter,
    column_2d_filter,
    first_true_1d,
    immutable_filter,
    mloc,
    nonzero_1d,
    resolve_dtype,
    resolve_dtype_iter,
    row_1d_filter,
    shape_filter,
)

from static_frame.core.container import ContainerOperand
from static_frame.core.container_util import (
    apply_binary_operator_blocks,
    apply_binary_operator_blocks_columnar,
    get_block_match,
)
from static_frame.core.display import Display, DisplayActive
from static_frame.core.doc_str import doc_inject
from static_frame.core.exception import AxisInvalid
from static_frame.core.index_correspondence import IndexCorrespondence, assign_via_ic
from static_frame.core.node_selector import InterGetItemLocReduces
from static_frame.core.util import (
    DEFAULT_FAST_SORT_KIND,
    DEFAULT_SORT_KIND,
    DTYPE_BOOL,
    DTYPE_OBJECT,
    EMPTY_ARRAY,
    EMPTY_ARRAY_OBJECT,
    FILL_VALUE_DEFAULT,
    INT_TYPES,
    KEY_ITERABLE_TYPES,
    KEY_MULTIPLE_TYPES,
    NULL_SLICE,
    STRING_TYPES,
    PositionsAllocator,
    TArraySignature,
    TDtypeSpecifier,
    TILocSelector,
    TILocSelectorCompound,
    TILocSelectorMany,
    TILocSelectorOne,
    TLabel,
    TNDArray1DBool,
    TShape,
    TSortKinds,
    TTupleCtor,
    TUFunc,
    array_shift,
    array_signature,
    array_to_groups_and_locations,
    array_ufunc_axis_skipna,
    arrays_equal,
    binary_transition,
    blocks_to_array_2d,
    concat_resolved,
    dtype_from_element,
    dtype_to_fill_value,
    full_for_fill,
    isfalsy_array,
    isin_array,
    isna_array,
    iterable_to_array_1d,
    iterable_to_array_nd,
    roll_1d,
    slices_from_targets,
    ufunc_dtype_to_dtype,
    validate_dtype_specifier,
    view_2d_as_1d,
)

TNDArrayAny = np.ndarray[tp.Any, tp.Any]
TDtypeAny = np.dtype[tp.Any]
TOptionalArrayList = tp.Optional[tp.List[TNDArrayAny]]
TNDArrayObject = np.ndarray[tp.Any, np.dtype[np.object_]]

if tp.TYPE_CHECKING:
    from static_frame.core.display_config import DisplayConfig  # pragma: no cover
    from static_frame.core.index_correspondence import (
        IndexCorrespondence,
    )  # pragma: no cover
    from static_frame.core.style_config import StyleConfig  # pragma: no cover


# ---------------------------------------------------------------------------


def group_match(
    blocks: 'TypeBlocks',
    *,
    axis: int,
    key: TILocSelector,
    drop: bool = False,
    extract: tp.Optional[int] = None,
    as_array: bool = False,
    group_source: tp.Optional[TNDArrayAny] = None,
) -> tp.Iterator[tp.Tuple[TLabel, TNDArrayAny, tp.Union[TypeBlocks, TNDArrayAny]]]:
    """
    Args:
        key: iloc selector on opposite axis
        drop: Optionally drop the target of the grouping as specified by ``key``.
        extract: if provided, will be used to select from the group on the opposite axis

    Returns:
        Generator of group, selection pairs, where selection is an np.ndarray. Returned is as an np.ndarray if key is more than one column.
    """
    # NOTE: in axis_values we determine zero size by looking for empty _blocks; not sure if that is appropriate here.
    if blocks._index.shape[0] == 0 or blocks._index.shape[1] == 0:  # zero sized
        return

    unique_axis = None

    if group_source is not None:
        pass
    elif axis == 0:
        # axis 0 means we return row groups; key is a column key
        group_source = blocks._extract_array(column_key=key)
    elif axis == 1:
        # axis 1 means we return column groups; key is a row key
        group_source = blocks._extract_array(row_key=key)
    else:
        raise AxisInvalid(f'invalid axis: {axis}')

    groups: tp.Iterable[tp.Any]
    groups, locations = array_to_groups_and_locations(
        group_source,
        axis,
    )

    if group_source.ndim > 1:
        # NOTE: this is expensive!
        # make the groups hashable for usage in index construction
        if axis == 0:
            groups = array_to_tuple_iter(groups)
        else:
            groups = array_to_tuple_iter(groups.T)

    if drop:
        # axis 0 means we return row groups; key is a column key
        shape = blocks._index.shape[1] if axis == 0 else blocks._index.shape[0]
        drop_mask = np.full(shape, True, dtype=DTYPE_BOOL)
        drop_mask[key] = False

    column_key: tp.Union[int, TNDArrayAny, None]
    row_key: tp.Union[int, TNDArrayAny, None]
    # this key is used to select which components are returned per group selection (where that group selection is on the opposite axis)

    func: tp.Callable[..., tp.Union[TypeBlocks, TNDArrayAny]] = (
        blocks._extract_array if as_array else blocks._extract
    )

    if axis == 0:
        if extract is not None:
            column_key = extract
        else:
            column_key = None if not drop else drop_mask
    else:
        if extract is not None:
            row_key = extract
        else:
            row_key = None if not drop else drop_mask

    # NOTE: we create one mutable Boolean array to serve as the selection for each group; as this array is yielded out, the caller must use it before the next iteration, which is assumed to alway be the case.
    selection = np.empty(len(locations), dtype=DTYPE_BOOL)

    for idx, g in enumerate(groups):
        # derive a Boolean array of fixed size showing where value in this group are found from the original TypeBlocks
        np.equal(locations, idx, out=selection)

        if axis == 0:  # return row
            yield (
                g,
                selection,
                func(
                    row_key=selection,
                    column_key=column_key,
                ),
            )
        else:  # return columns extractions
            yield (
                g,
                selection,
                func(
                    row_key=row_key,
                    column_key=selection,
                ),
            )


def group_sorted(
    blocks: TypeBlocks,
    *,
    axis: int,
    key: TILocSelector,
    drop: bool = False,
    extract: tp.Optional[int] = None,
    as_array: bool = False,
    group_source: tp.Optional[TNDArrayAny] = None,
) -> tp.Iterator[tp.Tuple[TLabel, slice, TypeBlocks | TNDArrayAny]]:
    """
    This method must be called on sorted TypeBlocks instance.

    Args:
        blocks: sorted TypeBlocks
        order: args
        key: iloc selector on opposite axis
        drop: Optionally drop the target of the grouping as specified by ``key``.
        axis: if 0, key is column selection, yield groups of rows; if 1, key is row selection, yield gruops of columns
        kind: Type of sort; a stable sort is required to preserve original odering.

    Returns:
        Generator of group, selection pairs, where selection is an np.ndarray. Returned is as an np.ndarray if key is more than one column.
    """
    # if extract is not None drop must be False
    # assert extract is not None and drop is False

    # NOTE: in axis_values we determine zero size by looking for empty _blocks; not sure if that is appropriate here.
    shape = blocks._index.shape
    if shape[0] == 0 or shape[1] == 0:  # zero sized
        return

    if group_source is not None:
        pass
        # NOTE: axis 1 transposition is not required as group_source is already prepared by h-stacking 1D arrays
    elif axis == 0:
        # axis 0 means we return row groups; key is a column key
        group_source = blocks._extract_array(column_key=key)
    elif axis == 1:
        # axis 1 means we return column groups; key is a row key
        group_source = blocks._extract_array(row_key=key)
        # for ndim==2, must present values from top to bottom
        group_source = group_source.T
    else:
        raise AxisInvalid(f'invalid axis: {axis}')

    if drop:
        # axis 0 means we return row groups; key is a column key
        drop_shape = shape[1] if axis == 0 else shape[0]
        drop_mask = np.full(drop_shape, True, dtype=DTYPE_BOOL)
        drop_mask[key] = False

    column_key: tp.Union[int, TNDArrayAny, None]
    row_key: tp.Union[int, TNDArrayAny, None]
    func: tp.Callable[..., tp.Union[TypeBlocks, TNDArrayAny]] = (
        blocks._extract_array if as_array else blocks._extract
    )
    # this key is used to select which components are returned per group selection (where that group selection is on the opposite axis)
    if axis == 0:
        if extract is not None:
            column_key = extract
        else:
            column_key = None if not drop else drop_mask
    else:
        if extract is not None:
            row_key = extract
        else:
            row_key = None if not drop else drop_mask

    # find iloc positions where new value is not equal to previous; drop the first as roll wraps
    if group_source.ndim == 2:
        group_to_tuple = True
        if group_source.dtype == DTYPE_OBJECT:
            # NOTE: cannot get view of object; use string
            consolidated = view_2d_as_1d(group_source.astype(str))
        else:
            consolidated = view_2d_as_1d(group_source)
        transitions = nonzero_1d(consolidated != roll_1d(consolidated, 1))[1:]
    else:
        group_to_tuple = False
        transitions = nonzero_1d(group_source != roll_1d(group_source, 1))[1:]

    start = 0
    if axis == 0 and group_to_tuple:
        for t in transitions:
            slc = slice(start, t)
            chunk = func(slc, column_key)
            yield tuple(group_source[start]), slc, chunk  # pyright: ignore
            start = t
    elif axis == 0 and not group_to_tuple:
        for t in transitions:
            slc = slice(start, t)
            chunk = func(slc, column_key)
            yield group_source[start], slc, chunk
            start = t
    elif axis == 1 and group_to_tuple:
        for t in transitions:
            slc = slice(start, t)
            chunk = func(row_key, slc)
            yield tuple(group_source[start]), slc, chunk  # pyright: ignore
            start = t
    elif axis == 1 and not group_to_tuple:
        for t in transitions:
            slc = slice(start, t)
            chunk = func(row_key, slc)
            yield group_source[start], slc, chunk
            start = t

    if start < len(group_source):
        slc = slice(start, None)
        if axis == 0:
            chunk = func(row_key=slc, column_key=column_key)
        else:
            chunk = func(row_key=row_key, column_key=slc)
        if group_to_tuple:
            yield tuple(group_source[start]), slc, chunk  # pyright: ignore
        else:
            yield group_source[start], slc, chunk


# -------------------------------------------------------------------------------


def assign_inner_from_iloc_by_unit(
    *,
    value: tp.Any,
    block: TNDArrayAny,
    row_target: TILocSelector,
    target_key: TILocSelector,
    t_shape: TShape,
    target_is_slice: bool,
    block_is_column: bool,
    row_key_is_null_slice: bool,
) -> tp.Tuple[tp.Any, TNDArrayAny]:
    if value.__class__ is np.ndarray:
        is_tuple = False
        value_dtype = value.dtype
    elif isinstance(value, tuple):
        is_tuple = True
        value_dtype = DTYPE_OBJECT
    else:  # all other inputs are elements
        is_tuple = False
        value_dtype = dtype_from_element(value)

    # match sliceable, when target_key is a slice (can be an element)
    if (
        target_is_slice
        and not isinstance(value, STRING_TYPES)
        and not is_tuple
        and hasattr(value, '__len__')
    ):
        if block_is_column:
            v_width = 1
            # if block is 1D, then we can only take 1 column if we have a 2d value
            value_piece_column_key: tp.Union[slice, int] = 0
        else:
            v_width = len(range(*target_key.indices(block.shape[1])))  # type: ignore
            # if block id 2D, can take up to v_width from value
            value_piece_column_key = slice(0, v_width)

        if value.__class__ is np.ndarray and value.ndim > 1:  # pyright: ignore
            value_piece = value[NULL_SLICE, value_piece_column_key]  # pyright: ignore
            # restore for next iter
            value = value[NULL_SLICE, slice(v_width, None)]  # pyright: ignore
        else:  # value is 1D array or tuple, assume assigning into a horizontal position
            value_piece = value[value_piece_column_key]
            value = value[slice(v_width, None)]
    else:  # not sliceable; this can be a single column
        value_piece = value

    if row_key_is_null_slice:  # will replace entire sub block, can be empty
        assigned_target = np.empty(t_shape, dtype=value_dtype)
    else:  # will need to mix types
        assigned_dtype = resolve_dtype(value_dtype, block.dtype)
        if block_is_column:
            assigned_target_pre = (
                block if block.ndim == 1 else block.reshape(block.shape[0])
            )  # make 1D
        else:
            assigned_target_pre = block[NULL_SLICE, target_key]
        if block.dtype == assigned_dtype:
            assigned_target = assigned_target_pre.copy()
        else:
            assigned_target = assigned_target_pre.astype(assigned_dtype)

    if assigned_target.ndim == 1:
        assigned_target[row_target] = value_piece
    else:
        # we are editing the entire assigned target sub block
        if is_tuple:
            # must do individual assignments to enforce element treatment
            for col in range(assigned_target.shape[1]):
                assigned_target[row_target, col] = value_piece
        else:
            assigned_target[row_target, NULL_SLICE] = value_piece

    assigned_target.flags.writeable = False
    return value, assigned_target


def assign_inner_from_iloc_by_sequence(
    *,
    value: tp.Any,
    block: TNDArrayAny,
    row_target: TILocSelector,
    target_key: TILocSelector,
    t_shape: TShape,
    target_is_slice: bool,
    block_is_column: bool,
    row_key_is_null_slice: bool,
) -> tp.Tuple[tp.Any, TNDArrayAny]:
    if value.__class__ is np.ndarray:
        # NOTE: might support object arrays...
        raise ValueError('an array cannot be used as a value')

    # match sliceable, when target_key is a slice (can be an element)
    value_piece: tp.Sequence[tp.Any] | TNDArrayAny
    if target_is_slice:
        if block_is_column:
            v_width = 1
            # if block is 1D, then we can only take 1 column if we have a 2d value
            value_piece_column_key: tp.Union[slice, int] = 0
        else:
            v_width = len(range(*target_key.indices(block.shape[1])))  # type: ignore
            # if block id 2D, can take up to v_width from value
            value_piece_column_key = slice(0, v_width)

        # value is tuple, assume assigning into a horizontal position
        value_piece = value[value_piece_column_key]
        value = value[slice(v_width, None)]

        if hasattr(value_piece, '__len__') and not isinstance(value_piece, str):
            value_piece, _ = iterable_to_array_1d(value_piece)
            value_dtype = resolve_dtype(value_piece.dtype, block.dtype)
        else:
            value_dtype = resolve_dtype(dtype_from_element(value_piece), block.dtype)
    elif len(value) == 1:
        # target must be an integer if it is not a slice
        value_piece = value[0]
        value = ()
        value_dtype = resolve_dtype(dtype_from_element(value_piece), block.dtype)
    elif len(value) > 1:
        raise ValueError('Value has incorrect length for this assignment.')
    else:
        # An empty iterable is not supported
        raise ValueError(f'No support for this value type in assignment: {value}')

    if row_key_is_null_slice:  # will replace entire sub block, can be empty
        assigned_target = np.empty(t_shape, dtype=value_dtype)
    else:  # will need to mix types
        assigned_dtype = resolve_dtype(value_dtype, block.dtype)
        if block_is_column:
            assigned_target_pre = (
                block if block.ndim == 1 else block.reshape(block.shape[0])
            )  # make 1D
        else:
            assigned_target_pre = block[NULL_SLICE, target_key]
        if block.dtype == assigned_dtype:
            assigned_target = assigned_target_pre.copy()
        else:
            assigned_target = assigned_target_pre.astype(assigned_dtype)

    if assigned_target.ndim == 1:
        assigned_target[row_target] = value_piece
    else:  # we are editing the entire assigned target sub block
        assigned_target[row_target, NULL_SLICE] = value_piece

    assigned_target.flags.writeable = False

    return value, assigned_target


# -------------------------------------------------------------------------------


class TypeBlocks(ContainerOperand):
    """An ordered collection of type-heterogenous, immutable NumPy arrays, providing an external array-like interface of a single, 2D array. Used by :obj:`Frame` for core, unindexed array management.

    A TypeBlocks instance can have a zero size shape (where the length of one axis is zero). Internally, when axis 0 (rows) is of size 0, we store similarly sized arrays. When axis 1 (columns) is of size 0, we do not store arrays, as such arrays do not define a type (as types are defined by columns).
    """

    __slots__ = (
        '_blocks',
        '_index',
    )

    STATIC = False

    # ---------------------------------------------------------------------------
    # constructors

    @classmethod
    def from_blocks(
        cls,
        raw_blocks: tp.Iterable[TNDArrayAny],
        shape_reference: tp.Optional[TShape] = None,
        own_data: bool = False,
    ) -> 'TypeBlocks':
        """
        Main constructor using iterator (or generator) of TypeBlocks; the order of the blocks defines the order of the columns contained.

        It is acceptable to construct blocks with a 0-sided shape.

        Args:
            raw_blocks: iterable (generator compatible) of NDArrays, or a single NDArray.
            shape_reference: optional argument to support cases where no blocks are found in the ``raw_blocks`` iterable, but the outer context is one with rows but no columns, or columns and no rows.
            own_data: If the caller knows all arrays are immutable, immutable_filter calls can be skipped.

        """
        blocks: tp.List[TNDArrayAny] = []  # ordered blocks
        index = BlockIndex()  # pyright: ignore

        # if a single block, no need to loop
        if raw_blocks.__class__ is np.ndarray:
            if index.register(raw_blocks):  # type: ignore
                blocks.append(immutable_filter(raw_blocks))  # type: ignore
        else:  # an iterable of blocks
            # we keep array with 0 rows but > 0 columns, as they take type space in the TypeBlocks object; arrays with 0 columns do not take type space and thus can be skipped entirely
            if own_data:  # skip immutable_filter
                for block in raw_blocks:
                    if index.register(block):
                        blocks.append(block)
            else:
                for block in raw_blocks:
                    if index.register(block):
                        blocks.append(immutable_filter(block))

            # blocks can be empty, and index with no registration has rows as -1
            if index.rows < 0:
                if shape_reference is not None:
                    index.register(EMPTY_ARRAY.reshape(shape_reference[0], 0))  # type: ignore
                else:
                    raise ErrorInitTypeBlocks(
                        'cannot derive a row_count from blocks; provide a shape reference'
                    )

        return cls(
            blocks=blocks,
            index=index,
        )

    @classmethod
    def from_element_items(
        cls,
        items: tp.Iterable[tp.Tuple[tp.Tuple[int, int], tp.Any]],
        shape: tp.Tuple[int, ...],
        dtype: TDtypeSpecifier | None,
        fill_value: object = FILL_VALUE_DEFAULT,
    ) -> 'TypeBlocks':
        """Given a generator of pairs of iloc coords and values, return a TypeBlock of the desired shape and dtype. This permits only uniformly typed data, as we have to create a single empty array first, then populate it."""
        fill_value = (
            fill_value
            if fill_value is not FILL_VALUE_DEFAULT
            else dtype_to_fill_value(dtype)
        )

        a = np.full(shape, fill_value=fill_value, dtype=dtype)
        for iloc, v in items:
            a[iloc] = v
        a.flags.writeable = False
        return cls.from_blocks(a)

    @classmethod
    def from_zero_size_shape(
        cls,
        shape: tp.Tuple[int, int] = (0, 0),
        get_col_dtype: tp.Optional[tp.Callable[[int], TDtypeSpecifier]] = None,
    ) -> 'TypeBlocks':
        """
        Given a shape where one or both axis is 0 (a zero sized array), return a TypeBlocks instance.
        """
        rows, columns = shape

        if not (rows == 0 or columns == 0):
            raise RuntimeError(f'invalid shape for empty TypeBlocks: {shape}')

        # as types are organized vertically, storing an array with 0 rows but > 0 columns is appropriate as it takes type space
        blocks: TNDArrayAny | tp.Iterable[TNDArrayAny]
        if rows == 0 and columns > 0:
            if get_col_dtype is None:
                blocks = np.empty(shape)
                blocks.flags.writeable = False
            else:
                blocks = (np.empty(rows, dtype=get_col_dtype(i)) for i in range(columns))
            return cls.from_blocks(blocks)

        # for arrays with 0 columns, favor storing shape alone and not creating an array object; the shape will be binding for future appending
        return cls.from_blocks((), shape_reference=shape)

    @staticmethod
    def vstack_blocks_to_blocks(
        type_blocks: tp.Sequence['TypeBlocks'],
        block_compatible: tp.Optional[bool] = None,
        reblock_compatible: tp.Optional[bool] = None,
    ) -> tp.Iterator[TNDArrayAny]:
        """
        Given a sequence of TypeBlocks with shape[1] equal to this TB's shape[1], return an iterator of consolidated arrays.
        """
        if block_compatible is None and reblock_compatible is None:
            block_compatible = True
            reblock_compatible = True
            previous_tb = None
            for tb in type_blocks:
                if previous_tb is not None:  # after the first
                    if block_compatible:
                        block_compatible &= tb.block_compatible(
                            previous_tb, axis=1
                        )  # only compare columns
                    if reblock_compatible:
                        reblock_compatible &= tb.reblock_compatible(previous_tb)
                previous_tb = tb

        if block_compatible or reblock_compatible:
            tb_proto: tp.Sequence[TypeBlocks]
            if not block_compatible and reblock_compatible:
                # after reblocking, will be compatible
                tb_proto = [tb.consolidate() for tb in type_blocks]
            else:  # blocks by column are compatible
                tb_proto = type_blocks

            # all TypeBlocks have the same number of blocks by here
            for block_idx in range(len(tb_proto[0]._blocks)):
                block_parts = []
                for tb_proto_idx in range(len(tb_proto)):
                    b = column_2d_filter(tb_proto[tb_proto_idx]._blocks[block_idx])
                    block_parts.append(b)
                yield concat_resolved(block_parts)  # returns immutable array
        else:  # blocks not alignable
            # break into single column arrays for maximum type integrity; there might be an alternative reblocking that could be more efficient, but determining that shape might be complex
            for i in range(type_blocks[0].shape[1]):
                block_parts = [tb._extract_array(column_key=i) for tb in type_blocks]
                yield concat_resolved(block_parts)

    # ---------------------------------------------------------------------------

    def __init__(
        self,
        *,
        blocks: tp.List[TNDArrayAny],
        index: BlockIndex,
    ) -> None:
        """
        Default constructor. We own all lists passed in to this constructor. This instance takes ownership of all lists passed to it.

        Args:
            blocks: A list of one or two-dimensional NumPy arrays. The list is owned by this instance.
            index: BlockIndex
        """
        self._blocks = blocks
        self._index = index

    # ---------------------------------------------------------------------------
    def __setstate__(
        self,
        state: tp.Tuple[object, tp.Mapping[str, tp.Any]],
    ) -> None:
        """
        Ensure that reanimated NP arrays are set not writeable.
        """
        for key, value in state[1].items():
            setattr(self, key, value)

        for b in self._blocks:
            b.flags.writeable = False

    def __deepcopy__(self, memo: tp.Dict[int, tp.Any]) -> 'TypeBlocks':
        obj = self.__class__.__new__(self.__class__)
        obj._blocks = [array_deepcopy(b, memo) for b in self._blocks]
        obj._index = self._index.copy()  # list of tuples of ints
        memo[id(self)] = obj
        return obj

    def __copy__(self) -> 'TypeBlocks':
        """
        Return shallow copy of this TypeBlocks. Underlying arrays are not copied.
        """
        return self.__class__(
            blocks=[b for b in self._blocks],
            index=self._index.copy(),  # list
        )

    def copy(self) -> 'TypeBlocks':
        """
        Return shallow copy of this TypeBlocks. Underlying arrays are not copied.
        """
        return self.__copy__()

    # ---------------------------------------------------------------------------
    # new properties

    def _iter_dtypes(self) -> tp.Iterator[TDtypeAny]:
        for b in self._blocks:
            dt = b.dtype
            if b.ndim == 1:
                yield dt
            else:  # PERF: repeat is much faster than a for loop
                yield from repeat(dt, b.shape[1])

    @property
    def dtypes(self) -> TNDArrayObject:
        """
        Return an immutable array that, for each realizable column (not each block), the dtype is given.
        """
        # this creates a new array every time it is called; could cache
        a = np.empty(self._index.columns, dtype=DTYPE_OBJECT)
        a[NULL_SLICE] = list(self._iter_dtypes())
        a.flags.writeable = False
        return a

    @property
    def shapes(self) -> TNDArrayAny:
        """
        Return an immutable array that, for each block, reports the shape as a tuple.
        """
        a = np.empty(len(self._blocks), dtype=DTYPE_OBJECT)
        a[:] = [b.shape for b in self._blocks]
        a.flags.writeable = False
        return a

    @property
    @doc_inject()
    def mloc(self) -> TNDArrayAny:
        """{doc_array}"""
        a = np.fromiter(
            (mloc(b) for b in self._blocks), count=len(self._blocks), dtype=np.int64
        )
        a.flags.writeable = False
        return a

    @property
    def unified(self) -> bool:
        return len(self._blocks) <= 1

    @property
    def unified_dtypes(self) -> bool:
        """Return True if all blocks have the same dtype."""
        # use blocks to iterate over fewer things
        if len(self._blocks) <= 1:
            return True
        dtypes = iter(d.dtype for d in self._blocks)
        # NOTE: could compare to index.dtype
        d_first = next(dtypes)
        for d in dtypes:
            if d != d_first:
                return False
        return True

    # ---------------------------------------------------------------------------
    # interfaces

    @property
    def iloc(self) -> InterGetItemLocReduces:  # type: ignore
        return InterGetItemLocReduces(self._extract_iloc)  # type: ignore

    # ---------------------------------------------------------------------------
    # common NP-style properties

    @property
    def shape(self) -> tp.Tuple[int, int]:
        return self._index.shape

    @property
    def ndim(self) -> int:
        return 2

    @property
    def size(self) -> int:
        return sum(b.size for b in self._blocks)

    @property
    def nbytes(self) -> int:
        return sum(b.nbytes for b in self._blocks)

    # ---------------------------------------------------------------------------
    # value extraction

    @property
    def values(self) -> TNDArrayAny:
        """Returns a consolidated NP array of the all blocks."""
        # provide a default dtype if one has not yet been set (an empty TypeBlocks, for example)
        # always return a 2D array
        return blocks_to_array_2d(
            blocks=self._blocks,
            shape=self._index.shape,
            dtype=self._index.dtype,
        )

    def axis_values(
        self,
        axis: int = 0,
        reverse: bool = False,
    ) -> tp.Iterator[TNDArrayAny]:
        """Generator of 1D arrays produced along an axis. Clients can expect to get an immutable array.

        Args:
            axis: 0 iterates over columns, 1 iterates over rows
        """
        # NOTE: might be renamed iter_arrays_by_axis

        if axis == 1:  # iterate over rows
            zero_size = not bool(self._blocks)
            unified = self.unified
            # key: tp.Union[int, slice]
            row_dtype = self._index.dtype
            row_length = self._index.rows

            if not reverse:
                row_idx_iter = range(row_length)
            else:
                row_idx_iter = range(row_length - 1, -1, -1)

            if zero_size:
                for _ in row_idx_iter:
                    yield EMPTY_ARRAY
            elif unified:
                b = self._blocks[0]
                for i in row_idx_iter:
                    if b.ndim == 1:  # slice to force array creation (not an element)
                        yield b[i : i + 1]
                    else:  # 2d array
                        yield b[i]
            else:
                # PERF: only creating and yielding one array at a time is shown to be slower; performance optimized: consolidate into a single array and then take slices
                # NOTE: this might force unnecessary type coercion if going to a tuple, but if going to an array, the type consolidation is necessary
                b = blocks_to_array_2d(
                    blocks=self._blocks,
                    shape=self._index.shape,
                    dtype=row_dtype,
                )
                for i in row_idx_iter:
                    yield b[i]

        elif axis == 0:  # iterate over columns
            if not reverse:
                block_column_iter = iter(self._index)
            else:
                block_column_iter = reversed(self._index)

            for block_idx, column in block_column_iter:
                b = self._blocks[block_idx]
                if b.ndim == 1:
                    yield b
                else:
                    yield b[NULL_SLICE, column]  # expected to be immutable
        else:
            raise AxisInvalid(f'no support for axis: {axis}')

    def element_items(
        self,
        axis: int = 0,
    ) -> tp.Iterator[tp.Tuple[tp.Tuple[int, ...], tp.Any]]:
        """
        Generator of pairs of iloc locations, values across entire TypeBlock. Used in creating a IndexHierarchy instance from a TypeBlocks.

        Args:
            axis: if 0, use row major iteration,  vary fastest along row.
        """
        shape = (
            self._index.shape if axis == 0 else (self._index.columns, self._index.rows)
        )

        for iloc in np.ndindex(shape):
            if axis != 0:  # invert
                iloc = (iloc[1], iloc[0])

            block_idx, column = self._index[iloc[1]]
            b = self._blocks[block_idx]
            if b.ndim == 1:
                yield iloc, b[iloc[0]]
            else:
                yield iloc, b[iloc[0], column]

    # ---------------------------------------------------------------------------
    # methods for evaluating compatibility with other blocks, and reblocking
    def _reblock_signature(self) -> tp.Iterator[tp.Tuple[TDtypeAny, int]]:
        """For anticipating if a reblock will result in a compatible block configuration for operator application, get the reblock signature, providing the dtype and size for each block without actually reblocking.

        This is a generator to permit lazy pairwise comparison.
        """
        group_dtype: tp.Optional[TDtypeAny] = (
            None  # store type found along contiguous blocks
        )
        group_cols = 0
        for block in self._blocks:
            if group_dtype is None:  # first block of a type
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

        assert group_dtype is not None
        if group_cols > 0:
            yield (group_dtype, group_cols)

    def block_compatible(
        self, other: 'TypeBlocks', axis: tp.Optional[int] = None
    ) -> bool:
        """Block compatible means that the blocks are the same shape. Type is not yet included in this evaluation.

        Args:
            axis: If True, the full shape is compared; if False, only the columns width is compared.
        """
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
        """
        Return True if post reblocking these TypeBlocks are compatible. This only compares columns in blocks, not the entire shape.
        """
        if self.shape[1] != other.shape[1]:
            return False
        # we only compare size, not the type
        return not any(
            a is None or b is None or a[1] != b[1]
            for a, b in zip_longest(self._reblock_signature(), other._reblock_signature())
        )

    @staticmethod
    def _concatenate_blocks(
        blocks: tp.Iterable[TNDArrayAny],
        dtype: TDtypeSpecifier,
        columns: int,
    ) -> TNDArrayAny:
        """Join blocks on axis 1, assuming the they have an appropriate dtype. This will always return a 2D array. This generally assumes that they dtype is aligned among the provided blocks."""
        # NOTE: when this is called we always have 2 or more blocks
        blocks_norm = [column_2d_filter(x) for x in blocks]
        rows = blocks_norm[0].shape[0]  # all 2D

        array = np.empty((rows, columns), dtype=dtype)
        np.concatenate(blocks_norm, axis=1, out=array)
        array.flags.writeable = False
        return array

    @classmethod
    def consolidate_blocks(
        cls,
        raw_blocks: tp.Iterable[TNDArrayAny],
    ) -> tp.Iterator[TNDArrayAny]:
        """
        Generator consumer, generator producer of np.ndarray, consolidating if types are exact matches.

        Returns: an Iterator of 1D or 2D arrays, consolidated if adjacent.
        """
        group_dtype: tp.Optional[TDtypeAny] = (
            None  # store type found along contiguous blocks
        )
        group = []

        for block in raw_blocks:
            if group_dtype is None:  # first block of a type
                group_dtype = block.dtype
                group.append(block)
                group_columns = 1 if block.ndim == 1 else block.shape[1]
                continue
            # NOTE: could be less strict and look for compatibility within dtype kind (or other compatible types)
            if block.dtype != group_dtype:
                # new group found, return stored
                if len(group) == 1:  # return reference without copy
                    # NOTE: using pop() here not shown to be faster
                    yield group[0]
                else:  # combine groups
                    yield cls._concatenate_blocks(group, group_dtype, group_columns)
                group_dtype = block.dtype
                group = [block]
                group_columns = 1 if block.ndim == 1 else block.shape[1]
            else:  # new block has same group dtype
                group.append(block)
                group_columns += 1 if block.ndim == 1 else block.shape[1]

        # always have one or more leftover
        if group:
            if len(group) == 1:
                yield group[0]
            else:
                yield cls._concatenate_blocks(group, group_dtype, group_columns)

    @classmethod
    def contiguous_columnar_blocks(
        cls,
        raw_blocks: tp.Iterable[TNDArrayAny],
    ) -> tp.Iterator[TNDArrayAny]:
        """
        Generator consumer, generator producer of np.ndarray, ensuring that blocks are contiguous in columnar access.

        Returns: an Iterator of 1D or 2D arrays, consolidated if adjacent.
        """

        for b in raw_blocks:
            # NOTE: 1D contiguous array sets both C_CONTIGUOUS and F_CONTIGUOUS; applying asfortranarray() to a 1D array is the same as ascontiguousarray(); columnar slices on F_CONTIGUOUS are contiguous;
            if not b.flags['F_CONTIGUOUS']:
                b = np.asfortranarray(b)
                b.flags.writeable = False
                yield b
            else:
                yield b

    def _reblock(self) -> tp.Iterator[TNDArrayAny]:
        """Generator of new block that consolidate adjacent types that are the same."""
        yield from self.consolidate_blocks(raw_blocks=self._blocks)

    def consolidate(self) -> 'TypeBlocks':
        """Return a new TypeBlocks that unifies all adjacent types."""
        return self.from_blocks(self.consolidate_blocks(raw_blocks=self._blocks))

    def contiguous_columnar(self) -> 'TypeBlocks':
        """Return a new TypeBlocks that makes all columns or column slices contiguous."""
        return self.from_blocks(self.contiguous_columnar_blocks(raw_blocks=self._blocks))

    # ---------------------------------------------------------------------------
    def resize_blocks_by_element(
        self,
        *,
        index_ic: tp.Optional[IndexCorrespondence],
        columns_ic: tp.Optional[IndexCorrespondence],
        fill_value: tp.Any,
    ) -> tp.Iterator[TNDArrayAny]:
        """
        Given index and column IndexCorrespondence objects, return a generator of resized blocks, extracting from self based on correspondence. Used for Frame.reindex(). Note that `fill_value` is an element.
        """
        if columns_ic is None and index_ic is None:
            yield from self._blocks
        elif columns_ic is None and index_ic is not None:
            for b in self._blocks:
                if index_ic.is_subset:
                    # works for both 1d and 2s arrays
                    yield b[index_ic.iloc_src]
                else:
                    shape: TShape = (
                        index_ic.size if b.ndim == 1 else (index_ic.size, b.shape[1])
                    )
                    values = full_for_fill(b.dtype, shape, fill_value)
                    assign_via_ic(index_ic, b, values)
                    yield values

        elif columns_ic is not None and index_ic is None:
            if not columns_ic.has_common:  # no columns in common
                shape = self.shape[0], columns_ic.size
                values = full_for_fill(None, shape, fill_value)
                values.flags.writeable = False
                yield values
            elif self.unified and columns_ic.is_subset:
                b = self._blocks[0]
                if b.ndim == 1:
                    yield b
                else:
                    yield b[NULL_SLICE, columns_ic.iloc_src]
            else:
                dst_to_src = dict(zip(columns_ic.iloc_dst, columns_ic.iloc_src))  # type: ignore [arg-type]
                for idx in range(columns_ic.size):
                    if idx in dst_to_src:
                        block_idx, block_col = self._index[dst_to_src[idx]]  # pyright: ignore
                        b = self._blocks[block_idx]
                        if b.ndim == 1:
                            yield b
                        else:
                            yield b[:, block_col]
                    else:  # just get an empty position, fill_value determines type
                        values = full_for_fill(None, self.shape[0], fill_value)
                        values.flags.writeable = False
                        yield values

        else:  # both defined
            assert columns_ic is not None and index_ic is not None  # mypy
            if not columns_ic.has_common or not index_ic.has_common:
                # no selection on either axis is an empty frame
                shape = index_ic.size, columns_ic.size
                values = full_for_fill(None, shape, fill_value)
                values.flags.writeable = False
                yield values
            elif self.unified and index_ic.is_subset and columns_ic.is_subset:
                b = self._blocks[0]
                if b.ndim == 1:
                    yield b[index_ic.iloc_src]
                else:
                    yield b[index_ic.iloc_src_fancy(), columns_ic.iloc_src]
            else:
                columns_dst_to_src = dict(zip(columns_ic.iloc_dst, columns_ic.iloc_src))  # type: ignore [arg-type]
                for idx in range(columns_ic.size):
                    if idx in columns_dst_to_src:
                        block_idx, block_col = self._index[columns_dst_to_src[idx]]  # pyright: ignore
                        b = self._blocks[block_idx]
                        if index_ic.is_subset:
                            if b.ndim == 1:
                                # NOTE: iloc_src is in the right order for dst
                                yield b[index_ic.iloc_src]
                            else:
                                yield b[index_ic.iloc_src, block_col]
                        else:  # need an empty to fill, compatible with this block
                            values = full_for_fill(b.dtype, index_ic.size, fill_value)
                            if b.ndim == 1:
                                assign_via_ic(index_ic, b, values)
                            else:
                                assign_via_ic(index_ic, b[NULL_SLICE, block_col], values)
                            yield values
                    else:
                        values = full_for_fill(None, index_ic.size, fill_value)
                        values.flags.writeable = False
                        yield values

    def resize_blocks_by_callable(
        self,
        *,
        index_ic: tp.Optional[IndexCorrespondence],
        columns_ic: tp.Optional[IndexCorrespondence],
        fill_value: tp.Callable[[int, TDtypeAny | None], tp.Any],
    ) -> tp.Iterator[TNDArrayAny]:
        """
        Given index and column IndexCorrespondence objects, return a generator of resized blocks, extracting from self based on correspondence. Used for Frame.reindex(). Note that `fill_value` is provided with a callable derived from FillValueAuto.
        """
        col_src = 0  # NOTE: tracking col_src increases complexity but is needed for using FillValueAuto

        if columns_ic is None and index_ic is None:
            yield from self._blocks
        elif columns_ic is None and index_ic is not None:
            for b in self._blocks:
                if index_ic.is_subset:  # no rows added
                    # works for both 1d and 2d arrays
                    yield b[index_ic.iloc_src]
                    col_src += 1 if b.ndim == 1 else b.shape[1]
                elif b.ndim == 1:
                    fv = fill_value(col_src, b.dtype)
                    values = full_for_fill(b.dtype, index_ic.size, fv)
                    assign_via_ic(index_ic, b, values)
                    yield values
                    col_src += 1
                else:
                    for pos in range(b.shape[1]):
                        fv = fill_value(col_src, b.dtype)
                        values = full_for_fill(b.dtype, index_ic.size, fv)
                        assign_via_ic(index_ic, b[NULL_SLICE, pos], values)
                        yield values
                        col_src += 1

        elif columns_ic is not None and index_ic is None:
            if not columns_ic.has_common:  # no columns in common
                for _ in range(columns_ic.size):
                    # we do not have a block to get a reference dtype in this situation; if a caller is using FillValueAuto, this has to fail; if a caller has given a mapping or sequence, this needs to work
                    fv = fill_value(col_src, None)
                    values = full_for_fill(None, self.shape[0], fv)
                    values.flags.writeable = False
                    yield values
                    col_src += 1
            elif self.unified and columns_ic.is_subset:
                b = self._blocks[0]
                if b.ndim == 1:
                    yield b
                else:
                    yield b[NULL_SLICE, columns_ic.iloc_src]
            else:
                dst_to_src = dict(zip(columns_ic.iloc_dst, columns_ic.iloc_src))  # type: ignore [arg-type]
                for idx in range(columns_ic.size):
                    if idx in dst_to_src:
                        block_idx, block_col = self._index[dst_to_src[idx]]  # pyright: ignore
                        b = self._blocks[block_idx]
                        if b.ndim == 1:
                            yield b
                        else:
                            yield b[:, block_col]
                    else:  # just get an empty position, fill_value determines type
                        fv = fill_value(col_src, None)
                        values = full_for_fill(None, self.shape[0], fv)
                        values.flags.writeable = False
                        yield values
                    col_src += 1

        else:  # both defined
            assert columns_ic is not None and index_ic is not None  # mypy
            if not columns_ic.has_common and not index_ic.has_common:
                for _ in range(columns_ic.size):
                    fv = fill_value(col_src, None)
                    values = full_for_fill(None, index_ic.size, fv)
                    values.flags.writeable = False
                    yield values
                    col_src += 1
            elif self.unified and index_ic.is_subset and columns_ic.is_subset:
                b = self._blocks[0]
                if b.ndim == 1:
                    # NOTE: iloc_src is in the right order for dst
                    yield b[index_ic.iloc_src]
                else:
                    yield b[index_ic.iloc_src_fancy(), columns_ic.iloc_src]
                col_src += 1
            else:
                columns_dst_to_src = dict(zip(columns_ic.iloc_dst, columns_ic.iloc_src))  # type: ignore [arg-type]
                for idx in range(columns_ic.size):
                    if idx in columns_dst_to_src:
                        block_idx, block_col = self._index[columns_dst_to_src[idx]]  # pyright: ignore
                        b = self._blocks[block_idx]
                        if index_ic.is_subset:
                            if b.ndim == 1:
                                yield b[index_ic.iloc_src]
                            else:
                                # NOTE: this is not using iloc_dst if iloc_src is a different order
                                yield b[index_ic.iloc_src, block_col]
                        else:  # need an empty to fill, compatible with this
                            fv = fill_value(col_src, b.dtype)
                            values = full_for_fill(b.dtype, index_ic.size, fv)
                            if b.ndim == 1:
                                assign_via_ic(index_ic, b, values)
                            else:
                                assign_via_ic(index_ic, b[NULL_SLICE, block_col], values)
                            yield values
                    else:
                        fv = fill_value(col_src, None)
                        values = full_for_fill(None, index_ic.size, fv)
                        values.flags.writeable = False
                        yield values
                    col_src += 1

    # ---------------------------------------------------------------------------
    def sort(
        self,
        axis: int | np.integer[tp.Any],
        key: TILocSelector,
        kind: TSortKinds = DEFAULT_SORT_KIND,
    ) -> tp.Tuple[TypeBlocks, TNDArrayAny]:
        """While sorting generally happens at the Frame level, some lower level operations will benefit from sorting on type blocks directly.

        Args:
            axis: 0 orders columns by row(s) given by ``key``; 1 orders rows by column(s) given by ``key``.
        """
        values_for_sort: tp.Optional[TNDArrayAny] = None
        values_for_lex: TOptionalArrayList = None

        if axis == 0:  # get a column ordering based on one or more rows
            cfsa: TNDArrayAny = self._extract_array(row_key=key)
            if cfsa.ndim == 1:
                values_for_sort = cfsa
            elif cfsa.ndim == 2 and cfsa.shape[0] == 1:
                values_for_sort = cfsa[0]
            else:
                values_for_lex = [cfsa[i] for i in range(cfsa.shape[0] - 1, -1, -1)]

        elif axis == 1:  # get a row ordering based on one or more columns
            if isinstance(key, INT_TYPES):
                values_for_sort = self._extract_array_column(key)
            else:  # TypeBlocks from here
                cfs: TypeBlocks = self._extract(column_key=key)  # get TypeBlocks
                if cfs.shape[1] == 1:
                    values_for_sort = cfs._extract_array_column(0)
                else:
                    values_for_lex = [
                        cfs._extract_array_column(i)
                        for i in range(cfs.shape[1] - 1, -1, -1)
                    ]
        else:
            raise AxisInvalid(f'invalid axis: {axis}')  # pragma: no cover

        if values_for_lex is not None:
            order = np.lexsort(values_for_lex)
        elif values_for_sort is not None:
            order = np.argsort(values_for_sort, kind=kind)
        else:
            raise RuntimeError('unable to resovle sort type')  # pragma: no cover

        if axis == 0:
            return self._extract(column_key=order), order  # order columns
        return self._extract(row_key=order), order

    def group(
        self,
        axis: int,
        key: TILocSelector,
        drop: bool = False,
        kind: TSortKinds = DEFAULT_SORT_KIND,
    ) -> tp.Iterator[tp.Tuple[TNDArrayAny, TNDArrayAny | slice, TypeBlocks]]:
        """
        Axis 0 groups on column values, axis 1 groups on row values

        NOTE: this interface should only be called in situations when we do not need to align Index objects, as this does the sort and holds on to the ordering; the alternative is to sort and call group_sorted directly.
        """
        # NOTE: using a stable sort is necessary for groups to retain initial ordering.
        try:
            blocks, _ = self.sort(key=key, axis=not axis, kind=kind)
            use_sorted = True
        except TypeError:  # raised on sorting issue
            use_sorted = False

        # when calling these group function, as_array is False, and thus the third-returned item is always a TypeBlocks
        if use_sorted:
            yield from group_sorted(blocks, axis=axis, key=key, drop=drop)  # pyright: ignore
        else:
            yield from group_match(self, axis=axis, key=key, drop=drop)  # pyright: ignore

    def group_extract(
        self,
        axis: int,
        key: TILocSelector,
        extract: int,
        kind: TSortKinds = DEFAULT_SORT_KIND,
    ) -> tp.Iterator[tp.Tuple[TLabel, TNDArrayAny | slice, TNDArrayAny]]:
        """
        This interface will do an extraction on the opposite axis if the extraction is a single row/column.
        """
        # NOTE: using a stable sort is necessary for groups to retain initial ordering.
        try:
            blocks, _ = self.sort(key=key, axis=not axis, kind=kind)
            use_sorted = True
        except TypeError:
            use_sorted = False

        # when calling these group function, as_array is True, and thus the third-returned item is always an array
        if use_sorted:
            yield from group_sorted(
                blocks,  # pyright: ignore
                axis=axis,
                key=key,
                drop=False,
                extract=extract,
                as_array=True,
            )
        else:
            yield from group_match(
                self,  # pyright: ignore
                axis=axis,
                key=key,
                drop=False,
                extract=extract,
                as_array=True,
            )

    # ---------------------------------------------------------------------------
    # transformations resulting in reduced dimensionality

    def ufunc_axis_skipna(
        self,
        *,
        skipna: bool,
        axis: int,
        ufunc: TUFunc,
        ufunc_skipna: TUFunc,
        composable: bool,
        dtypes: tp.Sequence[TDtypeAny],
        size_one_unity: bool,
    ) -> TNDArrayAny:
        """Apply a function that reduces blocks to a single axis. Note that this only works in axis 1 if the operation can be applied more than once, first by block, then by reduced blocks. This will not work for a ufunc like argmin, argmax, where the result of the function cannot be compared to the result of the function applied on a different block.

        Args:
            composable: when True, the function application will return a correct result by applying the function to blocks first, and then the result of the blocks (i.e., add, prod); where observation count is relevant (i.e., mean, var, std), this must be False.
            dtypes: if we know the return type of func, we can provide it here to avoid having to use the row dtype. We provide multiple values and attempt to match the kind: if the row dtype is the same as a kind in this tuple, we will use the row dtype.

        Returns:
            As this is a reduction of axis where the caller (a Frame) is likely to return a Series, this function is not a generator of blocks, but instead just returns a consolidated 1d array.
        """
        if axis < 0 or axis > 1:
            raise AxisInvalid(f'invalid axis: {axis}')

        func: tp.Callable[..., TNDArrayAny] = partial(
            array_ufunc_axis_skipna,
            skipna=skipna,
            ufunc=ufunc,
            ufunc_skipna=ufunc_skipna,
        )

        result: TNDArrayAny
        if self.unified:
            result = func(array=column_2d_filter(self._blocks[0]), axis=axis)
            result.flags.writeable = False
            return result

        shape: TShape
        if axis == 0:
            # reduce all rows to 1d with column width
            shape = self._index.columns
            pos = 0  # used below undex axis 0
        elif composable:  # axis 1
            # reduce all columns to 2d blocks with 1 column
            shape = (self._index.rows, len(self._blocks))
        else:  # axis 1, not block composable
            # Cannot do block-wise processing, must resolve to single array and return
            array = blocks_to_array_2d(
                blocks=self._blocks,
                shape=self._index.shape,
                dtype=self._index.dtype,
            )
            result = func(array=array, axis=axis)
            result.flags.writeable = False
            return result

        dtype: None | TDtypeAny
        if dtypes:
            # If dtypes were specified, we know we have specific targets in mind for output
            # Favor row_dtype's kind if it is in dtypes, else take first of passed dtypes
            row_dtype = self._index.dtype
            for dt in dtypes:
                if row_dtype == dt.kind:
                    dtype = row_dtype
                    break
            else:  # no break encountered
                dtype = dtypes[0]
        else:
            # row_dtype gives us the compatible dtype for all blocks, whether we are reducing vertically (axis 0) or horizontall (axis 1)
            ufunc_selected = ufunc_skipna if skipna else ufunc
            dtype = ufunc_dtype_to_dtype(ufunc_selected, self._index.dtype)
            if dtype is None:
                # if we do not have a mapping for this function and row dtype, try to get a compatible type for the result of the function applied to each block
                block_dtypes = []
                dtf: None | TDtypeAny
                for b in self._blocks:
                    dtf = ufunc_dtype_to_dtype(ufunc_selected, b.dtype)
                    if dtf is not None:
                        block_dtypes.append(dtf)
                if len(block_dtypes) == len(self._blocks):  # if all resolved
                    dtype = resolve_dtype_iter(block_dtypes)
                else:  # assume row_dtype is appropriate
                    dtype = self._index.dtype

        out = np.empty(shape, dtype=dtype)
        for idx, b in enumerate(self._blocks):
            if axis == 0:  # Combine rows, end with columns shape.
                if size_one_unity and b.size == 1 and not skipna:
                    # No function call is necessary; if skipna could turn NaN to zero.
                    end = pos + 1
                    # Can assign an array, even 2D, as an element if size is 1
                    out[pos] = b
                elif b.ndim == 1:
                    end = pos + 1
                    out[pos] = func(array=b, axis=axis)
                else:
                    span = b.shape[1]
                    end = pos + span
                    if span == 1:  # just one column, reducing to one value
                        out[pos] = func(array=b, axis=axis)
                    else:
                        func(array=b, axis=axis, out=out[pos:end])
                pos = end
            else:
                # Combine columns, end with block length shape and then call func again, for final result
                if b.size == 1 and size_one_unity and not skipna:
                    out[NULL_SLICE, idx] = b
                elif b.ndim == 1:
                    # if this is a composable, numeric single columns we just copy it and process it later; but if this is a logical application (and, or) then it is already Boolean
                    if out.dtype == DTYPE_BOOL and b.dtype != DTYPE_BOOL:
                        # making 2D with axis 0 func will result in element-wise operation
                        out[NULL_SLICE, idx] = func(array=column_2d_filter(b), axis=1)
                    else:  # otherwise, keep as is
                        out[NULL_SLICE, idx] = b
                else:
                    func(array=b, axis=axis, out=out[NULL_SLICE, idx])

        if axis == 0:  # nothing more to do
            out.flags.writeable = False
            return out

        # If axis 1 and composable, can call function one more time on remaining components. Note that composability is problematic in cases where overflow is possible
        result = func(array=out, axis=1)
        result.flags.writeable = False
        return result

    # ---------------------------------------------------------------------------
    def __round__(self, decimals: int = 0) -> 'TypeBlocks':
        """
        Return a TypeBlocks rounded to the given decimals. Negative decimals round to the left of the decimal point.
        """
        func = partial(np.round, decimals=decimals)
        # for now, we do not expose application of rounding on a subset of blocks, but is doable by setting the column_key
        return self.__class__(
            blocks=list(self._ufunc_blocks(column_key=NULL_SLICE, func=func)),  # type: ignore
            index=self._index.copy(),
        )

    def __len__(self) -> int:
        """Length, as with NumPy and Pandas, is the number of rows. Note that A shape of (3, 0) will return a length of 3, even though there is no data."""
        return self._index.rows

    @doc_inject()
    def display(
        self,
        config: tp.Optional[DisplayConfig] = None,
        /,
        *,
        style_config: tp.Optional[StyleConfig] = None,
    ) -> Display:
        """{doc}

        Args:
            {config}
        """
        # NOTE: the TypeBlocks Display is not composed into other Displays

        config = config or DisplayActive.get()
        d: tp.Optional[Display] = None
        outermost = True  # only for the first
        idx = 0
        h: str | type
        for block in self._blocks:
            block = column_2d_filter(block)
            # NOTE: we do not expect 0 width arrays
            h = '' if idx > 0 else self.__class__

            display = Display.from_values(
                block, header=h, config=config, outermost=outermost
            )
            if not d:  # assign first
                d = display
                outermost = False
            else:
                d.extend_display(display)
            # explicitly enumerate so as to not count no-width blocks
            idx += 1

        if d is None:
            # if we do not have blocks, provide an empty display
            d = Display.from_values(
                (), header=self.__class__, config=config, outermost=outermost
            )
        return d

    # ---------------------------------------------------------------------------
    # extraction utilities

    def _key_to_block_slices(
        self, key: TILocSelector, retain_key_order: bool = True
    ) -> tp.Iterator[tp.Tuple[int, tp.Union[slice, int]]]:
        """
        For a column key (an integer, slice, iterable, Boolean array), generate pairs of (block_idx, slice or integer) to cover all extractions. First, get the relevant index values (pairs of block id, column id), then convert those to contiguous slices. NOTE: integers are only returned when the input key is itself an integer.

        Args:
            retain_key_order: if False, returned slices will be in ascending order.

        Returns:
            A generator iterable of pairs, where values are pairs of either a block index and slice or, a block index and column index.
        """
        if key is None or (key.__class__ is slice and key == NULL_SLICE):
            yield from self._index.iter_block()
        elif isinstance(key, INT_TYPES):
            # the index has the pair block, column integer
            try:
                yield self._index[key]  # type: ignore
            except IndexError as e:
                raise KeyError(key) from e
        else:  # all cases where we try to get contiguous slices
            try:
                yield from self._index.iter_contiguous(
                    key, ascending=not retain_key_order
                )
            except TypeError as e:
                # BlockIndex will raise TypeErrors in a number of cases of bad inputs; some of these are not easy to change
                raise KeyError(key) from e

    # ---------------------------------------------------------------------------
    def _mask_blocks(
        self,
        row_key: TILocSelector = None,
        column_key: TILocSelector = None,
    ) -> tp.Iterator[TNDArrayAny]:
        """Return Boolean blocks of the same size and shape, where key selection sets values to True."""

        # this selects the columns; but need to return all blocks

        # block slices must be in ascending order, not key order
        block_slices = self._key_to_block_slices(column_key, retain_key_order=False)
        target_block_idx = target_slice = None
        targets_remain = True

        for block_idx, b in enumerate(self._blocks):
            mask = np.full(b.shape, False, dtype=bool)

            while targets_remain:
                # get target block and slice
                if target_block_idx is None:  # can be zero
                    try:
                        target_block_idx, target_slice = next(block_slices)
                    except StopIteration:
                        targets_remain = False
                        break

                if block_idx != target_block_idx:
                    break  # need to advance blocks

                if b.ndim == 1:  # given 1D array, our row key is all we need
                    mask[row_key] = True
                else:
                    if row_key is None:
                        mask[:, target_slice] = True
                    else:
                        mask[row_key, target_slice] = True

                target_block_idx = target_slice = None

            yield mask

    def _astype_blocks(
        self, column_key: TILocSelector, dtype: TDtypeSpecifier
    ) -> tp.Iterator[TNDArrayAny]:
        """
        Given any column selection, apply a single dtype.
        Generator-producer of np.ndarray.
        """
        dtype = validate_dtype_specifier(dtype)

        # block slices must be in ascending order, not key order
        block_slices = iter(self._key_to_block_slices(column_key, retain_key_order=False))

        target_slice: tp.Optional[tp.Union[slice, int]]

        target_block_idx = target_slice = None
        targets_remain = True
        target_start: int

        for block_idx, b in enumerate(self._blocks):
            parts = []
            part_start_last = 0

            while targets_remain:
                # get target block and slice
                if target_block_idx is None:  # can be zero
                    try:
                        target_block_idx, target_slice = next(block_slices)
                    except StopIteration:
                        targets_remain = False
                        break

                if block_idx != target_block_idx:
                    break  # need to advance blocks

                if dtype == b.dtype:
                    target_block_idx = target_slice = None
                    continue  # there may be more slices for this block

                if b.ndim == 1:  # given 1D array, our row key is all we need
                    # parts.append(b.astype(dtype))
                    parts.append(astype_array(b, dtype))
                    part_start_last = 1
                    target_block_idx = target_slice = None
                    break

                assert target_slice is not None
                # target_slice can be a slice or an integer
                if target_slice.__class__ is slice:
                    target_start = target_slice.start  # type: ignore
                    target_stop = target_slice.stop  # type: ignore
                else:  # it is an integer
                    target_start = target_slice  # type: ignore
                    target_stop = target_slice + 1  # type: ignore

                assert target_start is not None and target_stop is not None
                if target_start > part_start_last:
                    # yield un changed components before and after
                    parts.append(b[NULL_SLICE, slice(part_start_last, target_start)])

                # parts.append(b[NULL_SLICE, target_slice].astype(dtype))
                parts.append(astype_array(b[NULL_SLICE, target_slice], dtype))
                part_start_last = target_stop

                target_block_idx = target_slice = None

            # if this is a 1D block, we either convert it or do not, and thus either have parts or not, and do not need to get other part pieces of the block
            if b.ndim != 1 and part_start_last < b.shape[1]:
                parts.append(b[NULL_SLICE, slice(part_start_last, None)])

            if not parts:
                yield b  # no change for this block
            else:
                yield from parts

    def _astype_blocks_from_dtypes(
        self,
        dtype_factory: tp.Callable[[int], TDtypeAny | None],
    ) -> tp.Iterator[TNDArrayAny]:
        """
        Generator producer of np.ndarray.

        Args:
            dtype_factory: a function for mapping iloc positions to dtypes, often produced with get_col_dtype_factory().
        """
        iloc = 0
        for b in self._blocks:
            if b.ndim == 1:
                dtype = dtype_factory(iloc)
                if dtype is not None:
                    yield astype_array(b, dtype)
                else:
                    yield b
                iloc += 1
            else:
                group_start = 0
                for pos in range(b.shape[1]):
                    dtype = dtype_factory(iloc)
                    if pos == 0:
                        dtype_last = dtype
                    elif dtype != dtype_last:
                        # this dtype is different, so need to cast all up to (but not including) this one
                        if dtype_last is not None:
                            yield astype_array(
                                b[NULL_SLICE, slice(group_start, pos)], dtype_last
                            )
                        else:
                            yield b[NULL_SLICE, slice(group_start, pos)]
                        group_start = pos  # this is the start of a new group
                        dtype_last = dtype
                    # else: dtype is the same
                    iloc += 1
                # there is always one more to yield
                if dtype_last is not None:
                    yield astype_array(
                        b[NULL_SLICE, slice(group_start, None)], dtype_last
                    )
                else:
                    yield b[NULL_SLICE, slice(group_start, None)]

    def _consolidate_select_blocks(
        self,
        column_key: TILocSelector,
    ) -> tp.Iterator[TNDArrayAny]:
        """
        Given any column selection, consolidate when possible within that region.
        Generator-producer of np.ndarray.
        """
        # block slices must be in ascending order, not key order
        block_slices = self._key_to_block_slices(column_key, retain_key_order=False)

        target_slice: tp.Optional[tp.Union[slice, int]]

        target_block_idx = target_slice = None
        targets_remain = True
        target_start: int
        consolidate: tp.List[TNDArrayAny] = []

        def consolidate_and_clear() -> tp.Iterator[TNDArrayAny]:
            yield from self.consolidate_blocks(consolidate)
            consolidate.clear()

        for block_idx, b in enumerate(self._blocks):
            part_start_last = 0  # non-inclusive upper boundary, used to signal if block components have been read

            while targets_remain:
                # get target block and slice
                if target_block_idx is None:  # can be zero
                    try:
                        target_block_idx, target_slice = next(block_slices)
                    except StopIteration:
                        targets_remain = False
                        break

                if block_idx != target_block_idx:
                    yield from consolidate_and_clear()
                    yield b
                    part_start_last = 1 if b.ndim == 1 else b.shape[1]
                    break  # need to advance blocks

                if b.ndim == 1:
                    consolidate.append(b)
                    part_start_last = 1
                    target_block_idx = target_slice = None
                    break  # move on to next block

                assert target_slice is not None
                # target_slice can be a slice or an integer
                if target_slice.__class__ is slice:
                    target_start = (
                        target_slice.start  # type: ignore
                        if target_slice.start is not None  # type: ignore
                        else part_start_last
                    )
                    target_stop = (
                        target_slice.stop if target_slice.stop is not None else b.shape[1]  # type: ignore
                    )
                else:  # it is an integer
                    target_start = target_slice  # type: ignore
                    target_stop = target_slice + 1  # type: ignore

                assert target_start is not None and target_stop is not None

                if target_start > part_start_last:
                    yield from consolidate_and_clear()
                    # yield un changed components before and after
                    yield b[NULL_SLICE, slice(part_start_last, target_start)]

                consolidate.append(b[NULL_SLICE, target_slice])

                part_start_last = target_stop
                target_block_idx = target_slice = None
                if part_start_last == b.shape[1]:
                    # NOTE: this might be an optimization for related routines
                    break  # done with this block

            # if there are columns left in the block that are not targeted
            if b.ndim != 1 and part_start_last < b.shape[1]:
                yield from consolidate_and_clear()
                yield b[NULL_SLICE, slice(part_start_last, None)]
            elif part_start_last == 0:
                # no targets remain, and no partial blocks are targets
                yield from consolidate_and_clear()
                yield b

        yield from consolidate_and_clear()

    def _ufunc_blocks(
        self, column_key: TILocSelector, func: TUFunc
    ) -> tp.Iterator[TNDArrayAny]:
        """
        Return a new blocks after processing each columnar block with the passed ufunc. It is assumed the ufunc will retain the shape of the input 1D or 2D array. All blocks must be processed, which is different than _astype_blocks, which can check the type and skip processing some blocks.

        Generator producer of np.ndarray.
        """
        # block slices must be in ascending order, not key order
        block_slices = iter(self._key_to_block_slices(column_key, retain_key_order=False))

        target_slice: tp.Optional[tp.Union[slice, int]]

        target_block_idx = target_slice = None
        targets_remain = True
        target_start: int

        for block_idx, b in enumerate(self._blocks):
            parts = []
            part_start_last = 0

            while targets_remain:
                # get target block and slice
                if target_block_idx is None:  # can be zero
                    try:
                        target_block_idx, target_slice = next(block_slices)
                    except StopIteration:
                        targets_remain = False
                        break

                if block_idx != target_block_idx:
                    break  # need to advance blocks

                if b.ndim == 1:  # given 1D array, our row key is all we need
                    parts.append(func(b))
                    part_start_last = 1
                    target_block_idx = target_slice = None
                    break

                # target_slice can be a slice or an integer
                if target_slice.__class__ is slice:
                    if target_slice == NULL_SLICE:
                        target_start = 0
                        target_stop = b.shape[1]
                    else:
                        target_start = target_slice.start  # type: ignore
                        target_stop = target_slice.stop  # type: ignore
                else:  # it is an integer
                    target_start = target_slice  # type: ignore
                    target_stop = target_slice + 1  # type: ignore

                assert target_start is not None and target_stop is not None
                if target_start > part_start_last:
                    # yield unchanged components before and after
                    parts.append(b[NULL_SLICE, slice(part_start_last, target_start)])

                # apply func
                parts.append(func(b[NULL_SLICE, target_slice]))
                part_start_last = target_stop

                target_block_idx = target_slice = None

            # if this is a 1D block, we either convert it or do not, and thus either have parts or not, and do not need to get other part pieces of the block
            if b.ndim != 1 and part_start_last < b.shape[1]:
                parts.append(b[:, slice(part_start_last, None)])

            if not parts:
                yield b  # no change for this block
            else:
                yield from parts

    def _drop_blocks(
        self,
        row_key: TILocSelector = None,
        column_key: TILocSelector = None,
    ) -> tp.Iterator[TNDArrayAny]:
        """
        Generator producer of np.ndarray. Note that this appraoch should be more efficient than using selection/extraction, as here we are only concerned with columns.

        Args:
            column_key: Selection of columns to leave out of blocks.
        """
        if column_key is None:
            # the default should not be the null slice, which would drop all
            block_slices: tp.Iterator[tp.Tuple[int, tp.Union[slice, int]]] = iter(())
        else:
            if not self._blocks:
                raise IndexError('cannot drop columns from zero-blocks')
            # block slices must be in ascending order, not key order
            block_slices = iter(
                self._key_to_block_slices(column_key, retain_key_order=False)
            )

        if row_key.__class__ is np.ndarray and row_key.dtype == bool:  # type: ignore
            # row_key is used with np.delete, which does not support Boolean arrays; instead, convert to an array of integers
            row_key = PositionsAllocator.get(len(row_key))[row_key]  # type: ignore

        target_block_idx = target_slice = None
        targets_remain = True
        target_start: int

        for block_idx, b in enumerate(self._blocks):
            # for each block, we evaluate if we have any targets in that block and update the block accordingly; otherwise, we yield the block unchanged

            parts = []
            drop_block = False  # indicate entire block is dropped
            part_start_last = (
                0  # within this block, keep track of where our last change was started
            )

            while targets_remain:
                # get target block and slice; this is what we want to remove
                if target_block_idx is None:  # can be zero
                    try:
                        target_block_idx, target_slice = next(block_slices)
                    except StopIteration:
                        targets_remain = False
                        break

                if block_idx != target_block_idx:
                    break  # need to advance blocks

                if b.ndim == 1 or b.shape[1] == 1:  # given 1D array or 2D, 1 col array
                    part_start_last = 1
                    target_block_idx = target_slice = None
                    drop_block = True
                    break

                # target_slice can be a slice or an integer
                if target_slice.__class__ is slice:
                    if target_slice == NULL_SLICE:
                        target_start = 0
                        target_stop = b.shape[1]
                    else:
                        target_start = target_slice.start  # type: ignore
                        target_stop = target_slice.stop  # type: ignore
                else:  # it is an integer
                    target_start = target_slice  # type: ignore
                    target_stop = target_slice + 1  # type: ignore

                # assert target_start is not None and target_stop is not None
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

    def _shift_blocks_fill_by_element(
        self,
        row_shift: int = 0,
        column_shift: int = 0,
        wrap: bool = True,
        fill_value: tp.Any = np.nan,
    ) -> tp.Iterator[TNDArrayAny]:
        """
        Shift type blocks independently on rows or columns. When ``wrap`` is True, the operation is a roll-style shift; when ``wrap`` is False, shifted-out values are not replaced and are filled with ``fill_value``.

        Args:
            column_shift: a positive value moves column data to the right.
        """
        row_count = self._index.rows
        column_count = self._index.columns

        # new start index is the opposite of the shift; if shifting by 2, the new start is the second from the end
        index_start_pos = -(column_shift % column_count)
        row_start_pos = -(row_shift % row_count)

        block_head_iter: tp.Iterable[TNDArrayAny]
        block_tail_iter: tp.Iterable[TNDArrayAny]

        # possibly be truthy
        # index is columns here
        if wrap and index_start_pos == 0 and row_start_pos == 0:
            yield from self._blocks
        elif not wrap and column_shift == 0 and row_shift == 0:
            yield from self._blocks
        else:
            block_start_idx, block_start_column = self._index[
                index_start_pos
            ]  # modulo adjusted
            block_start = self._blocks[block_start_idx]

            if not wrap and abs(column_shift) >= column_count:  # no data will be retained
                # blocks will be set below
                block_head_iter = ()
                block_tail_iter = ()
            elif block_start_column == 0:  # we are starting at the start of the block
                block_head_iter = chain(
                    (block_start,), self._blocks[block_start_idx + 1 :]
                )
                block_tail_iter = self._blocks[:block_start_idx]
            else:
                block_head_iter = chain(
                    (block_start[:, block_start_column:],),
                    self._blocks[block_start_idx + 1 :],
                )
                block_tail_iter = chain(
                    self._blocks[:block_start_idx],
                    (block_start[:, :block_start_column],),
                )

            if not wrap:
                # provide a consolidated single block for missing values
                shape = (row_count, min(column_count, abs(column_shift)))
                empty = np.full(shape, fill_value)
                empty.flags.writeable = False

                # NOTE: this will overwrite values set above
                if column_shift > 0:
                    block_head_iter = (empty,)
                elif column_shift < 0:
                    block_tail_iter = (empty,)

            # NOTE: might consider not rolling when yielding an empty array
            for b in chain(block_head_iter, block_tail_iter):
                if (wrap and row_start_pos == 0) or (not wrap and row_shift == 0):
                    yield b
                else:  # do all row shifting here
                    array = array_shift(
                        array=b,
                        shift=row_shift,
                        axis=0,
                        wrap=wrap,
                        fill_value=fill_value,
                    )
                    array.flags.writeable = False
                    yield array

    def _shift_blocks_fill_by_callable(
        self,
        row_shift: int,
        column_shift: int,
        wrap: bool,
        get_col_fill_value: tp.Callable[[int, TDtypeAny | None], tp.Any],
    ) -> tp.Iterator[TNDArrayAny]:
        """
        Shift type blocks independently on rows or columns. When ``wrap`` is True, the operation is a roll-style shift; when ``wrap`` is False, shifted-out values are not replaced and are filled with ``get_col_fill_value``.
        """
        row_count = self._index.rows
        column_count = self._index.columns

        # new start index is the opposite of the shift; if shifting by 2, the new start is the second from the end
        index_start_pos = -(column_shift % column_count)
        row_start_pos = -(row_shift % row_count)

        block_head_iter: tp.Iterable[TNDArrayAny]
        block_tail_iter: tp.Iterable[TNDArrayAny]

        # possibly be truthy
        # index is columns here
        if wrap and index_start_pos == 0 and row_start_pos == 0:
            yield from self._blocks
        elif not wrap and column_shift == 0 and row_shift == 0:
            yield from self._blocks
        else:
            block_start_idx, block_start_column = self._index[index_start_pos]
            block_start = self._blocks[block_start_idx]

            if not wrap and abs(column_shift) >= column_count:  # no data will be retained
                # blocks will be set below
                block_head_iter = ()
                block_tail_iter = ()
            elif block_start_column == 0:
                # we are starting at the block, no tail, always yield;  captures all 1 dim block cases
                block_head_iter = chain(
                    (block_start,), self._blocks[block_start_idx + 1 :]
                )
                block_tail_iter = self._blocks[:block_start_idx]
            else:
                block_head_iter = chain(
                    (block_start[:, block_start_column:],),
                    self._blocks[block_start_idx + 1 :],
                )
                block_tail_iter = chain(
                    self._blocks[:block_start_idx],
                    (block_start[:, :block_start_column],),
                )

            if not wrap:
                # get the lesser of the existing number of columns or the shift
                fill_count = min(column_count, abs(column_shift))
                if column_shift > 0:
                    block_head_iter = (
                        np.full(row_count, get_col_fill_value(i, None))
                        for i in range(fill_count)
                    )
                elif column_shift < 0:
                    block_tail_iter = (
                        np.full(row_count, get_col_fill_value(i, None))
                        for i in range(column_count - fill_count, column_count)
                    )

            # NOTE: might consider not rolling when yielding an empty array
            col_idx = 0
            for b in chain(block_head_iter, block_tail_iter):
                if (wrap and row_start_pos == 0) or (not wrap and row_shift == 0):
                    yield b
                else:
                    for i in range(1 if b.ndim == 1 else b.shape[1]):
                        fv = get_col_fill_value(col_idx, b.dtype)
                        array = array_shift(
                            array=b if b.ndim == 1 else b[NULL_SLICE, i],
                            shift=row_shift,
                            axis=0,
                            wrap=wrap,
                            fill_value=fv,
                        )
                        array.flags.writeable = False
                        yield array
                        col_idx += 1

    # ---------------------------------------------------------------------------
    def _assign_from_iloc_by_blocks(
        self,
        values: tp.Iterable[TNDArrayAny],
        row_key: TILocSelector = None,
        column_key: TILocSelector = None,
    ) -> tp.Iterator[TNDArrayAny]:
        """
        Given row, column key selections, assign from an iterable of 1D or 2D block arrays.
        """
        target_block_slices = self._key_to_block_slices(column_key, retain_key_order=True)
        target_key: tp.Optional[tp.Union[int, slice]] = None
        target_block_idx: tp.Optional[int] = None
        targets_remain: bool = True
        target_is_slice: bool
        row_key_is_null_slice = row_key is None or (
            isinstance(row_key, slice) and row_key == NULL_SLICE
        )

        # get a mutable list in reverse order for pop/pushing
        values_source = list(values)
        values_source.reverse()

        for block_idx, b in enumerate(self._blocks):
            assigned_stop = 0  # exclusive maximum

            while targets_remain:
                if target_block_idx is None:
                    try:
                        target_block_idx, target_key = next(target_block_slices)
                    except StopIteration:
                        targets_remain = False  # stop entering while loop
                        break
                    target_is_slice = target_key.__class__ is slice
                    target_is_null_slice = target_is_slice and target_key == NULL_SLICE

                if block_idx != target_block_idx:
                    break  # need to advance blocks, keep targets

                if target_is_slice:
                    if target_is_null_slice:
                        t_start = 0
                        t_stop = b.shape[1] if b.ndim == 2 else 1
                        t_width = t_stop
                    else:
                        t_start = target_key.start  # type: ignore
                        t_stop = target_key.stop  # type: ignore
                        t_width = t_stop - t_start
                else:
                    t_start = target_key  # type: ignore
                    t_stop = t_start + 1  # pyright: ignore
                    t_width = 1

                # yield all block slices up to the target, then then the target; remain components of this block will be yielded on next iteration (if there is another target) or out of while by looking at assigned stop
                if t_start != 0:
                    yield b[NULL_SLICE, assigned_stop:t_start]
                if row_key_is_null_slice:
                    yield from get_block_match(t_width, values_source)
                else:
                    assigned_blocks = tuple(
                        column_2d_filter(a)
                        for a in get_block_match(t_width, values_source)
                    )
                    assigned_dtype = resolve_dtype_iter(
                        chain((a.dtype for a in assigned_blocks), (b.dtype,))
                    )
                    if b.ndim == 2:
                        assigned = b[NULL_SLICE, t_start:t_stop].astype(assigned_dtype)
                        assigned[row_key, NULL_SLICE] = concat_resolved(
                            assigned_blocks, axis=1
                        )
                    else:
                        assigned = b.astype(assigned_dtype)
                        assigned[row_key] = column_1d_filter(assigned_blocks[0])
                    assigned.flags.writeable = False
                    yield assigned

                assigned_stop = t_stop
                target_block_idx = target_key = None  # get a new target

            if assigned_stop == 0:
                yield b  # no targets were found for this block; or no targets remain
            elif b.ndim == 1 and assigned_stop == 1:
                pass
            elif b.ndim == 2 and assigned_stop < b.shape[1]:
                yield b[NULL_SLICE, assigned_stop:]

    def _assign_from_iloc_core(
        self,
        *,
        row_key: TILocSelector = None,
        column_key: TILocSelector = None,
        value: tp.Any = None,
        assign_inner: tp.Callable[
            [
                tp.Any,
                TNDArrayAny,
                TILocSelector,
                TILocSelector,
                TShape,
                bool,
                bool,
                bool,
            ],
            tp.Tuple[tp.Any, TNDArrayAny],
        ],
    ) -> tp.Iterator[TNDArrayAny]:
        # NOTE: this requires column_key to be ordered to work; we cannot use retain_key_order=False, as the passed `value` is ordered by that key
        target_block_slices = self._key_to_block_slices(column_key, retain_key_order=True)
        target_key: tp.Optional[tp.Union[int, slice]] = None
        target_block_idx: tp.Optional[int] = None
        targets_remain: bool = True
        target_is_slice: bool
        row_key_is_null_slice = row_key is None or (
            isinstance(row_key, slice) and row_key == NULL_SLICE
        )
        row_target = NULL_SLICE if row_key_is_null_slice else row_key

        for block_idx, b in enumerate(self._blocks):
            assigned_stop = 0  # exclusive maximum

            while targets_remain:
                if target_block_idx is None:  # can be zero
                    try:
                        target_block_idx, target_key = next(target_block_slices)
                    except StopIteration:
                        targets_remain = False  # stop entering while loop
                        break
                    target_is_slice = target_key.__class__ is slice
                    target_is_null_slice = target_is_slice and target_key == NULL_SLICE

                if block_idx != target_block_idx:
                    break  # need to advance blocks, keep targets

                t_start: int
                if not target_is_slice:
                    t_start = target_key  # type: ignore
                elif target_is_null_slice:
                    t_start = 0
                else:
                    t_start = target_key.start  # type: ignore

                if (
                    t_start > assigned_stop
                ):  # yield component from the last assigned position
                    b_component = b[
                        NULL_SLICE, slice(assigned_stop, t_start)
                    ]  # keeps writeable=False
                    yield b_component

                # at least one target we need to apply in the current block.
                block_is_column = b.ndim == 1 or (b.ndim > 1 and b.shape[1] == 1)

                # add empty components for the assignment region
                t_shape: TShape
                if target_is_slice and not block_is_column:
                    if target_is_null_slice:
                        t_width = b.shape[1]
                        t_shape = b.shape
                    else:  # can assume this slice has no strides
                        t_width = target_key.stop - target_key.start  # type: ignore
                        t_shape = (b.shape[0], t_width)
                else:  # b.ndim == 1 or target is an integer: get a 1d array
                    t_width = 1
                    t_shape = b.shape[0]

                value, block = assign_inner(  # type: ignore
                    value=value,  # pyright: ignore
                    block=b,
                    row_target=row_target,
                    target_key=target_key,
                    t_shape=t_shape,
                    target_is_slice=target_is_slice,
                    block_is_column=block_is_column,
                    row_key_is_null_slice=row_key_is_null_slice,
                )
                yield block

                assigned_stop = t_start + t_width
                target_block_idx = target_key = None  # get a new target

            if assigned_stop == 0:
                yield b  # no targets were found for this block; or no targets remain
            elif b.ndim == 1 and assigned_stop == 1:
                pass
            elif b.ndim == 2 and assigned_stop < b.shape[1]:
                yield b[NULL_SLICE, assigned_stop:]

    def _assign_from_iloc_by_unit(
        self,
        row_key: TILocSelector = None,
        column_key: TILocSelector = None,
        value: tp.Any = None,
    ) -> tp.Iterator[TNDArrayAny]:
        """Assign a single value (a tuple, array, or element) into all blocks, returning blocks of the same size and shape.

        Args:
            column_key: must be sorted in ascending order.
        """
        # given a list or other non-tuple iterable, convert it to an array here to determine dtype handling
        if (
            value.__class__ is not np.ndarray
            and hasattr(value, '__len__')
            and not isinstance(value, tuple)
            and not isinstance(value, STRING_TYPES)
        ):
            value, _ = iterable_to_array_1d(value)

        yield from self._assign_from_iloc_core(
            row_key=row_key,
            column_key=column_key,
            value=value,
            assign_inner=assign_inner_from_iloc_by_unit,  # type: ignore
        )

    def _assign_from_iloc_by_sequence(
        self,
        *,
        value: tp.Sequence[tp.Any],
        row_key: TILocSelector = None,
        column_key: TILocSelector = None,
    ) -> tp.Iterator[TNDArrayAny]:
        """Assign an iterable of appropriate size (a tuple) into all blocks, returning blocks of the same size and shape. If row-key is a multiple, the values will be replicated in all rows."""
        yield from self._assign_from_iloc_core(
            row_key=row_key,
            column_key=column_key,
            value=value,
            assign_inner=assign_inner_from_iloc_by_sequence,  # type: ignore
        )

    # ---------------------------------------------------------------------------

    def _assign_from_boolean_blocks_by_unit(
        self,
        targets: tp.Iterable[TNDArrayAny],
        value: object,
        value_valid: tp.Optional[TNDArrayAny],
    ) -> tp.Iterator[TNDArrayAny]:
        """Assign value (a single element or a matching array) into all blocks based on a Boolean arrays of shape equal to each block in these blocks, yielding blocks of the same size and shape. Value is set where the Boolean is True.

        Args:
            value: Must be a single value or an array
            value_valid: same size Boolean area to be combined with targets
        """
        if value.__class__ is np.ndarray:
            value_dtype = value.dtype  # type: ignore
            is_element = False
            if value_valid is not None:
                assert value_valid.shape == self.shape
        else:  # assumed to be non-string, non-iterable
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

                value_part = value[NULL_SLICE, value_slice][target]  # type: ignore
                start = end
            else:
                value_part = value

            # evaluate after updating target
            if not target.any():  # works for ndim 1 and 2
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

    def _assign_from_boolean_blocks_by_callable(
        self,
        targets: tp.Iterable[TNDArrayAny],
        get_col_fill_value: tp.Callable[[int, TDtypeAny], tp.Any],
    ) -> tp.Iterator[TNDArrayAny]:
        """Assign value (a single element) into blocks by integer column, based on a Boolean arrays of shape equal to each block in these blocks, yielding blocks of the same size and shape. The result of calling func with the column number is the value set where the Boolean is True.

        Args:
            get_col_fill_value: A callable that given the index position returns the value
            value_valid: same size Boolean area to be combined with targets
        """
        col = 0
        # value_slice: tp.Union[int, slice]

        for block, target in zip_longest(self._blocks, targets):
            # evaluate after updating target
            if not target.any():  # works for ndim 1 and 2
                yield block
                col += 1 if block.ndim == 1 else block.shape[1]
            elif block.ndim == 1:
                value = get_col_fill_value(col, block.dtype)
                value_dtype = dtype_from_element(value)
                assigned_dtype = resolve_dtype(value_dtype, block.dtype)
                if block.dtype == assigned_dtype:
                    assigned = block.copy()
                else:
                    assigned = block.astype(assigned_dtype)

                assigned[target] = value
                assigned.flags.writeable = False
                yield assigned
                col += 1
            else:
                target_flat: TNDArray1DBool = target.any(axis=0)  # type: ignore
                # NOTE: this implementation does maximal de-consolidation to ensure type resolution; this might instead collect fill values and find if they are unique accross blocks, but this would require them to be hashable or otherwise comparable, which they may not be
                for i in range(block.shape[1]):
                    if not target_flat[i]:
                        # no targets in this columns
                        yield block[NULL_SLICE, i]  # slices are immutable
                    else:
                        value = get_col_fill_value(col, block.dtype)
                        value_dtype = dtype_from_element(value)
                        assigned_dtype = resolve_dtype(value_dtype, block.dtype)

                        if block.dtype == assigned_dtype:
                            assigned = block[NULL_SLICE, i].copy()
                        else:
                            assigned = block[NULL_SLICE, i].astype(assigned_dtype)

                        assigned[target[NULL_SLICE, i]] = value
                        assigned.flags.writeable = False
                        yield assigned
                    col += 1

    def _assign_from_boolean_blocks_by_blocks(
        self,
        targets: tp.Iterable[TNDArrayAny],
        values: tp.Sequence[TNDArrayAny],
    ) -> tp.Iterator[TNDArrayAny]:
        """Assign values (derived from an iterable of arrays) into all blocks based on a Boolean arrays of shape equal to each block in these blocks. This yields blocks of the same size and shape. Value is set where the Boolean is True.

        This approach minimizes type coercion by reducing assigned values to columnar types.

        Args:
            targets: Boolean arrays aligned to blocks
            values: Sequence of 1D arrays with aggregate shape equal to targets
        """
        start = 0
        for block, target in zip_longest(self._blocks, targets):
            if block is None or target is None:
                raise RuntimeError('blocks or targets do not align')

            if block.ndim == 1:
                end = start + 1
            else:
                end = start + block.shape[1]

            if not target.any():  # works for ndim 1 and 2
                yield block
            else:
                values_for_block = values[start:end]  # get 1D array from tuple
                # target and block must be ndim=2
                for i in range(end - start):
                    if block.ndim == 1:  # will only do one iteration
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
                            assigned_dtype = resolve_dtype(
                                values_to_assign.dtype, block.dtype
                            )
                            if block.dtype == assigned_dtype:
                                assigned = block_sub.copy()
                            else:
                                assigned = block_sub.astype(assigned_dtype)
                            assigned[target_sub] = values_to_assign[target_sub]
                            assigned.flags.writeable = False
                            yield assigned
            start = end  # always update start

    # ---------------------------------------------------------------------------

    def _assign_from_bloc_by_unit(
        self,
        bloc_key: TNDArrayAny,
        value: tp.Any,  # an array, or element for single assignment
    ) -> tp.Iterator[TNDArrayAny]:
        """
        Given an Boolean array of targets, fill targets from value, where value is either a single value or an array. Unlike with _assign_from_boolean_blocks_by_unit, this method takes a single block_key.

        Args:
            value: can be a single element, or a single 2D array of shape equal to self.
        """
        if value.__class__ is np.ndarray:
            value_dtype = value.dtype
            is_element = False
            assert value.shape == self.shape
        else:
            value_dtype = dtype_from_element(value)
            is_element = True

        t_start = 0
        target_slice: tp.Union[int, slice]

        for block in self._blocks:
            if block.ndim == 1:
                t_end = t_start + 1
                target_slice = t_start
            else:
                t_end = t_start + block.shape[1]
                target_slice = slice(t_start, t_end)

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

            t_start = t_end  # always update start

    def _assign_from_bloc_by_blocks(
        self,
        bloc_key: TNDArrayAny,
        values: tp.Sequence[TNDArrayAny],
    ) -> tp.Iterator[TNDArrayAny]:
        """
        A given a single Boolean array bloc-key, assign with `value`, a sequence of appropriately sized arrays.
        """
        t_start = 0
        target_slice: tp.Union[int, slice]

        # get a mutable list in reverse order for pop/pushing
        values_source = list(values)
        values_source.reverse()

        for block in self._blocks:
            if block.ndim == 1:
                t_end = t_start + 1
                target_slice = t_start
                t_width = 1
            else:
                t_end = t_start + block.shape[1]
                target_slice = slice(t_start, t_end)
                t_width = t_end - t_start

            # target will only be 1D when block is 1d
            target = bloc_key[NULL_SLICE, target_slice]

            if not target.any():
                # must extract from values source even if no match
                for _ in get_block_match(t_width, values_source):
                    pass
                yield block
            else:  # something to assign, draw from values
                a_start = 0
                for value_part in get_block_match(t_width, values_source):
                    # loop over one or more blocks less then or equal to target
                    if value_part.ndim == 1:
                        a_end = a_start + 1
                    else:
                        a_end = a_start + value_part.shape[1]

                    assigned_dtype = resolve_dtype(value_part.dtype, block.dtype)

                    if block.ndim == 1:  # target is 1d
                        if block.dtype == assigned_dtype:
                            assigned = block.copy()
                        else:
                            assigned = block.astype(assigned_dtype)
                        # target is 1D, value_part may be 1D, 2D
                        assigned[target] = column_1d_filter(value_part)[target]

                    else:  # block is 2d, target is 2d, extract appropriate value
                        value_part = column_2d_filter(value_part)
                        a_slice = slice(a_start, a_end)
                        target_part = target[NULL_SLICE, a_slice]

                        if block.dtype == assigned_dtype:
                            assigned = block[NULL_SLICE, a_slice].copy()
                        else:
                            assigned = block[NULL_SLICE, a_slice].astype(assigned_dtype)

                        assigned[target_part] = value_part[target_part]

                    assigned.flags.writeable = False
                    yield assigned

                    a_start = a_end

            t_start = t_end

    def _assign_from_bloc_by_coordinate(
        self,
        bloc_key: TNDArrayAny,
        values_map: tp.Dict[tp.Tuple[int, int], tp.Any],
        values_dtype: TDtypeAny,
    ) -> tp.Iterator[TNDArrayAny]:
        """
        For assignment from a Series of coordinate/value pairs, as extracted via a bloc selection.

        Args:
            values_coord: will be sorted by column
        """
        t_start = 0
        target_slice: tp.Union[int, slice]

        for block in self._blocks:
            if block.ndim == 1:
                t_end = t_start + 1
                target_slice = t_start
            else:
                t_end = t_start + block.shape[1]
                target_slice = slice(t_start, t_end)

            # target will only be 1D when block is 1d
            target = bloc_key[NULL_SLICE, target_slice]

            if not target.any():
                yield block
            else:
                assigned_dtype = resolve_dtype(values_dtype, block.dtype)
                if block.dtype == assigned_dtype:
                    assigned = block.copy()
                else:
                    assigned = block.astype(assigned_dtype)

                # get coordinates and fill
                if block.ndim == 1:  # target will be 1D, may not be contiguous
                    for row_pos in nonzero_1d(target):
                        assigned[row_pos] = values_map[(row_pos, t_start)]
                else:
                    for row_pos, col_pos in zip(*np.nonzero(target)):  # pyright: ignore
                        assigned[row_pos, col_pos] = values_map[
                            (row_pos, t_start + col_pos)
                        ]

                assigned.flags.writeable = False
                yield assigned

            t_start = t_end  # always update start

    # ---------------------------------------------------------------------------

    @staticmethod
    def _is_single_row(
        row_key: TILocSelector,
        row_key_null: bool,
        rows: int,
    ) -> bool:
        single_row = False
        if row_key_null and rows == 1:
            # this codition used to only hold if the arg is a null slice; now if None too and shape has one row
            single_row = True
        elif isinstance(row_key, INT_TYPES):
            single_row = True
        elif row_key.__class__ is np.ndarray:
            if row_key.dtype == DTYPE_BOOL:  # type: ignore
                if row_key.sum() == 1:  # type: ignore
                    single_row = True
            elif len(row_key) == 1:  # type: ignore
                single_row = True
        elif isinstance(row_key, KEY_ITERABLE_TYPES) and len(row_key) == 1:
            # an iterable of index integers is expected here
            single_row = True
        return single_row

    def _slice_blocks(
        self,
        row_key: TILocSelector,
        column_key: TILocSelector,
        row_key_is_slice: bool,
        row_key_null: bool,
    ) -> tp.Iterator[TNDArrayAny]:
        """
        Generator of sliced blocks, given row and column key selectors.
        The result is suitable for passing to TypeBlocks constructor.

        This is expected to always return immutable arrays.
        """
        # if column_key_null
        if column_key is None or (
            column_key.__class__ is slice and column_key == NULL_SLICE
        ):
            if self._index.columns == 0:
                yield EMPTY_ARRAY.reshape(self._index.shape)[row_key]
            elif row_key_null:  # when column_key is full
                yield from self._blocks
            elif row_key_is_slice:
                for b in self._blocks:
                    yield b[
                        row_key
                    ]  # from 2D will always return 2D, from 1D will always return 1D, which properly be interpreted as a column
            else:
                single_row = self._is_single_row(row_key, row_key_null, self._index.rows)
                for b in self._blocks:
                    # selection works for both 1D (to an element) and 2D (two a 1D array)
                    b_row = b[row_key]
                    if b_row.__class__ is np.ndarray:
                        # if row selection results in a 1D array, we need to make it into a 2D array (with one row), otherwise it will be interpreted as a column
                        if single_row and b_row.ndim == 1:
                            # reshaping preserves writeable status
                            b_row = b_row.reshape(1, b_row.shape[0])
                        b_row.flags.writeable = (
                            False  # no-slice selections will be writeable
                        )
                        yield b_row
                    else:  # wrap element back into an array
                        # If `row_key` selects a non-array, we have selected an element; if b is an object dtype, we might have selected a list or other iterable that, if naively given to an array constructor, gets "flattened" into arrays. Thus, we create an empty and assign
                        b_fill = np.empty(1, dtype=b.dtype)
                        b_fill[0] = b_row
                        b_fill.flags.writeable = False
                        yield b_fill
        elif row_key_is_slice:  # column_key is not None
            # as both row and col keys are slices, this will never reduce to an element
            for block_idx, slc in self._key_to_block_slices(column_key):
                b = self._blocks[block_idx]
                if b.ndim == 1:  # given 1D array, our row key is all we need
                    if row_key_null:
                        yield b
                    else:
                        yield b[row_key]
                else:  # given 2D, use row key and column slice
                    if row_key_null:
                        yield b[NULL_SLICE, slc]
                    else:
                        yield b[row_key, slc]
        else:
            # convert column_key into a series of block slices; we have to do this as we stride blocks; do not have to convert row_key as can use directly per block slice
            single_row = self._is_single_row(row_key, row_key_null, self._index.rows)
            for block_idx, slc in self._key_to_block_slices(column_key):
                b = self._blocks[block_idx]
                if b.ndim == 1:  # given 1D array, our row key is all we need
                    if row_key_null:
                        b_sliced = b
                    else:
                        b_sliced = b[row_key]
                else:  # given 2D, use row key and column slice
                    if row_key_null:
                        b_sliced = b[NULL_SLICE, slc]
                    else:
                        b_sliced = b[row_key, slc]
                # optionally, apply additional selection, reshaping, or adjustments to what we got out of the block
                if b_sliced.__class__ is np.ndarray:
                    # if a single row, sliced 1d, rotate to 2D
                    if single_row and slc.__class__ is slice and b_sliced.ndim == 1:
                        b_sliced = b_sliced.reshape(1, b_sliced.shape[0])
                    b_sliced.flags.writeable = False
                    yield b_sliced
                else:  # a single element, wrap back up in array; assignment handles special cases with lists in object dtypes correctly
                    b_fill = np.empty(1, dtype=b.dtype)
                    b_fill[0] = b_sliced
                    b_fill.flags.writeable = False
                    yield b_fill

    def _extract_array(
        self, row_key: TILocSelector = None, column_key: TILocSelector = None
    ) -> TNDArrayAny:
        """Alternative extractor that returns just an ndarray, concatenating blocks as necessary. Used by internal clients that need to process row/column with an array.

        This will be consistent with NumPy as to the dimensionality returned: if a non-multi selection is made, 1D array will be returned.
        """
        row_key_is_slice = row_key.__class__ is slice
        row_key_null = row_key is None or (row_key_is_slice and row_key == NULL_SLICE)
        # identifying column_key as integer, then we only access one block, and can return directly without iterating over blocks
        if column_key is not None and isinstance(column_key, INT_TYPES):
            block_idx, column = self._index[column_key]  # type: ignore
            b = self._blocks[block_idx]
            if b.ndim == 1:
                if row_key_null:
                    return b
                array = b[row_key]
                if array.__class__ is np.ndarray:
                    array.flags.writeable = False
                return array

            if row_key_null:
                return b[NULL_SLICE, column]

            array = b[row_key, column]
            if array.__class__ is np.ndarray:
                array.flags.writeable = False
            return array

        # figure out shape from keys so as to not accumulate?
        blocks = []
        columns = 0
        for b in self._slice_blocks(  # a generator
            row_key, column_key, row_key_is_slice, row_key_null
        ):
            if b.ndim == 1:  # it is a single column
                columns += 1
            else:
                columns += b.shape[1]
            blocks.append(b)

        # if row_key is None or a multiple type, we do not force_1d; so invert
        force_1d = not (row_key_null or isinstance(row_key, KEY_MULTIPLE_TYPES))

        if len(blocks) == 1:
            if force_1d:
                return row_1d_filter(blocks[0])
            return column_2d_filter(blocks[0])

        row_dtype = (
            self._index.dtype
            if column_key is None
            else resolve_dtype_iter(b.dtype for b in blocks)
        )

        rows = 0 if not blocks else blocks[0].shape[0]
        array = blocks_to_array_2d(
            blocks=blocks,
            shape=(rows, columns),
            dtype=row_dtype,
        )
        if force_1d:
            return row_1d_filter(array)
        return array

    def _extract_array_column(
        self,
        key: int | np.integer[tp.Any],
    ) -> TNDArrayAny:
        """Alternative extractor that returns full-column arrays from single integer selection."""
        try:
            block_idx, column = self._index[key]  # type: ignore
        except IndexError as e:
            raise KeyError(key) from e

        b = self._blocks[block_idx]
        if b.ndim == 1:
            return b
        return b[NULL_SLICE, column]

    def iter_row_elements(
        self,
        key: int | np.integer[tp.Any],
    ) -> tp.Iterator[tp.Any]:
        """Alternative extractor that yields a full-row of values from a single integer selection. This will avoid any type coercion."""
        for b in self._blocks:
            if b.ndim == 1:
                yield b[key]
            else:
                yield from b[key]

    def iter_row_tuples(
        self,
        key: TILocSelector,
        *,
        constructor: TTupleCtor = tuple,
    ) -> tp.Iterator[tp.Tuple[tp.Any, ...]]:
        """Alternative extractor that yields tuples per row of values based on a selection of one or more columns. This interface yields all rows in the TypeBlocks."""
        if key is None or (key.__class__ is slice and key == NULL_SLICE):
            arrays = self._blocks
        else:
            arrays = list(self._slice_blocks(None, key, False, True))

        if len(arrays) == 1:
            a = arrays[0]
            if a.ndim > 1:
                for i in range(a.shape[0]):
                    yield constructor(a[i])  # pyright: ignore # works for 1D, 2D
            else:
                for v in a:
                    yield constructor((v,))  # pyright: ignore
        else:

            def chainer(i: int) -> tp.Any:
                for a in arrays:
                    if a.ndim > 1:
                        yield from a[i]
                    else:
                        yield a[i]

            for i in range(self._index.rows):
                yield constructor(chainer(i))  # pyright: ignore

    def iter_row_lists(self) -> tp.Iterator[list[tp.Any]]:
        """Alternative extractor that yields tuples per row with all values converted to objects, not scalars."""
        arrays = self._blocks
        rows = self._index.rows
        cols = self._index.columns
        if len(arrays) == 1:
            a = arrays[0]
            if a.ndim > 1:
                for i in range(rows):
                    yield a[i].tolist()
            else:
                for v in a:
                    yield [v]
        else:
            for i in range(rows):
                row = [None] * cols
                pos = 0
                stop = 0
                # NOTE: these conversions might violate SF3 expectations for non-objectable types
                for a in arrays:
                    if a.ndim > 1:
                        stop = pos + a.shape[1]
                        row[pos:stop] = a[i].tolist()
                        pos = stop
                    else:
                        row[pos] = a[i].item()
                        pos += 1
                yield row

    def iter_columns_tuples(
        self,
        key: TILocSelector,
        *,
        constructor: tp.Optional[TTupleCtor] = tuple,
    ) -> tp.Iterator[tp.Tuple[tp.Any, ...]]:
        """Alternative extractor that yields tuples per column of values based on a selection of one or more rows. This interface yields all columns in the TypeBlocks."""
        key_is_slice = key.__class__ is slice
        key_null = key is None or (key_is_slice and key == NULL_SLICE)
        if key_null:
            arrays = self._blocks
        else:
            arrays = list(self._slice_blocks(key, None, key_is_slice, key_null))

        if len(arrays) == 1:
            array = arrays[0]
            for i in range(array.shape[1]):
                yield constructor(array[NULL_SLICE, i])  # pyright: ignore
        else:

            def chainer() -> tp.Iterator[TNDArrayAny]:
                for a in arrays:
                    if a.ndim == 1:
                        yield a
                    else:
                        for i in range(a.shape[1]):
                            yield a[NULL_SLICE, i]

            yield from map(constructor, chainer())  # type: ignore

    def iter_columns_arrays(self) -> tp.Iterator[TNDArrayAny]:
        """Iterator of column arrays."""
        for b in self._blocks:
            if b.ndim == 1:
                yield b
            else:
                for i in range(b.shape[1]):
                    yield b[NULL_SLICE, i]

    @tp.overload
    def _extract(
        self, row_key: TILocSelectorMany, column_key: TILocSelectorMany
    ) -> TypeBlocks: ...

    @tp.overload
    def _extract(
        self, row_key: TILocSelectorMany, column_key: TILocSelectorOne
    ) -> TypeBlocks: ...

    @tp.overload
    def _extract(
        self, row_key: TILocSelectorOne, column_key: TILocSelectorMany
    ) -> TypeBlocks: ...

    @tp.overload
    def _extract(self, row_key: TILocSelectorOne) -> TypeBlocks: ...

    @tp.overload
    def _extract(self, row_key: TILocSelectorMany) -> TypeBlocks: ...

    @tp.overload
    def _extract(self, column_key: TILocSelectorOne) -> TypeBlocks: ...

    @tp.overload
    def _extract(self, column_key: TILocSelectorMany) -> TypeBlocks: ...

    @tp.overload
    def _extract(
        self, row_key: TILocSelectorOne, column_key: TILocSelectorOne
    ) -> tp.Any: ...

    def _extract(  # pyright: ignore
        self,
        row_key: TILocSelector = None,
        column_key: TILocSelector = None,
    ) -> tp.Any:
        """
        Return a TypeBlocks after performing row and column selection using iloc selection.

        Row and column keys can be:
            integer: single row/column selection
            slices: one or more contiguous selections
            iterable of integers: one or more non-contiguous and/or repeated selections

        Returns:
            TypeBlocks, or a single element if both are coordinates
        """
        row_key_is_slice = row_key.__class__ is slice
        row_key_null = row_key is None or (row_key_is_slice and row_key == NULL_SLICE)

        # identifying column_key as integer, then we only access one block, and can return directly without iterating over blocks
        if column_key is not None and isinstance(column_key, INT_TYPES):
            block_idx, column = self._index[column_key]  # type: ignore
            b: TNDArrayAny = self._blocks[block_idx]
            if b.ndim == 1:
                if row_key_null:  # return a column
                    return TypeBlocks.from_blocks(b)
                elif isinstance(row_key, INT_TYPES):
                    return b[row_key]  # return single item
                return TypeBlocks.from_blocks(b[row_key])

            if row_key_null:
                return TypeBlocks.from_blocks(b[NULL_SLICE, column])
            elif isinstance(row_key, INT_TYPES):
                return b[row_key, column]
            return TypeBlocks.from_blocks(b[row_key, column])

        # pass a generator to from_block; will return a TypeBlocks or a single element
        return self.from_blocks(
            self._slice_blocks(row_key, column_key, row_key_is_slice, row_key_null),
            shape_reference=self._index.shape,
            own_data=True,
        )

    @tp.overload
    def _extract_iloc(self, key: TILocSelector) -> TypeBlocks: ...

    @tp.overload
    def _extract_iloc(self, key: TILocSelectorCompound) -> tp.Any: ...

    def _extract_iloc(self, key: TILocSelectorCompound) -> tp.Any:
        if isinstance(key, tuple):
            return self._extract(*key)  # type: ignore # NOTE: needs specialization for 2D input
        return self._extract(row_key=key)

    def extract_iloc_mask(self, key: TILocSelectorCompound) -> 'TypeBlocks':
        if isinstance(key, tuple):
            return TypeBlocks.from_blocks(self._mask_blocks(*key))
        return TypeBlocks.from_blocks(self._mask_blocks(row_key=key))

    def extract_bloc(
        self,
        bloc_key: TNDArrayAny,
    ) -> tp.Tuple[TNDArrayAny, TNDArrayAny]:
        """
        Extract a 1D array from TypeBlocks, doing minimal type coercion. This returns results in row-major ordering.
        """
        parts = []
        coords = []

        dt_resolve: tp.Optional[TDtypeAny] = None
        size: int = 0
        target_slice: tp.Union[int, slice]

        t_start = 0
        for block in self._blocks:
            if block.ndim == 1:
                t_end = t_start + 1
                target_slice = t_start
            else:
                t_end = t_start + block.shape[1]
                target_slice = slice(t_start, t_end)

            # target will only be 1D when block is 1d
            target = bloc_key[NULL_SLICE, target_slice]
            if not target.any():
                t_start = t_end
                continue

            # will always reduce to a 1D array
            part = block[target]
            if dt_resolve is None:
                dt_resolve = part.dtype
            else:
                dt_resolve = resolve_dtype(dt_resolve, part.dtype)
            size += len(part)
            parts.append(part)

            # get coordinates
            if block.ndim == 1:  # target will be 1D, may not be contioguous
                for row_pos in nonzero_1d(target):
                    coords.append((row_pos, t_start))
            else:
                for row_pos, col_pos in zip(*np.nonzero(target)):  # pyright: ignore
                    coords.append((row_pos, t_start + col_pos))
            t_start = t_end

        # if size is zero, dt_resolve will be None
        if size == 0:
            return EMPTY_ARRAY_OBJECT, EMPTY_ARRAY

        array = np.empty(shape=size, dtype=dt_resolve)
        np.concatenate(parts, out=array)

        # NOTE: because we iterate by block, the caller will be exposed to block-level organization, which might result in a different label ordering. we sort integer tuples of coords here, and use that sort order to sort array; this is better than trying to sort the labels on the Series (labels that might not be sortable).

        coords_array = np.empty(len(array), dtype=DTYPE_OBJECT)
        coords_array[:] = coords  # force creation of 1D object array

        # NOTE: in this sort there should never be ties, so we can use an unstable sort
        order = np.argsort(coords_array, kind=DEFAULT_FAST_SORT_KIND)
        post = array[order]
        post.flags.writeable = False

        # NOTE: we do not need to set coords selection to not writable as it is not used to build blocks
        return coords_array[order], post

    # ---------------------------------------------------------------------------
    # assignment interfaces

    def extract_iloc_assign_by_unit(
        self,
        key: tp.Tuple[TILocSelector, TILocSelector],
        value: tp.Any,
    ) -> 'TypeBlocks':
        """
        Assign with value via a unit: a single array or element.
        """
        row_key, column_key = key
        return TypeBlocks.from_blocks(
            self._assign_from_iloc_by_unit(
                row_key=row_key, column_key=column_key, value=value
            )
        )

    def extract_iloc_assign_by_sequence(
        self,
        key: tp.Tuple[TILocSelector, TILocSelector],
        value: tp.Any,
    ) -> 'TypeBlocks':
        """
        Assign with value via a unit: a single array or element.
        """
        row_key, column_key = key
        return TypeBlocks.from_blocks(
            self._assign_from_iloc_by_sequence(
                row_key=row_key, column_key=column_key, value=value
            )
        )

    def extract_iloc_assign_by_blocks(
        self,
        key: tp.Tuple[TILocSelector, TILocSelector],
        values: tp.Iterable[TNDArrayAny],
    ) -> 'TypeBlocks':
        """
        Assign with value via an iterable of blocks.
        """
        row_key, column_key = key
        return TypeBlocks.from_blocks(
            self._assign_from_iloc_by_blocks(
                row_key=row_key,
                column_key=column_key,
                values=values,
            )
        )

    def extract_bloc_assign_by_unit(
        self, key: TNDArrayAny, value: tp.Any
    ) -> 'TypeBlocks':
        return TypeBlocks.from_blocks(
            self._assign_from_bloc_by_unit(bloc_key=key, value=value)
        )

    def extract_bloc_assign_by_blocks(
        self, key: TNDArrayAny, values: tp.Any
    ) -> 'TypeBlocks':
        return TypeBlocks.from_blocks(
            self._assign_from_bloc_by_blocks(bloc_key=key, values=values)
        )

    def extract_bloc_assign_by_coordinate(
        self,
        key: TNDArrayAny,
        values_map: tp.Dict[tp.Tuple[int, int], tp.Any],
        values_dtype: TDtypeAny,
    ) -> 'TypeBlocks':
        return TypeBlocks.from_blocks(
            self._assign_from_bloc_by_coordinate(
                bloc_key=key,
                values_map=values_map,
                values_dtype=values_dtype,
            )
        )

    # ---------------------------------------------------------------------------
    def drop(self, key: TILocSelectorCompound) -> 'TypeBlocks':
        """
        Drop rows or columns from a TypeBlocks instance.

        Args:
            key: if a single value, treated as a row key; if a tuple, treated as a pair of row, column keys.
        """
        if isinstance(key, tuple):
            # column dropping can leed to a TB with generator that yields nothing;
            return TypeBlocks.from_blocks(
                self._drop_blocks(*key), shape_reference=self._index.shape
            )
        return TypeBlocks.from_blocks(
            self._drop_blocks(row_key=key), shape_reference=self._index.shape
        )

    def __iter__(self) -> tp.Iterator['TypeBlocks']:
        raise NotImplementedError(
            'Amibigous whether or not to return np array or TypeBlocks'
        )

    def __getitem__(self, key: TILocSelectorCompound) -> tp.Any:
        """
        Returns a column, or a column slice.
        """
        # NOTE: if key is a tuple it means that multiple indices are being provided
        if isinstance(key, tuple):
            raise KeyError('__getitem__ does not support multiple indexers')
        return self._extract(row_key=None, column_key=key)

    # ---------------------------------------------------------------------------
    # operators

    def _ufunc_unary_operator(
        self,
        operator: tp.Callable[[TNDArrayAny], TNDArrayAny],
    ) -> 'TypeBlocks':
        # for now, do no reblocking; though, in many cases, operating on a unified block will be faster
        def operation() -> tp.Iterator[TNDArrayAny]:
            for b in self._blocks:
                result = operator(b)
                result.flags.writeable = False
                yield result

        return self.from_blocks(operation())

    # ---------------------------------------------------------------------------

    def _block_shape_slices(self) -> tp.Iterator[slice]:
        """Generator of slices necessary to slice a 1d array of length equal to the number of columns into a length suitable for each block."""
        start = 0
        for b in self._blocks:
            end = start + (1 if b.ndim == 1 else b.shape[1])
            yield slice(start, end)
            start = end

    def _ufunc_binary_operator(
        self,
        *,
        operator: tp.Callable[[TNDArrayAny, TNDArrayAny], TNDArrayAny],
        other: tp.Iterable[tp.Any],
        axis: int = 0,
        fill_value: object = np.nan,  # for interface compat
    ) -> 'TypeBlocks':
        """Axis is only relevant in the application of a 1D array to a 2D TypeBlocks, where axis 0 (the default) will apply the array per row, while axis 1 will apply the array per column."""
        self_operands: tp.Iterable[TNDArrayAny]
        other_operands: tp.Iterable[TNDArrayAny]

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
            elif self.shape == other.shape:
                # if the result of reblock does not result in compatible shapes, we have to use .values as operands; the dtypes can be different so we only have to check that they columns sizes, the second element of the signature, all match.
                if not self.reblock_compatible(other):
                    self_operands = (self.values,)
                    other_operands = (other.values,)
                else:
                    self_operands = self._reblock()
                    other_operands = other._reblock()
            else:  # raise same error as NP
                raise NotImplementedError(
                    'cannot apply binary operators to arbitrary TypeBlocks'
                )
        else:  # process other as an array
            self_operands = self._blocks
            if other.__class__ is not np.ndarray:
                other = iterable_to_array_nd(other)

            # handle dimensions
            if other.ndim == 0 or (other.ndim == 1 and len(other) == 1):  # type: ignore
                # a scalar: reference same value for each block position
                apply_column_2d_filter = False
                other_operands = (other for _ in range(len(self._blocks)))  # pyright: ignore
            elif other.ndim == 1:  # type: ignore
                if axis == 0 and len(other) == self._index.columns:  # type: ignore
                    # 1d array applied to the rows: chop to block width
                    apply_column_2d_filter = False
                    other_operands = (other[s] for s in self._block_shape_slices())  # type: ignore
                elif axis == 1 and len(other) == self._index.rows:  # type: ignore
                    columnar = True
                else:
                    raise NotImplementedError(
                        f'cannot apply binary operators with a 1D array along axis {axis}: {self.shape}, {other.shape}.'  # type: ignore
                    )
            elif other.ndim == 2 and other.shape == self._index.shape:  # type: ignore
                apply_column_2d_filter = True
                other_operands = (
                    other[NULL_SLICE, s]  # type: ignore
                    for s in self._block_shape_slices()
                )
            else:
                raise NotImplementedError(
                    f'cannot apply binary operators to arrays without alignable shapes: {self._index.shape}, {other.shape}.'  # type: ignore
                )

        if columnar:
            return self.from_blocks(
                apply_binary_operator_blocks_columnar(
                    values=self_operands,
                    other=other,  # type: ignore
                    operator=operator,
                )
            )

        return self.from_blocks(
            apply_binary_operator_blocks(
                values=self_operands,
                other=other_operands,
                operator=operator,
                apply_column_2d_filter=apply_column_2d_filter,
            )
        )

    # ---------------------------------------------------------------------------
    # transformations resulting in the same dimensionality

    def isin(self, other: tp.Any) -> 'TypeBlocks':
        """Return a new Boolean TypeBlocks that returns True if an element is in `other`."""
        if hasattr(other, '__len__') and len(other) == 0:
            array = np.full(self._index.shape, False, dtype=bool)
            array.flags.writeable = False
            return self.from_blocks(array)

        other, other_is_unique = iterable_to_array_1d(other)

        def blocks() -> tp.Iterator[TNDArrayAny]:
            for b in self._blocks:
                # yields immutable arrays
                yield isin_array(
                    array=b,
                    array_is_unique=False,  # expensive to determine
                    other=other,
                    other_is_unique=other_is_unique,
                )

        return self.from_blocks(blocks())

    def transpose(self) -> 'TypeBlocks':
        """Return a new TypeBlocks that transposes and concatenates all blocks."""
        if self.unified:
            # NOTE: transpositions of unified arrays are immutable
            array = column_2d_filter(self._blocks[0]).transpose()
        else:
            dtype = self._index.dtype
            blocks = []
            for b in self._blocks:
                b = column_2d_filter(b).transpose()
                if b.dtype != dtype:
                    b = b.astype(dtype)
                blocks.append(b)

            array = np.empty((self._index.columns, self._index.rows), dtype=dtype)
            np.concatenate(blocks, axis=0, out=array)
            array.flags.writeable = False
        return self.from_blocks(array)

    # ---------------------------------------------------------------------------
    #
    def boolean_apply_any(self, func: tp.Callable[[TNDArrayAny], TNDArrayAny]) -> bool:
        """Apply a Boolean-returning function to TypeBlocks and return a Boolean if any values are True. This takes advantage of short-circuiting and avoiding intermediary containers for better performance."""
        for b in self._blocks:
            if func(b).any():
                return True
        return False

    # ---------------------------------------------------------------------------
    # na handling

    def isna(self, include_none: bool = True) -> 'TypeBlocks':
        """Return a Boolean TypeBlocks where True is NaN or None."""

        def blocks() -> tp.Iterator[TNDArrayAny]:
            for b in self._blocks:
                bool_block = isna_array(b, include_none)
                bool_block.flags.writeable = False
                yield bool_block

        return self.from_blocks(blocks())

    def notna(self, include_none: bool = True) -> 'TypeBlocks':
        """Return a Boolean TypeBlocks where True is not NaN or None."""

        def blocks() -> tp.Iterator[TNDArrayAny]:
            for b in self._blocks:
                bool_block = np.logical_not(isna_array(b, include_none))
                bool_block.flags.writeable = False
                yield bool_block

        return self.from_blocks(blocks())

    # ---------------------------------------------------------------------------
    # falsy handling

    def isfalsy(self) -> 'TypeBlocks':
        """Return a Boolean TypeBlocks where True is falsy."""

        def blocks() -> tp.Iterator[TNDArrayAny]:
            for b in self._blocks:
                bool_block = isfalsy_array(b)
                bool_block.flags.writeable = False
                yield bool_block

        return self.from_blocks(blocks())

    def notfalsy(self) -> 'TypeBlocks':
        """Return a Boolean TypeBlocks where True is not falsy."""

        def blocks() -> tp.Iterator[TNDArrayAny]:
            for b in self._blocks:
                bool_block = np.logical_not(isfalsy_array(b))
                bool_block.flags.writeable = False
                yield bool_block

        return self.from_blocks(blocks())

    # ---------------------------------------------------------------------------
    def clip(
        self,
        lower: tp.Union[None, float, tp.Iterable[TNDArrayAny]],
        upper: tp.Union[None, float, tp.Iterable[TNDArrayAny]],
    ) -> 'TypeBlocks':
        """
        Apply clip to blocks. If clipping is not supported for a dtype, we will raise instead of silently returning the block unchanged.

        Args:
            lower, upper: a float, or iterable of array of aggregate shape equal to that of this TypeBlocks
        """
        lower_is_element = not hasattr(lower, '__len__')
        upper_is_element = not hasattr(upper, '__len__')

        lower_is_array = lower.__class__ is np.ndarray
        upper_is_array = upper.__class__ is np.ndarray

        # get a mutable list in reverse order for pop/pushing
        lower_source: None | float | TNDArrayAny | tp.List[TNDArrayAny]
        if lower_is_element or lower_is_array:
            lower_source = lower  # type: ignore
        else:
            lower_source = list(lower)  # type: ignore
            lower_source.reverse()

        upper_source: None | float | TNDArrayAny | tp.List[TNDArrayAny]
        if upper_is_element or upper_is_array:
            upper_source = upper  # type: ignore
        else:
            upper_source = list(upper)  # type: ignore
            upper_source.reverse()

        def get_block_match(
            start: int,  # relative to total size
            end: int,  # exclusive
            ndim: int,
            source: tp.Union[None, float, TNDArrayAny, tp.List[TNDArrayAny]],
            is_element: bool,
            is_array: bool,
        ) -> tp.Union[None, float, TNDArrayAny]:
            """
            Handle extraction of clip boundaries from multiple different types of sources. NOTE: ndim is the target ndim, and is only relevant when width is 1
            """
            if is_element:
                return source  # type: ignore

            width_target = end - start  # 1 is lowest value

            if is_array:  # if we have a homogenous 2D array
                block = source[NULL_SLICE, start:end]  # type: ignore
                func = column_1d_filter if ndim == 1 else column_2d_filter
                return func(block)

            assert isinstance(source, list)
            block = source.pop()
            width = shape_filter(block)[1]  # width is columns in next source (bounds)

            if width_target == 1:
                if width > 1:  # 2d array with more than one column
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
            raise RuntimeError('Unexepcted exit')  # pragma: no cover

        def blocks() -> tp.Iterator[TNDArrayAny]:
            start = end = 0
            lb: tp.Union[None, float, TNDArrayAny]
            ub: tp.Union[None, float, TNDArrayAny]
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

    # ---------------------------------------------------------------------------
    # fillna sided

    @staticmethod
    def _fill_missing_sided_axis_0(
        blocks: tp.Iterable[TNDArrayAny],
        value: tp.Any,
        func_target: TUFunc,
        sided_leading: bool,
    ) -> tp.Iterator[TNDArrayAny]:
        """Return a TypeBlocks where NaN or None are replaced in sided (leading or trailing) segments along axis 0, meaning vertically.

        Args:
            sided_leading: True sets the side to fill is the leading side; False sets the side to fill to the trailiing side.

        """
        if value.__class__ is np.ndarray:
            raise RuntimeError('cannot assign an array to fillna')

        sided_index = 0 if sided_leading else -1

        # store flag for when non longer need to check blocks, yield immediately

        for b in blocks:
            sel = func_target(b)  # True for is NaN
            ndim = sel.ndim

            if ndim == 1 and not sel[sided_index]:
                # if last value (bottom row) is not NaN, we can return block
                yield b
            elif ndim > 1 and ~sel[sided_index].any():  # if not any are NaN
                # can use this last-row observation below
                yield b
            else:
                assignable_dtype = resolve_dtype(dtype_from_element(value), b.dtype)
                if b.dtype == assignable_dtype:
                    assigned = b.copy()
                else:
                    assigned = b.astype(assignable_dtype)

                # make 2d look like 1D here
                if ndim == 1:
                    sel_nonzeros = ((0, sel),)
                else:
                    # only collect columns for sided NaNs
                    sel_nonzeros = (
                        (i, sel[:, i]) for i, j in enumerate(sel[sided_index]) if j
                    )  # type: ignore

                for idx, sel_nonzero in sel_nonzeros:
                    ft = first_true_1d(~sel_nonzero, forward=sided_leading)
                    if ft != -1:
                        if sided_leading:
                            sel_slice = slice(0, ft)
                        else:  # trailing
                            sel_slice = slice(ft + 1, None)
                    else:
                        sel_slice = NULL_SLICE

                    if ndim == 1:
                        assigned[sel_slice] = value
                    else:
                        assigned[sel_slice, idx] = value

                # done writing
                assigned.flags.writeable = False
                yield assigned

    @staticmethod
    def _fill_missing_sided_axis_1(
        blocks: tp.Iterable[TNDArrayAny],
        value: tp.Any,
        func_target: TUFunc,
        sided_leading: bool,
    ) -> tp.Iterator[TNDArrayAny]:
        """Return a TypeBlocks where NaN or None are replaced in sided (leading or trailing) segments along axis 1. Leading axis 1 fills rows, going from left to right.

        NOTE: blocks are generated in reverse order when sided_leading is False.

        Args:
            sided_leading: True sets the side to fill is the leading side; False sets the side to fill to the trailing side.

        """
        if value.__class__ is np.ndarray:
            raise RuntimeError('cannot assign an array to fillna')

        sided_index = 0 if sided_leading else -1

        # will need to re-reverse blocks coming out of this
        block_iter = blocks if sided_leading else reversed(blocks)  # type: ignore

        isna_exit_previous = None

        # iterate over blocks to observe NaNs contiguous horizontally
        for b in block_iter:
            sel = func_target(b)  # True for is NaN
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
                assignable_dtype = resolve_dtype(dtype_from_element(value), b.dtype)
                if b.dtype == assignable_dtype:
                    assigned = b.copy()
                else:
                    assigned = b.astype(assignable_dtype)

                if ndim == 1:
                    # if one dim, we simply fill nan values
                    assigned[isna_entry] = value
                else:
                    # only collect rows that have a sided NaN
                    candidates = (i for i, j in enumerate(isna_entry) if j)
                    sels_nonzero = ((i, sel[i]) for i in candidates)

                    for idx, sel_nonzero in sels_nonzero:
                        ft = first_true_1d(~sel_nonzero, forward=sided_leading)
                        if ft != -1:
                            if sided_leading:
                                sel_slice = slice(0, ft)
                            else:  # trailing
                                sel_slice = slice(ft + 1, None)
                        else:
                            sel_slice = NULL_SLICE
                        assigned[idx, sel_slice] = value

                assigned.flags.writeable = False
                yield assigned

            # always execute these lines after each yield
            # return True for next block only if all values are NaN in the row
            if ndim == 1:
                isna_exit_previous = isna_entry
            else:
                isna_exit_previous = sel.all(axis=1) & isna_exit_previous

    def fillna_leading(self, value: tp.Any, *, axis: int = 0) -> 'TypeBlocks':
        """Return a TypeBlocks instance replacing leading values with the passed `value`. Leading, axis 0 fills columns, going from top to bottom. Leading axis 1 fills rows, going from left to right."""
        if axis == 0:
            return self.from_blocks(
                self._fill_missing_sided_axis_0(
                    blocks=self._blocks,
                    value=value,
                    func_target=isna_array,
                    sided_leading=True,
                )
            )
        elif axis == 1:
            return self.from_blocks(
                self._fill_missing_sided_axis_1(
                    blocks=self._blocks,
                    value=value,
                    func_target=isna_array,
                    sided_leading=True,
                )
            )
        raise AxisInvalid(f'no support for axis {axis}')

    def fillna_trailing(self, value: tp.Any, *, axis: int = 0) -> 'TypeBlocks':
        """Return a TypeBlocks instance replacing trailing NaNs with the passed `value`. Trailing, axis 0 fills columns, going from bottom to top. Trailing axis 1 fills rows, going from right to left."""
        if axis == 0:
            return self.from_blocks(
                self._fill_missing_sided_axis_0(
                    blocks=self._blocks,
                    value=value,
                    func_target=isna_array,
                    sided_leading=False,
                )
            )
        elif axis == 1:
            # must reverse when not leading
            blocks = reversed(
                tuple(
                    self._fill_missing_sided_axis_1(
                        blocks=self._blocks,
                        value=value,
                        func_target=isna_array,
                        sided_leading=False,
                    )
                )
            )
            return self.from_blocks(blocks)

        raise AxisInvalid(f'no support for axis {axis}')

    def fillfalsy_leading(self, value: tp.Any, /, *, axis: int = 0) -> 'TypeBlocks':
        """Return a TypeBlocks instance replacing leading values with the passed `value`. Leading, axis 0 fills columns, going from top to bottom. Leading axis 1 fills rows, going from left to right."""
        if axis == 0:
            return self.from_blocks(
                self._fill_missing_sided_axis_0(
                    blocks=self._blocks,
                    value=value,
                    func_target=isfalsy_array,
                    sided_leading=True,
                )
            )
        elif axis == 1:
            return self.from_blocks(
                self._fill_missing_sided_axis_1(
                    blocks=self._blocks,
                    value=value,
                    func_target=isfalsy_array,
                    sided_leading=True,
                )
            )

        raise AxisInvalid(f'no support for axis {axis}')

    def fillfalsy_trailing(self, value: tp.Any, /, *, axis: int = 0) -> 'TypeBlocks':
        """Return a TypeBlocks instance replacing trailing NaNs with the passed `value`. Trailing, axis 0 fills columns, going from bottom to top. Trailing axis 1 fills rows, going from right to left."""
        if axis == 0:
            return self.from_blocks(
                self._fill_missing_sided_axis_0(
                    blocks=self._blocks,
                    value=value,
                    func_target=isfalsy_array,
                    sided_leading=False,
                )
            )
        elif axis == 1:
            # must reverse when not leading
            blocks = reversed(
                tuple(
                    self._fill_missing_sided_axis_1(
                        blocks=self._blocks,
                        value=value,
                        func_target=isfalsy_array,
                        sided_leading=False,
                    )
                )
            )
            return self.from_blocks(blocks)

        raise AxisInvalid(f'no support for axis {axis}')

    # ---------------------------------------------------------------------------
    # fillna directional

    @staticmethod
    def _fill_missing_directional_axis_0(
        blocks: tp.Iterable[TNDArrayAny],
        directional_forward: bool,
        func_target: TUFunc,
        limit: int = 0,
    ) -> tp.Iterator[TNDArrayAny]:
        """
        Do a directional fill along axis 0, meaning filling vertically, going top/bottom or bottom/top.

        Args:
            directional_forward: if True, start from the forward (top or left) side.
        """

        for b in blocks:
            sel = func_target(b)  # True for is NaN
            ndim = sel.ndim
            if not np.any(sel):
                yield b
            else:
                target_indexes = binary_transition(sel)

                if ndim == 1:
                    # make single array look like iterable of tuples
                    slots = 1
                    length = len(sel)

                elif ndim == 2:
                    slots = b.shape[1]  # axis 0 has column width
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
                            return sel[target_slice.start]  # type: ignore

                    else:  # 2D blocks
                        target_index = target_indexes[i]
                        if not target_index:
                            continue
                        target_values = b[target_index, i]

                        def slice_condition(target_slice: slice) -> bool:
                            # NOTE: start is never None
                            return sel[target_slice.start, i]  # type: ignore

                    for target_slice, value in slices_from_targets(
                        target_index=target_index,
                        target_values=target_values,
                        length=length,
                        directional_forward=directional_forward,
                        limit=limit,
                        slice_condition=slice_condition,
                    ):
                        if ndim == 1:
                            assigned[target_slice] = value
                        else:
                            assigned[target_slice, i] = value

                assigned.flags.writeable = False
                yield assigned

    @staticmethod
    def _fill_missing_directional_axis_1(
        blocks: tp.Iterable[TNDArrayAny],
        directional_forward: bool,
        func_target: TUFunc,
        limit: int = 0,
    ) -> tp.Iterator[TNDArrayAny]:
        """
        Do a directional fill along axis 1, or horizontally, going left to right or right to left.

        NOTE: blocks are generated in reverse order when directional_forward is False.

        """
        bridge_src_index = -1 if directional_forward else 0
        bridge_dst_index = 0 if directional_forward else -1

        # will need to re-reverse blocks coming out of this
        block_iter = blocks if directional_forward else reversed(blocks)  # type: ignore

        bridging_values: tp.Optional[TNDArrayAny] = None
        bridging_count: tp.Optional[TNDArrayAny] = None
        bridging_isna: tp.Optional[TNDArrayAny] = (
            None  # Boolean array describing isna of bridging values
        )

        for b in block_iter:
            sel = func_target(b)  # True for is NaN
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
            else:  # some NA in this block
                if bridging_values is None:
                    assigned = b.copy()
                    bridging_count = np.full(b.shape[0], 0)
                else:
                    assignable_dtype = resolve_dtype(bridging_values.dtype, b.dtype)
                    assigned = b.astype(assignable_dtype)

                if ndim == 1:
                    # a single array has either NaN or non-NaN values; will only fill in NaN if we have a caried value from the previous block
                    if bridging_values is not None:  # sel has at least one NaN
                        bridging_isnotna = ~bridging_isna  # type: ignore [operator]

                        sel_sided = sel & bridging_isnotna
                        if limit:
                            # set to false those values where bridging already at limit
                            sel_sided[bridging_count >= limit] = False  # type: ignore

                        # set values in assigned if there is a NaN here (sel_sided) and we are not beyond the count
                        assigned[sel_sided] = bridging_values[sel_sided]
                        # only increment positions that are NaN here and have not-nan bridging values
                        sel_count_increment = sel & bridging_isnotna
                        bridging_count[sel_count_increment] += 1  # type: ignore
                        # set unassigned to zero
                        bridging_count[~sel_count_increment] = 0  # type: ignore
                    else:
                        bridging_count = np.full(b.shape[0], 0)

                    bridging_values = assigned
                    bridging_isna = isna_array(bridging_values)

                elif ndim == 2:
                    slots = b.shape[0]  # axis 0 has column width
                    length = b.shape[1]

                    # set to True when can reset count to zero; this is always the case if the bridge src value is not NaN (before we do any filling)
                    bridging_count_reset = ~sel[:, bridge_src_index]

                    if bridging_values is not None:
                        bridging_isnotna = ~bridging_isna  # type: ignore

                        # find leading NaNs segments if they exist, and if there is as corrresponding non-nan value to bridge
                        isna_entry = sel[:, bridge_dst_index] & bridging_isnotna
                        # get a row of Booleans for plausible candidates
                        candidates = (i for i, j in enumerate(isna_entry) if j)
                        sels_nonzero = ((i, sel[i]) for i in candidates)

                        # get appropriate leading slice to cover nan region
                        for idx, sel_nonzero in sels_nonzero:
                            # indices of not-nan values, per row
                            ft = first_true_1d(~sel_nonzero, forward=directional_forward)
                            if ft != -1:
                                if directional_forward:
                                    sel_slice = slice(0, ft)
                                else:  # trailing
                                    sel_slice = slice(ft + 1, None)
                            else:
                                sel_slice = slice(0, length)

                            # truncate sel_slice by limit-
                            sided_len = len(range(*sel_slice.indices(length)))

                            if limit and bridging_count[idx] >= limit:  # type: ignore
                                # if already at limit, do not assign
                                bridging_count[idx] += sided_len  # type: ignore
                                continue
                            elif limit and (bridging_count[idx] + sided_len) >= limit:  # type: ignore
                                # trim slice to fit
                                shift = bridging_count[idx] + sided_len - limit  # type: ignore
                                # shift should only be positive only here
                                if directional_forward:
                                    sel_slice = slice(
                                        sel_slice.start, sel_slice.stop - shift
                                    )
                                else:
                                    sel_slice = slice(
                                        sel_slice.start + shift, sel_slice.stop
                                    )

                            # update with full length or limited length?
                            bridging_count[idx] += sided_len  # type: ignore
                            assigned[idx, sel_slice] = bridging_values[idx]

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
                            return sel[i, target_slice.start]  # type: ignore

                        target_slice = None
                        for target_slice, value in slices_from_targets(
                            target_index=target_index,
                            target_values=target_values,
                            length=length,
                            directional_forward=directional_forward,
                            limit=limit,
                            slice_condition=slice_condition,
                        ):
                            assigned[i, target_slice] = value

                        # update counts from the last slice; this will have already been limited if necessary, but need to reflext contiguous values going into the next block; if slices does not go to edge; will identify as needing as reset
                        if target_slice is not None:
                            bridging_count[i] = len(range(*target_slice.indices(length)))  # type: ignore

                    bridging_values = assigned[:, bridge_src_index]
                    bridging_isna = isna_array(bridging_values)

                    # if the birdging values is NaN now, it could not be filled, or was not filled enough, and thus does not continue a count; can set to zero
                    bridging_count_reset |= bridging_isna
                    bridging_count[bridging_count_reset] = 0  # type: ignore

                assigned.flags.writeable = False
                yield assigned

    def fillna_forward(self, limit: int = 0, *, axis: int = 0) -> 'TypeBlocks':
        """Return a new ``TypeBlocks`` after feeding forward the last non-null (NaN or None) observation across contiguous nulls. Forward axis 0 fills columns, going from top to bottom. Forward axis 1 fills rows, going from left to right."""
        if axis == 0:
            return self.from_blocks(
                self._fill_missing_directional_axis_0(
                    blocks=self._blocks,
                    directional_forward=True,
                    func_target=isna_array,
                    limit=limit,
                )
            )
        elif axis == 1:
            return self.from_blocks(
                self._fill_missing_directional_axis_1(
                    blocks=self._blocks,
                    directional_forward=True,
                    func_target=isna_array,
                    limit=limit,
                )
            )

        raise AxisInvalid(f'no support for axis {axis}')

    def fillna_backward(self, limit: int = 0, *, axis: int = 0) -> 'TypeBlocks':
        """Return a new ``TypeBlocks`` after feeding backward the last non-null (NaN or None) observation across contiguous nulls. Backward, axis 0 fills columns, going from bottom to top. Backward axis 1 fills rows, going from right to left."""
        if axis == 0:
            return self.from_blocks(
                self._fill_missing_directional_axis_0(
                    blocks=self._blocks,
                    directional_forward=False,
                    func_target=isna_array,
                    limit=limit,
                )
            )
        elif axis == 1:
            blocks = reversed(
                tuple(
                    self._fill_missing_directional_axis_1(
                        blocks=self._blocks,
                        directional_forward=False,
                        func_target=isna_array,
                        limit=limit,
                    )
                )
            )
            return self.from_blocks(blocks)

        raise AxisInvalid(f'no support for axis {axis}')

    def fillfalsy_forward(self, limit: int = 0, *, axis: int = 0) -> 'TypeBlocks':
        """Return a new ``TypeBlocks`` after feeding forward the last non-falsy observation across contiguous missing values. Forward axis 0 fills columns, going from top to bottom. Forward axis 1 fills rows, going from left to right."""
        if axis == 0:
            return self.from_blocks(
                self._fill_missing_directional_axis_0(
                    blocks=self._blocks,
                    directional_forward=True,
                    func_target=isfalsy_array,
                    limit=limit,
                )
            )
        elif axis == 1:
            return self.from_blocks(
                self._fill_missing_directional_axis_1(
                    blocks=self._blocks,
                    directional_forward=True,
                    func_target=isfalsy_array,
                    limit=limit,
                )
            )

        raise AxisInvalid(f'no support for axis {axis}')

    def fillfalsy_backward(self, limit: int = 0, *, axis: int = 0) -> 'TypeBlocks':
        """Return a new ``TypeBlocks`` after feeding backward the last non-falsy observation across contiguous missing values. Backward, axis 0 fills columns, going from bottom to top. Backward axis 1 fills rows, going from right to left."""
        if axis == 0:
            return self.from_blocks(
                self._fill_missing_directional_axis_0(
                    blocks=self._blocks,
                    directional_forward=False,
                    func_target=isfalsy_array,
                    limit=limit,
                )
            )
        elif axis == 1:
            blocks = reversed(
                tuple(
                    self._fill_missing_directional_axis_1(
                        blocks=self._blocks,
                        directional_forward=False,
                        func_target=isfalsy_array,
                        limit=limit,
                    )
                )
            )
            return self.from_blocks(blocks)

        raise AxisInvalid(f'no support for axis {axis}')

    # ---------------------------------------------------------------------------

    def drop_missing_to_keep_locations(
        self,
        axis: int = 0,
        condition: tp.Callable[..., TNDArrayAny] = np.all,
        *,
        func: tp.Callable[[TNDArrayAny], TNDArrayAny],
    ) -> tp.Tuple[tp.Optional[TNDArrayAny], tp.Optional[TNDArrayAny]]:
        """
        Return the row and column slices to extract the new TypeBlock. This is to be used by Frame, where the slices will be needed on the indices as well.

        Args:
            axis: Dimension to drop, where 0 will drop rows and 1 will drop columns based on the condition function applied to a Boolean array.
            func: A function that takes an array and returns a Boolean array.
        """
        # get a unified boolean array; as isna will always return a Boolean, we can simply take the first block out of consolidation
        unified = next(self.consolidate_blocks(func(b) for b in self._blocks))

        # flip axis to condition funcion
        to_drop: TNDArrayAny
        if unified.ndim == 2:
            condition_axis = 0 if axis else 1
            to_drop = condition(unified, axis=condition_axis)
        else:  # ndim == 1
            to_drop = unified
        to_keep = np.logical_not(to_drop)

        if axis == 1:
            row_key = None
            column_key = to_keep
        else:
            row_key = to_keep
            column_key = None

        return row_key, column_key

    def fill_missing_by_unit(
        self,
        value: object,
        value_valid: tp.Optional[TNDArrayAny] = None,
        *,
        func: tp.Callable[[TNDArrayAny], TNDArrayAny],
    ) -> 'TypeBlocks':
        """
        Return a new TypeBlocks instance that fills missing values with the passed value.

        Args:
            value: value to fill missing with; can be an element or a same-sized array.
            value_valid: Optionally provide a same-size array mask of the value setting (useful for carrying forward information from labels).
            func: A function that takes an array and returns a Boolean array.
        """
        return self.from_blocks(
            self._assign_from_boolean_blocks_by_unit(
                targets=(func(b) for b in self._blocks),
                value=value,
                value_valid=value_valid,
            )
        )

    def fill_missing_by_callable(
        self,
        *,
        func_missing: tp.Callable[[TNDArrayAny], TNDArrayAny],
        get_col_fill_value: tp.Callable[[int, TDtypeAny | None], tp.Any],
    ) -> 'TypeBlocks':
        """
        Return a new TypeBlocks instance that fills missing values with the passed value.

        Args:
            func_missing: A function that takes an array and returns a Boolean array.
        """
        return self.from_blocks(
            self._assign_from_boolean_blocks_by_callable(
                targets=(func_missing(b) for b in self._blocks),
                get_col_fill_value=get_col_fill_value,
            )
        )

    def fill_missing_by_values(
        self,
        values: tp.Sequence[TNDArrayAny],
        *,
        func: tp.Callable[[TNDArrayAny], TNDArrayAny],
    ) -> 'TypeBlocks':
        """
        Return a new TypeBlocks instance that fills missing values with the aligned columnar arrays.

        Args:
            values: iterable of arrays to be aligned as columns.
        """
        return self.from_blocks(
            self._assign_from_boolean_blocks_by_blocks(
                targets=(func(b) for b in self._blocks),
                values=values,
            )
        )

    def iter_block_signatures(self) -> tp.Iterator[TArraySignature]:
        """
        Yields:
            a hashable key that will match array that share the same data, or share slices from the same underlying data and have the same shape and strides.
        """
        yield from (
            array_signature(self._extract_array_column(i))
            for i in range(self._index.columns)
        )

    @doc_inject()
    def equals(
        self,
        other: tp.Any,
        /,
        *,
        compare_name: bool = False,
        compare_dtype: bool = False,
        compare_class: bool = False,
        skipna: bool = True,
    ) -> bool:
        """
        {doc} Underlying block structure is not considered in determining equality.

        Args:
            {compare_dtype}
            {compare_class}
            {skipna}
        """
        # NOTE: we include `name` such that all instances have the same interface
        if id(other) == id(self):
            return True

        # NOTE: there is only one TypeBlocks class, but better to be consistent
        if compare_class and self.__class__ != other.__class__:
            return False
        elif not isinstance(other, TypeBlocks):
            return False

        # same type from here
        if self._index.shape != other.shape:
            return False

        if compare_dtype:
            for d_self, d_other in zip(self._iter_dtypes(), other._iter_dtypes()):
                # have already validated shape
                if d_self != d_other:
                    return False

        # NOTE: cannot directly compare blocks as we cannot assume the same number of blocks means that the blocks are consolidated in the same way

        for i in range(self._index.columns):
            if not arrays_equal(
                self._extract_array_column(i),
                other._extract_array_column(i),
                skipna=skipna,
            ):
                return False
        return True

    # ---------------------------------------------------------------------------
    # mutate

    def append(self, block: TNDArrayAny) -> None:
        """Add a block; an array copy will not be made unless the passed in block is not immutable"""
        # NOTE: shape can be 0, 0 if empty, or any one dimension can be 0. if columns is 0 and rows is non-zero, that row count is binding for appending (though the array need no tbe appended); if columns is > 0 and rows is zero, that row is binding for appending (and the array should be appended).
        if self._index.register(block):
            self._blocks.append(immutable_filter(block))

    def extend(self, other: tp.Union['TypeBlocks', tp.Iterable[TNDArrayAny]]) -> None:
        """Extend this TypeBlock with the contents of another TypeBlocks instance, or an iterable of arrays. Note that an iterable of TypeBlocks is not currently supported."""
        # accept iterables of np.arrays
        blocks = other._blocks if isinstance(other, TypeBlocks) else other
        for block in blocks:
            if self._index.register(block):
                self._blocks.append(immutable_filter(block))
