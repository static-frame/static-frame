from __future__ import annotations

from itertools import repeat

import numpy as np
import typing_extensions as tp
from arraykit import factorize, group_reduce, resolve_dtype, resolve_dtype_iter

from static_frame.core.frame import Frame
from static_frame.core.index_auto import IndexAutoFactory, TIndexAutoFactory
from static_frame.core.index_base import IndexBase
from static_frame.core.node_selector import Interface, InterfaceBatch
from static_frame.core.series import Series
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.util import (
    DTYPE_FLOAT_DEFAULT,
    DTYPE_INT_DEFAULT,
    DTYPE_OBJECT,
    EMPTY_ARRAY,
    FRAME_INITIALIZER_DEFAULT,
    NULL_SLICE,
    IterNodeType,
    TCallableAny,
    TILocSelectorOne,
    TIndexCtorSpecifier,
    TIndexInitializer,
    TLabel,
    TName,
    TUFunc,
    concat_resolved,
    iterable_to_array_1d,
    ufunc_dtype_to_dtype,
)

if tp.TYPE_CHECKING:
    from static_frame.core.batch import Batch
    from static_frame.core.generic_aliases import TFrameAny

TNDArrayAny = np.ndarray[tp.Any, tp.Any]
TDtypeAny = np.dtype[tp.Any]

TFrameOrSeries = tp.Union[Frame, Series]
TFrameOrArray = tp.Union[Frame, TNDArrayAny]
TIterableFrameItems = tp.Iterable[tp.Tuple[TLabel, TFrameOrArray]]
TShape2D = tp.Tuple[int, int]

TListILocToFunc = tp.List[tp.Tuple[TILocSelectorOne, TUFunc]]
TListLabelToFunc = tp.List[tp.Tuple[TLabel, TUFunc]]

# Reduction functions whose per-group result an `arraykit.group_reduce` op reproduces
# exactly. Only functions proven equivalent are listed; anything else (nan-variants,
# mean/std, lambdas, ...) is absent and routes through the general per-group loop. Keyed
# by function identity, so the vectorized fast path is opt-in and never changes results.
_REDUCE_FUNC_TO_OP: tp.Dict[tp.Any, str] = {
    np.sum: 'sum',
    np.prod: 'prod',
    np.min: 'min',
    np.max: 'max',
    len: 'count',
    sum: 'sum',
    min: 'min',
    max: 'max',
}


def _reduce_column_plan(
    func: TUFunc,
    op: str,
    values: TNDArrayAny,
    unified: TDtypeAny | None,
) -> tp.Optional[tp.Tuple[TNDArrayAny, TDtypeAny]]:
    """
    Determine what to feed ``group_reduce`` for one column and the resulting dtype, or
    None to fall back to the loop.

    ``unified`` is None for iter_group (reduce each column at its native dtype) or the
    per-group 2D component's unified dtype for iter_group_array.
    """
    dt = values.dtype
    if op == 'count':  # group sizes: always int64, regardless of value dtype
        return values, DTYPE_INT_DEFAULT
    if unified is None:  # iter_group: native per-column dtype
        post_dt = ufunc_dtype_to_dtype(func, dt)
        if post_dt is None:
            return None
        return values, post_dt
    if unified == DTYPE_OBJECT:
        # iter_group_array on a mixed frame: the object 2D component reduces via Python
        # numbers, so the loop's value-inferred output is int64 for integer columns and
        # float64 for float columns (narrow widths are erased by object boxing)
        if dt.kind in 'iu':
            return values, DTYPE_INT_DEFAULT
        if dt.kind == 'f':
            return values.astype(DTYPE_FLOAT_DEFAULT), DTYPE_FLOAT_DEFAULT
        return None  # a non-numeric column in an object component
    # a numeric unified dtype: reduce each column cast to it (as the 2D array does)
    post_dt = ufunc_dtype_to_dtype(func, unified)
    if post_dt is None:
        return None
    acc = values if dt == unified else values.astype(unified)
    return acc, post_dt


# -------------------------------------------------------------------------------
class Reduce:
    """The `Reduce` interface exposes methods for applying functions to one or more `Frame`s that return a new `Frame`. The `Reduce` instance is configured via constructors on `ReduceDispatch`."""

    __slots__ = ()

    _INTERFACE: tp.Tuple[str, ...] = (
        'keys',
        '__iter__',
        'items',
        'values',
        'to_frame',
    )

    def _prepare_items(
        self,
        axis: int,
        items: TIterableFrameItems,
    ) -> tp.Tuple[tp.Sequence[TLabel], tp.Sequence[TFrameOrArray], TShape2D]:
        raise NotImplementedError()  # pragma: no cover

    def _get_iter(
        self,
        components: tp.Sequence[TFrameOrArray],
        labels: tp.Sequence[TLabel],
        shape: TShape2D,
        sample: TFrameOrArray,
        is_array: bool,
    ) -> tp.Iterator[Series | TFrameAny | TNDArrayAny]:
        raise NotImplementedError()  # pragma: no cover

    # ---------------------------------------------------------------------------
    # dictionary-like interface
    _items: TIterableFrameItems
    _axis: int

    def keys(self) -> tp.Iterator[TLabel]:
        labels, _, _ = self._prepare_items(
            self._axis,
            self._items,
        )
        yield from labels

    def __iter__(self) -> tp.Iterator[TLabel]:
        yield from self.keys()

    def items(self) -> tp.Iterator[tp.Tuple[TLabel, Series | TFrameAny | TNDArrayAny]]:
        labels, components, shape = self._prepare_items(
            self._axis,
            self._items,
        )
        if components:
            sample = components[0]
        else:  # an empty iterator
            sample = EMPTY_ARRAY

        return zip(
            labels,
            self._get_iter(
                components=components,
                labels=labels,
                shape=shape,
                sample=sample,
                is_array=sample.__class__ is np.ndarray,
            ),
        )

    def values(self) -> tp.Iterator[Series | TFrameAny | TNDArrayAny]:
        yield from (v for _, v in self.items())

    def to_frame(
        self,
        *,
        index: tp.Optional[tp.Union[TIndexInitializer, TIndexAutoFactory]] = None,
        columns: tp.Optional[tp.Union[TIndexInitializer, TIndexAutoFactory]] = None,
        index_constructor: TIndexCtorSpecifier = None,
        columns_constructor: TIndexCtorSpecifier = None,
        name: TName = None,
        consolidate_blocks: bool = False,
    ) -> TFrameAny:
        """
        Return a ``Frame`` after processing column reduction functions.
        """
        raise NotImplementedError()  # pragma: no cover
        # TODO: add `retain_labels`


class ReduceComponent(Reduce):
    """`ReduceComponent` reduces by applying a function to the entire component (an array or `Frame`) and collecting the resulting `Series` or `Frame`. If an "items" iterator is used, the function will be supplied two arguments, the label and the component."""

    __slots__ = (
        '_items',
        '_func',
        '_yield_type',
        '_axis',
        '_fill_value',
    )

    def __init__(
        self,
        items: TIterableFrameItems,
        func: TUFunc,
        yield_type: IterNodeType,
        axis: int = 1,
        fill_value: tp.Any = np.nan,
    ):
        """
        Args:
            axis_labels: Index on the axis used to label reductions.
        """
        self._items = items
        self._func = func
        self._yield_type = yield_type
        self._axis = axis
        self._fill_value = fill_value

    def _prepare_items(
        self,
        axis: int,
        items: TIterableFrameItems,
    ) -> tp.Tuple[tp.Sequence[TLabel], tp.Sequence[TFrameOrArray], TShape2D]:
        labels: tp.List[TLabel] = []
        components: tp.List[TFrameOrArray] = []
        for label, component in items:
            labels.append(label)
            components.append(component)
        return labels, components, (-1, -1)

    def _get_iter(
        self,
        components: tp.Sequence[TFrameOrArray],
        labels: tp.Sequence[TLabel],
        shape: TShape2D,
        sample: TFrameOrArray,
        is_array: bool,
    ) -> tp.Iterator[Series | TFrameAny | TNDArrayAny]:
        """
        Return an iterator of ``Series`` after processing column reduction functions.
        """
        if self._axis == 1:  # each component reduces to a row
            if self._yield_type == IterNodeType.VALUES:
                for f in components:
                    yield self._func(f)
            else:
                for label, f in zip(labels, components):
                    yield self._func(label, f)
        else:  # each component reduces to a column
            raise NotImplementedError()  # pragma: no cover

    # ---------------------------------------------------------------------------
    def to_frame(
        self,
        *,
        index: tp.Optional[tp.Union[TIndexInitializer, TIndexAutoFactory]] = None,
        columns: tp.Optional[tp.Union[TIndexInitializer, TIndexAutoFactory]] = None,
        index_constructor: TIndexCtorSpecifier = None,
        columns_constructor: TIndexCtorSpecifier = None,
        name: TName = None,
        consolidate_blocks: bool = False,
    ) -> TFrameAny:
        """
        Return a ``Frame`` after processing column reduction functions.
        """

        labels, components, _ = self._prepare_items(
            self._axis,
            self._items,
        )
        if components:
            sample = components[0]
        else:
            sample = EMPTY_ARRAY

        is_array = sample.__class__ is np.ndarray
        parts = self._get_iter(
            components=components,
            labels=labels,
            shape=(-1, -1),
            sample=sample,
            is_array=is_array,
        )
        if not is_array:
            return Frame.from_concat(
                parts,  # type: ignore
                axis=0,
                union=True,
                index=index,
                index_constructor=index_constructor,
                columns=columns,
                columns_constructor=columns_constructor,
                name=name,
                consolidate_blocks=consolidate_blocks,
                fill_value=self._fill_value,
            )
        part: tp.Iterable[TNDArrayAny] = list(parts)  # type: ignore
        if not part:
            block = FRAME_INITIALIZER_DEFAULT
        else:
            block = concat_resolved(part, 0)  # immutable
        if columns is None:
            columns = IndexAutoFactory
        if index is None:
            index = IndexAutoFactory
        return Frame(
            block,  # type: ignore
            index=index,
            index_constructor=index_constructor,
            columns=columns,
            columns_constructor=columns_constructor,
            name=name,
        )


class ReduceAxis(Reduce):
    """`ReduceAxis` reduces along an axis (i.e., columns) by applying one or more functions on each column to return a 1D Series (or array) for each component."""

    __slots__ = (
        '_axis',
        '_items',
        '_axis_labels',
        '_axis_len',
        '_yield_type',
    )

    _INTERFACE: tp.Tuple[str, ...] = (
        '__iter__',
        'keys',
        'values',
        'items',
        'to_frame',
    )

    _items: TIterableFrameItems
    _axis_labels: IndexBase | tp.Sequence[TLabel] | None
    _axis: int
    _axis_len: int
    _yield_type: IterNodeType

    @staticmethod
    def _derive_row_dtype_array(
        sample: TNDArrayAny,
        iloc_to_func: TListILocToFunc,
    ) -> TDtypeAny | None:
        dt_src = sample.dtype  # an array
        dtype: TDtypeAny | None = None
        for _, func in iloc_to_func:
            if not (dt := ufunc_dtype_to_dtype(func, dt_src)):
                return None
            if dtype is None:
                dtype = dt
            dtype = resolve_dtype(dtype, dt)
            if dtype == DTYPE_OBJECT:
                return dtype
        return dtype

    @staticmethod
    def _derive_row_dtype_frame(
        sample: Frame,
        iloc_to_func: TListILocToFunc,
    ) -> TDtypeAny | None:
        dt_src = sample._blocks.dtypes  # an array
        dtype = None
        for iloc, func in iloc_to_func:
            if not (dt := ufunc_dtype_to_dtype(func, dt_src[iloc])):
                return None
            if dtype is None:
                dtype = dt
            dtype = resolve_dtype(dtype, dt)
            if dtype == DTYPE_OBJECT:
                return dtype
        return dtype

    def _get_blocks(
        self,
        components: tp.Sequence[TFrameOrArray],
        labels: tp.Sequence[TLabel],
        shape: TShape2D,
        sample: TFrameOrArray,
        is_array: bool,
    ) -> tp.Sequence[TNDArrayAny]:
        raise NotImplementedError()  # pragma: no cover

    def _to_frame_fast(
        self,
        *,
        index: tp.Optional[tp.Union[TIndexInitializer, TIndexAutoFactory]],
        columns: tp.Optional[tp.Union[TIndexInitializer, TIndexAutoFactory]],
        index_constructor: TIndexCtorSpecifier,
        columns_constructor: TIndexCtorSpecifier,
        name: TName,
        consolidate_blocks: bool,
    ) -> tp.Optional[TFrameAny]:
        """A vectorized hash-group reduction, or None when it does not apply."""
        return None

    def _prepare_items(
        self,
        axis: int,
        items: TIterableFrameItems,
    ) -> tp.Tuple[tp.Sequence[TLabel], tp.Sequence[TFrameOrArray], TShape2D]:
        labels: tp.List[TLabel] = []
        components: tp.List[TFrameOrArray] = []
        for label, component in items:
            labels.append(label)
            # NOTE: could assert uniformity of shape / labels here
            components.append(component)

        shape = (len(labels), self._axis_len)  # axis == 1
        # shape = (self._axis_len, len(labels))

        return labels, components, shape

    # ---------------------------------------------------------------------------
    def to_frame(
        self,
        *,
        index: tp.Optional[tp.Union[TIndexInitializer, TIndexAutoFactory]] = None,
        columns: tp.Optional[tp.Union[TIndexInitializer, TIndexAutoFactory]] = None,
        index_constructor: TIndexCtorSpecifier = None,
        columns_constructor: TIndexCtorSpecifier = None,
        name: TName = None,
        consolidate_blocks: bool = False,
    ) -> TFrameAny:
        """
        Return a ``Frame`` after processing column reduction functions.
        """
        post = self._to_frame_fast(
            index=index,
            columns=columns,
            index_constructor=index_constructor,
            columns_constructor=columns_constructor,
            name=name,
            consolidate_blocks=consolidate_blocks,
        )
        if post is not None:
            return post

        labels, components, shape = self._prepare_items(
            self._axis,
            self._items,
        )
        if components:
            sample = components[0]
        else:  # return a zero-row Frame
            raise NotImplementedError()  # pragma: no cover

        is_array = sample.__class__ is np.ndarray
        blocks = self._get_blocks(components, labels, shape, sample, is_array)

        own_columns = False
        if columns is None:
            if isinstance(self._axis_labels, IndexBase):
                # NOTE: this is implicitly only ReduceAligned
                columns = self._axis_labels[[pair[0] for pair in self._iloc_to_func]]  # type: ignore
                own_columns = True
            else:
                columns = self._axis_labels
                own_columns = False

        # implement consolidate_blocks
        tb = TypeBlocks.from_blocks(blocks)
        if consolidate_blocks:
            tb = tb.consolidate()

        if self._axis == 1:
            if index is None:
                index = labels

        return Frame(
            tb,
            index=index,
            columns=columns,
            own_columns=own_columns,
            name=name,
            own_data=True,
            index_constructor=index_constructor,
            columns_constructor=columns_constructor,
        )


class ReduceAligned(ReduceAxis):
    """Utilities for Reducing a `Frame` (or many `Frame`) by applying functions to columns."""

    # Axis 1 will reduce components into rows (labels are the index, ilocs refer to column positions); axis 0 will reduce components into columns (labels are the column labels, ilocs refer to index positions).

    __slots__ = ('_iloc_to_func', '_group_source')

    def __init__(
        self,
        items: TIterableFrameItems,
        iloc_to_func: TListILocToFunc,
        axis_labels: IndexBase | tp.Sequence[TLabel],
        yield_type: IterNodeType,
        axis: int = 1,
        /,
        *,
        group_source: tp.Optional[tp.Tuple[TFrameAny, int, bool]] = None,
    ):
        """
        Args:
            axis_labels: Index on the axis used to label reductions.
            group_source: when reducing the groups of a single ``Frame`` by a single
                column key, ``(frame, key_iloc, as_array)``; enables the vectorized fast
                path. ``as_array`` marks iter_group_array (reduce at the component's
                unified 2D dtype) versus iter_group (reduce each column at its dtype).
        """
        self._items = items
        self._iloc_to_func = iloc_to_func
        self._axis_labels = axis_labels
        self._yield_type = yield_type
        self._axis = axis
        self._axis_len = len(self._iloc_to_func)
        self._group_source = group_source

    def _to_frame_fast(
        self,
        *,
        index: tp.Optional[tp.Union[TIndexInitializer, TIndexAutoFactory]],
        columns: tp.Optional[tp.Union[TIndexInitializer, TIndexAutoFactory]],
        index_constructor: TIndexCtorSpecifier,
        columns_constructor: TIndexCtorSpecifier,
        name: TName,
        consolidate_blocks: bool,
    ) -> tp.Optional[TFrameAny]:
        """
        Vectorized group reduction: factorize the single-column key once, then reduce
        each target column with ``arraykit.group_reduce`` -- an O(n) hash aggregation
        replacing the per-group Python loop. Returns None (fall back to the loop) unless
        the reduction is a single-column key on one Frame, axis-1 reduction, every
        function a recognized reducer, and every (dtype, op) supported by ``group_reduce``.

        For iter_group_array the reduction target is the component's unified 2D dtype: a
        numeric unified dtype reduces each column cast to it, while an object unified dtype
        (a mixed frame) reduces via Python numbers, so integer columns yield int64 and
        float columns yield float64 -- matching the loop's value-inferred output.

        Results are identical to the per-group loop for integer sum/prod, min/max, and
        count. Float ``sum``/``prod`` may differ from the loop by floating-point rounding
        (~1 ULP): ``group_reduce`` accumulates in array order while the loop's ``np.sum``
        accumulates in group order -- both valid, neither canonical.
        """
        source = self._group_source
        if source is None or self._axis != 1:
            return None

        # every function must map to a group_reduce op
        ops: tp.List[str] = []
        for _, func in self._iloc_to_func:
            op = _REDUCE_FUNC_TO_OP.get(func)
            if op is None:
                return None
            ops.append(op)

        frame, key_iloc, as_array = source
        blocks = frame._blocks
        key = np.ascontiguousarray(blocks._extract_array_column(key_iloc))
        if len(key) == 0:
            # zero groups: the loop path raises NotImplementedError; match it
            return None
        # the unified dtype iter_group_array would give each per-group 2D component
        unified = blocks._index.dtype if as_array else None
        try:
            # sort=True yields groups in the same (sorted, NaN-last) order as iter_group
            uniques, codes = factorize(key, sort=True)
        except (TypeError, ValueError):
            return None
        codes = np.ascontiguousarray(codes)
        size = len(uniques)

        out_blocks: tp.List[TNDArrayAny] = []
        for (iloc, func), op in zip(self._iloc_to_func, ops):
            values = blocks._extract_array_column(iloc)
            plan = _reduce_column_plan(func, op, values, unified)
            if plan is None:  # dtype the fast path cannot reproduce -> fall back
                return None
            acc_values, post_dt = plan
            try:
                block = group_reduce(
                    codes, size, np.ascontiguousarray(acc_values), op
                )
            except (TypeError, ValueError):
                # dtype/op unsupported by group_reduce (e.g. float32 sum) -> fall back
                return None
            if block.dtype != post_dt:  # cast the 64-bit accumulator to numpy's dtype
                block = block.astype(post_dt)
            block.flags.writeable = False
            out_blocks.append(block)

        own_columns = False
        if columns is None:
            if isinstance(self._axis_labels, IndexBase):
                columns = self._axis_labels[
                    [pair[0] for pair in self._iloc_to_func]
                ]
                own_columns = True
            else:
                columns = self._axis_labels

        tb = TypeBlocks.from_blocks(out_blocks)
        if consolidate_blocks:
            tb = tb.consolidate()
        if index is None:
            index = uniques

        return Frame(
            tb,
            index=index,
            columns=columns,
            own_columns=own_columns,
            name=name,
            own_data=True,
            index_constructor=index_constructor,
            columns_constructor=columns_constructor,
        )

    def _get_blocks(
        self,
        components: tp.Sequence[TFrameOrArray],
        labels: tp.Sequence[TLabel],
        shape: TShape2D,
        sample: TFrameOrArray,
        is_array: bool,
    ) -> tp.Sequence[TNDArrayAny]:
        blocks: tp.List[TNDArrayAny] = []
        v: TNDArrayAny | tp.List[tp.Any]

        if is_array:
            dtype = sample.dtype  # type: ignore
            if self._axis == 1:  # each component reduces to a row
                size = shape[0]
                for iloc, func in self._iloc_to_func:
                    post_dt = ufunc_dtype_to_dtype(func, dtype)
                    if post_dt is not None:
                        v = np.empty(size, dtype=post_dt)
                    else:
                        v = [None] * size
                    if self._yield_type == IterNodeType.VALUES:
                        for i, array in enumerate(components):
                            v[i] = func(array[NULL_SLICE, iloc])
                    else:
                        for i, (label, array) in enumerate(zip(labels, components)):
                            v[i] = func(label, array[NULL_SLICE, iloc])

                    if v.__class__ is not np.ndarray:
                        v, _ = iterable_to_array_1d(v, count=size)
                    v.flags.writeable = False
                    blocks.append(v)
            else:  # each component reduces to a column
                raise NotImplementedError()  # pragma: no cover

        else:  # component is a Frame
            dtypes = sample._blocks.dtypes  # type: ignore
            if self._axis == 1:
                # each component reduces to a row
                size = shape[0]
                for iloc, func in self._iloc_to_func:
                    post_dt = ufunc_dtype_to_dtype(func, dtypes[iloc])
                    if post_dt is not None:
                        v = np.empty(size, dtype=post_dt)
                    else:
                        v = [None] * size
                    if self._yield_type == IterNodeType.VALUES:
                        for i, frame in enumerate(components):
                            v[i] = func(frame._blocks._extract_array_column(iloc))  # type: ignore
                    else:
                        for i, (label, frame) in enumerate(zip(labels, components)):
                            v[i] = func(label, frame._blocks._extract_array_column(iloc))  # type: ignore
                    if v.__class__ is not np.ndarray:
                        v, _ = iterable_to_array_1d(v, count=size)
                    v.flags.writeable = False
                    blocks.append(v)
            else:  # each component reduces to a column
                raise NotImplementedError()  # pragma: no cover
        return blocks

    def _get_iter(
        self,
        components: tp.Sequence[TFrameOrArray],
        labels: tp.Sequence[TLabel],
        shape: TShape2D,
        sample: TFrameOrArray,
        is_array: bool,
    ) -> tp.Iterator[Series | TFrameAny | TNDArrayAny]:
        """
        Return an iterator of ``Series`` after processing column reduction functions.
        """
        index: IndexBase | tp.Sequence[TLabel]

        if isinstance(self._axis_labels, IndexBase):
            # TODO: handle static
            index = self._axis_labels[[pair[0] for pair in self._iloc_to_func]]  # pyright: ignore
            own_index = True
        elif self._axis_labels is not None:
            # a sequence of labels to be used
            index = self._axis_labels
            own_index = False
        else:
            raise NotImplementedError()  # pragma: no cover

        # We are yielding rows that result from each columnar function application; using the dtype of sample, the dtype expected from func, across all funcs, we can determine the resultant array dtype and not use a list, below
        assert self._axis == 1
        v: TNDArrayAny | tp.List[tp.Any]
        size = shape[1]

        if is_array:
            if self._yield_type == IterNodeType.VALUES:
                # this only works if IterNodeType.VALUES, as we cannot identify the function of it takes a pair
                dtype = self._derive_row_dtype_array(sample, self._iloc_to_func)  # type: ignore
                if dtype is not None:
                    for array in components:
                        v = np.empty(size, dtype=dtype)
                        for i, (iloc, func) in enumerate(self._iloc_to_func):
                            v[i] = func(array[NULL_SLICE, iloc])
                        v.flags.writeable = False
                        yield v
                else:
                    for array in components:
                        v = [None] * size
                        for i, (iloc, func) in enumerate(self._iloc_to_func):
                            v[i] = func(array[NULL_SLICE, iloc])
                        v, _ = iterable_to_array_1d(v, count=size)
                        yield v
            else:  # items
                for label, array in zip(labels, components):
                    v = [None] * size
                    for i, (iloc, func) in enumerate(self._iloc_to_func):
                        v[i] = func(label, array[NULL_SLICE, iloc])
                    v, _ = iterable_to_array_1d(v, count=size)
                    yield v

        else:  # component is a Frame
            if self._yield_type == IterNodeType.VALUES:
                dtype = self._derive_row_dtype_frame(sample, self._iloc_to_func)  # type: ignore
                if dtype is not None:
                    for label, f in zip(labels, components):
                        v = np.empty(size, dtype=dtype)
                        for i, (iloc, func) in enumerate(self._iloc_to_func):
                            v[i] = func(f._extract(NULL_SLICE, iloc))  # type: ignore
                        v.flags.writeable = False
                        yield Series(v, index=index, name=label, own_index=own_index)
                else:
                    for label, f in zip(labels, components):
                        v = [None] * size
                        for i, (iloc, func) in enumerate(self._iloc_to_func):
                            v[i] = func(f._extract(NULL_SLICE, iloc))  # type: ignore
                        v, _ = iterable_to_array_1d(v, count=size)
                        yield Series(v, index=index, name=label, own_index=own_index)
            else:  # items
                for label, f in zip(labels, components):
                    v = [None] * size
                    for i, (iloc, func) in enumerate(self._iloc_to_func):
                        v[i] = func(label, f._extract(NULL_SLICE, iloc))  # type: ignore
                    v, _ = iterable_to_array_1d(v, count=size)
                    yield Series(v, index=index, name=label, own_index=own_index)


class ReduceUnaligned(ReduceAxis):
    """Utilities for Reducing a `Frame` (or many `Frame`) by applying functions to columns."""

    __slots__ = (
        '_loc_to_func',
        '_fill_value',
    )

    def __init__(
        self,
        items: TIterableFrameItems,
        loc_to_func: TListLabelToFunc,
        axis_labels: tp.Sequence[TLabel] | None,
        yield_type: IterNodeType,
        axis: int = 1,
        fill_value: tp.Any = np.nan,
        /,
    ):
        """
        Args:
            axis_labels: Index on the axis used to label reductions.
        """
        self._items = items
        self._loc_to_func = loc_to_func
        self._axis_labels = axis_labels
        self._yield_type = yield_type
        self._axis = axis
        self._axis_len = len(self._loc_to_func)
        self._fill_value = fill_value

    def _get_blocks(
        self,
        components: tp.Sequence[TFrameOrArray],
        labels: tp.Sequence[TLabel],
        shape: TShape2D,
        sample: TFrameOrArray,
        is_array: bool,
    ) -> tp.Sequence[TNDArrayAny]:
        assert not is_array  # arrays cannot be supported for unaligned reduce

        blocks: tp.List[TNDArrayAny] = []
        v: TNDArrayAny | tp.List[tp.Any]
        assert self._axis == 1  # each component reduces to a row
        size = shape[0]

        for loc, func in self._loc_to_func:
            # NOTE: we cannot easily predict array type as the sample may not be representative
            v = [None] * size
            if self._yield_type == IterNodeType.VALUES:
                for i, frame in enumerate(components):
                    try:
                        iloc = frame.columns.loc_to_iloc(loc)  # type: ignore
                    except KeyError:
                        iloc = -1
                    if iloc >= 0:
                        v[i] = func(frame._blocks._extract_array_column(iloc))  # type: ignore
                    else:
                        v[i] = self._fill_value
            else:
                for i, (label, frame) in enumerate(zip(labels, components)):
                    try:
                        iloc = frame.columns.loc_to_iloc(loc)  # type: ignore
                    except KeyError:
                        iloc = -1
                    if iloc >= 0:
                        v[i] = func(label, frame._blocks._extract_array_column(iloc))  # type: ignore
                    else:
                        v[i] = self._fill_value

            v, _ = iterable_to_array_1d(v, count=size)
            v.flags.writeable = False
            blocks.append(v)

        return blocks

    def _get_iter(
        self,
        components: tp.Sequence[TFrameOrArray],
        labels: tp.Sequence[TLabel],
        shape: TShape2D,
        sample: TFrameOrArray,
        is_array: bool,
    ) -> tp.Iterator[Series | TFrameAny | TNDArrayAny]:
        """
        Return an iterator of ``Series`` after processing column reduction functions.
        """
        assert not is_array  # arrays cannot be supported for unaligned reduce
        assert self._axis == 1  # each component reduces to a row
        v: TNDArrayAny | tp.List[tp.Any]
        size = shape[1]
        fv = self._fill_value
        # NOTE: we cannot easily predict array type as we do not have a representative sample of the contained frame
        for label, f in zip(labels, components):
            v = [None] * size
            if self._yield_type == IterNodeType.VALUES:
                for i, (loc, func) in enumerate(self._loc_to_func):
                    try:
                        iloc = f.columns.loc_to_iloc(loc)  # type: ignore
                    except KeyError:
                        iloc = -1
                    if iloc >= 0:
                        v[i] = func(f._extract(NULL_SLICE, iloc))  # type: ignore
                    else:
                        v[i] = fv
            else:
                for i, (loc, func) in enumerate(self._loc_to_func):
                    try:
                        iloc = f.columns.loc_to_iloc(loc)  # type: ignore
                    except KeyError:
                        iloc = -1
                    if iloc >= 0:
                        v[i] = func(label, f._extract(NULL_SLICE, iloc))  # type: ignore
                    else:
                        v[i] = fv

            v, _ = iterable_to_array_1d(v, count=size)
            yield Series(v, index=self._axis_labels, name=label)


# -------------------------------------------------------------------------------

INTERFACE_REDUCE_DISPATCH: tp.Tuple[str, ...] = (
    'from_func',
    'from_map_func',
    'from_label_map',
    'from_label_pair_map',
)


class ReduceDispatch(Interface):
    """Interface for exposing `Reduce` constructors."""

    __slots__ = (
        '_items',
        '_yield_type',
        '_axis',
    )

    CLS_DELEGATE = Reduce
    _INTERFACE = INTERFACE_REDUCE_DISPATCH

    _items: TIterableFrameItems
    _yield_type: IterNodeType
    _axis: int

    def from_func(
        self,
        func: TUFunc,
        *,
        fill_value: tp.Any = np.nan,
    ) -> ReduceComponent:
        """For each `Frame`, and given a function `func` that returns either a `Series` or a `Frame`, call that function on each `Frame`."""
        return ReduceComponent(
            self._items,
            func,
            yield_type=self._yield_type,
            axis=self._axis,
            fill_value=fill_value,
        )

    def from_map_func(
        self,
        func: TUFunc,
        *,
        fill_value: tp.Any = np.nan,
    ) -> Reduce:
        raise NotImplementedError()  # pragma: no cover

    def from_label_map(
        self,
        func_map: tp.Mapping[TLabel, TUFunc],
        *,
        fill_value: tp.Any = np.nan,
    ) -> Reduce:
        raise NotImplementedError()  # pragma: no cover

    def from_label_pair_map(
        self,
        func_map: tp.Mapping[tp.Tuple[TLabel, TLabel], TUFunc],
        *,
        fill_value: tp.Any = np.nan,
    ) -> Reduce:
        raise NotImplementedError()  # pragma: no cover


class ReduceDispatchAligned(ReduceDispatch):
    """Interface for creating reductions from uniform collections of Frames."""

    __slots__ = ('_axis_labels', '_group_source')

    def __init__(
        self,
        items: TIterableFrameItems,
        axis_labels: IndexBase,  # always an index
        *,
        yield_type: IterNodeType,
        axis: int = 1,
        group_source: tp.Optional[tp.Tuple[TFrameAny, int, bool]] = None,
    ) -> None:
        """
        Args:
            axis_labels: Index on the axis used to label reductions.
            group_source: ``(frame, key_iloc, as_array)`` when these are the groups of a
                single Frame reduced by one column key; enables the vectorized fast path.
        """
        self._items = items
        self._axis_labels = axis_labels
        self._yield_type = yield_type
        self._axis = axis
        self._group_source = group_source

    def from_map_func(
        self,
        func: TUFunc,
        *,
        fill_value: tp.Any = np.nan,
    ) -> ReduceAligned:
        """
        For each `Frame`, reduce by applying, for each column, a function that reduces to (0-dimensional) elements, where the column label and function are given as a mapping. Column labels are retained.
        """
        iloc_to_func: TListILocToFunc = list(
            zip(
                range(len(self._axis_labels)),
                repeat(func),
            )
        )
        return ReduceAligned(
            self._items,
            iloc_to_func,
            self._axis_labels,
            self._yield_type,
            self._axis,
            group_source=self._group_source,
        )

    def from_label_map(
        self,
        func_map: tp.Mapping[TLabel, TUFunc],
        *,
        fill_value: tp.Any = np.nan,
    ) -> ReduceAligned:
        """
        For `Frame`, reduce by applying a function to each column, where the column label and function are given as a mapping. Column labels are retained.

        Args:
            func_map: a mapping of column labels to functions.
        """
        loc_to_iloc = self._axis_labels.loc_to_iloc

        iloc_to_func: TListILocToFunc = list(
            (loc_to_iloc(label), func) for label, func in func_map.items()
        )
        return ReduceAligned(
            self._items,
            iloc_to_func,
            self._axis_labels,
            self._yield_type,
            self._axis,
            group_source=self._group_source,
        )

    def from_label_pair_map(
        self,
        func_map: tp.Mapping[tp.Tuple[TLabel, TLabel], TUFunc],
        *,
        fill_value: tp.Any = np.nan,
    ) -> ReduceAligned:
        """
        For `Frame`, reduce by applying a function to a column and assigning the result a new label. Functions are provided as values in a mapping, where the key is tuple of source label, destination label.

        Args:
            func_map: a mapping of pairs of source label, destination label, to a function.

        """
        loc_to_iloc = self._axis_labels.loc_to_iloc

        iloc_to_func: TListILocToFunc = []
        axis_labels = []
        for (iloc, label), func in func_map.items():
            axis_labels.append(label)
            iloc_to_func.append((loc_to_iloc(iloc), func))
        # NOTE: ignore self._axis_labels
        return ReduceAligned(
            self._items,
            iloc_to_func,
            axis_labels,
            self._yield_type,
            self._axis,
        )


# -------------------------------------------------------------------------------
class ReduceDispatchUnaligned(ReduceDispatch):
    """Delegate interface for creating reductions from uniform collections of Frames."""

    _INTERFACE = INTERFACE_REDUCE_DISPATCH

    def __init__(
        self,
        items: TIterableFrameItems,
        *,
        axis: int = 1,
        yield_type: IterNodeType,
    ) -> None:
        """
        Args:
            axis_labels: Index on the axis used to label reductions.
        """
        self._items = items
        self._axis = axis
        self._yield_type = yield_type

    def from_map_func(
        self,
        func: TUFunc,
        *,
        fill_value: tp.Any = np.nan,
    ) -> ReduceComponent:
        # get a ReduceDispatchAligned
        if self._yield_type == IterNodeType.VALUES:

            def func_derived(f: Frame) -> Series:  # pyright: ignore
                return next(iter(f.reduce.from_map_func(func).values()))  # type: ignore
        else:

            def func_derived(l: TLabel, f: Frame) -> Series:
                return next(iter(f.reduce.from_map_func(func).values()))  # type: ignore

        return ReduceComponent(
            self._items,
            func_derived,  # type: ignore
            yield_type=self._yield_type,
            axis=self._axis,
            fill_value=fill_value,
        )

    def from_label_map(
        self,
        func_map: tp.Mapping[TLabel, TUFunc],
        *,
        fill_value: tp.Any = np.nan,
    ) -> ReduceUnaligned:
        """
        For `Frame`, reduce by applying a function to each column, where the column label and function are given as a mapping. Column labels are retained.

        Args:
            func_map: a mapping of column labels to functions.
        """
        loc_to_func: TListLabelToFunc = []
        axis_labels: tp.List[TLabel] = []
        for pair in func_map.items():
            axis_labels.append(pair[0])
            loc_to_func.append(pair)
        return ReduceUnaligned(
            self._items,
            loc_to_func,
            axis_labels,
            self._yield_type,
            self._axis,
            fill_value,
        )

    def from_label_pair_map(
        self,
        func_map: tp.Mapping[tp.Tuple[TLabel, TLabel], TUFunc],
        *,
        fill_value: tp.Any = np.nan,
    ) -> ReduceUnaligned:
        """
        For `Frame`, reduce by applying a function to a column and assigning the result a new label. Functions are provided as values in a mapping, where the key is tuple of source label, destination label.

        Args:
            func_map: a mapping of pairs of source label, destination label, to a function.

        """
        loc_to_func: TListLabelToFunc = []
        axis_labels = []
        for (loc, label), func in func_map.items():
            axis_labels.append(label)
            loc_to_func.append((loc, func))
        # NOTE: ignore self._axis_labels
        return ReduceUnaligned(
            self._items,
            loc_to_func,
            axis_labels,
            self._yield_type,
            self._axis,
            fill_value,
        )


# -------------------------------------------------------------------------------
class InterfaceBatchReduceDispatch(InterfaceBatch):
    """Alternate string interface specialized for the :obj:`Batch`."""

    __slots__ = ('_batch_apply',)
    _INTERFACE = INTERFACE_REDUCE_DISPATCH

    def __init__(
        self,
        batch_apply: tp.Callable[[TCallableAny], 'Batch'],
    ) -> None:
        self._batch_apply = batch_apply

    # ---------------------------------------------------------------------------
    def from_func(
        self,
        func: TUFunc,
        *,
        fill_value: tp.Any = np.nan,
    ) -> 'Batch':
        return self._batch_apply(
            lambda f: f.reduce.from_func(
                func,
                fill_value=fill_value,
            ).to_frame()
        )

    def from_map_func(
        self,
        func: TUFunc,
        *,
        fill_value: tp.Any = np.nan,
    ) -> 'Batch':
        return self._batch_apply(
            lambda f: f.reduce.from_map_func(
                func,
                fill_value=fill_value,
            ).to_frame()
        )

    def from_label_map(
        self,
        func_map: tp.Mapping[TLabel, TUFunc],
        *,
        fill_value: tp.Any = np.nan,
    ) -> 'Batch':
        return self._batch_apply(
            lambda f: f.reduce.from_label_map(
                func_map,
                fill_value=fill_value,
            ).to_frame()
        )

    def from_label_pair_map(
        self,
        func_map: tp.Mapping[tp.Tuple[TLabel, TLabel], TUFunc],
        *,
        fill_value: tp.Any = np.nan,
    ) -> 'Batch':
        return self._batch_apply(
            lambda f: f.reduce.from_label_pair_map(
                func_map,
                fill_value=fill_value,
            ).to_frame()
        )
