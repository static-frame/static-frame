from __future__ import annotations

from itertools import repeat

import numpy as np
import typing_extensions as tp
from arraykit import resolve_dtype

from static_frame.core.frame import Frame
from static_frame.core.generic_aliases import TFrameAny
from static_frame.core.index_auto import TIndexAutoFactory
from static_frame.core.index_base import IndexBase
from static_frame.core.series import Series
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import IterNodeType
from static_frame.core.util import TILocSelectorOne
from static_frame.core.util import TIndexCtorSpecifier
from static_frame.core.util import TIndexInitializer
from static_frame.core.util import TLabel
from static_frame.core.util import TName
from static_frame.core.util import TUFunc
from static_frame.core.util import iterable_to_array_1d
from static_frame.core.util import ufunc_dtype_to_dtype

TNDArrayAny = np.ndarray[tp.Any, tp.Any]
TDtypeAny = np.dtype[tp.Any]

TFrameOrSeries = tp.Union[Frame, Series]
TFrameOrArray = tp.Union[Frame, TNDArrayAny]
TIterableFrameItems = tp.Iterable[tp.Tuple[TLabel, TFrameOrArray]]
TShape2D = tp.Tuple[int, int]

TILocToFunc = tp.List[tp.Tuple[TILocSelectorOne, TUFunc]]
TLabelToFunc = tp.List[tp.Tuple[TLabel, TUFunc]]

#-------------------------------------------------------------------------------

class Reduce:

    #---------------------------------------------------------------------------
    # dictionary-like interface

    def keys(self) -> tp.Iterator[TLabel]:
        labels, _, _ = self._prepare_items(
                self._axis,
                self._items,
                )
        yield from labels

    def __iter__(self) -> tp.Iterator[TLabel]:
        yield from self.keys()

    def items(self) -> tp.Iterator[tp.Tuple[TLabel, Series]]:
        labels, components, shape = self._prepare_items(
                self._axis,
                self._items,
                )
        if components:
            sample = components[0]
        else: # return a zero-row Frame
            raise NotImplementedError()

        is_array = sample.__class__ is np.ndarray

        return zip(labels, self._get_iter(
                components=components,
                shape=shape,
                sample=sample,
                is_array=is_array,
                labels=labels,
                ))

    def values(self) -> tp.Iterator[Series]:
        yield from (v for _, v in self.items())


class ReduceComponent(Reduce):
    '''`ReduceComponent` reduces by applying a function to the entire component (an array or `Frame`) and collecting the resulting `Series` or `Frame`.
    '''

    __slots__ = (
        '_items',
        '_func',
        '_axis',
        '_yield_type',
    )
    def __init__(self,
            items: TIterableFrameItems,
            func: TUFunc,
            yield_type: IterNodeType,
            axis: int = 1,
            ):
        '''
        Args:
            axis_labels: Index on the axis used to label reductions.
        '''
        self._items = items
        self._func = func
        self._axis = axis
        self._yield_type = yield_type

    def _prepare_items(self,
            axis: int,
            items: TIterableFrameItems,
            ) -> tp.Tuple[tp.Sequence[TLabel], tp.Sequence[TFrameOrArray], TShape2D]:

        labels: tp.List[TLabel] = []
        components: tp.List[TFrameOrArray] = []
        for label, component in items:
            labels.append(label)
            components.append(component)
        return labels, components, (-1, -1)

    def _get_iter(self,
            components: tp.Sequence[TFrameOrArray],
            is_array: bool,
            labels: tp.Sequence[TLabel],
            ) -> tp.Iterator[Series | Frame]:
        '''
        Return an iterator of ``Series`` after processing column reduction functions.
        '''
        if self._axis == 1: # each component reduces to a row
            for label, f in zip(labels, components):
                yield self._func(f)
        else:  # each component reduces to a column
            raise NotImplementedError()

    #---------------------------------------------------------------------------
    def to_frame(self, *,
            index: tp.Optional[tp.Union[TIndexInitializer, TIndexAutoFactory]] = None,
            columns: tp.Optional[tp.Union[TIndexInitializer, TIndexAutoFactory]] = None,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            name: TName = None,
            consolidate_blocks: bool = False
        ) -> TFrameAny:
        '''
        Return a ``Frame`` after processing column reduction functions.
        '''

        labels, components, _ = self._prepare_items(
                self._axis,
                self._items,
                )
        if components:
            sample = components[0]
        else: # return a zero-row Frame
            raise NotImplementedError()

        is_array = sample.__class__ is np.ndarray
        parts = self._get_iter(
                components=components,
                is_array=is_array,
                labels=labels,
                )
        return Frame.from_concat(
                parts,
                axis=0,
                union=True,
                index=index,
                index_constructor=index_constructor,
                columns=columns,
                columns_constructor=columns_constructor,
                name=name,
                consolidate_blocks=consolidate_blocks,
                )

class ReduceAxis(Reduce):
    '''`ReduceAxis` reduces along an axis (i.e., columns) by applying one or more functions on each column to return a 1D Series (or array) for each component.
    '''
    __slots__ = (
            '_axis',
            '_items',
            '_axis_labels',
            '_axis_len',
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

    @staticmethod
    def _derive_row_dtype_array(
            sample: TNDArrayAny,
            iloc_to_func: TILocToFunc,
            ) -> TDtypeAny | None:
        dt_src = sample.dtype # an array
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
            iloc_to_func: TILocToFunc,
            ) -> TDtypeAny | None:
        dt_src = sample._blocks.dtypes # an array
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

    def _prepare_items(self,
            axis: int,
            items: TIterableFrameItems,
            ) -> tp.Tuple[tp.Sequence[TLabel], tp.Sequence[TFrameOrArray], TShape2D]:

        labels: tp.List[TLabel] = []
        components: tp.List[TFrameOrArray] = []

        for label, component in items:
            labels.append(label)
            # NOTE: could assert uniformity of shape / labels here
            components.append(component)

        if axis == 1:
            shape = (len(labels), self._axis_len)
        else:
            shape = (self._axis_len, len(labels))
        return labels, components, shape

    #---------------------------------------------------------------------------
    def to_frame(self, *,
            index: tp.Optional[tp.Union[TIndexInitializer, TIndexAutoFactory]] = None,
            columns: tp.Optional[tp.Union[TIndexInitializer, TIndexAutoFactory]] = None,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            name: TName = None,
            consolidate_blocks: bool = False
        ) -> TFrameAny:
        '''
        Return a ``Frame`` after processing column reduction functions.
        '''

        labels, components, shape = self._prepare_items(
                self._axis,
                self._items,
                )
        if components:
            sample = components[0]
        else: # return a zero-row Frame
            raise NotImplementedError()

        is_array = sample.__class__ is np.ndarray
        blocks = self._get_blocks(components, shape, sample, is_array)

        own_columns = False
        if columns is None:
            if isinstance(self._axis_labels, IndexBase):
                columns = self._axis_labels[[pair[0] for pair in self._iloc_to_func]]
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

        return Frame(tb,
                index=index,
                columns=columns,
                own_columns=own_columns,
                name=name,
                own_data=True,
                index_constructor=index_constructor,
                columns_constructor=columns_constructor,
                )


class ReduceAligned(ReduceAxis):
    '''Utilities for Reducing a `Frame` (or many `Frame`) by applying functions to columns.
    '''
    # Axis 1 will reduce components into rows (labels are the index, ilocs refer to column positions); axis 0 will reduce components into columns (labels are the column labels, ilocs refer to index positions).

    __slots__ = (
            '_iloc_to_func',
            )

    def __init__(self,
            items: TIterableFrameItems,
            iloc_to_func: TILocToFunc,
            axis_labels: IndexBase | tp.Sequence[TLabel],
            axis: int = 1,
            ):
        '''
        Args:
            axis_labels: Index on the axis used to label reductions.
        '''
        self._items = items
        self._iloc_to_func = iloc_to_func
        self._axis_labels = axis_labels
        self._axis = axis
        self._axis_len = len(self._iloc_to_func)

    def _get_blocks(self,
            components: tp.Sequence[TFrameOrArray],
            shape: TShape2D,
            sample: TFrameOrArray,
            is_array: bool,
            ) -> tp.Sequence[TNDArrayAny]:

        blocks: tp.List[TNDArrayAny] = []
        v: TNDArrayAny | tp.List[tp.Any]

        if is_array:
            dtype = sample.dtype # type: ignore
            if self._axis == 1: # each component reduces to a row
                size = shape[0]
                for iloc, func in self._iloc_to_func:
                    post_dt = ufunc_dtype_to_dtype(func, dtype)
                    if post_dt is not None:
                        v = np.empty(size, dtype=post_dt)
                    else:
                        v = [None] * size

                    for i, array in enumerate(components):
                        v[i] = func(array[NULL_SLICE, iloc])

                    if not v.__class__ is np.ndarray:
                        v, _ = iterable_to_array_1d(v, count=size)
                    v.flags.writeable = False # type: ignore
                    blocks.append(v) # type: ignore
            else:  # each component reduces to a column
                raise NotImplementedError()

        else: # component is a Frame
            dtypes = sample._blocks.dtypes # type: ignore
            if self._axis == 1:
                # each component reduces to a row
                size = shape[0]
                for iloc, func in self._iloc_to_func:
                    post_dt = ufunc_dtype_to_dtype(func, dtypes[iloc])
                    if post_dt is not None:
                        v = np.empty(size, dtype=post_dt)
                    else:
                        v = [None] * size

                    for i, frame in enumerate(components):
                        v[i] = func(frame._blocks._extract_array_column(iloc)) # type: ignore

                    if not v.__class__ is np.ndarray:
                        v, _ = iterable_to_array_1d(v, count=size)
                    v.flags.writeable = False # type: ignore
                    blocks.append(v) # type: ignore
            else: # each component reduces to a column
                raise NotImplementedError()
        return blocks

    def _get_iter(self,
            components: tp.Sequence[TFrameOrArray],
            shape: TShape2D,
            sample: TFrameOrArray,
            is_array: bool,
            labels: tp.Sequence[TLabel],
            ) -> tp.Iterator[Series]:
        '''
        Return an iterator of ``Series`` after processing column reduction functions.
        '''
        index: IndexBase | tp.Sequence[TLabel]

        if isinstance(self._axis_labels, IndexBase):
            index = self._axis_labels[[pair[0] for pair in self._iloc_to_func]]
            own_index = True
        else:
            index = self._axis_labels
            own_index = False

        # We are yielding rows that result from each columnar function application; using the dtype of sample, the dtype expected from func, across all funcs, we can determine the resultant array dtype and not use a list, below

        v: TNDArrayAny | tp.List[tp.Any]
        if is_array:
            dtype = self._derive_row_dtype_array(sample, self._iloc_to_func)

            if self._axis == 1: # each component reduces to a row
                size = shape[1]
                if dtype is not None:
                    for label, array in zip(labels, components):
                        v = np.empty(size, dtype=dtype)
                        for i, (iloc, func) in enumerate(self._iloc_to_func):
                            v[i] = func(array[NULL_SLICE, iloc])
                        v.flags.writeable = False
                        yield v
                else:
                    for label, array in zip(labels, components):
                        v = [None] * size
                        for i, (iloc, func) in enumerate(self._iloc_to_func):
                            v[i] = func(array[NULL_SLICE, iloc])
                        v, _ = iterable_to_array_1d(v, count=size)
                        yield v
            else:  # each component reduces to a column
                raise NotImplementedError()

        else: # component is a Frame
            if self._axis == 1: # each component reduces to a row
                dtype = self._derive_row_dtype_frame(sample, self._iloc_to_func)

                size = shape[1]
                if dtype is not None:
                    for label, f in zip(labels, components):
                        v = np.empty(size, dtype=dtype)
                        for i, (iloc, func) in enumerate(self._iloc_to_func):
                            v[i] = func(f._extract(NULL_SLICE, iloc)) # type: ignore
                        v.flags.writeable = False
                        yield Series(v, index=index, name=label, own_index=own_index)
                else:
                    for label, f in zip(labels, components):
                        v = [None] * size
                        for i, (iloc, func) in enumerate(self._iloc_to_func):
                            v[i] = func(f._extract(NULL_SLICE, iloc)) # type: ignore
                        v, _ = iterable_to_array_1d(v, count=size)
                        yield Series(v, index=index, name=label, own_index=own_index)
            else:  # each component reduces to a column
                raise NotImplementedError()


class ReduceUnaligned(ReduceAxis):
    '''Utilities for Reducing a `Frame` (or many `Frame`) by applying functions to columns.
    '''
    __slots__ = (
            '_loc_to_func',
            '_fill_value',
            )

    def __init__(self,
            items: TIterableFrameItems,
            loc_to_func: TILocToFunc,
            axis_labels: tp.Sequence[TLabel] | None,
            axis: int = 1,
            fill_value: to.Any = np.nan,
            ):
        '''
        Args:
            axis_labels: Index on the axis used to label reductions.
        '''
        self._items = items
        self._loc_to_func = loc_to_func
        self._axis_labels = axis_labels
        self._axis = axis
        self._axis_len = len(self._loc_to_func)
        self._fill_value = fill_value

    def _get_blocks(self,
            components: tp.Sequence[TFrameOrArray],
            shape: TShape2D,
            sample: TFrameOrArray,
            is_array: bool,
            ) -> tp.Sequence[TNDArrayAny]:

        assert not is_array # arrays cannot be supported for unaligned reduce

        blocks: tp.List[TNDArrayAny] = []
        v: TNDArrayAny | tp.List[tp.Any]

        if self._axis == 1: # each component reduces to a row
            size = shape[0]
            for loc, func in self._loc_to_func:
                # NOTE: we cannot easily predict array type as we do not have a representative sample of the contained frame
                v = [None] * size
                for i, frame in enumerate(components):
                    try:
                        iloc = frame.columns.loc_to_iloc(loc)
                        v[i] = func(frame._blocks._extract_array_column(iloc)) # type: ignore
                    except KeyError:
                        v[i] = self._fill_value
                v, _ = iterable_to_array_1d(v, count=size)
                v.flags.writeable = False
                blocks.append(v)
        else: # each component reduces to a column
            raise NotImplementedError()
        return blocks

    def _get_iter(self,
            components: tp.Sequence[TFrameOrArray],
            shape: TShape2D,
            sample: TFrameOrArray,
            is_array: bool,
            labels: tp.Sequence[TLabel],
            ) -> tp.Iterator[Series]:
        '''
        Return an iterator of ``Series`` after processing column reduction functions.
        '''
        assert not is_array # arrays cannot be supported for unaligned reduce

        v: TNDArrayAny | tp.List[tp.Any]
        if self._axis == 1: # each component reduces to a row
            size = shape[1]
            for label, f in zip(labels, components):
                # NOTE: we cannot easily predict array type as we do not have a representative sample of the contained frame
                v = [None] * size
                ilocs = []
                for i, (loc, func) in enumerate(self._loc_to_func):
                    iloc = f.columns.loc_to_iloc(loc)
                    ilocs.append(iloc)
                    v[i] = func(f._extract(NULL_SLICE, iloc)) # type: ignore
                v, _ = iterable_to_array_1d(v, count=size)
                index = f.columns[iloc]
                yield Series(v, index=index, name=label)
        else:  # each component reduces to a column
            raise NotImplementedError()


#-------------------------------------------------------------------------------
class ReduceDispatch:
    __slots__ = (
        '_axis',
        '_items',
        '_yield_type',
        )

    def from_func(self,
            func: TUFunc,
            *,
            fill_value: tp.Any = np.nan,
            ) -> ReduceComponent:
        # TODO: add `retain_labels` config
        return ReduceComponent(self._items,
                func,
                yield_type=self._yield_type,
                axis=self._axis,
                )


class ReduceDispatchAligned(ReduceDispatch):
    '''Delegate interface for creating reductions from uniform collections of Frames.
    '''

    __slots__ = (
        '_axis_labels',
        )

    _INTERFACE: tp.Tuple[str, ...] = (
        'from_func',
        'from_map_func',
        'from_label_map',
        'from_label_pair_map',
        )

    def __init__(self,
            items: TIterableFrameItems,
            axis_labels: IndexBase, # always an index
            *,
            yield_type: IterNodeType,
            axis: int = 1,
            ) -> None:
        '''
        Args:
            axis_labels: Index on the axis used to label reductions.
        '''
        self._axis = axis
        self._items = items
        self._axis_labels = axis_labels
        self._yield_type = yield_type

    def from_map_func(self, func: TUFunc) -> ReduceAligned:
        '''
        For `Frame`, reduce by applying, for each column, a function that reduces to (0-dimensional) elements, where the column label and function are given as a mapping. Column labels are retained.
        '''
        iloc_to_func: TILocToFunc = list(zip(range(len(self._axis_labels)), repeat(func)))
        return ReduceAligned(self._items, iloc_to_func, self._axis_labels, axis=self._axis)

    def from_label_map(self,
            func_map: tp.Mapping[TLabel, TUFunc],
            ) -> ReduceAligned:
        '''
        For `Frame`, reduce by applying a function to each column, where the column label and function are given as a mapping. Column labels are retained.

        Args:
            func_map: a mapping of column labels to functions.
        '''
        loc_to_iloc = self._axis_labels.loc_to_iloc

        iloc_to_func: TILocToFunc = list(
                (loc_to_iloc(label), func)
                for label, func in func_map.items())
        return ReduceAligned(self._items, iloc_to_func, self._axis_labels, axis=self._axis)

    def from_label_pair_map(self,
            func_map: tp.Mapping[tp.Tuple[TLabel, TLabel], TUFunc],
            ) -> ReduceAligned:
        '''
        For `Frame`, reduce by applying a function to a column and assigning the result a new label. Functions are provided as values in a mapping, where the key is tuple of source label, destination label.

        Args:
            func_map: a mapping of pairs of source label, destination label, to a function.

        '''
        loc_to_iloc = self._axis_labels.loc_to_iloc

        iloc_to_func: TILocToFunc = []
        axis_labels = []
        for (iloc, label), func in func_map.items():
            axis_labels.append(label)
            iloc_to_func.append((loc_to_iloc(iloc), func))
        # NOTE: ignore self._axis_labels
        return ReduceAligned(self._items, iloc_to_func, axis_labels, axis=self._axis)


#-------------------------------------------------------------------------------
class ReduceDispatchUnaligned(ReduceDispatch):
    '''Delegate interface for creating reductions from uniform collections of Frames.
    '''
    _INTERFACE: tp.Tuple[str, ...] = (
        'from_func',
        'from_label_map',
        'from_label_pair_map',
        )

    def __init__(self,
            items: TIterableFrameItems,
            *,
            axis: int = 1,
            yield_type: IterNodeType,
            ) -> None:
        '''
        Args:
            axis_labels: Index on the axis used to label reductions.
        '''
        self._items = items
        self._axis = axis
        self._yield_type = yield_type

    def from_map_func(self, func: TUFunc) -> ReduceAligned:
        def func_derived(f: Frame) -> Series:
            return f.reduce.from_map_func(func).to_frame() # return a series

        return ReduceComponent(self._items,
                func_derived,
                yield_type=self._yield_type,
                axis=self._axis,
                )

    def from_label_map(self,
            func_map: tp.Mapping[TLabel, TUFunc],
            fill_value: tp.Any = np.nan,
            ) -> ReduceUnaligned:
        '''
        For `Frame`, reduce by applying a function to each column, where the column label and function are given as a mapping. Column labels are retained.

        Args:
            func_map: a mapping of column labels to functions.
        '''
        loc_to_func: TLabelToFunc = list(func_map.items())
        return ReduceUnaligned(self._items,
                loc_to_func,
                None,
                axis=self._axis,
                fill_value=fill_value,
                )

    def from_label_pair_map(self,
            func_map: tp.Mapping[tp.Tuple[TLabel, TLabel], TUFunc],
            fill_value: tp.Any = np.nan,
            ) -> ReduceUnaligned:
        '''
        For `Frame`, reduce by applying a function to a column and assigning the result a new label. Functions are provided as values in a mapping, where the key is tuple of source label, destination label.

        Args:
            func_map: a mapping of pairs of source label, destination label, to a function.

        '''
        loc_to_func: TLabelToFunc = []
        axis_labels = []
        for (loc, label), func in func_map.items():
            axis_labels.append(label)
            loc_to_func.append((loc, func))
        # NOTE: ignore self._axis_labels
        return ReduceUnaligned(self._items,
                loc_to_func,
                axis_labels,
                axis=self._axis,
                fill_value=fill_value,
                )

