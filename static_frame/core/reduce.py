from __future__ import annotations

import numpy as np
import typing_extensions as tp

from static_frame.core.frame import Frame
from static_frame.core.generic_aliases import TFrameAny
from static_frame.core.index_auto import TIndexAutoFactory
from static_frame.core.series import Series
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import TIndexCtorSpecifier
from static_frame.core.util import TIndexInitializer
from static_frame.core.util import TLabel
from static_frame.core.util import TName
from static_frame.core.util import TUFunc
from static_frame.core.util import iterable_to_array_1d
from static_frame.core.util import ufunc_dtype_to_dtype
from static_frame.core.index_base import IndexBase

TNDArrayAny = np.ndarray[tp.Any, tp.Any] #pragma: no cover
TFrameOrSeries = tp.Union[Frame, Series]
TFrameOrArray = tp.Union[Frame, TNDArrayAny]
TIterableFrameItems = tp.Iterable[tp.Tuple[TLabel, TFrameOrArray]]
TShape2D = tp.Tuple[int, int]

#-------------------------------------------------------------------------------

class Reduce:
    '''Utilities for Reducing pairs of label, uniform Frame to a new Frame.
    Axis 1 will reduce components into rows (labels are the index, ilocs refer to column positions); axis 0 will reduce components into columns (labels are the column labels, ilocs refer to index positions).
    '''

    __slots__ = (
        '_iloc_to_func',
        '_axis',
        '_items',
        '_axis_labels',
        )

    def __init__(self,
            items: TIterableFrameItems,
            iloc_to_func: tp.Sequence[tp.Tuple[int, TUFunc]],
            axis_labels: IndexBase,
            *,
            axis: int = 1,
            ):
        '''
        Args:
            axis_labels: Index on the axis used to label reductions.
        '''
        self._axis = axis
        self._items = items
        self._iloc_to_func = iloc_to_func
        self._axis_labels = axis_labels

    @classmethod
    def from_func_map(cls,
            items: TIterableFrameItems,
            func_map: tp.Mapping[int, TUFunc],
            axis_labels: IndexBase,
            *,
            axis: int = 1,
            ) -> tp.Self:
        '''
        Args:
            func_map: a mapping of iloc positions to functions, or iloc position to an iterable of functions.
        '''
        iloc_to_func: tp.List[tp.Tuple[int, TUFunc]] = list(func_map.items())
        return cls(items, iloc_to_func, axis_labels, axis=axis)

    @classmethod
    def from_src_dst_func_map(cls,
            items: TIterableFrameItems,
            func_map: tp.Mapping[tp.Tuple[int, TLabel], TUFunc],
            *,
            axis: int = 1,
            ) -> tp.Self:

        iloc_to_func: tp.List[tp.Tuple[int, TUFunc]] = []
        axis_labels = []
        for (iloc, label), func in func_map.items():
            axis_labels.append(label)
            iloc_to_func.append((iloc, func))

        return cls(items, iloc_to_func, axis_labels, axis=axis)

    @staticmethod
    def _prepare_items(
            axis: int,
            func_count: int,
            items: TIterableFrameItems,
            ) -> tp.Tuple[tp.Sequence[TLabel], tp.Sequence[TFrameOrArray], TShape2D]:

        labels: tp.List[TLabel] = []
        components: tp.List[TFrameOrArray] = []

        for label, component in items:
            labels.append(label)
            # NOTE: could assert uniformity of shape / labels here
            components.append(component)

        if axis == 1:
            shape = (len(labels), func_count)
        else:
            shape = (func_count, len(labels))
        return labels, components, shape

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

    def to_frame(self, *,
            index: tp.Optional[tp.Union[TIndexInitializer, TIndexAutoFactory]] = None,
            columns: tp.Optional[tp.Union[TIndexInitializer, TIndexAutoFactory]] = None,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            name: TName = None,
            consolidate_blocks: bool = False
        ) -> TFrameAny:

        labels, components, shape = self._prepare_items(
                self._axis,
                len(self._iloc_to_func),
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
