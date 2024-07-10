from __future__ import annotations

import numpy as np
import typing_extensions as tp

from static_frame.core.frame import Frame
from static_frame.core.series import Series
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import TLabel
from static_frame.core.util import TShape
from static_frame.core.util import TUFunc
from static_frame.core.util import iterable_to_array_1d
from static_frame.core.util import ufunc_dtype_to_dtype

TFrameOrSeries = tp.Union[Frame, Series]
TIteratorFrameItems = tp.Iterator[tp.Tuple[TLabel, TFrameOrSeries]]

#-------------------------------------------------------------------------------

class Reduce:

    __slots__ = (
        '_iloc_to_func',
        '_axis',
        '_items',
        )

    def __init__(self,
            items: TIteratorFrameItems,
            iloc_to_funcs: tp.Mapping[int, tp.Iterable[TUFunc]],
            *,
            axis: int = 1,
            ):
        self._axis = axis
        self._items = items
        iloc_to_func = []
        for iloc, funcs in iloc_to_funcs.items():
            for func in funcs:
                iloc_to_func.append((iloc, func))
        self._iloc_to_func = iloc_to_func

    @classmethod
    def from_func_map(cls,
            items: TIteratorFrameItems,
            func_map: tp.Mapping[int, tp.Union[TUFunc, tp.Iterable[TUFunc]]],
            *,
            axis: int = 1,
            ):
        iloc_to_funcs = {}
        for iloc, func in func_map.items():
            if callable(func):
                iloc_to_funcs[iloc] = (func,)
            else:
                iloc_to_funcs[iloc] = func # assume an iterable?

        return cls(items, iloc_to_funcs)

    @staticmethod
    def _prepare_items(
            axis: int,
            func_count: int,
            items: TIteratorFrameItems,
            ) -> tp.Tuple[tp.Sequence[TLabel], TFrameOrSeries, TShape]:
        # this is an eager evaluation; this might all be deferred
        labels = []
        components = []

        # flatten iloc_to_funcs as we will do a single pass and need the length

        for label, component in items:
            labels.append(label)
            # NOTE: could assert uniformity of shape / labels here
            components.append(component)

        if axis == 1:
            shape = (len(labels), func_count)
        else:
            shape = (func_count, len(labels))
        return (labels, components, shape)

class ReduceArrays(Reduce):
    '''Utilities for Reducing pairs of label, uniform Frame to a new Frame.
    Axis 1 will reduce components into rows (labels are the index, ilocs refer to column positions); axis 0 will reduce components into columns (labels are the column labels, ilocs refer to index positions).
    '''

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
            dtype = components[0].dtype
        else:
            dtype = None

        if self._axis == 1:
            # each component reduces to a row
            blocks = [] # pre allocate arrays, or empty lists if necessary
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
                v.flags.writeable = False
                blocks.append(v)
        else:
            # each component reduces to a column
            pass

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
                name=name,
                own_data=True,
                index_constructor=index_constructor,
                columns_constructor=columns_constructor,
                )



Reduce = ReduceArrays