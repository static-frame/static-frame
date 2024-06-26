from __future__ import annotations

import numpy as np
import typing_extensions as tp

from static_frame.core.frame import Frame
from static_frame.core.series import Series
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import TLabel
from static_frame.core.util import TUFunc
from static_frame.core.util import iterable_to_array_1d
from static_frame.core.util import ufunc_dtype_to_dtype

TFrameOrSeries = tp.Union[Frame, Series]
TIteratorFrameItems = tp.Iterator[tp.Tuple[TLabel, TFrameOrSeries]]

#-------------------------------------------------------------------------------
class ReduceArrays:
    '''Utilities for Reducing pairs of label, uniform Frame to a new Frame.
    Axis 1 will reduce components into rows (labels are the index, ilocs refer to column positions); axis 0 will reduce components into columns (labels are the column labels, ilocs refer to index positions).
    '''
    __slots__ = (
        '_labels',
        '_arrays',
        '_iloc_to_func',
        '_axis',
        '_shape',
        '_dtype',
        )

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

    def __init__(self,
            items: TIteratorFrameItems,
            iloc_to_funcs: tp.Mapping[int, tp.Iterable[TUFunc]],
            *,
            axis: int = 1,
            ):
        self._axis = axis

        # this is an eager evaluation; this might all be deferred
        self._labels = []
        self._arrays = []
        self._dtype = None

        for label, array in items:
            self._labels.append(label)
            # NOTE: could assert uniformity of shape / labels here
            if self._dtype is None:
                self._dtype = array.dtype # take from first
            self._arrays.append(array)

        # flatten iloc_to_funcs as we will do a single pass and need the length
        iloc_to_func = []
        for iloc, funcs in iloc_to_funcs.items():
            for func in funcs:
                iloc_to_func.append((iloc, func))
        self._iloc_to_func = iloc_to_func

        if axis == 1:
            self._shape = (len(self._labels), len(self._iloc_to_func))
        else:
            self._shape = (len(self._iloc_to_func), len(self._labels))

    def to_frame(self, *,
            index: tp.Optional[tp.Union[TIndexInitializer, TIndexAutoFactory]] = None,
            columns: tp.Optional[tp.Union[TIndexInitializer, TIndexAutoFactory]] = None,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            name: TName = None,
            consolidate_blocks: bool = False
        ) -> TFrameAny:

        if self._axis == 1:
            # each component reduces to a row
            blocks = [] # pre allocate arrays, or empty lists if necessary
            size = self._shape[0]
            for iloc, func in self._iloc_to_func:
                post_dt = ufunc_dtype_to_dtype(func, self._dtype)
                if post_dt is not None:
                    v = np.empty(size, dtype=post_dt)
                else:
                    v = [None] * size
                for i, array in enumerate(self._arrays):
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
                index = self._labels

        return Frame(tb,
                index=index,
                columns=columns,
                name=name,
                own_data=True,
                index_constructor=index_constructor,
                columns_constructor=columns_constructor,
                )



Reduce = ReduceArrays