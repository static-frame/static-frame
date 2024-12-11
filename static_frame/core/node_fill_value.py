from __future__ import annotations

import numpy as np
import typing_extensions as tp

from static_frame.core.node_selector import Interface
from static_frame.core.node_selector import InterfaceBatch
from static_frame.core.node_selector import InterGetItemLocReduces
from static_frame.core.util import KEY_MULTIPLE_TYPES
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import OPERATORS
from static_frame.core.util import TCallableAny
from static_frame.core.util import TLabel
from static_frame.core.util import TLocSelector
from static_frame.core.util import TLocSelectorCompound

if tp.TYPE_CHECKING:
    from static_frame.core.batch import Batch  # pragma: no cover
    from static_frame.core.frame import Frame  # pragma: no cover
    from static_frame.core.hloc import HLoc  # pragma: no cover
    from static_frame.core.index import ILoc  # pragma: no cover
    from static_frame.core.index_base import IndexBase  # pragma: no cover
    from static_frame.core.node_selector import TFrameOrSeries  # pragma: no cover
    from static_frame.core.node_transpose import InterfaceBatchTranspose  # pragma: no cover
    from static_frame.core.node_transpose import InterfaceTranspose  # pragma: no cover
    from static_frame.core.series import Series  # pragma: no cover

    TSeriesAny = Series[tp.Any, tp.Any] #pragma: no cover
    TFrameAny = Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]] #pragma: no cover

TVContainer_co = tp.TypeVar('TVContainer_co',
        'Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]]',
        'Series[tp.Any, tp.Any]',
        covariant=True,
        )
INTERFACE_FILL_VALUE = (
        'loc',
        '__getitem__',
        'via_T',
        '__add__',
        '__sub__',
        '__mul__',
        # '__matmul__',
        '__truediv__',
        '__floordiv__',
        '__mod__',
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
        '__radd__',
        '__rsub__',
        '__rmul__',
        # '__rmatmul__',
        '__rtruediv__',
        '__rfloordiv__',
        )


class InterfaceFillValue(Interface, tp.Generic[TVContainer_co]):

    __slots__ = (
            '_container',
            '_fill_value',
            '_axis',
            )

    _INTERFACE = INTERFACE_FILL_VALUE

    def __init__(self,
            container: TVContainer_co,
            *,
            fill_value: tp.Any = np.nan,
            axis: int = 0,
            ) -> None:
        self._container: TVContainer_co = container
        self._fill_value = fill_value
        self._axis = axis

    #---------------------------------------------------------------------------
    @property
    def via_T(self) -> InterfaceTranspose[TFrameAny]:
        '''
        Interface for using binary operators with one-dimensional sequences, where the opperand is applied column-wise.
        '''
        from static_frame.core.frame import Frame
        from static_frame.core.node_transpose import InterfaceTranspose
        if not isinstance(self._container, Frame):
            raise NotImplementedError('via_T functionality only available on Frame')
        return InterfaceTranspose(
                container=self._container,
                fill_value=self._fill_value,
                )

    #---------------------------------------------------------------------------
    @staticmethod
    def _extract_key_attrs(
            key: TLocSelector | ILoc | HLoc,
            index: 'IndexBase',
            ) -> tp.Tuple[TLocSelector, bool, bool]:
        '''Given a loc-style key into the supplied `index`, return a selection (labels from that index) as well as Boolean attributes
        '''
        from static_frame.core.container_util import key_from_container_key
        from static_frame.core.hloc import HLoc

        key_is_null_slice: bool
        key_is_multiple: bool

        if key.__class__ is HLoc:
            labels = index._extract_loc(key) #type: ignore
            key_is_multiple = any(isinstance(k, KEY_MULTIPLE_TYPES) for k in key) #type: ignore
            key_is_null_slice = False
        elif key.__class__ is slice:
            labels = index._extract_loc(key) #type: ignore
            key_is_multiple = True
            key_is_null_slice = key == NULL_SLICE #type: ignore
        else:
            labels = key_from_container_key(index, key, expand_iloc=True)
            key_is_multiple = isinstance(labels, KEY_MULTIPLE_TYPES)
            key_is_null_slice = False

        return labels, key_is_multiple, key_is_null_slice

    def _extract_loc1d(self,
            key: TLocSelector = NULL_SLICE,
            ) -> TSeriesAny:
        '''This is only called if container is 1D
        '''
        from static_frame.core.container_util import get_col_fill_value_factory
        from static_frame.core.container_util import index_from_index

        labels, is_multiple, is_null_slice = self._extract_key_attrs(
                key,
                self._container._index,
                )
        fill_value = self._fill_value
        container: Series = self._container # type: ignore [assignment]

        if is_multiple:
            index = index_from_index(labels, container.index)
            return container.reindex(index, fill_value=fill_value, own_index=True)

        # if a single value, return it or the fill value
        fv = get_col_fill_value_factory(fill_value, None)(0, container.dtype)
        return container.get(labels, fv) #type: ignore

    def _extract_loc2d(self,
            row_key: TLocSelector = NULL_SLICE,
            column_key: TLocSelector = NULL_SLICE,
            ) -> tp.Union[TFrameAny, TSeriesAny]:
        '''
        NOTE: keys are loc keys; None is interpreted as selector, not a NULL_SLICE
        '''
        from static_frame.core.container_util import get_col_fill_value_factory
        from static_frame.core.container_util import index_from_index
        from static_frame.core.series import Series

        fill_value = self._fill_value
        container: Frame = self._container # type: ignore [assignment]

        row_labels, row_is_multiple, row_is_null_slice = self._extract_key_attrs(
                row_key,
                container._index,
                )
        column_labels, column_is_multiple, column_is_null_slice = self._extract_key_attrs(
                column_key,
                container._columns,
                )

        if row_is_multiple and column_is_multiple:
            # cannot reindex if loc keys are elements
            index = index_from_index(row_labels, container.index) if not row_is_null_slice else None
            columns = index_from_index(column_labels, container.columns) if not column_is_null_slice else None
            return container.reindex(
                    index=index,
                    columns=columns,
                    fill_value=fill_value,
                    own_index=index is not None,
                    own_columns=columns is not None,
                    )
        elif not row_is_multiple and not column_is_multiple: # selecting an element
            try:
                return container.loc[row_labels, column_labels]
            except KeyError:
                fv = get_col_fill_value_factory(fill_value, None)(0, None)
                return fv #type: ignore
        elif not row_is_multiple:
            # row is an element, return Series indexed by columns
            index = index_from_index(column_labels, container.columns)
            if row_labels in container._index: #type: ignore
                # NOTE: as row_labels might be a tuple, force second argument
                s = container.loc[row_labels, NULL_SLICE]
                return s.reindex(index, fill_value=fill_value, own_index=True)

            fv = get_col_fill_value_factory(fill_value, None)(0, None)
            return Series.from_element(fv,
                    index=index,
                    name=row_labels, # type: ignore
                    own_index=True,
                    )
        # columns is an element, return Series indexed by index
        if column_labels in container._columns: #type: ignore
            index = index_from_index(row_labels, container.index)
            s = container[column_labels] #type: ignore
            return s.reindex(index, fill_value=fill_value, own_index=True)

        index = index_from_index(row_labels, container.index)
        fv = get_col_fill_value_factory(fill_value, None)(0, None)
        return Series.from_element(fv,
                index=index,
                name=column_labels, # type: ignore
                own_index=True,
                )

    def _extract_loc2d_compound(self, key: TLocSelectorCompound) -> TFrameOrSeries:
        if isinstance(key, tuple):
            row_key, column_key = key # pyright: ignore
        else:
            row_key = key
            column_key = NULL_SLICE
        return self._extract_loc2d(row_key, column_key)

    #---------------------------------------------------------------------------
    @property
    def loc(self) -> InterGetItemLocReduces[TFrameOrSeries, tp.Any]:
        '''Label-based selection where labels not specified will define a new container containing those labels filled with the fill value.
        '''
        if self._container._NDIM == 1:
            return InterGetItemLocReduces(self._extract_loc1d)
        return InterGetItemLocReduces(self._extract_loc2d_compound)

    def __getitem__(self,  key: TLocSelector) -> TFrameOrSeries:
        '''Label-based selection where labels not specified will define a new container containing those labels filled with the fill value.
        '''
        if self._container._NDIM == 1:
            return self._extract_loc1d(key)
        return self._extract_loc2d(NULL_SLICE, key)

    #---------------------------------------------------------------------------
    def __add__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__add__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

    def __sub__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__sub__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

    def __mul__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__mul__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

    # def __matmul__(self, other: tp.Any) -> tp.Any:
    #     return self._container._ufunc_binary_operator(
    #             operator=OPERATORS['__matmul__'],
    #             other=other,
    #             axis=self._axis,
    #             fill_value=self._fill_value,
    #             )

    def __truediv__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__truediv__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

    def __floordiv__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__floordiv__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

    def __mod__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__mod__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

    def __pow__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__pow__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

    def __lshift__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__lshift__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

    def __rshift__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__rshift__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

    def __and__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__and__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

    def __xor__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__xor__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

    def __or__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__or__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

    def __lt__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__lt__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

    def __le__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__le__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

    def __eq__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__eq__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

    def __ne__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__ne__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

    def __gt__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__gt__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

    def __ge__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__ge__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

    #---------------------------------------------------------------------------
    def __radd__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__radd__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

    def __rsub__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__rsub__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

    def __rmul__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__rmul__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

    # def __rmatmul__(self, other: tp.Any) -> tp.Any:
    #     return self._container._ufunc_binary_operator(
    #             operator=OPERATORS['__rmatmul__'],
    #             other=other,
    #             axis=self._axis,
    #             fill_value=self._fill_value,
    #             )

    def __rtruediv__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__rtruediv__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

    def __rfloordiv__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__rfloordiv__'],
                other=other,
                axis=self._axis,
                fill_value=self._fill_value,
                )

#---------------------------------------------------------------------------
class InterfaceFillValueGO(InterfaceFillValue[TVContainer_co]): # only type is FrameGO

    __slots__ = ()
    _INTERFACE = InterfaceFillValue._INTERFACE + ( #type: ignore
            '__setitem__',
            )

    def __setitem__(self,
            key: TLabel,
            value: tp.Any,
            ) -> None:
        self._container.__setitem__(key, value, self._fill_value) #type: ignore

#---------------------------------------------------------------------------


class InterfaceBatchFillValue(InterfaceBatch):
    '''Alternate string interface specialized for the :obj:`Batch`.
    '''
    _INTERFACE = INTERFACE_FILL_VALUE

    __slots__ = (
            '_batch_apply',
            '_fill_value',
            '_axis',
            )

    def __init__(self,
            batch_apply: tp.Callable[[TCallableAny], 'Batch'],
            fill_value: object = np.nan,
            axis: int = 0,
            ) -> None:

        self._batch_apply = batch_apply
        self._fill_value = fill_value
        self._axis = axis


    #---------------------------------------------------------------------------
    @property
    def via_T(self) -> 'InterfaceBatchTranspose':
        '''
        Interface for using binary operators with one-dimensional sequences, where the opperand is applied column-wise.
        '''
        from static_frame.core.node_transpose import InterfaceBatchTranspose
        return InterfaceBatchTranspose(
                batch_apply=self._batch_apply,
                fill_value=self._fill_value,
                )

    #---------------------------------------------------------------------------
    @property
    def loc(self) -> InterGetItemLocReduces[Batch, tp.Any]:
        '''Label-based selection where labels not specified will define a new container containing those labels filled with the fill value.
        '''
        def func(key: TLocSelector) -> 'Batch':
            return self._batch_apply(
                    lambda c: InterfaceFillValue(c,
                            fill_value=self._fill_value,
                            axis=self._axis).loc[key]
                    )
        return InterGetItemLocReduces(func)

    def __getitem__(self,  key: TLocSelector) -> 'Batch':
        '''Label-based selection where labels not specified will define a new container containing those labels filled with the fill value.
        '''
        return self._batch_apply(
                lambda c: InterfaceFillValue(c,
                        fill_value=self._fill_value,
                        axis=self._axis).__getitem__(key)
                )

    #---------------------------------------------------------------------------
    def __add__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__add__(other)
            )

    def __sub__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__sub__(other)
            )

    def __mul__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__mul__(other)
            )

    # def __matmul__(self, other: tp.Any) -> 'Batch':

    def __truediv__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__truediv__(other)
            )

    def __floordiv__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__floordiv__(other)
            )

    def __mod__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__mod__(other)
            )

    def __pow__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__pow__(other)
            )

    def __lshift__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__lshift__(other)
            )

    def __rshift__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__rshift__(other)
            )

    def __and__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__and__(other)
            )

    def __xor__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__xor__(other)
            )

    def __or__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__or__(other)
            )

    def __lt__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__lt__(other)
            )

    def __le__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__le__(other)
            )

    def __eq__(self, other: tp.Any) -> 'Batch': #type: ignore
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__eq__(other)
            )

    def __ne__(self, other: tp.Any) -> 'Batch': #type: ignore
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__ne__(other)
            )

    def __gt__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__gt__(other)
            )

    def __ge__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__ge__(other)
            )

    #---------------------------------------------------------------------------
    def __radd__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__radd__(other)
            )

    def __rsub__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__rsub__(other)
            )

    def __rmul__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__rmul__(other)
            )

    # def __rmatmul__(self, other: tp.Any) -> 'Batch':

    def __rtruediv__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__rtruediv__(other)
            )

    def __rfloordiv__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceFillValue(c,
                    fill_value=self._fill_value,
                    axis=self._axis).__rfloordiv__(other)
            )
