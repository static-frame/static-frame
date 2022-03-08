import typing as tp

import numpy as np

from static_frame.core.node_selector import Interface
from static_frame.core.util import OPERATORS
from static_frame.core.node_selector import InterfaceGetItem
from static_frame.core.node_selector import InterfaceBatch
from static_frame.core.util import GetItemKeyTypeCompound
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import KEY_MULTIPLE_TYPES
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import AnyCallable

if tp.TYPE_CHECKING:
    from static_frame.core.batch import Batch  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.frame import Frame  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.frame import FrameGO  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index_base import IndexBase  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index import Index  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchy  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.series import Series  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.type_blocks import TypeBlocks  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.node_transpose import InterfaceTranspose #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.node_transpose import InterfaceBatchTranspose #pylint: disable = W0611 #pragma: no cover

TContainer = tp.TypeVar('TContainer',
        'Frame',
        'Series',
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


class InterfaceFillValue(Interface[TContainer]):

    __slots__ = (
            '_container',
            '_fill_value',
            '_axis',
            )

    INTERFACE = INTERFACE_FILL_VALUE

    def __init__(self,
            container: TContainer,
            *,
            fill_value: object = np.nan,
            axis: int = 0,
            ) -> None:
        self._container: TContainer = container
        self._fill_value = fill_value
        self._axis = axis

    #---------------------------------------------------------------------------
    @property
    def via_T(self) -> "InterfaceTranspose[Frame]":
        '''
        Interface for using binary operators with one-dimensional sequences, where the opperand is applied column-wise.
        '''
        from static_frame.core.node_transpose import InterfaceTranspose
        from static_frame.core.frame import Frame
        assert isinstance(self._container, Frame)
        return InterfaceTranspose(
                container=self._container,
                fill_value=self._fill_value,
                )


    #---------------------------------------------------------------------------
    @staticmethod
    def _extract_key_attrs(
            key: GetItemKeyType,
            index: 'IndexBase',
            ) -> tp.Tuple[GetItemKeyType, bool, bool]:
        key_is_multiple = isinstance(key, KEY_MULTIPLE_TYPES)
        if key.__class__ is slice:
            key_is_null_slice = key == NULL_SLICE
            key = index._extract_loc(key) #type: ignore
        else:
            key_is_null_slice = False
        return key, key_is_multiple, key_is_null_slice

    def _extract_loc1d(self,
            key: GetItemKeyType = NULL_SLICE,
            ) -> 'Series':
        key, is_multiple, is_null_slice = self._extract_key_attrs(
                key,
                self._container._index,
                )
        fill_value = self._fill_value
        container = self._container

        if is_multiple:
            return container.reindex(key if not is_null_slice else None, #type: ignore
                    fill_value=fill_value,
                    )
        return container.get(key, fill_value) #type: ignore

    def _extract_loc2d(self,
            row_key: GetItemKeyType = NULL_SLICE,
            column_key: GetItemKeyType = NULL_SLICE,
            ) -> tp.Union['Frame', 'Series']:
        '''
        NOTE: keys are loc keys; None is interpreted as selector, not a NULL_SLICE
        '''
        from static_frame.core.series import Series
        fill_value = self._fill_value
        container = self._container # always a Frame

        row_key, row_is_multiple, row_is_null_slice = self._extract_key_attrs(
                row_key,
                container._index,
                )
        column_key, column_is_multiple, column_is_null_slice = self._extract_key_attrs(
                column_key,
                container._columns, #type: ignore
                )

        if row_is_multiple and column_is_multiple:
            # cannot reindex if loc keys are elements
            return container.reindex( # type: ignore
                    index=row_key if not row_is_null_slice else None,
                    columns=column_key if not column_is_null_slice else None,
                    fill_value=fill_value,
                    )
        elif not row_is_multiple and not column_is_multiple: # selecting an element
            try:
                return container.loc[row_key, column_key]
            except KeyError:
                return fill_value #type: ignore
        elif not row_is_multiple:
            # row is an element, return Series indexed by columns
            if row_key in container._index: #type: ignore
                s = container.loc[row_key]
                return s.reindex(column_key, fill_value=fill_value) #type: ignore
            return Series.from_element(fill_value,
                    index=column_key,
                    name=row_key,
                    )
        # columns is an element, return Series indexed by index
        if column_key in container._columns: #type: ignore
            s = container[column_key]
            return s.reindex(row_key, fill_value=fill_value) #type: ignore
        return Series.from_element(fill_value,
                index=row_key,
                name=column_key,
                )

    def _extract_loc2d_compound(self, key: GetItemKeyTypeCompound) -> tp.Union['Frame', 'Series']:
        if isinstance(key, tuple):
            row_key, column_key = key
        else:
            row_key = key
            column_key = NULL_SLICE
        return self._extract_loc2d(row_key, column_key)

    #---------------------------------------------------------------------------
    @property
    def loc(self) -> InterfaceGetItem['Frame']:
        '''Label-based selection where labels not specified will define a new container containing those labels filled with the fill value.
        '''
        if self._container._NDIM == 1:
            return InterfaceGetItem(self._extract_loc1d)
        return InterfaceGetItem(self._extract_loc2d_compound)

    def __getitem__(self,  key: GetItemKeyType) -> tp.Union['Frame', 'Series']:
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
class InterfaceFillValueGO(InterfaceFillValue[TContainer]): # only type is FrameGO

    __slots__ = InterfaceFillValue.__slots__
    INTERFACE = InterfaceFillValue.INTERFACE + ( #type: ignore
            '__setitem__',
            )

    def __setitem__(self,
            key: tp.Hashable,
            value: tp.Any,
            ) -> None:
        self._container.__setitem__(key, value, self._fill_value) #type: ignore

#---------------------------------------------------------------------------


class InterfaceBatchFillValue(InterfaceBatch):
    '''Alternate string interface specialized for the :obj:`Batch`.
    '''
    INTERFACE = INTERFACE_FILL_VALUE

    __slots__ = (
            '_batch_apply',
            '_fill_value',
            '_axis',
            )

    def __init__(self,
            batch_apply: tp.Callable[[AnyCallable], 'Batch'],
            fill_value: object = np.nan,
            axis: int = 0,
            ) -> None:

        self._batch_apply = batch_apply
        self._fill_value = fill_value
        self._axis = axis


    #---------------------------------------------------------------------------
    @property
    def via_T(self) -> "InterfaceBatchTranspose":
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
    def loc(self) -> InterfaceGetItem['Frame']:
        '''Label-based selection where labels not specified will define a new container containing those labels filled with the fill value.
        '''
        def func(key: GetItemKeyType) -> 'Batch':
            return self._batch_apply(
                    lambda c: InterfaceFillValue(c,
                            fill_value=self._fill_value,
                            axis=self._axis).loc[key]
                    )
        return InterfaceGetItem(func)

    def __getitem__(self,  key: GetItemKeyType) -> 'Batch':
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
