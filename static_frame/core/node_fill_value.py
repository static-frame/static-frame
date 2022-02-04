import typing as tp

import numpy as np

from static_frame.core.node_selector import Interface
from static_frame.core.util import OPERATORS
from static_frame.core.node_selector import InterfaceGetItem
from static_frame.core.util import GetItemKeyTypeCompound
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import KEY_MULTIPLE_TYPES

if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.frame import FrameGO  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index import Index  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchy  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.series import Series  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.type_blocks import TypeBlocks  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.node_transpose import InterfaceTranspose #pylint: disable = W0611 #pragma: no cover

TContainer = tp.TypeVar('TContainer',
        'Frame',
        'Series',
        )

class InterfaceFillValue(Interface[TContainer]):

    __slots__ = (
            '_container',
            '_fill_value',
            '_axis',
            )
    INTERFACE = (
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

    def _extract_loc(self, key: GetItemKeyTypeCompound) -> 'Frame':

        # assyume if a key is None it is an element selection
        if isinstance(key, tuple):
            loc_row_key, loc_column_key = key
        else:
            loc_row_key = key
            loc_column_key = NULL_SLICE

        # if a key is a slice?
        # will raise if out of bound slice
        loc_row_is_multiple = isinstance(loc_row_key, KEY_MULTIPLE_TYPES)
        if loc_row_key.__class__ is slice:
            loc_row_key = self._container._index._extract_loc(loc_row_key)

        loc_column_is_multiple = isinstance(loc_column_key, KEY_MULTIPLE_TYPES)
        if loc_column_key.__class__ is slice:
            loc_column_key = self._container._columns._extract_loc(loc_column_key)

        if loc_row_is_multiple and loc_column_is_multiple:
            # cannot reindex if loc keys are elements
            return self._container.reindex(index=loc_row_key,
                    columns=loc_column_key,
                    fill_value=self._fill_value,
                    )
        elif not loc_row_is_multiple and not loc_column_is_multiple:
            # selecting an element
            try:
                return self._container.loc[loc_row_key, loc_column_key]
            except KeyError:
                return self._fill_value
        elif not loc_row_is_multiple:
            # row is an element, return Series indexed by columns
            if loc_row_key in self._container._index:
                s = self._container.loc[loc_row_key]
                return s.reindex(loc_column_key, fill_value=self._fill_value)
            else:
                from static_frame.core.series import Series
                return Series.from_element(self._fill_value, index=loc_column_key, name=loc_row_key)

        else:
            # columns is an element, return Series indexed by index
            pass


        # import ipdb; ipdb.set_trace()


    @property
    def loc(self) -> InterfaceGetItem['Frame']:
        return InterfaceGetItem(self._extract_loc)

    # implement __getitem__, on Frame select columns, on Series selections items

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

