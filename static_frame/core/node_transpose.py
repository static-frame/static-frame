
import typing as tp

import numpy as np

from static_frame.core.node_selector import Interface
from static_frame.core.util import OPERATORS


if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index import Index  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchy  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.series import Series  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.type_blocks import TypeBlocks  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.node_fill_value import InterfaceFillValue #pylint: disable = W0611 #pragma: no cov

TContainer = tp.TypeVar('TContainer',
        'Frame',
        'IndexHierarchy',
        )

class InterfaceTranspose(Interface[TContainer]):


    __slots__ = (
            '_container',
            '_fill_value',
            )
    INTERFACE = (
            'via_fill_value',
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
            ) -> None:
        self._container: TContainer = container
        self._fill_value = fill_value


    #---------------------------------------------------------------------------
    def via_fill_value(self,
            fill_value: object,
            ) -> "InterfaceFillValue[Frame]":
        '''
        Interface for using binary operators and methods with a pre-defined fill value.
        '''
        from static_frame.core.node_fill_value import InterfaceFillValue
        from static_frame.core.frame import Frame
        assert isinstance(self._container, Frame)
        return InterfaceFillValue(
                container=self._container,
                fill_value=fill_value,
                axis=1,
                )

    #---------------------------------------------------------------------------
    def __add__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__add__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

    def __sub__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__sub__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

    def __mul__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__mul__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

    # def __matmul__(self, other: tp.Any) -> tp.Any:
    #     return self._container._ufunc_binary_operator(
    #             operator=OPERATORS['__matmul__'],
    #             other=other,
    #             axis=1,
    #             )

    def __truediv__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__truediv__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

    def __floordiv__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__floordiv__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

    def __mod__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__mod__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

    def __pow__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__pow__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

    def __lshift__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__lshift__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

    def __rshift__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__rshift__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

    def __and__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__and__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

    def __xor__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__xor__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

    def __or__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__or__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

    def __lt__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__lt__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

    def __le__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__le__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

    def __eq__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__eq__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

    def __ne__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__ne__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

    def __gt__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__gt__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

    def __ge__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__ge__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

    #---------------------------------------------------------------------------
    def __radd__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__radd__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

    def __rsub__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__rsub__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

    def __rmul__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__rmul__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

    # def __rmatmul__(self, other: tp.Any) -> tp.Any:
    #     return self._container._ufunc_binary_operator(
    #             operator=OPERATORS['__rmatmul__'],
    #             other=other,
    #             axis=1,
    #             )

    def __rtruediv__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__rtruediv__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

    def __rfloordiv__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=OPERATORS['__rfloordiv__'],
                other=other,
                axis=1,
                fill_value=self._fill_value,
                )

