import operator as operator_mod

import typing as tp
# import numpy as np

from static_frame.core.node_selector import Interface


if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index import Index  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchy  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.series import Series  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.type_blocks import TypeBlocks  #pylint: disable = W0611 #pragma: no cover

TContainer = tp.TypeVar('TContainer',
        'Frame',
        'IndexHierarchy',
        )

class InterfaceTranspose(Interface[TContainer]):


    __slots__ = (
            '_container',
            )
    INTERFACE = (
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
            ) -> None:
        self._container: TContainer = container

    #---------------------------------------------------------------------------
    def __add__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=operator_mod.__add__,
                other=other,
                axis=1,
                )

    def __sub__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=operator_mod.__sub__,
                other=other,
                axis=1,
                )

    def __mul__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=operator_mod.__mul__,
                other=other,
                axis=1,
                )

    # def __matmul__(self, other: tp.Any) -> tp.Any:
    #     return self._container._ufunc_binary_operator(
    #             operator=operator_mod.__matmul__,
    #             other=other,
    #             axis=1,
    #             )

    def __truediv__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=operator_mod.__truediv__,
                other=other,
                axis=1,
                )

    def __floordiv__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=operator_mod.__floordiv__,
                other=other,
                axis=1,
                )

    def __mod__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=operator_mod.__mod__,
                other=other,
                axis=1,
                )

    def __pow__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=operator_mod.__pow__,
                other=other,
                axis=1,
                )

    def __lshift__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=operator_mod.__lshift__,
                other=other,
                axis=1,
                )

    def __rshift__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=operator_mod.__rshift__,
                other=other,
                axis=1,
                )

    def __and__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=operator_mod.__and__,
                other=other,
                axis=1,
                )

    def __xor__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=operator_mod.__xor__,
                other=other,
                axis=1,
                )

    def __or__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=operator_mod.__or__,
                other=other,
                axis=1,
                )

    def __lt__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=operator_mod.__lt__,
                other=other,
                axis=1,
                )

    def __le__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=operator_mod.__le__,
                other=other,
                axis=1,
                )

    def __eq__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=operator_mod.__eq__,
                other=other,
                axis=1,
                )

    def __ne__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=operator_mod.__ne__,
                other=other,
                axis=1,
                )

    def __gt__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=operator_mod.__gt__,
                other=other,
                axis=1,
                )

    def __ge__(self, other: tp.Any) -> tp.Any:
        return self._container._ufunc_binary_operator(
                operator=operator_mod.__ge__,
                other=other,
                axis=1,
                )

    #---------------------------------------------------------------------------
    def __radd__(self, other: tp.Any) -> tp.Any:
        operator = lambda rhs, lhs: operator_mod.__add__(lhs, rhs)
        operator.__name__ = 'r' + operator_mod.__add__.__name__
        return self._container._ufunc_binary_operator(
                operator=operator,
                other=other,
                axis=1,
                )

    def __rsub__(self, other: tp.Any) -> tp.Any:
        operator = lambda rhs, lhs: operator_mod.__sub__(lhs, rhs)
        operator.__name__ = 'r' + operator_mod.__sub__.__name__
        return self._container._ufunc_binary_operator(
                operator=operator,
                other=other,
                axis=1,
                )

    def __rmul__(self, other: tp.Any) -> tp.Any:
        operator = lambda rhs, lhs: operator_mod.__mul__(lhs, rhs)
        operator.__name__ = 'r' + operator_mod.__mul__.__name__
        return self._container._ufunc_binary_operator(
                operator=operator,
                other=other,
                axis=1,
                )

    # def __rmatmul__(self, other: tp.Any) -> tp.Any:
    #     operator = lambda rhs, lhs: operator_mod.__matmul__(lhs, rhs)
    #     operator.__name__ = 'r' + operator_mod.__matmul__.__name__
    #     return self._container._ufunc_binary_operator(
    #             operator=operator,
    #             other=other,
    #             axis=1,
    #             )

    def __rtruediv__(self, other: tp.Any) -> tp.Any:
        operator = lambda rhs, lhs: operator_mod.__truediv__(lhs, rhs)
        operator.__name__ = 'r' + operator_mod.__truediv__.__name__
        return self._container._ufunc_binary_operator(
                operator=operator,
                other=other,
                axis=1,
                )

    def __rfloordiv__(self, other: tp.Any) -> tp.Any:
        operator = lambda rhs, lhs: operator_mod.__floordiv__(lhs, rhs)
        operator.__name__ = 'r' + operator_mod.__floordiv__.__name__
        return self._container._ufunc_binary_operator(
                operator=operator,
                other=other,
                axis=1,
                )
