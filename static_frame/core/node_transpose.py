from __future__ import annotations

import numpy as np
import typing_extensions as tp

from static_frame.core.node_selector import Interface, InterfaceBatch
from static_frame.core.util import OPERATORS, TCallableAny

if tp.TYPE_CHECKING:
    from static_frame.core.batch import Batch  # pragma: no cover
    from static_frame.core.frame import Frame  # pragma: no cover
    from static_frame.core.index import Index  # #pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchy  # #pragma: no cover
    from static_frame.core.node_fill_value import (
        InterfaceBatchFillValue,
        InterfaceFillValue,  # pragma: no cover
    )  # pragma: no cover
    from static_frame.core.series import Series  # #pragma: no cover
    from static_frame.core.type_blocks import TypeBlocks  # #pragma: no cover

    TFrameAny = Frame[
        tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]
    ]  # pragma: no cover

TVContainer_co = tp.TypeVar(
    'TVContainer_co',
    'Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]]',
    'IndexHierarchy',
    covariant=True,
)

INTERFACE_TRANSPOSE = (
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


class InterfaceTranspose(Interface, tp.Generic[TVContainer_co]):
    __slots__ = (
        '_container',
        '_fill_value',
    )
    _INTERFACE = INTERFACE_TRANSPOSE

    def __init__(
        self,
        container: TVContainer_co,
        *,
        fill_value: object = np.nan,
    ) -> None:
        self._container: TVContainer_co = container
        self._fill_value = fill_value

    # ---------------------------------------------------------------------------
    def via_fill_value(
        self,
        fill_value: object,
        /,
    ) -> InterfaceFillValue[TFrameAny]:
        """
        Interface for using binary operators and methods with a pre-defined fill value.
        """
        from static_frame.core.frame import Frame
        from static_frame.core.node_fill_value import InterfaceFillValue

        assert isinstance(self._container, Frame)
        return InterfaceFillValue(
            container=self._container,
            fill_value=fill_value,
            axis=1,
        )

    # ---------------------------------------------------------------------------
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

    # ---------------------------------------------------------------------------
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


class InterfaceBatchTranspose(InterfaceBatch):
    __slots__ = (
        '_batch_apply',
        '_fill_value',
    )
    _INTERFACE = INTERFACE_TRANSPOSE

    def __init__(
        self,
        batch_apply: tp.Callable[[TCallableAny], 'Batch'],
        fill_value: object = np.nan,
    ) -> None:
        self._batch_apply = batch_apply
        self._fill_value = fill_value

    # ---------------------------------------------------------------------------
    def via_fill_value(
        self,
        fill_value: object,
        /,
    ) -> 'InterfaceBatchFillValue':
        """
        Interface for using binary operators and methods with a pre-defined fill value.
        """
        from static_frame.core.node_fill_value import InterfaceBatchFillValue

        return InterfaceBatchFillValue(
            batch_apply=self._batch_apply,
            fill_value=fill_value,
            axis=1,
        )

    # ---------------------------------------------------------------------------
    def __add__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__add__(other)
        )

    def __sub__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__sub__(other)
        )

    def __mul__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__mul__(other)
        )

    # def __matmul__(self, other: tp.Any) -> 'Batch':
    # pass

    def __truediv__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__truediv__(
                other
            )
        )

    def __floordiv__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__floordiv__(
                other
            )
        )

    def __mod__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__mod__(other)
        )

    def __pow__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__pow__(other)
        )

    def __lshift__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__lshift__(other)
        )

    def __rshift__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__rshift__(other)
        )

    def __and__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__and__(other)
        )

    def __xor__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__xor__(other)
        )

    def __or__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__or__(other)
        )

    def __lt__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__lt__(other)
        )

    def __le__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__le__(other)
        )

    def __eq__(self, other: tp.Any) -> 'Batch':  # type: ignore
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__eq__(other)
        )

    def __ne__(self, other: tp.Any) -> 'Batch':  # type: ignore
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__ne__(other)
        )

    def __gt__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__gt__(other)
        )

    def __ge__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__ge__(other)
        )

    # ---------------------------------------------------------------------------
    def __radd__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__radd__(other)
        )

    def __rsub__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__rsub__(other)
        )

    def __rmul__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__rmul__(other)
        )

    # def __rmatmul__(self, other: tp.Any) -> 'Batch':
    # pass

    def __rtruediv__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__rtruediv__(
                other
            )
        )

    def __rfloordiv__(self, other: tp.Any) -> 'Batch':
        return self._batch_apply(
            lambda c: InterfaceTranspose(c, fill_value=self._fill_value).__rfloordiv__(
                other
            )
        )
