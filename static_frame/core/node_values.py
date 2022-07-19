
import typing as tp

import numpy as np

from static_frame.core.node_selector import Interface
from static_frame.core.node_selector import InterfaceBatch
# from static_frame.core.util import OPERATORS
from static_frame.core.util import UFunc
from static_frame.core.util import AnyCallable
from static_frame.core.util import DtypeSpecifier
from static_frame.core.util import blocks_to_array_2d
from static_frame.core.type_blocks import TypeBlocks

if tp.TYPE_CHECKING:
    from static_frame.core.batch import Batch  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.frame import Frame  #pylint: disable = W0611 #pragma: no cover
    # from static_frame.core.index import Index  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchy  #pylint: disable = W0611 #pragma: no cover
    # from static_frame.core.series import Series  #pylint: disable = W0611 #pragma: no cover
    # from static_frame.core.type_blocks import TypeBlocks  #pylint: disable = W0611 #pragma: no cover
    # from static_frame.core.node_fill_value import InterfaceFillValue #pylint: disable = W0611 #pragma: no cover
    # from static_frame.core.node_fill_value import InterfaceBatchFillValue #pylint: disable = W0611 #pragma: no cover


TContainer = tp.TypeVar('TContainer',
        'Frame',
        'IndexHierarchy',
        )

INTERFACE_VALUES = (
        'apply',
        )

class InterfaceValues(Interface[TContainer]):

    __slots__ = (
            '_container',
            )
    INTERFACE = INTERFACE_VALUES

    def __init__(self,
            container: TContainer,
            ) -> None:
        self._container: TContainer = container

    #---------------------------------------------------------------------------
    def apply(self,
            func: UFunc,
            *,
            consolidate_blocks: bool = False,
            unify_blocks: bool = False,
            dtype: DtypeSpecifier = None,
            ) -> 'Frame':
        '''
        Interface for applying functions direclty to underly NumPy arrays.

        Args:
            consolidate_blocks: Group adjacent same-typed arrays into 2D arrays.
            unify_blocks: Group all arrays into single array, re-typing to an appropriate dtype.
            dtype: specify a dtype to be used in conversion before consolidation or unification, and before function application.
        '''
        if self._container._NDIM == 2:
            blocks = self._container._blocks._blocks

            if unify_blocks:
                dtype = self._container._blocks._row_dtype if dtype is None else dtype
                tb = TypeBlocks.from_blocks(func(blocks_to_array_2d(
                        blocks=blocks,
                        shape=self._container.shape,
                        dtype=dtype,
                        )))
            elif consolidate_blocks:
                if dtype is not None:
                    blocks = (b.astype(dtype) for b in blocks)
                tb = TypeBlocks.from_blocks(
                        func(b) for b in TypeBlocks.consolidate_blocks(blocks)
                        )
            else:
                if dtype is not None:
                    blocks = (func(b.astype(dtype)) for b in blocks)
                else:
                    blocks = (func(b) for b in blocks)
                tb = TypeBlocks.from_blocks(blocks)

            return self._container.__class__(
                    tb,
                    index=self._container.index,
                    columns=self._container.columns,
                    name=self._container.name,
                    own_index=True,
                    own_data=True,
                    own_columns=self._container.STATIC,
                    )
        # TODO: handl series
        raise NotImplementedError()

    def __array_ufunc__(self,
            ufunc: UFunc,
            method: str,
            *args: tp.Any,
            **kwargs: tp.Any,
            ) -> np.ndarray:
        '''Support for applying NumPy functions directly on containers, returning NumPy arrays.
        '''
        if method == '__call__':
            return ufunc(
                *[(arg if arg is not self else arg._container.values) for arg in args],
                **kwargs,
                )
        return NotImplemented #pragma: no cover


class InterfaceBatchValues(InterfaceBatch):

    __slots__ = (
            '_batch_apply',
            )
    INTERFACE = INTERFACE_VALUES

    def __init__(self,
            batch_apply: tp.Callable[[AnyCallable], 'Batch'],
            ) -> None:
        self._batch_apply = batch_apply

    #---------------------------------------------------------------------------
    def apply(self,
            func: UFunc,
            *,
            consolidate_blocks: bool = False,
            unify_blocks: bool = False,
            dtype: DtypeSpecifier = None,
            ) -> 'Batch':
        '''
        Interface for using binary operators and methods with a pre-defined fill value.
        '''
        return self._batch_apply(lambda c: c.via_values.apply(func,
                consolidate_blocks=consolidate_blocks,
                unify_blocks=unify_blocks,
                dtype=dtype,
                ))

    def __array_ufunc__(self,
            ufunc: UFunc,
            method: str,
            *args: tp.Any,
            **kwargs: tp.Any,
            ) -> 'Batch':
        '''Support for applying NumPy functions directly on containers, returning NumPy arrays.
        '''
        if method == '__call__':
            def func(c):
                nonlocal args
                args = [(arg if arg is not self else c.values) for arg in args]
                return ufunc(
                        *args,
                        **kwargs,
                        )
            return self._batch_apply(func)

        return NotImplemented #pragma: no cover
