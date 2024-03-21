from __future__ import annotations

import numpy as np
import typing_extensions as tp
from arraykit import column_2d_filter

from static_frame.core.node_selector import Interface
from static_frame.core.node_selector import InterfaceBatch
from static_frame.core.type_blocks import TypeBlocks
from static_frame.core.util import TCallableAny
from static_frame.core.util import TUFunc
from static_frame.core.util import blocks_to_array_2d

if tp.TYPE_CHECKING:
    from static_frame.core.batch import Batch  # pragma: no cover
    from static_frame.core.frame import Frame  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.index import Index  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchy  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.series import Series  # pylint: disable=W0611 #pragma: no cover
    TNDArrayAny = np.ndarray[tp.Any, tp.Any] #pragma: no cover
    TDtypeAny = np.dtype[tp.Any] #pragma: no cover

TVContainer_co = tp.TypeVar('TVContainer_co',
        'Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]]',
        'IndexHierarchy',
        'Series[tp.Any, tp.Any]',
        'Index[tp.Any]',
        covariant=True,
        )

INTERFACE_VALUES = (
        'apply',
        '__array_ufunc__',
        '__call__',
        )

VALID_UFUNC_ARRAY_METHODS = frozenset(('__call__',))


class InterfaceValues(Interface, tp.Generic[TVContainer_co]):
    '''
    If a user wants to call a ufunc and get back an array of variable dimensionality, they have to call that ufunc on one consolidated array via .values; any attempt at block-level manipulation will have to, under some scenarios, figure out how to combine the per-block results (and an appropriate type) into an array. This is undesirable. Instead, all applications of this interface must use UFuncs that retain dimensionality.
    '''
    __slots__ = (
            '_container',
            '_consolidate_blocks',
            '_unify_blocks',
            '_dtype',
            )
    _INTERFACE = INTERFACE_VALUES

    def __init__(self,
            container: TVContainer_co,
            *,
            consolidate_blocks: bool = False,
            unify_blocks: bool = False,
            dtype: TDtypeAny | None = None,
            ) -> None:
        self._container: TVContainer_co = container
        self._consolidate_blocks = consolidate_blocks
        self._unify_blocks = unify_blocks
        self._dtype: TDtypeAny | None = dtype

    def __call__(self,
            *,
            consolidate_blocks: bool = False,
            unify_blocks: bool = False,
            dtype: TDtypeAny | None = None,
            ) -> 'InterfaceValues[TVContainer_co]':
        '''
        Args:
            consolidate_blocks: Group adjacent same-typed arrays into 2D arrays.
            unify_blocks: Group all arrays into single array, re-typing to an appropriate dtype.
            dtype: specify a dtype to be used in conversion before consolidation or unification, and before function application.
        '''
        return self.__class__(self._container,
                consolidate_blocks=consolidate_blocks,
                unify_blocks=unify_blocks,
                dtype=dtype,
                )

    def __array_ufunc__(self,
            ufunc: TUFunc,
            method: str,
            *args: tp.Any,
            **kwargs: tp.Any,
            ) -> TVContainer_co:
        '''Support for applying NumPy functions directly on containers.
        '''
        from static_frame.core.frame import Frame
        from static_frame.core.series import Series


        if method not in VALID_UFUNC_ARRAY_METHODS:
            return NotImplemented #pragma: no cover

        def func(block: TNDArrayAny, normalize_2d: bool = True) -> TNDArrayAny:
            if normalize_2d:
                block = column_2d_filter(block)

            # NOTE: we assume that our target array (the passed in block) should alwasy be the first argument; then, we filter out arguments that are either this object or an InterfaceBatchValues instance
            args_final = [block]
            for arg in args:
                if arg is self or isinstance(arg, InterfaceBatchValues):
                    continue
                args_final.append(arg)
            # [(arg if arg is not self else block) for arg in args]
            array = ufunc(*args_final, **kwargs)
            array.flags.writeable = False
            return array

        if self._container._NDIM == 2:
            blocks: tp.Iterable[TNDArrayAny] = self._container._blocks._blocks #type: ignore

            if self._unify_blocks:
                dtype = self._container._blocks._index.dtype if self._dtype is None else self._dtype #type: ignore
                tb = TypeBlocks.from_blocks(func(blocks_to_array_2d(
                        blocks=blocks,
                        shape=self._container.shape, # type: ignore
                        dtype=dtype,
                        )))
            elif self._consolidate_blocks:
                if self._dtype is not None:
                    blocks = (b.astype(self._dtype) for b in blocks)
                tb = TypeBlocks.from_blocks(
                        func(b) for b in TypeBlocks.consolidate_blocks(blocks)
                        )
            else:
                if self._dtype is not None:
                    blocks = (func(b.astype(self._dtype)) for b in blocks)
                else:
                    blocks = (func(b) for b in blocks)
                tb = TypeBlocks.from_blocks(blocks)

            if isinstance(self._container, Frame):
                return self._container.__class__(
                        tb,
                        index=self._container.index,
                        columns=self._container.columns,
                        name=self._container.name,
                        own_index=True,
                        own_data=True,
                        own_columns=self._container.STATIC,
                        )
            #IndexHierarchy
            return self._container._from_type_blocks( #type: ignore
                tb,
                index_constructors=self._container._index_constructors, # type: ignore
                name=self._container._name,
                own_blocks=True,
                )
        # all 1D containers
        if self._dtype is not None:
            values = func(self._container.values.astype(self._dtype),
                    normalize_2d=False,
                    )
        else:
            values = func(self._container.values,
                    normalize_2d=False,
                    )

        if isinstance(self._container, Series):
            return self._container.__class__(values,
                    index=self._container.index,
                    name=self._container.name,
                    own_index=True,
                    )
        # else, Index
        return self._container.__class__(values, # type: ignore
                name=self._container.name,
                )


    def apply(self,
            func: TUFunc,
            *args: tp.Any,
            **kwargs: tp.Any,
            ) -> TVContainer_co:
        return self.__array_ufunc__(
                func,
                '__call__',
                *args,
                **kwargs,
                )

class InterfaceBatchValues(InterfaceBatch):

    __slots__ = (
            '_batch_apply',
            '_consolidate_blocks',
            '_unify_blocks',
            '_dtype',
            )
    _INTERFACE = INTERFACE_VALUES

    def __init__(self,
            batch_apply: tp.Callable[[TCallableAny], 'Batch'],
            *,
            consolidate_blocks: bool = False,
            unify_blocks: bool = False,
            dtype: TDtypeAny | None = None,
            ) -> None:
        self._batch_apply = batch_apply
        self._consolidate_blocks = consolidate_blocks
        self._unify_blocks = unify_blocks
        self._dtype = dtype

    #---------------------------------------------------------------------------
    def apply(self,
            func: TUFunc,
            *args: tp.Any,
            **kwargs: tp.Any,
            ) -> 'Batch':
        '''
        Interface for using binary operators and methods with a pre-defined fill value.
        '''
        return self._batch_apply(lambda c: c.via_values(
                consolidate_blocks=self._consolidate_blocks,
                unify_blocks=self._unify_blocks,
                dtype=self._dtype,
                ).apply(func, *args, **kwargs))

    def __call__(self,
            *,
            consolidate_blocks: bool = False,
            unify_blocks: bool = False,
            dtype: TDtypeAny | None = None,
            ) -> 'InterfaceBatchValues':
        '''
        Args:
            consolidate_blocks: Group adjacent same-typed arrays into 2D arrays.
            unify_blocks: Group all arrays into single array, re-typing to an appropriate dtype.
            dtype: specify a dtype to be used in conversion before consolidation or unification, and before function application.
        '''
        return self.__class__(self._batch_apply,
                consolidate_blocks=consolidate_blocks,
                unify_blocks=unify_blocks,
                dtype=dtype,
                )

    def __array_ufunc__(self,
            ufunc: TUFunc,
            method: str,
            *args: tp.Any,
            **kwargs: tp.Any,
            ) -> 'Batch':
        '''Support for applying NumPy functions directly on containers, returning NumPy arrays.
        '''
        # NOTE: want to fail method is not supported at call time of this function, not the deferred execution via Batch
        if method not in VALID_UFUNC_ARRAY_METHODS:
            return NotImplemented #pragma: no cover

        def func(c: TVContainer_co) -> TVContainer_co:
            return c.via_values(
                    consolidate_blocks=self._consolidate_blocks,
                    unify_blocks=self._unify_blocks,
                    dtype=self._dtype,
                    ).__array_ufunc__(ufunc,
                            method,
                            *args,
                            **kwargs,
                            )

        return self._batch_apply(func)

