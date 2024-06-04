from __future__ import annotations

import numpy as np
import typing_extensions as tp
from numpy.ma import MaskedArray

from static_frame.core.assign import Assign
from static_frame.core.doc_str import doc_inject
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import TBlocKey
from static_frame.core.util import TCallableAny
from static_frame.core.util import TDepthLevelSpecifier
from static_frame.core.util import TDtypeSpecifier
from static_frame.core.util import TDtypesSpecifier
from static_frame.core.util import TILocSelector
from static_frame.core.util import TILocSelectorCompound
from static_frame.core.util import TILocSelectorMany
from static_frame.core.util import TILocSelectorOne
from static_frame.core.util import TLabel
from static_frame.core.util import TLocSelector
from static_frame.core.util import TLocSelectorCompound
from static_frame.core.util import TLocSelectorMany

# from static_frame.core.util import TCallableAny

if tp.TYPE_CHECKING:
    from static_frame.core.batch import Batch  # pragma: no cover
    from static_frame.core.bus import Bus  # pragma: no cover
    from static_frame.core.frame import Frame  # pragma: no cover
    from static_frame.core.frame import FrameAssignILoc  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.frame import FrameAsType  # pragma: no cover
    from static_frame.core.frame import FrameGO  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.frame import FrameHE  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.index import Index  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.index_base import IndexBase  # pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchy  # pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchyAsType  # pragma: no cover
    from static_frame.core.series import Series  # pragma: no cover
    from static_frame.core.series import SeriesAssign  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.series import SeriesHE  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.type_blocks import TypeBlocks  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.yarn import Yarn  # pragma: no cover

    TNDArrayAny = np.ndarray[tp.Any, tp.Any] #pragma: no cover
    TDtypeAny = np.dtype[tp.Any] #pragma: no cover
    TSeriesAny = Series[tp.Any, tp.Any] #pragma: no cover
    TFrameAny = Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]] #pragma: no cover
    TBusAny = Bus[tp.Any] #pragma: no cover
    TYarnAny = Yarn[tp.Any] #pragma: no cover

#-------------------------------------------------------------------------------
TFrameOrSeries = tp.Union[
        'Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]]',
        'Series[tp.Any, tp.Any]',
        ]

TVContainer_co = tp.TypeVar('TVContainer_co',
        'Index[tp.Any]',
        'Series[tp.Any, tp.Any]',
        'SeriesHE[tp.Any, tp.Any]',
        'Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]]',
        'FrameGO[tp.Any, tp.Any]',
        'FrameHE[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]]',
        'TypeBlocks',
        'Bus[tp.Any]',
        'Batch',
        'Yarn[tp.Any]',
        'IndexHierarchy',
        'SeriesAssign',
        'FrameAssignILoc',
        np.ndarray[tp.Any, tp.Any],
        MaskedArray, # type: ignore
        TFrameOrSeries,
        covariant=True,
        )
TLocSelectorFunc = tp.TypeVar('TLocSelectorFunc',
        bound=tp.Callable[[TLocSelector], TVContainer_co] # pyright: ignore
        )
TILocSelectorFunc = tp.TypeVar('TILocSelectorFunc',
        bound=tp.Callable[[TILocSelector], TVContainer_co] # pyright: ignore
        )
TVDtype = tp.TypeVar('TVDtype', bound=np.generic, default=tp.Any) # pylint: disable=E1123

class Interface:
    __slots__ = ()
    _INTERFACE: tp.Tuple[str, ...] = ()

class InterfaceBatch:
    __slots__ = ()
    _INTERFACE: tp.Tuple[str, ...] = ()

class InterGetItemILocReduces(Interface, tp.Generic[TVContainer_co, TVDtype]):
    '''Interface for iloc selection that reduces dimensionality.
    '''
    __slots__ = ('_func',)
    _INTERFACE = ('__getitem__',)

    def __init__(self, func: tp.Union[
            tp.Callable[[TILocSelectorOne], TVDtype],
            tp.Callable[[TILocSelectorMany], TVContainer_co]]) -> None:
        self._func: tp.Union[
            tp.Callable[[TILocSelectorOne], TVDtype],
            tp.Callable[[TILocSelectorMany], TVContainer_co]] = func

    @tp.overload
    def __getitem__(self, key: TILocSelectorMany) -> TVContainer_co: ...

    @tp.overload
    def __getitem__(self, key: TILocSelectorOne) -> TVDtype: ...

    def __getitem__(self, key: TILocSelector) -> tp.Any:
        return self._func(key) # type: ignore


class InterGetItemILoc(Interface, tp.Generic[TVContainer_co]):
    '''Interface for iloc selection that does not reduce dimensionality.
    '''
    __slots__ = ('_func',)
    _INTERFACE = ('__getitem__',)

    def __init__(self, func: tp.Union[
            tp.Callable[[TILocSelectorOne], tp.Any],
            tp.Callable[[TILocSelectorMany], TVContainer_co]]) -> None:
        self._func: tp.Union[
            tp.Callable[[TILocSelectorOne], tp.Any],
            tp.Callable[[TILocSelectorMany], TVContainer_co]] = func

    def __getitem__(self, key: TILocSelector) -> TVContainer_co:
        return self._func(key) # type: ignore


class InterGetItemLocReduces(Interface, tp.Generic[TVContainer_co, TVDtype]):

    __slots__ = ('_func',)
    _INTERFACE = ('__getitem__',)

    _func: tp.Callable[[TLocSelector], TVContainer_co]

    def __init__(self,
            func: tp.Callable[[TLocSelector], TVContainer_co],
            ) -> None:
        self._func = func

    @tp.overload
    def __getitem__(self, key: TLocSelectorMany) -> TVContainer_co: ...

    @tp.overload
    def __getitem__(self, key: TLabel) -> TVDtype: ...

    def __getitem__(self, key: TLocSelector) -> tp.Any:
        return self._func(key)


class InterGetItemLoc(Interface, tp.Generic[TVContainer_co]):

    __slots__ = ('_func',)
    _INTERFACE = ('__getitem__',)

    _func: tp.Callable[[TLocSelector], TVContainer_co]

    def __init__(self, func: tp.Callable[[TLocSelector], TVContainer_co]) -> None:
        self._func = func

    def __getitem__(self, key: TLocSelector) -> TVContainer_co:
        return self._func(key)


class InterGetItemLocCompoundReduces(Interface, tp.Generic[TVContainer_co]):
    '''Interface for compound loc selection that reduces dimensionality. TVContainer_co is the outermost container
    '''

    __slots__ = ('_func',)
    _INTERFACE = ('__getitem__',)

    _func: tp.Callable[[TLocSelectorCompound], tp.Any]

    def __init__(self, func: tp.Callable[[TLocSelectorCompound], tp.Any]) -> None:
        self._func = func

    @tp.overload
    def __getitem__(self, key: tp.Tuple[TLabel, TLocSelectorMany]) -> TSeriesAny: ...

    @tp.overload
    def __getitem__(self, key: tp.Tuple[TLocSelectorMany, TLabel]) -> TSeriesAny: ...

    @tp.overload
    def __getitem__(self, key: tp.Tuple[TLocSelectorMany, TLocSelectorMany]) -> TVContainer_co: ...

    @tp.overload
    def __getitem__(self, key: tp.Tuple[tp.List[int], tp.List[int]]) -> TVContainer_co: ...

    @tp.overload
    def __getitem__(self, key: tp.Tuple[tp.List[str], tp.List[str]]) -> TVContainer_co: ...

    @tp.overload
    def __getitem__(self, key: tp.Tuple[TLabel, TLabel]) -> tp.Any: ...

    @tp.overload
    def __getitem__(self, key: TLabel) -> TSeriesAny: ...

    @tp.overload
    def __getitem__(self, key: TLocSelectorMany) -> TVContainer_co: ...

    @tp.overload
    def __getitem__(self, key: TLocSelectorCompound) -> tp.Any: ...

    def __getitem__(self, key: TLocSelectorCompound) -> tp.Any:
        return self._func(key)



class InterGetItemLocCompound(Interface, tp.Generic[TVContainer_co]):
    '''Interface for compound loc selection that does not reduce dimensionality. TVContainer_co is the only delivered container container
    '''

    __slots__ = ('_func',)
    _INTERFACE = ('__getitem__',)

    _func: tp.Callable[[TLocSelectorCompound], TVContainer_co]

    def __init__(self, func: tp.Callable[[TLocSelectorCompound], TVContainer_co]) -> None:
        self._func = func

    def __getitem__(self, key: TLocSelectorCompound) -> TVContainer_co:
        return self._func(key)



class InterGetItemILocCompoundReduces(Interface, tp.Generic[TVContainer_co]):

    __slots__ = ('_func',)
    _INTERFACE = ('__getitem__',)

    _func: tp.Callable[[TILocSelectorCompound], tp.Any]

    def __init__(self, func: tp.Callable[[TILocSelectorCompound], tp.Any]) -> None:
        self._func = func


    @tp.overload
    def __getitem__(self, key: TILocSelectorOne) -> TSeriesAny: ...

    @tp.overload
    def __getitem__(self, key: TILocSelectorMany) -> TVContainer_co: ...

    @tp.overload
    def __getitem__(self, key: tp.Tuple[TILocSelectorOne, TILocSelectorMany]) -> TSeriesAny: ...

    @tp.overload
    def __getitem__(self, key: tp.Tuple[TILocSelectorMany, TILocSelectorOne]) -> TSeriesAny: ...

    @tp.overload
    def __getitem__(self, key: tp.Tuple[TILocSelectorMany, TILocSelectorMany]) -> TVContainer_co: ...

    @tp.overload
    def __getitem__(self, key: tp.Tuple[TILocSelectorOne, TILocSelectorOne]) -> tp.Any: ...

    @tp.overload
    def __getitem__(self, key: TILocSelectorMany) -> TVContainer_co: ...

    @tp.overload
    def __getitem__(self, key: TILocSelectorCompound) -> tp.Any: ...

    def __getitem__(self, key: TILocSelectorCompound) -> tp.Any:
        return self._func(key)

class InterGetItemILocCompound(Interface, tp.Generic[TVContainer_co]):

    __slots__ = ('_func',)
    _INTERFACE = ('__getitem__',)

    _func: tp.Callable[[TILocSelectorCompound], TVContainer_co]

    def __init__(self, func: tp.Callable[[TILocSelectorCompound], TVContainer_co]) -> None:
        self._func = func

    def __getitem__(self, key: TILocSelectorCompound) -> TVContainer_co:
        return self._func(key)


class InterfaceGetItemBLoc(Interface, tp.Generic[TVContainer_co]):

    __slots__ = ('_func',)
    _INTERFACE = ('__getitem__',)

    _func: tp.Callable[[TBlocKey], TVContainer_co]

    def __init__(self, func: tp.Callable[[TBlocKey], TVContainer_co]) -> None:
        self._func = func

    def __getitem__(self, key: TBlocKey) -> TVContainer_co:
        return self._func(key)


#-------------------------------------------------------------------------------

class InterfaceSelectDuo(Interface, tp.Generic[TVContainer_co]):
    '''An instance to serve as an interface to all of iloc and loc
    '''

    __slots__ = (
            '_func_iloc',
            '_func_loc',
            )
    _INTERFACE = ('iloc', 'loc')

    def __init__(self, *,
            func_iloc: TILocSelectorFunc,
            func_loc: TLocSelectorFunc) -> None:

        self._func_iloc = func_iloc
        self._func_loc = func_loc

    @property
    def iloc(self) -> InterGetItemILocReduces[TVContainer_co, tp.Any]:
        return InterGetItemILocReduces(self._func_iloc)

    @property
    def loc(self) -> InterGetItemLocReduces[TVContainer_co, tp.Any]:
        return InterGetItemLocReduces(self._func_loc) # pyright: ignore

class InterfaceSelectTrio(Interface, tp.Generic[TVContainer_co]):
    '''An instance to serve as an interface to all of iloc, loc, and __getitem__ extractors. It is assumed that functionality that uses this interface returns containers that do not reduce their dimensionality.
    '''

    __slots__ = (
            '_func_iloc',
            '_func_loc',
            '_func_getitem',
            )
    _INTERFACE = ('__getitem__', 'iloc', 'loc')

    def __init__(self, *,
            func_iloc: TILocSelectorFunc,
            func_loc: TLocSelectorFunc,
            func_getitem: TLocSelectorFunc,
            ) -> None:

        self._func_iloc = func_iloc
        self._func_loc = func_loc
        self._func_getitem = func_getitem

    def __getitem__(self, key: TLocSelector) -> tp.Any:
        '''Label-based selection.
        '''
        return self._func_getitem(key)

    @property
    def iloc(self) -> InterGetItemILoc[TVContainer_co]:
        '''Integer-position based selection.'''
        return InterGetItemILoc(self._func_iloc)

    @property
    def loc(self) -> InterGetItemLoc[TVContainer_co]:
        '''Label-based selection.
        '''
        return InterGetItemLoc(self._func_loc) # pyright: ignore


class InterfaceSelectQuartet(Interface, tp.Generic[TVContainer_co]):
    '''An instance to serve as an interface to all of iloc, loc, and __getitem__ extractors.
    '''

    __slots__ = (
            '_func_iloc',
            '_func_loc',
            '_func_getitem',
            '_func_bloc',
            )
    _INTERFACE = ('__getitem__', 'iloc', 'loc', 'bloc')

    def __init__(self, *,
            func_iloc: TILocSelectorFunc,
            func_loc: TLocSelectorFunc,
            func_getitem: TLocSelectorFunc,
            func_bloc: tp.Any, # not sure what is the right type
            ) -> None:

        self._func_iloc = func_iloc
        self._func_loc = func_loc
        self._func_getitem = func_getitem
        self._func_bloc = func_bloc

    def __getitem__(self, key: TLocSelector) -> tp.Any:
        '''Label-based selection.
        '''
        return self._func_getitem(key)

    @property
    def bloc(self) -> InterGetItemLocReduces[TVContainer_co, tp.Any]:
        '''Boolean based assignment.'''
        return InterGetItemLocReduces(self._func_bloc)

    @property
    def iloc(self) -> InterGetItemILocReduces[TVContainer_co, tp.Any]:
        '''Integer-position based assignment.'''
        return InterGetItemILocReduces(self._func_iloc)

    @property
    def loc(self) -> InterGetItemLocReduces[TVContainer_co, tp.Any]:
        '''Label-based assignment.
        '''
        return InterGetItemLocReduces(self._func_loc) # pyright: ignore


#-------------------------------------------------------------------------------

class InterfaceAssignTrio(InterfaceSelectTrio[TVContainer_co]):
    '''For assignment with __getitem__, iloc, loc.
    '''

    __slots__ = ('delegate',)

    def __init__(self, *,
            func_iloc: TILocSelectorFunc,
            func_loc: TLocSelectorFunc,
            func_getitem: TLocSelectorFunc,
            delegate: tp.Type[Assign]
            ) -> None:
        InterfaceSelectTrio.__init__(self,
                func_iloc=func_iloc,
                func_loc=func_loc,
                func_getitem=func_getitem,
                )
        self.delegate = delegate #pylint: disable=E0237


class InterfaceAssignQuartet(InterfaceSelectQuartet[TVContainer_co]):
    '''For assignment with __getitem__, iloc, loc, bloc.
    '''
    __slots__ = ('delegate',)

    def __init__(self, *,
            func_iloc: TILocSelectorFunc,
            func_loc: TLocSelectorFunc,
            func_getitem: TLocSelectorFunc,
            func_bloc: tp.Any, # not sure what is the right type
            delegate: tp.Type[Assign]
            ) -> None:
        InterfaceSelectQuartet.__init__(self,
                func_iloc=func_iloc,
                func_loc=func_loc,
                func_getitem=func_getitem,
                func_bloc=func_bloc,
                )
        self.delegate = delegate #pylint: disable=E0237


#-------------------------------------------------------------------------------

class InterfaceFrameAsType(Interface, tp.Generic[TVContainer_co]):
    __slots__ = ('_func_getitem',)
    _INTERFACE = ('__getitem__', '__call__')

    def __init__(self,
            func_getitem: tp.Callable[[TLocSelector], 'FrameAsType']
            ) -> None:
        '''
        Args:
            _func_getitem: a callable that expects a _func_getitem key and returns a FrameAsType interface; for example, Frame._extract_getitem_astype.
        '''
        self._func_getitem = func_getitem

    @doc_inject(selector='selector')
    def __getitem__(self, key: TLocSelector) -> 'FrameAsType':
        '''Selector of columns by label.

        Args:
            key: {key_loc}
        '''
        return self._func_getitem(key)

    def __call__(self,
            dtype: TDtypeSpecifier,
            *,
            consolidate_blocks: bool = False,
            ) -> TFrameAny:
        '''
        Apply a single ``dtype`` to all columns.
        '''

        return self._func_getitem(NULL_SLICE)(
                dtype,
                consolidate_blocks=consolidate_blocks,
                )


class InterfaceIndexHierarchyAsType(Interface, tp.Generic[TVContainer_co]):
    __slots__ = ('_func_getitem',)
    _INTERFACE = ('__getitem__', '__call__')

    def __init__(self,
            func_getitem: tp.Callable[[TDepthLevelSpecifier], 'IndexHierarchyAsType']
            ) -> None:
        '''
        Args:
            _func_getitem: a callable that expects a _func_getitem key and returns a IndexHierarchyAsType interface; for example, Frame._extract_getitem_astype.
        '''
        self._func_getitem = func_getitem

    @doc_inject(selector='selector')
    def __getitem__(self, key: TDepthLevelSpecifier) -> 'IndexHierarchyAsType':
        '''Selector of columns by label.

        Args:
            key: {key_loc}
        '''
        return self._func_getitem(key)

    def __call__(self,
            dtype: TDtypeAny,
            *,
            consolidate_blocks: bool = False,
            ) -> 'IndexHierarchy':
        '''
        Apply a single ``dtype`` to all columns.
        '''
        return self._func_getitem(NULL_SLICE)(
                dtype,
                consolidate_blocks=consolidate_blocks,
                )



class BatchAsType:

    __slots__ = ('_batch_apply', '_column_key')

    def __init__(self,
            batch_apply: tp.Callable[[TCallableAny], 'Batch'],
            column_key: TLocSelector
            ) -> None:
        self._batch_apply = batch_apply
        self._column_key = column_key

    def __call__(self,
            dtypes: TDtypesSpecifier,
            *,
            consolidate_blocks: bool = False,
            ) -> 'Batch':
        return self._batch_apply(
                lambda c: c.astype[self._column_key](
                    dtypes,
                    consolidate_blocks=consolidate_blocks,
                    )
                )

class InterfaceBatchAsType(Interface, tp.Generic[TVContainer_co]):
    '''An instance to serve as an interface to __getitem__ extractors. Used by both :obj:`Frame` and :obj:`IndexHierarchy`.
    '''

    __slots__ = ('_batch_apply',)
    _INTERFACE = ('__getitem__', '__call__')

    def __init__(self,
            batch_apply: tp.Callable[[TCallableAny], 'Batch'],
            ) -> None:
        self._batch_apply = batch_apply

    @doc_inject(selector='selector')
    def __getitem__(self, key: TLocSelector) -> BatchAsType:
        '''Selector of columns by label.

        Args:
            key: {key_loc}
        '''
        return BatchAsType(batch_apply=self._batch_apply, column_key=key)

    def __call__(self, dtype: TDtypeAny) -> 'Batch':
        '''
        Apply a single ``dtype`` to all columns.
        '''
        return BatchAsType(
                batch_apply=self._batch_apply,
                column_key=NULL_SLICE,
                )(dtype)


#-------------------------------------------------------------------------------

class InterfaceConsolidate(Interface, tp.Generic[TVContainer_co]):
    '''An instance to serve as an interface to __getitem__ extractors.
    '''

    __slots__ = (
            '_container',
            '_func_getitem',
            )

    _INTERFACE = (
            '__getitem__',
            '__call__',
            'status',
            )

    def __init__(self,
            container: TVContainer_co,
            func_getitem: tp.Callable[[TLocSelector], TFrameAny]
            ) -> None:
        '''
        Args:
            _func_getitem: a callable that expects a _func_getitem key and returns a Frame interface.
        '''
        self._container: TVContainer_co = container
        self._func_getitem = func_getitem

    @doc_inject(selector='selector')
    def __getitem__(self, key: TLocSelector) -> TFrameAny:
        '''Return the full ``Frame``, selecting with ``key`` a subset of columns for consolidation.

        Args:
            key: {key_loc}
        '''
        return self._func_getitem(key)

    def __call__(self) -> TFrameAny:
        '''
        Apply consolidation to all columns.
        '''
        return self._func_getitem(NULL_SLICE)

    @property
    def status(self) -> TFrameAny:
        '''Display consolidation status of this Frame.
        '''
        from static_frame.core.frame import Frame

        flag_attrs: tp.Tuple[str, ...] = ('owndata', 'f_contiguous', 'c_contiguous')
        columns: IndexBase = self._container.columns # type: ignore

        def gen() -> tp.Iterator[tp.Sequence[tp.Any]]:
            iloc_start = 0

            for b in self._container._blocks._blocks: # type: ignore
                width = 1 if b.ndim == 1 else b.shape[1]

                iloc_end = iloc_start + width
                if iloc_end >= len(columns):
                    iloc_slice = slice(iloc_start, None)
                else:
                    iloc_slice = slice(iloc_start, iloc_end)

                sub = columns[iloc_slice] # returns a column
                iloc: tp.Union[int, slice]
                if len(sub) == 1:
                    loc = sub[0]
                    iloc = iloc_start
                else: # get inclusive slice
                    loc = slice(sub[0], sub[-1])
                    iloc = iloc_slice

                yield [loc, iloc, b.dtype, b.shape, b.ndim] + [
                    getattr(b.flags, attr) for attr in flag_attrs]

                iloc_start = iloc_end

        return Frame.from_records(gen(),
            columns=('loc', 'iloc', 'dtype', 'shape', 'ndim') + flag_attrs
            )



