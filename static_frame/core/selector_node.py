import typing as tp

import numpy as np


from static_frame.core.util import GetItemKeyType
from static_frame.core.util import NULL_SLICE

from static_frame.core.doc_str import doc_inject


if tp.TYPE_CHECKING:

    from static_frame.core.bus import Bus  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.frame import Frame  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.frame import FrameAsType  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.index import Index  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.series import Series  #pylint: disable = W0611 #pragma: no cover
    from static_frame.core.type_blocks import TypeBlocks  #pylint: disable = W0611 #pragma: no cover

#-------------------------------------------------------------------------------

TContainer = tp.TypeVar('TContainer', 'Index', 'Series', 'Frame', 'TypeBlocks', 'Bus')
GetItemFunc = tp.TypeVar('GetItemFunc', bound=tp.Callable[[GetItemKeyType], TContainer])

class InterfaceGetItem(tp.Generic[TContainer]):

    __slots__ = ('_func',)

    def __init__(self, func: tp.Callable[[GetItemKeyType], TContainer]) -> None:
        self._func: tp.Callable[[GetItemKeyType], TContainer] = func

    def __getitem__(self, key: GetItemKeyType) -> TContainer:
        return self._func(key)

#-------------------------------------------------------------------------------

class InterfaceSelection1D(tp.Generic[TContainer]):
    '''An instance to serve as an interface to all of iloc and loc
    '''

    __slots__ = (
            '_func_iloc',
            '_func_loc',
            )

    def __init__(self, *,
            func_iloc: GetItemFunc,
            func_loc: GetItemFunc) -> None:

        self._func_iloc = func_iloc
        self._func_loc = func_loc

    @property
    def iloc(self) -> InterfaceGetItem[TContainer]:
        return InterfaceGetItem(self._func_iloc)

    @property
    def loc(self) -> InterfaceGetItem[TContainer]:
        return InterfaceGetItem(self._func_loc)


#-------------------------------------------------------------------------------
class InterfaceSelection2D(tp.Generic[TContainer]):
    '''An instance to serve as an interface to all of iloc, loc, and __getitem__ extractors.
    '''

    __slots__ = (
            '_func_iloc',
            '_func_loc',
            '_func_getitem',
            )

    def __init__(self, *,
            func_iloc: GetItemFunc,
            func_loc: GetItemFunc,
            func_getitem: GetItemFunc,
            ) -> None:

        self._func_iloc = func_iloc
        self._func_loc = func_loc
        self._func_getitem = func_getitem

    def __getitem__(self, key: GetItemKeyType) -> tp.Any:
        '''Label-based selection.
        '''
        return self._func_getitem(key)

    @property
    def iloc(self) -> InterfaceGetItem[TContainer]:
        '''Integer-position based selection.'''
        return InterfaceGetItem(self._func_iloc)

    @property
    def loc(self) -> InterfaceGetItem[TContainer]:
        '''Label-based selection.
        '''
        return InterfaceGetItem(self._func_loc)

class InterfaceAssign2D(tp.Generic[TContainer]):
    '''An instance to serve as an interface to all of iloc, loc, and __getitem__ extractors.
    '''

    __slots__ = (
            '_func_iloc',
            '_func_loc',
            '_func_getitem',
            '_func_bloc',
            )

    def __init__(self, *,
            func_iloc: GetItemFunc,
            func_loc: GetItemFunc,
            func_getitem: GetItemFunc,
            func_bloc: tp.Any, # not sure what is the right type
            ) -> None:

        self._func_iloc = func_iloc
        self._func_loc = func_loc
        self._func_getitem = func_getitem
        self._func_bloc = func_bloc

    def __getitem__(self, key: GetItemKeyType) -> tp.Any:
        '''Label-based selection.
        '''
        return self._func_getitem(key)

    @property
    def bloc(self) -> InterfaceGetItem[TContainer]:
        '''Boolean based assignment.'''
        return InterfaceGetItem(self._func_bloc)

    @property
    def iloc(self) -> InterfaceGetItem[TContainer]:
        '''Integer-position based assignment.'''
        return InterfaceGetItem(self._func_iloc)

    @property
    def loc(self) -> InterfaceGetItem[TContainer]:
        '''Label-based assignment.
        '''
        return InterfaceGetItem(self._func_loc)

#-------------------------------------------------------------------------------

class InterfaceAsType:
    '''An instance to serve as an interface to __getitem__ extractors.
    '''

    __slots__ = ('_func_getitem',)

    def __init__(self,
            func_getitem: tp.Callable[[GetItemKeyType], 'FrameAsType']
            ) -> None:
        '''
        Args:
            _func_getitem: a callable that expects a _func_getitem key and returns a FrameAsType interface; for example, Frame._extract_getitem_astype.
        '''
        self._func_getitem = func_getitem

    @doc_inject(selector='selector')
    def __getitem__(self, key: GetItemKeyType) -> 'FrameAsType':
        '''Selector of columns by label.

        Args:
            key: {key_loc}
        '''
        return self._func_getitem(key)

    def __call__(self, dtype: np.dtype) -> 'Frame':
        return self._func_getitem(NULL_SLICE)(dtype)
