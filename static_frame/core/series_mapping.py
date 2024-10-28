from __future__ import annotations

from collections.abc import ItemsView
from collections.abc import Iterator
from collections.abc import KeysView
from collections.abc import Mapping
from collections.abc import ValuesView

import numpy as np
import typing_extensions as tp

from static_frame.core.container_util import is_element

if tp.TYPE_CHECKING:
    from static_frame.core.generic_aliases import TIndexAny  # pragma: no cover
    from static_frame.core.series import Series  # pragma: no cover

TVKeys = tp.TypeVar('TVKeys', bound=np.generic)
TVValues = tp.TypeVar('TVValues', bound=np.generic)


#-------------------------------------------------------------------------------
class SeriesMappingKeysView(KeysView[TVKeys]):
    def __init__(self, series: Series):
        KeysView.__init__(self, series.index) # type: ignore [arg-type]

class SeriesMappingItemsView(ItemsView[TVKeys, TVValues]):
    def __init__(self, series: Series[TIndexAny, TVValues]):
        ItemsView.__init__(self, series) # type: ignore [arg-type]

class SeriesMappingValuesView(ValuesView[TVValues]):

    def __init__(self, series: Series[TIndexAny, TVValues]):
        self._values = series.values
        ValuesView.__init__(self, series.values) # type: ignore [arg-type]

    def __contains__(self, key: object) -> bool:
        # linear time unavoidable
        return self._values.__contains__(key)

    def __iter__(self) -> Iterator[TVValues]:
        # ValueView base class wants to lookup keys to get values; this is more efficient.
        return iter(self._values)

#-------------------------------------------------------------------------------
class SeriesMapping(Mapping[TVKeys, TVValues]):
    '''A `collections.abc.Mapping` subclass that provides a view into the index and values of a `Series` as a compliant mapping type. This container is designed to be completely compatible with read-only `dict` and related interfaces. It does not copy underlying data and is immutable.'''

    _INTERFACE = (
        '__getitem__',
        '__iter__',
        '__len__',
        '__contains__',
        '__repr__',
        'keys',
        'values',
        'items',
        )

    def __init__(self, series: Series[TIndexAny, TVValues]):
        from static_frame.core.series import Series
        assert isinstance(series, Series)
        self._series = series

    def __getitem__(self, key: TVKeys) -> TVValues:
        #enforce that key must be an element
        if key.__class__ is slice or not is_element(key): # type: ignore [comparison-overlap]
            raise KeyError(str(key))
        try:
            return self._series._extract_loc(key) # type: ignore [no-any-return]
        except RuntimeError as e:
            # raise for mis-matched IH
            raise KeyError(str(e)) from None

    def __iter__(self) -> Iterator[TVKeys]:
        # for IndexHierarchy, these will be tuples
        return iter(self._series._index) # pyright: ignore

    def __len__(self) -> int:
        return len(self._series)

    def __contains__(self, key: object) -> bool:
        return key in self._series.index

    def __repr__(self) -> str:
        return '{}({{{}}})'.format(  # pylint: disable=C0209
            self.__class__.__name__,
            ', '.join(f'{k}: {v}' for k, v in self._series.items()),
            )

    #---------------------------------------------------------------------------

    def keys(self) -> SeriesMappingKeysView[TVKeys]:
        return SeriesMappingKeysView(self._series)

    def values(self) -> SeriesMappingValuesView[TVValues]:
        return SeriesMappingValuesView(self._series)

    def items(self) -> SeriesMappingItemsView[TVKeys, TVValues]:
        return SeriesMappingItemsView(self._series)
