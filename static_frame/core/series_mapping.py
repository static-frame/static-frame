from __future__ import annotations

from collections.abc import ItemsView
from collections.abc import Iterator
from collections.abc import KeysView
from collections.abc import Mapping
from collections.abc import ValuesView

import typing_extensions as tp

if tp.TYPE_CHECKING:
    from static_frame.core.series import Series

TVKeys = tp.TypeVar('TVKeys')
TVValues = tp.TypeVar('TVValues')

class SeriesMappingKeysView(KeysView[TVKeys]):
    def __init__(self, series: Series):
        if (m := series.index._map) is not None: # type: ignore [attr-defined]
            KeysView.__init__(self, m)
        else:
            # providing a range object for keys is far more efficient than creating a sequence or mapping, and implement __contains__
            KeysView.__init__(self, range(len(series))) # type: ignore [arg-type]

class SeriesMappingItemsView(ItemsView[TVKeys, TVValues]):
    def __init__(self, series: Series):
        ItemsView.__init__(self, series) # type: ignore [arg-type]

class SeriesMappingValuesView(ValuesView[TVValues]):
    def __init__(self, series: Series):
        self._values = series.values
        ValuesView.__init__(self, self._values) # type: ignore [arg-type]

    def __contains__(self, key: object) -> bool:
        # linear time unavoidable
        return key in self._values

    def __iter__(self) -> Iterator[TVValues]:
        # ValueView base class wants to lookup keys to get values; this is more efficient.
        return iter(self._values)


class SeriesMapping(Mapping[TVKeys, TVValues]):

    def __init__(self, series: Series):
        from static_frame.core.series import Series
        assert isinstance(series, Series)
        self._series = series

    def __getitem__(self, key: TVKeys) -> TVValues:
        # should we enforce that key must be an element
        try:
            return self._series._extract_loc(key) # type: ignore [no-any-return]
        except RuntimeError as e:
            # raise for mis-matched IH
            raise KeyError(str(e)) from None

    def __iter__(self) -> Iterator[TVKeys]:
        # for IndexHierarchy, these will be tuples
        return iter(self._series._index)

    def __len__(self) -> int:
        return len(self._series)

    def __contains__(self, key: object) -> bool:
        return key in self._series.index

    def __repr__(self) -> str:
        return '{}({{{}}})'.format(
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
