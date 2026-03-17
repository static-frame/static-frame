from __future__ import annotations

from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView

import typing_extensions as tp

from static_frame.core.container_util import is_element

if tp.TYPE_CHECKING:
    from static_frame.core.bus import Bus
    from static_frame.core.generic_aliases import TFrameAny

TVKeys = tp.TypeVar('TVKeys')


# -------------------------------------------------------------------------------
class BusMappingKeysView(KeysView[TVKeys]):
    def __init__(self, bus: Bus) -> None:
        KeysView.__init__(self, bus._index)  # pyright: ignore


class BusMappingItemsView(ItemsView[TVKeys, tp.Any]):
    def __init__(self, bus: Bus) -> None:
        self._bus = bus
        ItemsView.__init__(self, bus)  # pyright: ignore

    def __contains__(
        self,
        item: object,
        /,
    ) -> bool:
        if not isinstance(item, tuple) or len(item) != 2:
            return False
        key, value = item
        try:
            frame = self._bus._extract_loc(key)  # type: ignore[arg-type]
        except KeyError:
            return False
        return frame is value  # type: ignore[return-value]

    def __iter__(self) -> Iterator[tp.Tuple[TVKeys, TFrameAny]]:
        yield from self._bus.items()


class BusMappingValuesView(ValuesView[tp.Any]):
    def __init__(self, bus: Bus) -> None:
        self._bus = bus
        ValuesView.__init__(self, bus)  # pyright: ignore

    def __contains__(
        self,
        value: object,
        /,
    ) -> bool:
        for _, frame in self._bus.items():
            if frame is value:
                return True
        return False

    def __iter__(self) -> Iterator[TFrameAny]:
        for _, frame in self._bus.items():
            yield frame


# -------------------------------------------------------------------------------
class BusMapping(Mapping[TVKeys, tp.Any]):
    """A `collections.abc.Mapping` subclass that provides a view into the index and :obj:`Frame` values of a :obj:`Bus` as a compliant mapping type. This container is designed to be completely compatible with read-only ``dict`` and related interfaces. It does not copy underlying data and is immutable. Importantly, it holds on to the :obj:`Bus` and uses it directly, preserving the lazy loading paradigm of the :obj:`Bus`."""

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

    def __init__(self, bus: Bus) -> None:
        from static_frame.core.bus import Bus

        assert isinstance(bus, Bus)
        self._bus = bus

    def __getitem__(self, key: TVKeys) -> TFrameAny:
        if key.__class__ is slice or not is_element(key):  # type: ignore[comparison-overlap]
            raise KeyError(str(key))
        return self._bus._extract_loc(key)  # type: ignore[return-value, arg-type]

    def __iter__(self) -> Iterator[TVKeys]:
        return iter(self._bus._index)  # pyright: ignore

    def __len__(self) -> int:
        return len(self._bus)

    def __contains__(
        self,
        key: object,
        /,
    ) -> bool:
        return key in self._bus._index

    def __repr__(self) -> str:
        return '{}({{{}}})'.format(
            self.__class__.__name__,
            ', '.join(f'{k}: {v.__class__.__name__}' for k, v in self._bus.items()),
        )

    # ---------------------------------------------------------------------------

    def keys(self) -> BusMappingKeysView[TVKeys]:
        return BusMappingKeysView(self._bus)

    def values(self) -> BusMappingValuesView[tp.Any]:
        return BusMappingValuesView(self._bus)

    def items(self) -> BusMappingItemsView[TVKeys, TFrameAny]:
        return BusMappingItemsView(self._bus)
