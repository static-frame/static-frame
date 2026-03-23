from __future__ import annotations

from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView

import typing_extensions as tp

from static_frame.core.container_util import is_element

if tp.TYPE_CHECKING:
    from static_frame.core.bus import Bus
    from static_frame.core.generic_aliases import TFrameAny
    from static_frame.core.yarn import Yarn

TVKeys = tp.TypeVar('TVKeys')

TBusOrYarn = tp.Union['Bus', 'Yarn']


# -------------------------------------------------------------------------------
class BusMappingKeysView(KeysView[TVKeys]):
    def __init__(self, bus: TBusOrYarn) -> None:
        KeysView.__init__(self, bus._index)  # pyright: ignore

    def __reversed__(self) -> Iterator[TVKeys]:
        return reversed(self._mapping)  # type: ignore[call-overload]


class BusMappingItemsView(ItemsView[TVKeys, 'TFrameAny']):
    def __init__(self, bus: TBusOrYarn) -> None:
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
            frame = self._bus._extract_loc(key)
        except KeyError:
            return False

        return frame.equals(
            value, compare_name=True, compare_dtype=True, compare_class=True
        )

    def __iter__(self) -> Iterator[tp.Tuple[TVKeys, 'TFrameAny']]:
        yield from self._bus.items()  # pyright: ignore

    def __reversed__(self) -> Iterator[tp.Tuple[TVKeys, 'TFrameAny']]:
        for key in reversed(self._bus._index):  # type: ignore[call-overload]
            yield key, self._bus._extract_loc(key)  # type: ignore[misc]


class BusMappingValuesView(ValuesView['TFrameAny']):
    def __init__(self, bus: TBusOrYarn) -> None:
        self._bus = bus
        ValuesView.__init__(self, bus)  # pyright: ignore

    def __contains__(
        self,
        value: object,
        /,
    ) -> bool:
        for _, frame in self._bus.items():
            if frame.equals(
                value, compare_name=True, compare_dtype=True, compare_class=True
            ):
                return True
        return False

    def __iter__(self) -> Iterator['TFrameAny']:
        yield from self._bus._axis_element()

    def __reversed__(self) -> Iterator['TFrameAny']:
        for key in reversed(self._bus._index):  # type: ignore[call-overload]
            yield self._bus._extract_loc(key)  # type: ignore[misc]


# -------------------------------------------------------------------------------
class BusMapping(Mapping[TVKeys, 'TFrameAny']):
    """A `collections.abc.Mapping` subclass that provides a view into the index and :obj:`Frame` values of a :obj:`Bus` as a compliant mapping type. This container is designed to be completely compatible with read-only ``dict`` and related interfaces. It does not copy underlying data and is immutable. Importantly, it holds on to the :obj:`Bus` and uses it directly, preserving the lazy loading paradigm of the :obj:`Bus`."""

    __slots__ = ('_bus',)

    _INTERFACE = (
        '__getitem__',
        '__iter__',
        '__reversed__',
        '__len__',
        '__contains__',
        '__repr__',
        'keys',
        'values',
        'items',
    )

    def __init__(self, bus: 'Bus') -> None:
        from static_frame.core.bus import Bus

        assert isinstance(bus, Bus)
        self._bus: TBusOrYarn = bus

    def __getitem__(self, key: TVKeys) -> 'TFrameAny':
        if key.__class__ is slice or not is_element(key):
            raise KeyError(str(key))
        return self._bus._extract_loc(key)

    def __iter__(self) -> Iterator[TVKeys]:
        return iter(self._bus._index)  # pyright: ignore

    def __reversed__(self) -> Iterator[TVKeys]:
        return reversed(self._bus._index)  # pyright: ignore

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
            ', '.join(
                f'{k}: {v.__class__.__name__}'
                for k, v in zip(self._bus._index, self._bus._values_mutable)  # type: ignore[union-attr]
            ),
        )

    # ---------------------------------------------------------------------------

    def keys(self) -> BusMappingKeysView[TVKeys]:
        return BusMappingKeysView(self._bus)

    def values(self) -> BusMappingValuesView:
        return BusMappingValuesView(self._bus)

    def items(self) -> BusMappingItemsView[TVKeys]:
        return BusMappingItemsView(self._bus)


# -------------------------------------------------------------------------------
class YarnMapping(BusMapping[TVKeys]):
    """A `collections.abc.Mapping` subclass that provides a view into the index and :obj:`Frame` values of a :obj:`Yarn` as a compliant mapping type. This container is designed to be completely compatible with read-only ``dict`` and related interfaces. It does not copy underlying data and is immutable."""

    __slots__ = ()

    def __init__(self, yarn: 'Yarn') -> None:
        from static_frame.core.yarn import Yarn

        assert isinstance(yarn, Yarn)
        self._bus = yarn

    def __repr__(self) -> str:
        return '{}({{{}}})'.format(
            self.__class__.__name__,
            ', '.join(
                f'{k}: {v.__class__.__name__}'
                for k, v in zip(self._bus._index, self._bus._axis_element())
            ),
        )
