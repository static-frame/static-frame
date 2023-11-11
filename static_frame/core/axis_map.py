from __future__ import annotations

from copy import deepcopy
from functools import partial

import typing_extensions as tp
from arraykit import array_deepcopy

from static_frame.core.bus import Bus
from static_frame.core.exception import AxisInvalid
from static_frame.core.frame import Frame
from static_frame.core.index_auto import IndexAutoConstructorFactory
from static_frame.core.index_base import IndexBase
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.index_hierarchy import IndexHierarchyGO
from static_frame.core.index_hierarchy import TTreeNode
from static_frame.core.util import TCallableAny
from static_frame.core.util import TLabel

if tp.TYPE_CHECKING:
    from static_frame.core.yarn import Yarn  # pylint: disable=W0611 #pragma: no cover
    TYarnAny = Yarn[tp.Any] # pylint: disable=W0611 #pragma: no cover

TFrameAny = Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]] # type: ignore[type-arg]
TBusAny = Bus[tp.Any]

def get_extractor(
        deepcopy_from_bus: bool,
        is_array: bool,
        memo_active: bool,
        ) -> TCallableAny:
    '''
    Args:
        memo_active: enable usage of a common memoization dictionary accross all calls to extract from this extractor.
    '''
    if deepcopy_from_bus:
        memo: tp.Optional[tp.Dict[int, tp.Any]] = None if not memo_active else {}
        if is_array:
            return partial(array_deepcopy, memo=memo) # pyright: ignore
        return partial(deepcopy, memo=memo)
    return lambda x: x


def _bus_to_hierarchy_inner_hierarchies(
        bus: tp.Union[TBusAny, TYarnAny],
        axis: int,
        extractor: tp.Callable[[IndexBase], IndexBase],
        init_exception_cls: tp.Type[Exception],
        ) -> tp.Tuple[IndexHierarchy, IndexBase]:
    '''
    Specialized version of :func:`bus_to_hierarchy` for the case where Bus's frames contains only hierarchical indices on the axis of concatentation
    '''
    opposite: tp.Optional[IndexBase] = None

    def level_add(pair: tp.Tuple[TLabel, TFrameAny]) -> IndexHierarchy:
        nonlocal opposite
        label, frame = pair

        if axis == 0:
            axis0, axis1 = extractor(frame.index), frame.columns
        else:
            assert axis == 1
            axis0, axis1 = extractor(frame.columns), frame.index

        if opposite is None:
            opposite = extractor(axis1)
        else:
            if not opposite.equals(axis1):
                raise init_exception_cls('opposite axis must have equivalent indices')

        assert isinstance(axis0, IndexHierarchy) # true assert
        return axis0.level_add(label)

    items_iter = iter(bus.items())

    primary = IndexHierarchyGO(level_add(next(items_iter)))

    for level in items_iter:
        primary.extend(level_add(level))

    return IndexHierarchy(primary), opposite # type: ignore


def bus_to_hierarchy(
        bus: tp.Union[TBusAny, TYarnAny],
        axis: int,
        deepcopy_from_bus: bool,
        init_exception_cls: tp.Type[Exception],
        ) -> tp.Tuple[IndexHierarchy, IndexBase | None]:
    '''
    Given a :obj:`Bus` and an axis, derive a :obj:`IndexHierarchy`; also return and validate the :obj:`Index` of the opposite axis.
    '''
    # NOTE: need to extract just axis labels, not the full Frame; need new Store/Bus loaders just for label data
    extractor = get_extractor(deepcopy_from_bus, is_array=False, memo_active=False)

    first = tp.cast(TFrameAny, bus.iloc[0])
    if (
        (axis == 0 and isinstance(first.index, IndexHierarchy)) or
        (axis == 1 and isinstance(first.columns, IndexHierarchy))
    ):
        return _bus_to_hierarchy_inner_hierarchies(bus, axis, extractor, init_exception_cls)

    tree: TTreeNode = {}
    opposite: tp.Optional[IndexBase] = None

    for label, f in bus.items():
        if axis == 0:
            tree[label] = extractor(f.index)
            if opposite is None:
                opposite = extractor(f.columns)
            else:
                if not opposite.equals(f.columns):
                    raise init_exception_cls('opposite axis must have equivalent indices')
        elif axis == 1:
            tree[label] = extractor(f.columns)
            if opposite is None:
                opposite = extractor(f.index)
            else:
                if not opposite.equals(f.index):
                    raise init_exception_cls('opposite axis must have equivalent indices')
        else:
            raise AxisInvalid(f'invalid axis {axis}')

    # NOTE: we could try to collect index constructors by using the index of the Bus and observing the indices of the contained Frames, but it is not clear that will be better then using IndexAutoConstructorFactory

    return IndexHierarchy.from_tree(tree,
            index_constructors=IndexAutoConstructorFactory), opposite


def buses_to_hierarchy(
        buses: tp.Iterable[TBusAny],
        labels: tp.Iterable[TLabel],
        deepcopy_from_bus: bool,
        init_exception_cls: tp.Type[Exception],
        ) -> IndexHierarchy:
    '''
    Given an iterable of named :obj:`Bus` derive a :obj:`Series` with an :obj:`IndexHierarchy`.
    '''
    # NOTE: for now, the Returned Series will have bus Names as values; this requires the Yarn to store a dict, not a list
    extractor = get_extractor(deepcopy_from_bus, is_array=False, memo_active=False)

    tree = {}
    for label, bus in zip(labels, buses):
        if not isinstance(bus, Bus):
            raise init_exception_cls('Must provide an interable of Bus.')
        if label in tree:
            raise init_exception_cls(f'Bus names must be unique: {label} duplicated')
        tree[label] = extractor(bus._index)

    return IndexHierarchy.from_tree(tree, index_constructors=IndexAutoConstructorFactory)
