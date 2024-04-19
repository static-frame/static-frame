from __future__ import annotations

from copy import deepcopy
from functools import partial
from itertools import repeat

import typing_extensions as tp
from arraykit import array_deepcopy

from static_frame.core.bus import Bus
from static_frame.core.exception import AxisInvalid
from static_frame.core.generic_aliases import TBusAny
from static_frame.core.generic_aliases import TFrameAny
from static_frame.core.generic_aliases import TIndexAny
from static_frame.core.generic_aliases import TIndexIntDefault
from static_frame.core.index import Index
from static_frame.core.index_auto import IndexAutoConstructorFactory
from static_frame.core.index_base import IndexBase
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.index_hierarchy import IndexHierarchyGO
from static_frame.core.index_hierarchy import TTreeNode
from static_frame.core.util import DTYPE_INT_DEFAULT
from static_frame.core.util import TCallableAny
from static_frame.core.util import TLabel
from static_frame.core.util import TName
from static_frame.core.util import TNDArrayObject

if tp.TYPE_CHECKING:
    from static_frame.core.yarn import Yarn  # pragma: no cover
    TYarnAny = Yarn[tp.Any] #pragma: no cover

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


def buses_to_iloc_hierarchy(
        buses: tp.Iterable[TBusAny],
        deepcopy_from_bus: bool,
        init_exception_cls: tp.Type[Exception],
        ) -> IndexHierarchy[TIndexIntDefault, TIndexAny]:
    '''
    Given an iterable of named :obj:`Bus` derive a obj:`IndexHierarchy` with iloc labels on the outer depth, loc labels on the inner depth.
    '''
    extractor = get_extractor(deepcopy_from_bus, is_array=False, memo_active=False)

    tree: TTreeNode = {}
    for label, bus in enumerate(buses):
        if not isinstance(bus, Bus):
            raise init_exception_cls(f'Must provide an instance of a `Bus`, not {type(bus)}.')
        tree[label] = extractor(bus._index)

    ctor: tp.Callable[..., IndexBase] = partial(Index, dtype=DTYPE_INT_DEFAULT)
    return IndexHierarchy.from_tree(tree,
            index_constructors=[ctor, IndexAutoConstructorFactory], # type: ignore
            )

def buses_to_loc_hierarchy(
        buses: tp.Sequence[TBusAny] | TNDArrayObject,
        deepcopy_from_bus: bool,
        init_exception_cls: tp.Type[Exception],
        ) -> IndexHierarchy:
    '''
    Given an iterable of named :obj:`Bus` derive a obj:`IndexHierarchy` with loc labels on the outer depth, loc labels on the inner depth.
    '''
    # NOTE: for now, the Returned Series will have bus Names as values; this requires the Yarn to store a dict, not a list
    extractor = get_extractor(deepcopy_from_bus, is_array=False, memo_active=False)

    bus_names = set(bus.name for bus in buses)
    if len(bus_names) == len(buses):
        # reuse indexes
        tree = {}
        for bus in buses:
            tree[bus.name] = extractor(bus._index)
        return IndexHierarchy.from_tree(tree, index_constructors=IndexAutoConstructorFactory)

    # if Bus names are not unique, doing this permits discovering if resultant labels are unique
    def labels() -> tp.Iterator[tuple[TName, TLabel]]:
        for bus in buses:
            yield from zip(repeat(bus.name), bus.index)
    return IndexHierarchy.from_labels(labels(), index_constructors=IndexAutoConstructorFactory)
