import typing as tp
from functools import partial
from copy import deepcopy

from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.bus import Bus
from static_frame.core.index_base import IndexBase
from static_frame.core.index_level import TreeNodeT
from static_frame.core.exception import AxisInvalid
from static_frame.core.util import AnyCallable
from static_frame.core.util import array_deepcopy

if tp.TYPE_CHECKING:
    from static_frame.core.yarn import Yarn #pylint: disable=W0611 #pragma: no cover


def get_extractor(
        deepcopy_from_bus: bool,
        is_array: bool,
        memo_active: bool,
        ) -> AnyCallable:
    '''
    Args:
        memo_active: enable usage of a common memoization dictionary accross all calls to extract from this extractor.
    '''
    if deepcopy_from_bus:
        memo: tp.Optional[tp.Dict[int, tp.Any]] = None if not memo_active else {}
        if is_array:
            return partial(array_deepcopy, memo=memo)
        return partial(deepcopy, memo=memo)
    return lambda x: x


def bus_to_hierarchy(
        bus: tp.Union[Bus, 'Yarn'],
        axis: int,
        deepcopy_from_bus: bool,
        init_exception_cls: tp.Type[Exception],
        ) -> tp.Tuple[IndexHierarchy, IndexBase]:
    '''
    Given a :obj:`Bus` and an axis, derive a :obj:`IndexHierarchy`; also return and validate the :obj:`Index` of the opposite axis.
    '''
    # NOTE: need to extract just axis labels, not the full Frame; need new Store/Bus loaders just for label data
    extractor = get_extractor(deepcopy_from_bus, is_array=False, memo_active=False)

    def tree_extractor(index: IndexBase) -> tp.Union[IndexBase, TreeNodeT]:
        index = extractor(index)
        if isinstance(index, IndexHierarchy):
            return index.to_tree()
        return index

    tree: TreeNodeT = {}
    opposite: tp.Optional[IndexBase] = None

    for label, f in bus.items():
        if axis == 0:
            tree[label] = tree_extractor(f.index)
            if opposite is None:
                opposite = extractor(f.columns)
            else:
                if not opposite.equals(f.columns):
                    raise init_exception_cls('opposite axis must have equivalent indices')
        elif axis == 1:
            tree[label] = tree_extractor(f.columns)
            if opposite is None:
                opposite = extractor(f.index)
            else:
                if not opposite.equals(f.index):
                    raise init_exception_cls('opposite axis must have equivalent indices')
        else:
            raise AxisInvalid(f'invalid axis {axis}')

    return IndexHierarchy.from_tree(tree), opposite # type: ignore


def buses_to_hierarchy(
        buses: tp.Iterable[Bus],
        labels: tp.Iterable[tp.Hashable],
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
            raise init_exception_cls(f'Must provide an interable of Bus.')
        if label in tree:
            raise init_exception_cls(f'Bus names must be unique: {label} duplicated')
        tree[label] = extractor(bus._index)

    return IndexHierarchy.from_tree(tree)
