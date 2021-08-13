import typing as tp
from functools import partial
from copy import deepcopy

from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.bus import Bus
from static_frame.core.series import Series
from static_frame.core.index_base import IndexBase
from static_frame.core.exception import AxisInvalid
from static_frame.core.util import AnyCallable
from static_frame.core.util import array_deepcopy


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


class AxisMap:
    '''
    An AxisMap is a Series where index values point to Bus label as used by Quilt.
    '''

    def get_axis_series(
            tree: tp.Dict[tp.Hashable, IndexBase],
            ) -> Series:

        index = IndexHierarchy.from_tree(tree)
        return Series(
                index.values_at_depth(0), # store the labels as series values
                index=index,
                own_index=True,
                )

    @classmethod
    def from_bus(cls,
            bus: Bus,
            axis: int,
            deepcopy_from_bus: bool,
            init_exception_cls: tp.Type[Exception],
            ) -> tp.Tuple[Series, IndexBase]:
        '''
        Given a :obj:`Bus` and an axis, derive a :obj:`Series` with an :obj:`IndexHierarchy`; also return and validate the :obj:`Index` of the opposite axis.
        '''
        # NOTE: need to extract just axis labels, not the full Frame; need new Store/Bus loaders just for label data

        extractor = get_extractor(deepcopy_from_bus, is_array=False, memo_active=False)

        tree = {}
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
        return cls.get_axis_series(tree), opposite # type: ignore



class IndexMap:
    '''
    An IndexMap is a Series where index values point to Bus index positions as used by Yarn.
    '''

    def get_axis_series(
            tree: tp.Dict[tp.Hashable, IndexBase],
            ) -> Series:

        index = IndexHierarchy.from_tree(tree)
        return Series(
                index.values_at_depth(0), # store the labels as series values
                index=index,
                own_index=True,
                )

    @classmethod
    def from_buses(cls,
            buses: tp.Iterable[Bus],
            deepcopy_from_bus: bool,
            init_exception_cls: tp.Type[Exception],
            ) -> tp.Tuple[Series, IndexBase]:
        '''
        Given an iterable of named :obj:`Bus` derive a :obj:`Series` with an :obj:`IndexHierarchy`.
        '''
        # NOTE: for now, the Returned Series will have bus Names as values; this requires the Yarn to store a dict, not a list

        extractor = get_extractor(deepcopy_from_bus, is_array=False, memo_active=False)

        tree = {}
        for bus in buses:
            if bus.name in tree:
                raise init_exception_cls(f'Bus names must be unique: {bus.name} duplicated')
            tree[bus.name] = extractor(bus._series._index)

        return cls.get_axis_series(tree) # type: ignore





