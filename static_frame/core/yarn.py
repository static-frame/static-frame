import typing as tp
# from itertools import zip_longest
# from functools import partial
# from copy import deepcopy

import numpy as np

from static_frame.core.bus import Bus
from static_frame.core.container import ContainerBase
# from static_frame.core.container_util import axis_window_items
# from static_frame.core.display import Display
# from static_frame.core.display_config import DisplayConfig
from static_frame.core.doc_str import doc_inject
# from static_frame.core.exception import AxisInvalid
from static_frame.core.exception import ErrorInitYarn
# from static_frame.core.exception import NotImplementedAxis
# from static_frame.core.frame import Frame
# from static_frame.core.hloc import HLoc
from static_frame.core.index_base import IndexBase
from static_frame.core.index_hierarchy import IndexHierarchy
# from static_frame.core.node_iter import IterNodeAxis
# from static_frame.core.node_iter import IterNodeConstructorAxis
# from static_frame.core.node_iter import IterNodeType
# from static_frame.core.node_iter import IterNodeWindow
# from static_frame.core.node_iter import IterNodeApplyType
# from static_frame.core.node_selector import InterfaceGetItem
from static_frame.core.series import Series
# from static_frame.core.store import Store
# from static_frame.core.store import StoreConfigMapInitializer
from static_frame.core.store_client_mixin import StoreClientMixin
# from static_frame.core.store_hdf5 import StoreHDF5
# from static_frame.core.store_sqlite import StoreSQLite
# from static_frame.core.store_xlsx import StoreXLSX
# from static_frame.core.store_zip import StoreZipCSV
# from static_frame.core.store_zip import StoreZipParquet
# from static_frame.core.store_zip import StoreZipPickle
# from static_frame.core.store_zip import StoreZipTSV
# from static_frame.core.util import AnyCallable
# from static_frame.core.util import array_deepcopy
# from static_frame.core.util import duplicate_filter
# from static_frame.core.util import get_tuple_constructor
from static_frame.core.util import GetItemKeyType
# from static_frame.core.util import GetItemKeyTypeCompound
# from static_frame.core.util import INT_TYPES
from static_frame.core.util import NameType
# from static_frame.core.util import NULL_SLICE
# from static_frame.core.util import PathSpecifier
from static_frame.core.util import DTYPE_OBJECT

# from static_frame.core.util import concat_resolved
# from static_frame.core.style_config import StyleConfig
from static_frame.core.axis_map import buses_to_hierarchy



class Yarn(ContainerBase, StoreClientMixin):
    '''
    A :obj:`Series`-like container of ordered collections of :obj:`Bus`. If the labels of the index are unique accross all contained :obj:`Bus`, ``retain_labels`` can be set to ``False`` and underlying labels are simply concatenated; otherwise, ``retain_labels`` must be set to ``True`` and an additional depth-level is added to the index labels.
    '''

    __slots__ = (
            '_series',
            '_hierarchy',
            '_retain_labels',
            '_assign_index',
            '_index',
            '_deepcopy_from_bus',
            )

    _series: Series
    _hierarchy: tp.Optional[Series]
    _index: IndexBase
    _assign_index: bool

    _NDIM: int = 1

    @classmethod
    def from_buses(cls,
            buses: tp.Iterable[Bus],
            *,
            name: NameType = None,
            retain_labels: bool,
            ) -> 'Yarn':
        '''Return a :obj:`Yarn` from an iterable of :obj:`Bus`; labels will be drawn from :obj:`Bus.name`.
        '''
        series = Series.from_items(
                    ((b.name, b) for b in buses),
                    dtype=DTYPE_OBJECT,
                    name=name,
                    )
        return cls(series, retain_labels=retain_labels)

    #---------------------------------------------------------------------------
    def __init__(self,
            series: Series,
            *,
            retain_labels: bool,
            hierarchy: tp.Optional[IndexHierarchy] = None,
            deepcopy_from_bus: bool = False,
            ) -> None:
        '''
        Args:
            series: A :obj:`Series` of :obj:`Bus`.
        '''
        if series.dtype != DTYPE_OBJECT:
            raise ErrorInitYarn(
                    f'Series passed to initializer must have dtype object, not {series.dtype}')

        self._series = series # Bus by Bus label

        self._retain_labels = retain_labels
        self._hierarchy = hierarchy # pass in delegation moves
        #self._index assigned in _update_index_labels()
        self._deepcopy_from_bus = deepcopy_from_bus

        self._assign_index = True # Boolean to control deferred index creation

    #---------------------------------------------------------------------------
    # deferred loading of axis info

    def _update_index_labels(self) -> None:
        # _hierarchy might be None while we still need to set self._index
        if self._hierarchy is None:
            self._hierarchy = buses_to_hierarchy(
                    self._series.values,
                    deepcopy_from_bus=self._deepcopy_from_bus,
                    init_exception_cls=ErrorInitYarn,
                    )

        if not self._retain_labels:
            self._index = self._hierarchy.level_drop(1) #type: ignore
        else: # get hierarchical
            self._index = self._hierarchy

        self._assign_index = False


    #---------------------------------------------------------------------------
    # name interface

    @property #type: ignore
    @doc_inject()
    def name(self) -> NameType:
        '''{}'''
        return self._series._name

    def rename(self, name: NameType) -> 'Yarn':
        '''
        Return a new :obj:`Yarn` with an updated name attribute.

        Args:
            name
        '''
        # NOTE: do not need to call _update_index_labels; can continue to defer
        series = self._series.rename(name)
        return self.__class__(series,
                retain_labels=self._retain_labels,
                hierarchy=self._hierarchy,
                deepcopy_from_bus=self._deepcopy_from_bus,
                )

    #---------------------------------------------------------------------------
    def __len__(self) -> int:
        '''Length of values.
        '''
        if self._assign_index:
            self._update_index_labels()
        return self._index.__len__()

    #---------------------------------------------------------------------------
    # common attributes from the numpy array

    @property
    def dtype(self) -> np.dtype:
        '''
        Return the dtype of the realized NumPy array.

        Returns:
            :obj:`numpy.dtype`
        '''
        return DTYPE_OBJECT # always dtype object

    @property
    def shape(self) -> tp.Tuple[int]:
        '''
        Return a tuple describing the shape of the realized NumPy array.

        Returns:
            :obj:`Tuple[int]`
        '''
        if self._assign_index:
            self._update_index_labels()
        return (self._hierarchy.shape[0],) #type: ignore

    @property
    def ndim(self) -> int:
        '''
        Return the number of dimensions, which for a :obj:`Yarn` is always 1.

        Returns:
            :obj:`int`
        '''
        return self._NDIM

    @property
    def size(self) -> int:
        '''
        Return the size of the underlying NumPy array.

        Returns:
            :obj:`int`
        '''
        if self._assign_index:
            self._update_index_labels()
        return self._hierarchy.shape[0] #type: ignore

    #---------------------------------------------------------------------------

    @property
    def index(self) -> IndexBase:
        '''
        The index instance assigned to this container.

        Returns:
            :obj:`Index`
        '''
        if self._assign_index:
            self._update_index_labels()
        return self._index


    #---------------------------------------------------------------------------
    # extraction

    def _extract_iloc(self, key: GetItemKeyType) -> 'Yarn':
        '''
        Returns:
            Yarn or, if an element is selected, a Frame
        '''
        if self._assign_index:
            self._update_index_labels()

        target_hierarchy = self._hierarchy._extract_iloc(key)
        # get the outer-most index of the hierarchical index
        target_bus_index = target_hierarchy._levels.index

        # do avoid having to do a group by or other selection on the targetted bus, we create a Boolean array equal to the entire realized lengt.
        valid = np.full(len(self._index), False)
        valid[key] = True

        buses = np.empty(len(target_bus_index), dtype=DTYPE_OBJECT)
        # must run accross all labels to get incremental slices of Boolean array, but maybe there is a way to avoid
        pos = 0
        for bus_label, width in self._hierarchy.label_widths_at_depth(0):
            # this should always be a bus
            if bus_label not in target_bus_index:
                pos += width
                continue
            extract_per_bus = valid[pos: pos+width]
            pos += width

            idx = target_bus_index.loc_to_iloc(bus_label)
            buses[idx] = self._series[bus_label]._extract_iloc(extract_per_bus)

        buses.flags.writeable = False
        target_series = Series(buses,
                index=target_bus_index,
                own_index=True,
                name=self._series._name,
                )

        return self.__class__(target_series,
                retain_labels=self._retain_labels,
                hierarchy=target_hierarchy,
                deepcopy_from_bus=self._deepcopy_from_bus,
                )



    def _extract_loc(self, key: GetItemKeyType) -> 'Yarn':

        if self._assign_index:
            self._update_index_labels()

        # use the index active for this Yarn
        key_iloc = self._index._loc_to_iloc(key)
        return self._extract_iloc(key_iloc)


    @doc_inject(selector='selector')
    def __getitem__(self, key: GetItemKeyType) -> 'Yarn':
        '''Selector of values by label.

        Args:
            key: {key_loc}
        '''
        return self._extract_loc(key)




