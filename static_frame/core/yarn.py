import typing as tp
from itertools import zip_longest
from functools import partial
from copy import deepcopy

import numpy as np

from static_frame.core.bus import Bus
from static_frame.core.container import ContainerBase
from static_frame.core.container_util import axis_window_items
from static_frame.core.display import Display
from static_frame.core.display_config import DisplayConfig
from static_frame.core.doc_str import doc_inject
from static_frame.core.exception import AxisInvalid
from static_frame.core.exception import ErrorInitQuilt
from static_frame.core.exception import NotImplementedAxis
from static_frame.core.frame import Frame
from static_frame.core.hloc import HLoc
from static_frame.core.index_base import IndexBase
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.node_iter import IterNodeAxis
from static_frame.core.node_iter import IterNodeConstructorAxis
from static_frame.core.node_iter import IterNodeType
from static_frame.core.node_iter import IterNodeWindow
from static_frame.core.node_iter import IterNodeApplyType
from static_frame.core.node_selector import InterfaceGetItem
from static_frame.core.series import Series
from static_frame.core.store import Store
from static_frame.core.store import StoreConfigMapInitializer
from static_frame.core.store_client_mixin import StoreClientMixin
from static_frame.core.store_hdf5 import StoreHDF5
from static_frame.core.store_sqlite import StoreSQLite
from static_frame.core.store_xlsx import StoreXLSX
from static_frame.core.store_zip import StoreZipCSV
from static_frame.core.store_zip import StoreZipParquet
from static_frame.core.store_zip import StoreZipPickle
from static_frame.core.store_zip import StoreZipTSV
from static_frame.core.util import AnyCallable
from static_frame.core.util import array_deepcopy
from static_frame.core.util import duplicate_filter
from static_frame.core.util import get_tuple_constructor
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import GetItemKeyTypeCompound
from static_frame.core.util import INT_TYPES
from static_frame.core.util import NameType
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import PathSpecifier
from static_frame.core.util import concat_resolved
from static_frame.core.style_config import StyleConfig



class Yarn(ContainerBase, StoreClientMixin):
    '''
    A :obj:`Series`-like container of ordered collections of :obj:`Bus`. If the labels of the index are unique accross all contained :obj:`Bus`, ``retain_labels`` can be set to ``False`` and underlying labels are simply concatenated; otherwise, ``retain_labels`` must be set to ``True`` and an additional depth-level is added to the index labels.
    '''

    __slots__ = (
            '_buses',
            '_index_map',
            '_retain_labels',
            '_assign_axis',
            '_index',
            '_deepcopy_from_bus',
            '_name',
            )

    _buses: tp.Tuple[Bus]
    _index_map: tp.Optional[Series]
    _index: IndexBase
    _assign_index: bool
    _name: NameType

    _NDIM: int = 1


    def __init__(self,
            buses: tp.Iterable[Bus],
            *,
            retain_labels: bool,
            index_map: tp.Optional[Series] = None,
            deepcopy_from_bus: bool = False,
            name: NameType = None,
            ) -> None:
        '''
        Args:

        '''
        self._buses = buses if isinstance(buses, tuple) else tuple(buses)

        self._retain_labels = retain_labels
        self._index_map = index_map # pass in delegation moves
        self._deepcopy_from_bus = deepcopy_from_bus
        self._name = name

        # can creation until needed
        self._assign_index = True # Boolean to controll deferred axis index creation


    #---------------------------------------------------------------------------
    # deferred loading of axis info

    def _update_index_labels(self) -> None:
        if self._index_map is None:
            self._index_map = AxisMap.from_bus(
                    self._bus,
                    axis=self._axis,
                    deepcopy_from_bus=self._deepcopy_from_bus,
                    )

        if not self._retain_labels:
            self._index = self._index_map.index.level_drop(1) #type: ignore
        else: # get hierarchical
            self._index = self._index_map.index

        self._assign_index = False


    #---------------------------------------------------------------------------
    # name interfa

    @property #type: ignore
    @doc_inject()
    def name(self) -> NameType:
        '''{}'''
        return self._name

    def rename(self, name: NameType) -> 'Yarn':
        '''
        Return a new :obj:`Yarn` with an updated name attribute.

        Args:
            name
        '''
        return self.__class__(self._buses,
                retain_labels=self._retain_labels,
                index_map=self._index_map,
                deepcopy_from_bus=self._deepcopy_from_bus,
                name=name,
                )