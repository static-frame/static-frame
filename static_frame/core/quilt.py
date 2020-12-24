import typing as tp
from itertools import zip_longest

import numpy as np

from static_frame.core.container import ContainerOperand
from static_frame.core.store_client_mixin import StoreClientMixin
from static_frame.core.frame import Frame
from static_frame.core.index_base import IndexBase
from static_frame.core.bus import Bus
from static_frame.core.util import NameType
from static_frame.core.store import StoreConfigMapInitializer
from static_frame.core.doc_str import doc_inject
from static_frame.core.display_config import DisplayConfig
from static_frame.core.display import Display
from static_frame.core.series import Series
from static_frame.core.exception import AxisInvalid
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import GetItemKeyTypeCompound
from static_frame.core.node_selector import InterfaceGetItem
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.hloc import HLoc
from static_frame.core.util import duplicate_filter
from static_frame.core.util import INT_TYPES

class AxisMap:
    '''
    An AxisMap is a Series where index values point to string label in a Bus.
    '''

    @staticmethod
    def from_tree(
            tree: tp.Dict[tp.Hashable, IndexBase],
            ) -> Series:

        index = IndexHierarchy.from_tree(tree)
        return Series(
                index.values_at_depth(0), # store the labels as series values
                index=index,
                own_index=True,
                )

    @classmethod
    def from_bus(cls, bus: Bus, axis: int) -> Series:
        '''
        Given a :obj:`Bus` and an axis, derive a :obj:`Series` that maps the derived total index (concatenating all Bus components) to their Bus label.
        '''
        tree = {}
        # NOTE: need to extract just axis labels, not both
        for label, f in bus.items():
            if axis == 0:
                tree[label] = f.index
            elif axis == 1:
                tree[label] = f.columns
            else:
                raise AxisInvalid(f'invalid axis {axis}')

        return cls.from_tree(tree) # type: ignore


class Quilt(ContainerOperand, StoreClientMixin):

    __slots__ = (
            '_bus',
            '_axis',
            '_axis_map',
            '_retain_bus_labels',
            '_axis_opposite',
            '_assign_axis',
            '_columns',
            '_index',
            # '_name', # can use the name of the stored Bus
            # '_config', # stored in Bus
            # '_max_workers',
            # '_chunksize',
            # '_use_threads',
            )

    _bus: Bus
    _axis: int
    _axis_map: tp.Optional[Series]
    _axis_opposite: tp.Optional[IndexBase]
    _columns: IndexBase
    _index: IndexBase
    _assign_axis: bool

    _NDIM: int = 2

    @classmethod
    def from_frame(cls,
            frame: Frame,
            *,
            chunksize: int,
            retain_labels: bool,
            axis: int = 0,
            name: NameType = None,
            label_extractor: tp.Optional[tp.Callable[[IndexBase], tp.Hashable]] = None,
            config: StoreConfigMapInitializer = None,
            ) -> 'Quilt':
        '''
        Given a :obj:`Frame`, create a :obj:`Quilt` by partitioning it along the specified ``axis`` in units of ``chunksize``, where ``axis`` 0 partitions vertically (retaining aligned columns) nad 1 partions horizontally (retaining aligned index).

        Args:
            label_extractor: Function that, given the partitioned index component along the specified axis, returns a string label for that chunk.
        '''
        vector = frame._index if axis == 0 else frame._columns
        vector_len = len(vector)

        starts = range(0, vector_len, chunksize)
        if len(starts) == 1:
            ends: tp.Iterable[int] = (vector_len,)
        else:
            ends = range(starts[1], vector_len, chunksize)

        if label_extractor is None:
            label_extractor = lambda x: x.iloc[0] #type: ignore

        axis_map_components: tp.Dict[tp.Hashable, IndexBase] = {}

        def values() -> tp.Iterator[Frame]:
            for start, end in zip_longest(starts, ends, fillvalue=vector_len):
                if axis == 0: # along rows
                    f = frame.iloc[start:end]
                    label = label_extractor(f.index) #type: ignore
                    axis_map_components[label] = f.index
                elif axis == 1: # along columns
                    f = frame.iloc[:, start:end]
                    label = label_extractor(f.columns) #type: ignore
                    axis_map_components[label] = f.columns
                else:
                    raise AxisInvalid(f'invalid axis {axis}')
                yield f.rename(label)

        name = name if name else frame.name
        bus = Bus.from_frames(values(), config=config, name=name)

        axis_map = AxisMap.from_tree(axis_map_components)

        return cls(bus,
                axis=axis,
                axis_map=axis_map,
                retain_labels=retain_labels,
                )

    #---------------------------------------------------------------------------
    def __init__(self,
            bus: Bus,
            *,
            axis: int = 0,
            retain_labels: bool,
            axis_map: tp.Optional[Series] = None,
            axis_opposite: tp.Optional[IndexBase] = None,
            ) -> None:
        self._bus = bus
        self._axis = axis
        self._retain_bus_labels = retain_labels

        # defer creation until needed
        self._axis_map = axis_map
        self._axis_opposite = axis_opposite
        # will be set with re-axis
        # self._index = None
        # self._columns = None
        self._assign_axis = True


    #---------------------------------------------------------------------------
    # deferred loading of axis info

    def _update_axis_labels(self) -> None:
        if self._axis_map is None:
            self._axis_map = AxisMap.from_bus(self._bus, self._axis)

        if self._axis_opposite is None:
            # always assume thee first Frame in the Quilt is representative; otherwise, need to get a union index.
            if self._axis == 0: # get columns
                self._axis_opposite = self._bus.iloc[0].columns
            else:
                self._axis_opposite = self._bus.iloc[0].index

        if self._axis == 0:
            if not self._retain_bus_labels:
                self._index = self._axis_map.index.level_drop(1) #type: ignore
            else: # get hierarchical
                self._index = self._axis_map.index
            self._columns = self._axis_opposite
        else:
            if not self._retain_bus_labels:
                self._columns = self._axis_map.index.level_drop(1) #type: ignore
            else:
                self._columns = self._axis_map.index
            self._index = self._axis_opposite
        self._assign_axis = False

    #---------------------------------------------------------------------------
    # name interface

    @property #type: ignore
    @doc_inject()
    def name(self) -> NameType:
        '''{}'''
        return self._bus._name

    def rename(self, name: NameType) -> 'Quilt':
        '''
        Return a new Quilt with an updated name attribute.
        '''
        return self.__class__(self._bus.rename(name),
                axis=self._axis,
                retain_labels=self._retain_bus_labels,
                axis_map=self._axis_map,
                axis_opposite=self._axis_opposite,
                )

    #---------------------------------------------------------------------------

    def __repr__(self) -> str:
        '''Provide a display of the :obj:`Quilt` that does not exhaust the generator.
        '''
        if self.name:
            header = f'{self.__class__.__name__}: {self.name}'
        else:
            header = self.__class__.__name__
        return f'<{header} at {hex(id(self))}>'

    def display(self,
            config: tp.Optional[DisplayConfig] = None
            ) -> Display:
        '''Provide a :obj:`Frame`-style display of the :obj:`Quilt`.
        '''
        if self._assign_axis:
            self._update_axis_labels()
        return self._extract(NULL_SLICE, NULL_SLICE).display(config) #type: ignore

    #---------------------------------------------------------------------------
    # accessors

    @property #type: ignore
    @doc_inject(selector='values_2d', class_name='Quilt')
    def values(self) -> np.ndarray:
        '''
        {}
        '''
        if self._assign_axis:
            self._update_axis_labels()
        return self._extract(NULL_SLICE, NULL_SLICE).values

    @property
    def index(self) -> IndexBase:
        '''The ``IndexBase`` instance assigned for row labels.
        '''
        if self._assign_axis:
            self._update_axis_labels()
        return self._index

    @property
    def columns(self) -> IndexBase:
        '''The ``IndexBase`` instance assigned for column labels.
        '''
        if self._assign_axis:
            self._update_axis_labels()
        return self._columns

    #---------------------------------------------------------------------------

    @property
    def shape(self) -> tp.Tuple[int, int]:
        '''
        Return a tuple describing the shape of the underlying NumPy array.

        Returns:
            :obj:`tp.Tuple[int]`
        '''
        if self._assign_axis:
            self._update_axis_labels()
        return len(self._index), len(self._columns)

    @property
    def ndim(self) -> int:
        '''
        Return the number of dimensions, which for a `Frame` is always 2.

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
        if self._assign_axis:
            self._update_axis_labels()
        return len(self._index) * len(self._columns)

    @property
    def nbytes(self) -> int:
        '''
        Return the total bytes of the underlying NumPy arrays.

        Returns:
            :obj:`int`
        '''
        # return self._blocks.nbytes
        if self._assign_axis:
            self._update_axis_labels()
        return sum(f.nbytes for _, f in self._bus.items())


    #---------------------------------------------------------------------------

    def _extract(self,
            row_key: GetItemKeyType = None,
            column_key: GetItemKeyType = None,
            ) -> tp.Union[Frame, Series]:
        '''
        Extract based on iloc selection.
        '''
        parts: tp.List[tp.Any] = []

        row_key = NULL_SLICE if row_key is None else row_key
        column_key = NULL_SLICE if column_key is None else column_key

        sel = np.full(len(self._axis_map), False) #type: ignore
        if self._axis == 0:
            sel_key = row_key
            opposite_key = column_key
        else:
            sel_key = column_key
            opposite_key = row_key

        if isinstance(sel_key, INT_TYPES):
            sel_reduces = True
        else:
            sel_reduces = False

        sel[sel_key] = True
        sel.flags.writeable = False
        sel_map = Series(sel, index=self._axis_map.index, own_index=True) #type: ignore
        # get ordered unique Bus labels from AxisMap Series values; cannot use .unique as need order
        axis_map_sub = self._axis_map.iloc[sel_key] #type: ignore
        if not isinstance(axis_map_sub, Series): # we have an element integer
            bus_keys = (axis_map_sub,)
        else:
            bus_keys = duplicate_filter(axis_map_sub.values) #type: ignore

        for key_count, key in enumerate(bus_keys):
            sel_component = sel_map[HLoc[key]].values # get Boolean array

            if self._axis == 0:
                component = self._bus.loc[key].iloc[sel_component, opposite_key]
                if key_count == 0:
                    component_is_series = isinstance(component, Series)
                if self._retain_bus_labels:
                    # component might be a Series, can call the same with first arg
                    component = component.relabel_level_add(key)
                if sel_reduces: # make Frame into a Series, Series into an element
                    component = component.iloc[0]
            else:
                component = self._bus.loc[key].iloc[opposite_key, sel_component]
                if key_count == 0:
                    component_is_series = isinstance(component, Series)
                if self._retain_bus_labels:
                    if component_is_series:
                        component = component.relabel_level_add(key)
                    else:
                        component = component.relabel_level_add(columns=key)
                if sel_reduces: # make Frame into a Series, Series into an element
                    if component_is_series:
                        component = component.iloc[0]
                    else:
                        component = component.iloc[NULL_SLICE, 0]
            parts.append(component)

        # import ipdb; ipdb.set_trace()
        if len(parts) == 1:
            return parts.pop() #type: ignore
        if component_is_series:
            return Series.from_concat(parts)
        return Frame.from_concat(parts, axis=self._axis) #type: ignore
        # raise NotImplementedError(f'no handling for {parts[0]}')


    # NOTE: the following methods are nearly duplicated from Frame

    def _extract_iloc(self, key: GetItemKeyTypeCompound) -> tp.Union[Series, Frame]:
        '''
        Give a compound key, return a new Frame. This method simply handles the variabiliyt of single or compound selectors.
        '''
        if self._assign_axis:
            self._update_axis_labels()
        if isinstance(key, tuple):
            return self._extract(*key)
        return self._extract(row_key=key)

    def _compound_loc_to_iloc(self,
            key: GetItemKeyTypeCompound) -> tp.Tuple[GetItemKeyType, GetItemKeyType]:
        '''
        Given a compound iloc key, return a tuple of row, column keys. Assumes the first argument is always a row extractor.
        '''
        if isinstance(key, tuple):
            loc_row_key, loc_column_key = key
            iloc_column_key = self._columns.loc_to_iloc(loc_column_key)
        else:
            loc_row_key = key
            iloc_column_key = None

        iloc_row_key = self._index.loc_to_iloc(loc_row_key)
        return iloc_row_key, iloc_column_key

    def _extract_loc(self, key: GetItemKeyTypeCompound) -> tp.Union[Series, Frame]:
        if self._assign_axis:
            self._update_axis_labels()
        return self._extract(*self._compound_loc_to_iloc(key))

    def _compound_loc_to_getitem_iloc(self,
            key: GetItemKeyTypeCompound) -> tp.Tuple[GetItemKeyType, GetItemKeyType]:
        '''Handle a potentially compound key in the style of __getitem__. This will raise an appropriate exception if a two argument loc-style call is attempted.
        '''
        iloc_column_key = self._columns.loc_to_iloc(key)
        return None, iloc_column_key

    @doc_inject(selector='selector')
    def __getitem__(self, key: GetItemKeyType) -> tp.Union[Frame, Series]:
        '''Selector of columns by label.

        Args:
            key: {key_loc}
        '''
        if self._assign_axis:
            self._update_axis_labels()
        return self._extract(*self._compound_loc_to_getitem_iloc(key))



    #---------------------------------------------------------------------------
    # interfaces

    @property
    def loc(self) -> InterfaceGetItem['Frame']:
        return InterfaceGetItem(self._extract_loc) #type: ignore

    @property
    def iloc(self) -> InterfaceGetItem['Frame']:
        return InterfaceGetItem(self._extract_iloc) #type: ignore


    #---------------------------------------------------------------------------
