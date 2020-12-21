import typing as tp
from itertools import zip_longest
from itertools import repeat

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




class AxisMap:
    '''
    An AxisMap is a Series where index values point to string label in a Bus.
    '''

    @staticmethod
    def from_components(
            components: tp.Iterable[tp.Tuple[tp.Iterable[tp.Hashable], str]],
            ) -> Series:

        def items() -> tp.Iterator[tp.Tuple[tp.Hashable, str]]:
            for axis_labels, label in components:
                yield from zip(axis_labels, repeat(label))

        return Series.from_items(items(), dtype=str)


    @staticmethod
    def from_bus(bus: Bus, axis: int) -> Series:
        '''
        Given a :obj:`Bus` and an axis, derive a :obj:`Series` that maps the derived total index (concatenating all Bus components) to their Bus label.
        '''
        def items() -> tp.Iterator[tp.Tuple[tp.Hashable, str]]:
            for label, f in bus.items():
                if axis == 0:
                    yield from zip(f.index, repeat(label))
                elif axis == 1:
                    yield from zip(f.columns, repeat(label))
                else:
                    raise AxisInvalid(f'invalid axis {axis}')

        dtype = bus._series.index.values.dtype
        return Series.from_items(items(), dtype=dtype)




class Quilt(ContainerOperand, StoreClientMixin):

    __slots__ = (
            '_bus',
            '_axis',
            '_axis_map',
            '_axis_opposite',
            '_recache',
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
    _recache: bool

    _NDIM: int = 2

    @classmethod
    def from_frame(cls,
            frame: Frame,
            *,
            chunksize: int,
            axis: int = 0,
            name: NameType = None,
            label_extractor: tp.Optional[tp.Callable[[IndexBase], str]] = None,
            config: StoreConfigMapInitializer = None,
            ) -> 'Quilt':
        '''
        Given a :obj:`Frame`, create a :obj:`Quilt` by partitioning it along the specified ``axis`` in units of ``chunksize``, where ``axis`` 0 partitions vertically (retaining aligned columns) nad 1 partions horizontally (retaining aligned index).

        Args:
            label_extractor: Function that, given the partitioned index component along the specified axis, returns a string label for that chunk.
        '''
        vector = frame._index if axis == 0 else frame._columns

        starts = range(0, len(vector), chunksize)
        ends = range(starts[1], len(vector), chunksize)

        if label_extractor is None:
            label_extractor = lambda x: x.iloc[0] #type: ignore

        axis_map_components = []

        def values() -> tp.Iterator[Frame]:
            for start, end in zip_longest(starts, ends, fillvalue=len(vector)):
                if axis == 0: # along rows
                    f = frame.iloc[start:end]
                    label = label_extractor(f.index) #type: ignore
                    axis_map_components.append((f.index, label))
                elif axis == 1: # along columns
                    f = frame.iloc[:, start:end]
                    label = label_extractor(f.columns) #type: ignore
                    axis_map_components.append((f.columns, label))
                else:
                    raise AxisInvalid(f'invalid axis {axis}')
                yield f.rename(label)

        name = name if name else frame.name
        bus = Bus.from_frames(values(), config=config, name=name)

        axis_map = AxisMap.from_components(axis_map_components)

        return cls(bus, axis=axis, axis_map=axis_map)

    #---------------------------------------------------------------------------
    def __init__(self,
            bus: Bus,
            *,
            axis: int = 0,
            axis_map: tp.Optional[Series] = None,
            axis_opposite: tp.Optional[IndexBase] = None,
            ) -> None:
        self._bus = bus
        self._axis = axis

        # defer creation until needed
        self._axis_map = axis_map
        self._axis_opposite = axis_opposite

        if axis_map is None or axis_opposite is None:
            self._recache = True
        else:
            self._recache = False

    #---------------------------------------------------------------------------
    # deferred loading of axis info

    def _update_array_cache(self) -> None:
        if self._axis_map is None:
            self._axis_map = AxisMap.from_bus(self._bus, self._axis)
        if self._axis_opposite is None:
            # always assume thee first Frame in the Quilt is representative; otherwise, need to get a union index.
            if self._axis == 0: # get columns
                self._axis_opposite = self._bus.iloc[0].columns
            else:
                self._axis_opposite = self._bus.iloc[0].index
        self._recache = False


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
        return self.__class__(self._bus.rename(name))

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
        if self._recache:
            self._update_array_cache()
        raise NotImplementedError()

    #---------------------------------------------------------------------------
    # accessors

    # @property
    # @doc_inject(selector='values_2d', class_name='Quilt')
    # def values(self) -> np.ndarray:
    #     '''
    #     {}
    #     '''
    #     if self._recache:
    #         self._update_array_cache()
    #     raise NotImplementedError()

    @property
    def index(self) -> IndexBase:
        '''The ``IndexBase`` instance assigned for row labels.
        '''
        if self._recache:
            self._update_array_cache()
        return self._axis_map.index if self._axis == 0 else self._axis_opposite #type: ignore

    @property
    def columns(self) -> IndexBase:
        '''The ``IndexBase`` instance assigned for column labels.
        '''
        if self._recache:
            self._update_array_cache()
        return self._axis_opposite if self._axis == 0 else self._axis_map.index #type: ignore

    #---------------------------------------------------------------------------

    @property
    def shape(self) -> tp.Tuple[int, int]:
        '''
        Return a tuple describing the shape of the underlying NumPy array.

        Returns:
            :obj:`tp.Tuple[int]`
        '''
        if self._recache:
            self._update_array_cache()
        return len(self.index), len(self.columns)

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
        if self._recache:
            self._update_array_cache()
        return len(self.index) * len(self.columns)

    # @property
    # def nbytes(self) -> int:
    #     '''
    #     Return the total bytes of the underlying NumPy array.

    #     Returns:
    #         :obj:`int`
    #     '''
    #     # return self._blocks.nbytes
    #     if self._recache:
    #         self._update_array_cache()
    #     raise NotImplementedError()


    #---------------------------------------------------------------------------

    def _extract(self,
            row_key: GetItemKeyType = None,
            column_key: GetItemKeyType = None,
            ) -> tp.Union[Frame, Series]:
        '''
        Extract based on iloc selection.
        '''
        if self._recache:
            self._update_array_cache()

        parts = []
        if row_key is None:
            row_key = NULL_SLICE

        if self._axis == 0:
            # get ordered unique values; cannot use .unique as need order
            bus_keys = dict.fromkeys(self._axis_map.iloc[row_key].values) #type: ignore
            for key in bus_keys:
                # need to trim bus-part after extraction
                parts.append(self._bus.loc[key].iloc[NULL_SLICE, column_key])

        if isinstance(parts[0], Series):
            return Series.from_concat(parts)
        elif isinstance(parts[0], Frame):
            return Frame.from_concat(parts, axis=self._axis)
        raise NotImplementedError(f'no handling for {parts[0]}')


    # NOTE: the following methods are duplicated from Frame

    def _extract_iloc(self, key: GetItemKeyTypeCompound) -> tp.Union[Series, Frame]:
        '''
        Give a compound key, return a new Frame. This method simply handles the variabiliyt of single or compound selectors.
        '''
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
            iloc_column_key = self.columns.loc_to_iloc(loc_column_key)
        else:
            loc_row_key = key
            iloc_column_key = None

        iloc_row_key = self.index.loc_to_iloc(loc_row_key)
        return iloc_row_key, iloc_column_key

    def _extract_loc(self, key: GetItemKeyTypeCompound) -> tp.Union[Series, Frame]:
        return self._extract(*self._compound_loc_to_iloc(key))

    def _compound_loc_to_getitem_iloc(self,
            key: GetItemKeyTypeCompound) -> tp.Tuple[GetItemKeyType, GetItemKeyType]:
        '''Handle a potentially compound key in the style of __getitem__. This will raise an appropriate exception if a two argument loc-style call is attempted.
        '''
        iloc_column_key = self.columns.loc_to_iloc(key)
        return None, iloc_column_key

    @doc_inject(selector='selector')
    def __getitem__(self, key: GetItemKeyType) -> tp.Union[Frame, Series]:
        '''Selector of columns by label.

        Args:
            key: {key_loc}
        '''
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
