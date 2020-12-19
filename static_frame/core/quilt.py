import typing as tp
from itertools import zip_longest
from itertools import repeat

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





class AxisMap:

    @staticmethod
    def from_components(
            components: tp.Iterable[tp.Tuple[tp.Iterable[tp.Hashable], str]],
            ) -> Series:
        def items():
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
            # '_name', # can use the name of the stored Bus
            # '_config', # stored in Bus
            # '_max_workers',
            # '_chunksize',
            # '_use_threads',
            )

    _bus: Bus
    _axis: int


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

        label_extractor = label_extractor if label_extractor else lambda x: x.iloc[0]

        axis_map_components = []

        def values() -> tp.Iterator[Frame]:
            for start, end in zip_longest(starts, ends, fillvalue=len(vector)):
                if axis == 0: # along rows
                    f = frame.iloc[start:end]
                    label = label_extractor(f.index)
                    axis_map_components.append((f.index, label))
                elif axis == 1: # along columns
                    f = frame.iloc[:, start:end]
                    label = label_extractor(f.columns)
                    axis_map_components.append((f.columns, label))
                else:
                    raise AxisInvalid(f'invalid axis {axis}')
                yield f.rename(label)

        name = name if name else frame.name
        bus = Bus.from_frames(values(), config=config, name=name)

        axis_map = AxisMap.from_components(axis_map_components)

        return cls(bus, axis=axis, axis_map=axis_map)

    def __init__(self,
            bus: Bus,
            *,
            axis: int = 0,
            axis_map: tp.Optional[Series] = None,
            ) -> None:
        self._bus = bus
        self._axis = axis

        # defer creation until needed
        self._axis_map: Series = axis_map




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
        pass

    #---------------------------------------------------------------------------
