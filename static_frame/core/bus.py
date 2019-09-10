
import typing as tp

from static_frame.core.series import Series
from static_frame.core.frame import Frame

from static_frame.core.exception import ErrorInitBus
from static_frame.core.util import GetItemKeyType
from static_frame.core.display import DisplayConfig
from static_frame.core.display import DisplayActive
from static_frame.core.display import Display
from static_frame.core.display import DisplayHeader

class Bus:

    __slots__ = (
        '_series',
        )


    @classmethod
    def from_frames(cls, *frames) -> 'Bus':
        series = Series.from_items(
                    ((f.name, f) for f in frames),
                    dtype=object
                    )
        return cls(series)

    def __init__(self, series: Series):

        for value in series.values:
            if not isinstance(value, Frame):
                raise ErrorInitBus(f'supplied {value} is not a frame')

        self._series = series


    def __getattr__(self, name) -> tp.Any:
        return getattr(self._series, name)

    def __getitem__(self, key: GetItemKeyType) -> Series:
        return self._series.__getitem__(key)

    def __reversed__(self) -> tp.Iterator[tp.Hashable]:
        return reversed(self._series._index)

    def __len__(self) -> int:
        return self._series.__len__()


    def display(self,
            config: tp.Optional[DisplayConfig] = None
            ) -> Display:
        '''Return a Display of the Series.
        '''
        config = config or DisplayActive.get()

        d = Display([],
                config=config,
                outermost=True,
                index_depth=1,
                columns_depth=2) # series and index header

        display_index = self._index.display(config=config)
        d.extend_display(display_index)

        d.extend_display(Display.from_values(
                self.values,
                header='',
                config=config))

        display_cls = Display.from_values((),
                header=DisplayHeader(self.__class__, self._series._name),
                config=config)
        d.insert_displays(display_cls.flatten())
        return d

    def __repr__(self):
        return repr(self.display())