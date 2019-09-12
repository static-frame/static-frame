
import typing as tp

from static_frame.core.series import Series
from static_frame.core.frame import Frame
# from static_frame.core.frame import Index

from static_frame.core.store import Store
from static_frame.core.store import StoreZipCSV
from static_frame.core.store import StoreZipTSV


from static_frame.core.exception import ErrorInitBus
from static_frame.core.util import GetItemKeyType
from static_frame.core.display import DisplayConfig
from static_frame.core.display import DisplayActive
from static_frame.core.display import Display
from static_frame.core.display import DisplayHeader



class Bus:

    __slots__ = (
        '_series',
        '_store'
        )

    _series: Series
    _store: Store


    @staticmethod
    def _empty_series(labels: tp.Iterable[str]):
        # make an object dtype
        return Series(None, index=labels, dtype=object)

    @classmethod
    def from_frames(cls, *frames) -> 'Bus':
        # TODO: fail if a name is None
        series = Series.from_items(
                    ((f.name, f) for f in frames),
                    dtype=object
                    )
        return cls(series)


    @classmethod
    def from_zip_tsv(cls, fp: str):
        store = StoreZipTSV(fp)
        return cls(cls._empty_series(store.labels()), store=store)

    @classmethod
    def from_zip_csv(cls, fp: str):
        store = StoreZipCSV(fp)
        return cls(cls._empty_series(store.labels()), store=store)


    #---------------------------------------------------------------------------
    def __init__(self,
            series: Series,
            *,
            store: tp.Optional[Store] = None
            ):

        for value in series.values:
            if not isinstance(value, Frame) and not value is None:
                raise ErrorInitBus(f'supplied {value.__class__} is not a frame')

        self._series = series
        self._store = store


    def _key_to_labels(self,
            key: GetItemKeyType
            ) -> tp.Iterator[int]:
        '''
        Given a get-item key, translate to an iterator of loc positions.
        '''
        # key maybe a selection, slice, or Boolean
        labels = self.index.values[self.index.loc_to_iloc(key)]
        if isinstance(labels, str): # single values
            return (labels,)
        return labels

    def __getattr__(self, name) -> tp.Any:
        return getattr(self._series, name)

    def __getitem__(self, key: GetItemKeyType) -> Series:

        if self._series.isna().any():
            labels_to_load = set(self._key_to_labels(key))

            def gen():
                for label, frame in self._series.items():
                    if frame is None and label in labels_to_load:
                        frame = self._store.read(label)
                    yield label, frame

            self._series = Series.from_items(gen(), dtype=object)

        return self._series.__getitem__(key)

    def __reversed__(self) -> tp.Iterator[tp.Hashable]:
        return reversed(self._series._index)

    def __len__(self) -> int:
        return self._series.__len__()

    #---------------------------------------------------------------------------
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


    #---------------------------------------------------------------------------
    def to_zip_tsv(self, fp) -> None:
        store = StoreZipTSV(fp)
        store.write(self.items())

    def to_zip_csv(self, fp) -> None:
        store = StoreZipCSV(fp)
        store.write(self.items())
