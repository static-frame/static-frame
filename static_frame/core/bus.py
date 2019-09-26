
import typing as tp

import numpy as np # type: ignore


from static_frame.core.series import Series
from static_frame.core.frame import Frame
# from static_frame.core.frame import Index

from static_frame.core.store import Store
from static_frame.core.store import StoreZipCSV
from static_frame.core.store import StoreZipTSV
from static_frame.core.store import StoreZipPickle


from static_frame.core.exception import ErrorInitBus
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import DTYPE_BOOL
# from static_frame.core.util import DTYPE_INT_DEFAULT
from static_frame.core.util import DTYPE_FLOAT_DEFAULT

from static_frame.core.util import PathSpecifier


from static_frame.core.display import DisplayConfig
from static_frame.core.display import DisplayActive
from static_frame.core.display import Display
from static_frame.core.display import DisplayHeader

from static_frame.core.container import ContainerBase


class FrameDefferedMeta(type):
    def __repr__(cls):
        return f'<{cls.__name__}>'

class FrameDeferred(metaclass=FrameDefferedMeta):
    '''
    Token placeholder for :obj:`Frame` not yet loaded.
    '''
    pass


class Bus(ContainerBase):

    __slots__ = (
        '_series',
        '_store'
        )

    _series: Series
    _store: Store


    @staticmethod
    def _empty_series(labels: tp.Iterable[str]):
        # make an object dtype
        return Series(FrameDeferred, index=labels, dtype=object)

    @classmethod
    def from_frames(cls, frames: tp.Iterable[Frame]) -> 'Bus':
        '''Return a ``Bus`` from an iterable of ``Frame``; labels will be drawn from :obj:`Frame.name`.
        '''
        series = Series.from_items(
                    ((f.name, f) for f in frames),
                    dtype=object
                    )
        return cls(series)

    #---------------------------------------------------------------------------
    # constructors by data format

    @classmethod
    def from_zip_tsv(cls, fp: PathSpecifier):
        store = StoreZipTSV(fp)
        return cls(cls._empty_series(store.labels()), store=store)

    @classmethod
    def from_zip_csv(cls, fp: PathSpecifier):
        store = StoreZipCSV(fp)
        return cls(cls._empty_series(store.labels()), store=store)

    @classmethod
    def from_zip_pickle(cls, fp: PathSpecifier):
        store = StoreZipPickle(fp)
        return cls(cls._empty_series(store.labels()), store=store)


    #---------------------------------------------------------------------------
    def __init__(self,
            series: Series,
            *,
            store: tp.Optional[Store] = None
            ):

        if series.dtype != DTYPE_OBJECT:
            raise ErrorInitBus(
                    f'Series passed to initializer must have dtype object, not {series.dtype}')
        for value in series.values:
            if not isinstance(value, Frame) and not value is FrameDeferred:
                raise ErrorInitBus(f'supplied {value.__class__} is not a frame')
        # labels need to be stings, do an explicit check

        self._series = series
        self._store = store

    #---------------------------------------------------------------------------
    # delegation

    def __getattr__(self, name) -> tp.Any:
        if name == 'interface':
            return getattr(self.__class__, 'interface')

        return getattr(self._series, name)

    #---------------------------------------------------------------------------
    # cache management

    def _key_to_labels(self,
            key: GetItemKeyType
            ) -> tp.Iterable[tp.Hashable]:
        '''
        Given a get-item key, translate to an iterator of loc positions.
        '''
        # key may be a selection, slice, or Boolean
        iloc_key = self.index.loc_to_iloc(key)
        if isinstance(iloc_key, int):
            return [self.index.values[iloc_key],] # needs to be a list for usage in loc assignment
        return self.index.values[iloc_key]

    def _cache_not_complete(self) -> bool:
        # return (self._series == FrameDeferred).any()
        for v in self._series.values:
            if v is FrameDeferred:
                return True
        return False

    def _cache_all_incomplete(self) -> bool:
        # return (self._series == FrameDeferred).all()
        for v in self._series.values:
            if v is not FrameDeferred:
                return False
        return True


    def _update_series_cache(self, key: GetItemKeyType) -> None:
        '''
        Update the Series cache with the key specified, where key can be any GetItemKeyType.
        '''
        # if any of the Series values are not loaded
        if self._cache_not_complete():

            # iterate and filter
            labels_to_load = set(self._key_to_labels(key))

            array = np.empty(shape=len(self._index), dtype=object)
            for idx, (label, frame) in enumerate(self._series.items()):
                if frame is FrameDeferred and label in labels_to_load:
                    frame = self._store.read(label)
                array[idx] = frame
            array.flags.writeable = False
            self._series = Series(array, index=self._index, dtype=object)


            # # alt implementation using assignment; assume that this does not copy
            # index_assign = self._key_to_labels(key)

            # # pre-allocate array and assign in a loop
            # values_assign = np.empty(len(index_assign), dtype=object)
            # for idx, label in enumerate(index_assign):
            #     values_assign[idx] = self._store.read(label)
            # values_assign.flags.writeable = False

            # insert = Series(values_assign,
            #         index=index_assign,
            #         )

            # self._series = self._series.assign[index_assign](insert)


    def _update_series_cache_all(self):
        '''Load all Tables contained in this Bus.
        '''
        # import ipdb; ipdb.set_trace()
        if self._cache_not_complete():

            array = np.empty(shape=len(self._index), dtype=object)
            for idx, (label, frame) in enumerate(self._series.items()):
                if frame is FrameDeferred:
                    frame = self._store.read(label)
                array[idx] = frame
            array.flags.writeable = False
            self._series = Series(array, index=self._index, dtype=object)


            # def gen():
            #     for label, frame in self._series.items():
            #         if frame is FrameDeferred:
            #             frame = self._store.read(label)
            #         yield label, frame
            # self._series = Series.from_items(gen(), dtype=object)


    #---------------------------------------------------------------------------
    def __getitem__(self, key: GetItemKeyType) -> Series:
        self._update_series_cache(key=key)
        return self._series.__getitem__(key)

    def __reversed__(self) -> tp.Iterator[tp.Hashable]:
        return reversed(self._series._index)

    def __len__(self) -> int:
        return self._series.__len__()


    #---------------------------------------------------------------------------
    # dictionary-like interface

    def items(self) -> tp.Iterator[tp.Tuple[tp.Any, tp.Any]]:
        '''Iterator of pairs of index label and value.
        '''
        self._update_series_cache_all()
        yield from self._series.items()


    #---------------------------------------------------------------------------
    def display(self,
            config: tp.Optional[DisplayConfig] = None
            ) -> Display:
        '''Return a Display of the Bus.
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

    #---------------------------------------------------------------------------
    # extended disciptors

    @property
    def mloc(self) -> Series:
        '''Returns a Series of tuples of dtypes, one for each loaded Frame.
        '''
        if self._cache_all_incomplete():
            return Series(None, index=self._series._index)

        def gen() -> tp.Iterator[tp.Tuple[str, tp.Optional[tp.Tuple[int, ...]]]]:
            for label, f in zip(self._series._index, self._series.values):
                if f is FrameDeferred:
                    yield label, None
                else:
                    yield label, tuple(f.mloc)
        return Series.from_items(gen())

    @property
    def dtypes(self) -> Frame:
        '''Returns a Frame of dtypes for all loaded Frames.
        '''
        if self._cache_all_incomplete():
            return Frame(index=self._series.index)

        f = Frame.from_concat(
                frames=(f.dtypes for f in self._series.values if f is not FrameDeferred),
                fill_value=None,
                ).reindex(index=self._series.index, fill_value=None)
        return f

    @property
    def shapes(self) -> Series:
        '''A :obj:`Series` describing the shape of each loaded :obj:`Frame`.

        Returns:
            :obj:`tp.Tuple[int]`
        '''
        values = (f.shape if f is not FrameDeferred else None for f in self._series.values)
        return Series(values, index=self._series._index, dtype=object, name='shape')


    @property
    def nbytes(self) -> int:
        '''Total bytes of data currently loaded in the Bus.
        '''
        return sum(f.nbytes if f is not FrameDeferred else 0 for f in self._series.values)

    @property
    def status(self) -> Frame:
        '''
        Return a
        '''
        def gen() -> Series:

            yield Series((False if f is FrameDeferred else True for f in self._series.values),
                    index=self._series._index,
                    dtype=DTYPE_BOOL,
                    name='loaded')

            for attr, dtype, missing in (
                    ('size', DTYPE_FLOAT_DEFAULT, np.nan),
                    ('nbytes', DTYPE_FLOAT_DEFAULT, np.nan),
                    ('shape', DTYPE_OBJECT, None)
                    ):

                values = (getattr(f, attr) if f is not FrameDeferred
                        else missing for f in self._series.values)
                yield Series(values, index=self._series._index, dtype=dtype, name=attr)

        return Frame.from_concat(gen(), axis=1)


    #---------------------------------------------------------------------------
    def to_zip_tsv(self, fp) -> None:
        store = StoreZipTSV(fp)
        store.write(self.items())

    def to_zip_csv(self, fp) -> None:
        store = StoreZipCSV(fp)
        store.write(self.items())

    def to_zip_pickle(self, fp) -> None:
        store = StoreZipPickle(fp)
        store.write(self.items())
