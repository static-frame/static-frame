
import typing as tp

import numpy as np

from static_frame.core.container import ContainerBase
from static_frame.core.display import Display
from static_frame.core.display import DisplayActive
from static_frame.core.display import DisplayHeader
from static_frame.core.display_config import DisplayConfig
from static_frame.core.doc_str import doc_inject
from static_frame.core.exception import ErrorInitBus
from static_frame.core.frame import Frame
from static_frame.core.index_auto import RelabelInput
from static_frame.core.index_base import IndexBase
from static_frame.core.node_iter import IterNodeNoArg
from static_frame.core.node_iter import IterNodeType
from static_frame.core.node_selector import InterfaceGetItem
from static_frame.core.node_selector import InterfaceSelectTrio
from static_frame.core.series import Series
from static_frame.core.store import Store
from static_frame.core.store import StoreConfigMap
from static_frame.core.store import StoreConfigMapInitializer
from static_frame.core.store_client_mixin import StoreClientMixin
from static_frame.core.store_hdf5 import StoreHDF5
from static_frame.core.store_sqlite import StoreSQLite
from static_frame.core.store_xlsx import StoreXLSX
from static_frame.core.store_zip import StoreZipCSV
from static_frame.core.store_zip import StoreZipParquet
from static_frame.core.store_zip import StoreZipPickle
from static_frame.core.store_zip import StoreZipTSV
from static_frame.core.util import DEFAULT_SORT_KIND
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import DTYPE_FLOAT_DEFAULT
from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import IndexInitializer
from static_frame.core.util import INT_TYPES
from static_frame.core.util import NameType
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import PathSpecifier

#-------------------------------------------------------------------------------
class FrameDeferredMeta(type):
    def __repr__(cls) -> str:
        return f'<{cls.__name__}>'

class FrameDeferred(metaclass=FrameDeferredMeta):
    '''
    Token placeholder for :obj:`Frame` not yet loaded.
    '''

#-------------------------------------------------------------------------------
class Bus(ContainerBase, StoreClientMixin): # not a ContainerOperand
    '''
    A randomly-accessible container of :obj:`Frame`. When created from a multi-table storage format (such as a zip-pickle or XLSX), a Bus will lazily read in components as they are accessed. When combined with the ``max_persist`` parameter, a Bus will not hold on to more than ``max_persist`` references, permitting low-memory reading of collections of :obj:`Frame`.
    '''

    __slots__ = (
        '_loaded',
        '_loaded_all',
        '_series',
        '_store',
        '_config',
        '_last_accessed',
        '_max_persist',
        )

    _series: Series
    _store: tp.Optional[Store]
    _config: StoreConfigMap
    _name: NameType

    STATIC = False
    _NDIM: int = 1

    @staticmethod
    def _deferred_series(labels: tp.Iterable[tp.Hashable]) -> Series:
        '''
        Return an object ``Series`` of ``FrameDeferred`` objects, based on the passed in ``labels``.
        '''
        # NOTE: need to accept an  IndexConstructor to support reanimating Index subtypes, IH
        return Series.from_element(FrameDeferred, index=labels, dtype=DTYPE_OBJECT)

    @classmethod
    def from_frames(cls,
            frames: tp.Iterable[Frame],
            *,
            config: StoreConfigMapInitializer = None,
            name: NameType = None,
            ) -> 'Bus':
        '''Return a :obj:`Bus` from an iterable of :obj:`Frame`; labels will be drawn from :obj:`Frame.name`.
        '''
        series = Series.from_items(
                    ((f.name, f) for f in frames),
                    dtype=DTYPE_OBJECT,
                    name=name,
                    )
        return cls(series, config=config)

    @classmethod
    def from_items(cls,
            pairs: tp.Iterable[tp.Tuple[tp.Hashable, Frame]],
            *,
            config: StoreConfigMapInitializer = None,
            name: NameType = None,
            index_constructor: tp.Optional[tp.Callable[..., IndexBase]] = None
            ) -> 'Bus':
        '''Return a :obj:`Bus` from an iterable of pairs of label, :obj:`Frame`.

        Returns:
            :obj:`Bus`
        '''
        series = Series.from_items(pairs,
                dtype=DTYPE_OBJECT,
                name=name,
                index_constructor=index_constructor,
                )
        return cls(series, config=config)

    @classmethod
    def from_dict(cls,
            mapping: tp.Dict[tp.Hashable, tp.Any],
            *,
            config: StoreConfigMapInitializer = None,
            name: NameType = None,
            index_constructor: tp.Optional[tp.Callable[..., IndexBase]] = None
            ) -> 'Bus':
        '''Bus construction from a dictionary, where the first pair value is the index and the second is the value.

        Args:
            mapping: a dictionary or similar mapping interface.
            dtype: dtype or valid dtype specifier.

        Returns:
            :obj:`Bus`
        '''
        series = Series.from_dict(mapping,
                dtype=DTYPE_OBJECT,
                name=name,
                index_constructor=index_constructor,
                )
        return cls(series, config=config)

    #---------------------------------------------------------------------------
    # constructors by data format
    @classmethod
    def _from_store(cls,
            store: Store,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            ) -> 'Bus':
        return cls(cls._deferred_series(store.labels(config=config)),
                store=store,
                config=config,
                max_persist=max_persist,
                )


    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_zip_tsv(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            ) -> 'Bus':
        '''
        Given a file path to zipped TSV :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreZipTSV(fp)
        return cls._from_store(store,
                config=config,
                max_persist=max_persist,
                )

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_zip_csv(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            ) -> 'Bus':
        '''
        Given a file path to zipped CSV :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreZipCSV(fp)
        return cls._from_store(store,
                config=config,
                max_persist=max_persist,
                )

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_zip_pickle(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            ) -> 'Bus':
        '''
        Given a file path to zipped pickle :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreZipPickle(fp)
        return cls._from_store(store,
                config=config,
                max_persist=max_persist,
                )

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_zip_parquet(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            ) -> 'Bus':
        '''
        Given a file path to zipped parquet :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreZipParquet(fp)
        return cls._from_store(store,
                config=config,
                max_persist=max_persist,
                )

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_xlsx(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            ) -> 'Bus':
        '''
        Given a file path to an XLSX :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        # how to pass configuration for multiple sheets?
        store = StoreXLSX(fp)
        return cls._from_store(store,
                config=config,
                max_persist=max_persist,
                )

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_sqlite(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            ) -> 'Bus':
        '''
        Given a file path to an SQLite :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreSQLite(fp)
        return cls._from_store(store,
                config=config,
                max_persist=max_persist,
                )

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_hdf5(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            ) -> 'Bus':
        '''
        Given a file path to a HDF5 :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreHDF5(fp)
        return cls._from_store(store,
                config=config,
                max_persist=max_persist,
                )

    #---------------------------------------------------------------------------
    @doc_inject(selector='bus_init')
    def __init__(self,
            series: Series,
            *,
            store: tp.Optional[Store] = None,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            ):
        '''
        Default Bus constructor.

        {args}
        '''

        if series.dtype != DTYPE_OBJECT:
            raise ErrorInitBus(
                    f'Series passed to initializer must have dtype object, not {series.dtype}')

        if max_persist is not None:
            # use an (ordered) dictionary to give use an ordered set, simply pointing to None for all keys
            self._last_accessed: tp.Dict[str, None] = {}

        # do a one time iteration of series
        def gen() -> tp.Iterator[bool]:
            for label, value in series.items():
                if isinstance(value, Frame):
                    if max_persist is not None:
                        self._last_accessed[label] = None
                    yield True
                elif value is FrameDeferred:
                    yield False
                else:
                    raise ErrorInitBus(f'supplied {value.__class__} is not a Frame or FrameDeferred.')

        self._loaded = np.fromiter(gen(), dtype=DTYPE_BOOL, count=len(series))
        self._loaded_all = self._loaded.all()
        self._series = series
        self._store = store

        # Not handling cases of max_persist being greater than the length of the Series (might floor to length)
        if max_persist is not None and max_persist < self._loaded.sum():
            raise ErrorInitBus('max_persis cannot be less than the number of already loaded Frames')
        self._max_persist = max_persist

        # providing None will result in default; providing a StoreConfig or StoreConfigMap will return an appropriate map
        self._config = StoreConfigMap.from_initializer(config)

    #---------------------------------------------------------------------------
    def _derive(self,
            series: Series,
            ) -> 'Bus':
        '''Utility for creating derived Bus
        '''
        return self.__class__(series,
                store=self._store,
                config=self._config,
                max_persist=self._max_persist,
                )

    # ---------------------------------------------------------------------------
    def __reversed__(self) -> tp.Iterator[tp.Hashable]:
        '''
        Returns a reverse iterator on the series' index.

        Returns:
            :obj:`Index`
        '''
        return reversed(self._series._index) #type: ignore

    #---------------------------------------------------------------------------
    # name interface

    @property #type: ignore
    @doc_inject()
    def name(self) -> NameType:
        '''{}'''
        return self._series._name

    def rename(self, name: NameType) -> 'Bus':
        '''
        Return a new Series with an updated name attribute.
        '''
        series = self._series.rename(name)
        return self._derive(series)

    #---------------------------------------------------------------------------
    # interfaces

    @property
    def loc(self) -> InterfaceGetItem['Bus']:
        return InterfaceGetItem(self._extract_loc)

    @property
    def iloc(self) -> InterfaceGetItem['Bus']:
        return InterfaceGetItem(self._extract_iloc)

    @property
    def drop(self) -> InterfaceSelectTrio['Bus']:
        '''
        Interface for dropping elements from :obj:`static_frame.Bus`.
        '''
        return InterfaceSelectTrio( #type: ignore
                func_iloc=self._drop_iloc,
                func_loc=self._drop_loc,
                func_getitem=self._drop_loc
                )

    #---------------------------------------------------------------------------
    @property
    def iter_element(self) -> IterNodeNoArg['Bus']:
        '''
        Iterator of elements.
        '''
        return IterNodeNoArg(
                container=self,
                function_items=self._axis_element_items,
                function_values=self._axis_element,
                yield_type=IterNodeType.VALUES
                )

    @property
    def iter_element_items(self) -> IterNodeNoArg['Bus']:
        '''
        Iterator of label, element pairs.
        '''
        return IterNodeNoArg(
                container=self,
                function_items=self._axis_element_items,
                function_values=self._axis_element,
                yield_type=IterNodeType.ITEMS
                )

    #---------------------------------------------------------------------------
    # index manipulation

    @doc_inject(selector='reindex', class_name='Bus')
    def reindex(self,
            index: IndexInitializer,
            *,
            fill_value: tp.Any,
            own_index: bool = False,
            check_equals: bool = True
            ) -> 'Bus':
        '''
        {doc}

        Args:
            index: {index_initializer}
            columns: {index_initializer}
            {fill_value}
            {own_index}
        '''
        series = self._series.reindex(index,
                fill_value=fill_value,
                own_index=own_index,
                check_equals=check_equals,
                )
        return self._derive(series)

    @doc_inject(selector='relabel', class_name='Bus')
    def relabel(self,
            index: tp.Optional[RelabelInput]
            ) -> 'Bus':
        '''
        {doc}

        Args:
            index: {relabel_input}
        '''
        series = self._series.relabel(index)
        return self._derive(series)

    @doc_inject(selector='relabel_flat', class_name='Bus')
    def relabel_flat(self) -> 'Bus':
        '''
        {doc}
        '''
        series = self._series.relabel_flat()
        return self._derive(series)

    @doc_inject(selector='relabel_level_add', class_name='Bus')
    def relabel_level_add(self,
            level: tp.Hashable
            ) -> 'Bus':
        '''
        {doc}

        Args:
            level: {level}
        '''
        series = self._series.relabel_level_add(level)
        return self._derive(series)

    @doc_inject(selector='relabel_level_drop', class_name='Bus')
    def relabel_level_drop(self,
            count: int = 1
            ) -> 'Bus':
        '''
        {doc}

        Args:
            count: {count}
        '''
        series = self._series.relabel_level_drop(count)
        return self._derive(series)

    def rehierarch(self,
            depth_map: tp.Sequence[int]
            ) -> 'Bus':
        '''
        Return a new :obj:`Bus` with new a hierarchy based on the supplied ``depth_map``.
        '''
        series = self._series.rehierarch(depth_map)
        return self._derive(series)

    #---------------------------------------------------------------------------
    # na handling

    # NOTE: not implemented, as a Bus must contain only Frame or FrameDeferred

    #---------------------------------------------------------------------------
    # cache management

    def _iloc_to_labels(self,
            key: GetItemKeyType
            ) -> np.ndarray:
        '''
        Given a get-item key, translate to an iterator of loc positions.
        '''
        if isinstance(key, int):
            return [self.index.values[key],] # needs to be a list for usage in loc assignment
        return self.index.values[key]

    @staticmethod
    def _store_reader(
            store: Store,
            config: StoreConfigMap,
            labels: tp.Iterator[tp.Hashable],
            max_persist: tp.Optional[int],
            ) -> tp.Iterator[Frame]:
        '''
        Read as many labels as possible from Store, then yield back each one at a time. If max_persist is active, max_persist will set the maximum number of Frame to load per read. Using Store.read_many is shown to have significant performance benefits on large collections of Frame.
        '''
        if max_persist is None:
            for frame in store.read_many(labels, config=config):
                yield frame
        elif max_persist > 1:
            coll = []
            for label in labels:
                coll.append(label)
                if len(coll) == max_persist:
                    for frame in store.read_many(coll, config=config):
                        yield frame
                    coll.clear()
            if coll: # less than max persist remaining
                for frame in store.read_many(coll, config=config):
                    yield frame
        else: # max persist is 1
            for label in labels:
                yield store.read(label, config=config[labels])


    def _update_series_cache_iloc(self, key: GetItemKeyType) -> None:
        '''
        Update the Series cache with the key specified, where key can be any iloc GetItemKeyType.

        Args:
            key: always an iloc key.
        '''
        max_persist_active = self._max_persist is not None

        load = False if self._loaded_all else not self._loaded[key].all()
        if not load and not max_persist_active:
            return

        index = self._series.index
        if not load and max_persist_active: # must update LRU position
            labels = (index.iloc[key],) if isinstance(key, INT_TYPES) else index.iloc[key].values
            for label in labels: # update LRU position
                self._last_accessed[label] = self._last_accessed.pop(label, None)
            return

        if self._store is None: # there has to be a Store defined if we are partially loaded
            raise RuntimeError('no store defined')
        if max_persist_active:
            loaded_count = self._loaded.sum()

        array = self._series.values.copy() # not a deepcopy
        targets = self._series.iloc[key] # key is iloc key

        if not isinstance(targets, Series):
            label = index[key] #type: ignore [unreachable]
            targets_items = ((label, targets),) # present element as items
            store_reader = (self._store.read(label, config=self._config[label]) for _ in  range(1))
        else: # more than one Frame
            store_reader = self._store_reader(
                    store=self._store,
                    config=self._config,
                    labels=(label for label, f in targets.items() if f is FrameDeferred),
                    max_persist=self._max_persist,
                    )
            targets_items = targets.items()

        for label, frame in targets_items:
            idx = index._loc_to_iloc(label)

            if max_persist_active: # update LRU position
                self._last_accessed[label] = self._last_accessed.pop(label, None)

            if frame is FrameDeferred:
                frame = next(store_reader)

            if not self._loaded[idx]:
                # as we are iterating from `targets`, we might be holding on to references of Frames that we already removed in `array`; in this case we do not need to `read`, but we still need to update the new array
                array[idx] = frame
                self._loaded[idx] = True # update loaded status
                if max_persist_active:
                    loaded_count += 1

            if max_persist_active and loaded_count > self._max_persist:
                label_remove = next(iter(self._last_accessed))
                del self._last_accessed[label_remove]
                idx_remove = index._loc_to_iloc(label_remove)
                self._loaded[idx_remove] = False
                array[idx_remove] = FrameDeferred
                loaded_count -= 1

        array.flags.writeable = False
        self._series = Series(array,
                index=self._series._index,
                dtype=object,
                own_index=True,
                )
        self._loaded_all = self._loaded.all()

    #---------------------------------------------------------------------------
    # extraction

    def _extract_iloc(self, key: GetItemKeyType) -> 'Bus':
        '''
        Returns:
            Bus or, if an element is selected, a Frame
        '''
        self._update_series_cache_iloc(key=key)

        # iterable selection should be handled by NP
        values = self._series.values[key]

        if not values.__class__ is np.ndarray: # if we have a single element
            return values #type: ignore

        series = Series(
                values,
                index=self._series._index.iloc[key],
                name=self._series._name)

        return self._derive(series)

    def _extract_loc(self, key: GetItemKeyType) -> 'Bus':

        iloc_key = self._series._index._loc_to_iloc(key)

        # NOTE: if we update before slicing, we change the local and the object handed back
        self._update_series_cache_iloc(key=iloc_key)

        values = self._series.values[iloc_key]

        if not values.__class__ is np.ndarray: # if we have a single element
            return values #type: ignore

        series = Series(values,
                index=self._series._index.iloc[iloc_key],
                own_index=True,
                name=self._series._name)

        return self._derive(series)


    @doc_inject(selector='selector')
    def __getitem__(self, key: GetItemKeyType) -> 'Bus':
        '''Selector of values by label.

        Args:
            key: {key_loc}
        '''
        return self._extract_loc(key)

    #---------------------------------------------------------------------------
    # utilities for alternate extraction: drop

    def _drop_iloc(self, key: GetItemKeyType) -> 'Bus':
        series = self._series._drop_iloc(key)
        return self._derive(series)

    def _drop_loc(self, key: GetItemKeyType) -> 'Bus':
        return self._drop_iloc(self._series._index._loc_to_iloc(key))

    #---------------------------------------------------------------------------
    # axis functions

    def _axis_element_items(self,
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, tp.Any]]:
        '''Generator of index, value pairs, equivalent to Series.items(). Repeated to have a common signature as other axis functions.
        '''
        yield from zip(self._series._index, self._series.values)

    def _axis_element(self,
            ) -> tp.Iterator[tp.Any]:
        yield from self._series.values

    #---------------------------------------------------------------------------
    # dictionary-like interface; these will force loadings contained Frame

    def items(self) -> tp.Iterator[tp.Tuple[tp.Hashable, Frame]]:
        '''Iterator of pairs of :obj:`Bus` label and contained :obj:`Frame`.
        '''
        if self._max_persist is None: # load all at once if possible
            if not self._loaded_all:
                self._update_series_cache_iloc(key=NULL_SLICE)
            yield from self._series.items()

        else: # force new iteration to account for max_persist
            for i, label in enumerate(self._series._index):
                yield label, self._extract_iloc(i) #type: ignore

    _items_store = items

    @property
    def values(self) -> np.ndarray:
        '''A 1D object array of all Frame contained in the Bus.
        '''
        if self._max_persist is None: # load all at once if possible
            if not self._loaded_all:
                self._update_series_cache_iloc(key=NULL_SLICE)
            return self._series.values

        # force new iteration to account for max_persist
        post = np.empty(self.__len__(), dtype=object)
        for i, _ in enumerate(self._series._index):
            post[i] = self._extract_iloc(i)
        post.flags.writeable = False

        return post

    #---------------------------------------------------------------------------
    def __len__(self) -> int:
        '''Length of values.
        '''
        return self._series.__len__()

    @doc_inject()
    def display(self,
            config: tp.Optional[DisplayConfig] = None
            ) -> Display:
        '''{doc}

        Args:
            {config}
        '''
        # NOTE: the key change over serires is providing the Bus as the displayed class
        config = config or DisplayActive.get()
        display_cls = Display.from_values((),
                header=DisplayHeader(self.__class__, self._series._name),
                config=config)
        return self._series._display(config, display_cls)

    #---------------------------------------------------------------------------
    # extended discriptors; in general, these do not force loading Frame

    @property
    def mloc(self) -> Series:
        '''Returns a Series of tuples of dtypes, one for each loaded Frame.
        '''
        if not self._loaded.any():
            return Series.from_element(None, index=self._series._index)

        def gen() -> tp.Iterator[tp.Tuple[tp.Hashable, tp.Optional[tp.Tuple[int, ...]]]]:
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
        if not self._loaded.any():
            return Frame(index=self._series.index)

        f = Frame.from_concat(
                frames=(f.dtypes for f in self._series.values if f is not FrameDeferred),
                fill_value=None,
                ).reindex(index=self._series.index, fill_value=None)
        return tp.cast(Frame, f)

    @property
    def shapes(self) -> Series:
        '''A :obj:`Series` describing the shape of each loaded :obj:`Frame`. Unloaded :obj:`Frame` will have a shape of None.

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
        Return a :obj:`Frame` indicating loaded status, size, bytes, and shape of all loaded :obj:`Frame`.
        '''
        def gen() -> tp.Iterator[Series]:

            yield Series(self._loaded,
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

        return tp.cast(Frame, Frame.from_concat(gen(), axis=1))


    #---------------------------------------------------------------------------
    # common attributes from the numpy array

    @property
    def dtype(self) -> np.dtype:
        '''
        Return the dtype of the underlying NumPy array.

        Returns:
            :obj:`numpy.dtype`
        '''
        return self._series.values.dtype

    @property
    def shape(self) -> tp.Tuple[int]:
        '''
        Return a tuple describing the shape of the underlying NumPy array.

        Returns:
            :obj:`Tuple[int]`
        '''
        return self._series.values.shape #type: ignore

    @property
    def ndim(self) -> int:
        '''
        Return the number of dimensions, which for a `Series` is always 1.

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
        return self._series.values.size #type: ignore

    #---------------------------------------------------------------------------

    @property
    def index(self) -> IndexBase:
        '''
        The index instance assigned to this container.

        Returns:
            :obj:`Index`
        '''
        return self._series._index

    @property
    def _index(self) -> IndexBase:
        return self._series._index

    #---------------------------------------------------------------------------
    # dictionary-like interface

    def keys(self) -> IndexBase:
        '''
        Iterator of index labels.

        Returns:
            :obj:`Iterator[Hashable]`
        '''
        return self._series._index

    def __iter__(self) -> tp.Iterator[tp.Hashable]:
        '''
        Iterator of index labels, same as :obj:`static_frame.Series.keys`.

        Returns:
            :obj:`Iterator[Hashasble]`
        '''
        return self._series._index.__iter__()

    def __contains__(self, value: tp.Hashable) -> bool:
        '''
        Inclusion of value in index labels.

        Returns:
            :obj:`bool`
        '''
        return self._series._index.__contains__(value)

    def get(self, key: tp.Hashable,
            default: tp.Any = None,
            ) -> tp.Any:
        '''
        Return the value found at the index key, else the default if the key is not found.

        Returns:
            :obj:`Any`
        '''
        if key not in self._series._index:
            return default
        return self._series.__getitem__(key)

    #---------------------------------------------------------------------------
    @doc_inject()
    def equals(self,
            other: tp.Any,
            *,
            compare_name: bool = False,
            compare_dtype: bool = False,
            compare_class: bool = False,
            skipna: bool = True,
            ) -> bool:
        '''
        {doc}

        Note: this will attempt to load and compare all Frame managed by the Bus.

        Args:
            {compare_name}
            {compare_dtype}
            {compare_class}
            {skipna}
        '''

        if id(other) == id(self):
            return True

        if compare_class and self.__class__ != other.__class__:
            return False
        elif not isinstance(other, Bus):
            return False

        # NOTE: dtype self._series is always object
        if len(self._series) != len(other._series):
            return False

        if compare_name and self._series._name != other._series._name:
            return False

        if not self._series.index.equals(
                other._series.index,
                compare_name=compare_name,
                compare_dtype=compare_dtype,
                compare_class=compare_class,
                skipna=skipna,
                ):
            return False

        # can zip because length of Series already match
        # using .values will force loading all Frame into memory; better to use items() to permit collection
        for (_, frame_self), (_, frame_other) in zip(self.items(), other.items()):
            if not frame_self.equals(frame_other,
                    compare_name=compare_name,
                    compare_dtype=compare_dtype,
                    compare_class=compare_class,
                    skipna=skipna,
                    ):
                return False

        return True

    #---------------------------------------------------------------------------
    # transformations resulting in changed dimensionality

    @doc_inject(selector='head', class_name='Bus')
    def head(self, count: int = 5) -> 'Bus':
        '''{doc}

        Args:
            {count}

        Returns:
            :obj:`Bus`
        '''
        return self.iloc[:count]

    @doc_inject(selector='tail', class_name='Bus')
    def tail(self, count: int = 5) -> 'Bus':
        '''{doc}s

        Args:
            {count}

        Returns:
            :obj:`Bus`
        '''
        return self.iloc[-count:]


    #---------------------------------------------------------------------------
    # transformations resulting in the same dimensionality

    @doc_inject(selector='sort')
    def sort_index(self,
            *,
            ascending: bool = True,
            kind: str = DEFAULT_SORT_KIND,
            key: tp.Optional[tp.Callable[[IndexBase], tp.Union[np.ndarray, IndexBase]]] = None,
            ) -> 'Bus':
        '''
        Return a new Bus ordered by the sorted Index.

        Args:
            *
            ascending: {ascending}
            kind: {kind}
            key: {key}

        Returns:
            :obj:`Bus`
        '''
        series = self._series.sort_index(
                ascending=ascending,
                kind=kind,
                key=key,
                )
        return self._derive(series)

    @doc_inject(selector='sort')
    def sort_values(self,
            *,
            ascending: bool = True,
            kind: str = DEFAULT_SORT_KIND,
            key: tp.Callable[['Series'], tp.Union[np.ndarray, 'Series']],
            ) -> 'Bus':
        '''
        Return a new Bus ordered by the sorted values. Note that as a Bus contains Frames, a `key` argument must be provided to extract a sortable value, and this key function will process a :obj:`Series` of :obj:`Frame`.

        Args:
            *
            ascending: {ascending}
            kind: {kind}
            key: {key}

        Returns:
            :obj:`Bus`
        '''
        values = self.values # this will handle max_persist, but will deliver an array with all Frame loaded
        cfs = Series(values,
                index=self._series.index,
                own_index=True,
                name=self._series.name,
                )

        series = cfs.sort_values(
                ascending=ascending,
                kind=kind,
                key=key,
                )

        return self._derive(series)


    def roll(self,
            shift: int,
            *,
            include_index: bool = False,
            ) -> 'Bus':
        '''Return a Bus with values rotated forward and wrapped around the index (with a positive shift) or backward and wrapped around the index (with a negative shift).

        Args:
            shift: Positive or negative integer shift.
            include_index: Determine if the Index is shifted with the underlying data.

        Returns:
            :obj:`Bus`
        '''
        series = self._series.roll(shift=shift, include_index=include_index)
        return self._derive(series)

    def shift(self,
            shift: int,
            *,
            fill_value: tp.Any,
            ) -> 'Bus':
        '''Return a Bus with values shifted forward on the index (with a positive shift) or backward on the index (with a negative shift).

        Args:
            shift: Positive or negative integer shift.
            fill_value: Value to be used to fill data missing after the shift.

        Returns:
            :obj:`Bus`
        '''
        series = self._series.shift(shift=shift, fill_value=fill_value)
        return self._derive(series)

