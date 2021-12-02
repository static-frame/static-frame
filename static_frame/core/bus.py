import typing as tp

import numpy as np

from static_frame.core.container import ContainerBase
from static_frame.core.display import Display
from static_frame.core.display import DisplayActive
from static_frame.core.display import DisplayHeader
from static_frame.core.display_config import DisplayConfig
from static_frame.core.doc_str import doc_inject
from static_frame.core.exception import ErrorInitBus
from static_frame.core.exception import ErrorInitIndexNonUnique
from static_frame.core.frame import Frame
from static_frame.core.index_auto import RelabelInput
from static_frame.core.index_base import IndexBase
from static_frame.core.node_iter import IterNodeNoArg
from static_frame.core.node_iter import IterNodeApplyType
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
from static_frame.core.store_zip import StoreZipNPZ
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
from static_frame.core.util import BoolOrBools
from static_frame.core.util import NAME_DEFAULT
from static_frame.core.util import IndexConstructor
from static_frame.core.style_config import StyleConfig
# from static_frame.core.index_auto import IndexAutoFactory
from static_frame.core.index_auto import IndexAutoFactoryType


#-------------------------------------------------------------------------------
class FrameDeferredMeta(type):
    def __repr__(cls) -> str:
        return f'<{cls.__name__}>'

class FrameDeferred(metaclass=FrameDeferredMeta):
    '''
    Token placeholder for :obj:`Frame` not yet loaded.
    '''

BusItemsType = tp.Iterable[tp.Tuple[
        tp.Hashable, tp.Union[Frame, tp.Type[FrameDeferred]]]]

FrameIterType = tp.Iterator[Frame]

#-------------------------------------------------------------------------------
class Bus(ContainerBase, StoreClientMixin): # not a ContainerOperand
    '''
    A randomly-accessible container of :obj:`Frame`. When created from a multi-table storage format (such as a zip-pickle or XLSX), a Bus will lazily read in components as they are accessed. When combined with the ``max_persist`` parameter, a Bus will not hold on to more than ``max_persist`` references, permitting low-memory reading of collections of :obj:`Frame`.
    '''

    __slots__ = (
        '_loaded',
        '_loaded_all',
        '_values_mutable',
        '_index',
        '_name',
        '_store',
        '_config',
        '_last_accessed',
        '_max_persist',
        )

    _values_mutable: np.ndarray
    _index: IndexBase
    _store: tp.Optional[Store]
    _config: StoreConfigMap
    _name: NameType

    STATIC = False
    _NDIM: int = 1

    @staticmethod
    def _deferred_series(
            labels: tp.Iterable[tp.Hashable],
            *,
            index_constructor: IndexConstructor = None,
            ) -> Series:
        '''
        Return an object ``Series`` of ``FrameDeferred`` objects, based on the passed in ``labels``.
        '''
        # NOTE: need to accept an  IndexConstructor to support reanimating Index subtypes, IH
        return Series.from_element(FrameDeferred,
                index=labels,
                dtype=DTYPE_OBJECT,
                index_constructor=index_constructor,
                )

    @classmethod
    def from_frames(cls,
            frames: tp.Iterable[Frame],
            *,
            index_constructor: IndexConstructor = None,
            config: StoreConfigMapInitializer = None,
            name: NameType = None,
            ) -> 'Bus':
        '''Return a :obj:`Bus` from an iterable of :obj:`Frame`; labels will be drawn from :obj:`Frame.name`.
        '''
        try:
            series = Series.from_items(
                        ((f.name, f) for f in frames),
                        dtype=DTYPE_OBJECT,
                        name=name,
                        index_constructor=index_constructor,
                        )
        except ErrorInitIndexNonUnique:
            raise ErrorInitIndexNonUnique("Frames do not have unique names.") from None

        return cls(series, config=config, own_data=True)

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
        return cls(series, config=config, own_data=True)

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
        return cls(series, config=config, own_data=True)

    @classmethod
    def from_concat(cls,
            containers: tp.Iterable['Bus'],
            *,
            index: tp.Optional[tp.Union[IndexInitializer, IndexAutoFactoryType]] = None,
            name: NameType = NAME_DEFAULT,
            ) -> 'Bus':
        '''
        Concatenate multiple :obj:`Bus` into a new :obj:`Bus`. All :obj:`Bus` will load all :obj:`Frame` into memory if any are deferred.
        '''
        # will extract .values, .index from Bus, which will correct load from Store as needed
        series = Series.from_concat(containers, index=index, name=name)
        return cls(series, own_data=True)

    @classmethod
    def from_series(cls,
            series: Series,
            *,
            store: tp.Optional[Store] = None,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            own_data: bool = False,
            ) -> 'Bus':
        '''
        Create a :obj:`Bus` from a :obj:`Series` of :obj:`Frame`.
        '''
        # NOTE: this interface is for 0.9 after the default Bus no longer accepts a Series
        return cls(series,
                store=store,
                config=config,
                max_persist=max_persist,
                own_data=own_data,
                )

    #---------------------------------------------------------------------------
    # constructors by data format
    @classmethod
    def _from_store(cls,
            store: Store,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            index_constructor: IndexConstructor = None,
            ) -> 'Bus':
        return cls(cls._deferred_series(
                        store.labels(config=config),
                        index_constructor=index_constructor,
                        ),
                store=store,
                config=config,
                max_persist=max_persist,
                own_data=True,
                )

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_zip_tsv(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            index_constructor: IndexConstructor = None,
            ) -> 'Bus':
        '''
        Given a file path to zipped TSV :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreZipTSV(fp)
        return cls._from_store(store,
                config=config,
                max_persist=max_persist,
                index_constructor=index_constructor,
                )

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_zip_csv(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            index_constructor: IndexConstructor = None,
            ) -> 'Bus':
        '''
        Given a file path to zipped CSV :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreZipCSV(fp)
        return cls._from_store(store,
                config=config,
                max_persist=max_persist,
                index_constructor=index_constructor,
                )

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_zip_pickle(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            index_constructor: IndexConstructor = None,
            ) -> 'Bus':
        '''
        Given a file path to zipped pickle :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreZipPickle(fp)
        return cls._from_store(store,
                config=config,
                max_persist=max_persist,
                index_constructor=index_constructor,
                )

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_zip_npz(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            index_constructor: IndexConstructor = None,
            ) -> 'Bus':
        '''
        Given a file path to zipped parquet :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreZipNPZ(fp)
        return cls._from_store(store,
                config=config,
                max_persist=max_persist,
                index_constructor=index_constructor,
                )

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_zip_parquet(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            index_constructor: IndexConstructor = None,
            ) -> 'Bus':
        '''
        Given a file path to zipped parquet :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreZipParquet(fp)
        return cls._from_store(store,
                config=config,
                max_persist=max_persist,
                index_constructor=index_constructor,
                )

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_xlsx(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            index_constructor: IndexConstructor = None,
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
                index_constructor=index_constructor,
                )

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_sqlite(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            index_constructor: IndexConstructor = None,
            ) -> 'Bus':
        '''
        Given a file path to an SQLite :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreSQLite(fp)
        return cls._from_store(store,
                config=config,
                max_persist=max_persist,
                index_constructor=index_constructor,
                )

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_hdf5(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            index_constructor: IndexConstructor = None,
            ) -> 'Bus':
        '''
        Given a file path to a HDF5 :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreHDF5(fp)
        return cls._from_store(store,
                config=config,
                max_persist=max_persist,
                index_constructor=index_constructor,
                )

    #---------------------------------------------------------------------------
    @doc_inject(selector='bus_init')
    def __init__(self,
            series: Series,
            *,
            store: tp.Optional[Store] = None,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            own_data: bool = False,
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

        if own_data:
            self._values_mutable = series.values
            self._values_mutable.flags.writeable = True
        else:
            self._values_mutable = series.values.copy()

        self._index = series._index
        self._name = series._name
        self._store = store

        # Not handling cases of max_persist being greater than the length of the Series (might floor to length)
        if max_persist is not None and max_persist < self._loaded.sum():
            raise ErrorInitBus('max_persist cannot be less than the number of already loaded Frames')
        self._max_persist = max_persist

        # providing None will result in default; providing a StoreConfig or StoreConfigMap will return an appropriate map
        self._config = StoreConfigMap.from_initializer(config)

    #---------------------------------------------------------------------------
    def _derive(self,
            series: Series,
            *,
            own_data: bool = False,
            ) -> 'Bus':
        '''Utility for creating a derived Bus, propagating the associated ``Store`` and configuration. This can be used if the passed `series` is a subset or re-ordering of self._series; however, if the index has been transformed, this method should not be used, as, if there is a Store, the labels are no longer found in that Store.
        '''
        return self.__class__(series,
                store=self._store,
                config=self._config,
                max_persist=self._max_persist,
                own_data=own_data,
                )

    # ---------------------------------------------------------------------------
    def __reversed__(self) -> tp.Iterator[tp.Hashable]:
        '''
        Returns a reverse iterator on the :obj:`Bus` index.

        Returns:
            :obj:`Index`
        '''
        return reversed(self._index) #type: ignore

    # def __copy__(self) -> 'Bus':
    #     '''
    #     Return a new Bus, holding new references to Frames as well as a link to the a new Store instance.
    #     '''
    #     return self.__class__(series,
    #             store=self._store.__copy__(),
    #             config=self._config,
    #             max_persiste=self._max_persist,
    #             )

    #---------------------------------------------------------------------------
    # name interface

    @property #type: ignore
    @doc_inject()
    def name(self) -> NameType:
        '''{}'''
        return self._name

    def rename(self, name: NameType) -> 'Bus':
        '''
        Return a new :obj:`Bus` with an updated name attribute.
        '''
        series = self._to_series_state().rename(name)
        return self._derive(series, own_data=True)

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
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.SERIES_VALUES,
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
                yield_type=IterNodeType.ITEMS,
                apply_type=IterNodeApplyType.SERIES_VALUES,
                )

    #---------------------------------------------------------------------------
    # index manipulation

    # NOTE: must return a new Bus with fully-realized Frames, as cannot gaurantee usage of a Store after labels have been changed.

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
        series = self.to_series().reindex(index,
                fill_value=fill_value,
                own_index=own_index,
                check_equals=check_equals,
                )
        return self.__class__(series, config=self._config)

    @doc_inject(selector='relabel', class_name='Bus')
    def relabel(self,
            index: tp.Optional[RelabelInput]
            ) -> 'Bus':
        '''
        {doc}

        Args:
            index: {relabel_input}
        '''
        series = self.to_series().relabel(index)
        return self.__class__(series, config=self._config)


    @doc_inject(selector='relabel_flat', class_name='Bus')
    def relabel_flat(self) -> 'Bus':
        '''
        {doc}
        '''
        series = self.to_series().relabel_flat()
        return self.__class__(series, config=self._config)

    @doc_inject(selector='relabel_level_add', class_name='Bus')
    def relabel_level_add(self,
            level: tp.Hashable
            ) -> 'Bus':
        '''
        {doc}

        Args:
            level: {level}
        '''
        series = self.to_series().relabel_level_add(level)
        return self.__class__(series, config=self._config)


    @doc_inject(selector='relabel_level_drop', class_name='Bus')
    def relabel_level_drop(self,
            count: int = 1
            ) -> 'Bus':
        '''
        {doc}

        Args:
            count: {count}
        '''
        series = self.to_series().relabel_level_drop(count)
        return self.__class__(series, config=self._config)

    def rehierarch(self,
            depth_map: tp.Sequence[int]
            ) -> 'Bus':
        '''
        Return a new :obj:`Bus` with new a hierarchy based on the supplied ``depth_map``.
        '''
        series = self.to_series().rehierarch(depth_map)
        return self.__class__(series, config=self._config)


    #---------------------------------------------------------------------------
    # na / falsy handling

    # NOTE: not implemented, as a Bus must contain only Frame or FrameDeferred

    #---------------------------------------------------------------------------
    # cache management

    @staticmethod
    def _store_reader(
            store: Store,
            config: StoreConfigMap,
            labels: tp.Iterator[tp.Hashable],
            max_persist: tp.Optional[int],
            ) -> FrameIterType:
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
               # try to collect max_persist-sized bundles in coll, then use read_many to get all at once, then clear if we have more to iter
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

        index = self._index
        if not load and max_persist_active: # must update LRU position
            labels = (index.iloc[key],) if isinstance(key, INT_TYPES) else index.iloc[key].values
            for label in labels: # update LRU position
                self._last_accessed[label] = self._last_accessed.pop(label, None)
            return

        if self._store is None: # there has to be a Store defined if we are partially loaded
            raise RuntimeError('no store defined')
        if max_persist_active:
            loaded_count = self._loaded.sum()

        array = self._values_mutable
        target_values = array[key]
        target_labels = self._index.iloc[key]
        # targets = self._series.iloc[key] # key is iloc key

        store_reader: FrameIterType
        targets_items: BusItemsType

        if not isinstance(target_values, np.ndarray):
            targets_items = ((target_labels, target_values),) # present element as items
            store_reader = (self._store.read(target_labels,
                    config=self._config[target_labels]) for _ in range(1))
        else: # more than one Frame
            store_reader = self._store_reader(
                    store=self._store,
                    config=self._config,
                    labels=(label for label, f in zip(target_labels, target_values)
                            if f is FrameDeferred),
                    max_persist=self._max_persist,
                    )
            targets_items = zip(target_labels, target_values)

        # Iterate over items that have been selected; there must be at least 1 FrameDeffered among this selection
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

        self._loaded_all = self._loaded.all()

    def unpersist(self) -> None:
        '''Replace loaded :obj:`Frame` with :obj:`FrameDeferred`.
        '''
        if self._store is None:
            # have this be a no-op so that Yarn or Quilt can call regardless of Store
            return

        if self._max_persist is not None:
            last_accessed = self._last_accessed
        else:
            last_accessed = dict.fromkeys(self.index)

        index = self._index
        array = self._values_mutable

        for label_remove in last_accessed:
            idx_remove = index._loc_to_iloc(label_remove)
            self._loaded[idx_remove] = False
            array[idx_remove] = FrameDeferred

        last_accessed.clear()
        self._loaded_all = False

    #---------------------------------------------------------------------------
    # extraction

    def _extract_iloc(self, key: GetItemKeyType) -> 'Bus':
        '''
        Returns:
            Bus or, if an element is selected, a Frame
        '''
        self._update_series_cache_iloc(key=key)

        # iterable selection should be handled by NP
        values = self._values_mutable[key]

        # NOTE: Bus only stores Frame and FrameDeferred, can rely on check with values
        if not values.__class__ is np.ndarray: # if we have a single element
            return values #type: ignore

        # values will be copied and made immutable
        series = Series(
                values,
                index=self._index.iloc[key],
                name=self._name,
                )
        return self._derive(series, own_data=True)

    def _extract_loc(self, key: GetItemKeyType) -> 'Bus':
        iloc_key = self._index._loc_to_iloc(key)
        return self._extract_iloc(iloc_key)

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
        series = self._to_series_state()._drop_iloc(key)
        return self._derive(series, own_data=True)

    def _drop_loc(self, key: GetItemKeyType) -> 'Bus':
        return self._drop_iloc(self._index._loc_to_iloc(key))

    #---------------------------------------------------------------------------
    # axis functions

    def _axis_element_items(self,
            ) -> tp.Iterator[tp.Tuple[tp.Hashable, Frame]]:
        '''Generator of index, value pairs, equivalent to Series.items(). Repeated to have a common signature as other axis functions.
        '''
        yield from self.items()

    def _axis_element(self,
            ) -> tp.Iterator[tp.Any]:
        if self._loaded_all:
            yield from self._values_mutable
        elif self._max_persist is None: # load all at once if possible
            if not self._loaded_all:
                self._update_series_cache_iloc(key=NULL_SLICE)
            yield from self._values_mutable
        elif self._max_persist > 1:
            i = 0
            i_max = len(self._index.values)
            while i < i_max:
                key = slice(i, min(i + self._max_persist, i_max))
                # draw values to force usage of read_many in _store_reader
                self._update_series_cache_iloc(key=key)
                for j in range(key.start, key.stop):
                    yield self._values_mutable[j]
                i += self._max_persist
        else: # max_persist is 1
            for i in range(self.__len__()):
                self._update_series_cache_iloc(key=i)
                yield self._values_mutable[i]

    #---------------------------------------------------------------------------
    # dictionary-like interface; these will force loading contained Frame

    def items(self) -> tp.Iterator[tp.Tuple[tp.Hashable, Frame]]:
        '''Iterator of pairs of :obj:`Bus` label and contained :obj:`Frame`.
        '''
        if self._loaded_all:
            yield from zip(self._index, self._values_mutable)
        elif self._max_persist is None: # load all at once if possible
            if not self._loaded_all:
                self._update_series_cache_iloc(key=NULL_SLICE)
            yield from zip(self._index, self._values_mutable)
        elif self._max_persist > 1:
            labels = self._index.values
            i = 0
            i_max = len(labels)
            while i < i_max:
                key = slice(i, min(i + self._max_persist, i_max))
                labels_select = labels[key] # may over select
                # draw values to force usage of read_many in _store_reader
                self._update_series_cache_iloc(key=key)
                yield from zip(labels_select, self._values_mutable[key])
                i += self._max_persist
        else: # max_persist is 1
            for i, label in enumerate(self._index.values):
                self._update_series_cache_iloc(key=i)
                yield label, self._values_mutable[i]

    _items_store = items

    @property
    def values(self) -> np.ndarray:
        '''A 1D object array of all :obj:`Frame` contained in the :obj:`Bus`. The returned ``np.ndarray`` will have ``Frame``; this will never return an array with ``FrameDeferred``, but ``max_persist`` will be observed in reading from the Store.
        '''
        # NOTE: when self._values_mutable is fully loaded, it could become immutable and avoid a copy

        if self._loaded_all:
            post = self._values_mutable.copy()
            post.flags.writeable = False
            return post

        if self._max_persist is None: # load all at once if possible
            # b._loaded_all must be False
            self._update_series_cache_iloc(key=NULL_SLICE)
            post = self._values_mutable.copy()
            post.flags.writeable = False
            return post

        # return a new array; force new iteration to account for max_persist
        post = np.empty(self.__len__(), dtype=object)

        if self._max_persist > 1:
            i = 0
            i_max = len(self._index.values)
            while i < i_max:
                key = slice(i, min(i + self._max_persist, i_max))
                # draw values to force usage of read_many in _store_reader
                self._update_series_cache_iloc(key=key)
                post[key] = self._values_mutable[key]
                i += self._max_persist
        else: # max_persist is 1
            for i in range(self.__len__()):
                self._update_series_cache_iloc(key=i)
                post[i] = self._values_mutable[i]

        post.flags.writeable = False
        return post

    #---------------------------------------------------------------------------
    def __len__(self) -> int:
        '''Length of values.
        '''
        return self._index.__len__()

    @doc_inject()
    def display(self,
            config: tp.Optional[DisplayConfig] = None,
            *,
            style_config: tp.Optional[StyleConfig] = None,
            ) -> Display:
        '''{doc}

        Args:
            {config}
        '''
        # NOTE: the key change over serires is providing the Bus as the displayed class
        config = config or DisplayActive.get()
        display_cls = Display.from_values((),
                header=DisplayHeader(self.__class__, self._name),
                config=config)
        return self._to_series_state()._display(config,
                display_cls=display_cls,
                style_config=style_config,
                )

    #---------------------------------------------------------------------------
    # extended discriptors; in general, these do not force loading Frame

    @property
    def mloc(self) -> Series:
        '''Returns a :obj:`Series` showing a tuple of memory locations within each loaded Frame.
        '''
        if not self._loaded.any():
            return Series.from_element(None, index=self._index)

        def gen() -> tp.Iterator[tp.Tuple[tp.Hashable, tp.Optional[tp.Tuple[int, ...]]]]:
            for label, f in zip(self._index, self._values_mutable):
                if f is FrameDeferred:
                    yield label, None
                else:
                    yield label, tuple(f.mloc)

        return Series.from_items(gen())

    @property
    def dtypes(self) -> Frame:
        '''Returns a :obj:`Frame` of dtype per column for all loaded Frames.
        '''
        if not self._loaded.any():
            return Frame(index=self._index)

        f = Frame.from_concat(
                frames=(f.dtypes for f in self._values_mutable if f is not FrameDeferred),
                fill_value=None,
                ).reindex(index=self._index, fill_value=None)
        return tp.cast(Frame, f)

    @property
    def shapes(self) -> Series:
        '''A :obj:`Series` describing the shape of each loaded :obj:`Frame`. Unloaded :obj:`Frame` will have a shape of None.

        Returns:
            :obj:`Series`
        '''
        values = (f.shape if f is not FrameDeferred else None for f in self._values_mutable)
        return Series(values, index=self._index, dtype=object, name='shape')

    @property
    def nbytes(self) -> int:
        '''Total bytes of data currently loaded in the Bus.
        '''
        return sum(f.nbytes if f is not FrameDeferred else 0 for f in self._values_mutable)

    @property
    def status(self) -> Frame:
        '''
        Return a :obj:`Frame` indicating loaded status, size, bytes, and shape of all loaded :obj:`Frame`.
        '''
        def gen() -> tp.Iterator[Series]:

            yield Series(self._loaded,
                    index=self._index,
                    dtype=DTYPE_BOOL,
                    name='loaded')

            for attr, dtype, missing in (
                    ('size', DTYPE_FLOAT_DEFAULT, np.nan),
                    ('nbytes', DTYPE_FLOAT_DEFAULT, np.nan),
                    ('shape', DTYPE_OBJECT, None)
                    ):

                values = (getattr(f, attr) if f is not FrameDeferred
                        else missing for f in self._values_mutable)
                yield Series(values, index=self._index, dtype=dtype, name=attr)

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
        return DTYPE_OBJECT

    @property
    def shape(self) -> tp.Tuple[int]:
        '''
        Return a tuple describing the shape of the underlying NumPy array.

        Returns:
            :obj:`Tuple[int]`
        '''
        return self._values_mutable.shape #type: ignore

    @property
    def ndim(self) -> int:
        '''
        Return the number of dimensions, which for a :obj:`Bus` is always 1.

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
        return self._values_mutable.size #type: ignore

    #---------------------------------------------------------------------------

    @property
    def index(self) -> IndexBase:
        '''
        The index instance assigned to this container.

        Returns:
            :obj:`Index`
        '''
        return self._index

    # @property
    # def _index(self) -> IndexBase:
    #     return self._series._index

    #---------------------------------------------------------------------------
    # dictionary-like interface

    def keys(self) -> IndexBase:
        '''
        Iterator of index labels.

        Returns:
            :obj:`Iterator[Hashable]`
        '''
        return self._index

    def __iter__(self) -> tp.Iterator[tp.Hashable]:
        '''
        Iterator of index labels, same as :obj:`static_frame.Series.keys`.

        Returns:
            :obj:`Iterator[Hashasble]`
        '''
        return self._index.__iter__()

    def __contains__(self, value: tp.Hashable) -> bool:
        '''
        Inclusion of value in index labels.

        Returns:
            :obj:`bool`
        '''
        return self._index.__contains__(value)

    def get(self, key: tp.Hashable,
            default: tp.Any = None,
            ) -> tp.Any:
        '''
        Return the value found at the index key, else the default if the key is not found.

        Returns:
            :obj:`Any`
        '''
        if key not in self._index:
            return default
        # will always return an element
        return self._extract_loc(key=key)

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
        if len(self._index) != len(other._index):
            return False

        if compare_name and self._name != other._name:
            return False

        if not self._index.equals(
                other._index,
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
            ascending: BoolOrBools = True,
            kind: str = DEFAULT_SORT_KIND,
            key: tp.Optional[tp.Callable[[IndexBase], tp.Union[np.ndarray, IndexBase]]] = None,
            ) -> 'Bus':
        '''
        Return a new Bus ordered by the sorted Index.

        Args:
            *
            {ascendings}
            {kind}
            {key}

        Returns:
            :obj:`Bus`
        '''
        series = self._to_series_state().sort_index(
                ascending=ascending,
                kind=kind,
                key=key,
                )
        return self._derive(series, own_data=True)

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
            {ascending}
            {kind}
            {key}

        Returns:
            :obj:`Bus`
        '''
        values = self.values # this will handle max_persist, but will deliver an array with all Frame loaded
        cfs = Series(values,
                index=self._index,
                own_index=True,
                name=self._name,
                )
        series = cfs.sort_values(
                ascending=ascending,
                kind=kind,
                key=key,
                )
        return self._derive(series, own_data=True)


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
        series = self._to_series_state().roll(shift=shift, include_index=include_index)
        return self._derive(series, own_data=True)

    def shift(self,
            shift: int,
            *,
            fill_value: tp.Any,
            ) -> 'Bus':
        '''Return a :obj:`Bus` with values shifted forward on the index (with a positive shift) or backward on the index (with a negative shift).

        Args:
            shift: Positive or negative integer shift.
            fill_value: Value to be used to fill data missing after the shift.

        Returns:
            :obj:`Bus`
        '''
        series = self._to_series_state().shift(shift=shift, fill_value=fill_value)
        return self._derive(series, own_data=True)

    #---------------------------------------------------------------------------
    # exporter

    def _to_series_state(self) -> Series:
        # the mutable array will be copied in the Series construction
        return Series(self._values_mutable,
                index=self._index,
                own_index=True,
                name=self._name,
                )

    def to_series(self) -> Series:
        '''Return a :obj:`Series` with the :obj:`Frame` contained in this :obj:`Bus`. If the :obj:`Bus` is associated with a :obj:`Store`, all :obj:`Frame` will be loaded into memory and the returned :obj:`Bus` will no longer be associated with the :obj:`Store`.
        '''
        # values returns an immutable array and will fully realize from Store
        return Series(self.values,
                index=self._index,
                own_index=True,
                name=self._name,
                )
