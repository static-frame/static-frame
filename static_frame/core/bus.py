from __future__ import annotations

from itertools import chain
from itertools import zip_longest

import numpy as np
import typing_extensions as tp

from static_frame.core.container import ContainerBase
from static_frame.core.container_util import index_from_optional_constructor
from static_frame.core.container_util import iter_component_signature_bytes
from static_frame.core.display import Display
from static_frame.core.display import DisplayActive
from static_frame.core.display import DisplayHeader
from static_frame.core.display_config import DisplayConfig
from static_frame.core.doc_str import doc_inject
from static_frame.core.exception import ErrorInitBus
from static_frame.core.exception import ErrorInitIndexNonUnique
from static_frame.core.frame import Frame
from static_frame.core.index import Index
from static_frame.core.index_auto import TIndexAutoFactory
from static_frame.core.index_auto import TRelabelInput
from static_frame.core.index_base import IndexBase
from static_frame.core.node_iter import IterNodeApplyType
from static_frame.core.node_iter import IterNodeNoArgReducible
from static_frame.core.node_selector import InterfaceSelectTrio
from static_frame.core.node_selector import InterGetItemILocReduces
from static_frame.core.node_selector import InterGetItemLocReduces
from static_frame.core.series import Series
from static_frame.core.store import Store
from static_frame.core.store_client_mixin import StoreClientMixin
from static_frame.core.store_config import StoreConfigMap
from static_frame.core.store_config import StoreConfigMapInitializer
from static_frame.core.store_duckdb import StoreDuckDB
from static_frame.core.store_hdf5 import StoreHDF5
from static_frame.core.store_sqlite import StoreSQLite
from static_frame.core.store_xlsx import StoreXLSX
from static_frame.core.store_zip import StoreZipCSV
from static_frame.core.store_zip import StoreZipNPY
from static_frame.core.store_zip import StoreZipNPZ
from static_frame.core.store_zip import StoreZipParquet
from static_frame.core.store_zip import StoreZipPickle
from static_frame.core.store_zip import StoreZipTSV
from static_frame.core.style_config import StyleConfig
from static_frame.core.util import DEFAULT_SORT_KIND
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import DTYPE_FLOAT_DEFAULT
from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import INT_TYPES
from static_frame.core.util import NAME_DEFAULT
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import ZIP_LONGEST_DEFAULT
from static_frame.core.util import IterNodeType
from static_frame.core.util import TBoolOrBools
from static_frame.core.util import TILocSelector
from static_frame.core.util import TIndexCtorSpecifier
from static_frame.core.util import TIndexCtorSpecifiers
from static_frame.core.util import TIndexInitializer
from static_frame.core.util import TLabel
from static_frame.core.util import TLocSelector
from static_frame.core.util import TName
from static_frame.core.util import TNDArrayObject
from static_frame.core.util import TPathSpecifier
from static_frame.core.util import TSortKinds


#-------------------------------------------------------------------------------
class FrameDeferredMeta(type):
    def __repr__(cls) -> str:
        return f'<{cls.__name__}>'

class FrameDeferred(metaclass=FrameDeferredMeta):
    '''
    Token placeholder for :obj:`Frame` not yet loaded.
    '''
#-------------------------------------------------------------------------------

if tp.TYPE_CHECKING:
    from static_frame.core.generic_aliases import TFrameAny  # pragma: no cover
    from static_frame.core.generic_aliases import TIndexHierarchyAny  # pragma: no cover
    from static_frame.core.generic_aliases import TSeriesAny  # pragma: no cover

    TNDArrayAny = np.ndarray[tp.Any, tp.Any] #pragma: no cover
    TDtypeAny = np.dtype[tp.Any] #pragma: no cover
    TDtypeObject = np.dtype[np.object_] #pragma: no cover
    TSeriesObject = Series[tp.Any, np.object_] #pragma: no cover

    TBusItems = tp.Iterable[tp.Tuple[ #pragma: no cover
            TLabel, tp.Union[TFrameAny, tp.Type[FrameDeferred]]]] #pragma: no cover

    TIterFrame = tp.Iterator[TFrameAny] #pragma: no cover

#-------------------------------------------------------------------------------
TVIndex = tp.TypeVar('TVIndex', bound=IndexBase, default=tp.Any) # pylint: disable=E1123

class Bus(ContainerBase, StoreClientMixin, tp.Generic[TVIndex]): # not a ContainerOperand
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

    _values_mutable: TNDArrayAny
    _index: IndexBase
    _store: tp.Optional[Store]
    _config: StoreConfigMap
    _name: TName

    STATIC = False
    _NDIM: int = 1

    @classmethod
    def from_items(cls,
            pairs: tp.Iterable[tp.Tuple[TLabel, TFrameAny]],
            *,
            config: StoreConfigMapInitializer = None,
            name: TName = None,
            index_constructor: tp.Optional[tp.Callable[..., IndexBase]] = None
            ) -> tp.Self:
        '''Return a :obj:`Bus` from an iterable of pairs of label, :obj:`Frame`.

        Returns:
            :obj:`Bus`
        '''
        frames = []
        index = []
        for i, f in pairs: # might be a generator
            index.append(i)
            frames.append(f)

        return cls(frames,
                index=index,
                index_constructor=index_constructor,
                name=name,
                config=config,
                )

    @classmethod
    def from_frames(cls,
            frames: tp.Iterable[TFrameAny],
            *,
            index_constructor: TIndexCtorSpecifier = None,
            config: StoreConfigMapInitializer = None,
            name: TName = None,
            ) -> tp.Self:
        '''Return a :obj:`Bus` from an iterable of :obj:`Frame`; labels will be drawn from :obj:`Frame.name`.
        '''
        try:
            return cls.from_items(((f.name, f) for f in frames),
                    index_constructor=index_constructor,
                    config=config,
                    name=name,
                    )
        except ErrorInitIndexNonUnique:
            raise ErrorInitIndexNonUnique('Frames do not have unique names.') from None

    @classmethod
    def from_dict(cls,
            mapping: tp.Dict[TLabel, TFrameAny],
            *,
            name: TName = None,
            index_constructor: tp.Optional[tp.Callable[..., IndexBase]] = None
            ) -> tp.Self:
        '''Bus construction from a mapping of labels and :obj:`Frame`.

        Args:
            mapping: a dictionary or similar mapping interface.

        Returns:
            :obj:`Bus`
        '''
        return cls(frames=mapping.values(),
                index=mapping.keys(),
                index_constructor=index_constructor,
                name=name,
                )

    @classmethod
    def from_series(cls,
            series: TSeriesAny,
            *,
            store: tp.Optional[Store] = None,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            own_data: bool = False,
            ) -> tp.Self:
        '''
        Create a :obj:`Bus` from a :obj:`Series` of :obj:`Frame`.
        '''
        # NOTE: this interface is for 0.9 after the default Bus no longer accepts a Series
        return cls(series.values,
                index=series.index,
                store=store,
                config=config,
                max_persist=max_persist,
                own_data=own_data,
                own_index=True,
                name=series.name,
                )

    @classmethod
    def from_concat(cls,
            containers: tp.Iterable[TBusAny],
            *,
            index: tp.Optional[tp.Union[TIndexInitializer, TIndexAutoFactory]] = None,
            name: TName = NAME_DEFAULT,
            ) -> tp.Self:
        '''
        Concatenate multiple :obj:`Bus` into a new :obj:`Bus`. All :obj:`Bus` will load all :obj:`Frame` into memory if any are deferred.
        '''
        # will extract .values, .index from Bus, which will correct load from Store as needed
        # NOTE: useful to use Series here as it handles aligned names, IndexAutoFactory, etc.
        series: TSeriesObject = Series.from_concat(containers, index=index, name=name)
        return cls.from_series(series, own_data=True)

    #---------------------------------------------------------------------------
    # constructors by data format
    @classmethod
    def _from_store(cls,
            store: Store,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            index_constructor: TIndexCtorSpecifier = None,
            ) -> tp.Self:
        return cls(None, # will generate FrameDeferred array
                index=store.labels(config=config),
                index_constructor=index_constructor,
                store=store,
                config=config,
                max_persist=max_persist,
                own_data=True,
                )

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_zip_tsv(cls,
            fp: TPathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            index_constructor: TIndexCtorSpecifier = None,
            ) -> tp.Self:
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
            fp: TPathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            index_constructor: TIndexCtorSpecifier = None,
            ) -> tp.Self:
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
            fp: TPathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            index_constructor: TIndexCtorSpecifier = None,
            ) -> tp.Self:
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
            fp: TPathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            index_constructor: TIndexCtorSpecifier = None,
            ) -> tp.Self:
        '''
        Given a file path to zipped NPZ :obj:`Bus` store, return a :obj:`Bus` instance.

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
    def from_zip_npy(cls,
            fp: TPathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            index_constructor: TIndexCtorSpecifier = None,
            ) -> tp.Self:
        '''
        Given a file path to zipped NPY :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreZipNPY(fp)
        return cls._from_store(store,
                config=config,
                max_persist=max_persist,
                index_constructor=index_constructor,
                )

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_zip_parquet(cls,
            fp: TPathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            index_constructor: TIndexCtorSpecifier = None,
            ) -> tp.Self:
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
            fp: TPathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            index_constructor: TIndexCtorSpecifier = None,
            ) -> tp.Self:
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
            fp: TPathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            index_constructor: TIndexCtorSpecifier = None,
            ) -> tp.Self:
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
    def from_duckdb(cls,
            fp: TPathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            index_constructor: TIndexCtorSpecifier = None,
            ) -> tp.Self:
        '''
        Given a file path to an DuckDB :obj:`Bus` store, return a :obj:`Bus` instance.

        {args}
        '''
        store = StoreDuckDB(fp)
        return cls._from_store(store,
                config=config,
                max_persist=max_persist,
                index_constructor=index_constructor,
                )

    @classmethod
    @doc_inject(selector='bus_constructor')
    def from_hdf5(cls,
            fp: TPathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            index_constructor: TIndexCtorSpecifier = None,
            ) -> tp.Self:
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
    def __init__(self,
            frames: TNDArrayAny | tp.Iterable[TFrameAny | tp.Type[FrameDeferred]] | None,
            *,
            index: TIndexInitializer,
            index_constructor: TIndexCtorSpecifier = None,
            name: TName = NAME_DEFAULT,
            store: tp.Optional[Store] = None,
            config: StoreConfigMapInitializer = None,
            max_persist: tp.Optional[int] = None,
            own_index: bool = False,
            own_data: bool = False,
            ):

        '''
        Default Bus constructor.

        {args}
        '''
        if max_persist is not None:
            # use an (ordered) dictionary to give use an ordered set, simply pointing to None for all keys
            self._last_accessed: tp.Dict[TLabel, None] = {}

        if own_index:
            self._index = index #type: ignore
        else:
            self._index = index_from_optional_constructor(index,
                    default_constructor=Index,
                    explicit_constructor=index_constructor
                    )
        count = len(self._index) # pyright: ignore
        frames_array: TNDArrayAny
        self._loaded: TNDArrayAny
        load_array: bool | np.bool_
        self._loaded_all: bool | np.bool_

        if frames is None:
            if store is None:
                raise ErrorInitBus('Cannot initialize a :obj:`Bus` with neither `frames` nor `store`.')
            self._values_mutable = np.full(count, FrameDeferred, dtype=DTYPE_OBJECT)
            self._loaded = np.full(count, False, dtype=DTYPE_BOOL)
            self._loaded_all = False
        else:
            if frames.__class__ is np.ndarray:
                if frames.dtype != DTYPE_OBJECT: #type: ignore
                    raise ErrorInitBus(
                            f'Series passed to initializer must have dtype object, not {frames.dtype}') #type: ignore
                frames_array = frames # type: ignore
                load_array = False
            else:
                if own_data:
                    raise ErrorInitBus('Cannot use `own_data` when not supplying an array.')
                frames_array = np.empty(count, dtype=DTYPE_OBJECT)
                load_array = True

            self._loaded = np.empty(count, dtype=DTYPE_BOOL)
            # do a one time iteration of series

            for i, (label, value) in enumerate(zip_longest(
                    index,
                    frames,
                    fillvalue=ZIP_LONGEST_DEFAULT,
                    )):
                if label is ZIP_LONGEST_DEFAULT or value is ZIP_LONGEST_DEFAULT:
                    raise ErrorInitBus('frames and index are not of equal length')

                if load_array:
                    frames_array[i] = value

                if value is FrameDeferred:
                    self._loaded[i] = False
                elif isinstance(value, Frame): # permit FrameGO?
                    if max_persist is not None:
                        self._last_accessed[label] = None
                    self._loaded[i] = True
                else:
                    raise ErrorInitBus(f'supplied {value.__class__} is not a Frame or FrameDeferred.')

            self._loaded_all = self._loaded.all()

            if own_data or load_array:
                self._values_mutable = frames_array
            else:
                self._values_mutable = frames_array.copy()
            self._values_mutable.flags.writeable = True

        # self._index = index
        self._name = None if name is NAME_DEFAULT else name
        self._store = store

        # Not handling cases of max_persist being greater than the length of the Series (might floor to length)
        if max_persist is not None and max_persist < self._loaded.sum():
            raise ErrorInitBus('max_persist cannot be less than the number of already loaded Frames')
        self._max_persist = max_persist

        # providing None will result in default; providing a StoreConfig or StoreConfigMap will return an appropriate map
        self._config = StoreConfigMap.from_initializer(config)

    #---------------------------------------------------------------------------
    def _derive_from_series(self,
            series: TSeriesObject,
            *,
            own_data: bool = False,
            ) -> tp.Self:
        '''Utility for creating a derived Bus, propagating the associated ``Store`` and configuration. This can be used if the passed `series` is a subset or re-ordering of self._series; however, if the index has been transformed, this method should not be used, as, if there is a Store, the labels are no longer found in that Store.
        '''
        # NOTE: there may be a more efficient path than using a Series
        return self.__class__.from_series(series,
                store=self._store,
                config=self._config,
                max_persist=self._max_persist,
                own_data=own_data,
                )

    # ---------------------------------------------------------------------------
    def __reversed__(self) -> tp.Iterator[TLabel]:
        '''
        Returns a reverse iterator on the :obj:`Bus` index.

        Returns:
            :obj:`Index`
        '''
        return reversed(self._index)

    #---------------------------------------------------------------------------
    # name interface

    @property
    @doc_inject()
    def name(self) -> TName:
        '''{}'''
        return self._name

    def rename(self, name: TName) -> tp.Self:
        '''
        Return a new :obj:`Bus` with an updated name attribute.
        '''
        # NOTE: do not want to use .values as this will force loading all Frames; use _values_mutable and let a copy be made by constructor
        return self.__class__(self._values_mutable,
                index=self._index,
                name=name,
                store=self._store,
                config=self._config,
                max_persist=self._max_persist,
                own_index=True,
                own_data=False,
                )

    #---------------------------------------------------------------------------
    # interfaces

    @property
    def loc(self) -> InterGetItemLocReduces[TBusAny, np.object_]:
        return InterGetItemLocReduces(self._extract_loc)

    @property
    def iloc(self) -> InterGetItemILocReduces[TBusAny, np.object_]:
        return InterGetItemILocReduces(self._extract_iloc)

    @property
    def drop(self) -> InterfaceSelectTrio[TBusAny]:
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
    def iter_element(self) -> IterNodeNoArgReducible[TBusAny]:
        '''
        Iterator of elements.
        '''
        return IterNodeNoArgReducible(
                container=self,
                function_items=self._axis_element_items,
                function_values=self._axis_element,
                yield_type=IterNodeType.VALUES,
                apply_type=IterNodeApplyType.SERIES_VALUES,
                )

    @property
    def iter_element_items(self) -> IterNodeNoArgReducible[TBusAny]:
        '''
        Iterator of label, element pairs.
        '''
        return IterNodeNoArgReducible(
                container=self,
                function_items=self._axis_element_items,
                function_values=self._axis_element,
                yield_type=IterNodeType.ITEMS,
                apply_type=IterNodeApplyType.SERIES_VALUES,
                )

    #---------------------------------------------------------------------------
    # index manipulation

    # NOTE: must return a new Bus with fully-realized Frames, as cannot guarantee usage of a Store after labels have been changed.

    @doc_inject(selector='reindex', class_name='Bus')
    def reindex(self,
            index: TIndexInitializer,
            *,
            fill_value: tp.Any,
            own_index: bool = False,
            check_equals: bool = True
            ) -> tp.Self:
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
        # NOTE: do not propagate store after reindex
        return self.__class__.from_series(series, config=self._config)

    @doc_inject(selector='relabel', class_name='Bus')
    def relabel(self,
            index: tp.Optional[TRelabelInput],
            *,
            index_constructor: TIndexCtorSpecifier = None,
            ) -> tp.Self:
        '''
        {doc}

        Args:
            index: {relabel_input_index}
        '''
        # NOTE: can be done without going trhough a series
        series = self.to_series().relabel(index, index_constructor=index_constructor)
        # NOTE: do not propagate store after relabel
        return self.__class__.from_series(series, config=self._config)

    @doc_inject(selector='relabel_flat', class_name='Bus')
    def relabel_flat(self) -> tp.Self:
        '''
        {doc}
        '''
        series = self.to_series().relabel_flat()
        return self.__class__.from_series(series, config=self._config)

    @doc_inject(selector='relabel_level_add', class_name='Bus')
    def relabel_level_add(self,
            level: TLabel
            ) -> tp.Self:
        '''
        {doc}

        Args:
            level: {level}
        '''
        series = self.to_series().relabel_level_add(level)
        return self.__class__.from_series(series, config=self._config)


    @doc_inject(selector='relabel_level_drop', class_name='Bus')
    def relabel_level_drop(self,
            count: int = 1
            ) -> tp.Self:
        '''
        {doc}

        Args:
            count: {count}
        '''
        series = self.to_series().relabel_level_drop(count)
        return self.__class__.from_series(series, config=self._config)

    def rehierarch(self,
            depth_map: tp.Sequence[int],
            *,
            index_constructors: TIndexCtorSpecifiers = None,
            ) -> tp.Self:
        '''
        Return a new :obj:`Bus` with new a hierarchy based on the supplied ``depth_map``.
        '''
        series = self.to_series().rehierarch(
                depth_map,
                index_constructors=index_constructors,
                )
        return self.__class__.from_series(series, config=self._config)


    #---------------------------------------------------------------------------
    # na / falsy handling

    # NOTE: not implemented, as a Bus must contain only Frame or FrameDeferred

    #---------------------------------------------------------------------------
    # cache management

    def _update_values_mutable_iloc(self, key: TILocSelector) -> None:
        '''
        Update _values_mutable with the key specified, where key can be any iloc.

        Args:
            key: always an iloc key.
        '''
        max_persist = self._max_persist
        max_persist_active = max_persist is not None

        target_loaded = self._loaded[key]
        target_loaded_count = target_loaded.sum()
        load = False if self._loaded_all else not target_loaded.all()
        if not load and not max_persist_active:
            return

        index = self._index
        label: TLabel
        key_is_element = isinstance(key, INT_TYPES)

        if not load and max_persist_active: # must update LRU position
            labels = (index.iloc[key],) if key_is_element else index.iloc[key].values
            for label in labels: # update LRU position
                self._last_accessed[label] = self._last_accessed.pop(label, None)
            return

        if self._store is None: # Store must be defined if we are partially loaded
            raise RuntimeError('no store defined')

        array = self._values_mutable
        # selection might result in an element so types here are not precise
        target_values: TNDArrayAny = array[key]

        target_labels: TNDArrayAny | TIndexHierarchyAny
        if self._index._NDIM == 2:
            # if an IndexHierarchy, using .values results in a 2D array that might coerce types; thus, must keep an index
            target_labels = self._index[key] # type: ignore[assignment]
        else:
            # if an 1D index, we can immediately reduce to an array
            target_labels = self._index.values[key]

        target_count = 1 if key_is_element else len(target_labels)

        if max_persist_active:
            loaded_count = self._loaded.sum()
            loaded_available = max_persist - loaded_count
            loaded_needed = target_count - target_loaded_count

        store_reader: TIterFrame
        targets_items: TBusItems

        # NOTE: prepare iterable of pairs of label, Frame / FrameDeferred; ensure that for every FrameDeferred, the appropriate Frame is loaded and yielded from the store_reader in order. We must ensure within the target of requested Frame we do not delete any previously-loaded Frame. If max_persist is less than the target, reduce the target to max_persist.

        if key_is_element:
            store_reader = iter((self._store.read(target_labels, config=self._config[target_labels]),)) # type: ignore
            targets_items = ((target_labels, target_values),) # type: ignore
        # more than one Frame
        elif (not max_persist_active
                or max_persist == 1
                or loaded_needed <= loaded_available # pyright: ignore
                ):
            # only read-in labels that are deferred; as loaded_needed is less than loaded_available, no Frame will be removed
            if target_loaded_count:
                labels_to_read = target_labels[~target_loaded]
            else: # no targets are loaded
                labels_to_read = target_labels

            store_reader = self._store.read_many(labels_to_read, config=self._config)
            targets_items = zip(target_labels, target_values)
        # max_persist_active, must delete some Frame
        else:
            if loaded_needed <= max_persist:
                # loaded_needed is less than _max_persist but greater than loaded_available, meaning that some Frame have to be deleted. we must ensure we do not delete a Frame we already have loaded within the target region, so move them to the back of the LRU
                if target_loaded_count:
                    # update LRU position to ensure we do not delete in target
                    for label in target_labels[target_loaded]: # update LRU position
                        self._last_accessed[label] = self._last_accessed.pop(label, None)
                    labels_to_read = target_labels[~target_loaded]
                else: # no targets are loaded
                    labels_to_read = target_labels

            else: # loaded_needed > max_persist:
                # Need to load more than max_persist, so limit to last max_persist-length components. All other Frame, if loaded, will be deleted
                # assert max_persist < len(target_labels)
                target_labels = target_labels[-max_persist:] # type: ignore # pylint: disable=E1130
                target_values = target_values[-max_persist:] # type: ignore # pylint: disable=E1130
                target_loaded = target_loaded[-max_persist:] # type: ignore # pylint: disable=E1130

                if target_loaded.any():
                    # update LRU position to ensure we do not delete in target
                    for label in target_labels[target_loaded]:
                        self._last_accessed[label] = self._last_accessed.pop(label, None)
                    labels_to_read = target_labels[~target_loaded]
                else:
                    # no targets are loaded, will only load a subset of targets of size equal to max_persist; can unpersist everything else
                    labels_to_read = target_labels

                    array[self._loaded] = FrameDeferred
                    self._loaded[NULL_SLICE] = False
                    self._last_accessed.clear()

            store_reader = self._store.read_many(labels_to_read, config=self._config)
            targets_items = zip(target_labels, target_values)

        # Iterate over items that have been selected; there must be at least 1 FrameDeferred among this selection. Note that we iterate over all Frame in the target, not just those form the store, as we need to update LRU positions for all values in the target
        for label, frame in targets_items: # pyright: ignore
            idx = index._loc_to_iloc(label)

            if frame is FrameDeferred:
                frame = next(store_reader)
                array[idx] = frame
                self._loaded[idx] = True # update loaded status
                if max_persist_active:
                    loaded_count += 1

            if max_persist_active: # update LRU position
                self._last_accessed[label] = self._last_accessed.pop(label, None)

                if loaded_count > max_persist: # pyright: ignore
                    label_remove = next(iter(self._last_accessed))
                    del self._last_accessed[label_remove]
                    idx_remove = index._loc_to_iloc(label_remove)
                    self._loaded[idx_remove] = False
                    array[idx_remove] = FrameDeferred
                    loaded_count -= 1



        self._loaded_all = self._loaded.all()

    def unpersist(self) -> None:
        '''Replace all loaded :obj:`Frame` with :obj:`FrameDeferred`.
        '''
        if self._store is None:
            # no-op so Yarn or Quilt can call regardless of Store
            return

        self._values_mutable[self._loaded] = FrameDeferred
        self._loaded[NULL_SLICE] = False
        self._loaded_all = False

        if self._max_persist is not None:
            self._last_accessed.clear()

    #---------------------------------------------------------------------------
    # extraction

    def _extract_iloc(self, key: TILocSelector) -> tp.Self:
        '''
        Returns:
            Bus or, if an element is selected, a Frame
        '''
        self._update_values_mutable_iloc(key=key)

        # iterable selection should be handled by NP
        values: tp.Any = self._values_mutable[key]

        # NOTE: Bus only stores Frame and FrameDeferred, can rely on check with values
        if not values.__class__ is np.ndarray: # if we have a single element
            return values #type: ignore

        return self.__class__(values,
                index=self._index.iloc[key],
                name=self._name,
                store=self._store,
                config=self._config,
                max_persist=self._max_persist,
                own_index=True,
                own_data=False, # force immutable copy
                )

    def _extract_loc(self, key: TLocSelector) -> tp.Self:
        iloc_key = self._index._loc_to_iloc(key)
        return self._extract_iloc(iloc_key)

    @doc_inject(selector='selector')
    def __getitem__(self, key: TLocSelector) -> tp.Self:
        '''Selector of values by label.

        Args:
            key: {key_loc}
        '''
        return self._extract_loc(key)

    #---------------------------------------------------------------------------
    # utilities for alternate extraction: drop

    def _drop_iloc(self, key: TILocSelector) -> tp.Self:
        series = self._to_series_state()._drop_iloc(key)
        return self._derive_from_series(series, own_data=True)

    def _drop_loc(self, key: TLocSelector) -> tp.Self:
        return self._drop_iloc(self._index._loc_to_iloc(key))

    #---------------------------------------------------------------------------
    # axis functions

    def _axis_element_items(self,
            ) -> tp.Iterator[tp.Tuple[TLabel, TFrameAny]]:
        '''Generator of index, value pairs, equivalent to Series.items(). Repeated to have a common signature as other axis functions.
        '''
        yield from self.items()

    def _axis_element(self,
            ) -> tp.Iterator[tp.Any]:
        if self._loaded_all:
            yield from self._values_mutable
        elif self._max_persist is None: # load all at once if possible
            if not self._loaded_all:
                self._update_values_mutable_iloc(key=NULL_SLICE)
            yield from self._values_mutable
        elif self._max_persist > 1:
            i = 0
            i_max = len(self._index.values)
            while i < i_max:
                # draw values up to size of max_persist
                key = slice(i, min(i + self._max_persist, i_max))
                self._update_values_mutable_iloc(key=key)
                for j in range(key.start, key.stop):
                    yield self._values_mutable[j]
                i += self._max_persist
        else: # max_persist is 1
            for i in range(self.__len__()):
                self._update_values_mutable_iloc(key=i)
                yield self._values_mutable[i]

    #---------------------------------------------------------------------------
    # dictionary-like interface; these will force loading contained Frame

    def items(self) -> tp.Iterator[tp.Tuple[TLabel, TFrameAny]]:
        '''Iterator of pairs of :obj:`Bus` label and contained :obj:`Frame`.
        '''
        if self._loaded_all:
            yield from zip(self._index, self._values_mutable)
        elif self._max_persist is None: # load all at once if possible
            if not self._loaded_all:
                self._update_values_mutable_iloc(key=NULL_SLICE)
            yield from zip(self._index, self._values_mutable)
        elif self._max_persist > 1:
            # if _max_persist is greater than 1, load as many Frame as possible (up to the max persist) at a time; this optimizes read operations from the Store
            labels = self._index.values
            i = 0
            i_max = len(labels)
            while i < i_max:
                key = slice(i, min(i + self._max_persist, i_max))
                labels_select = labels[key] # may over select
                self._update_values_mutable_iloc(key=key)
                yield from zip(labels_select, self._values_mutable[key])
                i += self._max_persist
        else: # max_persist is 1
            for i, label in enumerate(self._index.values):
                self._update_values_mutable_iloc(key=i)
                yield label, self._values_mutable[i]

    _items_store = items

    @property
    def values(self) -> TNDArrayObject:
        '''A 1D object array of all :obj:`Frame` contained in the :obj:`Bus`. The returned ``np.ndarray`` will have ``Frame``; this will never return an array with ``FrameDeferred``, but ``max_persist`` will be observed in reading from the Store.
        '''
        # NOTE: when self._values_mutable is fully loaded, it could become immutable and avoid a copy. However, with unpersist(), we might unload all Frame

        if self._loaded_all:
            post = self._values_mutable.copy()
            post.flags.writeable = False
            return post

        if self._max_persist is None: # load all at once if possible
            # b._loaded_all must be False
            self._update_values_mutable_iloc(key=NULL_SLICE)
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
                self._update_values_mutable_iloc(key=key)
                post[key] = self._values_mutable[key]
                i += self._max_persist
        else: # max_persist is 1
            for i in range(self.__len__()):
                self._update_values_mutable_iloc(key=i)
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
    def mloc(self) -> TSeriesObject:
        '''Returns a :obj:`Series` showing a tuple of memory locations within each loaded Frame.
        '''
        if not self._loaded.any():
            return Series.from_element(None, index=self._index)

        def gen() -> tp.Iterator[tp.Tuple[TLabel, tp.Optional[tp.Tuple[int, ...]]]]:
            for label, f in zip(self._index, self._values_mutable):
                if f is FrameDeferred:
                    yield label, None
                else:
                    yield label, tuple(f.mloc)

        return Series.from_items(gen())

    @property
    def dtypes(self) -> TFrameAny:
        '''Returns a :obj:`Frame` of dtype per column for all loaded Frames.
        '''
        if not self._loaded.any():
            return Frame(index=self._index)

        f: TFrameAny = Frame.from_concat(
                frames=(f.dtypes for f in self._values_mutable if f is not FrameDeferred),
                fill_value=None,
                ).reindex(index=self._index, fill_value=None)
        return f

    @property
    def shapes(self) -> TSeriesObject:
        '''A :obj:`Series` describing the shape of each loaded :obj:`Frame`. Unloaded :obj:`Frame` will have a shape of None.

        Returns:
            :obj:`Series`
        '''
        values = (f.shape if f is not FrameDeferred else None for f in self._values_mutable)
        return Series(values, index=self._index, dtype=DTYPE_OBJECT, name='shape')

    @property
    def nbytes(self) -> int:
        '''Total bytes of data currently loaded in the Bus.
        '''
        return sum(f.nbytes if f is not FrameDeferred else 0 for f in self._values_mutable)

    @property
    def status(self) -> TFrameAny:
        '''
        Return a :obj:`Frame` indicating loaded status, size, bytes, and shape of all loaded :obj:`Frame`.
        '''
        def gen() -> tp.Iterator[TSeriesAny]:

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

        return Frame.from_concat(gen(), axis=1)


    #---------------------------------------------------------------------------
    # common attributes from the numpy array

    @property
    def dtype(self) -> TDtypeObject:
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
        return self._values_mutable.size

    #---------------------------------------------------------------------------

    @property
    def index(self) -> IndexBase:
        '''
        The index instance assigned to this container.

        Returns:
            :obj:`Index`
        '''
        return self._index

    #---------------------------------------------------------------------------
    # dictionary-like interface

    def keys(self) -> IndexBase:
        '''
        Iterator of index labels.

        Returns:
            :obj:`Iterator[Hashable]`
        '''
        return self._index

    def __iter__(self) -> tp.Iterator[TLabel]:
        '''
        Iterator of index labels, same as :obj:`static_frame.Series.keys`.

        Returns:
            :obj:`Iterator[Hashasble]`
        '''
        return self._index.__iter__()

    def __contains__(self, value: TLabel) -> bool:
        '''
        Inclusion of value in index labels.

        Returns:
            :obj:`bool`
        '''
        return self._index.__contains__(value)

    def get(self, key: TLabel,
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
    def head(self, count: int = 5) -> TBusAny:
        '''{doc}

        Args:
            {count}

        Returns:
            :obj:`Bus`
        '''
        return self.iloc[:count]

    @doc_inject(selector='tail', class_name='Bus')
    def tail(self, count: int = 5) -> TBusAny:
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
            ascending: TBoolOrBools = True,
            kind: TSortKinds = DEFAULT_SORT_KIND,
            key: tp.Optional[tp.Callable[[IndexBase], tp.Union[TNDArrayAny, IndexBase]]] = None,
            ) -> tp.Self:
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
        return self._derive_from_series(series, own_data=True)

    @doc_inject(selector='sort')
    def sort_values(self,
            *,
            ascending: bool = True,
            kind: TSortKinds = DEFAULT_SORT_KIND,
            key: tp.Callable[[TSeriesAny], tp.Union[TNDArrayAny, TSeriesAny]],
            ) -> tp.Self:
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
        cfs = self.to_series() # force loading all Frame
        series = cfs.sort_values(
                ascending=ascending,
                kind=kind,
                key=key,
                )
        return self._derive_from_series(series, own_data=True)

    def roll(self,
            shift: int,
            *,
            include_index: bool = False,
            ) -> tp.Self:
        '''Return a Bus with values rotated forward and wrapped around the index (with a positive shift) or backward and wrapped around the index (with a negative shift).

        Args:
            shift: Positive or negative integer shift.
            include_index: Determine if the Index is shifted with the underlying data.

        Returns:
            :obj:`Bus`
        '''
        series = self._to_series_state().roll(shift=shift, include_index=include_index)
        return self._derive_from_series(series, own_data=True)

    def shift(self,
            shift: int,
            *,
            fill_value: tp.Any,
            ) -> tp.Self:
        '''Return a :obj:`Bus` with values shifted forward on the index (with a positive shift) or backward on the index (with a negative shift).

        Args:
            shift: Positive or negative integer shift.
            fill_value: Value to be used to fill data missing after the shift.

        Returns:
            :obj:`Bus`
        '''
        series = self._to_series_state().shift(shift=shift, fill_value=fill_value)
        return self._derive_from_series(series, own_data=True)

    #---------------------------------------------------------------------------
    # exporter

    def _to_series_state(self) -> TSeriesObject:
        # the mutable array will be copied in the Series construction
        return Series(self._values_mutable,
                index=self._index,
                own_index=True,
                name=self._name,
                )

    def to_series(self) -> TSeriesObject:
        '''Return a :obj:`Series` with the :obj:`Frame` contained in this :obj:`Bus`. If the :obj:`Bus` is associated with a :obj:`Store`, all :obj:`Frame` will be loaded into memory and the returned :obj:`Bus` will no longer be associated with the :obj:`Store`.
        '''
        # values returns an immutable array and will fully realize from Store
        return Series(self.values,
                index=self._index,
                own_index=True,
                name=self._name,
                )

    def _to_signature_bytes(self,
            include_name: bool = True,
            include_class: bool = True,
            encoding: str = 'utf-8',
            ) -> bytes:

        v = (f._to_signature_bytes(
                include_name=include_name,
                include_class=include_class,
                encoding=encoding,
                ) for f in self._axis_element())

        return b''.join(chain(
                iter_component_signature_bytes(self,
                        include_name=include_name,
                        include_class=include_class,
                        encoding=encoding),
                (self._index._to_signature_bytes(
                        include_name=include_name,
                        include_class=include_class,
                        encoding=encoding),),
                v))


TBusAny = Bus[tp.Any]




