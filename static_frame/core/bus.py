from __future__ import annotations

from datetime import datetime
from datetime import timezone
from itertools import chain
from itertools import islice
from itertools import zip_longest
from pathlib import Path

import numpy as np
import typing_extensions as tp

from static_frame.core.container import ContainerBase
from static_frame.core.container_util import index_from_optional_constructor
from static_frame.core.container_util import iter_component_signature_bytes
from static_frame.core.display import Display
from static_frame.core.display import DisplayActive
from static_frame.core.display import DisplayHeader
from static_frame.core.doc_str import doc_inject
from static_frame.core.exception import ErrorInitBus
from static_frame.core.exception import ErrorInitIndexNonUnique
from static_frame.core.exception import immutable_type_error_factory
from static_frame.core.frame import Frame
from static_frame.core.index import Index
from static_frame.core.index_base import IndexBase
from static_frame.core.node_iter import IterNodeApplyType
from static_frame.core.node_iter import IterNodeNoArgReducible
from static_frame.core.node_selector import InterfacePersist
from static_frame.core.node_selector import InterfaceSelectTrio
from static_frame.core.node_selector import InterGetItemILocReduces
from static_frame.core.node_selector import InterGetItemLocReduces
from static_frame.core.series import Series
from static_frame.core.store_client_mixin import StoreClientMixin
from static_frame.core.store_config import StoreConfigMap
from static_frame.core.store_config import StoreConfigMapInitializer
from static_frame.core.store_sqlite import StoreSQLite
from static_frame.core.store_xlsx import StoreXLSX
from static_frame.core.store_zip import StoreZipCSV
from static_frame.core.store_zip import StoreZipNPY
from static_frame.core.store_zip import StoreZipNPZ
from static_frame.core.store_zip import StoreZipParquet
from static_frame.core.store_zip import StoreZipPickle
from static_frame.core.store_zip import StoreZipTSV
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
from static_frame.core.util import bytes_to_size_label

if tp.TYPE_CHECKING:
    from collections.abc import Container  # pragma: no cover

    from static_frame.core.display_config import DisplayConfig  # pragma: no cover
    from static_frame.core.index_auto import TIndexAutoFactory  # pragma: no cover
    from static_frame.core.index_auto import TRelabelInput  # pragma: no cover
    from static_frame.core.store import Store  # pragma: no cover
    from static_frame.core.style_config import StyleConfig  # pragma: no cover


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
    from static_frame.core.generic_aliases import TSeriesAny  # pragma: no cover

    TNDArrayAny = np.ndarray[tp.Any, tp.Any] #pragma: no cover
    TDtypeAny = np.dtype[tp.Any] #pragma: no cover
    TDtypeObject = np.dtype[np.object_] #pragma: no cover
    TSeriesObject = Series[tp.Any, np.object_] #pragma: no cover

    TBusItems = tp.Iterable[tp.Tuple[ #pragma: no cover
            TLabel, tp.Union[TFrameAny, tp.Type[FrameDeferred]]]] #pragma: no cover

    TIterFrame = tp.Iterator[TFrameAny] #pragma: no cover

#-------------------------------------------------------------------------------
TVIndex = tp.TypeVar('TVIndex', bound=IndexBase, default=tp.Any)

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
        '_last_loaded',
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
            /,
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
            /,
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
            /,
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
        return cls(mapping.values(),
                index=mapping.keys(),
                index_constructor=index_constructor,
                name=name,
                )

    @classmethod
    def from_series(cls,
            series: TSeriesAny,
            /,
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
            /,
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
            /,
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
            /,
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
            /,
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
            /,
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
            /,
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
            /,
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
            /,
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
            /,
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

    #---------------------------------------------------------------------------
    def __init__(self,
            frames: TNDArrayAny | tp.Iterable[TFrameAny | tp.Type[FrameDeferred]] | None,
            /,
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
            if max_persist < 1:
                raise ErrorInitBus('Cannot initialize a :obj:`Bus` with `max_persist` less than 1; use `None` to disable `max_persist`.')
            # use an dict to give use an ordered set pointing to None for all keys
            self._last_loaded: tp.Dict[TLabel, None] = {}

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
                        self._last_loaded[label] = None
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


    def __copy__(self) -> tp.Self:
        '''
        Return a shallow copy of this :obj:`Bus`.
        '''
        # NOTE: do not want to use .values as this will force loading all Frames; use _values_mutable and let a copy be made by constructor
        # NOTE: the copy will retain the same loaded Frame as the origin
        return self.__class__(self._values_mutable,
                index=self._index,
                name=self._name,
                store=self._store,
                config=self._config,
                max_persist=self._max_persist,
                own_index=True,
                own_data=False, # force copy of _values_mutable
                )

    def copy(self) -> tp.Self:
        '''
        Return a shallow copy of this :obj:`Bus`.
        '''
        return self.__copy__()

    #---------------------------------------------------------------------------
    # name interface

    @property
    @doc_inject()
    def name(self) -> TName:
        '''{}'''
        return self._name

    def rename(self, name: TName, /,) -> tp.Self:
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

    @property
    def persist(self) -> InterfacePersist[TBusAny]:
        '''
        Interface for selectively (or completely) pre-load `Frame` from a store to optimize subsequent single `Frame` extraction.
        '''
        return InterfacePersist(
                func_iloc=self._persist_iloc,
                func_loc=self._persist_loc,
                func_getitem=self._persist_loc,
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

    # NOTE: these methods are not implemented, as a Bus must contain only Frame or FrameDeferred

    #---------------------------------------------------------------------------
    # cache management

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
            self._last_loaded.clear()

    #---------------------------------------------------------------------------

    def _unpersist_next(self, labels_retain: Container[TLabel]) -> None:
        '''Remove the next available loaded Frame. This does not adjust self._loaded_all.
        Args:
            labels_retain: container of labels that are in the assignment region that are not being loaded and need to be retained.
        '''
        # avoid mutating self._last_loaded while iterating; avoid creating a list if not needed
        restore: list[TLabel] | None = None
        for label_remove in self._last_loaded:
            if label_remove not in labels_retain:
                break
            if restore is None:
                restore = [label_remove]
            else:
                restore.append(label_remove)
        if restore is not None:
            for label in restore: # move to back
                self._last_loaded[label] = self._last_loaded.pop(label)

        del self._last_loaded[label_remove]
        idx_remove = self._index._loc_to_iloc(label_remove)
        self._loaded[idx_remove] = False
        self._values_mutable[idx_remove] = FrameDeferred

    def _persist_one(self,
            pos: int | np.integer[tp.Any],
            update_last_loaded: bool,
            ) -> Frame:
        label = self._index[pos]
        f: Frame = self._store.read(label, config=self._config) # type: ignore
        self._values_mutable[pos] = f
        self._loaded[pos] = True # update loaded status
        if update_last_loaded:
            self._last_loaded[label] = None
        return f

    #---------------------------------------------------------------------------

    def _update_mutable_persistent_one(self,
                key: TILocSelector,
                ) -> None:
        if self._loaded_all or not isinstance(key, INT_TYPES) or self._loaded[key]:
            return
        _ = self._persist_one(key, False)
        self._loaded_all = self._loaded.all()

    def _update_mutable_persistent_iter(self) -> tp.Iterator[Frame]:
        values_mutable = self._values_mutable
        if self._loaded_all:
            yield from values_mutable

        index = self._index
        loaded = self._loaded # Boolean array
        loaded_count = loaded.sum()
        size = len(loaded)

        labels_to_load: tp.Iterable[TLabel]
        if index._NDIM == 2:  # if an IndexHierarchy avoid going to an array
            labels_to_load = index[~loaded]
        else:
            labels_to_load = index.values[~loaded]

        store_reader = self._store.read_many(labels_to_load, config=self._config) # type: ignore
        for idx in range(size): # iter over all values
            if not loaded[idx]:
                f = next(store_reader)
                values_mutable[idx] = f
                loaded[idx] = True
                loaded_count += 1
                self._loaded_all = loaded_count == size
                yield f
            else:
                yield values_mutable[idx]

    def _update_mutable_persistant_many(self,
                key: TILocSelector,
                ) -> None:
        '''Load all `Frame` targeted by `key`.
        '''
        if self._loaded_all:
            return

        index = self._index
        loaded = self._loaded # Boolean array
        values_mutable = self._values_mutable

        if key.__class__ is slice and key == NULL_SLICE:
            labels_unloaded = ~loaded
        else:
            targets = np.zeros(len(loaded), dtype=DTYPE_BOOL)
            targets[key] = True
            labels_unloaded = ~loaded & targets

        if labels_unloaded.any():
            labels_to_load: tp.Iterable[TLabel]
            if index._NDIM == 2:  # if an IndexHierarchy avoid going to an array
                labels_to_load = index[labels_unloaded]
            else:
                labels_to_load = index.values[labels_unloaded]

            for idx, f in zip(
                    index.positions[labels_unloaded],
                    self._store.read_many(labels_to_load, config=self._config) # type: ignore[union-attr]
                    ):
                values_mutable[idx] = f

            loaded[labels_unloaded] = True
            self._loaded_all = self._loaded.all()

    #---------------------------------------------------------------------------

    def _update_mutable_max_persist_one(self,
                key: TILocSelector,
                ) -> None:
        '''
        For loading a single element; keys beyond a single element are accepted but are a no-op.
        '''
        if self._loaded_all or not isinstance(key, INT_TYPES):
            return

        loaded = self._loaded # Boolean array
        loaded_count = loaded.sum()
        if loaded[key]:
            return

        size = len(loaded)
        loaded_count += 1
        if loaded_count > self._max_persist:
            self._unpersist_next(())
            loaded_count -= 1

        _ = self._persist_one(key, True)
        self._loaded_all = loaded_count == size


    def _update_mutable_max_persist_iter(self) -> tp.Iterator[Frame]:
        '''Iterator of all values in the context of max_persist
        '''
        values_mutable = self._values_mutable
        if self._loaded_all:
            yield from values_mutable

        max_persist: int = self._max_persist # type: ignore[assignment]
        index = self._index
        loaded = self._loaded # Boolean array
        last_loaded = self._last_loaded
        loaded_count = loaded.sum()
        size = len(loaded)

        if max_persist > 1:
            i = 0
            labels_to_load: tp.Iterable[TLabel]
            while i < size:
                i_end = i + max_persist
                targets = np.zeros(size, dtype=DTYPE_BOOL)
                targets[i: i_end] = True
                labels_unloaded = ~loaded & targets
                if not labels_unloaded.any():
                    yield from islice(values_mutable, i, i_end)
                else:
                    if index._NDIM == 2:  # if an IndexHierarchy avoid going to an array
                        labels_to_load = index[labels_unloaded]
                    else:
                        labels_to_load = index.values[labels_unloaded]

                    labels_to_keep = index[targets] # keep as index for lookup
                    store_reader = self._store.read_many(labels_to_load, config=self._config) # type: ignore
                    for idx in range(i, min(i_end, size)):
                        if loaded[idx]:
                            yield values_mutable[idx]
                        else:
                            loaded_count += 1
                            if loaded_count > max_persist:
                                self._unpersist_next(labels_to_keep)
                                loaded_count -= 1
                            f = next(store_reader)
                            values_mutable[idx] = f
                            loaded[idx] = True
                            last_loaded[index[idx]] = None
                            self._loaded_all = loaded_count == size
                            yield f
                i = i_end
        else: # max_persist is 1
            for i in range(size):
                if loaded[i]:
                    yield values_mutable[i]
                else:
                    loaded_count += 1
                    if loaded_count > max_persist:
                        self._unpersist_next(())
                        loaded_count -= 1
                    f = self._persist_one(i, True)
                    self._loaded_all = loaded_count == size
                    yield f


    def _update_mutable_max_persist_many(self,
                key: TILocSelector,
                ) -> None:
        '''Load all `Frame` targeted by `key`.
        '''
        if self._loaded_all:
            return

        index = self._index
        loaded = self._loaded # Boolean array
        last_loaded = self._last_loaded
        values_mutable = self._values_mutable
        max_persist: int = self._max_persist # type: ignore[assignment]
        loaded_count = loaded.sum()
        size = len(loaded)

        if key.__class__ is slice and key == NULL_SLICE:
            targets = np.ones(size, dtype=DTYPE_BOOL)
            labels_unloaded = ~loaded
        else:
            targets = np.zeros(size, dtype=DTYPE_BOOL)
            targets[key] = True
            labels_unloaded = ~loaded & targets

        if labels_unloaded.any():
            labels_to_load: tp.Iterable[TLabel]
            if index._NDIM == 2:  # if an IndexHierarchy avoid going to an array
                labels_to_load = index[labels_unloaded]
            else:
                labels_to_load = index.values[labels_unloaded]

            to_load_count = len(labels_to_load) # type: ignore
            if to_load_count + loaded_count <= max_persist:
                for label, idx, f in zip(
                        labels_to_load,
                        index.positions[labels_unloaded],
                        self._store.read_many(labels_to_load, config=self._config) # type: ignore[union-attr]
                        ):
                    values_mutable[idx] = f
                    last_loaded[label] = None

                loaded[labels_unloaded] = True
                self._loaded_all = to_load_count + loaded_count == size

            else: # load only max_persist count from targets
                # from original targets, find max_persist number of indices
                mp_key = np.nonzero(targets)[0][-max_persist:]
                targets = np.zeros(size, dtype=DTYPE_BOOL)
                targets[mp_key] = True
                labels_unloaded = ~loaded & targets

                if labels_unloaded.any():
                    if index._NDIM == 2:  # if an IndexHierarchy avoid going to an array
                        labels_to_load = index[labels_unloaded]
                    else:
                        labels_to_load = index.values[labels_unloaded]
                    labels_to_keep = index[targets] # an index for lookup
                    store_reader = self._store.read_many(labels_to_load, config=self._config) # type: ignore[union-attr]
                    for label, idx in zip(
                            labels_to_load,
                            index.positions[labels_unloaded],
                            ):
                        loaded_count += 1
                        if loaded_count > max_persist:
                            self._unpersist_next(labels_to_keep)
                            loaded_count -= 1
                        f = next(store_reader)
                        values_mutable[idx] = f
                        last_loaded[label] = None

                    loaded[labels_unloaded] = True
                    self._loaded_all = loaded_count == size

    #---------------------------------------------------------------------------
    def _persist_iloc(self, key: TILocSelector) -> None:
        if self._max_persist is None:
            self._update_mutable_persistant_many(key)
        else:
            self._update_mutable_max_persist_many(key)

    def _persist_loc(self, key: TLocSelector) -> None:
        return self._persist_iloc(self._index._loc_to_iloc(key))

    #---------------------------------------------------------------------------
    # extraction

    def _extract_iloc(self, key: TILocSelector) -> tp.Self:
        '''
        Returns:
            Bus or, if an element is selected, a Frame
        '''
        if self._max_persist is None:
            self._update_mutable_persistent_one(key)
        else:
            self._update_mutable_max_persist_one(key)

        # iterable selection should be handled by NP
        values: tp.Any = self._values_mutable[key]

        # NOTE: Bus only stores Frame and FrameDeferred, can rely on check with values
        if values.__class__ is not np.ndarray: # if we have a single element
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

    def __setitem__(self, key: TLabel, value: tp.Any) -> None:
        raise immutable_type_error_factory(self.__class__, '', key, value)

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
        max_persist = self._max_persist
        if self._loaded_all:
            yield from self._values_mutable
        elif max_persist is None:
            yield from self._update_mutable_persistent_iter()
        else:
            yield from self._update_mutable_max_persist_iter()

    #---------------------------------------------------------------------------
    # dictionary-like interface; these will force loading contained Frame

    def items(self) -> tp.Iterator[tp.Tuple[TLabel, TFrameAny]]:
        '''Iterator of pairs of :obj:`Bus` label and contained :obj:`Frame`.
        '''
        if self._max_persist is None: # load all at once if possible
            yield from zip(self._index, self._update_mutable_persistent_iter())
        else:
            yield from zip(self._index, self._update_mutable_max_persist_iter())

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

        max_persist = self._max_persist
        if max_persist is None: # load all at once if possible
            post = np.fromiter(self._update_mutable_persistent_iter(), dtype=DTYPE_OBJECT, count=self.__len__())
        else:
            post = np.fromiter(self._update_mutable_max_persist_iter(), dtype=DTYPE_OBJECT, count=self.__len__())
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
            /,
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
                (f.dtypes for f in self._values_mutable if f is not FrameDeferred),
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

    @property
    def inventory(self) -> TFrameAny:
        '''Return a :obj:`Frame` indicating file_path, last-modified time, and size of underlying disk-based data stores if used for this :obj:`Bus`.
        '''
        records = []
        index = [self._name]
        if self._store is not None:
            fp = Path(self._store._fp)
            size = bytes_to_size_label(fp.stat().st_size)
            utc = datetime.fromtimestamp(
                    self._store._last_modified,
                    timezone.utc).isoformat()
            records.append([str(fp), utc, size])
        else:
            records.append(['', '', ''])
        return Frame.from_records(records,
                columns=('path', 'last_modified', 'size'),
                index=index,
                )

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

    def __contains__(self, value: TLabel, /,) -> bool:
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
            /,
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
    def head(self, count: int = 5, /,) -> TBusAny:
        '''{doc}

        Args:
            {count}

        Returns:
            :obj:`Bus`
        '''
        return self.iloc[:count]

    @doc_inject(selector='tail', class_name='Bus')
    def tail(self, count: int = 5, /,) -> TBusAny:
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
            /,
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
        series = self._to_series_state().roll(shift, include_index=include_index)
        return self._derive_from_series(series, own_data=True)

    def shift(self,
            shift: int,
            /,
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
        series = self._to_series_state().shift(shift, fill_value=fill_value)
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




