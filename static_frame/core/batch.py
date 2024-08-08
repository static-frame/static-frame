from __future__ import annotations

import numpy as np
import typing_extensions as tp

from static_frame.core.bus import Bus
from static_frame.core.container import ContainerOperand
from static_frame.core.display import Display
from static_frame.core.display import DisplayActive
from static_frame.core.display import DisplayHeader
from static_frame.core.display_config import DisplayConfig
from static_frame.core.doc_str import doc_inject
from static_frame.core.doc_str import doc_update
from static_frame.core.exception import BatchIterableInvalid
from static_frame.core.frame import Frame
from static_frame.core.index import Index
from static_frame.core.index_auto import TIndexAutoFactory
from static_frame.core.index_auto import TRelabelInput
from static_frame.core.node_dt import InterfaceBatchDatetime
from static_frame.core.node_fill_value import InterfaceBatchFillValue
from static_frame.core.node_re import InterfaceBatchRe
from static_frame.core.node_selector import InterfaceBatchAsType
from static_frame.core.node_selector import InterfaceGetItemBLoc
from static_frame.core.node_selector import InterfaceSelectTrio
from static_frame.core.node_selector import InterGetItemILocCompound
from static_frame.core.node_selector import InterGetItemLocCompound
from static_frame.core.node_str import InterfaceBatchString
from static_frame.core.node_transpose import InterfaceBatchTranspose
from static_frame.core.node_values import InterfaceBatchValues
from static_frame.core.reduce import InterfaceBatchReduceDispatch
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
from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import ELEMENT_TUPLE
from static_frame.core.util import NAME_DEFAULT
from static_frame.core.util import TBlocKey
from static_frame.core.util import TBoolOrBools
from static_frame.core.util import TCallableAny
from static_frame.core.util import TDtypeSpecifier
from static_frame.core.util import TILocSelectorCompound
from static_frame.core.util import TIndexCtorSpecifier
from static_frame.core.util import TIndexCtorSpecifiers
from static_frame.core.util import TIndexInitializer
from static_frame.core.util import TKeyOrKeys
from static_frame.core.util import TLabel
from static_frame.core.util import TLocSelector
from static_frame.core.util import TLocSelectorCompound
from static_frame.core.util import TName
from static_frame.core.util import TPathSpecifier
from static_frame.core.util import TUFunc
from static_frame.core.util import get_concurrent_executor

TFrameOrSeries = tp.Union[Frame, Series]
TIteratorFrameItems = tp.Iterator[tp.Tuple[TLabel, TFrameOrSeries]]
TGeneratorFrameItems = tp.Callable[..., TIteratorFrameItems]

if tp.TYPE_CHECKING:
    TNDArrayAny = np.ndarray[tp.Any, tp.Any] #pragma: no cover
    TDtypeAny = np.dtype[tp.Any] #pragma: no cover

TSeriesAny = Series[tp.Any, tp.Any]
TFrameAny = Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]]
TBusAny = Bus[tp.Any]

#-------------------------------------------------------------------------------
# family of executor functions normalized in signature (taking a single tuple of args) for usage in processor pool calls

def normalize_container(post: tp.Any
        ) -> TFrameOrSeries:
    # post might be an element, promote to a Series to permit concatenation
    if post.__class__ is np.ndarray:
        if post.ndim == 1:
            return Series(post)
        elif post.ndim == 2:
            return Frame(post)
        # let ndim 0 pass
    if not isinstance(post, (Frame, Series)):
        # NOTE: do not set index as (container.name,), as this can lead to diagonal formations; will already be paired with stored labels
        return Series.from_element(post, index=ELEMENT_TUPLE)
    return post

def call_func(bundle: tp.Tuple[TFrameOrSeries, TCallableAny]
        ) -> TFrameOrSeries:
    container, func = bundle
    return func(container) # type: ignore

def call_func_items(bundle: tp.Tuple[TFrameOrSeries, TCallableAny, TLabel]
        ) -> TFrameOrSeries:
    container, func, label = bundle
    return func(label, container) # type: ignore

def call_attr(bundle: tp.Tuple[TFrameOrSeries, str, tp.Any, tp.Any]
        ) -> TFrameOrSeries:
    container, attr, args, kwargs = bundle
    func = getattr(container, attr)
    return func(*args, **kwargs) # type: ignore

#-------------------------------------------------------------------------------
class Batch(ContainerOperand, StoreClientMixin):
    '''
    A lazy, sequentially evaluated container of :obj:`Frame` that broadcasts operations on contained :obj:`Frame` by return new :obj:`Batch` instances. Full evaluation of operations only occurs when iterating or calling an exporter, such as ``to_frame()`` or ``to_series()``.
    '''

    __slots__ = (
            '_items',
            '_name',
            '_config',
            '_max_workers',
            '_chunksize',
            '_use_threads',
            '_mp_context',
            )

    _config: StoreConfigMap

    @classmethod
    def from_frames(cls,
            frames: tp.Iterable[TFrameAny],
            *,
            name: TName = None,
            config: StoreConfigMapInitializer = None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
            mp_context: tp.Optional[str] = None,
            ) -> 'Batch':
        '''Return a :obj:`Batch` from an iterable of :obj:`Frame`; labels will be drawn from :obj:`Frame.name`.
        '''
        return cls(((f.name, f) for f in frames),
                name=name,
                config=config,
                max_workers=max_workers,
                chunksize=chunksize,
                use_threads=use_threads,
                mp_context=mp_context,
                )

    #---------------------------------------------------------------------------
    # constructors by data format

    @classmethod
    def _from_store(cls,
            store: Store,
            *,
            config: StoreConfigMapInitializer = None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
            mp_context: tp.Optional[str] = None,
            ) -> 'Batch':
        config_map = StoreConfigMap.from_initializer(config)

        items = ((label, store.read(label, config=config_map[label]))
                for label in store.labels(config=config_map))

        return cls(items,
                config=config,
                max_workers=max_workers,
                chunksize=chunksize,
                use_threads=use_threads,
                mp_context=mp_context,
                )

    @classmethod
    @doc_inject(selector='batch_constructor')
    def from_zip_tsv(cls,
            fp: TPathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
            mp_context: tp.Optional[str] = None,
            ) -> 'Batch':
        '''
        Given a file path to zipped TSV :obj:`Batch` store, return a :obj:`Batch` instance.

        {args}
        '''
        store = StoreZipTSV(fp)
        return cls._from_store(store,
                config=config,
                max_workers=max_workers,
                chunksize=chunksize,
                use_threads=use_threads,
                mp_context=mp_context,
                )

    @classmethod
    @doc_inject(selector='batch_constructor')
    def from_zip_csv(cls,
            fp: TPathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
            mp_context: tp.Optional[str] = None,
            ) -> 'Batch':
        '''
        Given a file path to zipped CSV :obj:`Batch` store, return a :obj:`Batch` instance.

        {args}
        '''
        store = StoreZipCSV(fp)
        return cls._from_store(store,
                config=config,
                max_workers=max_workers,
                chunksize=chunksize,
                use_threads=use_threads,
                mp_context=mp_context,
                )

    @classmethod
    @doc_inject(selector='batch_constructor')
    def from_zip_pickle(cls,
            fp: TPathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
            mp_context: tp.Optional[str] = None,
            ) -> 'Batch':
        '''
        Given a file path to zipped pickle :obj:`Batch` store, return a :obj:`Batch` instance.

        {args}
        '''
        store = StoreZipPickle(fp)
        return cls._from_store(store,
                config=config,
                max_workers=max_workers,
                chunksize=chunksize,
                use_threads=use_threads,
                mp_context=mp_context,
                )

    @classmethod
    @doc_inject(selector='batch_constructor')
    def from_zip_npz(cls,
            fp: TPathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
            mp_context: tp.Optional[str] = None,
            ) -> 'Batch':
        '''
        Given a file path to zipped NPZ :obj:`Batch` store, return a :obj:`Batch` instance.

        {args}
        '''
        store = StoreZipNPZ(fp)
        return cls._from_store(store,
                config=config,
                max_workers=max_workers,
                chunksize=chunksize,
                use_threads=use_threads,
                mp_context=mp_context,
                )

    @classmethod
    @doc_inject(selector='batch_constructor')
    def from_zip_npy(cls,
            fp: TPathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
            mp_context: tp.Optional[str] = None,
            ) -> 'Batch':
        '''
        Given a file path to zipped NPY :obj:`Batch` store, return a :obj:`Batch` instance.

        {args}
        '''
        store = StoreZipNPY(fp)
        return cls._from_store(store,
                config=config,
                max_workers=max_workers,
                chunksize=chunksize,
                use_threads=use_threads,
                mp_context=mp_context,
                )

    @classmethod
    @doc_inject(selector='batch_constructor')
    def from_zip_parquet(cls,
            fp: TPathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
            mp_context: tp.Optional[str] = None,
            ) -> 'Batch':
        '''
        Given a file path to zipped parquet :obj:`Batch` store, return a :obj:`Batch` instance.

        {args}
        '''
        store = StoreZipParquet(fp)
        return cls._from_store(store,
                config=config,
                max_workers=max_workers,
                chunksize=chunksize,
                use_threads=use_threads,
                mp_context=mp_context,
                )


    @classmethod
    @doc_inject(selector='batch_constructor')
    def from_xlsx(cls,
            fp: TPathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
            mp_context: tp.Optional[str] = None,
            ) -> 'Batch':
        '''
        Given a file path to an XLSX :obj:`Batch` store, return a :obj:`Batch` instance.

        {args}
        '''
        # how to pass configuration for multiple sheets?
        store = StoreXLSX(fp)
        return cls._from_store(store,
                config=config,
                max_workers=max_workers,
                chunksize=chunksize,
                use_threads=use_threads,
                mp_context=mp_context,
                )


    @classmethod
    @doc_inject(selector='batch_constructor')
    def from_sqlite(cls,
            fp: TPathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
            mp_context: tp.Optional[str] = None,
            ) -> 'Batch':
        '''
        Given a file path to an SQLite :obj:`Batch` store, return a :obj:`Batch` instance.

        {args}
        '''
        store = StoreSQLite(fp)
        return cls._from_store(store,
                config=config,
                max_workers=max_workers,
                chunksize=chunksize,
                use_threads=use_threads,
                mp_context=mp_context,
                )

    @classmethod
    @doc_inject(selector='batch_constructor')
    def from_duckdb(cls,
            fp: TPathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
            mp_context: tp.Optional[str] = None,
            ) -> 'Batch':
        '''
        Given a file path to an DuckDB :obj:`Batch` store, return a :obj:`Batch` instance.

        {args}
        '''
        store = StoreDuckDB(fp)
        return cls._from_store(store,
                config=config,
                max_workers=max_workers,
                chunksize=chunksize,
                use_threads=use_threads,
                mp_context=mp_context,
                )

    @classmethod
    @doc_inject(selector='batch_constructor')
    def from_hdf5(cls,
            fp: TPathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
            mp_context: tp.Optional[str] = None,
            ) -> 'Batch':
        '''
        Given a file path to a HDF5 :obj:`Batch` store, return a :obj:`Batch` instance.

        {args}
        '''
        store = StoreHDF5(fp)
        return cls._from_store(store,
                config=config,
                max_workers=max_workers,
                chunksize=chunksize,
                use_threads=use_threads,
                mp_context=mp_context,
                )

    #---------------------------------------------------------------------------
    def __init__(self,
            items: TIteratorFrameItems,
            *,
            name: TName = None,
            config: StoreConfigMapInitializer = None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
            mp_context: tp.Optional[str] = None,
            ):
        '''
        Default constructor of a :obj:`Batch`.

        {args}
        '''
        self._items = items # might be a generator!
        self._name = name

        self._config = StoreConfigMap.from_initializer(config)

        self._max_workers = max_workers
        self._chunksize = chunksize
        self._use_threads = use_threads
        self._mp_context = mp_context

    #---------------------------------------------------------------------------
    def _derive(self,
            gen: TGeneratorFrameItems,
            name: TName = None,
            ) -> 'Batch':
        '''Utility for creating derived Batch
        '''
        return self.__class__(gen(),
                name=name if name is not None else self._name,
                config=self._config,
                max_workers=self._max_workers,
                chunksize=self._chunksize,
                use_threads=self._use_threads,
                )

    @property
    def via_container(self) -> 'Batch':
        '''
        Return a new Batch with all values wrapped in either a :obj:`Frame` or :obj:`Series`.
        '''
        def gen() -> TIteratorFrameItems:
            for label, v in self._items:
                yield label, normalize_container(v)
        return self._derive(gen)

    #---------------------------------------------------------------------------
    # name interface

    @property
    @doc_inject()
    def name(self) -> TName:
        '''{}'''
        return self._name

    #---------------------------------------------------------------------------
    @property
    def shapes(self) -> Series[Index[tp.Any], np.object_]:
        '''A :obj:`Series` describing the shape of each iterated :obj:`Frame`.

        Returns:
            :obj:`tp.Tuple[int]`
        '''
        items = ((label, f.shape) for label, f in self._items)
        return Series.from_items(items, name='shape', dtype=DTYPE_OBJECT)


    def display(self,
            config: tp.Optional[DisplayConfig] = None,
            *,
            style_config: tp.Optional[StyleConfig] = None,
            ) -> Display:
        '''Provide a :obj:`Series`-style display of the :obj:`Batch`. Note that if the held iterator is a generator, this display will exhaust the generator.
        '''
        config = config or DisplayActive.get()

        items = ((label, f.__class__) for label, f in self._items)
        series: TSeriesAny = Series.from_items(items, name=self._name)

        display_cls = Display.from_values((),
                header=DisplayHeader(self.__class__, self._name),
                config=config)
        return series._display(config,
                display_cls=display_cls,
                style_config=style_config,
                )

    def __repr__(self) -> str:
        '''Provide a display of the :obj:`Batch` that does not exhaust the generator.
        '''
        if self._name:
            header = f'{self.__class__.__name__}: {self._name}'
        else:
            header = self.__class__.__name__
        return f'<{header} max_workers={self._max_workers}>'

    #---------------------------------------------------------------------------
    # core function application routines

    def _iter_items(self) -> TIteratorFrameItems:
        '''Iter pairs in items, providing helpful exception of a pair is not found. Thies is necessary as we cannot validate the items until we actually do an iteration, and the iterable might be an iterator.
        '''
        for pair in self._items:
            try:
                label, frame = pair
            except ValueError:
                raise BatchIterableInvalid() from None
            yield label, frame

    def _apply_pool(self,
            labels: tp.List[TLabel],
            arg_iter: tp.Iterator[tp.Tuple[tp.Any, ...]],
            caller: tp.Callable[..., TFrameOrSeries],
            ) -> 'Batch':

        pool_executor = get_concurrent_executor(
                use_threads=self._use_threads,
                max_workers=self._max_workers,
                mp_context=self._mp_context,
                )

        def gen_pool() -> TIteratorFrameItems:
            with pool_executor() as executor:
                yield from zip(labels,
                        executor.map(caller, arg_iter, chunksize=self._chunksize)
                        )
        return self._derive(gen_pool)

    def _apply_pool_except(self,
            labels: tp.List[TLabel],
            arg_iter: tp.Iterator[tp.Tuple[tp.Any, ...]],
            caller: tp.Callable[..., TFrameOrSeries],
            exception: tp.Type[Exception],
            ) -> 'Batch':

        if self._chunksize != 1:
            raise NotImplementedError('Cannot use apply_except idioms with chunksize other than 1')

        pool_executor = get_concurrent_executor(
                use_threads=self._use_threads,
                max_workers=self._max_workers,
                mp_context=self._mp_context,
                )

        def gen_pool() -> TIteratorFrameItems:
            futures = []
            with pool_executor() as executor:
                for args in arg_iter:
                    futures.append(executor.submit(caller, args))

                for label, future in zip(labels, futures):
                    try:
                        container = future.result()
                    except exception:
                        continue
                    yield label, container

        return self._derive(gen_pool)

    def _apply_attr(self,
            *args: tp.Any,
            attr: str,
            **kwargs: tp.Any,
            ) -> 'Batch':
        '''
        Apply a method on a Frame given as an attr string.
        '''
        if self._max_workers is None:
            def gen() -> TIteratorFrameItems:
                for label, frame in self._iter_items():
                    yield label, call_attr((frame, attr, args, kwargs))
            return self._derive(gen)

        labels = []
        def arg_gen() -> tp.Iterator[tp.Tuple[TFrameOrSeries, str, tp.Any, tp.Any]]:
            for label, frame in self._iter_items():
                labels.append(label)
                yield frame, attr, args, kwargs

        return self._apply_pool(labels, arg_gen(), call_attr)

    def apply(self, func: TCallableAny) -> 'Batch':
        '''
        Apply a function to each :obj:`Frame` contained in this :obj:`Frame`, where a function is given the :obj:`Frame` as an argument.
        '''
        if self._max_workers is None:
            def gen() -> TIteratorFrameItems:
                for label, frame in self._iter_items():
                    yield label, call_func((frame, func))
            return self._derive(gen)

        labels = []
        def arg_gen() -> tp.Iterator[tp.Tuple[TFrameOrSeries, TCallableAny]]:
            for label, frame in self._iter_items():
                labels.append(label)
                yield frame, func

        return self._apply_pool(labels, arg_gen(), call_func)

    def apply_except(self,
            func: TCallableAny,
            exception: tp.Type[Exception],
            ) -> 'Batch':
        '''
        Apply a function to each :obj:`Frame` contained in this :obj:`Frame`, where a function is given the :obj:`Frame` as an argument. Exceptions raised that matching the `except` argument will be silenced.
        '''
        if self._max_workers is None:
            def gen() -> TIteratorFrameItems:
                for label, frame in self._iter_items():
                    try:
                        yield label, call_func((frame, func))
                    except exception:
                        pass
            return self._derive(gen)

        labels = []
        def arg_gen() -> tp.Iterator[tp.Tuple[TFrameOrSeries, TCallableAny]]:
            for label, frame in self._iter_items():
                labels.append(label)
                yield frame, func

        return self._apply_pool_except(labels,
                arg_gen(),
                call_func,
                exception,
                )

    def apply_items(self, func: TCallableAny) -> 'Batch':
        '''
        Apply a function to each :obj:`Frame` contained in this :obj:`Frame`, where a function is given the pair of label, :obj:`Frame` as an argument.
        '''
        if self._max_workers is None:
            def gen() -> TIteratorFrameItems:
                for label, frame in self._iter_items():
                    yield label, call_func_items((frame, func, label))
            return self._derive(gen)

        labels = []
        def arg_gen() -> tp.Iterator[tp.Tuple[TFrameOrSeries, TCallableAny, TLabel]]:
            for label, frame in self._iter_items():
                labels.append(label)
                yield frame, func, label

        return self._apply_pool(labels, arg_gen(), call_func_items)

    def apply_items_except(self,
            func: TCallableAny,
            exception: tp.Type[Exception],
            ) -> 'Batch':
        '''
        Apply a function to each :obj:`Frame` contained in this :obj:`Frame`, where a function is given the pair of label, :obj:`Frame` as an argument. Exceptions raised that matching the `except` argument will be silenced.
        '''
        if self._max_workers is None:
            def gen() -> TIteratorFrameItems:
                for label, frame in self._iter_items():
                    try:
                        yield label, call_func_items((frame, func, label))
                    except exception:
                        pass
            return self._derive(gen)

        labels = []
        def arg_gen() -> tp.Iterator[tp.Tuple[TFrameOrSeries, TCallableAny, TLabel]]:
            for label, frame in self._iter_items():
                labels.append(label)
                yield frame, func, label

        return self._apply_pool_except(labels,
                arg_gen(),
                call_func_items,
                exception,
                )

    #---------------------------------------------------------------------------
    @property
    def reduce(self) -> InterfaceBatchReduceDispatch:
        '''Return a ``ReduceAligned`` interface, permitting function application per column or on entire containers.
        '''
        return InterfaceBatchReduceDispatch(self.apply)

    #---------------------------------------------------------------------------
    # extraction

    def _extract_iloc(self, key: TILocSelectorCompound) -> 'Batch':
        return self._apply_attr(
                attr='_extract_iloc',
                key=key
                )
    def _extract_loc(self, key: TLocSelectorCompound) -> 'Batch':
        return self._apply_attr(
                attr='_extract_loc',
                key=key
                )

    def _extract_bloc(self, key: TBlocKey) -> 'Batch':
        return self._apply_attr(
                attr='_extract_bloc',
                key=key
                )

    def __getitem__(self, key: TLocSelector) -> 'Batch':
        ''
        return self._apply_attr(
                attr='__getitem__',
                key=key
                )

    #---------------------------------------------------------------------------
    def _drop_iloc(self, key: TILocSelectorCompound) -> 'Batch':
        return self._apply_attr(
                attr='_drop_iloc',
                key=key
                )

    def _drop_loc(self, key: TLocSelectorCompound) -> 'Batch':
        return self._apply_attr(
                attr='_drop_loc',
                key=key
                )

    def _drop_getitem(self, key: TLocSelectorCompound) -> 'Batch':
        return self._apply_attr(
                attr='_drop_getitem',
                key=key
                )

    #---------------------------------------------------------------------------
    # interfaces

    @property
    def loc(self) -> InterGetItemLocCompound['Batch']:
        return InterGetItemLocCompound(self._extract_loc)

    @property
    def iloc(self) -> InterGetItemILocCompound['Batch']:
        return InterGetItemILocCompound(self._extract_iloc)

    @property
    def bloc(self) -> InterfaceGetItemBLoc['Batch']:
        return InterfaceGetItemBLoc(self._extract_bloc)

    @property
    def drop(self) -> InterfaceSelectTrio['Batch']:
        return InterfaceSelectTrio( #type: ignore
            func_iloc=self._drop_iloc,
            func_loc=self._drop_loc,
            func_getitem=self._drop_getitem)

    # NOTE: note sure if assign interfaces would work in this context

    #---------------------------------------------------------------------------
    # dictionary-like interface
    # these methods operate on the Batch itself, not the contained Frames

    def keys(self) -> tp.Iterator[TLabel]:
        '''
        Iterator of :obj:`Frame` labels.
        '''
        for k, _ in self._iter_items():
            yield k

    def __iter__(self) -> tp.Iterator[TLabel]:
        '''
        Iterator of :obj:`Frame` labels, same as :obj:`Batch.keys`.
        '''
        yield from self.keys()

    @property
    def values(self) -> tp.Iterator[TFrameOrSeries]: # type: ignore # NOTE: this violates the supertype
        '''
        Return an iterator of values (:obj:`Frame` or :obj:`Series`) stored in this :obj:`Batch`.
        '''
        return (v for _, v in self._iter_items())

    def items(self) -> TIteratorFrameItems:
        '''
        Iterator of labels, :obj:`Frame`.
        '''
        return self._iter_items()

    _items_store = items

    #---------------------------------------------------------------------------
    # axis and shape ufunc methods

    def _ufunc_unary_operator(self,
            operator: TUFunc
            ) -> 'Batch':
        return self._apply_attr(
                attr='_ufunc_unary_operator',
                operator=operator
                )

    def _ufunc_binary_operator(self, *,
            operator: TUFunc,
            other: tp.Any,
            fill_value: object = np.nan,
            ) -> 'Batch':
        return self._apply_attr(
                attr='_ufunc_binary_operator',
                operator=operator,
                other=other,
                )

    def _ufunc_axis_skipna(self, *,
            axis: int,
            skipna: bool,
            ufunc: TUFunc,
            ufunc_skipna: TUFunc,
            composable: bool,
            dtypes: tp.Tuple[TDtypeAny, ...],
            size_one_unity: bool
            ) -> 'Batch':
        return self._apply_attr(
                attr='_ufunc_axis_skipna',
                axis=axis,
                skipna=skipna,
                ufunc=ufunc,
                ufunc_skipna=ufunc_skipna,
                composable=composable,
                dtypes=dtypes,
                size_one_unity=size_one_unity,
                )

    def _ufunc_shape_skipna(self, *,
            axis: int,
            skipna: bool,
            ufunc: TUFunc,
            ufunc_skipna: TUFunc,
            composable: bool,
            dtypes: tp.Tuple[TDtypeAny, ...],
            size_one_unity: bool
            ) -> 'Batch':

        return self._apply_attr(
                attr='_ufunc_shape_skipna',
                axis=axis,
                skipna=skipna,
                ufunc=ufunc,
                ufunc_skipna=ufunc_skipna,
                composable=composable,
                dtypes=dtypes,
                size_one_unity=size_one_unity,
                )


    #---------------------------------------------------------------------------
    # via interfaces

    @property
    def via_values(self) -> InterfaceBatchValues:
        '''
        Interface for applying a function to values in this container.
        '''
        return InterfaceBatchValues(self.apply)

    @property
    def via_str(self) -> InterfaceBatchString:
        '''
        Interface for applying string methods to elements in this container.
        '''
        return InterfaceBatchString(self.apply)

    @property
    def via_dt(self) -> InterfaceBatchDatetime:
        '''
        Interface for applying datetime properties and methods to elements in this container.
        '''
        return InterfaceBatchDatetime(self.apply)

    @property
    def via_T(self) -> InterfaceBatchTranspose:
        '''
        Interface for using binary operators with one-dimensional sequences, where the opperand is applied column-wise.
        '''
        return InterfaceBatchTranspose(self.apply)

    def via_fill_value(self,
            fill_value: object = np.nan,
            ) -> InterfaceBatchFillValue:
        '''
        Interface for using binary operators and methods with a pre-defined fill value.
        '''
        return InterfaceBatchFillValue(self.apply,
                fill_value=fill_value,
                )

    def via_re(self,
            pattern: str,
            flags: int = 0,
            ) -> InterfaceBatchRe:
        '''
        Interface for applying regular expressions to elements in this container.
        '''
        return InterfaceBatchRe(self.apply,
                pattern=pattern,
                flags=flags,
                )

    #---------------------------------------------------------------------------
    # transformations resulting in the same dimensionality

    @property
    def astype(self) -> InterfaceBatchAsType['Batch']:
        '''
        Return a new Batch with astype transformed.
        '''
        return InterfaceBatchAsType(self.apply)

    def rename(self,
            name: TName = NAME_DEFAULT,
            *,
            index: TName = NAME_DEFAULT,
            columns: TName = NAME_DEFAULT,
            ) -> 'Batch':
        '''
        Return a new Batch with an updated name attribute.
        '''
        return self._apply_attr(
                attr='rename',
                name=name,
                index=index,
                columns=columns,
                )

    def sort_index(self,
            *,
            ascending: bool = True,
            kind: str = DEFAULT_SORT_KIND
            ) -> 'Batch':
        '''
        Return a new :obj:`Batch` with contained :obj;`Frame` ordered by the sorted ``index``.
        '''
        return self._apply_attr(
                attr='sort_index',
                ascending=ascending,
                kind=kind,
                )

    def sort_columns(self,
            *,
            ascending: bool = True,
            kind: str = DEFAULT_SORT_KIND
            ) -> 'Batch':
        '''
        Return a new :obj:`Batch` with contained :obj:`Frame` ordered by the sorted ``columns``.
        '''
        return self._apply_attr(
                attr='sort_columns',
                ascending=ascending,
                kind=kind,
                )

    def sort_values(self,
            label: TKeyOrKeys,
            *,
            ascending: bool = True,
            axis: int = 1,
            kind: str = DEFAULT_SORT_KIND) -> 'Batch':
        '''
        Return a new :obj:`Batch` with contained :obj:`Frame` ordered by the sorted values, where values are given by single column or iterable of columns.

        Args:
            label: a label or iterable of keys.
        '''
        return self._apply_attr(
                attr='sort_values',
                label=label,
                ascending=ascending,
                axis=axis,
                kind=kind,
                )

    def isin(self, other: tp.Any) -> 'Batch':
        '''
        Return a new :obj:`Batch` with contained :obj:`Frame` as a same-sized Boolean :obj:`Frame` that shows if the same-positioned element is in the passed iterable.
        '''
        return self._apply_attr(
                attr='isin',
                other=other,
                )

    @doc_inject(class_name='Batch')
    def clip(self, *,
            lower: tp.Optional[tp.Union[float, TSeriesAny, TFrameAny]] = None,
            upper: tp.Optional[tp.Union[float, TSeriesAny, TFrameAny]] = None,
            axis: tp.Optional[int] = None
            ) -> 'Batch':
        '''{}

        Args:
            lower: value, :obj:`Series`, :obj:`Frame`
            upper: value, :obj:`Series`, :obj:`Frame`
            axis: required if ``lower`` or ``upper`` are given as a :obj:`Series`.
        '''
        return self._apply_attr(
                attr='clip',
                lower=lower,
                upper=upper,
                axis=axis,
                )

    def transpose(self) -> 'Batch':
        '''Transpose. Return a :obj:`Frame` with ``index`` as ``columns`` and vice versa.
        '''
        return self._apply_attr(attr='transpose')

    @property
    def T(self) -> 'Batch':
        '''Transpose. Return a :obj:`Frame` with ``index`` as ``columns`` and vice versa.
        '''
        return self._apply_attr(attr='transpose')



    @doc_inject(selector='duplicated')
    def duplicated(self, *,
            axis: int = 0,
            exclude_first: bool = False,
            exclude_last: bool = False) -> 'Batch':
        '''
        Return an axis-sized Boolean :obj:`Series` that shows True for all rows (axis 0) or columns (axis 1) duplicated.

        Args:
            {axis}
            {exclude_first}
            {exclude_last}
        '''
        return self._apply_attr(
                attr='duplicated',
                axis=axis,
                exclude_first=exclude_first,
                exclude_last=exclude_last,
                )

    @doc_inject(selector='duplicated')
    def drop_duplicated(self, *,
            axis: int = 0,
            exclude_first: bool = False,
            exclude_last: bool = False
            ) -> 'Batch':
        '''
        Return a :obj:`Batch` with contained :obj:`Frame` with duplicated rows (axis 0) or columns (axis 1) removed. All values in the row or column are compared to determine duplication.

        Args:
            {axis}
            {exclude_first}
            {exclude_last}
        '''
        return self._apply_attr(
                attr='drop_duplicated',
                axis=axis,
                exclude_first=exclude_first,
                exclude_last=exclude_last,
                )

    # as only useful on Frame, perhaps skip?
    # def set_index(self,
    # def set_index_hierarchy(self,
    # def unset_index(self, *,

    def __round__(self, decimals: int = 0) -> 'Batch':
        '''
        Return a :obj:`Batch` with contained :obj:`Frame` rounded to the given decimals. Negative decimals round to the left of the decimal point.

        Args:
            decimals: number of decimals to round to.
        '''
        return self._apply_attr(
                attr='__round__',
                decimals=decimals,
                )

    def roll(self,
            index: int = 0,
            columns: int = 0,
            *,
            include_index: bool = False,
            include_columns: bool = False,
            ) -> 'Batch':
        '''
        Roll columns and/or rows by positive or negative integer counts, where columns and/or rows roll around the axis.

        Args:
            include_index: Determine if index is included in index-wise rotation.
            include_columns: Determine if column index is included in index-wise rotation.
        '''
        return self._apply_attr(
                attr='roll',
                index=index,
                columns=columns,
                include_index=include_index,
                include_columns=include_columns,
                )

    def shift(self,
            index: int = 0,
            columns: int = 0,
            fill_value: tp.Any = np.nan) -> 'Batch':
        '''
        Shift columns and/or rows by positive or negative integer counts, where columns and/or rows fall of the axis and introduce missing values, filled by `fill_value`.
        '''
        return self._apply_attr(
                attr='shift',
                index=index,
                columns=columns,
                fill_value=fill_value,
                )

    # ---------------------------------------------------------------------------
    # na handling
    def isna(self) -> 'Batch':
        '''
        Return a :obj:`Batch` with contained, same-indexed :obj:`Frame` indicating True which values are NaN or None.
        '''
        return self._apply_attr(attr='isna')

    def notna(self) -> 'Batch':
        '''
        Return a :obj:`Batch` with contained, same-indexed :obj:`Frame` indicating True which values are not NaN or None.
        '''
        return self._apply_attr(attr='notna')

    def dropna(
            self,
            axis: int = 0, condition: tp.Callable[[TNDArrayAny], bool] = np.all,
            ) -> 'Batch':
        '''
        Return a :obj:`Batch` with contained :obj:`Frame` after removing rows (axis 0) or columns (axis 1) where any or all values are NA (NaN or None). The condition is determined by a NumPy ufunc that process the Boolean array returned by ``isna()``; the default is ``np.all``.

        Args:
            axis:
            condition:
        '''
        return self._apply_attr(
                attr='dropna',
                axis=axis,
                condition=condition
                )

    # ---------------------------------------------------------------------------
    # falsy handling
    def isfalsy(self) -> 'Batch':
        '''
        Return a :obj:`Batch` with contained, same-indexed :obj:`Frame` indicating True which values are Falsy.
        '''
        return self._apply_attr(attr='isfalsy')

    def notfalsy(self) -> 'Batch':
        '''
        Return a :obj:`Batch` with contained, same-indexed :obj:`Frame` indicating True which values are not Falsy.
        '''
        return self._apply_attr(attr='notfalsy')

    def dropfalsy(self,
            axis: int = 0, condition: tp.Callable[[TNDArrayAny], bool] = np.all,
            ) -> 'Batch':
        '''
        Return a :obj:`Batch` with contained :obj:`Frame` after removing rows (axis 0) or columns (axis 1) where any or all values are NA (NaN or None). The condition is determined by a NumPy ufunc that process the Boolean array returned by ``isna()``; the default is ``np.all``.

        Args:
            axis:
            condition:
        '''
        return self._apply_attr(
                attr='dropfalsy',
                axis=axis,
                condition=condition
                )


    # ---------------------------------------------------------------------------
    # na filling

    def fillna(self,
            value: tp.Any
            ) -> 'Batch':
        '''
        Return a new :obj:`Batch` with contained :obj:`Frame` after filling null (NaN or None) with the provided ``value``.
        '''
        return self._apply_attr(
                attr='fillna',
                value=value,
                )

    def fillna_leading(self,
            value: tp.Any,
            *,
            axis: int = 0
            ) -> 'Batch':
        '''
        Return a new :obj:`Batch` with contained :obj:`Frame` after filling leading (and only leading) null (NaN or None) with the provided ``value``.

        Args:
            {value}
            {axis}
        '''

        return self._apply_attr(
                attr='fillna_leading',
                value=value,
                axis=axis
                )

    def fillna_trailing(self,
            value: tp.Any,
            *,
            axis: int = 0,
            ) -> 'Batch':
        '''
        Return a new :obj:`Batch` with contained :obj:`Frame` after filling trailing (and only trailing) null (NaN or None) with the provided ``value``.

        Args:
            {value}
            {axis}
        '''
        return self._apply_attr(
            attr='fillna_trailing',
            value=value,
            axis=axis
            )

    def fillna_forward(self,
            limit: int = 0,
            *,
            axis: int = 0,
            ) -> 'Batch':
        '''
        Return a new :obj:`Batch` with contained :obj:`Frame` after filling forward null (NaN or None) with the last observed value.

        Args:
            {limit}
            {axis}
        '''
        return self._apply_attr(
            attr='fillna_forward',
            limit=limit,
            axis=axis,
            )

    def fillna_backward(self,
            limit: int = 0,
            *,
            axis: int = 0,
            ) -> 'Batch':
        '''
        Return a new :obj:`Batch` with contained :obj:`Frame` after filling backward null (NaN or None) with the first observed value.

        Args:
            {limit}
            {axis}
        '''
        return self._apply_attr(
            attr='fillna_backward',
            limit=limit,
            axis=axis,
            )

    # ---------------------------------------------------------------------------
    # falsy filling

    def fillfalsy(self,
            value: tp.Any
            ) -> 'Batch':
        '''
        Return a new :obj:`Batch` with contained :obj:`Frame` after filling falsy values with the provided ``value``.
        '''
        return self._apply_attr(
                attr='fillfalsy',
                value=value,
                )

    def fillfalsy_leading(self,
            value: tp.Any,
            *,
            axis: int = 0,
            ) -> 'Batch':
        '''
        Return a new :obj:`Batch` with contained :obj:`Frame` after filling leading (and only leading) falsy values with the provided ``value``.

        Args:
            {value}
            {axis}
        '''
        return self._apply_attr(
            attr='fillfalsy_leading',
            value=value,
            axis=axis,
            )

    def fillfalsy_trailing(self,
            value: tp.Any,
            *,
            axis: int = 0,
            ) -> 'Batch':
        '''
        Return a new :obj:`Batch` with contained :obj:`Frame` after filling trailing (and only trailing) falsy values with the provided ``value``.

        Args:
            {value}
            {axis}
        '''
        return self._apply_attr(
            attr='fillfalsy_trailing',
            value=value,
            axis=axis,
            )

    def fillfalsy_forward(self,
            limit: int = 0,
            axis: int = 0,
            ) -> 'Batch':
        '''
        Return a new :obj:`Batch` with contained :obj:`Frame` after filling forward falsy values with the last observed value.

        Args:
            {limit}
            {axis}
        '''
        return self._apply_attr(
            attr='fillfalsy_forward',
            limit=limit,
            axis=axis,
            )

    def fillfalsy_backward(self,
            limit: int = 0,
            *,
            axis: int = 0
            ) -> 'Batch':
        '''
        Return a new :obj:`Batch` with contained :obj:`Frame` after filling backward falsy values with the first observed value.

        Args:
            {limit}
            {axis}
        '''
        return self._apply_attr(
            attr='fillfalsy_backward',
            limit=limit,
            axis=axis,
            )


    # ---------------------------------------------------------------------------
    # index and relabel
    def relabel(self,
            index: tp.Optional[TRelabelInput] = None,
            columns: tp.Optional[TRelabelInput] = None,
            *,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            ) -> 'Batch':

        return self._apply_attr(
            attr='relabel',
            index=index,
            columns=columns,
            index_constructor=index_constructor,
            columns_constructor=columns_constructor,
            )

    def unset_index(self,
            *,
            names: tp.Iterable[TLabel] = (),
            consolidate_blocks: bool = False,
            columns_constructors: TIndexCtorSpecifiers = None
            ) -> 'Batch':

        return self._apply_attr(
            attr='unset_index',
            names=names,
            consolidate_blocks=consolidate_blocks,
            columns_constructors=columns_constructors
            )

    def reindex(self,
            index: tp.Optional[TIndexInitializer] = None,
            columns: tp.Optional[TIndexInitializer] = None,
            *,
            fill_value: object = np.nan,
            own_index: bool = False,
            own_columns: bool = False,
            check_equals: bool = True,
            ) -> 'Batch':

        return self._apply_attr(
            attr='reindex',
            index=index,
            columns=columns,
            fill_value=fill_value,
            own_index=own_index,
            own_columns=own_columns,
            check_equals=check_equals,
            )

    def relabel_flat(self,
            index: bool = False,
            columns: bool = False,
            ) -> 'Batch':

        return self._apply_attr(
            attr='relabel_flat',
            index=index,
            columns=columns
            )

    def relabel_level_add(self,
            index: TLabel = None,
            columns: TLabel = None,
            *,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None
            ) -> 'Batch':

        return self._apply_attr(
            attr='relabel_level_add',
            index=index,
            columns=columns,
            index_constructor=index_constructor,
            columns_constructor=columns_constructor,
        )

    def relabel_level_drop(self,
            index: int = 0,
            columns: int = 0
            ) -> 'Batch':

        return self._apply_attr(
            attr='relabel_level_drop',
            index=index,
            columns=columns
            )

    def relabel_shift_in(self,
            key: TLocSelector,
            *,
            axis: int = 0,
            ) -> 'Batch':

        return self._apply_attr(
            attr='relabel_shift_in',
            key=key,
            axis=axis
            )

    # ---------------------------------------------------------------------------
    # rank

    def rank_ordinal(self,
            *,
            axis: int = 0,
            skipna: bool = True,
            ascending: TBoolOrBools = True,
            start: int = 0,
            fill_value: tp.Any = np.nan
            ) -> 'Batch':

        return self._apply_attr(
            attr='rank_ordinal',
            axis=axis,
            skipna=skipna,
            ascending=ascending,
            start=start,
            fill_value=fill_value
            )

    def rank_dense(self,
            *,
            axis: int = 0,
            skipna: bool = True,
            ascending: TBoolOrBools = True,
            start: int = 0,
            fill_value: tp.Any = np.nan
            ) -> 'Batch':

        return self._apply_attr(
            attr='rank_dense',
            axis=axis,
            skipna=skipna,
            ascending=ascending,
            start=start,
            fill_value=fill_value
            )

    def rank_min(self,
            *,
            axis: int = 0,
            skipna: bool = True,
            ascending: TBoolOrBools = True,
            start: int = 0,
            fill_value: tp.Any = np.nan
            ) -> 'Batch':

        return self._apply_attr(
            attr='rank_min',
            axis=axis,
            skipna=skipna,
            ascending=ascending,
            start=start,
            fill_value=fill_value
            )

    def rank_max(self,
            *,
            axis: int = 0,
            skipna: bool = True,
            ascending: TBoolOrBools = True,
            start: int = 0,
            fill_value: tp.Any = np.nan
            ) -> 'Batch':

        return self._apply_attr(
            attr='rank_max',
            axis=axis,
            skipna=skipna,
            ascending=ascending,
            start=start,
            fill_value=fill_value
            )

    def rank_mean(self,
            *,
            axis: int = 0,
            skipna: bool = True,
            ascending: TBoolOrBools = True,
            start: int = 0,
            fill_value: tp.Any = np.nan
            ) -> 'Batch':

        return self._apply_attr(
            attr='rank_mean',
            axis=axis,
            skipna=skipna,
            ascending=ascending,
            start=start,
            fill_value=fill_value
            )

    #---------------------------------------------------------------------------
    # transformations resulting in changed dimensionality

    def count(self, *,
            skipna: bool = True,
            skipfalsy: bool = False,
            unique: bool = False,
            axis: int = 0,
            ) -> 'Batch':
        '''Apply count on contained Frames.
        '''
        return self._apply_attr(
                attr='count',
                skipna=skipna,
                skipfalsy=skipfalsy,
                unique=unique,
                axis=axis,
                )

    @doc_inject(selector='sample')
    def sample(self,
            index: tp.Optional[int] = None,
            columns: tp.Optional[int] = None,
            *,
            seed: tp.Optional[int] = None,
            ) -> 'Batch':
        '''Apply sample on contained Frames.

        Args:
            {index}
            {columns}
            {seed}
        '''
        return self._apply_attr(
                attr='sample',
                index=index,
                columns=columns,
                seed=seed,
                )

    @doc_inject(selector='head', class_name='Batch')
    def head(self, count: int = 5) -> 'Batch':
        '''{doc}

        Args:
            {count}
        '''
        return self._apply_attr(
                attr='head',
                count=count,
                )

    @doc_inject(selector='tail', class_name='Batch')
    def tail(self, count: int = 5) -> 'Batch':
        '''{doc}

        Args:
            {count}
        '''
        return self._apply_attr(
                attr='tail',
                count=count,
                )

    @doc_inject(selector='argminmax')
    def loc_min(self, *,
            skipna: bool = True,
            axis: int = 0
            ) -> 'Batch':
        '''
        Return the labels corresponding to the minimum value found.

        Args:
            {skipna}
            {axis}
        '''
        return self._apply_attr(
                attr='loc_min',
                skipna=skipna,
                axis=axis,
                )

    @doc_inject(selector='argminmax')
    def iloc_min(self, *,
            skipna: bool = True,
            axis: int = 0
            ) -> 'Batch':
        '''
        Return the integer indices corresponding to the minimum values found.

        Args:
            {skipna}
            {axis}
        '''
        return self._apply_attr(
                attr='iloc_min',
                skipna=skipna,
                axis=axis,
                )

    @doc_inject(selector='argminmax')
    def loc_max(self, *,
            skipna: bool = True,
            axis: int = 0
            ) -> 'Batch':
        '''
        Return the labels corresponding to the maximum values found.

        Args:
            {skipna}
            {axis}
        '''
        return self._apply_attr(
                attr='loc_max',
                skipna=skipna,
                axis=axis,
                )

    @doc_inject(selector='argminmax')
    def iloc_max(self, *,
            skipna: bool = True,
            axis: int = 0
            ) -> 'Batch':
        '''
        Return the integer indices corresponding to the maximum values found.

        Args:
            {skipna}
            {axis}
        '''
        return self._apply_attr(
                attr='iloc_max',
                skipna=skipna,
                axis=axis,
                )

    def cov(self, *,
            axis: int = 1,
            ddof: int = 1,
            ) -> 'Batch':
        '''
        Compute a covariance matrix.

        Args:
            axis: if 0, each row represents a variable, with observations as columns; if 1, each column represents a variable, with observations as rows. Defaults to 1.
            ddof: Delta degrees of freedom, defaults to 1.
        '''
        return self._apply_attr(
                attr='cov',
                axis=axis,
                ddof=ddof,
                )

    def corr(self, *,
            axis: int = 1,
            ) -> 'Batch':
        '''
        Compute a correlation matrix.

        Args:
            axis: if 0, each row represents a variable, with observations as columns; if 1, each column represents a variable, with observations as rows. Defaults to 1.
        '''
        return self._apply_attr(
                attr='corr',
                axis=axis,
                )

    #---------------------------------------------------------------------------
    # utility function to numpy array

    def unique(self, *,
            axis: tp.Optional[int] = None,
            ) -> 'Batch':
        '''
        Return a NumPy array of unqiue values. If the axis argument is provied, uniqueness is determined by columns or row.

        '''
        return self._apply_attr(
                attr='unique',
                axis=axis,
                )

    #---------------------------------------------------------------------------
    # exporter

    def to_series(self, *,
        dtype: TDtypeSpecifier = None,
        name: TName = None,
        index_constructor: TIndexCtorSpecifier = None
        ) -> TSeriesAny:
        '''
        Consolidate stored values into a new :obj:`Series` using the stored labels as the index.
        '''
        return Series.from_items(self._items,
                dtype=dtype,
                name=name,
                index_constructor=index_constructor,
                )

    def to_frame(self, *,
            axis: int = 0,
            union: bool = True,
            index: tp.Optional[tp.Union[TIndexInitializer, TIndexAutoFactory]] = None,
            columns: tp.Optional[tp.Union[TIndexInitializer, TIndexAutoFactory]] = None,
            index_constructor: TIndexCtorSpecifier = None,
            columns_constructor: TIndexCtorSpecifier = None,
            name: TName = None,
            fill_value: object = np.nan,
            consolidate_blocks: bool = False
        ) -> TFrameAny:
        '''
        Consolidate stored :obj:`Frame` into a new :obj:`Frame` using the stored labels as the index on the provided ``axis`` using :obj:`Frame.from_concat`. This assumes that that the contained :obj:`Frame` have been reduced to a single dimension along the provided `axis`.
        '''
        labels = []
        containers: tp.List[TFrameOrSeries] = []
        ndim1d = True
        for label, container in self._items:
            container = normalize_container(container)
            labels.append(label)
            ndim1d &= container.ndim == 1
            containers.append(container)

        name = name if name is not None else self._name
        if ndim1d:
            if axis == 0 and index is None:
                index = labels
            if axis == 1 and columns is None:
                columns = labels
            return Frame.from_concat(
                    containers,
                    axis=axis,
                    union=union,
                    index=index,
                    columns=columns,
                    index_constructor=index_constructor,
                    columns_constructor=columns_constructor,
                    name=name,
                    fill_value=fill_value,
                    consolidate_blocks=consolidate_blocks,
                    )
        # produce a hierarchical index to return all Frames
        f: TFrameAny = Frame.from_concat_items(
                zip(labels, containers),
                axis=axis,
                union=union,
                name=name,
                fill_value=fill_value,
                consolidate_blocks=consolidate_blocks,
                index_constructor=index_constructor,
                columns_constructor=columns_constructor,
                )
        if index is not None or columns is not None:
            # this relabels, as that is how Frame.from_concat works
            # NOTE: we need to apply index_constructor, columns_constructors if defined
            f = f.relabel(index=index, columns=columns)
        return f

    def to_bus(self,
            *,
            index_constructor: TIndexCtorSpecifier = None,
            ) -> TBusAny:
        '''Realize the :obj:`Batch` as an :obj:`Bus`. Note that, as a :obj:`Bus` must have all labels (even if :obj:`Frame` are loaded lazily), this :obj:`Batch` will be exhausted.
        '''
        frames: tp.List[TFrameAny] = []
        index = []
        for i, f in self.items():
            index.append(i)
            if isinstance(f, Series):
                frames.append(f.to_frame())
            else:
                frames.append(f)

        return Bus(frames,
                index=index,
                index_constructor=index_constructor,
                config=self._config,
                name=self._name,
                )

    def _to_signature_bytes(self,
            include_name: bool = True,
            include_class: bool = True,
            encoding: str = 'utf-8',
            ) -> bytes:

        return self.to_bus()._to_signature_bytes(
                include_name=include_name,
                include_class=include_class,
                encoding=encoding,
                )


doc_update(Batch.__init__, selector='batch_init')
