import typing as tp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from static_frame.core.bus import Bus
from static_frame.core.container import ContainerOperand
from static_frame.core.display import Display
from static_frame.core.display import DisplayActive
from static_frame.core.display import DisplayHeader
from static_frame.core.display_config import DisplayConfig
from static_frame.core.doc_str import doc_inject
from static_frame.core.frame import Frame
from static_frame.core.index_auto import IndexAutoFactoryType
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
from static_frame.core.util import AnyCallable
from static_frame.core.util import Bloc2DKeyType
from static_frame.core.util import DEFAULT_SORT_KIND
from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import ELEMENT_TUPLE
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import GetItemKeyTypeCompound
from static_frame.core.util import IndexInitializer
from static_frame.core.util import KeyOrKeys
from static_frame.core.util import NameType
from static_frame.core.util import PathSpecifier
from static_frame.core.util import UFunc
from static_frame.core.style_config import StyleConfig


FrameOrSeries = tp.Union[Frame, Series]
IteratorFrameItems = tp.Iterator[tp.Tuple[tp.Hashable, FrameOrSeries]]
GeneratorFrameItems = tp.Callable[..., IteratorFrameItems]


#-------------------------------------------------------------------------------
# family of executor functions normalized in signature (taking a single tuple of args) for usage in processor pool calls

def normalize_container(post: tp.Any
        ) -> FrameOrSeries:
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

def call_func(bundle: tp.Tuple[FrameOrSeries, AnyCallable]
        ) -> FrameOrSeries:
    container, func = bundle
    return normalize_container(func(container))

def call_func_items(bundle: tp.Tuple[FrameOrSeries, AnyCallable, tp.Hashable]
        ) -> FrameOrSeries:
    container, func, label = bundle
    return normalize_container(func(label, container))

def call_attr(bundle: tp.Tuple[FrameOrSeries, str, tp.Any, tp.Any]
        ) -> FrameOrSeries:
    container, attr, args, kwargs = bundle
    func = getattr(container, attr)
    return normalize_container(func(*args, **kwargs))

#-------------------------------------------------------------------------------
class Batch(ContainerOperand, StoreClientMixin):
    '''
    A lazy, sequentially evaluated container of :obj:`Frame` that broadcasts operations on contained :obj:`Frame` by return new :obj:`Batch` instances. Full evaluation of operations only occurs when iterating or calling an exporter.
    '''

    __slots__ = (
            '_items',
            '_name',
            '_config',
            '_max_workers',
            '_chunksize',
            '_use_threads',
            )

    _config: StoreConfigMap

    @classmethod
    def from_frames(cls,
            frames: tp.Iterable[Frame],
            *,
            name: NameType = None,
            config: StoreConfigMapInitializer = None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
            ) -> 'Batch':
        '''Return a :obj:`Batch` from an iterable of :obj:`Frame`; labels will be drawn from :obj:`Frame.name`.
        '''
        return cls(((f.name, f) for f in frames),
                name=name,
                config=config,
                max_workers=max_workers,
                chunksize=chunksize,
                use_threads=use_threads,
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
            ) -> 'Batch':
        config_map = StoreConfigMap.from_initializer(config)

        items = ((label, store.read(label, config=config_map[label]))
                for label in store.labels(config=config_map))

        return cls(items,
                config=config,
                max_workers=max_workers,
                chunksize=chunksize,
                use_threads=use_threads,
                                )


    @classmethod
    @doc_inject(selector='batch_constructor')
    def from_zip_tsv(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
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
                                )

    @classmethod
    @doc_inject(selector='batch_constructor')
    def from_zip_csv(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
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
                                )

    @classmethod
    @doc_inject(selector='batch_constructor')
    def from_zip_pickle(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
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
                                )


    @classmethod
    @doc_inject(selector='batch_constructor')
    def from_zip_parquet(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
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
                                )


    @classmethod
    @doc_inject(selector='batch_constructor')
    def from_xlsx(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
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
                                )


    @classmethod
    @doc_inject(selector='batch_constructor')
    def from_sqlite(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
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
                                )


    @classmethod
    @doc_inject(selector='batch_constructor')
    def from_hdf5(cls,
            fp: PathSpecifier,
            *,
            config: StoreConfigMapInitializer = None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
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
                                )

    #---------------------------------------------------------------------------

    @doc_inject(selector='batch_init')
    def __init__(self,
            items: IteratorFrameItems,
            *,
            name: NameType = None,
            config: StoreConfigMapInitializer = None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
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

    #---------------------------------------------------------------------------
    def _derive(self,
            gen: GeneratorFrameItems,
            name: NameType = None,
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

    #---------------------------------------------------------------------------
    # name interface

    @property #type: ignore
    @doc_inject()
    def name(self) -> NameType:
        '''{}'''
        return self._name

    def rename(self, name: NameType) -> 'Batch':
        '''
        Return a new Batch with an updated name attribute.
        '''
        def gen() -> IteratorFrameItems:
            yield from self._items
        return self._derive(gen, name=name)

    #---------------------------------------------------------------------------
    @property
    def shapes(self) -> Series:
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
        series = Series.from_items(items, name=self._name)

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
        return f'<{header} at {hex(id(self))}>'

    #---------------------------------------------------------------------------
    # core function application routines

    def _apply_pool(self,
            labels: tp.List[tp.Hashable],
            arg_iter: tp.Iterator[tp.Tuple[tp.Any, ...]],
            caller: tp.Callable[..., FrameOrSeries],
            ) -> 'Batch':

        pool_executor = ThreadPoolExecutor if self._use_threads else ProcessPoolExecutor

        def gen_pool() -> IteratorFrameItems:
            with pool_executor(max_workers=self._max_workers) as executor:
                yield from zip(labels,
                        executor.map(caller, arg_iter, chunksize=self._chunksize)
                        )
        return self._derive(gen_pool)

    def _apply_pool_except(self,
            labels: tp.List[tp.Hashable],
            arg_iter: tp.Iterator[tp.Tuple[tp.Any, ...]],
            caller: tp.Callable[..., FrameOrSeries],
            exception: tp.Type[Exception],
            ) -> 'Batch':

        if self._chunksize != 1:
            raise NotImplementedError('Cannot use apply_except idioms with chunksize other than 1')

        pool_executor = ThreadPoolExecutor if self._use_threads else ProcessPoolExecutor

        def gen_pool() -> IteratorFrameItems:
            futures = []
            with pool_executor(max_workers=self._max_workers) as executor:
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
            def gen() -> IteratorFrameItems:
                for label, frame in self._items:
                    yield label, call_attr((frame, attr, args, kwargs))
            return self._derive(gen)

        labels = []
        def arg_gen() -> tp.Iterator[tp.Tuple[FrameOrSeries, str, tp.Any, tp.Any]]:
            for label, frame in self._items:
                labels.append(label)
                yield frame, attr, args, kwargs

        return self._apply_pool(labels, arg_gen(), call_attr)

    def apply(self, func: AnyCallable) -> 'Batch':
        '''
        Apply a function to each :obj:`Frame` contained in this :obj:`Frame`, where a function is given the :obj:`Frame` as an argument.
        '''
        if self._max_workers is None:
            def gen() -> IteratorFrameItems:
                for label, frame in self._items:
                    yield label, call_func((frame, func))
            return self._derive(gen)

        labels = []
        def arg_gen() -> tp.Iterator[tp.Tuple[FrameOrSeries, AnyCallable]]:
            for label, frame in self._items:
                labels.append(label)
                yield frame, func

        return self._apply_pool(labels, arg_gen(), call_func)

    def apply_except(self,
            func: AnyCallable,
            exception: tp.Type[Exception],
            ) -> 'Batch':
        '''
        Apply a function to each :obj:`Frame` contained in this :obj:`Frame`, where a function is given the :obj:`Frame` as an argument. Exceptions raised that matching the `except` argument will be silenced.
        '''
        if self._max_workers is None:
            def gen() -> IteratorFrameItems:
                for label, frame in self._items:
                    try:
                        yield label, call_func((frame, func))
                    except exception:
                        pass
            return self._derive(gen)

        labels = []
        def arg_gen() -> tp.Iterator[tp.Tuple[FrameOrSeries, AnyCallable]]:
            for label, frame in self._items:
                labels.append(label)
                yield frame, func

        return self._apply_pool_except(labels,
                arg_gen(),
                call_func,
                exception,
                )

    def apply_items(self, func: AnyCallable) -> 'Batch':
        '''
        Apply a function to each :obj:`Frame` contained in this :obj:`Frame`, where a function is given the pair of label, :obj:`Frame` as an argument.
        '''
        if self._max_workers is None:
            def gen() -> IteratorFrameItems:
                for label, frame in self._items:
                    yield label, call_func_items((frame, func, label))
            return self._derive(gen)

        labels = []
        def arg_gen() -> tp.Iterator[tp.Tuple[FrameOrSeries, AnyCallable, tp.Hashable]]:
            for label, frame in self._items:
                labels.append(label)
                yield frame, func, label

        return self._apply_pool(labels, arg_gen(), call_func_items)

    def apply_items_except(self,
            func: AnyCallable,
            exception: tp.Type[Exception],
            ) -> 'Batch':
        '''
        Apply a function to each :obj:`Frame` contained in this :obj:`Frame`, where a function is given the pair of label, :obj:`Frame` as an argument. Exceptions raised that matching the `except` argument will be silenced.
        '''
        if self._max_workers is None:
            def gen() -> IteratorFrameItems:
                for label, frame in self._items:
                    try:
                        yield label, call_func_items((frame, func, label))
                    except exception:
                        pass
            return self._derive(gen)

        labels = []
        def arg_gen() -> tp.Iterator[tp.Tuple[FrameOrSeries, AnyCallable, tp.Hashable]]:
            for label, frame in self._items:
                labels.append(label)
                yield frame, func, label

        return self._apply_pool_except(labels,
                arg_gen(),
                call_func_items,
                exception,
                )

    #---------------------------------------------------------------------------
    # extraction

    def _extract_iloc(self, key: GetItemKeyTypeCompound) -> 'Batch':
        return self._apply_attr(
                attr='_extract_iloc',
                key=key
                )
    def _extract_loc(self, key: GetItemKeyTypeCompound) -> 'Batch':
        return self._apply_attr(
                attr='_extract_loc',
                key=key
                )

    def _extract_bloc(self, key: Bloc2DKeyType) -> 'Batch':
        return self._apply_attr(
                attr='_extract_bloc',
                key=key
                )

    def __getitem__(self, key: GetItemKeyType) -> 'Batch':
        ''
        return self._apply_attr(
                attr='__getitem__',
                key=key
                )

    #---------------------------------------------------------------------------
    def _drop_iloc(self, key: GetItemKeyTypeCompound) -> 'Batch':
        return self._apply_attr(
                attr='_drop_iloc',
                key=key
                )

    def _drop_loc(self, key: GetItemKeyTypeCompound) -> 'Batch':
        return self._apply_attr(
                attr='_drop_loc',
                key=key
                )

    def _drop_getitem(self, key: GetItemKeyTypeCompound) -> 'Batch':
        return self._apply_attr(
                attr='_drop_getitem',
                key=key
                )

    #---------------------------------------------------------------------------
    # interfaces

    @property
    def loc(self) -> InterfaceGetItem['Batch']:
        return InterfaceGetItem(self._extract_loc)

    @property
    def iloc(self) -> InterfaceGetItem['Batch']:
        return InterfaceGetItem(self._extract_iloc)

    @property
    def bloc(self) -> InterfaceGetItem['Batch']:
        return InterfaceGetItem(self._extract_bloc)

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

    def keys(self) -> tp.Iterator[tp.Hashable]:
        '''
        Iterator of :obj:`Frame` labels.
        '''
        for k, _ in self._items:
            yield k

    def __iter__(self) -> tp.Iterator[tp.Hashable]:
        '''
        Iterator of :obj:`Frame` labels, same as :obj:`Batch.keys`.
        '''
        yield from self.keys()

    @property
    def values(self) -> tp.Iterator[FrameOrSeries]:
        '''
        Return an iterator of values (:obj:`Frame` or :obj:`Series`) stored in this :obj:`Batch`.
        '''
        return (v for _, v in self._items)

    def items(self) -> IteratorFrameItems:
        '''
        Iterator of labels, :obj:`Frame`.
        '''
        return self._items.__iter__()

    _items_store = items

    #---------------------------------------------------------------------------
    # axis and shape ufunc methods

    def _ufunc_unary_operator(self,
            operator: UFunc
            ) -> 'Batch':
        return self._apply_attr(
                attr='_ufunc_unary_operator',
                operator=operator
                )

    def _ufunc_binary_operator(self, *,
            operator: UFunc,
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
            ufunc: UFunc,
            ufunc_skipna: UFunc,
            composable: bool,
            dtypes: tp.Tuple[np.dtype, ...],
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
            ufunc: UFunc,
            ufunc_skipna: UFunc,
            composable: bool,
            dtypes: tp.Tuple[np.dtype, ...],
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
    # transformations resulting in the same dimensionality

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
            label: KeyOrKeys,
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
            lower: tp.Optional[tp.Union[float, Series, Frame]] = None,
            upper: tp.Optional[tp.Union[float, Series, Frame]] = None,
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

    #---------------------------------------------------------------------------
    # transformations resulting in changed dimensionality

    def count(self, *,
            skipna: bool = True,
            axis: int = 0,
            ) -> 'Batch':
        '''Apply count on contained Frames.
        '''
        return self._apply_attr(
                attr='count',
                skipna=skipna,
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

    def to_frame(self, *,
            axis: int = 0,
            union: bool = True,
            index: tp.Optional[tp.Union[IndexInitializer, IndexAutoFactoryType]] = None,
            columns: tp.Optional[tp.Union[IndexInitializer, IndexAutoFactoryType]] = None,
            name: NameType = None,
            fill_value: object = np.nan,
            consolidate_blocks: bool = False
        ) -> Frame:
        '''
        Consolidate stored :obj:`Frame` into a new :obj:`Frame` using the stored labels as the index on the provided ``axis`` using :obj:`Frame.from_concat`. This assumes that that the contained :obj:`Frame` have been reduced to single dimension along the provided `axis`.
        '''
        labels = []
        containers: tp.List[FrameOrSeries] = []
        ndim1d = True
        for label, container in self._items:
            labels.append(label)
            ndim1d &= container.ndim == 1
            containers.append(container)

        name = name if name is not None else self._name

        if ndim1d:
            if axis == 0 and index is None:
                index = labels
            if axis == 1 and columns is None:
                columns = labels
            # import ipdb; ipdb.set_trace()
            return Frame.from_concat( #type: ignore
                    containers,
                    axis=axis,
                    union=union,
                    index=index,
                    columns=columns,
                    name=name,
                    fill_value=fill_value,
                    consolidate_blocks=consolidate_blocks,
                    )
        # produce a hierarchical index to return all Frames
        f = Frame.from_concat_items(
                zip(labels, containers),
                axis=axis,
                union=union,
                name=name,
                fill_value=fill_value,
                consolidate_blocks=consolidate_blocks,
                )
        if index is not None or columns is not None:
            # this relabels, as that is how Frame.from_concat works
            f = f.relabel(index=index, columns=columns)
        return f

    def to_bus(self) -> 'Bus':
        '''Realize the :obj:`Batch` as an :obj:`Bus`. Note that, as a :obj:`Bus` must have all labels (even if :obj:`Frame` are loaded lazily), this :obj:`Batch` will be exhausted.
        '''
        series = Series.from_items(
                self.items(),
                name=self._name,
                dtype=DTYPE_OBJECT)

        return Bus(series, config=self._config)


