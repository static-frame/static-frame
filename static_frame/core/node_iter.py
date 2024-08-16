'''
Tools for iterators in Series and Frame. These components are imported by both series.py and frame.py; these components also need to be able to return Series and Frame, and thus use deferred, function-based imports.
'''
from __future__ import annotations

from enum import Enum
from functools import partial

import numpy as np
import typing_extensions as tp
from arraykit import name_filter

from static_frame.core.container_util import group_from_container
from static_frame.core.doc_str import doc_inject
# from static_frame.core.util import TUFunc
from static_frame.core.util import KEY_ITERABLE_TYPES
from static_frame.core.util import IterNodeType
from static_frame.core.util import TCallableAny
from static_frame.core.util import TDepthLevel
from static_frame.core.util import TDtypeSpecifier
from static_frame.core.util import TIndexCtorSpecifier
from static_frame.core.util import TLabel
from static_frame.core.util import TMapping
from static_frame.core.util import TName
from static_frame.core.util import TTupleCtor
from static_frame.core.util import get_concurrent_executor
from static_frame.core.util import iterable_to_array_1d

if tp.TYPE_CHECKING:
    from static_frame.core.bus import Bus  # pragma: no cover
    from static_frame.core.frame import Frame  # pragma: no cover
    from static_frame.core.index import Index  # pragma: no cover
    from static_frame.core.quilt import Quilt  # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.reduce import ReduceDispatch  # pragma: no cover
    from static_frame.core.series import Series  # pragma: no cover
    from static_frame.core.yarn import Yarn  # pragma: no cover

    TNDArrayAny = np.ndarray[tp.Any, tp.Any] #pragma: no cover
    # TDtypeAny = np.dtype[tp.Any] #pragma: no cover
    TSeriesAny = Series[tp.Any, tp.Any] #pragma: no cover
    TFrameAny = Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]] #pragma: no cover
    TBusAny = Bus[tp.Any] #pragma: no cover
    TYarnAny = Yarn[tp.Any] #pragma: no cover
    TFrameOrSeries = tp.Union[TSeriesAny, TFrameAny] # pragma: no cover
    TFrameOrArray = tp.Union[Frame, TNDArrayAny] # pragma: no cover

TContainerAny = tp.TypeVar('TContainerAny',
        'Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]]',
        'Series[tp.Any, tp.Any]',
        'Bus[tp.Any]',
        'Quilt',
        'Yarn[tp.Any]',
        )

class IterNodeApplyType(Enum):
    SERIES_VALUES = 0
    SERIES_ITEMS = 1 # only used for iter_window_*
    SERIES_ITEMS_GROUP_VALUES = 2
    SERIES_ITEMS_GROUP_LABELS = 3
    FRAME_ELEMENTS = 4
    INDEX_LABELS = 5

    @classmethod
    def is_items(cls, apply_type: 'IterNodeApplyType') -> bool:
        '''Return True if the apply_constructor to be used consumes items; otherwise, the apply_constructor consumes values alone.
        '''
        if (apply_type is cls.SERIES_VALUES
                or apply_type is cls.INDEX_LABELS
                ):
            return False
        return True


# NOTE: the generic type here is the type returned from calls to apply()
class IterNodeDelegate(tp.Generic[TContainerAny]):
    '''
    Delegate returned from :obj:`static_frame.IterNode`, providing iteration as well as a family of apply methods.
    '''

    __slots__ = (
            '_func_values',
            '_func_items',
            '_yield_type',
            '_apply_constructor',
            '_apply_type',
            '_container',
            )

    _INTERFACE: tp.Tuple[str, ...] = (
            'apply',
            'apply_iter',
            'apply_iter_items',
            'apply_pool',
            ) # should include __iter__() ?

    def __init__(self,
            func_values: tp.Callable[..., tp.Iterable[tp.Any]],
            func_items: tp.Callable[..., tp.Iterable[tp.Tuple[tp.Any, tp.Any]]],
            yield_type: IterNodeType,
            apply_constructor: tp.Callable[..., TContainerAny],
            apply_type: IterNodeApplyType,
            container: TFrameOrSeries,
        ) -> None:
        '''
        Args:
            apply_constructor: Callable (generally a class) used to construct the object returned from apply(); must take an iterator of items.
        '''
        self._func_values = func_values
        self._func_items = func_items
        self._yield_type = yield_type
        self._apply_constructor: tp.Callable[..., TContainerAny] = apply_constructor
        self._apply_type = apply_type
        self._container = container

    #---------------------------------------------------------------------------

    def _apply_iter_items_parallel(self,
            func: TCallableAny,
            *,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
            mp_context: tp.Optional[str] = None,
            ) -> tp.Iterator[tp.Tuple[tp.Any, tp.Any]]:

        if not callable(func): # support array, Series mapping
            func = getattr(func, '__getitem__')

        # use side effect list population to create keys when iterating over values
        func_keys = []

        if self._yield_type is IterNodeType.VALUES:
            def arg_gen() -> tp.Iterator[tp.Any]: #pylint: disable=E0102
                for k, v in self._func_items():
                    func_keys.append(k)
                    yield v
        else:
            def arg_gen() -> tp.Iterator[tp.Any]: #pylint: disable=E0102
                for k, v in self._func_items():
                    func_keys.append(k)
                    yield k, v

        pool_executor = get_concurrent_executor(
                use_threads=use_threads,
                max_workers=max_workers,
                mp_context=mp_context,
                )

        with pool_executor() as executor:
            yield from zip(func_keys,
                    executor.map(func, arg_gen(), chunksize=chunksize)
                    )

    def _apply_iter_parallel(self,
            func: TCallableAny,
            *,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
            mp_context: tp.Optional[str] = None,
            ) -> tp.Iterator[tp.Any]:

        if not callable(func): # support array, Series mapping
            func = getattr(func, '__getitem__')

        # use side effect list population to create keys when iterating over values
        arg_gen = (self._func_values if self._yield_type is IterNodeType.VALUES
                else self._func_items)

        pool_executor = get_concurrent_executor(
                use_threads=use_threads,
                max_workers=max_workers,
                mp_context=mp_context,
                )

        with pool_executor() as executor:
            yield from executor.map(func, arg_gen(), chunksize=chunksize)

    #---------------------------------------------------------------------------
    @doc_inject(selector='apply')
    def apply_iter_items(self,
            func: TCallableAny,
            ) -> tp.Iterator[tp.Tuple[tp.Any, tp.Any]]:
        '''
        {doc} A generator of resulting key, value pairs.

        Args:
            {func}

        Yields:
            Pairs of label, value after function application.
        '''
        # depend on yield type, we determine what the passed in function expects to
        if self._yield_type is IterNodeType.VALUES:
            yield from ((k, func(v)) for k, v in self._func_items())
        else:
            yield from ((k, func(k, v)) for k, v in self._func_items())

    @doc_inject(selector='apply')
    def apply_iter(self,
            func: TCallableAny
            ) -> tp.Iterator[tp.Any]:
        '''
        {doc} A generator of resulting values.

        Args:
            {func}

        Yields:
            Values after function application.
        '''
        if self._yield_type is IterNodeType.VALUES:
            yield from (func(v) for v in self._func_values())
        else:
            yield from (func(k, v) for k, v in self._func_items())

    @doc_inject(selector='apply')
    def apply(self,
            func: TCallableAny,
            *,
            dtype: TDtypeSpecifier = None,
            name: TName = None,
            index_constructor: tp.Optional[TIndexCtorSpecifier] = None,
            columns_constructor: tp.Optional[TIndexCtorSpecifier] = None,
            ) -> TContainerAny:
        '''
        {doc} Returns a new container.

        Args:
            {func}
            {dtype}
        '''
        if not callable(func):
            raise RuntimeError('use map_fill(), map_any(), or map_all() for applying a mapping type')

        if IterNodeApplyType.is_items(self._apply_type):
            apply_func = self.apply_iter_items
        else:
            apply_func = self.apply_iter

        if self._apply_type is IterNodeApplyType.FRAME_ELEMENTS:
            # can always pass columns_constructor
            return self._apply_constructor(
                    apply_func(func),
                    dtype=dtype,
                    name=name,
                    index_constructor=index_constructor,
                    columns_constructor=columns_constructor,
                    )

        if columns_constructor is not None:
            raise RuntimeError('Cannot use `columns_constructor` in this type of apply.')
        return self._apply_constructor(
                apply_func(func),
                dtype=dtype,
                name=name,
                index_constructor=index_constructor,
                )

    @doc_inject(selector='apply')
    def apply_pool(self,
            func: TCallableAny,
            *,
            dtype: TDtypeSpecifier = None,
            name: TName = None,
            index_constructor: tp.Optional[TIndexCtorSpecifier]= None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False
            ) -> TContainerAny:
        '''
        {doc} Employ parallel processing with either the ProcessPoolExecutor or ThreadPoolExecutor.

        Args:
            {func}
            *
            {dtype}
            {name}
            {max_workers}
            {chunksize}
            {use_threads}
        '''
        # only use when we need pairs of values to dynamically create an Index
        if IterNodeApplyType.is_items(self._apply_type):
            apply_func = self._apply_iter_items_parallel
        else:
            apply_func = self._apply_iter_parallel
        return self._apply_constructor(
                apply_func(func,
                        max_workers=max_workers,
                        chunksize=chunksize,
                        use_threads=use_threads,
                        ),
                dtype=dtype,
                name=name,
                index_constructor=index_constructor,
                )

    #---------------------------------------------------------------------------
    def __iter__(self) -> tp.Union[
            tp.Iterator[tp.Any],
            tp.Iterator[tp.Tuple[tp.Any, tp.Any]]
            ]:
        '''
        Return a generator based on the yield type.
        '''
        if self._yield_type is IterNodeType.VALUES:
            yield from self._func_values()
        else:
            yield from self._func_items()


class IterNodeDelegateReducible(IterNodeDelegate[TContainerAny]):
    '''
    Delegate returned from :obj:`static_frame.IterNode`, providing iteration as well as a family of apply methods.
    '''

    __slots__ = ()

    _INTERFACE = IterNodeDelegate._INTERFACE + (
            'reduce',
            )

    @property
    def reduce(self) -> ReduceDispatch:
        '''For each iterated compoent, apply a function per column.
        '''
        from static_frame.core.bus import Bus
        from static_frame.core.reduce import ReduceDispatchAligned
        from static_frame.core.reduce import ReduceDispatchUnaligned
        from static_frame.core.yarn import Yarn

        if self._container.ndim == 1:
            if not isinstance(self._container, (Bus, Yarn)):
                raise NotImplementedError('No support for 1D containers.') # pragma: no cover
            return ReduceDispatchUnaligned(
                    self._func_items(),
                    yield_type=self._yield_type,
                    )

        # self._func_items is partialed with kwargs specific to that function
        if self._func_items.keywords.get('drop', False): # type: ignore
            key = self._func_items.keywords['key'] # type: ignore
            axis_labels = self._container.columns.drop.loc[key] # type: ignore
        else:
            axis_labels = self._container.columns # type: ignore
        # always use the items iterator, as we always want labelled values
        return ReduceDispatchAligned(
                self._func_items(),
                axis_labels,
                yield_type=self._yield_type,
                )


class IterNodeDelegateMapable(IterNodeDelegate[TContainerAny]):
    '''
    Delegate returned from :obj:`static_frame.IterNode`, providing iteration as well as a family of apply methods.
    '''

    __slots__ = ()

    _INTERFACE = IterNodeDelegate._INTERFACE + (
            'map_all',
            'map_all_iter',
            'map_all_iter_items',
            'map_any',
            'map_any_iter',
            'map_any_iter_items',
            'map_fill',
            'map_fill_iter',
            'map_fill_iter_items',
            )

    @doc_inject(selector='map_any')
    def map_any_iter_items(self,
            mapping: TMapping
            ) -> tp.Iterator[tp.Tuple[tp.Any, tp.Any]]:
        '''
        {doc} A generator of resulting key, value pairs.

        Args:
            {mapping}
        '''
        get = getattr(mapping, 'get')
        if self._yield_type is IterNodeType.VALUES:
            yield from ((k, get(v, v)) for k, v in self._func_items())
        else:
            yield from ((k, get((k,  v), v)) for k, v in self._func_items())

    @doc_inject(selector='map_any')
    def map_any_iter(self,
            mapping: TMapping,
            ) -> tp.Iterator[tp.Any]:
        '''
        {doc} A generator of resulting values.

        Args:
            {mapping}
        '''
        get = getattr(mapping, 'get')
        if self._yield_type is IterNodeType.VALUES:
            yield from (get(v, v) for v in self._func_values())
        else:
            yield from (get((k,  v), v) for k, v in self._func_items())

    @doc_inject(selector='map_any')
    def map_any(self,
            mapping: TMapping,
            *,
            dtype: TDtypeSpecifier = None,
            name: TName = None,
            index_constructor: tp.Optional[TIndexCtorSpecifier] = None,
            ) -> TContainerAny:
        '''
        {doc} Returns a new container.

        Args:
            {mapping}
            {dtype}
        '''
        if IterNodeApplyType.is_items(self._apply_type):
            return self._apply_constructor(
                    self.map_any_iter_items(mapping),
                    dtype=dtype,
                    index_constructor=index_constructor,
                    name=name,
                    )
        return self._apply_constructor(
                self.map_any_iter(mapping),
                dtype=dtype,
                index_constructor=index_constructor,
                name=name,
                )

    #---------------------------------------------------------------------------
    @doc_inject(selector='map_fill')
    def map_fill_iter_items(self,
            mapping: TMapping,
            *,
            fill_value: tp.Any = np.nan,
            ) -> tp.Iterator[tp.Tuple[tp.Any, tp.Any]]:
        '''
        {doc} A generator of resulting key, value pairs.

        Args:
            {mapping}
            {fill_value}
        '''
        get = getattr(mapping, 'get')
        if self._yield_type is IterNodeType.VALUES:
            yield from ((k, get(v, fill_value)) for k, v in self._func_items())
        else:
            yield from ((k, get((k,  v), fill_value)) for k, v in self._func_items())

    @doc_inject(selector='map_fill')
    def map_fill_iter(self,
            mapping: TMapping,
            *,
            fill_value: tp.Any = np.nan,
            ) -> tp.Iterator[tp.Any]:
        '''
        {doc} A generator of resulting values.

        Args:
            {mapping}
            {fill_value}
        '''
        get = getattr(mapping, 'get')
        if self._yield_type is IterNodeType.VALUES:
            yield from (get(v, fill_value) for v in self._func_values())
        else:
            yield from (get((k,  v), fill_value) for k, v in self._func_items())

    @doc_inject(selector='map_fill')
    def map_fill(self,
            mapping: TMapping,
            *,
            fill_value: tp.Any = np.nan,
            dtype: TDtypeSpecifier = None,
            name: TName = None,
            index_constructor: tp.Optional[TIndexCtorSpecifier] = None,
            ) -> TContainerAny:
        '''
        {doc} Returns a new container.

        Args:
            {mapping}
            {fill_value}
            {dtype}
        '''
        if IterNodeApplyType.is_items(self._apply_type):
            return self._apply_constructor(
                    self.map_fill_iter_items(mapping, fill_value=fill_value),
                    dtype=dtype,
                    name=name,
                    index_constructor=index_constructor,
                    )
        return self._apply_constructor(
                self.map_fill_iter(mapping, fill_value=fill_value),
                dtype=dtype,
                name=name,
                index_constructor=index_constructor,
                )


    #---------------------------------------------------------------------------
    @doc_inject(selector='map_all')
    def map_all_iter_items(self,
            mapping: TMapping
            ) -> tp.Iterator[tp.Tuple[tp.Any, tp.Any]]:
        '''
        {doc} A generator of resulting key, value pairs.

        Args:
            {mapping}
        '''
        # want exception to raise if key not found
        func = getattr(mapping, '__getitem__')
        if self._yield_type is IterNodeType.VALUES:
            yield from ((k, func(v)) for k, v in self._func_items())
        else:
            yield from ((k, func((k,  v))) for k, v in self._func_items())

    @doc_inject(selector='map_all')
    def map_all_iter(self,
            mapping: TMapping
            ) -> tp.Iterator[tp.Any]:
        '''
        {doc} A generator of resulting values.

        Args:
            {mapping}
        '''
        func = getattr(mapping, '__getitem__')
        if self._yield_type is IterNodeType.VALUES:
            yield from (func(v) for v in self._func_values())
        else:
            yield from (func((k,  v)) for k, v in self._func_items())

    @doc_inject(selector='map_all')
    def map_all(self,
            mapping: TMapping,
            *,
            dtype: TDtypeSpecifier = None,
            name: TName = None,
            index_constructor: tp.Optional[TIndexCtorSpecifier] = None,
            ) -> TContainerAny:
        '''
        {doc} Returns a new container.

        Args:
            {mapping}
            {dtype}
        '''
        if IterNodeApplyType.is_items(self._apply_type):
            return self._apply_constructor(
                    self.map_all_iter_items(mapping),
                    dtype=dtype,
                    name=name,
                    index_constructor=index_constructor,
                    )
        return self._apply_constructor(
                self.map_all_iter(mapping),
                dtype=dtype,
                name=name,
                index_constructor=index_constructor,
                )

#-------------------------------------------------------------------------------

class IterNode(tp.Generic[TContainerAny]):
    '''Interface to a type of iteration on :obj:`static_frame.Series` and :obj:`static_frame.Frame`.
    '''
    # Stores two version of a generator function: one to yield single values, another to yield items pairs. The latter is needed in all cases, as when we use apply we return a Series, and need to have recourse to an index.

    __slots__ = (
        '_container',
        '_func_values',
        '_func_items',
        '_yield_type',
        '_apply_type',
        )
    CLS_DELEGATE = IterNodeDelegate

    def __init__(self, *,
            container: TContainerAny,
            function_values: tp.Callable[..., tp.Iterable[tp.Any]],
            function_items: tp.Callable[..., tp.Iterable[tp.Tuple[tp.Any, tp.Any]]],
            yield_type: IterNodeType,
            apply_type: IterNodeApplyType,
            ) -> None:
        '''
        Args:
            function_values: will be partialed with arguments given with __call__.
            function_items: will be partialed with arguments given with __call__.
        '''
        self._container: TContainerAny = container
        self._func_values = function_values
        self._func_items = function_items
        self._yield_type = yield_type
        self._apply_type = apply_type

    #---------------------------------------------------------------------------
    # apply constructors

    def to_series_from_values(self,
            values: tp.Iterator[tp.Any],
            *,
            dtype: TDtypeSpecifier,
            name: TName = None,
            index_constructor: tp.Optional[TIndexCtorSpecifier] = None,
            axis: int = 0,
            ) -> TSeriesAny:
        from static_frame.core.series import Series

        # Creating a Series that will have the same index as source container
        if self._container._NDIM == 2 and axis == 0:
            index = self._container._columns #type: ignore
            own_index = False
        else:
            index = self._container._index
            own_index = True

        if index_constructor is not None:
            index = index_constructor(index)

        # PERF: passing count here permits faster generator realization
        array, _ = iterable_to_array_1d(
                values,
                count=index.shape[0],
                dtype=dtype,
                )
        return Series(array,
                name=name,
                index=index,
                own_index=own_index,
                )

    def to_series_from_items(self,
            pairs: tp.Iterable[tp.Tuple[TLabel, tp.Any]],
            *,
            dtype: TDtypeSpecifier = None,
            name: TName = None,
            index_constructor: tp.Optional[TIndexCtorSpecifier]= None,
            axis: int = 0,
            ) -> TSeriesAny:
        from static_frame.core.series import Series

        # apply_constructor should be implemented to take a pairs of label, value; only used for iter_window
        # axis 0 iters windows labelled by the index, axis 1 iters windows labelled by the columns

        if self._container._NDIM == 2 and axis == 1:
            index_constructor = (index_constructor
                    if index_constructor is not None
                    else self._container._columns.from_labels) #type: ignore
            name_index = self._container._columns._name #type: ignore
        else:
            index_constructor = (index_constructor
                    if index_constructor is not None
                    else self._container._index.from_labels)
            name_index = self._container._index._name

        index_constructor_final = partial(
                index_constructor,
                name=name_index,
                )
        # always return a Series
        return Series.from_items(
                pairs=pairs,
                dtype=dtype,
                name=name,
                index_constructor=index_constructor_final,
                )

    def to_series_from_group_items(self,
            pairs: tp.Iterable[tp.Tuple[TLabel, tp.Any]],
            *,
            dtype: TDtypeSpecifier = None,
            name: TName = None,
            index_constructor: tp.Optional[TIndexCtorSpecifier]= None,
            name_index: TName = None,
            ) -> TSeriesAny:
        from static_frame.core.index import Index
        from static_frame.core.series import Series

        # NOTE: when used on labels, this key is given; when used on labels (indices) depth_level is given; only take the key if it is a hashable (a string or a tuple, not a slice, list, or array)

        index_constructor = partial(
                Index if index_constructor is None else index_constructor,
                name=name_index,
                )
        return Series.from_items(
                pairs=pairs,
                dtype=dtype,
                name=name,
                index_constructor=index_constructor
                )

    def to_frame_from_elements(self,
            items: tp.Iterable[tp.Tuple[
                    tp.Tuple[TLabel, TLabel], tp.Any]],
            *,
            dtype: TDtypeSpecifier = None,
            name: TName = None,
            index_constructor: tp.Optional[TIndexCtorSpecifier]= None,
            columns_constructor: tp.Optional[TIndexCtorSpecifier]= None,
            axis: int = 0,
            ) -> TFrameAny:
        # NOTE: this is only called from `Frame` to produce a new `Frame`

        from static_frame.core.frame import Frame

        if index_constructor is not None:
            index = index_constructor(self._container._index)
        else:
            index = self._container._index

        assert isinstance(self._container, Frame) # mypy

        if columns_constructor is not None:
            columns = columns_constructor(self._container._columns)
        else:
            columns = self._container._columns

        return self._container.__class__.from_element_items(
                items,
                index=index,
                columns=columns,
                axis=axis,
                own_index=True,
                own_columns=True,
                name=name,
                )

    def to_index_from_labels(self,
            values: tp.Iterator[TLabel], #pylint: disable=function-redefined
            dtype: TDtypeSpecifier = None,
            name: TName = None,
            index_constructor: tp.Optional[TIndexCtorSpecifier]= None,
            ) -> TNDArrayAny:
        # NOTE: name argument is for common interface
        if index_constructor is not None:
            raise RuntimeError('index_constructor not supported with this interface')
        # PERF: passing count here permits faster generator realization
        shape = self._container.shape
        array, _ = iterable_to_array_1d(values, count=shape[0], dtype=dtype)
        return array

    #---------------------------------------------------------------------------
    def _get_delegate_kwargs(self,
            **kwargs: object,
            ) -> tp.Dict[str, tp.Any]:
        '''
        In usage as an iteator, the args passed here are expected to be argument for the core iterators, i.e., axis arguments.

        Args:
            kwargs: kwarg args to be passed to both self._func_values and self._func_items
        '''
        from static_frame.core.frame import Frame

        # all kwargs, like ``drop```, are partialed into func_values, func_items
        func_values = partial(self._func_values, **kwargs)
        func_items = partial(self._func_items, **kwargs)

        axis: int = kwargs.get('axis', 0) # type: ignore

        apply_constructor: tp.Callable[..., tp.Union[TFrameAny, TSeriesAny]]

        if self._apply_type is IterNodeApplyType.SERIES_VALUES:
            apply_constructor = partial(self.to_series_from_values, axis=axis)

        elif self._apply_type is IterNodeApplyType.SERIES_ITEMS:
            apply_constructor = partial(self.to_series_from_items, axis=axis)

        elif self._apply_type is IterNodeApplyType.SERIES_ITEMS_GROUP_VALUES:
            try:
                name_index = name_filter(kwargs.get('key', None))
            except TypeError:
                name_index = None
            apply_constructor = partial(self.to_series_from_group_items,
                    name_index=name_index,
                    )

        elif self._apply_type is IterNodeApplyType.SERIES_ITEMS_GROUP_LABELS:
            # will always have `depth_level` in kwargs, and for Frame an axis; could attempt to get name from the index if it has a name
            name_index = None
            apply_constructor = partial(self.to_series_from_group_items,
                    name_index=name_index,
                    )

        elif self._apply_type is IterNodeApplyType.FRAME_ELEMENTS:
            assert isinstance(self._container, Frame) # for typing
            apply_constructor = partial(self.to_frame_from_elements, axis=axis)

        elif self._apply_type is IterNodeApplyType.INDEX_LABELS:
            apply_constructor = self.to_index_from_labels # type: ignore

        else:
            raise NotImplementedError(self._apply_type) #pragma: no cover

        return dict(
                func_values=func_values,
                func_items=func_items,
                yield_type=self._yield_type,
                apply_constructor=tp.cast(tp.Callable[..., TContainerAny], apply_constructor),
                apply_type=self._apply_type,
                container=self._container,
                )

    def get_delegate(self,
            **kwargs: object,
            ) -> IterNodeDelegate[TContainerAny]:
        return IterNodeDelegate(**self._get_delegate_kwargs(**kwargs))

    def get_delegate_reducible(self,
            **kwargs: object,
            ) -> IterNodeDelegateReducible[TContainerAny]:
        return IterNodeDelegateReducible(**self._get_delegate_kwargs(**kwargs))

    def get_delegate_mapable(self,
            **kwargs: object,
            ) -> IterNodeDelegateMapable[TContainerAny]:
        return IterNodeDelegateMapable(**self._get_delegate_kwargs(**kwargs))

#-------------------------------------------------------------------------------
# specialize IterNode based on arguments given to __call__

class IterNodeNoArg(IterNode[TContainerAny]):

    __slots__ = ()
    CLS_DELEGATE = IterNodeDelegate

    def __call__(self,
            ) -> IterNodeDelegate[TContainerAny]:
        return IterNode.get_delegate(self)


class IterNodeNoArgMapable(IterNode[TContainerAny]):

    __slots__ = ()
    CLS_DELEGATE = IterNodeDelegateMapable

    def __call__(self,
            ) -> IterNodeDelegateMapable[TContainerAny]:
        return IterNode.get_delegate_mapable(self)

class IterNodeNoArgReducible(IterNode[TContainerAny]):

    __slots__ = ()
    CLS_DELEGATE = IterNodeDelegateReducible

    def __call__(self,
            ) -> IterNodeDelegateReducible[TContainerAny]:
        return IterNode.get_delegate_reducible(self)


class IterNodeAxisElement(IterNode[TContainerAny]):

    __slots__ = ()
    CLS_DELEGATE = IterNodeDelegateMapable

    def __call__(self,
            *,
            axis: int = 0
            ) -> IterNodeDelegateMapable[TContainerAny]:
        return IterNode.get_delegate_mapable(self, axis=axis)

class IterNodeAxis(IterNode[TContainerAny]):

    __slots__ = ()

    def __call__(self,
            *,
            axis: int = 0
            ) -> IterNodeDelegateMapable[TContainerAny]:
        return IterNode.get_delegate_mapable(self, axis=axis)

class IterNodeConstructorAxis(IterNode[TContainerAny]):

    __slots__ = ()
    CLS_DELEGATE = IterNodeDelegateMapable

    def __call__(self,
            *,
            axis: int = 0,
            constructor: tp.Optional[TTupleCtor] = None,
            ) -> IterNodeDelegateMapable[TContainerAny]:
        return IterNode.get_delegate_mapable(self,
                axis=axis,
                constructor=constructor,
                )

class IterNodeGroup(IterNode[TContainerAny]):
    '''
    Iterator on 1D groupings where no args are required (but axis is retained for compatibility)
    '''

    __slots__ = ()

    def __call__(self,
            *,
            axis: int = 0
            ) -> IterNodeDelegate[TContainerAny]:
        return IterNode.get_delegate(self, axis=axis)

class IterNodeGroupAxis(IterNode[TContainerAny]):
    '''
    Iterator on 2D groupings where key and axis are required.
    '''

    __slots__ = ()
    CLS_DELEGATE = IterNodeDelegateReducible

    def __call__(self,
            key: KEY_ITERABLE_TYPES, # type: ignore
            *,
            axis: int = 0,
            drop: bool = False,
            ) -> IterNodeDelegateReducible[TContainerAny]:
        return IterNode.get_delegate_reducible(self, key=key, axis=axis, drop=drop)


class IterNodeGroupOther(IterNode[TContainerAny]):
    '''
    Iterator on 1D groupings where group values are provided.
    '''
    __slots__ = ()
    CLS_DELEGATE = IterNodeDelegate

    def __call__(self,
            other: tp.Union[TNDArrayAny, Index[tp.Any], TSeriesAny, tp.Iterable[tp.Any]],
            *,
            fill_value: tp.Any = np.nan,
            axis: int = 0
            ) -> IterNodeDelegate[TContainerAny]:

        index_ref = (self._container._index if axis == 0
                else self._container._columns) # type: ignore
        group_source = group_from_container(
                index=index_ref,
                group_source=other,
                fill_value=fill_value,
                axis=axis,
                )
        # kwargs are partialed into func_values, func_items
        return IterNode.get_delegate(self,
                axis=axis,
                group_source=group_source,
                )

class IterNodeGroupOtherReducible(IterNode[TContainerAny]):
    '''
    Iterator on 1D groupings where group values are provided.
    '''
    __slots__ = ()
    CLS_DELEGATE = IterNodeDelegateReducible

    def __call__(self,
            other: tp.Union[TNDArrayAny, Index[tp.Any], TSeriesAny, tp.Iterable[tp.Any]],
            *,
            fill_value: tp.Any = np.nan,
            axis: int = 0
            ) -> IterNodeDelegateReducible[TContainerAny]:

        index_ref = (self._container._index if axis == 0
                else self._container._columns) # type: ignore
        group_source = group_from_container(
                index=index_ref,
                group_source=other,
                fill_value=fill_value,
                axis=axis,
                )
        # kwargs are partialed into func_values, func_items
        return IterNode.get_delegate_reducible(self,
                axis=axis,
                group_source=group_source,
                )

class IterNodeDepthLevel(IterNode[TContainerAny]):

    __slots__ = ()

    def __call__(self,
            depth_level: tp.Optional[TDepthLevel] = None
            ) -> IterNodeDelegateMapable[TContainerAny]:
        return IterNode.get_delegate_mapable(self, depth_level=depth_level)


class IterNodeDepthLevelAxis(IterNode[TContainerAny]):

    __slots__ = ()

    def __call__(self,
            depth_level: TDepthLevel = 0,
            *,
            axis: int = 0
            ) -> IterNodeDelegate[TContainerAny]:
        return IterNode.get_delegate(self, depth_level=depth_level, axis=axis)


class IterNodeWindow(IterNode[TContainerAny]):

    __slots__ = ()
    CLS_DELEGATE = IterNodeDelegate

    def __call__(self, *,
            size: int,
            axis: int = 0,
            step: int = 1,
            window_sized: bool = True,
            window_func: tp.Optional[TCallableAny] = None,
            window_valid: tp.Optional[TCallableAny] = None,
            label_shift: int = 0,
            label_missing_skips: bool = True,
            label_missing_raises: bool = False,
            start_shift: int = 0,
            size_increment: int = 0,
            ) -> IterNodeDelegate[TContainerAny]:
        return IterNode.get_delegate(self,
                axis=axis,
                size=size,
                step=step,
                window_sized=window_sized,
                window_func=window_func,
                window_valid=window_valid,
                label_shift=label_shift,
                label_missing_skips=label_missing_skips,
                label_missing_raises=label_missing_raises,
                start_shift=start_shift,
                size_increment=size_increment,
                )

class IterNodeWindowReducible(IterNode[TContainerAny]):

    __slots__ = ()
    CLS_DELEGATE = IterNodeDelegateReducible

    def __call__(self, *,
            size: int,
            axis: int = 0,
            step: int = 1,
            window_sized: bool = True,
            window_func: tp.Optional[TCallableAny] = None,
            window_valid: tp.Optional[TCallableAny] = None,
            label_shift: int = 0,
            label_missing_skips: bool = True,
            label_missing_raises: bool = False,
            start_shift: int = 0,
            size_increment: int = 0,
            ) -> IterNodeDelegateReducible[TContainerAny]:
        return IterNode.get_delegate_reducible(self,
                axis=axis,
                size=size,
                step=step,
                window_sized=window_sized,
                window_func=window_func,
                window_valid=window_valid,
                label_shift=label_shift,
                label_missing_skips=label_missing_skips,
                label_missing_raises=label_missing_raises,
                start_shift=start_shift,
                size_increment=size_increment,
                )

