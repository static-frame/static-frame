'''
Tools for iterators in Series and Frame. These components are imported by both series.py and frame.py; these components also need to be able to return Series and Frame, and thus use deferred, function-based imports.
'''

import typing as tp
from enum import Enum
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
# import multiprocessing as mp
# mp_context = mp.get_context('spawn')

import numpy as np
from arraykit import name_filter

from static_frame.core.doc_str import doc_inject
from static_frame.core.util import AnyCallable
from static_frame.core.util import DepthLevelSpecifier
from static_frame.core.util import DtypeSpecifier
from static_frame.core.util import KEY_ITERABLE_TYPES
from static_frame.core.util import Mapping
from static_frame.core.util import NameType
from static_frame.core.util import TupleConstructorType
from static_frame.core.util import iterable_to_array_1d
from static_frame.core.util import IndexConstructor
# from static_frame.core.util import array_from_iterator


if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.series import Series # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.quilt import Quilt # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.bus import Bus # pylint: disable=W0611 #pragma: no cover
    from static_frame.core.yarn import Yarn # pylint: disable=W0611 #pragma: no cover


FrameOrSeries = tp.TypeVar('FrameOrSeries', 'Frame', 'Series', 'Bus', 'Quilt', 'Yarn')
PoolArgGen = tp.Callable[[], tp.Union[tp.Iterator[tp.Any], tp.Iterator[tp.Tuple[tp.Any, tp.Any]]]]
# FrameSeriesIndex = tp.TypeVar('FrameSeriesIndex', 'Frame', 'Series', 'Index')


class IterNodeType(Enum):
    VALUES = 1
    ITEMS = 2


class IterNodeApplyType(Enum):
    SERIES_VALUES = 0
    SERIES_ITEMS = 1 # only used for iter_window_*
    SERIES_ITEMS_GROUP_VALUES = 2
    SERIES_ITEMS_GROUP_LABELS = 3
    FRAME_ELEMENTS = 4
    INDEX_LABELS = 5

    @classmethod
    def is_items(cls, apply_type: 'IterNodeApplyType') -> bool:
        if apply_type is cls.SERIES_VALUES or apply_type is cls.INDEX_LABELS:
            return False
        return True



class IterNodeDelegate(tp.Generic[FrameOrSeries]):
    '''
    Delegate returned from :obj:`static_frame.IterNode`, providing iteration as well as a family of apply methods.
    '''

    __slots__ = (
            '_func_values',
            '_func_items',
            '_yield_type',
            '_apply_constructor',
            '_apply_type',
            )

    INTERFACE = (
            'apply',
            'apply_iter',
            'apply_iter_items',
            'apply_pool',
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

    def __init__(self,
            func_values: tp.Callable[..., tp.Iterable[tp.Any]],
            func_items: tp.Callable[..., tp.Iterable[tp.Tuple[tp.Any, tp.Any]]],
            yield_type: IterNodeType,
            apply_constructor: tp.Callable[..., FrameOrSeries],
            apply_type: IterNodeApplyType,
        ) -> None:
        '''
        Args:
            apply_constructor: Callable (generally a class) used to construct the object returned from apply(); must take an iterator of items.
        '''
        self._func_values = func_values
        self._func_items = func_items
        self._yield_type = yield_type
        self._apply_constructor: tp.Callable[..., FrameOrSeries] = apply_constructor
        self._apply_type = apply_type

    #---------------------------------------------------------------------------

    def _apply_iter_items_parallel(self,
            func: AnyCallable,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
            ) -> tp.Iterator[tp.Tuple[tp.Any, tp.Any]]:

        pool_executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

        if not callable(func): # NOTE: when is func not a callable?
            func = getattr(func, '__getitem__')

        # use side effect list population to create keys when iterating over values
        func_keys = []
        arg_gen: PoolArgGen

        if self._yield_type is IterNodeType.VALUES:
            def arg_gen() -> tp.Iterator[tp.Any]: #pylint: disable=E0102
                for k, v in self._func_items():
                    func_keys.append(k)
                    yield v
        else:
            def arg_gen() -> tp.Iterator[tp.Tuple[tp.Any, tp.Any]]: #pylint: disable=E0102
                for k, v in self._func_items():
                    func_keys.append(k)
                    yield k, v

        with pool_executor(max_workers=max_workers) as executor:
            yield from zip(func_keys,
                    executor.map(func, arg_gen(), chunksize=chunksize)
                    )

    def _apply_iter_parallel(self,
            func: AnyCallable,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False,
            ) -> tp.Iterator[tp.Any]:

        pool_executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

        if not callable(func): # NOTE: when is func not a callable?
            func = getattr(func, '__getitem__') # COV_MISSING

        # use side effect list population to create keys when iterating over values
        arg_gen = (self._func_values if self._yield_type is IterNodeType.VALUES
                else self._func_items)

        with pool_executor(max_workers=max_workers) as executor:
            yield from executor.map(func, arg_gen(), chunksize=chunksize)

    #---------------------------------------------------------------------------
    # public interface

    @doc_inject(selector='map_any')
    def map_any_iter_items(self,
            mapping: Mapping
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
            mapping: Mapping,
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
            yield from (get((k,  v), v) for k, v in self._func_items()) # COV_MISSING

    @doc_inject(selector='map_any')
    def map_any(self,
            mapping: Mapping,
            *,
            dtype: DtypeSpecifier = None,
            name: NameType = None,
            index_constructor: tp.Optional[IndexConstructor] = None,
            ) -> FrameOrSeries:
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
            mapping: Mapping,
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
            mapping: Mapping,
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
            yield from (get((k,  v), fill_value) for k, v in self._func_items()) # COV_MISSING

    @doc_inject(selector='map_fill')
    def map_fill(self,
            mapping: Mapping,
            *,
            fill_value: tp.Any = np.nan,
            dtype: DtypeSpecifier = None,
            name: NameType = None,
            index_constructor: tp.Optional[IndexConstructor] = None,
            ) -> FrameOrSeries:
        '''
        {doc} Returns a new container.

        Args:
            {mapping}
            {fill_value}
            {dtype}
        '''
        if IterNodeApplyType.is_items(self._apply_type):
            return self._apply_constructor( # COV_MISSING
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
            mapping: Mapping
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
            mapping: Mapping
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
            yield from (func((k,  v)) for k, v in self._func_items()) # COV_MISSING

    @doc_inject(selector='map_all')
    def map_all(self,
            mapping: Mapping,
            *,
            dtype: DtypeSpecifier = None,
            name: NameType = None,
            index_constructor: tp.Optional[IndexConstructor] = None,
            ) -> FrameOrSeries:
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


    #---------------------------------------------------------------------------
    @doc_inject(selector='apply')
    def apply_iter_items(self,
            func: AnyCallable,
            ) -> tp.Iterator[tp.Tuple[tp.Any, tp.Any]]:
        '''
        {doc} A generator of resulting key, value pairs.

        Args:
            {func}
        '''
        # depend on yield type, we determine what the passed in function expects to
        if self._yield_type is IterNodeType.VALUES:
            yield from ((k, func(v)) for k, v in self._func_items())
        else:
            yield from ((k, func(k, v)) for k, v in self._func_items())

    @doc_inject(selector='apply')
    def apply_iter(self,
            func: AnyCallable
            ) -> tp.Iterator[tp.Any]:
        '''
        {doc} A generator of resulting values.

        Args:
            {func}
        '''
        if self._yield_type is IterNodeType.VALUES:
            yield from (func(v) for v in self._func_values())
        else:
            yield from (func(k, v) for k, v in self._func_items())

    @doc_inject(selector='apply')
    def apply(self,
            func: AnyCallable,
            *,
            dtype: DtypeSpecifier = None,
            name: NameType = None,
            index_constructor: tp.Optional[IndexConstructor]= None,
            ) -> FrameOrSeries:
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

        return self._apply_constructor(
                apply_func(func),
                dtype=dtype,
                name=name,
                index_constructor=index_constructor,
                )

    @doc_inject(selector='apply')
    def apply_pool(self,
            func: AnyCallable,
            *,
            dtype: DtypeSpecifier = None,
            name: NameType = None,
            index_constructor: tp.Optional[IndexConstructor]= None,
            max_workers: tp.Optional[int] = None,
            chunksize: int = 1,
            use_threads: bool = False
            ) -> FrameOrSeries:
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



#-------------------------------------------------------------------------------

_ITER_NODE_SLOTS = (
        '_container',
        '_func_values',
        '_func_items',
        '_yield_type',
        '_apply_type'
        )

class IterNode(tp.Generic[FrameOrSeries]):
    '''Interface to a type of iteration on :obj:`static_frame.Series` and :obj:`static_frame.Frame`.
    '''
    # Stores two version of a generator function: one to yield single values, another to yield items pairs. The latter is needed in all cases, as when we use apply we return a Series, and need to have recourse to an index.

    __slots__ = _ITER_NODE_SLOTS

    def __init__(self, *,
            container: FrameOrSeries,
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
        self._container: FrameOrSeries = container
        self._func_values = function_values
        self._func_items = function_items
        self._yield_type = yield_type
        self._apply_type = apply_type

    #---------------------------------------------------------------------------
    # apply constructors

    def to_series_values(self,
            values: tp.Iterator[tp.Any],
            *,
            dtype: DtypeSpecifier,
            name: NameType = None,
            index_constructor: tp.Optional[IndexConstructor] = None,
            axis: int = 0,
            ) -> 'Series':
        from static_frame.core.series import Series

        # Creating a Series that will have the same index as source container
        if self._container._NDIM == 2 and axis == 0:
            index = self._container._columns #type: ignore
            own_index = False
        else:
            index = self._container._index
            own_index = True

        if index_constructor is not None:
            index = index_constructor(index) # COV_MISSING

        # PERF: passing count here permits faster generator realization
        values, _ = iterable_to_array_1d(
                values,
                count=index.shape[0],
                dtype=dtype,
                )
        return Series(values,
                name=name,
                index=index,
                own_index=own_index,
                )

    def to_series_items(self,
            pairs: tp.Iterable[tp.Tuple[tp.Hashable, tp.Any]],
            *,
            dtype: DtypeSpecifier = None,
            name: NameType = None,
            index_constructor: tp.Optional[IndexConstructor]= None,
            axis: int = 0,
            ) -> 'Series':
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

    def to_series_items_group(self,
            pairs: tp.Iterable[tp.Tuple[tp.Hashable, tp.Any]],
            *,
            dtype: DtypeSpecifier = None,
            name: NameType = None,
            index_constructor: tp.Optional[IndexConstructor]= None,
            name_index: NameType = None,
            ) -> 'Series':
        from static_frame.core.series import Series
        from static_frame.core.index import Index

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

    def to_frame_elements(self,
            items: tp.Iterable[tp.Tuple[
                    tp.Tuple[tp.Hashable, tp.Hashable], tp.Any]],
            *,
            dtype: DtypeSpecifier = None,
            name: NameType = None,
            index_constructor: tp.Optional[IndexConstructor]= None,
            axis: int = 0,
            ) -> 'Frame':
        from static_frame.core.frame import Frame

        index_constructor = (self._container._index.from_labels
                if index_constructor is None else index_constructor)

        assert isinstance(self._container, Frame)
        return self._container.__class__.from_element_items(
                items,
                index=self._container._index,
                columns=self._container._columns,
                axis=axis,
                own_index=True,
                index_constructor=index_constructor,
                columns_constructor=self._container._columns.from_labels,
                name=name,
                )

    def to_index_labels(self,
            values: tp.Iterator[tp.Hashable], #pylint: disable=function-redefined
            dtype: DtypeSpecifier = None,
            name: NameType = None,
            index_constructor: tp.Optional[IndexConstructor]= None,
            ) -> np.ndarray:
        # NOTE: name argument is for common interface
        if index_constructor is not None:
            raise RuntimeError('index_constructor not supported with this interface') # COV_MISSING
        # PERF: passing count here permits faster generator realization
        shape = self._container.shape
        array, _ = iterable_to_array_1d(values, count=shape[0], dtype=dtype)
        return array

    #---------------------------------------------------------------------------
    def get_delegate(self,
            **kwargs: object
            ) -> IterNodeDelegate[FrameOrSeries]:
        '''
        In usage as an iteator, the args passed here are expected to be argument for the core iterators, i.e., axis arguments.

        Args:
            kwargs: kwarg args to be passed to both self._func_values and self._func_items
        '''
        from static_frame.core.frame import Frame

        # all kwargs, like ``drop```, are partialed into func_values, func_items
        func_values = partial(self._func_values, **kwargs)
        func_items = partial(self._func_items, **kwargs)

        axis = kwargs.get('axis', 0)

        apply_constructor: tp.Callable[..., tp.Union['Frame', 'Series']]

        if self._apply_type is IterNodeApplyType.SERIES_VALUES:
            apply_constructor = partial(self.to_series_values, axis=axis)

        elif self._apply_type is IterNodeApplyType.SERIES_ITEMS:
            apply_constructor = partial(self.to_series_items, axis=axis)

        elif self._apply_type is IterNodeApplyType.SERIES_ITEMS_GROUP_VALUES:
            try:
                name_index = name_filter(kwargs.get('key', None))
            except TypeError:
                name_index = None
            apply_constructor = partial(self.to_series_items_group,
                    name_index=name_index,
                    )

        elif self._apply_type is IterNodeApplyType.SERIES_ITEMS_GROUP_LABELS:
            # will always have `depth_level` in kwargs, and for Frame an axis; could attempt to get name from the index if it has a name
            name_index = None
            apply_constructor = partial(self.to_series_items_group,
                    name_index=name_index,
                    )

        elif self._apply_type is IterNodeApplyType.FRAME_ELEMENTS:
            assert isinstance(self._container, Frame) # for typing
            apply_constructor = partial(self.to_frame_elements, axis=axis)

        elif self._apply_type is IterNodeApplyType.INDEX_LABELS:
            apply_constructor = self.to_index_labels

        else:
            raise NotImplementedError(self._apply_type) #pragma: no cover

        return IterNodeDelegate(
                func_values=func_values,
                func_items=func_items,
                yield_type=self._yield_type,
                apply_constructor=tp.cast(tp.Callable[..., FrameOrSeries], apply_constructor),
                apply_type=self._apply_type,
                )


#-------------------------------------------------------------------------------
# specialize IterNode based on arguments given to __call__

class IterNodeNoArg(IterNode[FrameOrSeries]):

    __slots__ = _ITER_NODE_SLOTS

    def __call__(self,
            ) -> IterNodeDelegate[FrameOrSeries]:
        return IterNode.get_delegate(self)


class IterNodeAxis(IterNode[FrameOrSeries]):

    __slots__ = _ITER_NODE_SLOTS

    def __call__(self,
            *,
            axis: int = 0
            ) -> IterNodeDelegate[FrameOrSeries]:
        return IterNode.get_delegate(self, axis=axis)


class IterNodeConstructorAxis(IterNode[FrameOrSeries]):

    __slots__ = _ITER_NODE_SLOTS

    def __call__(self,
            *,
            axis: int = 0,
            constructor: tp.Optional[TupleConstructorType] = None,
            ) -> IterNodeDelegate[FrameOrSeries]:
        return IterNode.get_delegate(self,
                axis=axis,
                constructor=constructor,
                )

class IterNodeGroup(IterNode[FrameOrSeries]):
    '''
    Iterator on 1D groupings where no args are required (but axis is retained for compatibility)
    '''

    __slots__ = _ITER_NODE_SLOTS

    def __call__(self,
            *,
            axis: int = 0
            ) -> IterNodeDelegate[FrameOrSeries]:
        return IterNode.get_delegate(self, axis=axis)

class IterNodeGroupAxis(IterNode[FrameOrSeries]):
    '''
    Iterator on 2D groupings where key and axis are required.
    '''

    __slots__ = _ITER_NODE_SLOTS

    def __call__(self,
            key: KEY_ITERABLE_TYPES, # type: ignore
            *,
            axis: int = 0,
            drop: bool = False,
            ) -> IterNodeDelegate[FrameOrSeries]:
        return IterNode.get_delegate(self, key=key, axis=axis, drop=drop)


class IterNodeDepthLevel(IterNode[FrameOrSeries]):

    __slots__ = _ITER_NODE_SLOTS

    def __call__(self,
            depth_level: tp.Optional[DepthLevelSpecifier] = None
            ) -> IterNodeDelegate[FrameOrSeries]:
        return IterNode.get_delegate(self, depth_level=depth_level)


class IterNodeDepthLevelAxis(IterNode[FrameOrSeries]):

    __slots__ = _ITER_NODE_SLOTS

    def __call__(self,
            depth_level: DepthLevelSpecifier = 0,
            *,
            axis: int = 0
            ) -> IterNodeDelegate[FrameOrSeries]:
        return IterNode.get_delegate(self, depth_level=depth_level, axis=axis)


class IterNodeWindow(IterNode[FrameOrSeries]):

    __slots__ = _ITER_NODE_SLOTS

    def __call__(self, *,
            size: int,
            axis: int = 0,
            step: int = 1,
            window_sized: bool = True,
            window_func: tp.Optional[AnyCallable] = None,
            window_valid: tp.Optional[AnyCallable] = None,
            label_shift: int = 0,
            start_shift: int = 0,
            size_increment: int = 0,
            ) -> IterNodeDelegate[FrameOrSeries]:
        return IterNode.get_delegate(self,
                axis=axis,
                size=size,
                step=step,
                window_sized=window_sized,
                window_func=window_func,
                window_valid=window_valid,
                label_shift=label_shift,
                start_shift=start_shift,
                size_increment=size_increment,
                )

