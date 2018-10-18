'''
Tools for iterators in Series and Frame. These components are imported by both series.py and frame.py; these components also need to be able to return Series and Frame, and thus use deferred, function-based imports.
'''


import typing as tp
from enum import Enum


from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor


from static_frame.core.util import CallableOrMapping

from functools import partial





class IterNodeApplyType(Enum):
    SERIES_ITEMS = 1
    FRAME_ELEMENTS = 2


class IterNodeType(Enum):
    VALUES = 1
    ITEMS = 2


class IterNodeDelegate:
    '''
    Delegate returned from :py:class:`IterNode`, providing iteration as well as a family of apply methods.
    '''

    __slots__ = (
            '_func_values',
            '_func_items',
            '_yield_type',
            '_apply_constructor'
            )

    def __init__(self,
            func_values,
            func_items,
            yield_type: IterNodeType,
            apply_constructor) -> None:
        '''
        Args:
            apply_constructor: Callable (generally a class) used to construct the object returned from apply(); must take an iterator of items.
        '''
        self._func_values = func_values
        self._func_items= func_items
        self._yield_type = yield_type
        self._apply_constructor = apply_constructor

    #---------------------------------------------------------------------------

    def _apply_iter_items_parallel(self,
            func: CallableOrMapping,
            max_workers=None,
            chunksize=1,
            use_threads=False,
            ) -> tp.Generator[tp.Tuple[tp.Any, tp.Any], None, None]:

        pool_executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

        if not callable(func):
            func = getattr(func, '__getitem__')

        # use side effect list population to create keys when iterating over values
        func_keys = []
        if self._yield_type is IterNodeType.VALUES:
            def arg_gen():
                for k, v in self._func_items():
                    func_keys.append(k)
                    yield v
        else:
            def arg_gen():
                for k, v in self._func_items():
                    func_keys.append(k)
                    yield k, v

        with pool_executor(max_workers=max_workers) as executor:
            yield from zip(func_keys,
                    executor.map(func, arg_gen(), chunksize=chunksize)
                    )

    #---------------------------------------------------------------------------
    # public interface


    def apply_iter_items(self,
            func: CallableOrMapping) -> tp.Generator[tp.Tuple[tp.Any, tp.Any], None, None]:
        '''
        Generator that applies function to each element iterated and yields the pair of element and the result.

        Args:
            func: A function or a mapping object that defines __getitem__ and __contains__. If a mpping is given and a value is not found in the mapping, the value is returned unchanged (this deviates from Pandas Series.map, which inserts NaNs)
        '''
        condition = None
        if not callable(func):
            # if the key is not in the map, we return the value unaltered
            condition = getattr(func, '__contains__')
            func = getattr(func, '__getitem__')

        # apply always calls the items function
        for k, v in self._func_items():
            if condition and not condition(v):
                if self._yield_type is IterNodeType.VALUES:
                    yield k, v
                else: # items, give both keys and values to function
                    yield k, (k, v)
            else:
                # depend on yield type, we determine what the passed in function expects to take
                if self._yield_type is IterNodeType.VALUES:
                    yield k, func(v)
                else: # items, give both keys and values to function
                    yield k, func(k, v)


    def apply_iter(self,
            func: CallableOrMapping
            ) -> tp.Generator[tp.Any, None, None]:
        '''
        Generator that applies the passed function to each element iterated and yields the result.

        Args:
            func: A function, or a mapping object that defines __getitem__. If a mapping is given, all values must be found in the mapping.
        '''
        yield from (v for _, v in self.apply_iter_items(func=func))


    def apply(self,
            func: CallableOrMapping,
            dtype=None
            ) -> tp.Union['static_frame.Series', 'static_frame.Frame']:
        '''
        Apply passed function to each object iterated, where the object depends on the creation of this instance.

        Args:
            func: A function, or a mapping object that defines __getitem__. If a mapping is given, all values must be found in the mapping.
            dtype: Type used to create the returned array.
        '''
        return self._apply_constructor(
                self.apply_iter_items(func=func),
                dtype=dtype)


    def apply_pool(self,
            func: CallableOrMapping,
            dtype=None,
            max_workers: tp.Optional[int]=None,
            chunksize: int=1,
            use_threads: bool=False
            ) -> tp.Union['static_frame.Series', 'static_frame.Frame']:
        '''
        Apply passed function to each object iterated, where the object depends on the creation of this instance. Employ parallel processing with either the ProcessPoolExecutor or ThreadPoolExecutor.

        Args:
            func: A function, or a mapping object that defines __getitem__. If a mapping is given, all values must be found in the mapping.
            dtype: Type used to create the returned array.
            max_workers: Passed to the pool_executor, where None defaults to the max number of machine processes.
            chunksize: Passed to the pool executor.
            use_thread: When True, the ThreadPoolExecutor will be used rather than the default ProcessPoolExecutor.
        '''
        return self._apply_constructor(
                self._apply_iter_items_parallel(
                        func=func,
                        max_workers=max_workers,
                        chunksize=chunksize,
                        use_threads=use_threads),
                dtype=dtype)

    def __iter__(self):
        '''
        Return a generator based on the yield type.
        '''
        if self._yield_type is IterNodeType.VALUES:
            yield from self._func_values()
        else:
            yield from self._func_items()


class IterNode:
    '''Iterface to a type of iteration on :py:class:`Series` and :py:class:`Frame`.
    '''
    # '''Stores two version of a generator function: one to yield single values, another to yield items pairs. The latter is needed in all cases, as when we use apply we return a Series, and need to have recourse to an index.
    # '''

    __slots__ = ('_container',
            '_func_values',
            '_func_items',
            '_yield_type',
            '_apply_type'
            )

    def __init__(self, *,
            container: tp.Union['static_frame.Series', 'static_frame.Frame'],
            function_values,
            function_items,
            yield_type: IterNodeType,
            apply_type: IterNodeApplyType=IterNodeApplyType.SERIES_ITEMS
            ) -> None:
        self._container = container
        self._func_values = function_values
        self._func_items = function_items
        self._yield_type = yield_type
        self._apply_type = apply_type

    def __call__(self, *args, **kwargs):
        '''
        In usage as an iteator, the args passed here are expected to be argument for the core iterators, i.e., axis arguments.
        '''
        from static_frame.core.series import Series
        from static_frame.core.frame import Frame

        func_values = partial(self._func_values, *args, **kwargs)
        func_items = partial(self._func_items, *args, **kwargs)

        if self._apply_type is IterNodeApplyType.SERIES_ITEMS:
            apply_constructor = Series.from_items
        elif self._apply_type is IterNodeApplyType.FRAME_ELEMENTS:
            apply_constructor = partial(Frame.from_element_loc_items,
                    index=self._container._index,
                    columns=self._container._columns)
        else:
            raise NotImplementedError()

        return IterNodeDelegate(
                func_values=func_values,
                func_items=func_items,
                yield_type=self._yield_type,
                apply_constructor=apply_constructor
                )





