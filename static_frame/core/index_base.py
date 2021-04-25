import typing as tp

import numpy as np

from static_frame.core.container import ContainerOperand
from static_frame.core.display import Display
from static_frame.core.display import DisplayActive
from static_frame.core.display_config import DisplayConfig
from static_frame.core.display_config import DisplayFormats
from static_frame.core.doc_str import doc_inject
from static_frame.core.exception import ErrorInitIndex
from static_frame.core.util import DepthLevelSpecifier
from static_frame.core.util import GetItemKeyType
from static_frame.core.util import NameType
from static_frame.core.util import PathSpecifierOrFileLike
from static_frame.core.util import UFunc
from static_frame.core.util import write_optional_file
from static_frame.core.util import iterable_to_array_1d
from static_frame.core.util import resolve_dtype
from static_frame.core.util import dtype_from_element


if tp.TYPE_CHECKING:
    import pandas #pylint: disable=W0611 #pragma: no cover
    from static_frame.core.series import Series #pylint: disable=W0611,C0412 #pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchy #pylint: disable=W0611,C0412 #pragma: no cover
    from static_frame.core.index_auto import RelabelInput #pylint: disable=W0611,C0412 #pragma: no cover

I = tp.TypeVar('I', bound='IndexBase')

class IndexBase(ContainerOperand):
    '''
    All indices are derived from ``IndexBase``, including ``Index`` and ``IndexHierarchy``.
    '''

    __slots__ = () # defined in derived classes

    #---------------------------------------------------------------------------

    _recache: bool
    _name: NameType
    values: np.ndarray
    positions: np.ndarray
    depth: int

    loc: tp.Any
    iloc: tp.Any # this does not work: InterfaceGetItem[I]
    dtype: np.dtype

    __pos__: tp.Callable[['IndexBase'], np.ndarray]
    __neg__: tp.Callable[['IndexBase'], np.ndarray]
    __abs__: tp.Callable[['IndexBase'], np.ndarray]
    __invert__: tp.Callable[['IndexBase'], np.ndarray]
    __add__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __sub__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __mul__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __matmul__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __truediv__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __floordiv__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __mod__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    # __divmod__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __pow__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __lshift__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __rshift__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __and__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __xor__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __or__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __lt__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __le__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __eq__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __ne__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __gt__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __ge__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __radd__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __rsub__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __rmul__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __rtruediv__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    __rfloordiv__: tp.Callable[['IndexBase', tp.Any], np.ndarray]
    # __len__: tp.Callable[['IndexBase'], int]

    _IMMUTABLE_CONSTRUCTOR: tp.Callable[..., 'IndexBase']
    _MUTABLE_CONSTRUCTOR: tp.Callable[..., 'IndexBase']

    _UFUNC_UNION: tp.Callable[[np.ndarray, np.ndarray, bool], np.ndarray]
    _UFUNC_INTERSECTION: tp.Callable[[np.ndarray, np.ndarray, bool], np.ndarray]
    _UFUNC_DIFFERENCE: tp.Callable[[np.ndarray, np.ndarray, bool], np.ndarray]

    label_widths_at_depth: tp.Callable[[I, int], tp.Iterator[tp.Tuple[tp.Hashable, int]]]

    #---------------------------------------------------------------------------
    # base class interface, mostly for mypy

    #---------------------------------------------------------------------------
    # constructors

    @classmethod
    def from_pandas(cls,
            value: 'pandas.Index',
            ) -> 'IndexBase':
        '''
        Given a Pandas index, return the appropriate IndexBase derived class.
        '''
        import pandas
        if not isinstance(value, pandas.Index):
            raise ErrorInitIndex(f'from_pandas must be called with a Pandas Index object, not: {type(value)}')

        from static_frame import Index
        from static_frame import IndexGO
        from static_frame import IndexHierarchy
        from static_frame import IndexHierarchyGO
        from static_frame import IndexNanosecond
        from static_frame import IndexNanosecondGO
        from static_frame.core.index_datetime import IndexDatetime

        if isinstance(value, pandas.MultiIndex):
            # iterating over a hierarchical index will iterate over labels
            name: tp.Optional[tp.Tuple[tp.Hashable, ...]] = tuple(value.names)
            # if not assigned Pandas returns None for all components, which will raise issue if trying to unset this index.
            if all(n is None for n in name): #type: ignore
                name = None
            depth = value.nlevels

            if not cls.STATIC:
                return IndexHierarchyGO.from_labels(value,
                        name=name,
                        depth_reference=depth)
            return IndexHierarchy.from_labels(value,
                    name=name,
                    depth_reference=depth)
        elif isinstance(value, pandas.DatetimeIndex):
            # if IndexDatetime, use cls, else use IndexNanosecond
            if issubclass(cls, IndexDatetime):
                return cls(value, name=value.name)
            else:
                if not cls.STATIC:
                    return IndexNanosecondGO(value, name=value.name)
                return IndexNanosecond(value, name=value.name)

        if not cls.STATIC:
            return IndexGO(value, name=value.name)
        return Index(value, name=value.name)


    @classmethod
    def from_labels(cls: tp.Type[I],
            labels: tp.Iterable[tp.Sequence[tp.Hashable]],
            *,
            name: tp.Optional[tp.Hashable] = None
            ) -> I:
        raise NotImplementedError()

    def __init__(self, initializer: tp.Any = None,
            *,
            name: tp.Optional[tp.Hashable] = None
            ):
        # trivial init for mypy; not called by derived class
        pass

    #---------------------------------------------------------------------------

    def __len__(self) -> int:
        raise NotImplementedError() #pragma: no cover

    def __iter__(self) -> tp.Iterator[tp.Hashable]:
        raise NotImplementedError() #pragma: no cover

    def __contains__(self, value: tp.Hashable) -> bool:
        raise NotImplementedError() #pragma: no cover

    @property
    def shape(self) -> tp.Tuple[int, ...]:
        raise NotImplementedError() #pragma: no cover

    @property
    def ndim(self) -> int:
        raise NotImplementedError() #pragma: no cover

    def values_at_depth(self,
            depth_level: DepthLevelSpecifier = 0
            ) -> np.ndarray:
        raise NotImplementedError() #pragma: no cover

    def _extract_iloc(self: I, key: GetItemKeyType) -> tp.Union[I, tp.Hashable]:
        raise NotImplementedError() #pragma: no cover

    def _update_array_cache(self) -> None:
        raise NotImplementedError()

    def copy(self: I) -> I:
        raise NotImplementedError()

    def relabel(self: I, mapper: 'RelabelInput') -> I:
        raise NotImplementedError() #pragma: no cover

    def rename(self: I, name: NameType) -> I:
        raise NotImplementedError() #pragma: no cover

    def _drop_iloc(self: I, key: GetItemKeyType) -> I:
        raise NotImplementedError() #pragma: no cover

    def isin(self, other: tp.Iterable[tp.Any]) -> np.ndarray:
        raise NotImplementedError() #pragma: no cover

    def roll(self: I, shift: int) -> I:
        raise NotImplementedError() #pragma: no cover

    def fillna(self: I, value: tp.Any) -> I:
        raise NotImplementedError() #pragma: no cover

    def _sample_and_key(self: I,
            count: int = 1,
            *,
            seed: tp.Optional[int] = None,
            ) -> tp.Tuple[I, np.ndarray]:
        raise NotImplementedError() #pragma: no cover

    def level_add(self, level: tp.Hashable) -> 'IndexHierarchy':
        raise NotImplementedError() #pragma: no cover

    def display(self, config: tp.Optional[DisplayConfig] = None) -> Display:
        raise NotImplementedError()

    #---------------------------------------------------------------------------

    @doc_inject(selector='sample')
    def sample(self: I,
            count: int = 1,
            *,
            seed: tp.Optional[int] = None,
            ) -> I:
        '''{doc}

        Args:
            {count}
            {seed}
        '''
        container, _ = self._sample_and_key(count=count, seed=seed)
        return container

    #---------------------------------------------------------------------------

    @doc_inject(selector='searchsorted', label_type='iloc (integer)')
    def iloc_searchsorted(self,
            values: tp.Any,
            *,
            side_left: bool = True,
            ) -> tp.Union[tp.Hashable, tp.Iterable[tp.Hashable]]:
        '''
        {doc}

        Args:
            {values}
            {side_left}
        '''
        if not isinstance(values, str) and hasattr(values, '__len__'):
            if not values.__class__ is np.ndarray:
                values, _ = iterable_to_array_1d(values)
        return np.searchsorted(self.values, #type: ignore [no-any-return]
                values,
                'left' if side_left else 'right',
                )

    @doc_inject(selector='searchsorted', label_type='loc (label)')
    def loc_searchsorted(self,
            values: tp.Any,
            *,
            side_left: bool = True,
            fill_value: tp.Any = np.nan,
            ) -> tp.Union[tp.Hashable, tp.Iterable[tp.Hashable]]:
        '''
        {doc}

        Args:
            {values}
            {side_left}
            {fill_value}
        '''
        sel = self.iloc_searchsorted(values, side_left=side_left)

        length = self.__len__()
        if sel.ndim == 0 and sel == length: # an element:
            return fill_value #type: ignore [no-any-return]

        mask = sel == length
        if not mask.any():
            return self.values[sel] #type: ignore [no-any-return]

        post = np.empty(len(sel),
                dtype=resolve_dtype(self.dtype,
                dtype_from_element(fill_value))
                )
        sel[mask] = 0 # set out of range values to zero
        post[:] = self.values[sel]
        post[mask] = fill_value
        post.flags.writeable = False
        return post #type: ignore [no-any-return]

    #---------------------------------------------------------------------------

    def _loc_to_iloc(self,
            key: GetItemKeyType,
            ) -> GetItemKeyType:
        raise NotImplementedError()

    def loc_to_iloc(self,
            key: GetItemKeyType,
            ) -> GetItemKeyType:
        raise NotImplementedError()

    def __getitem__(self: I,
            key: GetItemKeyType
            ) -> tp.Union[I, tp.Hashable]:
        raise NotImplementedError() #pragma: no cover

    #---------------------------------------------------------------------------
    # name interface

    @property #type: ignore
    @doc_inject()
    def name(self) -> NameType:
        '''{}'''
        return self._name

    def _name_is_names(self) -> bool:
        return isinstance(self._name, tuple) and len(self._name) == self.depth

    @property
    def names(self) -> tp.Tuple[str, ...]:
        '''
        Provide a suitable iterable of names for usage in output formats that require a field name as string for the index.
        '''
        template = '__index{}__' # arrow does __index_level_0__
        depth = self.depth
        name = self._name

        def gen() -> tp.Iterator[str]:
            if name and depth == 1:
                yield str(name)
            # try to use name only if it is a tuple of the right size
            elif name and self._name_is_names():
                for n in name: #type: ignore [attr-defined]
                    yield str(n)
            else:
                for i in range(depth):
                    yield template.format(i)

        return tuple(gen())


    #---------------------------------------------------------------------------
    # transformations resulting in reduced dimensionality

    @doc_inject(selector='head', class_name='Index')
    def head(self: I, count: int = 5) -> I:
        '''{doc}

        Args:
            {count}
        '''
        return self.iloc[:count] #type: ignore

    @doc_inject(selector='tail', class_name='Index')
    def tail(self: I, count: int = 5) -> I:
        '''{doc}

        Args:
            {count}
        '''
        return self.iloc[-count:] #type: ignore

    #---------------------------------------------------------------------------
    # set operations

    def _ufunc_set(self: I,
            func: tp.Callable[[np.ndarray, np.ndarray, bool], np.ndarray],
            other: tp.Union['IndexBase', tp.Iterable[tp.Hashable]]
            ) -> I:
        raise NotImplementedError() #pragma: no cover

    def intersection(self: I, *others: tp.Union['IndexBase', tp.Iterable[tp.Hashable]]) -> I:
        '''
        Perform intersection with one or many Index, container, or NumPy array. Identical comparisons retain order.
        '''
        # NOTE: must get UFunc off of class to avoid automatic addition of self to signature
        func = self.__class__._UFUNC_INTERSECTION
        if len(others) == 1:
            return self._ufunc_set(func, others[0])

        post = self
        for other in others:
            post = post._ufunc_set(func, other)
        return post

    def union(self: I, *others: tp.Union['IndexBase', tp.Iterable[tp.Hashable]]) -> I:
        '''
        Perform union with another Index, container, or NumPy array. Identical comparisons retain order.
        '''
        func = self.__class__._UFUNC_UNION
        if len(others) == 1:
            return self._ufunc_set(func, others[0])

        post = self
        for other in others:
            post = post._ufunc_set(func, other)
        return post


    def difference(self: I, other: tp.Union['IndexBase', tp.Iterable[tp.Hashable]]) -> I:
        '''
        Perform difference with another Index, container, or NumPy array. Retains order.
        '''
        return self._ufunc_set(
                self.__class__._UFUNC_DIFFERENCE,
                other)

    #---------------------------------------------------------------------------
    # metaclass-applied functions

    def _ufunc_shape_skipna(self, *,
            axis: int,
            skipna: bool,
            ufunc: UFunc,
            ufunc_skipna: UFunc,
            composable: bool,
            dtypes: tp.Tuple[np.dtype, ...],
            size_one_unity: bool
            ) -> np.ndarray:
        '''
        For Index and IndexHierarchy, _ufunc_shape_skipna and _ufunc_axis_skipna are defined the same.

        Returns:
            immutable NumPy array.
        '''
        return self._ufunc_axis_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=ufunc,
                ufunc_skipna=ufunc_skipna,
                composable=composable, # shape on axis 1 is never composable
                dtypes=dtypes,
                size_one_unity=size_one_unity
                )

    #---------------------------------------------------------------------------
    # exporters

    @doc_inject(class_name='Index')
    def to_html(self,
            config: tp.Optional[DisplayConfig] = None
            ) -> str:
        '''
        {}
        '''
        config = config or DisplayActive.get(type_show=False)
        config = config.to_display_config(
                display_format=DisplayFormats.HTML_TABLE,
                )
        return repr(self.display(config))

    @doc_inject(class_name='Index')
    def to_html_datatables(self,
            fp: tp.Optional[PathSpecifierOrFileLike] = None,
            *,
            show: bool = True,
            config: tp.Optional[DisplayConfig] = None
            ) -> tp.Optional[str]:
        '''
        {}
        '''
        config = config or DisplayActive.get(type_show=False)
        config = config.to_display_config(
                display_format=DisplayFormats.HTML_DATATABLES,
                )
        content = repr(self.display(config))
        # path_filter called internally
        fp = write_optional_file(content=content, fp=fp)

        if fp and show:
            import webbrowser #pragma: no cover
            webbrowser.open_new_tab(fp) #pragma: no cover

        return fp

    def to_pandas(self) -> 'pandas.Series':
        raise NotImplementedError() #pragma: no cover


