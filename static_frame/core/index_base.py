import typing as tp
import numpy as np

from static_frame.core.util import mloc
from static_frame.core.util import PathSpecifierOrFileLike
from static_frame.core.util import write_optional_file
# from static_frame.core.util import IndexInitializer
# from static_frame.core.util import IndexConstructor
from static_frame.core.util import UFunc


from static_frame.core.display import DisplayFormats
from static_frame.core.display import DisplayActive
from static_frame.core.display import DisplayConfig
from static_frame.core.display import Display

from static_frame.core.doc_str import doc_inject
from static_frame.core.container import ContainerOperand
# from static_frame.core.selector_node import InterfaceGetItem
# from static_frame.core.selector_node import TContainer
# from static_frame.core.exception import ErrorInitIndex

if tp.TYPE_CHECKING:
    import pandas #pylint: disable=W0611 #pragma: no cover
    from static_frame.core.series import Series #pylint: disable=W0611 #pragma: no cover


I = tp.TypeVar('I', bound='IndexBase')

class IndexBase(ContainerOperand):
    '''
    All indices are dervied from ``IndexBase``, including ``Index`` and ``IndexHierarchy``.
    '''

    __slots__ = () # defined in dervied classes

    #---------------------------------------------------------------------------
    # type defs

    _map: tp.Optional[tp.Dict[tp.Hashable, int]]
    _labels: np.ndarray
    _positions: np.ndarray
    _recache: bool
    _name: tp.Hashable
    values: np.ndarray
    depth: int
    iloc: tp.Any # this does not work: InterfaceGetItem[I]

    __pos__: tp.Callable[['IndexBase'], np.ndarray]
    __neg__: tp.Callable[['IndexBase'], np.ndarray]
    __abs__: tp.Callable[['IndexBase'], np.ndarray]
    __invert__: tp.Callable[['IndexBase'], np.ndarray]
    __add__: tp.Callable[['IndexBase', object], np.ndarray]
    __sub__: tp.Callable[['IndexBase', object], np.ndarray]
    __mul__: tp.Callable[['IndexBase', object], np.ndarray]
    __matmul__: tp.Callable[['IndexBase', object], np.ndarray]
    __truediv__: tp.Callable[['IndexBase', object], np.ndarray]
    __floordiv__: tp.Callable[['IndexBase', object], np.ndarray]
    __mod__: tp.Callable[['IndexBase', object], np.ndarray]
    # __divmod__: tp.Callable[['IndexBase', object], np.ndarray]
    __pow__: tp.Callable[['IndexBase', object], np.ndarray]
    __lshift__: tp.Callable[['IndexBase', object], np.ndarray]
    __rshift__: tp.Callable[['IndexBase', object], np.ndarray]
    __and__: tp.Callable[['IndexBase', object], np.ndarray]
    __xor__: tp.Callable[['IndexBase', object], np.ndarray]
    __or__: tp.Callable[['IndexBase', object], np.ndarray]
    __lt__: tp.Callable[['IndexBase', object], np.ndarray]
    __le__: tp.Callable[['IndexBase', object], np.ndarray]
    __eq__: tp.Callable[['IndexBase', object], np.ndarray]
    __ne__: tp.Callable[['IndexBase', object], np.ndarray]
    __gt__: tp.Callable[['IndexBase', object], np.ndarray]
    __ge__: tp.Callable[['IndexBase', object], np.ndarray]
    __radd__: tp.Callable[['IndexBase', object], np.ndarray]
    __rsub__: tp.Callable[['IndexBase', object], np.ndarray]
    __rmul__: tp.Callable[['IndexBase', object], np.ndarray]
    __rtruediv__: tp.Callable[['IndexBase', object], np.ndarray]
    __rfloordiv__: tp.Callable[['IndexBase', object], np.ndarray]
    __len__: tp.Callable[['IndexBase'], int]

    _IMMUTABLE_CONSTRUCTOR: tp.Callable[..., 'IndexBase']
    _MUTABLE_CONSTRUCTOR: tp.Callable[..., 'IndexBase']

    _UFUNC_UNION: tp.Callable[[np.ndarray, np.ndarray, bool], np.ndarray]
    _UFUNC_INTERSECTION: tp.Callable[[np.ndarray, np.ndarray, bool], np.ndarray]
    _UFUNC_DIFFERENCE: tp.Callable[[np.ndarray, np.ndarray, bool], np.ndarray]

    label_widths_at_depth: tp.Callable[[I, int], tp.Iterator[tp.Tuple[tp.Hashable, int]]]


    #---------------------------------------------------------------------------
    # class attrs

    STATIC: bool = True

    def _ufunc_axis_skipna(self, *,
            axis: int,
            skipna: bool,
            ufunc: UFunc,
            ufunc_skipna: UFunc,
            composable: bool,
            dtypes: tp.Tuple[np.dtype, ...],
            size_one_unity: bool
            ) -> np.ndarray:
        raise NotImplementedError()

    def _update_array_cache(self) -> None:
        raise NotImplementedError()


    def copy(self: I) -> I:
        raise NotImplementedError()


    def display(self, config: tp.Optional[DisplayConfig] = None) -> Display:
        raise NotImplementedError()


    @classmethod
    def from_labels(cls: tp.Type[I], labels: tp.Iterable[tp.Sequence[tp.Hashable]]) -> I:
        raise NotImplementedError()


    #---------------------------------------------------------------------------
    # constructors

    @classmethod
    def from_pandas(cls,
            value: 'pandas.DataFrame',
            ) -> 'IndexBase':
        '''
        Given a Pandas index, return the appropriate IndexBase derived class.
        '''
        import pandas
        from static_frame import Index
        from static_frame import IndexGO
        from static_frame import IndexHierarchy
        from static_frame import IndexHierarchyGO
        from static_frame import IndexNanosecond
        from static_frame import IndexNanosecondGO
        from static_frame.core.index_datetime import IndexDatetime

        if isinstance(value, pandas.MultiIndex):
            # iterating over a hierarchucal index will iterate over labels
            name = tuple(value.names)
            if not cls.STATIC:
                return IndexHierarchyGO.from_labels(value, name=name)
            return IndexHierarchy.from_labels(value, name=name)
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

    #---------------------------------------------------------------------------
    # name interface

    @property #type: ignore
    @doc_inject()
    def name(self) -> tp.Hashable:
        '''{}'''
        return self._name

    @property
    def names(self) -> tp.Tuple[str, ...]:
        '''
        Provide a suitable iterable of names for usage in output formats that require a field name for the index.
        '''
        template = '__index{}__' # arrow does __index_level_0__

        def gen() -> tp.Iterator[str]:
            if self._name and isinstance(self._name, tuple):
                for name in self._name:
                    yield name
            else:
                start = 0
                if self._name:
                    # If self._name is defined, try to use it as the first label
                    yield str(self._name) # might be other hashable
                    start = 1
                for i in range(start, self.depth):
                    yield template.format(i)

        names = tuple(gen())
        if len(names) != self.depth:
            raise RuntimeError(f'unexpected names formation: {names}, does not meet depth {self.depth}')
        return names

    #---------------------------------------------------------------------------
    # common attributes from the numpy array

    @property # type: ignore
    @doc_inject()
    def mloc(self) -> int:
        '''{doc_int}
        '''
        if self._recache:
            self._update_array_cache()
        return mloc(self._labels)

    @property
    def dtype(self) -> np.dtype:
        '''
        Return the dtype of the underlying NumPy array.

        Returns:
            :obj:`numpy.dtype`
        '''
        if self._recache:
            self._update_array_cache()
        return self._labels.dtype

    @property
    def shape(self) -> tp.Tuple[int, ...]:
        '''
        Return a tuple describing the shape of the underlying NumPy array.

        Returns:
            :obj:`tp.Tuple[int]`
        '''
        if self._recache:
            self._update_array_cache()
        return tp.cast(tp.Tuple[int, ...], self.values.shape)

    @property
    def ndim(self) -> int:
        '''
        Return the number of dimensions.

        Returns:
            :obj:`int`
        '''
        if self._recache:
            self._update_array_cache()
        return tp.cast(int, self._labels.ndim)

    @property
    def size(self) -> int:
        '''
        Return the size of the underlying NumPy array.

        Returns:
            :obj:`int`
        '''
        if self._recache:
            self._update_array_cache()
        return tp.cast(int, self._labels.size)

    @property
    def nbytes(self) -> int:
        '''
        Return the total bytes of the underlying NumPy array.

        Returns:
            :obj:`int`
        '''
        if self._recache:
            self._update_array_cache()
        return tp.cast(int, self._labels.nbytes)

    def __bool__(self) -> bool:
        '''
        True if this container has size.
        '''
        if self._recache:
            self._update_array_cache()
        return bool(self._labels.size)

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
            other: tp.Union['IndexBase', 'Series']
            ) -> I:
        '''
        Utility function for preparing and collecting values for Indices to produce a new Index.
        '''
        if self._recache:
            self._update_array_cache()

        if isinstance(other, np.ndarray):
            opperand = other
            assume_unique = False
        elif isinstance(other, IndexBase):
            opperand = other.values
            assume_unique = True # can always assume unique
        elif isinstance(other, ContainerOperand):
            opperand = other.values
            assume_unique = False
        else:
            raise NotImplementedError(f'no support for {other}')

        cls = self.__class__

        # using assume_unique will permit retaining order when opperands are identical
        labels = func(self._labels, opperand, assume_unique=assume_unique) # type: ignore

        if id(labels) == id(self._labels):
            # NOTE: favor using cls constructor here as it permits maximal sharing of static resources and the underlying dictionary
            return cls(self) # type: ignore
        return cls.from_labels(labels)


    def intersection(self: I, other: tp.Union['IndexBase', 'Series']) -> I:
        '''
        Perform intersection with another Index, container, or NumPy array. Identical comparisons retain order.
        '''
        # NOTE: must get UFunc off of class to avoid automatic addition of self to signature
        return self._ufunc_set(
                self.__class__._UFUNC_INTERSECTION,
                other)

    def union(self: I, other: 'IndexBase') -> I:
        '''
        Perform union with another Index, container, or NumPy array. Identical comparisons retain order.
        '''
        return self._ufunc_set(
                self.__class__._UFUNC_UNION,
                other)

    def difference(self: I, other: 'IndexBase') -> I:
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
    # common display

    def _repr_html_(self) -> str:
        '''
        Provide HTML representation for Jupyter Notebooks.
        '''
        # modify the active display to be force HTML
        config = DisplayActive.get(
                display_format=DisplayFormats.HTML_TABLE,
                type_show=False
                )
        return repr(self.display(config))

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
