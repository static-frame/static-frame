import typing as tp
import numpy as np  # type: ignore

from static_frame.core.util import mloc
from static_frame.core.util import FilePathOrFileLike
from static_frame.core.util import write_optional_file
from static_frame.core.util import IndexInitializer
from static_frame.core.util import IndexConstructor
from static_frame.core.util import UFunc

from static_frame.core.display import DisplayFormats
from static_frame.core.display import DisplayActive
from static_frame.core.display import DisplayConfig
from static_frame.core.display import Display

from static_frame.core.doc_str import doc_inject
from static_frame.core.container import ContainerBase

if tp.TYPE_CHECKING:

    import pandas as pd  # type: ignore


I = tp.TypeVar('I', bound='IndexBase')

class IndexBase(ContainerBase):

    __slots__ = () # defined in dervied classes

    _map: tp.Dict[tp.Hashable, tp.Any]
    _labels: np.ndarray
    _positions: np.ndarray
    _recache: bool
    _loc_is_iloc: bool
    _name: tp.Hashable
    values: np.ndarray


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



    STATIC = True

    _IMMUTABLE_CONSTRUCTOR: tp.Callable[..., 'IndexBase']
    _MUTABLE_CONSTRUCTOR: tp.Callable[..., 'IndexBase']

    _UFUNC_UNION: tp.Callable[[np.ndarray, np.ndarray], np.ndarray]
    _UFUNC_INTERSECTION: tp.Callable[[np.ndarray, np.ndarray], np.ndarray]

    def _ufunc_axis_skipna(self, *,
            axis: int,
            skipna: bool,
            ufunc: UFunc,
            ufunc_skipna: UFunc,
            dtype: tp.Optional[np.dtype] = None,
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
            value: 'pd.DataFrame',
            *,
            is_go: bool = False) -> 'IndexBase':
        '''
        Given a Pandas index, return the appropriate IndexBase derived class.
        '''
        import pandas
        from static_frame import Index
        from static_frame import IndexGO
        from static_frame import IndexDate
        from static_frame import IndexHierarchy
        from static_frame import IndexHierarchyGO

        if isinstance(value, pandas.MultiIndex):
            # iterating over a hierarchucal index will iterate over labels
            name = tuple(value.names)
            if is_go:
                return IndexHierarchyGO.from_labels(value, name=name)
            return IndexHierarchy.from_labels(value, name=name)
        elif isinstance(value, pandas.DatetimeIndex):
            if is_go:
                raise NotImplementedError('No grow-only version of IndexDate yet exists')
            return IndexDate(value, name=value.name)

        if is_go:
            return IndexGO(value, name=value.name)
        return Index(value, name=value.name)




    #---------------------------------------------------------------------------
    # name interface

    @property
    def name(self) -> tp.Hashable:
        return self._name

    #---------------------------------------------------------------------------
    # common attributes from the numpy array

    @property
    def mloc(self) -> int:
        '''Memory location
        '''
        if self._recache:
            self._update_array_cache()
        return mloc(self._labels)

    @property
    def dtype(self) -> np.dtype:
        '''
        Return the dtype of the underlying NumPy array.

        Returns:
            :py:class:`numpy.dtype`
        '''
        if self._recache:
            self._update_array_cache()
        return self._labels.dtype

    @property
    def shape(self) -> tp.Tuple[int, ...]:
        '''
        Return a tuple describing the shape of the underlying NumPy array.

        Returns:
            :py:class:`tp.Tuple[int]`
        '''
        if self._recache:
            self._update_array_cache()
        return tp.cast(tp.Tuple[int, ...], self.values.shape)

    @property
    def ndim(self) -> int:
        '''
        Return the number of dimensions.

        Returns:
            :py:class:`int`
        '''
        if self._recache:
            self._update_array_cache()
        return tp.cast(int, self._labels.ndim)

    @property
    def size(self) -> int:
        '''
        Return the size of the underlying NumPy array.

        Returns:
            :py:class:`int`
        '''
        if self._recache:
            self._update_array_cache()
        return tp.cast(int, self._labels.size)

    @property
    def nbytes(self) -> int:
        '''
        Return the total bytes of the underlying NumPy array.

        Returns:
            :py:class:`int`
        '''
        if self._recache:
            self._update_array_cache()
        return tp.cast(int, self._labels.nbytes)


    #---------------------------------------------------------------------------
    # set operations

    def intersection(self: I, other: 'IndexBase') -> I:
        if self._recache:
            self._update_array_cache()

        if isinstance(other, np.ndarray):
            opperand = other
        else: # assume we can get it from a .values attribute
            opperand = other.values

        cls = self.__class__
        return cls.from_labels(cls._UFUNC_INTERSECTION(self._labels, opperand))

    def union(self: I, other: 'IndexBase') -> I:
        if self._recache:
            self._update_array_cache()

        if isinstance(other, np.ndarray):
            opperand = other
        else: # assume we can get it from a .values attribute
            opperand = other.values

        cls = self.__class__
        return cls.from_labels(cls._UFUNC_UNION(self._labels, opperand))

    #---------------------------------------------------------------------------
    # dictionary-like interface

    def __iter__(self) -> tp.Iterator[tp.Hashable]:
        '''Iterate over labels.
        '''
        if self._recache:
            self._update_array_cache()
        return tp.cast(tp.Iterator[tp.Hashable], self._labels.__iter__())

    def __reversed__(self) -> tp.Iterator[tp.Hashable]:
        '''
        Returns a reverse iterator on the index labels.
        '''
        if self._recache:
            self._update_array_cache()
        return reversed(self._labels)

    #---------------------------------------------------------------------------
    # metaclass-applied functions

    def _ufunc_shape_skipna(self, *,
            axis: int,
            skipna: bool,
            ufunc: UFunc,
            ufunc_skipna: UFunc,
            dtype: tp.Optional[np.dtype] = None,
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
                dtype=dtype
                )


    #---------------------------------------------------------------------------
    # common display

    def __repr__(self) -> str:
        return repr(self.display())

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
            fp: tp.Optional[FilePathOrFileLike] = None,
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
        fp = write_optional_file(content=content, fp=fp)

        if fp and show:
            import webbrowser
            webbrowser.open_new_tab(fp)

        return fp





def index_from_optional_constructor(
        value: tp.Union[IndexInitializer, IndexBase],
        default_constructor: IndexConstructor,
        ) -> IndexBase:
    '''
    Given a value that might be an Index or and IndexInitializer, determine if that value is an Index, and if so, determine if a copy has to be made; otherwise, use the default_constructor
    '''
    if isinstance(value, IndexBase) and isinstance(default_constructor, IndexBase):
        # if default is STATIC, use value is not STATIC, get an immutabel
        if default_constructor.STATIC:
            if not value.STATIC:
                # use the immutabel alternative
                return value._IMMUTABLE_CONSTRUCTOR(value)
            return value # immutable, can reuse
        else: # default constructor is mutable
            if not value.STATIC: # both are mutable
                return value.copy()
            # need to return a mutable version of something that is not mutabel
            # TODO: not sure how to do this
            return default_constructor(value)

    return default_constructor(value)
