import typing as tp
import numpy as np

from static_frame.core.util import mloc
from static_frame.core.util import FilePathOrFileLike
from static_frame.core.util import write_optional_file
from static_frame.core.util import IndexInitializer
from static_frame.core.util import IndexConstructor

from static_frame.core.display import DisplayFormats
from static_frame.core.display import DisplayActive
from static_frame.core.display import DisplayConfig

from static_frame.core.doc_str import doc_inject

class IndexBase:

    __slots__ = () # defined in dervied classes

    STATIC = True

    _IMMUTABLE_CONSTRUCTOR = None
    _MUTABLE_CONSTRUCTOR = None

    _UFUNC_UNION = None
    _UFUNC_INTERSECTION = None


    #---------------------------------------------------------------------------
    # constructors

    @classmethod
    def from_pandas(cls,
            value,
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
    def mloc(self):
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
    def shape(self) -> tp.Tuple[int]:
        '''
        Return a tuple describing the shape of the underlying NumPy array.

        Returns:
            :py:class:`tp.Tuple[int]`
        '''
        if self._recache:
            self._update_array_cache()
        return self.values.shape

    @property
    def ndim(self) -> int:
        '''
        Return the number of dimensions.

        Returns:
            :py:class:`int`
        '''
        if self._recache:
            self._update_array_cache()
        return self._labels.ndim

    @property
    def size(self) -> int:
        '''
        Return the size of the underlying NumPy array.

        Returns:
            :py:class:`int`
        '''
        if self._recache:
            self._update_array_cache()
        return self._labels.size

    @property
    def nbytes(self) -> int:
        '''
        Return the total bytes of the underlying NumPy array.

        Returns:
            :py:class:`int`
        '''
        if self._recache:
            self._update_array_cache()
        return self._labels.nbytes


    #---------------------------------------------------------------------------
    # set operations

    def intersection(self, other) -> 'Index':
        if self._recache:
            self._update_array_cache()

        if isinstance(other, np.ndarray):
            opperand = other
        else: # assume we can get it from a .values attribute
            opperand = other.values

        cls = self.__class__
        return cls.from_labels(cls._UFUNC_INTERSECTION(self._labels, opperand))

    def union(self, other) -> 'Index':
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

    def __iter__(self):
        '''Iterate over labels.
        '''
        if self._recache:
            self._update_array_cache()
        return self._labels.__iter__()

    def __reversed__(self) -> tp.Iterable[tp.Hashable]:
        '''
        Returns a reverse iterator on the index labels.
        '''
        if self._recache:
            self._update_array_cache()
        return reversed(self._labels)

    #---------------------------------------------------------------------------
    # metaclass-applied functions

    def _ufunc_shape_skipna(self, *,
            axis,
            skipna,
            ufunc,
            ufunc_skipna,
            dtype=None
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

    def _repr_html_(self):
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
            ):
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
            ) -> str:
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
    if isinstance(value, IndexBase):
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
