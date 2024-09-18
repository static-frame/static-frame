from __future__ import annotations

from functools import partial
from itertools import chain

import numpy as np
import typing_extensions as tp

from static_frame.core.container import ContainerOperandSequence
from static_frame.core.container_util import IMTOAdapter
from static_frame.core.container_util import imto_adapter_factory
from static_frame.core.container_util import index_many_to_one
from static_frame.core.display import Display
from static_frame.core.display import DisplayActive
from static_frame.core.display_config import DisplayConfig
from static_frame.core.display_config import DisplayFormats
from static_frame.core.doc_str import doc_inject
from static_frame.core.exception import ErrorInitIndex
from static_frame.core.node_dt import InterfaceDatetime
from static_frame.core.node_iter import IterNodeDepthLevel
from static_frame.core.node_re import InterfaceRe
from static_frame.core.node_str import InterfaceString
from static_frame.core.style_config import STYLE_CONFIG_DEFAULT
from static_frame.core.style_config import StyleConfig
from static_frame.core.style_config import style_config_css_factory
from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import OPERATORS
from static_frame.core.util import ManyToOneType
from static_frame.core.util import TDepthLevel
from static_frame.core.util import TILocSelector
from static_frame.core.util import TILocSelectorMany
from static_frame.core.util import TILocSelectorOne
from static_frame.core.util import TIndexCtorSpecifier
from static_frame.core.util import TKeyTransform
from static_frame.core.util import TLabel
from static_frame.core.util import TLocSelector
from static_frame.core.util import TLocSelectorMany
from static_frame.core.util import TName
from static_frame.core.util import TPathSpecifierOrTextIO
from static_frame.core.util import TUFunc
from static_frame.core.util import isfalsy_array
from static_frame.core.util import isna_array
from static_frame.core.util import write_optional_file

if tp.TYPE_CHECKING:
    import pandas  # pragma: no cover

    from static_frame.core.index_auto import TRelabelInput  # pylint: disable=W0611,C0412 #pragma: no cover
    from static_frame.core.index_hierarchy import IndexHierarchy  # pylint: disable=W0611,C0412 #pragma: no cover
    from static_frame.core.series import Series  # pylint: disable=W0611,C0412 #pragma: no cover
    TNDArrayAny = np.ndarray[tp.Any, tp.Any] #pragma: no cover
    TNDArrayBool = np.ndarray[tp.Any, np.dtype[np.bool_]] #pragma: no cover
    TDtypeAny = np.dtype[tp.Any] #pragma: no cover

I = tp.TypeVar('I', bound='IndexBase')

class IndexBase(ContainerOperandSequence):
    '''
    All indices are derived from ``IndexBase``, including ``Index`` and ``IndexHierarchy``.
    '''

    __slots__ = () # defined in derived classes

    #---------------------------------------------------------------------------

    _recache: bool
    _name: TName
    depth: int
    _NDIM: int

    loc: tp.Any
    iloc: tp.Any # this does not work: InterGetItemLocReduces[I]

    #---------------------------------------------------------------------------
    def _ufunc_unary_operator(self, operator: TUFunc) -> TNDArrayAny:
        raise NotImplementedError() #pragma: no cover

    @property
    def positions(self) -> TNDArrayAny:
        raise NotImplementedError() #pragma: no cover

    #---------------------------------------------------------------------------
    def __pos__(self) -> TNDArrayAny:
        return self._ufunc_unary_operator(OPERATORS['__pos__'])

    def __neg__(self) -> TNDArrayAny:
        return self._ufunc_unary_operator(OPERATORS['__neg__'])

    def __abs__(self) -> TNDArrayAny:
        return self._ufunc_unary_operator(OPERATORS['__abs__'])

    def __invert__(self) -> TNDArrayAny:
        return self._ufunc_unary_operator(OPERATORS['__invert__'])

    __add__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __sub__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __mul__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __matmul__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __truediv__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __floordiv__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __mod__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    # __divmod__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __pow__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __lshift__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __rshift__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __and__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __xor__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __or__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __lt__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __le__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __eq__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __ne__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __gt__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __ge__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __radd__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __rsub__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __rmul__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __rtruediv__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    __rfloordiv__: tp.Callable[['IndexBase', tp.Any], TNDArrayAny]
    # __len__: tp.Callable[['IndexBase'], int]

    _IMMUTABLE_CONSTRUCTOR: tp.Callable[..., 'IndexBase']
    _MUTABLE_CONSTRUCTOR: tp.Callable[..., 'IndexBase']

    def label_widths_at_depth(self,
            depth_level: TDepthLevel = 0
            ) -> tp.Iterator[tp.Tuple[TLabel, int]]:
        raise NotImplementedError() #pragma: no cover

    #---------------------------------------------------------------------------
    # base class interface, mostly for mypy

    #---------------------------------------------------------------------------
    # constructors

    @classmethod
    def from_pandas(cls,
            value: 'pandas.Index',
            ) -> IndexBase:
        '''
        Given a Pandas index, return the appropriate IndexBase derived class.
        '''
        import pandas
        if not isinstance(value, pandas.Index):
            raise ErrorInitIndex(f'from_pandas must be called with a Pandas Index object, not: {type(value)}')

        from static_frame import Index
        from static_frame import IndexGO
        from static_frame import IndexNanosecond
        from static_frame import IndexNanosecondGO
        from static_frame.core.index_datetime import IndexDatetime

        if isinstance(value, pandas.DatetimeIndex):
            # if IndexDatetime, use cls, else use IndexNanosecond
            if issubclass(cls, IndexDatetime):
                return cls(value, name=value.name)

            if not cls.STATIC:
                return IndexNanosecondGO(value, name=value.name)
            return IndexNanosecond(value, name=value.name)

        if not cls.STATIC:
            return IndexGO(value, name=value.name)
        return Index(value, name=value.name)


    @classmethod
    def from_labels(cls: tp.Type[I],
            labels: tp.Iterable[tp.Sequence[TLabel]],
            *,
            name: tp.Optional[TLabel] = None
            ) -> I:
        raise NotImplementedError() #pragma: no cover

    def __init__(self, initializer: tp.Any = None,
            *,
            name: tp.Optional[TLabel] = None
            ):
        # trivial init for mypy; not called by derived class
        pass

    #---------------------------------------------------------------------------

    def __len__(self) -> int:
        raise NotImplementedError() #pragma: no cover

    def __iter__(self) -> tp.Iterator[TLabel]:
        raise NotImplementedError() #pragma: no cover

    def __contains__(self, value: TLabel) -> bool:
        raise NotImplementedError() #pragma: no cover

    @property
    def iter_label(self) -> IterNodeDepthLevel[tp.Any]:
        raise NotImplementedError() #pragma: no cover

    @property
    def shape(self) -> tp.Tuple[int, ...]:
        raise NotImplementedError() #pragma: no cover

    @property
    def ndim(self) -> int:
        raise NotImplementedError() #pragma: no cover

    def values_at_depth(self,
            depth_level: TDepthLevel = 0
            ) -> TNDArrayAny:
        raise NotImplementedError() #pragma: no cover

    @property
    def index_types(self) -> Series[tp.Any, np.object_]:
        # NOTE: this implementation is here due to pydoc.render_doc call that led to calling this base class method
        from static_frame.core.series import Series
        return Series((), dtype=DTYPE_OBJECT) # pragma: no cover


    @tp.overload
    def _extract_iloc(self, key: TILocSelectorOne) -> TLabel: ...

    @tp.overload
    def _extract_iloc(self, key: TILocSelectorMany) -> tp.Self: ...

    def _extract_iloc(self, key: TILocSelector) -> tp.Any:
        raise NotImplementedError() #pragma: no cover

    def _extract_iloc_by_int(self, key: int | np.integer[tp.Any]) -> TLabel:
        raise NotImplementedError() #pragma: no cover

    def _update_array_cache(self) -> None:
        raise NotImplementedError()

    def copy(self: I) -> I:
        raise NotImplementedError()

    def relabel(self: I, mapper: 'TRelabelInput') -> I:
        raise NotImplementedError() #pragma: no cover

    def rename(self: I, name: TName) -> I:
        raise NotImplementedError() #pragma: no cover

    def _drop_iloc(self: I, key: TILocSelector) -> I:
        raise NotImplementedError() #pragma: no cover

    def isin(self, other: tp.Iterable[tp.Any]) -> TNDArrayAny:
        raise NotImplementedError() #pragma: no cover

    def roll(self: I, shift: int) -> I:
        raise NotImplementedError() #pragma: no cover

    def fillna(self: I, value: tp.Any) -> I:
        raise NotImplementedError() #pragma: no cover

    def _sample_and_key(self: I,
            count: int = 1,
            *,
            seed: tp.Optional[int] = None,
            ) -> tp.Tuple[I, TNDArrayAny]:
        raise NotImplementedError() #pragma: no cover

    def level_add(self,
            level: TLabel,
            *,
            index_constructor: TIndexCtorSpecifier = None,
            ) -> 'IndexHierarchy':
        raise NotImplementedError() #pragma: no cover

    def display(self,
            config: tp.Optional[DisplayConfig] = None,
            *,
            style_config: tp.Optional[StyleConfig] = None,
            ) -> Display:
        raise NotImplementedError()

    # ufunc shape skipna methods -----------------------------------------------

    def _ufunc_shape_skipna(self, *,
            axis: int,
            skipna: bool,
            ufunc: TUFunc,
            ufunc_skipna: TUFunc,
            composable: bool,
            dtypes: tp.Tuple[TDtypeAny, ...],
            size_one_unity: bool
            ) -> TNDArrayAny:
        # not sure if these make sense on TypeBlocks, as they reduce dimensionality
        raise NotImplementedError() #pragma: no cover

    @doc_inject(selector='ufunc_skipna')
    def cumsum(self,
            axis: int = 0,
            skipna: bool = True,
            ) -> TNDArrayAny:
        '''Return the cumulative sum over the specified axis.

        {args}
        '''
        return self._ufunc_shape_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=np.cumsum,
                ufunc_skipna=np.nancumsum,
                composable=False,
                dtypes=(),
                size_one_unity=True
                )

    @doc_inject(selector='ufunc_skipna')
    def cumprod(self,
            axis: int = 0,
            skipna: bool = True,
            ) -> TNDArrayAny:
        '''Return the cumulative product over the specified axis.

        {args}
        '''
        return self._ufunc_shape_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=np.cumprod,
                ufunc_skipna=np.nancumprod,
                composable=False,
                dtypes=(),
                size_one_unity=True
                )


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

    def _loc_to_iloc(self,
            key: TLocSelector,
            key_transform: TKeyTransform = None,
            partial_selection: bool = False,
            ) -> TILocSelector:
        raise NotImplementedError() #pragma: no cover

    @tp.overload
    def loc_to_iloc(self, key: TLabel) -> TILocSelectorOne: ...

    @tp.overload
    def loc_to_iloc(self, key: TLocSelectorMany) -> TILocSelectorMany: ...

    def loc_to_iloc(self,
            key: TLocSelector,
            ) -> TILocSelector:
        raise NotImplementedError() #pragma: no cover

    @tp.overload
    def __getitem__(self, key: TILocSelectorOne) -> tp.Any: ...

    @tp.overload
    def __getitem__(self, key: TILocSelectorMany) -> tp.Self: ...

    def __getitem__(self: I,
            key: TILocSelector
            ) -> tp.Any:
        raise NotImplementedError() #pragma: no cover

    #---------------------------------------------------------------------------
    # name interface

    @property
    @doc_inject()
    def name(self) -> TName:
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
                # name is a tuple of length equal to depth
                for n in name: # type: ignore
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
            others: tp.Iterable[tp.Union['IndexBase', tp.Iterable[TLabel]]],
            many_to_one_type: ManyToOneType,
            ) -> I:
        '''Normalize inputs and call `index_many_to_one`.
        '''

        if self._recache:
            self._update_array_cache()

        imtoaf = partial(imto_adapter_factory,
                depth=self.depth,
                name=self.name,
                ndim=self.ndim,
                )

        indices: tp.Iterable[tp.Union[IndexBase, IMTOAdapter]]

        if hasattr(others, '__len__') and len(others) == 1: # type: ignore
            # NOTE: having only one `other` is far more common than many others; thus, optimize for that case by not using an iterator
            indices = (self, imtoaf(others[0])) # type: ignore
        else:
            indices = chain((self,), (imtoaf(other) for other in others))

        return index_many_to_one( # type: ignore
                indices,
                cls_default=self.__class__,
                many_to_one_type=many_to_one_type,
                )

    def intersection(self: I, *others: tp.Union['IndexBase', tp.Iterable[TLabel]]) -> I:
        '''
        Perform intersection with one or many Index, container, or NumPy array. Identical comparisons retain order.
        '''
        return self._ufunc_set(others, ManyToOneType.INTERSECT)

    def union(self: I, *others: tp.Union['IndexBase', tp.Iterable[TLabel]]) -> I:
        '''
        Perform union with another Index, container, or NumPy array. Identical comparisons retain order.
        '''
        return self._ufunc_set(others, ManyToOneType.UNION)

    def difference(self: I, *others: tp.Union['IndexBase', tp.Iterable[TLabel]]) -> I:
        '''
        Perform difference with another Index, container, or NumPy array. Retains order.
        '''
        return self._ufunc_set(others, ManyToOneType.DIFFERENCE)


    #---------------------------------------------------------------------------
    # na handling

    def isna(self) -> TNDArrayBool:
        '''
        Return a same-shaped, Boolean :obj:`ndarray` indicating which values are NaN or None.
        '''
        array = isna_array(self.values)
        array.flags.writeable = False
        return array

    def notna(self) -> TNDArrayBool:
        '''
        Return a same-shaped, Boolean :obj:`ndarray` indicating which values are NaN or None.
        '''
        array = np.logical_not(isna_array(self.values))
        array.flags.writeable = False
        return array

    #---------------------------------------------------------------------------
    # falsy handling

    def isfalsy(self) -> TNDArrayBool:
        '''
        Return a same-shaped, Boolean :obj:`ndarray` indicating which values are falsy.
        '''
        array = isfalsy_array(self.values)
        array.flags.writeable = False
        return array

    def notfalsy(self) -> TNDArrayBool:
        '''
        Return a same-shaped, Boolean :obj:`ndarray` indicating which values are falsy.
        '''
        array = np.logical_not(isfalsy_array(self.values))
        array.flags.writeable = False
        return array

    #---------------------------------------------------------------------------
    # via interfaces

    @property
    def via_str(self) -> InterfaceString[TNDArrayAny]:
        raise NotImplementedError() #pragma: no cover

    @property
    def via_dt(self) -> InterfaceDatetime[TNDArrayAny]:
        raise NotImplementedError() #pragma: no cover

    def via_re(self,
            pattern: str,
            flags: int = 0,
            ) -> InterfaceRe[TNDArrayAny]:
        raise NotImplementedError() #pragma: no cover

    #---------------------------------------------------------------------------
    # exporters

    @doc_inject(class_name='Index')
    def to_html(self,
            config: tp.Optional[DisplayConfig] = None,
            style_config: tp.Optional[StyleConfig] = STYLE_CONFIG_DEFAULT,
            ) -> str:
        '''
        {}
        '''
        config = config or DisplayActive.get(type_show=False)
        config = config.to_display_config(
                display_format=DisplayFormats.HTML_TABLE,
                )

        style_config = style_config_css_factory(style_config, self)
        return repr(self.display(config, style_config=style_config))

    @doc_inject(class_name='Index')
    def to_html_datatables(self,
            fp: tp.Optional[TPathSpecifierOrTextIO] = None,
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
        fp_post: tp.Optional[str] = write_optional_file(content=content, fp=fp)

        if fp_post is not None and show:
            import webbrowser  # pragma: no cover
            webbrowser.open_new_tab(fp_post) #pragma: no cover

        return fp_post

    def to_pandas(self) -> 'pandas.Series':
        raise NotImplementedError() #pragma: no cover

    def _to_signature_bytes(self,
            include_name: bool = True,
            include_class: bool = True,
            encoding: str = 'utf-8',
            ) -> bytes:
        raise NotImplementedError() #pragma: no cover

