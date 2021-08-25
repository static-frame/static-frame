import typing as tp
from functools import partial

import numpy as np

from static_frame.core.display import Display
from static_frame.core.display import DisplayActive
from static_frame.core.display_config import DisplayConfig
from static_frame.core.display_config import DisplayFormats
from static_frame.core.doc_str import doc_inject
from static_frame.core.interface_meta import InterfaceMeta
from static_frame.core.util import DTYPE_FLOAT_DEFAULT
from static_frame.core.util import DTYPES_BOOL
from static_frame.core.util import DTYPES_INEXACT
from static_frame.core.util import EMPTY_TUPLE
from static_frame.core.util import UFunc
from static_frame.core.util import ufunc_all
from static_frame.core.util import ufunc_any
from static_frame.core.util import ufunc_nanall
from static_frame.core.util import ufunc_nanany
from static_frame.core.util import OPERATORS
from static_frame.core.style_config import StyleConfig

if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame #pylint: disable=W0611 #pragma: no cover

T = tp.TypeVar('T')

#-------------------------------------------------------------------------------
class ContainerBase(metaclass=InterfaceMeta):
    '''
    Root of all containers. Most containers, like Series, Frame, and Index, inherit from ContainerOperand; only Bus inherits from ContainerBase.
    '''
    __slots__ = EMPTY_TUPLE

    #---------------------------------------------------------------------------
    # class attrs

    STATIC: bool = True

    #---------------------------------------------------------------------------
    # common display functions

    @property #type: ignore
    @doc_inject()
    def interface(self) -> 'Frame':
        '''{}'''
        from static_frame.core.interface import InterfaceSummary
        return InterfaceSummary.to_frame(self.__class__)

    def display(self,
            config: tp.Optional[DisplayConfig] = None,
            *,
            style_config: tp.Optional[StyleConfig] = None,
            ) -> Display:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return repr(self.display())

    @doc_inject(selector='display')
    def display_tall(self,
            config: tp.Optional[DisplayConfig] = None
            ) -> Display:
        '''Maximize vertical presentation. {doc}

        Args:
            {config}
        '''
        config = config or DisplayActive.get()
        args = config.to_dict()
        args.update(dict(
                display_rows=np.inf,
                cell_max_width=np.inf,
                cell_max_width_leftmost=np.inf,
                ))
        return self.display(config=DisplayConfig(**args))

    @doc_inject(selector='display')
    def display_wide(self,
            config: tp.Optional[DisplayConfig] = None
            ) -> Display:
        '''Maximize horizontal presentation. {doc}

        Args:
            {config}
        '''
        config = config or DisplayActive.get()
        args = config.to_dict()
        args.update(dict(
                display_columns=np.inf,
                cell_max_width=np.inf,
                cell_max_width_leftmost=np.inf,
                ))
        return self.display(config=DisplayConfig(**args))

    #---------------------------------------------------------------------------
    def equals(self,
            other: tp.Any,
            *,
            compare_name: bool = False,
            compare_dtype: bool = False,
            compare_class: bool = False,
            skipna: bool = True,
            ) -> bool:
        raise NotImplementedError() #pragma: no cover


    #---------------------------------------------------------------------------
    def __bool__(self) -> bool:
        '''
        Raises ValueError to prohibit ambiguous use of truethy evaluation.
        '''
        raise ValueError('The truth value of a container is ambiguous. For a truthy indicator of non-empty status, use the `size` attribute.')

    #---------------------------------------------------------------------------
    def to_visidata(self) -> None:
        '''Open an interactive VisiData session.
        '''
        from static_frame.core.display_visidata import view_sf #pragma: no cover
        view_sf(self) #type: ignore [no-untyped-call] #pragma: no cover


class ContainerOperand(ContainerBase):
    '''Base class of all containers that support opperators.'''

    __slots__ = EMPTY_TUPLE

    interface: 'Frame' # property that returns a Frame
    values: np.ndarray

    def _ufunc_unary_operator(self: T, operator: UFunc) -> T:
        raise NotImplementedError() #pragma: no cover

    def _ufunc_binary_operator(self: T, *,
            operator: UFunc,
            other: tp.Any,
            fill_value: object = np.nan,
            ) -> T:
        raise NotImplementedError() #pragma: no cover

    #---------------------------------------------------------------------------
    def __pos__(self) -> 'ContainerOperand':
        return self._ufunc_unary_operator(OPERATORS['__pos__'])

    def __neg__(self) -> 'ContainerOperand':
        return self._ufunc_unary_operator(OPERATORS['__neg__'])

    def __abs__(self) -> 'ContainerOperand':
        return self._ufunc_unary_operator(OPERATORS['__abs__'])

    def __invert__(self) -> 'ContainerOperand':
        return self._ufunc_unary_operator(OPERATORS['__invert__'])

    #---------------------------------------------------------------------------
    def __add__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__add__'], other=other)

    def __sub__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__sub__'], other=other)

    def __mul__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__mul__'], other=other)

    def __matmul__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__matmul__'], other=other)

    def __truediv__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__truediv__'], other=other)

    def __floordiv__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__floordiv__'], other=other)

    def __mod__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__mod__'], other=other)

    # def __divmod__:

    def __pow__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__pow__'], other=other)

    def __lshift__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__lshift__'], other=other)

    def __rshift__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__rshift__'], other=other)

    def __and__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__and__'], other=other)

    def __xor__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__xor__'], other=other)

    def __or__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__or__'], other=other)

    def __lt__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__lt__'], other=other)

    def __le__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__le__'], other=other)

    def __eq__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__eq__'], other=other)

    def __ne__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__ne__'], other=other)

    def __gt__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__gt__'], other=other)

    def __ge__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__ge__'], other=other)

    #---------------------------------------------------------------------------
    def __radd__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__radd__'], other=other)

    def __rsub__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__rsub__'], other=other)

    def __rmul__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__rmul__'], other=other)

    def __rmatmul__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__rmatmul__'], other=other)

    def __rtruediv__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__rtruediv__'], other=other)

    def __rfloordiv__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__rfloordiv__'], other=other)

    # --------------------------------------------------------------------------
    # ufunc axis skipna methods: applied along an axis, reducing dimensionality.
    # NOTE: as argmin and argmax have iloc/loc interetaions, they are implemented on derived containers

    def _ufunc_axis_skipna(self, *,
            axis: int,
            skipna: bool,
            ufunc: UFunc,
            ufunc_skipna: UFunc,
            composable: bool,
            dtypes: tp.Tuple[np.dtype, ...],
            size_one_unity: bool
            ) -> np.ndarray:
        '''
        Args:
            dtypes: iterable of valid dtypes that can be returned; first is default of not match
            composable: if partial solutions can be processed per block for axis 1 computations
            size_one_unity: if the result of the operation on size 1 objects is that value
        '''
        # not sure if these make sense on TypeBlocks, as they reduce dimensionality
        raise NotImplementedError() #pragma: no cover


    @doc_inject(selector='ufunc_skipna')
    def all(self,
            axis: int = 0,
            skipna: bool = True,
            out: tp.Optional[np.ndarray] = None,
            ) -> tp.Any:
        '''Logical ``and`` over values along the specified axis.

        {args}
        '''
        return self._ufunc_axis_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=ufunc_all,
                ufunc_skipna=ufunc_nanall,
                composable=True,
                dtypes=DTYPES_BOOL,
                size_one_unity=False
                )

    @doc_inject(selector='ufunc_skipna')
    def any(self,
            axis: int = 0,
            skipna: bool = True,
            out: tp.Optional[np.ndarray] = None,
            ) -> tp.Any:
        '''Logical ``or`` over values along the specified axis.

        {args}
        '''
        return self._ufunc_axis_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=ufunc_any,
                ufunc_skipna=ufunc_nanany,
                composable=True,
                dtypes=DTYPES_BOOL,
                size_one_unity=False # Overflow amongst heterogenous types accross columns
                )

    @doc_inject(selector='ufunc_skipna')
    def sum(self,
            axis: int = 0,
            skipna: bool = True,
            out: tp.Optional[np.ndarray] = None,
            ) -> tp.Any:
        '''Sum values along the specified axis.

        {args}
        '''
        return self._ufunc_axis_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=np.sum,
                ufunc_skipna=np.nansum,
                composable=False,
                dtypes=EMPTY_TUPLE, # float or int, row type will match
                size_one_unity=True
                )

    @doc_inject(selector='ufunc_skipna')
    def min(self,
            axis: int = 0,
            skipna: bool = True,
            out: tp.Optional[np.ndarray] = None,
            ) -> tp.Any:
        '''Return the minimum along the specified axis.

        {args}
        '''
        return self._ufunc_axis_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=np.min,
                ufunc_skipna=np.nanmin,
                composable=True,
                dtypes=EMPTY_TUPLE,
                size_one_unity=True
                )

    @doc_inject(selector='ufunc_skipna')
    def max(self,
            axis: int = 0,
            skipna: bool = True,
            ) -> tp.Any:
        '''Return the maximum along the specified axis.

        {args}
        '''
        return self._ufunc_axis_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=np.max,
                ufunc_skipna=np.nanmax,
                composable=True,
                dtypes=EMPTY_TUPLE,
                size_one_unity=True
                )

    @doc_inject(selector='ufunc_skipna')
    def mean(self,
            axis: int = 0,
            skipna: bool = True,
            out: tp.Optional[np.ndarray] = None,
            ) -> tp.Any:
        '''Return the mean along the specified axis.

        {args}
        '''
        return self._ufunc_axis_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=np.mean,
                ufunc_skipna=np.nanmean,
                composable=False,
                dtypes=DTYPES_INEXACT, # neads to at least be float, but complex if necessary
                size_one_unity=True
                )

    @doc_inject(selector='ufunc_skipna')
    def median(self,
            axis: int = 0,
            skipna: bool = True,
            out: tp.Optional[np.ndarray] = None,
            ) -> tp.Any:
        '''Return the median along the specified axis.

        {args}
        '''
        return self._ufunc_axis_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=np.median,
                ufunc_skipna=np.nanmedian,
                composable=False,
                dtypes=DTYPES_INEXACT,
                size_one_unity=True
                )

    @doc_inject(selector='ufunc_skipna')
    def std(self,
            axis: int = 0,
            skipna: bool = True,
            ddof: int = 0,
            out: tp.Optional[np.ndarray] = None,
            ) -> tp.Any:
        '''Return the standard deviaton along the specified axis.

        {args}
        '''
        return self._ufunc_axis_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=partial(np.std, ddof=ddof),
                ufunc_skipna=partial(np.nanstd, ddof=ddof),
                composable=False,
                dtypes=(DTYPE_FLOAT_DEFAULT,), # Ufuncs only return real result.
                size_one_unity=False
                )

    @doc_inject(selector='ufunc_skipna')
    def var(self,
            axis: int = 0,
            skipna: bool = True,
            ddof: int = 0,
            out: tp.Optional[np.ndarray] = None,
            ) -> tp.Any:
        '''Return the variance along the specified axis.

        {args}
        '''
        return self._ufunc_axis_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=partial(np.var, ddof=ddof),
                ufunc_skipna=partial(np.nanvar, ddof=ddof),
                composable=False,
                dtypes=(DTYPE_FLOAT_DEFAULT,), # Ufuncs only return real result.
                size_one_unity=False
                )

    @doc_inject(selector='ufunc_skipna')
    def prod(self,
            axis: int = 0,
            skipna: bool = True,
            out: tp.Optional[np.ndarray] = None,
            ) -> tp.Any:
        '''Return the product along the specified axis.

        {args}
        '''
        return self._ufunc_axis_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=np.prod,
                ufunc_skipna=np.nanprod,
                composable=False, # Block compbinations with overflow and NaNs require this.
                dtypes=EMPTY_TUPLE,
                size_one_unity=True
                )

    # ufunc shape skipna methods -----------------------------------------------

    def _ufunc_shape_skipna(self, *,
            axis: int,
            skipna: bool,
            ufunc: UFunc,
            ufunc_skipna: UFunc,
            composable: bool,
            dtypes: tp.Tuple[np.dtype, ...],
            size_one_unity: bool
            ) -> np.ndarray:
        # not sure if these make sense on TypeBlocks, as they reduce dimensionality
        raise NotImplementedError() #pragma: no cover

    @doc_inject(selector='ufunc_skipna')
    def cumsum(self,
            axis: int = 0,
            skipna: bool = True,
            ) -> tp.Any:
        '''Return the cumulative sum over the specified axis.

        {args}
        '''
        return self._ufunc_shape_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=np.cumsum,
                ufunc_skipna=np.nancumsum,
                composable=False,
                dtypes=EMPTY_TUPLE,
                size_one_unity=True
                )

    @doc_inject(selector='ufunc_skipna')
    def cumprod(self,
            axis: int = 0,
            skipna: bool = True,
            ) -> tp.Any:
        '''Return the cumulative product over the specified axis.

        {args}
        '''
        return self._ufunc_shape_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=np.cumprod,
                ufunc_skipna=np.nancumprod,
                composable=False,
                dtypes=EMPTY_TUPLE,
                size_one_unity=True
                )

    #---------------------------------------------------------------------------
    def _repr_html_(self) -> str:
        '''
        Provide HTML representation for Jupyter Notebooks.
        '''
        # NOTE: We observe that Jupyter will window big content into scrollable component, so do not limit output and introduce ellipsis.

        config = DisplayActive.get(
                display_format=DisplayFormats.HTML_TABLE,
                type_show=False,
                display_columns=np.inf,
                display_rows=np.inf,
                )
        # modify the active display to be for HTML
        return repr(self.display(config))


