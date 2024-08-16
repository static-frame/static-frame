from __future__ import annotations

from functools import partial

import numpy as np
import typing_extensions as tp

from static_frame.core.display import Display
from static_frame.core.display import DisplayActive
from static_frame.core.display_config import DisplayConfig
from static_frame.core.display_config import DisplayFormats
from static_frame.core.doc_str import doc_inject
from static_frame.core.exception import ErrorNotTruthy
from static_frame.core.interface_meta import InterfaceMeta
from static_frame.core.memory_measure import MemoryDisplay
from static_frame.core.node_fill_value import InterfaceBatchFillValue
from static_frame.core.node_hashlib import InterfaceHashlib
from static_frame.core.node_transpose import InterfaceBatchTranspose
from static_frame.core.style_config import StyleConfig
from static_frame.core.util import DTYPE_FLOAT_DEFAULT
from static_frame.core.util import DTYPES_BOOL
from static_frame.core.util import DTYPES_INEXACT
from static_frame.core.util import INT64_MAX
from static_frame.core.util import OPERATORS
from static_frame.core.util import UFUNC_TO_REVERSE_OPERATOR
from static_frame.core.util import TDtypeSpecifier
from static_frame.core.util import TName
from static_frame.core.util import TUFunc
from static_frame.core.util import ufunc_all
from static_frame.core.util import ufunc_any
from static_frame.core.util import ufunc_nanall
from static_frame.core.util import ufunc_nanany
from static_frame.core.util import ufunc_nanprod
from static_frame.core.util import ufunc_nansum

if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame  # pragma: no cover
    from static_frame.core.type_clinic import TypeClinic  # pragma: no cover
    TNDArrayAny = np.ndarray[tp.Any, tp.Any] #pragma: no cover
    TDtypeAny = np.dtype[tp.Any] #pragma: no cover
    TFrameAny = Frame[tp.Any, tp.Any, tp.Unpack[tp.Tuple[tp.Any, ...]]] #pragma: no cover

T = tp.TypeVar('T')

#-------------------------------------------------------------------------------
class ContainerBase(metaclass=InterfaceMeta):
    '''
    Root of all containers. The core containers, like Series, Frame, and Index, inherit from ContainerOperand. The higher-order containers, like Bus, Quilt, Batch, and Yarn, inherit from ContainerBase.
    '''
    __slots__ = ()

    #---------------------------------------------------------------------------
    # class attrs

    STATIC: bool = True

    #---------------------------------------------------------------------------
    # common display functions

    @property
    @doc_inject()
    def interface(self) -> TFrameAny:
        '''{}'''
        from static_frame.core.interface import InterfaceSummary
        return InterfaceSummary.to_frame(self.__class__)

    @property
    def via_type_clinic(self) -> TypeClinic:
        from static_frame.core.type_clinic import TypeClinic
        return TypeClinic(self)


    # def __sizeof__(self) -> int:
        # NOTE: implementing this to use memory_total is difficult, as we cannot pass in self without an infinite loop; trying to leave out self but keep its components returns a slightly different result as we miss the "native" (shallow) __sizeof__ components (and possible GC components as well).
        # return memory_total(self, format=MeasureFormat.REFERENCED)

    @property
    def name(self) -> TName:
        return None

    def _memory_label_component_pairs(self,
            ) -> tp.Iterable[tp.Tuple[str, tp.Any]]:
        return ()

    @property
    def memory(self) -> MemoryDisplay:
        '''Return a :obj:`MemoryDisplay`, providing the size in memory of this object. For compound containers, component sizes will also be provided. Size can be interpreted through six combinations of three configurations:

        L: Local: memory ignoring referenced array data provided via views.
        LM: Local Materialized: memory where arrays that are locally owned report their byte payload
        LMD: Local Materialized Data: locally owned memory of arrays byte payloads, excluding all other components

        R: Referenced: memory including referenced array data provided via views
        RM: Referenced Materialized: memory where arrays that are locally owned or referenced report their byte payload
        RMD: Referenced Materialized Data: localy owned and referenced array byte payloads, excluding all other components
        '''
        label_component_pairs = self._memory_label_component_pairs()
        return MemoryDisplay.from_any(self,
                label_component_pairs=label_component_pairs,
                )

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
        Raises ValueError to prohibit ambiguous use of truthy evaluation.
        '''
        raise ErrorNotTruthy()

    def __lt__(self, other: tp.Any) -> tp.Any:
        return NotImplemented #pragma: no cover

    def __le__(self, other: tp.Any) -> tp.Any:
        return NotImplemented #pragma: no cover

    def __eq__(self, other: tp.Any) -> tp.Any:
        return NotImplemented #pragma: no cover

    def __ne__(self, other: tp.Any) -> tp.Any:
        return NotImplemented #pragma: no cover

    def __gt__(self, other: tp.Any) -> tp.Any:
        return NotImplemented #pragma: no cover

    def __ge__(self, other: tp.Any) -> tp.Any:
        return NotImplemented #pragma: no cover

    #---------------------------------------------------------------------------

    def _to_signature_bytes(self,
            include_name: bool = True,
            include_class: bool = True,
            encoding: str = 'utf-8',
            ) -> bytes:
        raise NotImplementedError() #pragma: no cover

    @property
    def via_hashlib(self) -> InterfaceHashlib:
        '''
        Interface for deriving cryptographic hashes from this container.
        '''
        return InterfaceHashlib(
                to_bytes=self._to_signature_bytes,
                include_name=True,
                include_class=True,
                encoding='utf-8',
                )

    def to_visidata(self) -> None:
        '''Open an interactive VisiData session.
        '''
        from static_frame.core.display_visidata import view_sf  # pragma: no cover
        view_sf(self) # type: ignore  #pragma: no cover




class ContainerOperandSequence(ContainerBase):
    '''Base class of all sequence-like containers that support operators but tend to decay to NumPy array, not specialized container subclasses. IndexBase inherits from this class.'''

    __slots__ = ()

    interface: TFrameAny # property that returns a Frame
    # values: TNDArrayAny

    # NOTE: the return type here is intentionally broad as it will get specialized in derived classes
    def _ufunc_binary_operator(self, *,
            operator: TUFunc,
            other: tp.Any,
            fill_value: object = np.nan,
            ) -> tp.Any:
        raise NotImplementedError() #pragma: no cover

    @property
    def values(self) -> TNDArrayAny:
        raise NotImplementedError() #pragma: no cover

    #---------------------------------------------------------------------------
    def __add__(self, other: tp.Any) -> tp.Any:
        if other.__class__ is InterfaceBatchFillValue or other.__class__ is InterfaceBatchTranspose:
            return NotImplemented
        return self._ufunc_binary_operator(operator=OPERATORS['__add__'], other=other)

    def __sub__(self, other: tp.Any) -> tp.Any:
        if other.__class__ is InterfaceBatchFillValue or other.__class__ is InterfaceBatchTranspose:
            return NotImplemented
        return self._ufunc_binary_operator(operator=OPERATORS['__sub__'], other=other)

    def __mul__(self, other: tp.Any) -> tp.Any:
        if other.__class__ is InterfaceBatchFillValue or other.__class__ is InterfaceBatchTranspose:
            return NotImplemented
        return self._ufunc_binary_operator(operator=OPERATORS['__mul__'], other=other)

    def __matmul__(self, other: tp.Any) -> tp.Any:
        return self._ufunc_binary_operator(operator=OPERATORS['__matmul__'], other=other)

    def __truediv__(self, other: tp.Any) -> tp.Any:
        if other.__class__ is InterfaceBatchFillValue or other.__class__ is InterfaceBatchTranspose:
            return NotImplemented
        return self._ufunc_binary_operator(operator=OPERATORS['__truediv__'], other=other)

    def __floordiv__(self, other: tp.Any) -> tp.Any:
        if other.__class__ is InterfaceBatchFillValue or other.__class__ is InterfaceBatchTranspose:
            return NotImplemented
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
    def __array__(self, dtype: TDtypeSpecifier = None) -> TNDArrayAny:
        '''
        Support the __array__ interface, returning an array of values.
        '''
        if dtype is None:
            return self.values
        array: TNDArrayAny = self.values.astype(dtype)
        return array

    def __array_ufunc__(self,
            ufunc: TUFunc,
            method: str,
            *args: tp.Any,
            **kwargs: tp.Any,
            ) -> tp.Any:
        '''Support for NumPy elements or arrays on the left hand of binary operators.
        '''
        if len(args) == 2 and args[1] is self and method == '__call__':
            # self is right-hand side of binary operator with NumPy object
            return self._ufunc_binary_operator(
                    operator=UFUNC_TO_REVERSE_OPERATOR[ufunc],
                    other=args[0],
                    )
        return NotImplemented  #pragma: no cover

    # NOTE: this method will support aribitrary np functions; we choosen not to support these as not all functions make sense for SF containers
    # def __array_function__(self, func, types, args, kwargs):
    #     raise NotImplementedError(f'no support for {func}')

    # --------------------------------------------------------------------------
    # ufunc axis skipna methods: applied along an axis, reducing dimensionality.
    def _ufunc_axis_skipna(self, *,
            axis: int,
            skipna: bool,
            ufunc: TUFunc,
            ufunc_skipna: TUFunc,
            composable: bool,
            dtypes: tp.Tuple[TDtypeAny, ...],
            size_one_unity: bool
            ) -> tp.Any: # usually a Series
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
            out: tp.Optional[TNDArrayAny] = None,
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
            out: tp.Optional[TNDArrayAny] = None,
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
            allna: int = 0,
            out: tp.Optional[TNDArrayAny] = None,
            ) -> tp.Any:
        '''Sum values along the specified axis.

        {args}
        '''
        return self._ufunc_axis_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=np.sum,
                ufunc_skipna=partial(ufunc_nansum, allna=allna),
                # ufunc_skipna=np.nansum,
                composable=False,
                dtypes=(), # float or int, row type will match except Boolean
                size_one_unity=True,
                )

    @doc_inject(selector='ufunc_skipna')
    def min(self,
            axis: int = 0,
            skipna: bool = True,
            out: tp.Optional[TNDArrayAny] = None,
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
                dtypes=(),
                size_one_unity=True
                )

    @doc_inject(selector='ufunc_skipna')
    def max(self,
            axis: int = 0,
            skipna: bool = True,
            out: tp.Optional[TNDArrayAny] = None,
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
                dtypes=(),
                size_one_unity=True
                )

    @doc_inject(selector='ufunc_skipna')
    def mean(self,
            axis: int = 0,
            skipna: bool = True,
            out: tp.Optional[TNDArrayAny] = None,
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
            out: tp.Optional[TNDArrayAny] = None,
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
            out: tp.Optional[TNDArrayAny] = None,
            ) -> tp.Any:
        '''Return the standard deviaton along the specified axis.

        {args}
        '''
        return self._ufunc_axis_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=partial(np.std, ddof=ddof), #type: ignore
                ufunc_skipna=partial(np.nanstd, ddof=ddof), #type: ignore
                composable=False,
                dtypes=(DTYPE_FLOAT_DEFAULT,), # Ufuncs only return real result.
                size_one_unity=False
                )

    @doc_inject(selector='ufunc_skipna')
    def var(self,
            axis: int = 0,
            skipna: bool = True,
            ddof: int = 0,
            out: tp.Optional[TNDArrayAny] = None,
            ) -> tp.Any:
        '''Return the variance along the specified axis.

        {args}
        '''
        return self._ufunc_axis_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=partial(np.var, ddof=ddof), #type: ignore
                ufunc_skipna=partial(np.nanvar, ddof=ddof), #type: ignore
                composable=False,
                dtypes=(DTYPE_FLOAT_DEFAULT,), # Ufuncs only return real result.
                size_one_unity=False
                )

    @doc_inject(selector='ufunc_skipna')
    def prod(self,
            axis: int = 0,
            skipna: bool = True,
            allna: int = 1,
            out: tp.Optional[TNDArrayAny] = None,
            ) -> tp.Any:
        '''Return the product along the specified axis.

        {args}
        '''
        return self._ufunc_axis_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=np.prod,
                ufunc_skipna=partial(ufunc_nanprod, allna=allna),
                composable=False, # Block combinations with overflow and NaNs require this.
                dtypes=(),
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
                display_columns=INT64_MAX,
                display_rows=INT64_MAX,
                )
        # modify the active display to be for HTML
        return repr(self.display(config))




class ContainerOperand(ContainerOperandSequence):
    '''Base class of all mapping-like containers that support operators. Series, TypeBlocks, and Frame inherit from this class. These containers preserve the type in unary and binary operations.'''

    __slots__ = ()

    #---------------------------------------------------------------------------
    def __pos__(self) -> tp.Self:
        return self._ufunc_unary_operator(OPERATORS['__pos__'])

    def __neg__(self) -> tp.Self:
        return self._ufunc_unary_operator(OPERATORS['__neg__'])

    def __abs__(self) -> tp.Self:
        return self._ufunc_unary_operator(OPERATORS['__abs__'])

    def __invert__(self) -> tp.Self:
        return self._ufunc_unary_operator(OPERATORS['__invert__'])

    #---------------------------------------------------------------------------

    def _ufunc_unary_operator(self: T, operator: TUFunc) -> T:
        raise NotImplementedError() #pragma: no cover

    def _ufunc_binary_operator(self: T, *,
            operator: TUFunc,
            other: tp.Any,
            fill_value: object = np.nan,
            ) -> T:
        raise NotImplementedError() #pragma: no cover

    # ufunc shape skipna methods -----------------------------------------------

    def _ufunc_shape_skipna(self, *,
            axis: int,
            skipna: bool,
            ufunc: TUFunc,
            ufunc_skipna: TUFunc,
            composable: bool,
            dtypes: tp.Tuple[TDtypeAny, ...],
            size_one_unity: bool
            ) -> tp.Any:
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
                dtypes=(),
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
                dtypes=(),
                size_one_unity=True
                )
