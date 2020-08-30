import typing as tp
from itertools import chain
from itertools import product
from functools import wraps
import operator as operator_mod

import numpy as np

from static_frame.core.display import Display
from static_frame.core.display import DisplayActive
from static_frame.core.display_config import DisplayConfig
from static_frame.core.display_config import DisplayFormats
from static_frame.core.doc_str import doc_inject
from static_frame.core.util import AnyCallable
from static_frame.core.util import DTYPE_FLOAT_DEFAULT
from static_frame.core.util import DTYPE_INT_KINDS
from static_frame.core.util import DTYPE_INEXACT_KINDS
from static_frame.core.util import DTYPE_NAT_KINDS
from static_frame.core.util import DTYPE_STR_KINDS
from static_frame.core.util import DTYPES_BOOL
from static_frame.core.util import DTYPES_INEXACT
from static_frame.core.util import EMPTY_TUPLE
from static_frame.core.util import isna_array
from static_frame.core.util import UFunc
from static_frame.core.interface_meta import InterfaceMeta


if tp.TYPE_CHECKING:
    from static_frame.core.frame import Frame #pylint: disable=W0611 #pragma: no cover

T = tp.TypeVar('T')


_UFUNC_UNARY_OPERATORS = (
        '__pos__',
        '__neg__',
        '__abs__',
        '__invert__')

_UFUNC_BINARY_OPERATORS = (
        '__add__',
        '__sub__',
        '__mul__',
        '__matmul__',
        '__truediv__',
        '__floordiv__',
        '__mod__',
        #'__divmod__', this returns two np.arrays when called on an np array
        '__pow__',
        '__lshift__',
        '__rshift__',
        '__and__',
        '__xor__',
        '__or__',
        '__lt__',
        '__le__',
        '__eq__',
        '__ne__',
        '__gt__',
        '__ge__',
        )

_UFUNC_OPERATORS_MAP = {k: getattr(operator_mod, k)
        for k in chain(_UFUNC_UNARY_OPERATORS, _UFUNC_BINARY_OPERATORS)
        }

# all right are binary
_RIGHT_OPERATOR_MAP = {
        '__radd__': '__add__',
        '__rsub__': '__sub__',
        '__rmul__': '__mul__',
        '__rmatmul__': '__matmul__',
        '__rtruediv__': '__truediv__',
        '__rfloordiv__': '__floordiv__',
        }


#-------------------------------------------------------------------------------
def _ufunc_logical_skipna(
        array: np.ndarray,
        ufunc: AnyCallable,
        skipna: bool,
        axis: int = 0,
        out: tp.Optional[np.ndarray] = None
        ) -> np.ndarray:
    '''
    Given a logical (and, or) ufunc that does not support skipna, implement skipna behavior.
    '''
    if ufunc != np.all and ufunc != np.any:
        raise NotImplementedError(f'unsupported ufunc ({ufunc}); use np.all or np.any')

    if len(array) == 0:
        # TODO: handle if this is ndim == 2 and has no length
        # any() of an empty array is False
        return ufunc == np.all

    kind = array.dtype.kind

    #---------------------------------------------------------------------------
    # types that cannot have NA
    if kind == 'b':
        return ufunc(array, axis=axis, out=out)
    if kind in DTYPE_INT_KINDS:
        return ufunc(array, axis=axis, out=out)
    if kind in DTYPE_STR_KINDS:
        # only string in object arrays can be converted to bool, where the empty string will be evaluated as False; here, manually check
        return ufunc(array != '', axis=axis, out=out)

    #---------------------------------------------------------------------------
    # types that can have NA

    if kind in DTYPE_INEXACT_KINDS:
        isna = isna_array(array)
        hasna = isna.any() # returns single value for 1d, 2d
        if hasna and skipna:
            fill_value = 0.0 if ufunc == np.any else 1.0
            v = array.copy()
            v[isna] = fill_value
            return ufunc(v, axis=axis, out=out)
        elif hasna and not skipna:
            # if array.ndim == 1:
            #     return np.nan
            raise TypeError('cannot propagate NaN without expanding to object array result')
        return ufunc(array, axis=axis, out=out)

    if kind in DTYPE_NAT_KINDS:
        isna = isna_array(array)
        hasna = isna.any() # returns single value for 1d, 2d
        # all dates are truthy, special handling only to propagate NaNs
        if hasna and not skipna:
            # if array.ndim == 1:
            #     return NAT
            raise TypeError('cannot propagate NaN without expanding to object array result')
        # to ignore NaN, simply fall back on all-truth behavior, below

    if kind == 'O':
        # all object types: convert to boolean aray then process
        isna = isna_array(array)
        hasna = isna.any() # returns single value for 1d, 2d
        if hasna and skipna:
            # supply True for np.all, False for np.any
            fill_value = False if ufunc == np.any else True
            v = array.copy()
            v = v.astype(bool) # nan will be converted to True
            v[isna] = fill_value
        elif hasna and not skipna:
            # if array.ndim == 1:
            #     return np.nan
            raise TypeError('cannot propagate NaN without expanding to object array result')
        else:
            v = array.astype(bool)
        return ufunc(v, axis=axis, out=out)

    # all types other than strings or objects assume truthy
    if array.ndim == 1:
        return True
    return np.full(array.shape[0 if axis else 1], fill_value=True, dtype=bool)


def _all(array: np.ndarray,
        axis: int = 0,
        out: tp.Optional[np.ndarray] = None
        ) -> np.ndarray:
    return _ufunc_logical_skipna(array,
            ufunc=np.all,
            skipna=False,
            axis=axis,
            out=out)

_all.__doc__ = np.all.__doc__

def _any(array: np.ndarray,
        axis: int = 0,
        out: tp.Optional[np.ndarray] = None
        ) -> np.ndarray:
    return _ufunc_logical_skipna(array,
            ufunc=np.any,
            skipna=False,
            axis=axis,
            out=out)

_any.__doc__ = np.any.__doc__

def _nanall(array: np.ndarray,
        axis: int = 0,
        out: tp.Optional[np.ndarray] = None
        ) -> np.ndarray:
    return _ufunc_logical_skipna(array,
            ufunc=np.all,
            skipna=True,
            axis=axis,
            out=out)

def _nanany(array: np.ndarray,
        axis: int = 0,
        out: tp.Optional[np.ndarray] = None
        ) -> np.ndarray:
    return _ufunc_logical_skipna(array,
            ufunc=np.any,
            skipna=True,
            axis=axis,
            out=out)

#-------------------------------------------------------------------------------

class ContainerOperandMeta(InterfaceMeta):
    '''Auto-populate binary and unary methods based on instance methods named `_ufunc_unary_operator` and `_ufunc_binary_operator`.
    '''

    @staticmethod
    def create_ufunc_operator(
            func_name: str,
            opperand_count: int = 1,
            reverse: bool = False,
            ) -> tp.Union[tp.Callable[[tp.Any], tp.Any], tp.Callable[[tp.Any, tp.Any], tp.Any]]:
        '''
        Given a func_name, derive the method to live on the Container.
        '''
        # operator module defines alias to funcs with names like __add__, etc
        if not reverse:
            operator_func = getattr(operator_mod, func_name)
            func_wrapper = operator_func
        else:
            unreversed_operator_func = getattr(
                    operator_mod,
                    _RIGHT_OPERATOR_MAP[func_name])
            # flip the order of the arguments
            operator_func = lambda rhs, lhs: unreversed_operator_func(lhs, rhs)
            # construct a __name__ that will look the name we get from for the unreversed operator; these are names without the leading and trailing dunders, like "matmul", we we just add an r for reverse.
            operator_func.__name__ = 'r' + unreversed_operator_func.__name__
            func_wrapper = unreversed_operator_func

        func: tp.Union[tp.Callable[[tp.Any], tp.Any],
                tp.Callable[[tp.Any, tp.Any], tp.Any]]

        if opperand_count == 1:
            assert not reverse # cannot reverse a single operand
            def func(self: tp.Any) -> tp.Any: #pylint: disable=E0102
                return self._ufunc_unary_operator(operator_func)
        elif opperand_count == 2:
            def func(self: tp.Any, other: tp.Any) -> tp.Any: #pylint: disable=E0102
                return self._ufunc_binary_operator(operator=operator_func, other=other)
        else:
            raise NotImplementedError() #pragma: no cover

        f = wraps(func_wrapper)(func)
        f.__name__ = func_name
        return f

    def __new__(mcs, #type: ignore
            name: str,
            bases: tp.Tuple[type, ...],
            attrs: tp.Dict[str, object
            ]) -> type: #must return a subtype of "ContainerOperandMeta"
        '''
        Create and assign all autopopulated functions. This __new__ is on the metaclass, not the class, and is thus only called once per class.
        '''
        for opperand_count, func_name in chain(
                product((1,), _UFUNC_UNARY_OPERATORS),
                product((2,), _UFUNC_BINARY_OPERATORS)):

            attrs[func_name] = mcs.create_ufunc_operator(
                    func_name,
                    opperand_count=opperand_count)

        for func_name in _RIGHT_OPERATOR_MAP:
            attrs[func_name] = mcs.create_ufunc_operator(
                    func_name,
                    opperand_count=2,
                    reverse=True)

        return type.__new__(mcs, name, bases, attrs)


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
            config: tp.Optional[DisplayConfig] = None
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




class ContainerOperand(ContainerBase, metaclass=ContainerOperandMeta):
    '''Base class of all containers that support opperators.'''

    __slots__ = EMPTY_TUPLE

    interface: 'Frame' # property that returns a Frame
    values: np.ndarray

    __pos__: tp.Callable[[T], T]
    __neg__: tp.Callable[[T], T]
    __abs__: tp.Callable[[T], T]
    __invert__: tp.Callable[[T], T]
    __add__: tp.Callable[[T, object], T]
    __sub__: tp.Callable[[T, object], T]
    __mul__: tp.Callable[[T, object], T]
    __matmul__: tp.Callable[[T, object], T]
    __truediv__: tp.Callable[[T, object], T]
    __floordiv__: tp.Callable[[T, object], T]
    __mod__: tp.Callable[[T, object], T]
    # __divmod__: tp.Callable[[T, object], T]
    __pow__: tp.Callable[[T, object], T]
    __lshift__: tp.Callable[[T, object], T]
    __rshift__: tp.Callable[[T, object], T]
    __and__: tp.Callable[[T, object], T]
    __xor__: tp.Callable[[T, object], T]
    __or__: tp.Callable[[T, object], T]
    __lt__: tp.Callable[[T, object], T]
    __le__: tp.Callable[[T, object], T]
    __eq__: tp.Callable[[T, object], T]  #type: ignore
    __ne__: tp.Callable[[T, object], T]  #type: ignore
    __gt__: tp.Callable[[T, object], T]
    __ge__: tp.Callable[[T, object], T]
    __radd__: tp.Callable[[T, object], T]
    __rsub__: tp.Callable[[T, object], T]
    __rmul__: tp.Callable[[T, object], T]
    __rtruediv__: tp.Callable[[T, object], T]
    __rfloordiv__: tp.Callable[[T, object], T]


    # methods are overwritten by metaclass, but defined here for typing

    # ufunc axis skipna methods ------------------------------------------------
    # ufuncs that are applied along an axis, reducing dimensionality. NOTE: as argmin and argmax have iloc/loc interetaions, they are implemented on derived containers
    # dtypes: iterable of valid dtypes that can be returned; first is default of not match
    # composable: if partial solutions can be processed per block for axis 1 computations
    # size_one_unity: if the result of the operation on size 1 objects is that value

    def _ufunc_axis_skipna(self, *,
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
    def all(self,
            axis: int = 0,
            skipna: bool = True,
            out: tp.Optional[np.ndarray] = None,
            ) -> tp.Any:
        '''Logical and over values along the specified axis.

        {args}
        '''
        return self._ufunc_axis_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=_all,
                ufunc_skipna=_nanall,
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
        '''Logical or over values along the specified axis.

        {args}
        '''
        return self._ufunc_axis_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=_any,
                ufunc_skipna=_nanany,
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
            out: tp.Optional[np.ndarray] = None,
            ) -> tp.Any:
        '''Return the standard deviaton along the specified axis.

        {args}
        '''
        return self._ufunc_axis_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=np.std,
                ufunc_skipna=np.nanstd,
                composable=False,
                dtypes=(DTYPE_FLOAT_DEFAULT,), # Ufuncs only return real result.
                size_one_unity=False
                )

    @doc_inject(selector='ufunc_skipna')
    def var(self,
            axis: int = 0,
            skipna: bool = True,
            out: tp.Optional[np.ndarray] = None,
            ) -> tp.Any:
        '''Return the variance along the specified axis.

        {args}
        '''
        return self._ufunc_axis_skipna(
                axis=axis,
                skipna=skipna,
                ufunc=np.var,
                ufunc_skipna=np.nanvar,
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

