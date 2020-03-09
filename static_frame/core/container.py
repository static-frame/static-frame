import typing as tp

from itertools import chain
from itertools import product
from functools import wraps

import operator as operator_mod
from collections import namedtuple

import numpy as np

from static_frame.core.util import EMPTY_TUPLE
from static_frame.core.util import AnyCallable
from static_frame.core.util import DTYPE_INT_KIND
from static_frame.core.util import DTYPE_STR_KIND
from static_frame.core.util import DTYPE_NAN_KIND
from static_frame.core.util import DTYPE_NAT_KIND
from static_frame.core.util import isna_array

# from static_frame.core.util import DTYPE_BOOL
# from static_frame.core.util import DTYPE_FLOAT_DEFAULT

from static_frame.core.util import DTYPES_BOOL
from static_frame.core.util import DTYPES_INEXACT
from static_frame.core.util import DTYPE_FLOAT_DEFAULT
# from static_frame.core.util import NAT

from static_frame.core.doc_str import DOC_TEMPLATE

from static_frame.core.display import Display
from static_frame.core.display import DisplayConfig
# from static_frame.core.display import DisplayConfigs
from static_frame.core.display import DisplayActive

from static_frame.core.doc_str import doc_inject


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
# NOTE: this was an approach to doing nan propagation when skipna=False; this approach could not use the out argument, which is used in TypeBlocks. Forcing type coercion to object occasionally is hard to reason about, and makes TypeBlock implementation difficult. Thus, we raise a TypeError.

# def _ufunc_logical_withna(
#         array: np.ndarray,
#         isna: np.ndarray,
#         ufunc: AnyCallable,
#         axis: int,
#         element_missing: tp.Any,
#         out: tp.Optional[np.ndarray] = None
#         ) -> np.ndarray:
#     '''
#     Perform a logical (and, or) ufunc on an array that has already been identified as having a NULL (as given in the `isna` array), propagating `element_missing` (NaN or NaT) on an axis if ndim > 1.
#     '''
#     if array.ndim == 1:
#         return element_missing
#     if out is not None:
#         # cann use out if we need to do an astype conversion
#         raise NotImplementedError()
#     # do not need fill values, as will set to nan after eval
#     v = array.astype(bool) # object, datetime64 arrays can be converted to bool
#     v = ufunc(v, axis=axis).astype(object) # get the axis result
#     v[np.any(isna, axis=axis)] = np.nan # propagate NaN
#     return v

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
    if kind in DTYPE_INT_KIND:
        return ufunc(array, axis=axis, out=out)
    if kind in DTYPE_STR_KIND:
        # only string in object arrays can be converted to bool, where the empty string will be evaluated as False; here, manually check
        return ufunc(array != '', axis=axis, out=out)

    #---------------------------------------------------------------------------
    # types that can have NA

    if kind in DTYPE_NAN_KIND:
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

    if kind in DTYPE_NAT_KIND:
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

UfuncSkipnaAttrs = namedtuple('UfuncSkipnaAttrs', (
        'ufunc',
        'ufunc_skipna',
        'dtypes', # iterable of valid dtypes that can be returned; first is default of not match
        'composable', # if partial solutions can be processed per block for axis 1 computations
        'doc_header',
        'size_one_unity', # if the result of the operation on size 1 objects is that value
))

# class UfuncSkipnaAttrs(tp.NamedTuple):
#     ufunc: AnyCallable
#     ufunc_skipna: AnyCallable
#     dtypes: tp.Tuple[np.dtype, ...] # iterable of valid dtypes that can be returned; first is default of not match
#     composable: bool # if partial solutions can be processed per block
#     doc_header: str
#     size_one_unity: bool # if the result of the operation on size 1 objects is that value

# ufuncs that are applied along an axis, reducing dimensionality. NOTE: as argmin and argmax have iloc/loc interetaions, they are implemetned on derived containers
UFUNC_AXIS_SKIPNA: tp.Dict[str, UfuncSkipnaAttrs] = {
        'all': UfuncSkipnaAttrs(
            _all,
            _nanall,
            DTYPES_BOOL,
            True,
            'Logical and over values along the specified axis.',
            False,
            ),
        'any': UfuncSkipnaAttrs(
            _any,
            _nanany,
            DTYPES_BOOL,
            True,
            'Logical or over values along the specified axis.',
            False, # Overflow amongst hetergenoust tyes accross columns
            ),
        'sum': UfuncSkipnaAttrs(
            np.sum,
            np.nansum,
            EMPTY_TUPLE, # float or int, row type will match
            False,
            'Sum values along the specified axis.',
            True,
            ),
        'min': UfuncSkipnaAttrs(
            np.min,
            np.nanmin,
            EMPTY_TUPLE,
            True,
            'Return the minimum along the specified axis.',
            True,
            ),
        'max': UfuncSkipnaAttrs(
            np.max,
            np.nanmax,
            EMPTY_TUPLE,
            True,
            'Return the maximum along the specified axis.',
            True,
            ),
        'mean': UfuncSkipnaAttrs(
            np.mean,
            np.nanmean,
            DTYPES_INEXACT, # neads to at least be float, but complex if necessary
            False,
            'Return the mean along the specified axis.',
            True,
            ),
        'median': UfuncSkipnaAttrs(
            np.median,
            np.nanmedian,
            DTYPES_INEXACT,
            False,
            'Return the median along the specified axis.',
            True,
            ),
        'std': UfuncSkipnaAttrs(
            np.std,
            np.nanstd,
            (DTYPE_FLOAT_DEFAULT,), # Ufuncs only return real result.
            False,
            'Return the standard deviaton along the specified axis.',
            False,
            ),
        'var': UfuncSkipnaAttrs(
            np.var,
            np.nanvar,
            (DTYPE_FLOAT_DEFAULT,), # Ufuncs only return real result.
            False,
            'Return the variance along the specified axis.',
            False,
            ),
        'prod': UfuncSkipnaAttrs(
            np.prod,
            np.nanprod,
            EMPTY_TUPLE, # float or int, row type will match
            False, # Block compbinations with overflow and NaNs require this.
            'Return the product along the specified axis.',
            True,
            ),
        }

# ufuncs that retain the shape and dimensionality
UFUNC_SHAPE_SKIPNA: tp.Dict[str, UfuncSkipnaAttrs] = {
        'cumsum': UfuncSkipnaAttrs(
            np.cumsum,
            np.nancumsum,
            EMPTY_TUPLE,
            False,
            'Return the cumulative sum over the specified axis.',
            True,
            ),
        'cumprod': UfuncSkipnaAttrs(
            np.cumprod,
            np.nancumprod,
            EMPTY_TUPLE,
            False,
            'Return the cumulative product over the specified axis.',
            True,
            ),
        }


#-------------------------------------------------------------------------------
class ContainerMeta(type):
    '''Lowest level metaclass for providing interface property on class.
    '''

    @property #type: ignore
    @doc_inject()
    def interface(cls) -> 'Frame':
        '''{}'''
        from static_frame.core.interface import InterfaceSummary
        return InterfaceSummary.to_frame(cls)


class ContainerOperandMeta(ContainerMeta):
    '''Auto-populate binary and unary methods based on instance methods named `_ufunc_unary_operator` and `_ufunc_binary_operator`.
    '''

    @staticmethod
    def create_ufunc_operator(func_name: str,
            opperand_count: int = 1,
            reverse: bool = False,
            ) -> tp.Union[tp.Callable[[tp.Any], tp.Any], tp.Callable[[tp.Any, tp.Any], tp.Any]]:
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
            assert not reverse # cannot reverse a single opperand
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

    @staticmethod
    def create_ufunc_axis_skipna(func_name: str) -> AnyCallable:
        nt = UFUNC_AXIS_SKIPNA[func_name]
        ufunc = nt.ufunc
        ufunc_skipna = nt.ufunc_skipna
        dtypes = nt.dtypes
        composable = nt.composable
        doc = nt.doc_header
        size_one_unity = nt.size_one_unity

        # these become the common defaults for all of these functions
        def func(self: tp.Any,
                axis: int = 0,
                skipna: bool = True,
                out: tp.Optional[np.ndarray] = None,
                ) -> tp.Any:
            # some NP contexts might call this function with out given as a kwarg; add it here for compatibility, but ensrue it is not being used
            assert out is None
            return self._ufunc_axis_skipna(
                    axis=axis,
                    skipna=skipna,
                    ufunc=ufunc,
                    ufunc_skipna=ufunc_skipna,
                    composable=composable,
                    dtypes=dtypes,
                    size_one_unity=size_one_unity
                    )

        func.__name__ = func_name
        func.__doc__ = DOC_TEMPLATE.ufunc_skipna.format(header=doc)
        return func


    @staticmethod
    def create_ufunc_shape_skipna(func_name: str) -> AnyCallable:
        # ufunc, ufunc_skipna, dtypes, composable, doc, suze_one_unity = UFUNC_SHAPE_SKIPNA[func_name]
        nt = UFUNC_SHAPE_SKIPNA[func_name]
        ufunc = nt.ufunc
        ufunc_skipna = nt.ufunc_skipna
        dtypes = nt.dtypes
        composable = nt.composable
        doc = nt.doc_header
        size_one_unity = nt.size_one_unity

        # these become the common defaults for all of these functions
        def func(self: tp.Any,
                axis: int = 0,
                skipna: bool = True,
                ) -> tp.Any:
            return self._ufunc_shape_skipna(
                    axis=axis,
                    skipna=skipna,
                    ufunc=ufunc,
                    ufunc_skipna=ufunc_skipna,
                    composable=composable,
                    dtypes=dtypes,
                    size_one_unity=size_one_unity
                    )

        func.__name__ = func_name
        func.__doc__ = DOC_TEMPLATE.ufunc_skipna.format(header=doc)
        return func


    def __new__(mcs,
            name: str,
            bases: tp.Tuple[type, ...],
            attrs: tp.Dict[str, object
            ]) -> type:
        '''
        Create and assign all autopopulated functions.
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

        for func_name in UFUNC_AXIS_SKIPNA:
            attrs[func_name] = mcs.create_ufunc_axis_skipna(func_name)

        for func_name in UFUNC_SHAPE_SKIPNA:
            attrs[func_name] = mcs.create_ufunc_shape_skipna(func_name)

        return type.__new__(mcs, name, bases, attrs)


class ContainerBase(metaclass=ContainerMeta):
    '''
    Root of all containers. Most containers, like Series, Frame, and Index, inherit from ContainerOperaand; only Bus inherits from ContainerBase.
    '''
    __slots__ = EMPTY_TUPLE
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




class ContainerOperand(ContainerBase, metaclass=ContainerOperandMeta):
    '''Base class of all containers that support opperators.'''

    __slots__ = EMPTY_TUPLE

    interface: 'Frame'

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

    def all(self, axis: int = 0, skipna: bool = True) -> bool:
        raise NotImplementedError() # pragma: no cover

    def any(self, axis: int = 0, skipna: bool = True) -> bool:
        raise NotImplementedError() # pragma: no cover

    def sum(self, axis: int = 0, skipna: bool = True) -> tp.Any:
        raise NotImplementedError() # pragma: no cover

    def min(self, axis: int = 0, skipna: bool = True) -> tp.Any:
        raise NotImplementedError() # pragma: no cover

    def max(self, axis: int = 0, skipna: bool = True) -> tp.Any:
        raise NotImplementedError() # pragma: no cover

    def mean(self, axis: int = 0, skipna: bool = True) -> tp.Any:
        raise NotImplementedError() # pragma: no cover

    def median(self, axis: int = 0, skipna: bool = True) -> tp.Any:
        raise NotImplementedError() # pragma: no cover

    def std(self, axis: int = 0, skipna: bool = True) -> tp.Any:
        raise NotImplementedError() # pragma: no cover

    def var(self, axis: int = 0, skipna: bool = True) -> tp.Any:
        raise NotImplementedError() # pragma: no cover

    def prod(self, axis: int = 0, skipna: bool = True) -> tp.Any:
        raise NotImplementedError() # pragma: no cover

    def cumsum(self, axis: int = 0, skipna: bool = True) -> tp.Any:
        raise NotImplementedError() # pragma: no cover

    def cumprod(self, axis: int = 0, skipna: bool = True) -> tp.Any:
        raise NotImplementedError() # pragma: no cover


