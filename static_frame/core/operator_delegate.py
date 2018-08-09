import typing as tp

from itertools import chain
from itertools import product
from functools import wraps

import operator as operator_mod


import numpy as np

from static_frame.core.util import _DTYPE_INT_KIND
from static_frame.core.util import _DTYPE_STR_KIND


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

# all reverse are binary
# should be RIGHT, not REVERSE
_REVERSE_OPERATOR_MAP = {
        '__radd__': '__add__',
        '__rsub__': '__sub__',
        '__rmul__': '__mul__',
        '__rtruediv__': '__truediv__',
        '__rfloordiv__': '__floordiv__',
        }


def _ufunc_logical_skipna(array: np.ndarray,
        ufunc: tp.Callable,
        skipna: bool,
        axis: int=0,
        out=None
        ) -> np.ndarray:
    '''
    Given a logical (and, or) ufunc that does not support skipna, implement skipna behavior.
    '''
    if ufunc != np.all and ufunc != np.any:
        raise Exception('unsupported ufunc')

    if len(array) == 0:
        # TODO: handle if this is ndim == 2 and has no length
        if ufunc == np.all:
            return True
        return False # any() of an empty array is False

    if array.dtype.kind == 'b':
        # if boolean execute first
        return ufunc(array, axis=axis, out=out)
    elif array.dtype.kind == 'f':
        if skipna:
            # replace nans with nonzero value; faster to use masked array?
            v = array.copy()
            v[np.isnan(array)] = 0
            return ufunc(v, axis=axis, out=out)
        return ufunc(array, axis=axis, out=out)
    elif array.dtype.kind in _DTYPE_INT_KIND:
        return ufunc(array, axis=axis, out=out)

    # all types other than strings or objects" assume truthy
    elif array.dtype.kind != 'O' and array.dtype.kind not in _DTYPE_STR_KIND:
        if array.ndim == 1:
            return True
        return np.full(array.shape[0 if axis else 1], fill_value=True, dtype=bool)

    # convert to boolean aray then process
    if skipna:
        v = np.fromiter(((False if x is np.nan else bool(x)) for x in array.flat),
                count=array.size,
                dtype=bool).reshape(array.shape)
    else:
        v = np.fromiter((bool(x) for x in array.flat),
                count=array.size,
                dtype=bool).reshape(array.shape)
    return ufunc(v, axis=axis, out=out)


def _all(array, axis=0, out=None):
    return _ufunc_logical_skipna(array, ufunc=np.all, skipna=False, axis=axis, out=out)

_all.__doc__ = np.all.__doc__

def _any(array, axis=0, out=None):
    return _ufunc_logical_skipna(array, ufunc=np.any, skipna=False, axis=axis, out=out)

_any.__doc__ = np.any.__doc__

def _nanall(array, axis=0, out=None):
    return _ufunc_logical_skipna(array, ufunc=np.all, skipna=True, axis=axis, out=out)

def _nanany(array, axis=0, out=None):
    return _ufunc_logical_skipna(array, ufunc=np.any, skipna=True, axis=axis, out=out)




_UFUNC_AXIS_SKIPNA = {
        'all': (_all, _nanall, bool),
        'any': (_any, _nanany, bool),
        'sum': (np.sum, np.nansum, None),
        'min': (np.min, np.nanmin, None),
        'max': (np.max, np.nanmax, None),
        'mean': (np.mean, np.nanmean, None),
        'std': (np.std, np.nanstd, None),
        'var': (np.var, np.nanvar, None),
        'prod': (np.prod, np.nanprod, None),
        'cumsum': (np.cumsum, np.nancumsum, None),
        'cumprod': (np.cumprod, np.nancumprod, None)
        }

class MetaOperatorDelegate(type):
    '''Auto-populate binary and unary methods based on instance methods named `_ufunc_unary_operator` and `_ufunc_binary_operator`.
    '''

    @staticmethod
    def create_ufunc_operator(func_name, opperand_count=1, reverse=False):
        # operator module defines alias to funcs with names like __add__, etc
        if not reverse:
            operator_func = getattr(operator_mod, func_name)
            func_wrapper = operator_func
        else:
            unreversed_operator_func = getattr(operator_mod, _REVERSE_OPERATOR_MAP[func_name])
            # flip the order of the arguments
            operator_func = lambda rhs, lhs: unreversed_operator_func(lhs, rhs)
            func_wrapper = unreversed_operator_func

        if opperand_count == 1:
            assert not reverse # cannot reverse a single opperand
            def func(self):
                return self._ufunc_unary_operator(operator_func)
        elif opperand_count == 2:
            def func(self, other):
                return self._ufunc_binary_operator(operator=operator_func, other=other)
        else:
            raise NotImplementedError()

        f = wraps(func_wrapper)(func)
        f.__name__ = func_name
        return f

    @staticmethod
    def create_ufunc_axis_skipna(func_name):
        ufunc, ufunc_skipna, dtype = _UFUNC_AXIS_SKIPNA[func_name]

        # these become the common defaults for all of these functions
        def func(self, axis=0, skipna=True, **_):
            return self._ufunc_axis_skipna(
                    axis=axis,
                    skipna=skipna,
                    ufunc=ufunc,
                    ufunc_skipna=ufunc_skipna,
                    dtype=dtype)

        f = wraps(ufunc)(func) # not sure if this is correct
        f.__name__ = func_name
        return f

    def __new__(mcs, name, bases, attrs):
        '''
        Create and assign all autopopulated functions.
        '''
        for opperand_count, func_name in chain(
                product((1,), _UFUNC_UNARY_OPERATORS),
                product((2,), _UFUNC_BINARY_OPERATORS)):
            attrs[func_name] = mcs.create_ufunc_operator(
                    func_name,
                    opperand_count=opperand_count)
        for func_name in _REVERSE_OPERATOR_MAP:
            attrs[func_name] = mcs.create_ufunc_operator(
                    func_name,
                    opperand_count=2,
                    reverse=True)

        for func_name in _UFUNC_AXIS_SKIPNA:
            attrs[func_name] = mcs.create_ufunc_axis_skipna(func_name)

        return type.__new__(mcs, name, bases, attrs)

