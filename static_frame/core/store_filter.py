
import typing as tp

import numpy as np

from static_frame.core.util import DTYPE_INT_KIND
from static_frame.core.util import DTYPE_STR_KIND
from static_frame.core.util import DTYPE_NAN_KIND
from static_frame.core.util import DTYPE_NAT_KIND
from static_frame.core.util import DTYPE_COMPLEX_KIND

# from static_frame.core.util import DTYPE_DATETIME_KIND
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import DTYPE_OBJECT

from static_frame.core.util import FLOAT_TYPES
# from static_frame.core.util import AnyCallable
from static_frame.core.util import EMPTY_SET


class StoreFilter:
    '''
    Uitlity for defining and applying translation in stored values, as needed for XLSX and other writers.
    '''

    __slots__ = (
            'from_nan',
            'from_none',
            'from_posinf',
            'from_neginf',
            'to_nan',
            'to_none',
            'to_posinf',
            'to_neginf',

            '_FLOAT_FUNC_TO_FROM',
            '_EQUAL_FUNC_TO_FROM',
            '_TYPE_TO_TO_SET',
            '_TYPE_TO_TO_TUPLE',
            )

    # from type to string
    from_nan: tp.Optional[str]
    from_none: tp.Optional[str]
    from_posinf: tp.Optional[str]
    from_neginf: tp.Optional[str]

    # fro string to type
    to_nan: tp.FrozenSet[str]
    to_none: tp.FrozenSet[str]
    to_posinf: tp.FrozenSet[str]
    to_neginf: tp.FrozenSet[str]

    # cannot use AnyCallable here
    _FLOAT_FUNC_TO_FROM: tp.Tuple[tp.Tuple[tp.Any, tp.Optional[str]], ...]
    _EQUAL_FUNC_TO_FROM: tp.Tuple[tp.Tuple[tp.Any, tp.Optional[str]], ...]
    _TYPE_TO_TO_SET: tp.Tuple[tp.Tuple[tp.Any, tp.FrozenSet[str]], ...]
    _TYPE_TO_TO_TUPLE: tp.Tuple[tp.Tuple[tp.Any, tp.Tuple[str, ...]], ...]

    def __init__(self,
            # from type to str
            from_nan: tp.Optional[str] = '',
            from_none: tp.Optional[str] = 'None',
            from_posinf: tp.Optional[str] = 'inf',
            from_neginf: tp.Optional[str] = '-inf',
            # str to type
            to_nan: tp.FrozenSet[str] = frozenset(('', 'nan', 'NaN', 'NAN', 'NULL', '#N/A')),
            to_none: tp.FrozenSet[str] = frozenset(('None',)),
            to_posinf: tp.FrozenSet[str] = frozenset(('inf',)),
            to_neginf: tp.FrozenSet[str] = frozenset(('-inf',)),
            ) -> None:

        self.from_nan = from_nan
        self.from_none = from_none
        self.from_posinf = from_posinf
        self.from_neginf = from_neginf

        self.to_nan = to_nan
        self.to_none = to_none
        self.to_posinf = to_posinf
        self.to_neginf = to_neginf

        # assumed faster to define these per instance than at the class level; this avoids having to use a getattr call to get a handle to the instance method, as wold be necessary if this was on th eclass


        # None has to be handled separately
        self._FLOAT_FUNC_TO_FROM = (
                (np.isnan, self.from_nan),
                (np.isposinf, self.from_posinf),
                (np.isneginf, self.from_neginf)
                )

        # for object array processing
        self._EQUAL_FUNC_TO_FROM = (
                # NOTE: this using the same heuristic as util.isna_array,, which may not be the besr choice for non-standard objects
                (lambda x: np.not_equal(x, x), self.from_nan),
                (lambda x: np.equal(x, None), self.from_none),
                (lambda x: np.equal(x, np.inf), self.from_posinf),
                (lambda x: np.equal(x, -np.inf), self.from_neginf)
                )

        self._TYPE_TO_TO_SET = (
                (np.nan, self.to_nan),
                (None, self.to_none),
                (np.inf, self.to_posinf),
                (-np.inf, self.to_neginf)
                )

        # for using isin, cannot use a set, so pre-convert to tuples here
        self._TYPE_TO_TO_TUPLE = (
                (np.nan, tuple(self.to_nan)),
                (None, tuple(self.to_none)),
                (np.inf, tuple(self.to_posinf)),
                (-np.inf, tuple(self.to_neginf)),
                )

    def from_type_filter_array(self,
            array: np.ndarray
            ) -> np.ndarray:
        '''Given an array, replace types with strings
        '''
        kind = array.dtype.kind
        dtype = array.dtype

        if kind in DTYPE_INT_KIND or kind in DTYPE_STR_KIND or dtype == DTYPE_BOOL:
            return array # no replacements posible

        if kind in DTYPE_NAN_KIND:
            # if all(v is None for _, v in self._FLOAT_FUNC_TO_FROM):
            #     return array

            post = None # defer creating until we have a match
            for func, value_replace in self._FLOAT_FUNC_TO_FROM:
                if value_replace is not None:
                    # cannot use these ufuncs on complex array
                    if (array.dtype.kind == DTYPE_COMPLEX_KIND
                            and (func == np.isposinf or func == np.isneginf)):
                        continue
                    found = func(array)

                    if found.any():
                        if post is None:
                            # need to store string replacements in object type
                            post = array.astype(object) # get a copy to mutate
                        post[found] = value_replace
            return post if post is not None else array

        if kind in DTYPE_NAT_KIND:
            raise NotImplementedError() # np.isnat

        if dtype == DTYPE_OBJECT:
            post = None
            for func, value_replace in self._EQUAL_FUNC_TO_FROM:
                if value_replace is not None:
                    found = func(array)
                    if found.any():
                        if post is None:
                            post = array.copy() # get a copy to mutate
                        post[found] = value_replace
            return post if post is not None else array

        return array

    def from_type_filter_element(self,
            value: tp.Any
            ) -> tp.Any:
        '''
        Filter single values to string.
        '''
        # apply to all types
        if self.from_none is not None and value is None:
            return self.from_none

        if isinstance(value, FLOAT_TYPES):
            for func, value_replace in self._FLOAT_FUNC_TO_FROM:
                if value_replace is not None and func(value):
                    return value_replace
        return value


    def to_type_filter_array(self,
            array: np.ndarray
            ) -> np.ndarray:
        '''Given an array, replace strings with types.
        '''
        kind = array.dtype.kind
        dtype = array.dtype

        # nothin to do with ints, floats, or bools
        if (kind in DTYPE_INT_KIND
                or kind in DTYPE_NAN_KIND
                or dtype == DTYPE_BOOL
                ):
            return array # no replacements posible

        # need to only check object or float
        if kind in DTYPE_STR_KIND or dtype == DTYPE_OBJECT:
            # for string types, cannot use np.equal
            post = None
            for value_replace, matching in self._TYPE_TO_TO_TUPLE:
                if matching:
                    found = np.isin(array, matching)
                    if found.any():
                        if post is None:
                            post = array.astype(object) # get a copy to mutate
                        post[found] = value_replace
            return post if post is not None else array

        return array


    def to_type_filter_element(self,
            value: tp.Any
            ) -> tp.Any:
        '''
        Given a value wich may be an encoded string, decode into a type.
        '''
        if isinstance(value, str):
            for value_replace, matching in self._TYPE_TO_TO_SET:
                if value in matching:
                    return value_replace
        return value

    def to_type_filter_iterable(self, iterable: tp.Iterable[tp.Any]) -> tp.Iterator[tp.Any]:
        for value in iterable:
            yield self.to_type_filter_element(value)



STORE_FILTER_DEFAULT = StoreFilter()

STORE_FILTER_DISABLE = StoreFilter(
            from_nan=None,
            from_none=None,
            from_posinf=None,
            from_neginf=None,
            # str to type
            to_nan=EMPTY_SET,
            to_none=EMPTY_SET,
            to_posinf=EMPTY_SET,
            to_neginf=EMPTY_SET,
            )
