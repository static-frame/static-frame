
import typing as tp

import numpy as np

from static_frame.core.interface_meta import InterfaceMeta
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import DTYPE_COMPLEX_KIND
from static_frame.core.util import DTYPE_INT_KINDS
from static_frame.core.util import DTYPE_INEXACT_KINDS
from static_frame.core.util import DTYPE_NAT_KINDS
from static_frame.core.util import DTYPE_OBJECT
from static_frame.core.util import DTYPE_STR_KINDS
from static_frame.core.util import EMPTY_SET
from static_frame.core.util import FLOAT_TYPES
from static_frame.core.util import COMPLEX_TYPES
from static_frame.core.util import EMPTY_TUPLE
from static_frame.core.util import DTYPE_OBJECT_KIND
from static_frame.core.util import DTYPE_FLOAT_KIND
from static_frame.core.util import NAT
from static_frame.core.util import NAT_STR
from static_frame.core.util import DT64_YEAR
from static_frame.core.util import DT64_MONTH

# from static_frame.core.util import InexactTypes

class StoreFilter(metaclass=InterfaceMeta):
    '''
    Utility for defining and applying translation of values going to and from a data store, as needed for XLSX and other writers.
    '''

    __slots__ = (
            'from_nan',
            'from_nat',
            'from_none',
            'from_posinf',
            'from_neginf',
            'to_nan',
            'to_nat',
            'to_none',
            'to_posinf',
            'to_neginf',

            '_FLOAT_FUNC_TO_FROM',
            '_EQUAL_FUNC_TO_FROM',
            '_TYPE_TO_TO_SET',
            '_TYPE_TO_TO_TUPLE',

            'value_format_float_scientific',
            'value_format_float_positional',
            'value_format_complex_scientific',
            'value_format_complex_positional',

            '_value_format_active',
            )

    # from type to string (encoding into the data store)
    from_nan: tp.Optional[str]
    from_nat: tp.Optional[str]
    from_none: tp.Optional[str]
    from_posinf: tp.Optional[str]
    from_neginf: tp.Optional[str]

    # from string to type (decoding from the data store)
    to_nan: tp.FrozenSet[str]
    to_nat: tp.FrozenSet[str]
    to_none: tp.FrozenSet[str]
    to_posinf: tp.FrozenSet[str]
    to_neginf: tp.FrozenSet[str]

    # formatting for inexact types, from type to string
    value_format_float_scientific: tp.Optional[str]
    value_format_float_positional: tp.Optional[str]
    value_format_complex_scientific: tp.Optional[str]
    value_format_complex_positional: tp.Optional[str]

    # reference collections defined with values given above; cannot use AnyCallable here
    _FLOAT_FUNC_TO_FROM: tp.Tuple[tp.Tuple[tp.Any, tp.Optional[str]], ...]
    _EQUAL_FUNC_TO_FROM: tp.Tuple[tp.Tuple[tp.Any, tp.Optional[str]], ...]
    _TYPE_TO_TO_SET: tp.Tuple[tp.Tuple[tp.Any, tp.FrozenSet[str]], ...]
    _TYPE_TO_TO_TUPLE: tp.Tuple[tp.Tuple[tp.Any, tp.Tuple[str, ...]], ...]

    def __init__(self, *,
            # from type to str
            from_nan: tp.Optional[str] = '',
            from_nat: tp.Optional[str] = '',
            from_none: tp.Optional[str] = 'None',
            from_posinf: tp.Optional[str] = 'inf',
            from_neginf: tp.Optional[str] = '-inf',
            # str to type
            to_nan: tp.FrozenSet[str] = frozenset(('', 'nan', 'NaN', 'NAN', 'NULL', '#N/A')),
            to_nat: tp.FrozenSet[str] = frozenset(EMPTY_TUPLE), # do not assume there are NaTs.
            to_none: tp.FrozenSet[str] = frozenset(('None',)),
            to_posinf: tp.FrozenSet[str] = frozenset(('inf',)),
            to_neginf: tp.FrozenSet[str] = frozenset(('-inf',)),
            # from float to str
            value_format_float_positional: tp.Optional[str] = None,
            value_format_float_scientific: tp.Optional[str] = None,
            value_format_complex_positional: tp.Optional[str] = None,
            value_format_complex_scientific: tp.Optional[str] = None,
            ) -> None:

        self.from_nan = from_nan
        self.from_nat = from_nat
        self.from_none = from_none
        self.from_posinf = from_posinf
        self.from_neginf = from_neginf

        self.to_nan = to_nan
        self.to_nat = to_nat
        self.to_none = to_none
        self.to_posinf = to_posinf
        self.to_neginf = to_neginf

        self.value_format_float_positional = value_format_float_positional
        self.value_format_float_scientific = value_format_float_scientific
        self.value_format_complex_positional = value_format_complex_positional
        self.value_format_complex_scientific = value_format_complex_scientific

        self._value_format_active = (
                self.value_format_float_positional is not None or
                self.value_format_float_scientific is not None or
                self.value_format_complex_positional is not None or
                self.value_format_complex_scientific is not None
                )

        # assumed faster to define these per instance than at the class level
        # None has to be handled separately
        self._FLOAT_FUNC_TO_FROM = (
                (np.isnan, self.from_nan),
                (np.isposinf, self.from_posinf),
                (np.isneginf, self.from_neginf)
                )

        # for object array processing
        self._EQUAL_FUNC_TO_FROM = (
                # NOTE: this using the same heuristic as util.isna_array, which may not be the best choice for non-standard objects
                (lambda x: np.not_equal(x, x), self.from_nan),
                (lambda x: np.equal(x, None), self.from_none),
                (lambda x: np.equal(x, np.inf), self.from_posinf),
                (lambda x: np.equal(x, -np.inf), self.from_neginf)
                )

        #-----------------------------------------------------------------------
        # these are used for converting from strings to types
        self._TYPE_TO_TO_SET = (
                (np.nan, self.to_nan),
                (NAT, self.to_nat),
                (None, self.to_none),
                (np.inf, self.to_posinf),
                (-np.inf, self.to_neginf)
                )

        # for using isin, cannot use a set, so pre-convert to tuples here
        self._TYPE_TO_TO_TUPLE = (
                (np.nan, tuple(self.to_nan)),
                (NAT, tuple(self.to_nat)),
                (None, tuple(self.to_none)),
                (np.inf, tuple(self.to_posinf)),
                (-np.inf, tuple(self.to_neginf)),
                )

    # --------------------------------------------------------------------------
    # converting from types (in memory) to datastore


    def _format_inexact_element(self,
            value: tp.Any,
            kind: str,
            ) -> tp.Any:
        '''
        Must let unexact types pass, as object arrays will have mixed types.
        '''
        if kind == DTYPE_OBJECT_KIND:
            is_float = isinstance(value, FLOAT_TYPES)
            is_complex = False if is_float else isinstance(value, COMPLEX_TYPES)
        elif kind == DTYPE_FLOAT_KIND:
            is_float = True
            is_complex = False
        elif kind == DTYPE_COMPLEX_KIND:
            is_float = False
            is_complex = True

        if not is_float and not is_complex:
            return value

        # NOTE: similar move in Display.to_cell
        # must call built-in str() to get native realization per element
        is_scientific = 'e' in str(value)
        if is_float:
            if self.value_format_float_scientific is not None and is_scientific:
                return self.value_format_float_scientific.format(value)
            elif self.value_format_float_positional is not None:
                return self.value_format_float_positional.format(value)
            return value

        # is_complex
        if self.value_format_complex_scientific is not None and is_scientific:
            return self.value_format_complex_scientific.format(value)
        elif self.value_format_complex_positional is not None:
            return self.value_format_complex_positional.format(value)
        return value


    def _format_inexact_array(self,
            array: np.ndarray,
            array_object: tp.Optional[np.ndarray],
            ) -> np.ndarray:
        '''
        Args:
            array_object: if we have already created an object array, use it as destination, mutating values in-place. ``array`` and ``array_object`` can be the same array.
        '''
        # NOTE: assume only called on object or inexact dtypes, and when at least one of the value_format attributes is non-None
        kind = array.dtype.kind
        if array_object is None:
            if kind == DTYPE_OBJECT_KIND:
                array_object = array.copy()
            else:
                array_object = array.astype(DTYPE_OBJECT)

        func = self._format_inexact_element
        for iloc, e in np.ndenumerate(array):
            array_object[iloc] = func(e, kind)

        return array_object

    def from_type_filter_array(self,
            array: np.ndarray
            ) -> np.ndarray:
        '''Given an array, replace types with strings
        '''
        kind = array.dtype.kind
        dtype = array.dtype

        if kind in DTYPE_INT_KINDS or kind in DTYPE_STR_KINDS or dtype == DTYPE_BOOL:
            return array # no replacements possible

        kind_is_complex = kind == DTYPE_COMPLEX_KIND
        kind_is_object = kind == DTYPE_OBJECT_KIND

        if kind in DTYPE_INEXACT_KINDS or kind_is_object:
            func_value_replace_pairs = (
                    self._EQUAL_FUNC_TO_FROM if kind_is_object
                    else self._FLOAT_FUNC_TO_FROM)

            post = None # defer creating until we have a match
            for func, value_replace in func_value_replace_pairs:
                if value_replace is not None:
                    # cannot use these ufuncs on complex array
                    if kind_is_complex and (func == np.isposinf or func == np.isneginf):
                        continue
                    found = func(array)
                    if found.any():
                        if post is None:
                            # need to store string replacements in object type
                            # astype always returns a copy by default
                            post = array.astype(DTYPE_OBJECT)
                        post[found] = value_replace

            array_final = post if post is not None else array
            if self._value_format_active:
                return self._format_inexact_array(array_final, post)
            return array_final

        if kind in DTYPE_NAT_KINDS:
            post = None
            if array.dtype == DT64_YEAR or array.dtype == DT64_MONTH:
                post = array.astype(str) # nat will go to "NaT"

            if post is not None and post.dtype.kind in DTYPE_STR_KINDS:
                is_nat = post == NAT_STR
            else: # still datetime
                is_nat = np.isnat(array)

            # we always force datetime64 to object, as most formats (i.e., XLSX) are not prepared to write them
            post = post if post is not None else array.astype(DTYPE_OBJECT)
            if is_nat.any():
                post[is_nat] = self.from_nat
            return post if post is not None else array

        raise NotImplementedError(f'no handling for dtype {dtype}') #pragma: no cover

    def from_type_filter_element(self,
            value: tp.Any
            ) -> tp.Any:
        '''
        Filter single values to string.
        '''
        # apply to all types
        if self.from_none is not None and value is None:
            return self.from_none

        is_float = isinstance(value, FLOAT_TYPES)
        is_complex = False if is_float else isinstance(value, COMPLEX_TYPES)

        if is_float or is_complex:
            for func, value_replace in self._FLOAT_FUNC_TO_FROM:
                if value_replace is not None:
                    if is_complex and (func == np.isposinf or func == np.isneginf):
                        continue
                    if func(value):
                        return value_replace
        if isinstance(value, np.datetime64):
            if np.isnat(value):
                value = self.from_nat
            elif value.dtype == DT64_YEAR or value.dtype == DT64_MONTH:
                value = str(value) # convert year, month to string

        if self._value_format_active:
            if is_float:
                kind = DTYPE_FLOAT_KIND
            elif is_complex:
                kind = DTYPE_COMPLEX_KIND
            else:
                kind = DTYPE_OBJECT_KIND
            return self._format_inexact_element(value, kind)
        return value

    #---------------------------------------------------------------------------
    # converting from strings (in data store) to types

    def to_type_filter_array(self,
            array: np.ndarray
            ) -> np.ndarray:
        '''Given an array, replace strings with types.
        '''
        kind = array.dtype.kind
        dtype = array.dtype

        # nothin to do with ints, floats, or bools
        if (kind in DTYPE_INT_KINDS
                or kind in DTYPE_INEXACT_KINDS
                or dtype == DTYPE_BOOL
                ):
            return array # no replacements possible

        # need to only check object or float
        if kind in DTYPE_STR_KINDS or dtype == DTYPE_OBJECT:
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
