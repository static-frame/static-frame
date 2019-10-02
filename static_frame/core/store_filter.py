
import typing as tp

import numpy as np # type: ignore

from static_frame.core.util import DTYPE_INT_KIND
from static_frame.core.util import DTYPE_STR_KIND
from static_frame.core.util import DTYPE_NAN_KIND
from static_frame.core.util import DTYPE_NAT_KIND

# from static_frame.core.util import DTYPE_DATETIME_KIND
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import DTYPE_OBJECT

from static_frame.core.util import FLOAT_TYPES


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
            'to_neginf'
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

    def __init__(self,
            # from type to str
            from_nan: tp.Optional[str] = '',
            from_none: tp.Optional[str] = 'None',
            from_posinf: tp.Optional[str] = '+inf',
            from_neginf: tp.Optional[str] = '-inf',
            # str to type
            to_nan: tp.FrozenSet[str] = frozenset(('', 'NaN', 'NAN', 'NULL', '#N/A')),
            to_none: tp.FrozenSet[str] = frozenset(('None',)),
            to_posinf: tp.FrozenSet[str] = frozenset(('inf', '+inf')),
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


    def from_type_filter_array(self,
            array: np.ndarray
            ) -> np.ndarray:
        '''Given an array, replace values approrpriately
        '''
        kind = array.dtype.kind
        dtype = array.dtype

        if kind in DTYPE_INT_KIND or kind in DTYPE_STR_KIND or dtype == DTYPE_BOOL:
            return array # no replacements posible

        if kind in DTYPE_NAN_KIND:
            if (self.from_nan is None
                    and self.from_posinf is None
                    and self.from_neginf is None):
                return array

            # can have all but None
            post = array.astype(object) # get a copy to mutate

            for func, value in (
                    (np.isnan, self.from_nan),
                    (np.isposinf, self.from_posinf),
                    (np.isneginf, self.from_neginf)
                    ):
                if value is not None:
                    post[func(array)] = value

            # if self.from_nan is not None:
            #     post[np.isnan(array)] = self.from_nan
            # if self.from_posinf is not None:
            #     post[np.isposinf(array)] = self.from_posinf
            # if self.from_neginf is not None:
            #     post[np.isneginf(array)] = self.from_neginf

            return post

        if kind in DTYPE_NAT_KIND:
            raise NotImplementedError() # np.isnat

        if dtype == DTYPE_OBJECT:
            if (self.from_nan is None
                    and self.from_none is None
                    and self.from_posinf is None
                    and self.from_neginf is None):
                return array

            array = array.copy() # get a copy to mutate

            if self.from_nan is not None:
                # NOTE: this using the same heuristic as util.isna_array,, which may not be the besr choice for non-standard objects
                array[np.not_equal(array, array)] = self.from_nan

            # if self.from_none is not None:
            #     array[np.equal(array, None)] = self.from_none
            # if self.from_posinf is not None:
            #     array[np.equal(array, np.inf)] = self.from_posinf
            # if self.from_neginf is not None:
            #     array[np.equal(array, -np.inf)] = self.from_neginf

            for equal_to, value in (
                    (None, self.from_none),
                    (np.inf, self.from_posinf),
                    (-np.inf, self.from_neginf)
                    ):
                if value is not None:
                    array[np.equal(array, equal_to)] = value

            return array

        return array

    def from_type_filter_value(self,
            value: tp.Any
            ) -> tp.Any:
        '''
        Filter single values to string.
        '''

        if self.from_none is not None and value is None:
            return self.from_none

        if isinstance(value, FLOAT_TYPES):
            if self.from_nan is not None and np.isnan(value):
                return self.from_nan





STORE_FILTER_DEFAULT = StoreFilter()

