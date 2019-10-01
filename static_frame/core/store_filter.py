# REPLACE_DEFAULT = object()

# ReplaceTotp.Set[str] = tp.Dict[tp.Hashable, str]
# ReplaceFromStr = tp.Dict[str, tp.Set[str]]

# REPLACE_VALID_KEYS = frozenset((np.nan, None, np.inf, -np.inf))

# REPLACE_TO_STR_DEFAULT = {
#         np.nan: '',
#         None: 'None',
#         np.inf: 'inf',
#         -np.inf: '-inf'
#         }
# REPLACE_FROM_STR_DEFAULT = {
#         np.nan: frozenset(('', 'NaN', 'NAN', 'NULL', '#N/A')),
#         None: frozenset(('None',)),
#         np.inf: frozenset(('inf',)),
#         -np.inf: frozenset(('-inf',))
#         }


# def valid_replace(replace: tp.Dict[tp.Hashable, tp.Any]) -> None:
#     '''Retrun Boolean of the replace dictionary has valid keys.
#     '''
#     # NOTE: could be a replace filter
#     if not (replace.keys() | REPLACE_VALID_KEYS) == REPLACE_VALID_KEYS:
#         raise RuntimeError(f'invalid replace keys; must include only {REPLACE_VALID_KEYS}')

import typing as tp

import numpy as np # type: ignore

from static_frame.core.util import DTYPE_INT_KIND
from static_frame.core.util import DTYPE_STR_KIND
from static_frame.core.util import DTYPE_NAN_KIND
from static_frame.core.util import DTYPE_NAT_KIND

# from static_frame.core.util import DTYPE_DATETIME_KIND
from static_frame.core.util import DTYPE_BOOL
from static_frame.core.util import DTYPE_OBJECT



class StoreFilter:
    '''
    Uitlity for defining and applying translation in stored values, as needed for XLSX and other writers.
    '''

    __slots__ = (
            'to_nan',
            'to_None',
            'to_posinf',
            'to_neginf',
            'from_nan',
            'from_None',
            'from_posinf',
            'from_neginf'
            )

    to_nan: tp.Optional[str]
    to_None: tp.Optional[str]
    to_posinf: tp.Optional[str]
    to_neginf: tp.Optional[str]

    from_nan: tp.Set[str]
    from_None: tp.Set[str]
    from_posinf: tp.Set[str]
    from_neginf: tp.Set[str]

    def __init__(self,
            to_nan: tp.Optional[str],
            to_None: tp.Optional[str],
            to_posinf: tp.Optional[str],
            to_neginf: tp.Optional[str],

            from_nan: tp.Set[str],
            from_None: tp.Set[str],
            from_posinf: tp.Set[str],
            from_neginf: tp.Set[str]
            ) -> None:

        self.to_nan = to_nan
        self.to_None = to_None
        self.to_posinf = to_posinf
        self.to_neginf = to_neginf

        self.from_nan = from_nan
        self.from_None = from_None
        self.from_posinf = from_posinf
        self.from_neginf = from_neginf


    def filter_array(self,
            array: np.ndarray
            ) -> np.ndarray:
        '''Given an array, replace values approrpriately
        '''
        kind = array.dtype.kind
        dtype = array.dtype

        if kind in DTYPE_INT_KIND or kind in DTYPE_STR_KIND or dtype == DTYPE_BOOL:
            return array # no replacements possilbe

        if kind in DTYPE_NAN_KIND:
            # can have all but None
            post = array.astype(object) # get a copy to mutate

            if self.to_nan is not None:
                post[np.isnan(array)] = self.to_nan
            if self.to_posinf is not None:
                post[np.isposinf(array)] = self.to_posinf
            if self.to_neginf is not None:
                post[np.isneginf(array)] = self.to_neginf

            return post

        if kind in DTYPE_NAT_KIND:
            raise NotImplementedError() # np.isnat

        if dtype == DTYPE_OBJECT:
            array = array.copy() # get a copy to mutate

            if self.to_nan is not None:
                # NOTE: this using the same heuristic as util.isna_array,, which may not be the besr choice for non-standard objects
                array[np.not_equal(array, array)] = self.to_nan

            if self.to_posinf is not None:
                array[np.equal(array, np.inf)] = self.to_posinf
            if self.to_neginf is not None:
                array[np.equal(array, -np.inf)] = self.to_neginf
            if self.to_None is not None:
                array[np.equal(array, None)] = self.to_None

            return array

        return array


STORE_FILTER_DEFAULT = StoreFilter(
        to_nan = '',
        to_None = 'None',
        to_posinf = '+inf',
        to_neginf = '-inf',
        from_nan = frozenset(('', 'NaN', 'NAN', 'NULL', '#N/A')),
        from_None = frozenset(('None',)),
        from_posinf = frozenset(('inf', '+inf')),
        from_neginf = frozenset(('-inf',))
)

