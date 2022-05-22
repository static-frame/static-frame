
import typing as tp

import numpy as np

from static_frame.core.util import NAT
from static_frame.core.util import NAT_TD64


class FillValueAuto:
    '''Define, per NumPy dtype kind, a value to be used for filling missing values.
    '''
    __slots__ = tuple('biufcmMOSUV')

    def __init__(self,
            b: tp.Any = False, # np.bool_(False)
            i: tp.Any = 0,
            u: tp.Any = 0,
            f: tp.Any = np.nan,
            c: tp.Any = np.nan,
            m: tp.Any = NAT_TD64,
            M: tp.Any = NAT,
            O: tp.Any = None,
            S: tp.Any = b'',
            U: tp.Any = '',
            V: tp.Any = b'\0',
            ) -> None:

        self.b = b
        self.i = i
        self.u = u
        self.f = f
        self.c = c
        self.m = m
        self.M = M
        self.O = O
        self.S = S
        self.U = U
        self.V = V

    def __call__(self, dtype: np.dtype) -> tp.Any:
        return getattr(self, dtype.kind)
