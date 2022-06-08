
import typing as tp

import numpy as np

from static_frame.core.util import NAT
from static_frame.core.util import NAT_TD64

FILL_VALUE_AUTO_DEFAULT = object()

class FillValueAuto:
    '''Define, per NumPy dtype kind, a value to be used for filling missing values.
    '''
    __slots__ = tuple('biufcmMOSUV')

    @classmethod
    def from_default(cls,
            b: tp.Any = False, # np.bool_(False)
            i: tp.Any = 0,
            u: tp.Any = 0,
            f: tp.Any = np.nan,
            c: tp.Any = complex(np.nan, np.nan),
            m: tp.Any = NAT_TD64,
            M: tp.Any = NAT,
            O: tp.Any = None,
            S: tp.Any = b'',
            U: tp.Any = '',
            V: tp.Any = b'\0',
    ) -> 'FillValueAuto':
        '''Create a ``FileValueAuto`` instance based on a default selected to prohibit type coercions.

        Args:
            b: fill value for bool kind
            i: fill value for integer kind
            u: fill value for unsigned integer kind
            f: fill value for float kind
            c: fill value for complex kind
            m: fill value for timedelta62 kind
            M: fill value for datetime64 kind
            O: fill value for object kind
            S: fill value for bytes kind
            U: fill value for unicode kind
            V: fill value for void kind
        '''
        return cls(
                b=b,
                i=i,
                u=u,
                f=f,
                c=c,
                m=m,
                M=M,
                O=O,
                S=S,
                U=U,
                V=V,
                )

    def __init__(self,
            b: tp.Any = FILL_VALUE_AUTO_DEFAULT,
            i: tp.Any = FILL_VALUE_AUTO_DEFAULT,
            u: tp.Any = FILL_VALUE_AUTO_DEFAULT,
            f: tp.Any = FILL_VALUE_AUTO_DEFAULT,
            c: tp.Any = FILL_VALUE_AUTO_DEFAULT,
            m: tp.Any = FILL_VALUE_AUTO_DEFAULT,
            M: tp.Any = FILL_VALUE_AUTO_DEFAULT,
            O: tp.Any = FILL_VALUE_AUTO_DEFAULT,
            S: tp.Any = FILL_VALUE_AUTO_DEFAULT,
            U: tp.Any = FILL_VALUE_AUTO_DEFAULT,
            V: tp.Any = FILL_VALUE_AUTO_DEFAULT,
            ) -> None:
        '''
        Args:
            b: fill value for bool kind
            i: fill value for integer kind
            u: fill value for unsigned integer kind
            f: fill value for float kind
            c: fill value for complex kind
            m: fill value for timedelta62 kind
            M: fill value for datetime64 kind
            O: fill value for object kind
            S: fill value for bytes kind
            U: fill value for unicode kind
            V: fill value for void kind
        '''
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

    def __getitem__(self, dtype: np.dtype) -> tp.Any:
        fv = getattr(self, dtype.kind)
        if fv is FILL_VALUE_AUTO_DEFAULT:
            raise RuntimeError(f'FillValueAuto requested value for kind {dtype.kind} not defined.')
        return fv
