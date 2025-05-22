import typing as tp

import numpy as np

import static_frame as sf


def process1(x: np.ndarray[tuple[int], np.dtype[np.signedinteger]]): ...


a1 = np.empty(100, dtype=np.int16)
process1(a1) # mypy passes

a2 = np.empty(100, dtype=np.uint8)
process1(a2) # mypy fails
# nptyping.py:13: error: Argument 1 to "process1" has incompatible type "ndarray[tuple[int], dtype[unsignedinteger[_8Bit]]]"; expected "ndarray[tuple[int], dtype[signedinteger[Any]]]"  [arg-type]


a3 = np.empty((100, 100, 100), dtype=np.int64)
process1(a3) # mypy fails
# nptyping.py:18: error: Argument 1 to "process1" has incompatible type "ndarray[tuple[int, int, int], dtype[signedinteger[_64Bit]]]"; expected "ndarray[tuple[int], dtype[signedinteger[Any]]]"  [arg-type]



import static_frame as sf


@sf.CallGuard.check
def process2(x: np.ndarray[tuple[int], np.dtype[np.signedinteger]]): ...

b1 = np.empty(100, dtype=np.uint8)
# process2(b1)
# static_frame.core.type_clinic.ClinicError:
# In args of (x: ndarray[tuple[int], dtype[signedinteger]]) -> Any
# └── In arg x
#     └── ndarray[tuple[int], dtype[signedinteger]]
#         └── dtype[signedinteger]
#             └── Expected signedinteger, provided uint8 invalid

b2 = np.empty((10, 100), dtype=np.int8)
process2(b2)
# static_frame.core.type_clinic.ClinicError:
# In args of (x: ndarray[tuple[int], dtype[signedinteger]]) -> Any
# └── In arg x
#     └── ndarray[tuple[int], dtype[signedinteger]]
#         └── tuple[int]
#             └── Expected tuple length of 1, provided tuple length of 2