import numpy as np
# import static_frame as sf
import typing as tp


def process1(x: np.ndarray[tuple[int], np.dtype[np.integer]]): ...


a1 = np.empty(100, dtype=np.int16)
process1(a1) # mypy passes

a2 = np.empty(100, dtype=np.float32)
process1(a2) # mypy fails
# nptyping.py:12: error: Argument 1 to "process1" has incompatible type "ndarray[tuple[int], dtype[floating[_32Bit]]]"; expected "ndarray[tuple[int], dtype[integer[Any]]]"


a3 = np.empty((100, 100, 100), dtype=np.int64)
process1(a3) # mypy fails
# nptyping.py:18: error: Argument 1 to "process1" has incompatible type "ndarray[tuple[int, int, int], dtype[signedinteger[_64Bit]]]"; expected "ndarray[tuple[int], dtype[integer[Any]]]"



