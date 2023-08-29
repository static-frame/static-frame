import numpy as np
import numpy.typing as npt

import static_frame as sf


def run_idx_int(idx: sf.Index[np.int64]) -> bool:
    return True

def run_bool(idx: bool) -> bool:
    return True


def test_generics_a() -> None:

    idx1: sf.Index[np.int64] = sf.Index((2, 3))
    idx2: sf.Index[np.unicode_] = sf.Index(('a', 'b'))

    x = run_idx_int(idx1)
    # y = run_idx_int(idx2) # this fails

    # x = run_bool(idx1)

    dt1: np.dtype[np.int64] = idx1.dtype
    # dt2: np.dtype[np.int64] = idx2.dtype # this fails



