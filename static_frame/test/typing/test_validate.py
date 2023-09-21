import numpy as np
import typing_extensions as tp

# from static_frame.core.validate import check_type
from static_frame.test.test_case import skip_nple119


@skip_nple119
def test_ndarray_a() -> None:
    v = np.array([False, True, False])
    # NOTE: must type this as a dytpe, not just a a generic
    h1 = np.ndarray[tp.Any, np.dtype[np.bool_]]

    # check_type(v, h1)
