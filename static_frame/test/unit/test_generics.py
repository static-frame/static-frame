import numpy as np

import static_frame as sf


def test_generics_a() -> None:

    idx1: sf.Index[int] = sf.Index((2, 3))
    idx2: sf.Index[str] = sf.Index(('a', 'b'))

    def run(idx: sf.Index[int]) -> None:
        pass

    run(idx1)
    run(idx2)

    dt1: np.dtype[str] = idx1.dtype

    # ih1: sf.IndexHierarchy[int, bool] = sf.IndexHierarchy.from_product((1, 2), (True, False))


