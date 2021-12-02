import unittest

import numpy as np

from static_frame.test.test_case import TestCase
from static_frame.core.frame import Frame
from static_frame.test.test_case import temp_file

class TestUnit(TestCase):

    def test_frame_to_npz_a(self) -> None:
        # this forced a RuntimeError: File size unexpectedly exceeded ZIP64 limit error until the `force_zip64` parameter was added
        a1 = np.arange(15_000 * 18_000).reshape(15_000, 18_000)
        f1 = Frame(a1)

        with temp_file('.npz') as fp:
            f1.to_npz(fp)
            # f2 = Frame.from_npz(fp)
            # f1.equals(f2, compare_dtype=True, compare_class=True, compare_name=True)



if __name__ == '__main__':
    unittest.main()

