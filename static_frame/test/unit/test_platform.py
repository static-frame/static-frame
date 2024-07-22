from __future__ import annotations

import static_frame as sf
from static_frame.test.test_case import TestCase
from static_frame.test.test_case import skip_no_hdf5


class TestUnit(TestCase):

    @skip_no_hdf5
    def test_platform_a(self) -> None:
        post = sf.Platform.to_series()
        self.assertTrue(post.shape[0] > 10)

        d = sf.Platform.display()
        self.assertTrue(len(str(d)) > 100)


if __name__ == '__main__':
    import unittest
    unittest.main()
