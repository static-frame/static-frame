

import unittest

import frame_fixtures as ff
# import numpy as np

# from static_frame import Frame
# from static_frame import Series
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    def test_frame_via_re_a(self) -> None:
        f1 = ff.parse('s(3,3)|c(I,str)|i(I,str)|v(int)')



if __name__ == '__main__':
    unittest.main()

