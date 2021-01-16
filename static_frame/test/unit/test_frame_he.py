

import unittest

from static_frame import FrameHE
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    def test_frame_he_slotted_a(self) -> None:

        f1 = FrameHE.from_element(1, index=(1,2), columns=(3,4,5))

        with self.assertRaises(AttributeError):
            f1.g = 30 #type: ignore #pylint: disable=E0237
        with self.assertRaises(AttributeError):
            f1.__dict__ #pylint: disable=W0104













if __name__ == '__main__':
    unittest.main()

