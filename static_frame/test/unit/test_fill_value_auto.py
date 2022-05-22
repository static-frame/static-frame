import numpy as np
# import frame_fixtures as ff

from static_frame.test.test_case import TestCase
from static_frame.core.fill_value_auto import FillValueAuto
from static_frame.core.util import NAT
from static_frame.core.util import NAT_TD64

class TestUnit(TestCase):

    def test_fill_value_auto_a(self) -> None:

        fva = FillValueAuto()
        self.assertEqual(fva(np.dtype(int)), 0)
        self.assertEqual(fva(np.dtype(bool)), False)
        self.assertIs(fva(np.dtype(np.datetime64)), NAT)
        self.assertIs(fva(np.dtype(np.timedelta64)), NAT_TD64)
        self.assertEqual(fva(np.dtype('U4')), '')

        # import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    import unittest
    unittest.main()
