from __future__ import annotations

import numpy as np

from static_frame.core.fill_value_auto import FillValueAuto
from static_frame.core.util import EMPTY_TIMEDELTA, NAT, NAT_TD64
from static_frame.test.test_case import TestCase

# import frame_fixtures as ff


class TestUnit(TestCase):
    def test_temporal_constants_use_explicit_unit(self) -> None:
        self.assertEqual(np.datetime_data(NAT.dtype), ('Y', 1))
        self.assertEqual(np.datetime_data(NAT_TD64.dtype), ('ns', 1))
        self.assertEqual(np.datetime_data(EMPTY_TIMEDELTA.dtype), ('ns', 1))

    def test_fill_value_auto_a(self) -> None:
        fva = FillValueAuto.from_default()
        self.assertEqual(fva[np.dtype(int)], 0)
        self.assertEqual(fva[np.dtype(bool)], False)
        self.assertIs(fva[np.dtype(np.datetime64)], NAT)
        self.assertIs(fva[np.dtype(np.timedelta64)], NAT_TD64)
        self.assertEqual(fva[np.dtype('U4')], '')

    def test_fill_value_auto_b(self) -> None:
        fva = FillValueAuto(O=None)
        self.assertEqual(fva[np.dtype(object)], None)

        with self.assertRaises(RuntimeError):
            self.assertEqual(fva[np.dtype(int)], 0)


if __name__ == '__main__':
    import unittest

    unittest.main()
