from __future__ import annotations

import numpy as np

from static_frame.core.index_base import IndexBase
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):
    def test_index_base_slotted_a(self) -> None:
        idx1 = IndexBase()

        with self.assertRaises(AttributeError):
            idx1.g = 30  # type: ignore
        with self.assertRaises(AttributeError):
            idx1.__dict__

    def test_index_base_not_implemented(self) -> None:
        idx1 = IndexBase()

        with self.assertRaises(NotImplementedError):
            idx1._ufunc_axis_skipna(
                axis=0,
                skipna=False,
                ufunc=np.sum,
                ufunc_skipna=np.nansum,
                composable=True,
                dtypes=(),
                size_one_unity=True,
            )

        with self.assertRaises(NotImplementedError):
            idx1._update_array_cache()

        with self.assertRaises(NotImplementedError):
            idx1.copy()

        with self.assertRaises(NotImplementedError):
            idx1.copy()

        with self.assertRaises(NotImplementedError):
            idx1.display()

        with self.assertRaises(NotImplementedError):
            idx1.from_labels(())


if __name__ == '__main__':
    import unittest

    unittest.main()
