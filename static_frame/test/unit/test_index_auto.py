
import unittest
import numpy as np  # type: ignore

from static_frame.core.index import Index
from static_frame.core.index import IndexGO

from static_frame.test.test_case import TestCase
from static_frame.core.index_auto import IndexAutoFactory

class TestUnit(TestCase):

    def test_index_auto_factory_a(self) -> None:

        idx1 = IndexAutoFactory.from_is_static(4, is_static=True)
        self.assertEqual(idx1._loc_is_iloc, True)
        self.assertEqual(len(idx1), 4)
        self.assertEqual(idx1.STATIC, True)

    def test_index_auto_factory_b(self) -> None:

        idx1 = IndexAutoFactory.from_is_static(8, is_static=False)
        self.assertEqual(idx1._loc_is_iloc, True)
        self.assertEqual(len(idx1), 8)
        self.assertEqual(idx1.STATIC, False)

        # go funcitonality
        assert isinstance(idx1, IndexGO)
        idx1.append(8)
        self.assertEqual(idx1._loc_is_iloc, True)
        self.assertEqual(len(idx1), 9)


    def test_index_auto_factory_c(self) -> None:

        idx1 = IndexAutoFactory.from_constructor(5, constructor=Index)
        # when using an alternate constructor, loc_is_iloc will not be set
        self.assertEqual(idx1._loc_is_iloc, False)
        self.assertEqual(len(idx1), 5)
        self.assertEqual(idx1.STATIC, True)



if __name__ == '__main__':
    unittest.main()
