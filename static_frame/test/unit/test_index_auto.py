
import unittest

from static_frame.core.index import Index
from static_frame.core.index import IndexGO

from static_frame.test.test_case import TestCase
from static_frame.core.index_auto import IndexAutoFactory

class TestUnit(TestCase):

    def test_index_auto_factory_a(self) -> None:

        idx1 = IndexAutoFactory.from_optional_constructor(4,
                default_constructor=Index)
        self.assertEqual(idx1._map is None, True) #type: ignore
        self.assertEqual(len(idx1), 4)
        self.assertEqual(idx1.STATIC, True)

    def test_index_auto_factory_b(self) -> None:

        idx1 = IndexAutoFactory.from_optional_constructor(8,
                default_constructor=IndexGO)
        self.assertEqual(idx1._map is None, True) #type: ignore
        self.assertEqual(len(idx1), 8)
        self.assertEqual(idx1.STATIC, False)

        # go funcitonality
        assert isinstance(idx1, IndexGO)
        idx1.append(8)
        self.assertEqual(idx1._map is None, True)
        self.assertEqual(len(idx1), 9)


    def test_index_auto_factory_c(self) -> None:

        idx1 = IndexAutoFactory.from_optional_constructor(5,
                default_constructor=IndexGO,
                explicit_constructor=Index)
        # when using an alternate constructor, loc_is_iloc will not be set
        self.assertEqual(idx1._map is None, False) #type: ignore
        self.assertEqual(len(idx1), 5)
        self.assertEqual(idx1.STATIC, True)



if __name__ == '__main__':
    unittest.main()
