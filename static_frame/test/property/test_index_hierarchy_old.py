from hypothesis import given
from static_frame.test.test_case import TestCase

from static_frame.test.property.strategies import get_index_hierarchyold_any

from static_frame import IndexHierarchyOld


class TestUnit(TestCase):

    #---------------------------------------------------------------------------

    @given(get_index_hierarchyold_any())
    def test_index_display(self, ih: IndexHierarchyOld) -> None:

        d1 = ih.display()
        self.assertTrue(len(d1) > 0)

        d2 = ih.display_tall()
        self.assertTrue(len(d2) > 0)

        d3 = ih.display_wide()
        self.assertTrue(len(d3) > 0)

    @given(get_index_hierarchyold_any())
    def test_index_to_frame(self, ih: IndexHierarchyOld) -> None:
        f1 = ih.to_frame()
        self.assertEqual(f1.shape, ih.shape)
