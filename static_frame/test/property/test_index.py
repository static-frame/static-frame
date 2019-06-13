
import unittest

import numpy as np

from hypothesis import strategies as st
from hypothesis import given
from hypothesis import example
from hypothesis import reproduce_failure

from static_frame.test.property.strategies import get_labels

from static_frame.test.test_case import TestCase

from static_frame import Index
from static_frame import IndexGO
from static_frame import Series
from static_frame import Frame
from static_frame import FrameGO
from static_frame import TypeBlocks
from static_frame import Display
from static_frame import mloc
from static_frame import DisplayConfig
from static_frame import IndexHierarchy
from static_frame import IndexHierarchyGO




class TestUnit(TestCase):


    @given(get_labels())
    def test_index_values_len(self, values):

        def property_values(cls, values):
            #Property: that the length of the index is the length of the (unique) values.
            index = cls(values)
            self.assertEqual(len(index), len(values))
            self.assertEqual(len(index.values), len(values))

        property_values(Index, values)
        property_values(IndexGO, values)

    @given(get_labels())
    def test_index_values_list(self, values):

        def property_values(cls, values):
            index = cls(values)
            # must cast both sides to the dtype, as some int to float conversions result in different floats
            self.assertAlmostEqualValues(index.values, np.array(values, dtype=index.values.dtype))

        property_values(Index, values)
        property_values(IndexGO, values)


    @given(get_labels())
    def test_index_loc_to_iloc_element(self, values):

        def property_loc_to_iloc_element(cls, values):
            index = cls(values)
            for i, v in enumerate(values):
                self.assertEqual(index.loc_to_iloc(v), i)

        property_loc_to_iloc_element(Index, values)
        property_loc_to_iloc_element(IndexGO, values)


    @given(get_labels())
    def test_index_loc_to_iloc_slice(self, values):

        def property_loc_to_iloc_slice(cls, values):
            # Property: that the key translates to the appropriate slice.
            index = cls(values)
            for i, v in enumerate(values):
                self.assertEqual(index.loc_to_iloc(slice(v, None)), slice(i, None))

        property_loc_to_iloc_slice(Index, values)
        property_loc_to_iloc_slice(IndexGO, values)


    @given(get_labels(min_size=1))
    def test_index_loc_to_iloc_slice(self, values):

        def property_loc_to_iloc_slice(cls, values):
            '''
            Property: that the key translates to the appropriate slice.
            '''
            index = cls(values)
            for i, v in enumerate(values):
                self.assertEqual(index.loc_to_iloc(slice(v, None)), slice(i, None))

        property_loc_to_iloc_slice(Index, values)
        property_loc_to_iloc_slice(IndexGO, values)



    @given(get_labels(min_size=2))
    def test_index_go_append(self, values):

        index = IndexGO(values[:-1])
        length_start = len(index)
        index.append(values[-1])
        length_end = len(index)
        self.assertEqual(length_start + 1, length_end)



if __name__ == '__main__':
    unittest.main()
