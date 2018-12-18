
import unittest

import numpy as np

from hypothesis import strategies as st
from hypothesis import given
from hypothesis import example

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


# index_integer = st.builds(Index, st.lists(st.integers(), unique=True))
list_integers = st.lists(st.integers(), unique=True)
list_floats = st.lists(st.floats(), unique=True)
list_text = st.lists(st.text(), unique=True)
list_mixed = st.lists(st.one_of(
        st.integers(), st.floats(), st.characters()), unique=True)

labels = st.one_of(list_mixed, list_integers, list_floats, list_text)


class TestUnit(TestCase):


    def _property_values(self, cls, values):
        '''
        Property: that the length of the index is the length of the (unique) values.
        '''
        index = cls(values)
        self.assertEqual(len(index), len(values))

    def _property_loc_to_iloc_element(self, cls, values):
        '''
        Property: that the key translates to the ordered position.
        '''
        index = cls(values)
        for i, v in enumerate(values):
            self.assertEqual(index.loc_to_iloc(v), i)

    def _property_loc_to_iloc_slice(self, cls, values):
        '''
        Property: that the key translates to the appropriate slice.
        '''
        # print(values)
        index = cls(values)
        for i, v in enumerate(values):
            self.assertEqual(index.loc_to_iloc(slice(v, None)), slice(i, None))


    @given(labels)
    def test_index_values(self, values):
        self._property_values(Index, values)
        self._property_values(IndexGO, values)


    @given(labels)
    def test_index_loc_to_iloc_element(self, values):
        self._property_loc_to_iloc_element(Index, values)
        self._property_loc_to_iloc_element(IndexGO, values)


    @given(labels)
    def test_index_loc_to_iloc_slice(self, values):
        self._property_loc_to_iloc_slice(Index, values)
        self._property_loc_to_iloc_slice(IndexGO, values)




if __name__ == '__main__':
    unittest.main()
