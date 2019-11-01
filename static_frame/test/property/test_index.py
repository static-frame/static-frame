
import typing as tp
import unittest

import numpy as np

# from hypothesis import strategies as st
from hypothesis import given  # type: ignore

from static_frame.test.property.strategies import get_labels

from static_frame.test.test_case import TestCase

from static_frame import Index
from static_frame import IndexGO
# from static_frame import Series
# from static_frame import Frame
# from static_frame import FrameGO
# from static_frame import TypeBlocks
# from static_frame import Display



class TestUnit(TestCase):


    @given(get_labels())  # type: ignore
    def test_index_values_len(self, values: tp.Sequence[tp.Hashable]) -> None:

        def property_values(cls: tp.Type[Index], values: tp.Sequence[tp.Hashable]) -> None:
            #Property: that the length of the index is the length of the (unique) values.
            index = cls(values)
            self.assertEqual(len(index), len(values))
            self.assertEqual(len(index.values), len(values))

        property_values(Index, values)
        property_values(IndexGO, values)

    @given(get_labels())  # type: ignore
    def test_index_values_list(self, values: tp.Sequence[tp.Hashable]) -> None:

        def property_values(cls: tp.Type[Index], values: tp.Iterable[tp.Hashable]) -> None:
            index = cls(values)
            # must cast both sides to the dtype, as some int to float conversions result in different floats
            self.assertAlmostEqualValues(index.values, np.array(values, dtype=index.values.dtype))

        property_values(Index, values)
        property_values(IndexGO, values)


    @given(get_labels())  # type: ignore
    def test_index_loc_to_iloc_element(self, values: tp.Sequence[tp.Hashable]) -> None:

        def property_loc_to_iloc_element(cls: tp.Type[Index], values: tp.Iterable[tp.Hashable]) -> None:
            index = cls(values)
            for i, v in enumerate(values):
                self.assertEqual(index.loc_to_iloc(v), i)

        property_loc_to_iloc_element(Index, values)
        property_loc_to_iloc_element(IndexGO, values)

    @given(get_labels(min_size=1))  # type: ignore
    def test_index_loc_to_iloc_slice(self, values: tp.Sequence[tp.Hashable]) -> None:

        def property_loc_to_iloc_slice(cls: tp.Type[Index], values: tp.Iterable[tp.Hashable]) -> None:
            # Property: that the key translates to the appropriate slice.
            index = cls(values)
            for i, v in enumerate(values):
                # insure that we get teh same slice going through loc that we would get by direct iloc
                if v is None:
                    self.assertEqual(index.loc_to_iloc(slice(v, None)), slice(None))
                else:
                    self.assertEqual(index.loc_to_iloc(slice(v, None)), slice(i, None))  # type: ignore  # https://github.com/python/typeshed/pull/3024

        property_loc_to_iloc_slice(Index, values)
        property_loc_to_iloc_slice(IndexGO, values)



    @given(get_labels(min_size=2))  # type: ignore
    def test_index_go_append(self, values: tp.Sequence[tp.Hashable]) -> None:

        index = IndexGO(values[:-1])
        length_start = len(index)
        index.append(values[-1])
        length_end = len(index)
        self.assertEqual(length_start + 1, length_end)



if __name__ == '__main__':
    unittest.main()
