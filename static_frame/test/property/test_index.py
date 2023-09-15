import numpy as np
import typing_extensions as tp
from arraykit import isna_element
from hypothesis import given

from static_frame import Index
from static_frame import IndexGO
from static_frame.core.util import TLabel
from static_frame.test.property.strategies import get_index_any
from static_frame.test.property.strategies import get_labels
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    @given(get_labels())
    def test_index_values_len(self, values: tp.Sequence[TLabel]) -> None:

        def property_values(cls: tp.Type[Index], values: tp.Sequence[TLabel]) -> None:
            #Property: that the length of the index is the length of the (unique) values.
            index = cls(values)
            self.assertEqual(len(index), len(values))
            self.assertEqual(len(index.values), len(values))

        property_values(Index, values)
        property_values(IndexGO, values)

    @given(get_labels())
    def test_index_values_list(self, values: tp.Sequence[TLabel]) -> None:

        def property_values(cls: tp.Type[Index], values: tp.Iterable[TLabel]) -> None:
            index = cls(values)
            # must cast both sides to the dtype, as some int to float conversions result in different floats
            self.assertAlmostEqualValues(index.values, np.array(values, dtype=index.values.dtype))

        property_values(Index, values)
        property_values(IndexGO, values)

    @given(get_labels())
    def test_index_loc_to_iloc_element(self, values: tp.Sequence[TLabel]) -> None:

        def property_loc_to_iloc_element(cls: tp.Type[Index], values: tp.Iterable[TLabel]) -> None:
            index = cls(values)
            for i, v in enumerate(values):
                self.assertEqual(index._loc_to_iloc(v), i)

        property_loc_to_iloc_element(Index, values)
        property_loc_to_iloc_element(IndexGO, values)

    @given(get_labels(min_size=1))
    def test_index_loc_to_iloc_slice(self, values: tp.Sequence[TLabel]) -> None:

        def property_loc_to_iloc_slice(cls: tp.Type[Index], values: tp.Iterable[TLabel]) -> None:
            # Property: that the key translates to the appropriate slice.
            index = cls(values)
            for i, v in enumerate(values):
                # insure that we get teh same slice going through loc that we would get by direct iloc
                if v is None:
                    self.assertEqual(index._loc_to_iloc(slice(v, None)), slice(None))
                else:
                    self.assertEqual(index._loc_to_iloc(slice(v, None)), slice(i, None))

        property_loc_to_iloc_slice(Index, values)
        property_loc_to_iloc_slice(IndexGO, values)

    @given(get_labels(min_size=2))
    def test_index_go_append(self, values: tp.Sequence[TLabel]) -> None:

        index = IndexGO(values[:-1])
        length_start = len(index)
        index.append(values[-1])
        length_end = len(index)
        self.assertEqual(length_start + 1, length_end)

    @given(get_labels(min_size=1))
    def test_index_isin(self, labels: tp.Sequence[TLabel]) -> None:
        index = Index(labels)
        self.assertTrue(index.isin((labels[0],))[0])

    #---------------------------------------------------------------------------

    @given(get_index_any())
    def test_index_display(self, index: Index) -> None:

        d1 = index.display()
        self.assertTrue(len(d1) > 0)

        d2 = index.display_tall()
        self.assertTrue(len(d2) > 0)

        d3 = index.display_wide()
        self.assertTrue(len(d3) > 0)

    @given(get_index_any())
    def test_index_to_series(self, index: Index) -> None:
        s1 = index.to_series()
        self.assertEqual(len(s1), len(index))

    #---------------------------------------------------------------------------

    @given(get_index_any())
    def test_index_iloc_map_a(self, index: Index) -> None:
        # We can't handle NaNs
        if any(map(isna_element, index.values)):
            return

        np.random.seed(0)
        ilocs = np.arange(len(index)) # Need new array, will shuffle in-place
        np.random.shuffle(ilocs)

        other = tp.cast(Index, index.iloc[ilocs])

        expected = index.iter_label().apply(other._loc_to_iloc)
        post = index._index_iloc_map(other)
        self.assertEqual(post.tolist(), expected.tolist())


if __name__ == '__main__':
    import unittest
    unittest.main()
