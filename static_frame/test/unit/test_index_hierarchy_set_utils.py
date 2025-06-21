from __future__ import annotations

import unittest
from string import ascii_letters

import numpy as np

from static_frame.core.index_base import IndexBase
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.index_hierarchy_set_utils import (
    index_hierarchy_difference,
    index_hierarchy_intersection,
    index_hierarchy_union,
)
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):
    def test_index_hierarchy_union(self) -> None:
        """
        NOTE: This test only exists to prove that the new union function returns the
        same result as the old one.
        """
        ih = IndexHierarchy.from_product(tuple(ascii_letters), range(100), [True, False])

        size = len(ih) // 100
        half = size // 2

        indices = []
        for i in range(100):
            if i == 0:
                sl = slice(0, size * (i + 1) + half)
            elif i == 100 - 1:
                sl = slice(size * i - half, None)
            else:
                sl = slice(size * i - half, size * (i + 1) + half)

            indices.append(ih.iloc[sl])

        # 1.51 s ± 63.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        expected = IndexBase.union(*indices).sort()  # type: ignore

        # 17.4 ms ± 162 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        actual = index_hierarchy_union(*indices).sort()

        self.assertTrue(
            actual.equals(expected.sort()),
            msg=(expected.rename('expected'), actual.rename('actual')),
        )

    def test_index_hierarchy_intersection(self) -> None:
        """
        NOTE: This test only exists to prove that the new intersection function returns the
        same result as the old one.
        """
        ih = IndexHierarchy.from_product(tuple(ascii_letters), range(100), [True, False])

        indices = []
        for i in range(100):
            indices.append(ih.iloc[i:])

        # 4.44 s ± 66.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        expected = IndexBase.intersection(*indices).sort()  # type: ignore

        # 219 ms ± 1.34 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        actual = index_hierarchy_intersection(*indices).sort()

        self.assertTrue(
            actual.equals(expected),
            msg=(expected.rename('expected'), actual.rename('actual')),
        )

    def test_index_hierarchy_difference(self) -> None:
        """
        NOTE: This test only exists to prove that the new difference function returns the
        same result as the old one.
        """
        ih = IndexHierarchy.from_product(tuple(ascii_letters), range(100), [True, False])

        size = len(ih) // 100
        half = size // 2

        indices = []
        for i in range(100):
            if i == 0:
                sl = slice(0, size * (i + 1) + half)
            elif i == 100 - 1:
                sl = slice(size * i - half, None)
            else:
                sl = slice(size * i - half, size * (i + 1) + half)

            indices.append(ih.iloc[sl])

        # 46.5 ms ± 3.35 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
        expected = IndexBase.difference(*indices).sort()  # type: ignore

        # 23.2 ms ± 564 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
        actual = index_hierarchy_difference(*indices).sort()

        self.assertTrue(
            actual.equals(expected),
            msg=(expected.rename('expected'), actual.rename('actual')),
        )

    def test_index_hierarchy_ops_with_object_mappings(self) -> None:
        # Addresses issue found in #1073
        ih1 = IndexHierarchy.from_labels(np.arange(0, 1_300).reshape(130, 10))
        ih2 = IndexHierarchy.from_labels(np.arange(1_300, 2_600).reshape(130, 10))

        ih3 = ih1.intersection(ih2)
        ih4 = ih1.union(ih2)
        ih5 = ih1.difference(ih2)
        ih6 = ih2.intersection(ih1)
        ih7 = ih2.union(ih1)
        ih8 = ih2.difference(ih1)

        assert ih3.shape == (0, 10)
        assert ih4.shape == (260, 10)
        assert ih5.shape == (130, 10)
        assert ih6.shape == (0, 10)
        assert ih7.shape == (260, 10)
        assert ih8.shape == (130, 10)

        assert ih3.equals(ih6, compare_dtype=True, compare_name=True, compare_class=True)
        assert ih4.equals(ih7, compare_dtype=True, compare_name=True, compare_class=True)
        assert not ih5.equals(ih8)


if __name__ == '__main__':
    unittest.main()
