from __future__ import annotations

import unittest
from string import ascii_letters

from static_frame.core.index_base import IndexBase
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.index_hierarchy_set_utils import index_hierarchy_difference
from static_frame.core.index_hierarchy_set_utils import index_hierarchy_intersection
from static_frame.core.index_hierarchy_set_utils import index_hierarchy_union
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    def test_index_hierarchy_union(self) -> None:
        '''
        NOTE: This test only exists to prove that the new union function returns the
        same result as the old one.
        '''
        ih = IndexHierarchy.from_product(tuple(ascii_letters), range(100), [True, False])

        size = len(ih) //  100
        half = size // 2

        indices = []
        for i in range(100):
            if i == 0:
                sl = slice(0, size*(i+1) + half)
            elif i == 100 - 1:
                sl = slice(size * i - half, None)
            else:
                sl = slice(size * i - half, size*(i+1) + half)

            indices.append(ih.iloc[sl])

        # 1.51 s ± 63.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        expected = IndexBase.union(*indices).sort() # pylint: disable=E1120

        # 17.4 ms ± 162 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        actual = index_hierarchy_union(*indices).sort()

        self.assertTrue(actual.equals(expected.sort()), msg=(expected.rename("expected"), actual.rename("actual")))

    def test_index_hierarchy_intersection(self) -> None:
        '''
        NOTE: This test only exists to prove that the new intersection function returns the
        same result as the old one.
        '''
        ih = IndexHierarchy.from_product(tuple(ascii_letters), range(100), [True, False])

        indices = []
        for i in range(100):
            indices.append(ih.iloc[i:])

        # 4.44 s ± 66.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        expected = IndexBase.intersection(*indices).sort() # pylint: disable=E1120

        # 219 ms ± 1.34 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        actual = index_hierarchy_intersection(*indices).sort()

        self.assertTrue(actual.equals(expected), msg=(expected.rename("expected"), actual.rename("actual")))

    def test_index_hierarchy_difference(self) -> None:
        '''
        NOTE: This test only exists to prove that the new difference function returns the
        same result as the old one.
        '''
        ih = IndexHierarchy.from_product(tuple(ascii_letters), range(100), [True, False])

        size = len(ih) //  100
        half = size // 2

        indices = []
        for i in range(100):
            if i == 0:
                sl = slice(0, size*(i+1) + half)
            elif i == 100 - 1:
                sl = slice(size * i - half, None)
            else:
                sl = slice(size * i - half, size*(i+1) + half)

            indices.append(ih.iloc[sl])

        # 46.5 ms ± 3.35 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
        expected = IndexBase.difference(*indices).sort() # pylint: disable=E1120

        # 23.2 ms ± 564 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
        actual = index_hierarchy_difference(*indices).sort()

        self.assertTrue(actual.equals(expected), msg=(expected.rename("expected"), actual.rename("actual")))


if __name__ == '__main__':
    unittest.main()
