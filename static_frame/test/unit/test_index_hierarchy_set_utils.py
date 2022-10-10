import unittest
import pytest
from string import ascii_letters

from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.index_hierarchy_set_utils import index_hierarchy_union
from static_frame.core.index_hierarchy_set_utils import index_hierarchy_difference
from static_frame.core.index_hierarchy_set_utils import index_hierarchy_intersection
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    def test_index_hierarchy_union(self) -> None:
        '''
        NOTE: This test will be unnecessary once `__or__` uses the new union function.

        You can delete this test once the migration completes.
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

        # Traditional approach: ~1.85 seconds
        expected = indices[0]
        for index in indices[1:]:
            # TODO: Is this a bug?
            with pytest.raises((NotImplementedError, TypeError)):
                expected |= index

            expected = expected.union(index)

        # New approach: 23.8 ms
        result = index_hierarchy_union(*indices).sort()

        self.assertTrue(result.equals(expected.sort()), msg=(expected.rename("expected"), result.rename("result")))

    def test_index_hierarchy_intersection(self) -> None:
        '''
        NOTE: This test will be unnecessary once `__and__` uses the new union function.

        You can delete this test once the migration completes.
        '''
        ih = IndexHierarchy.from_product(tuple(ascii_letters), range(100), [True, False])

        indices = []
        for i in range(100):
            indices.append(ih.iloc[i:])

        # Traditional approach: ~5.97 seconds
        expected = indices[0]
        for index in indices[1:]:
            # TODO: Is this a bug?
            with pytest.raises((NotImplementedError, TypeError)):
                expected &= index

            expected = expected.intersection(index)

        # New approach: 189 ms
        result = index_hierarchy_intersection(*indices).sort()

        self.assertTrue(result.equals(expected.sort()), msg=(expected.rename("expected"), result.rename("result")))

    def test_index_hierarchy_difference(self) -> None:
        '''
        NOTE: This test will be unnecessary once `difference` uses the new union function.

        You can delete this test once the migration completes.
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

        # Traditional approach: ~116 ms
        expected = indices[0]
        for index in indices[1:]:
            expected = expected.difference(index)

        # New approach: 21.5 ms
        result = index_hierarchy_difference(*indices).sort()

        self.assertTrue(result.equals(expected.sort()), msg=(expected.rename("expected"), result.rename("result")))


if __name__ == '__main__':
    unittest.main()
