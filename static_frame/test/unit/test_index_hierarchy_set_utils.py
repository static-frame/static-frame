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

        # Traditional approach: ~2.25 seconds
        expected = indices[0]
        for index in indices[1:]:
            # TODO: This is a bug!
            with pytest.raises((NotImplementedError, TypeError)):
                expected |= index

            expected = expected.union(index)

        # New approach: 28.8 ms
        result = index_hierarchy_union(*indices)

        assert result.sort().equals(expected)


if __name__ == '__main__':
    unittest.main()
