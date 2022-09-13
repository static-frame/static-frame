import unittest
from sys import getsizeof

import numpy as np
from automap import FrozenAutoMap

from static_frame.core.util import getsizeof_recursive
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):
    def test_int(self) -> None:
        self.assertEqual(getsizeof_recursive([2]), getsizeof(2))

    def test_set(self) -> None:
        self.assertEqual(
            getsizeof_recursive([set(['a', 'b', 4])]),
            sum(getsizeof(e) for e in [
                'a', 'b', 4,
                set(['a', 'b', 4])
            ])
        )

    def test_frozenset(self) -> None:
        self.assertEqual(
            getsizeof_recursive([frozenset(['a', 'b', 4])]),
            sum(getsizeof(e) for e in [
                'a', 'b', 4,
                frozenset(['a', 'b', 4])
            ])
        )

    def test_np_array(self) -> None:
        self.assertEqual(
            getsizeof_recursive([np.arange(3)]),
            getsizeof(np.arange(3))
        )

    def test_frozenautomap(self) -> None:
        self.assertEqual(
            getsizeof_recursive([FrozenAutoMap(['a', 'b', 'c'])]),
            getsizeof(FrozenAutoMap(['a', 'b', 'c']))
        )

    def test_dict(self) -> None:
        self.assertEqual(
            getsizeof_recursive([{ 'a': 2, 'b': 3, 'c': (4, 5) }]),
            sum(getsizeof(e) for e in [
                'a', 2,
                'b', 3,
                'c', 4, 5, (4, 5),
                { 'a': 2, 'b': 3, 'c': (4, 5) }
            ])
        )

    def test_tuple(self) -> None:
        self.assertEqual(getsizeof_recursive([(2, 3, 4)]), sum(getsizeof(e) for e in [
            2, 3, 4,
            (2, 3, 4)
        ]))


    def test_nested_tuple(self) -> None:
        self.assertEqual(getsizeof_recursive([(2, 'b', (2, 3))]), sum(getsizeof(e) for e in [
            2, 'b', 3,
            (2, 3),
            (2, 'b', (2, 3))
        ]))

    def test_larger_values_is_larger(self) -> None:
        a = ('a', 'b', 'c')
        b = ('abc', 'def', 'ghi')
        self.assertTrue(getsizeof_recursive([a]) < getsizeof_recursive([b]))

    def test_more_values_is_larger_a(self) -> None:
        a = ('a', 'b', 'c')
        b = ('a', 'b', 'c', 'd')
        self.assertTrue(getsizeof_recursive([a]) < getsizeof_recursive([b]))

    def test_more_values_is_larger_b(self) -> None:
        a = ('a', 'b', 'c')
        b = 'd'
        self.assertTrue(getsizeof_recursive([a]) < getsizeof_recursive([a, b]))

    def test_more_values_is_larger_nested_a(self) -> None:
        a = ('a', (2, (8, 9), 4), 'c')
        b = ('a', (2, (8, 9, 10), 4), 'c')
        self.assertTrue(getsizeof_recursive([a]) < getsizeof_recursive([b]))

    def test_more_values_is_larger_nested_b(self) -> None:
        a = np.array(['a', [2, (8, 9), 4], 'c'], dtype=object)
        b = np.array(['a', [2, (8, 9, 10), 4], 'c'], dtype=object)
        self.assertTrue(getsizeof_recursive([a]) < getsizeof_recursive([b]))

if __name__ == '__main__':
    unittest.main()