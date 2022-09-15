import unittest
from sys import getsizeof

import numpy as np
from automap import FrozenAutoMap  # pylint: disable=E0611

from static_frame.core.util import getsizeof_total
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):
    def test_int(self) -> None:
        obj = 2
        self.assertEqual(getsizeof_total(obj), getsizeof(2))

    def test_set(self) -> None:
        obj = set(['a', 'b', 4])
        self.assertEqual(
            getsizeof_total(obj),
            sum(getsizeof(e) for e in (
                'a', 'b', 4,
                set(['a', 'b', 4]),
            ))
        )

    def test_frozenset(self) -> None:
        obj = frozenset(['a', 'b', 4])
        self.assertEqual(
            getsizeof_total(obj),
            sum(getsizeof(e) for e in (
                'a', 'b', 4,
                frozenset(['a', 'b', 4]),
            ))
        )

    def test_np_array(self) -> None:
        obj = np.arange(3)
        self.assertEqual(
            getsizeof_total(obj),
            getsizeof(np.arange(3))
        )

    def test_frozenautomap(self) -> None:
        obj = FrozenAutoMap(['a', 'b', 'c'])
        self.assertEqual(
            getsizeof_total(obj),
            getsizeof(FrozenAutoMap(['a', 'b', 'c']))
        )

    def test_dict(self) -> None:
        obj = { 'a': 2, 'b': 3, 'c': (4, 5) }
        self.assertEqual(
            getsizeof_total(obj),
            sum(getsizeof(e) for e in (
                'a', 2,
                'b', 3,
                'c', 4, 5, (4, 5),
                { 'a': 2, 'b': 3, 'c': (4, 5) },
            ))
        )

    def test_tuple(self) -> None:
        obj = (2, 3, 4)
        self.assertEqual(getsizeof_total(obj), sum(getsizeof(e) for e in (
            2, 3, 4,
            (2, 3, 4),
        )))

    def test_nested_tuple(self) -> None:
        obj = (2, 'b', (2, 3))
        self.assertEqual(getsizeof_total(obj), sum(getsizeof(e) for e in (
            2, 'b', 3,
            (2, 3),
            (2, 'b', (2, 3)),
        )))

    def test_predefined_seen(self) -> None:
        obj = (4, 5, (2, 8))
        seen = set([id(2), id(3)])
        self.assertEqual(getsizeof_total(obj, seen=seen), sum(getsizeof(e) for e in (
            4, 5,
            8,
            (2, 8),
            (4, 5, (2, 8))
        )))

    # TODO: Add predefined_seen where it includes all elements already

    def test_larger_values_is_larger(self) -> None:
        a = ('a', 'b', 'c')
        b = ('abc', 'def', 'ghi')
        self.assertTrue(getsizeof_total(a) < getsizeof_total(b))

    def test_more_values_is_larger_a(self) -> None:
        a = ('a', 'b', 'c')
        b = ('a', 'b', 'c', 'd')
        self.assertTrue(getsizeof_total(a) < getsizeof_total(b))

    def test_more_values_is_larger_b(self) -> None:
        a = ('a', 'b', 'c')
        b = 'd'
        self.assertTrue(getsizeof_total([a]) < getsizeof_total([a, b]))

    def test_more_values_is_larger_nested_a(self) -> None:
        a = ('a', (2, (8, 9), 4), 'c')
        b = ('a', (2, (8, 9, 10), 4), 'c')
        self.assertTrue(getsizeof_total(a) < getsizeof_total(b))

    def test_more_values_is_larger_nested_b(self) -> None:
        a = np.array(['a', [2, (8, 9), 4], 'c'], dtype=object)
        b = np.array(['a', [2, (8, 9, 10), 4], 'c'], dtype=object)
        self.assertTrue(getsizeof_total(a) < getsizeof_total(b))


if __name__ == '__main__':
    unittest.main()
