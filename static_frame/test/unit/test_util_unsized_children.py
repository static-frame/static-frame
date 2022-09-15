import unittest

import numpy as np
from automap import FrozenAutoMap  # pylint: disable=E0611

from static_frame.core.util import _unsized_children
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):
    def test_none(self) -> None:
        self.assertEqual(tuple(_unsized_children(None)), ())

    def test_int(self) -> None:
        self.assertEqual(tuple(_unsized_children(3)), ())

    def test_str(self) -> None:
        self.assertEqual(tuple(_unsized_children('abc')), ())

    def test_float(self) -> None:
        self.assertEqual(tuple(_unsized_children(4.5)), ())

    def test_numpy_array_int(self) -> None:
        obj = np.array([2, 3, 4], dtype=np.int64)
        self.assertEqual(tuple(_unsized_children(obj)), ())

    def test_numpy_array_object(self) -> None:
        obj = np.array([2, 'a', 4], dtype=np.object)
        self.assertEqual(tuple(_unsized_children(obj)), (2, 'a', 4))

    def test_tuple(self) -> None:
        obj = (2, 3, 4)
        self.assertEqual(tuple(_unsized_children(obj)), (2, 3, 4))

    def test_tuple_nested(self) -> None:
        obj = (2, ('a', 'b', ('c', 'd')), 4)
        self.assertEqual(tuple(_unsized_children(obj)), (2, ('a', 'b', ('c', 'd')), 4))

    def test_list(self) -> None:
        obj = [2, 3, 4]
        self.assertEqual(tuple(_unsized_children(obj)), (2, 3, 4))

    def test_list_nested(self) -> None:
        obj = [2, ('a', 'b', ('c', 'd')), 4]
        self.assertEqual(tuple(_unsized_children(obj)), (2, ('a', 'b', ('c', 'd')), 4))

    def test_set(self) -> None:
        obj = set((2, 3, 4))
        self.assertEqual(tuple(_unsized_children(obj)), (2, 3, 4))

    def test_frozenset(self) -> None:
        obj = frozenset((2, 3, 4))
        self.assertEqual(tuple(_unsized_children(obj)), (2, 3, 4))

    def test_dict(self) -> None:
        obj = { 'a': 2, 'b': 3, 'c': 4 }
        self.assertEqual(tuple(_unsized_children(obj)), ('a', 2, 'b', 3, 'c', 4))

    def test_frozenautomap(self) -> None:
        obj = FrozenAutoMap([2, 3, 4])
        self.assertEqual(tuple(_unsized_children(obj)), ())


if __name__ == '__main__':
    unittest.main()
