import unittest

import numpy as np
from automap import FrozenAutoMap

from static_frame.core.util import _nested_sizable_elements
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):
    def test_all_nested_elements_none(self) -> None:
        self.assertEqual(tuple(_nested_sizable_elements(None)), (None,))

    def test_all_nested_elements_int(self) -> None:
        self.assertEqual(tuple(_nested_sizable_elements(3)), (3,))

    def test_all_nested_elements_str(self) -> None:
        self.assertEqual(tuple(_nested_sizable_elements('abc')), ('abc',))

    def test_all_nested_elements_float(self) -> None:
        self.assertEqual(tuple(_nested_sizable_elements(4.5)), (4.5,))

    def test_all_nested_elements_numpy_array_int(self) -> None:
        obj = np.array([2, 3, 4], dtype=np.int64)
        self.assertEqual(tuple(_nested_sizable_elements(obj)), (obj,))

    def test_all_nested_elements_numpy_array_object(self) -> None:
        obj = np.array([2, 'a', 4], dtype=np.object)
        self.assertEqual(tuple(_nested_sizable_elements(obj)), (2, 'a', 4, obj))

    def test_all_nested_elements_tuple(self) -> None:
        obj = (2, 3, 4)
        self.assertEqual(tuple(_nested_sizable_elements(obj)), (2, 3, 4, obj))

    def test_all_nested_elements_tuple_nested(self) -> None:
        obj = (2, ('a', 'b', ('c', 'd')), 4)
        self.assertEqual(
            tuple(_nested_sizable_elements(obj)),
            (2, 'a', 'b', 'c', 'd', ('c', 'd'), ('a', 'b', ('c', 'd')), 4, obj)
        )

    def test_all_nested_elements_tuple_nested_existing_seen(self) -> None:
        cd = ('c', 'd')
        obj = (2, ('a', 'b', cd), 4)
        seen = set(id(el) for el in ('a', 'c', 'd', cd))
        self.assertEqual(
            tuple(_nested_sizable_elements(obj, seen=seen)),
            (2, 'b', ('a', 'b', ('c', 'd')), 4, obj)
        )

    def test_all_nested_elements_list(self) -> None:
        obj = [2, 3, 4]
        self.assertEqual(tuple(_nested_sizable_elements(obj)), (2, 3, 4, obj))

    def test_all_nested_elements_list_nested(self) -> None:
        obj = [2, ('a', 'b', ('c', 'd')), 4]
        self.assertEqual(
            tuple(_nested_sizable_elements(obj)),
            (2, 'a', 'b', 'c', 'd', ('c', 'd'), ('a', 'b', ('c', 'd')), 4, obj)
        )

    def test_all_nested_elements_set(self) -> None:
        obj = set((2, 3, 4))
        self.assertEqual(tuple(_nested_sizable_elements(obj)), (2, 3, 4, obj))

    def test_all_nested_elements_frozenset(self) -> None:
        obj = frozenset((2, 3, 4))
        self.assertEqual(tuple(_nested_sizable_elements(obj)), (2, 3, 4, obj))

    def test_all_nested_elements_dict(self) -> None:
        obj = { 'a': 2, 'b': 3, 'c': 4 }
        self.assertEqual(tuple(_nested_sizable_elements(obj)), ('a', 2, 'b', 3, 'c', 4, obj))

    def test_all_nested_elements_frozenautomap(self) -> None:
        obj = FrozenAutoMap([2, 3, 4])
        self.assertEqual(tuple(_nested_sizable_elements(obj)), (obj,))


if __name__ == '__main__':
    unittest.main()