import unittest

import numpy as np
from automap import FrozenAutoMap  # pylint: disable=E0611

from static_frame.core.util import MemoryMeasurements
from static_frame.test.test_case import TestCase

_nested_sizable_elements = MemoryMeasurements._nested_sizable_elements
_unsized_children = MemoryMeasurements._unsized_children

class TestUnit(TestCase):
    #---------------------------------------------------------------------------
    # MemoryMeasurements._nested_sizable_elements

    def test_nested_sizable_elements_none(self) -> None:
        self.assertEqual(tuple(_nested_sizable_elements(None, seen=set())), (None,))

    def test_nested_sizable_elements_int(self) -> None:
        self.assertEqual(tuple(_nested_sizable_elements(3, seen=set())), (3,))

    def test_nested_sizable_elements_str(self) -> None:
        self.assertEqual(tuple(_nested_sizable_elements('abc', seen=set())), ('abc',))

    def test_nested_sizable_elements_float(self) -> None:
        self.assertEqual(tuple(_nested_sizable_elements(4.5, seen=set())), (4.5,))

    def test_nested_sizable_elements_numpy_array_int(self) -> None:
        obj = np.array([2, 3, 4], dtype=np.int64)
        self.assertEqual(tuple(_nested_sizable_elements(obj, seen=set())), (obj,))

    def test_nested_sizable_elements_numpy_array_object(self) -> None:
        obj = np.array([2, 'a', 4], dtype=np.object)
        self.assertEqual(tuple(_nested_sizable_elements(obj, seen=set())), (2, 'a', 4, obj))

    def test_nested_sizable_elements_tuple(self) -> None:
        obj = (2, 3, 4)
        self.assertEqual(tuple(_nested_sizable_elements(obj, seen=set())), (2, 3, 4, obj))

    def test_nested_sizable_elements_tuple_nested(self) -> None:
        obj = (2, ('a', 'b', ('c', 'd')), 4)
        self.assertEqual(
            tuple(_nested_sizable_elements(obj, seen=set())),
            (2, 'a', 'b', 'c', 'd', ('c', 'd'), ('a', 'b', ('c', 'd')), 4, obj)
        )

    def test_nested_sizable_elements_tuple_nested_existing_seen(self) -> None:
        cd = ('c', 'd')
        obj = (2, ('a', 'b', cd), 4)
        seen = set(id(el) for el in ('a', 'c', 'd', cd))
        self.assertEqual(
            tuple(_nested_sizable_elements(obj, seen=seen)),
            (2, 'b', ('a', 'b', ('c', 'd')), 4, obj)
        )

    def test_nested_sizable_elements_list(self) -> None:
        obj = [2, 3, 4]
        self.assertEqual(tuple(_nested_sizable_elements(obj, seen=set())), (2, 3, 4, obj))

    def test_nested_sizable_elements_list_nested(self) -> None:
        obj = [2, ('a', 'b', ('c', 'd')), 4]
        self.assertEqual(
            tuple(_nested_sizable_elements(obj, seen=set())),
            (2, 'a', 'b', 'c', 'd', ('c', 'd'), ('a', 'b', ('c', 'd')), 4, obj)
        )

    def test_nested_sizable_elements_set(self) -> None:
        obj = set((2, 3, 4))
        self.assertEqual(tuple(_nested_sizable_elements(obj, seen=set())), (2, 3, 4, obj))

    def test_nested_sizable_elements_frozenset(self) -> None:
        obj = frozenset((2, 3, 4))
        self.assertEqual(tuple(_nested_sizable_elements(obj, seen=set())), (2, 3, 4, obj))

    def test_nested_sizable_elements_dict(self) -> None:
        obj = { 'a': 2, 'b': 3, 'c': 4 }
        self.assertEqual(tuple(_nested_sizable_elements(obj, seen=set())), ('a', 2, 'b', 3, 'c', 4, obj))

    def test_nested_sizable_elements_frozenautomap(self) -> None:
        obj = FrozenAutoMap([2, 3, 4])
        self.assertEqual(tuple(_nested_sizable_elements(obj, seen=set())), (obj,))
    
    #---------------------------------------------------------------------------
    # MemoryMeasurements._unsized_children

    def test_unsized_children_none(self) -> None:
        self.assertEqual(tuple(_unsized_children(None)), ())

    def test_unsized_children_int(self) -> None:
        self.assertEqual(tuple(_unsized_children(3)), ())

    def test_unsized_children_str(self) -> None:
        self.assertEqual(tuple(_unsized_children('abc')), ())

    def test_unsized_children_float(self) -> None:
        self.assertEqual(tuple(_unsized_children(4.5)), ())

    def test_unsized_children_numpy_array_int(self) -> None:
        obj = np.array([2, 3, 4], dtype=np.int64)
        self.assertEqual(tuple(_unsized_children(obj)), ())

    def test_unsized_children_numpy_array_object(self) -> None:
        obj = np.array([2, 'a', 4], dtype=np.object)
        self.assertEqual(tuple(_unsized_children(obj)), (2, 'a', 4))

    def test_unsized_children_tuple(self) -> None:
        obj = (2, 3, 4)
        self.assertEqual(tuple(_unsized_children(obj)), (2, 3, 4))

    def test_unsized_children_tuple_nested(self) -> None:
        obj = (2, ('a', 'b', ('c', 'd')), 4)
        self.assertEqual(tuple(_unsized_children(obj)), (2, ('a', 'b', ('c', 'd')), 4))

    def test_unsized_children_list(self) -> None:
        obj = [2, 3, 4]
        self.assertEqual(tuple(_unsized_children(obj)), (2, 3, 4))

    def test_unsized_children_list_nested(self) -> None:
        obj = [2, ('a', 'b', ('c', 'd')), 4]
        self.assertEqual(tuple(_unsized_children(obj)), (2, ('a', 'b', ('c', 'd')), 4))

    def test_unsized_children_set(self) -> None:
        obj = set((2, 3, 4))
        self.assertEqual(tuple(_unsized_children(obj)), (2, 3, 4))

    def test_unsized_children_frozenset(self) -> None:
        obj = frozenset((2, 3, 4))
        self.assertEqual(tuple(_unsized_children(obj)), (2, 3, 4))

    def test_unsized_children_dict(self) -> None:
        obj = { 'a': 2, 'b': 3, 'c': 4 }
        self.assertEqual(tuple(_unsized_children(obj)), ('a', 2, 'b', 3, 'c', 4))

    def test_unsized_children_frozenautomap(self) -> None:
        obj = FrozenAutoMap([2, 3, 4])
        self.assertEqual(tuple(_unsized_children(obj)), ())


if __name__ == '__main__':
    unittest.main()
