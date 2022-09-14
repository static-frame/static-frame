import unittest
from sys import getsizeof

import numpy as np
from automap import FrozenAutoMap

from static_frame.core.util import all_nested_elements
from static_frame.core.util import get_unsized_children_iter
from static_frame.core.util import getsizeof_recursive
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):
    # get_unsized_children
    #------------------------------------------------------------------------------------
    def test_unsized_children_none(self) -> None:
        self.assertEqual(tuple(get_unsized_children_iter(None)), ())

    def test_unsized_children_int(self) -> None:
        self.assertEqual(tuple(get_unsized_children_iter(3)), ())

    def test_unsized_children_str(self) -> None:
        self.assertEqual(tuple(get_unsized_children_iter('abc')), ())

    def test_unsized_children_float(self) -> None:
        self.assertEqual(tuple(get_unsized_children_iter(4.5)), ())

    def test_unsized_children_numpy_array_int(self) -> None:
        obj = np.array([2, 3, 4], dtype=np.int64)
        self.assertEqual(tuple(get_unsized_children_iter(obj)), ())

    def test_unsized_children_numpy_array_object(self) -> None:
        obj = np.array([2, 'a', 4], dtype=np.object)
        self.assertEqual(tuple(get_unsized_children_iter(obj)), (2, 'a', 4))

    def test_unsized_children_tuple(self) -> None:
        obj = (2, 3, 4)
        self.assertEqual(tuple(get_unsized_children_iter(obj)), (2, 3, 4))

    def test_unsized_children_tuple_nested(self) -> None:
        obj = (2, ('a', 'b', ('c', 'd')), 4)
        self.assertEqual(tuple(get_unsized_children_iter(obj)), (2, ('a', 'b', ('c', 'd')), 4))

    def test_unsized_children_list(self) -> None:
        obj = [2, 3, 4]
        self.assertEqual(tuple(get_unsized_children_iter(obj)), (2, 3, 4))

    def test_unsized_children_list_nested(self) -> None:
        obj = [2, ('a', 'b', ('c', 'd')), 4]
        self.assertEqual(tuple(get_unsized_children_iter(obj)), (2, ('a', 'b', ('c', 'd')), 4))

    def test_unsized_children_set(self) -> None:
        obj = set((2, 3, 4))
        self.assertEqual(tuple(get_unsized_children_iter(obj)), (2, 3, 4))

    def test_unsized_children_frozenset(self) -> None:
        obj = frozenset((2, 3, 4))
        self.assertEqual(tuple(get_unsized_children_iter(obj)), (2, 3, 4))

    def test_unsized_children_dict(self) -> None:
        obj = { 'a': 2, 'b': 3, 'c': 4 }
        self.assertEqual(tuple(get_unsized_children_iter(obj)), ('a', 2, 'b', 3, 'c', 4))

    def test_unsized_children_frozenautomap(self) -> None:
        obj = FrozenAutoMap([2, 3, 4])
        self.assertEqual(tuple(get_unsized_children_iter(obj)), ())

    # all_nested_elements
    #------------------------------------------------------------------------------------
    def test_all_nested_elements_none(self) -> None:
        self.assertEqual(tuple(all_nested_elements(None)), (None,))

    def test_all_nested_elements_int(self) -> None:
        self.assertEqual(tuple(all_nested_elements(3)), (3,))

    def test_all_nested_elements_str(self) -> None:
        self.assertEqual(tuple(all_nested_elements('abc')), ('abc',))

    def test_all_nested_elements_float(self) -> None:
        self.assertEqual(tuple(all_nested_elements(4.5)), (4.5,))

    def test_all_nested_elements_numpy_array_int(self) -> None:
        obj = np.array([2, 3, 4], dtype=np.int64)
        self.assertEqual(tuple(all_nested_elements(obj)), (obj,))

    def test_all_nested_elements_numpy_array_object(self) -> None:
        obj = np.array([2, 'a', 4], dtype=np.object)
        self.assertEqual(tuple(all_nested_elements(obj)), (2, 'a', 4, obj))

    def test_all_nested_elements_tuple(self) -> None:
        obj = (2, 3, 4)
        self.assertEqual(tuple(all_nested_elements(obj)), (2, 3, 4, obj))

    def test_all_nested_elements_tuple_nested(self) -> None:
        obj = (2, ('a', 'b', ('c', 'd')), 4)
        self.assertEqual(
            tuple(all_nested_elements(obj)),
            (2, 'a', 'b', 'c', 'd', ('c', 'd'), ('a', 'b', ('c', 'd')), 4, obj)
        )

    def test_all_nested_elements_tuple_nested_existing_seen(self) -> None:
        cd = ('c', 'd')
        obj = (2, ('a', 'b', cd), 4)
        seen = set(id(el) for el in ('a', 'c', 'd', cd))
        self.assertEqual(
            tuple(all_nested_elements(obj, seen=seen)),
            (2, 'b', ('a', 'b', ('c', 'd')), 4, obj)
        )

    def test_all_nested_elements_list(self) -> None:
        obj = [2, 3, 4]
        self.assertEqual(tuple(all_nested_elements(obj)), (2, 3, 4, obj))

    def test_all_nested_elements_list_nested(self) -> None:
        obj = [2, ('a', 'b', ('c', 'd')), 4]
        self.assertEqual(
            tuple(all_nested_elements(obj)),
            (2, 'a', 'b', 'c', 'd', ('c', 'd'), ('a', 'b', ('c', 'd')), 4, obj)
        )

    def test_all_nested_elements_set(self) -> None:
        obj = set((2, 3, 4))
        self.assertEqual(tuple(all_nested_elements(obj)), (2, 3, 4, obj))

    def test_all_nested_elements_frozenset(self) -> None:
        obj = frozenset((2, 3, 4))
        self.assertEqual(tuple(all_nested_elements(obj)), (2, 3, 4, obj))

    def test_all_nested_elements_dict(self) -> None:
        obj = { 'a': 2, 'b': 3, 'c': 4 }
        self.assertEqual(tuple(all_nested_elements(obj)), ('a', 2, 'b', 3, 'c', 4, obj))

    def test_all_nested_elements_frozenautomap(self) -> None:
        obj = FrozenAutoMap([2, 3, 4])
        self.assertEqual(tuple(all_nested_elements(obj)), (obj,))

    # getsizeof_recursive
    #------------------------------------------------------------------------------------
    def test_int(self) -> None:
        obj = 2
        self.assertEqual(getsizeof_recursive(obj), getsizeof(2))

    def test_set(self) -> None:
        obj = set(['a', 'b', 4])
        self.assertEqual(
            getsizeof_recursive(obj),
            sum(getsizeof(e) for e in (
                'a', 'b', 4,
                set(['a', 'b', 4]),
            ))
        )

    def test_frozenset(self) -> None:
        obj = frozenset(['a', 'b', 4])
        self.assertEqual(
            getsizeof_recursive(obj),
            sum(getsizeof(e) for e in (
                'a', 'b', 4,
                frozenset(['a', 'b', 4]),
            ))
        )

    def test_np_array(self) -> None:
        obj = np.arange(3)
        self.assertEqual(
            getsizeof_recursive(obj),
            getsizeof(np.arange(3))
        )

    def test_frozenautomap(self) -> None:
        obj = FrozenAutoMap(['a', 'b', 'c'])
        self.assertEqual(
            getsizeof_recursive(obj),
            getsizeof(FrozenAutoMap(['a', 'b', 'c']))
        )

    def test_dict(self) -> None:
        obj = { 'a': 2, 'b': 3, 'c': (4, 5) }
        self.assertEqual(
            getsizeof_recursive(obj),
            sum(getsizeof(e) for e in (
                'a', 2,
                'b', 3,
                'c', 4, 5, (4, 5),
                { 'a': 2, 'b': 3, 'c': (4, 5) },
            ))
        )

    def test_tuple(self) -> None:
        obj = (2, 3, 4)
        self.assertEqual(getsizeof_recursive(obj), sum(getsizeof(e) for e in (
            2, 3, 4,
            (2, 3, 4),
        )))

    def test_nested_tuple(self) -> None:
        obj = (2, 'b', (2, 3))
        self.assertEqual(getsizeof_recursive(obj), sum(getsizeof(e) for e in (
            2, 'b', 3,
            (2, 3),
            (2, 'b', (2, 3)),
        )))

    def test_predefined_seen(self) -> None:
        obj = (4, 5, (2, 8))
        seen = set([id(2), id(3)])
        self.assertEqual(getsizeof_recursive(obj, seen=seen), sum(getsizeof(e) for e in (
            4, 5,
            8,
            (2, 8),
            (4, 5, (2, 8))
        )))

    def test_larger_values_is_larger(self) -> None:
        a = ('a', 'b', 'c')
        b = ('abc', 'def', 'ghi')
        self.assertTrue(getsizeof_recursive(a) < getsizeof_recursive(b))

    def test_more_values_is_larger_a(self) -> None:
        a = ('a', 'b', 'c')
        b = ('a', 'b', 'c', 'd')
        self.assertTrue(getsizeof_recursive(a) < getsizeof_recursive(b))

    def test_more_values_is_larger_b(self) -> None:
        a = ('a', 'b', 'c')
        b = 'd'
        self.assertTrue(getsizeof_recursive([a]) < getsizeof_recursive([a, b]))

    def test_more_values_is_larger_nested_a(self) -> None:
        a = ('a', (2, (8, 9), 4), 'c')
        b = ('a', (2, (8, 9, 10), 4), 'c')
        self.assertTrue(getsizeof_recursive(a) < getsizeof_recursive(b))

    def test_more_values_is_larger_nested_b(self) -> None:
        a = np.array(['a', [2, (8, 9), 4], 'c'], dtype=object)
        b = np.array(['a', [2, (8, 9, 10), 4], 'c'], dtype=object)
        self.assertTrue(getsizeof_recursive(a) < getsizeof_recursive(b))


if __name__ == '__main__':
    unittest.main()