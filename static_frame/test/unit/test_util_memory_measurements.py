import unittest

import numpy as np
from automap import FrozenAutoMap  # pylint: disable=E0611

from static_frame.core.util import MemoryMeasurements
from static_frame.test.test_case import TestCase

_unsized_children = MemoryMeasurements._unsized_children
_sizable_slot_attrs = MemoryMeasurements._sizable_slot_attrs
nested_sizable_elements = MemoryMeasurements.nested_sizable_elements

class TestUnit(TestCase):
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

    def test_unsized_children_numpy_array_object_complex_has_unique_ids(self) -> None:
        # See comment in MemoryMeasurements._unsized_children for more detail
        # on what this is testing and why it is needed
        obj = np.array([np.array([None, None, i // 2]) for i in range(10)])
        self.assertEqual(len(set(id(el) for el in _unsized_children(obj))), 10)

    #---------------------------------------------------------------------------
    # MemoryMeasurements._sizable_slot_attrs

    def test_sizable_slot_attrs_empty(self) -> None:
        class A:
            pass
        obj = A()
        self.assertEqual(tuple(_sizable_slot_attrs(obj)), ())

    def test_sizable_slot_attrs_simple(self) -> None:
        class A:
            __slots__ = (
                'apples',
                'bananas',
                'carrots',
                'dumplings',
                'eggs'
            )
            def __init__(self) -> None:
                self.apples = 'a'
                self.bananas = 'b'
                self.carrots = 'c'
                self.dumplings = 'd'
                self.eggs = 'e'
        obj = A()
        self.assertEqual(frozenset(_sizable_slot_attrs(obj)), frozenset(('a', 'b', 'c', 'd', 'e')))

    def test_sizable_slot_attrs_not_all_initialized(self) -> None:
        class A:
            __slots__ = (
                'apples',
                'bananas',
                'carrots',
                'dumplings',
                'eggs'
            )
            def __init__(self) -> None:
                self.apples = 'a'
                self.bananas = 'b'
                self.eggs = 'e'
        obj = A()
        self.assertEqual(frozenset(_sizable_slot_attrs(obj)), frozenset(('a', 'b', 'e')))

    def test_sizable_slot_attrs_inheritance_1_layer(self) -> None:
        class A:
            __slots__ = (
                'apples',
                'bananas',
            )
            def __init__(self) -> None:
                self.apples = 'a'
                self.bananas = 'b'
        class B(A):
            __slots__ = (
                'carrots',
                'dumplings',
                'eggs'
            )
            def __init__(self) -> None:
                super().__init__()
                self.carrots = 'c'
                self.dumplings = 'd'
                self.eggs = 'e'
        obj = B()
        self.assertEqual(frozenset(_sizable_slot_attrs(obj)), frozenset(('a', 'b', 'c', 'd', 'e')))

    def test_sizable_slot_attrs_inheritance_2_layers(self) -> None:
        class A:
            __slots__ = (
                'apples',
                'bananas',
            )
            def __init__(self) -> None:
                self.apples = 'a'
                self.bananas = 'b'
        class B(A):
            __slots__ = (
                'carrots',
                'dumplings',
            )
            def __init__(self) -> None:
                super().__init__()
                self.carrots = 'c'
                self.dumplings = 'd'
        class C(B):
            __slots__ = (
                'eggs',
            )
            def __init__(self) -> None:
                super().__init__()
                self.eggs = 'e'
        obj = C()
        sizables = frozenset(_sizable_slot_attrs(obj))
        self.assertEqual(sizables, frozenset(('a', 'b', 'c', 'd', 'e')))

    def test_sizable_slot_attrs_inheritance_multiple(self) -> None:
        class A:
            __slots__ = (
                'apples',
                'bananas',
                'carrots',
                'dumplings',
            )
            def __init__(self) -> None:
                self.apples = 'a'
                self.bananas = 'b'
                self.carrots = 'c'
                self.dumplings = 'd'
        class B:
            __slots__ = ()
        class C(A, B):
            __slots__ = (
                'eggs',
            )
            def __init__(self) -> None:
                super().__init__()
                self.eggs = 'e'
        obj = C()
        self.assertEqual(frozenset(_sizable_slot_attrs(obj)), frozenset(('a', 'b', 'c', 'd', 'e')))

    # NOTE: From https://docs.python.org/3/reference/datamodel.html#notes-on-using-slots
    # > Multiple inheritance with multiple slotted parent classes can be used, but only one
    # >   parent is allowed to have attributes created by slots (the other bases must have empty
    # >   slot layouts) - violations raise TypeError.

    #---------------------------------------------------------------------------
    # MemoryMeasurements.nested_sizable_elements

    def test_nested_sizable_elements_none(self) -> None:
        self.assertEqual(tuple(nested_sizable_elements(None, seen=set())), (None,))

    def test_nested_sizable_elements_int(self) -> None:
        self.assertEqual(tuple(nested_sizable_elements(3, seen=set())), (3,))

    def test_nested_sizable_elements_str(self) -> None:
        self.assertEqual(tuple(nested_sizable_elements('abc', seen=set())), ('abc',))

    def test_nested_sizable_elements_float(self) -> None:
        self.assertEqual(tuple(nested_sizable_elements(4.5, seen=set())), (4.5,))

    def test_nested_sizable_elements_numpy_array_int(self) -> None:
        obj = np.array([2, 3, 4], dtype=np.int64)
        self.assertEqual(tuple(nested_sizable_elements(obj, seen=set())), (obj,))

    def test_nested_sizable_elements_numpy_array_object(self) -> None:
        obj = np.array([2, 'a', 4], dtype=np.object)
        self.assertEqual(tuple(nested_sizable_elements(obj, seen=set())), (2, 'a', 4, obj))

    def test_nested_sizable_elements_tuple(self) -> None:
        obj = (2, 3, 4)
        self.assertEqual(tuple(nested_sizable_elements(obj, seen=set())), (2, 3, 4, obj))

    def test_nested_sizable_elements_tuple_nested(self) -> None:
        obj = (2, ('a', 'b', ('c', 'd')), 4)
        self.assertEqual(
            tuple(nested_sizable_elements(obj, seen=set())),
            (2, 'a', 'b', 'c', 'd', ('c', 'd'), ('a', 'b', ('c', 'd')), 4, obj)
        )

    def test_nested_sizable_elements_tuple_nested_existing_seen(self) -> None:
        cd = ('c', 'd')
        obj = (2, ('a', 'b', cd), 4)
        seen = set(id(el) for el in ('a', 'c', 'd', cd))
        self.assertEqual(
            tuple(nested_sizable_elements(obj, seen=seen)),
            (2, 'b', ('a', 'b', ('c', 'd')), 4, obj)
        )

    def test_nested_sizable_elements_list(self) -> None:
        obj = [2, 3, 4]
        self.assertEqual(tuple(nested_sizable_elements(obj, seen=set())), (2, 3, 4, obj))

    def test_nested_sizable_elements_list_nested(self) -> None:
        obj = [2, ('a', 'b', ('c', 'd')), 4]
        self.assertEqual(
            tuple(nested_sizable_elements(obj, seen=set())),
            (2, 'a', 'b', 'c', 'd', ('c', 'd'), ('a', 'b', ('c', 'd')), 4, obj)
        )

    def test_nested_sizable_elements_set(self) -> None:
        obj = set((2, 3, 4))
        self.assertEqual(tuple(nested_sizable_elements(obj, seen=set())), (2, 3, 4, obj))

    def test_nested_sizable_elements_frozenset(self) -> None:
        obj = frozenset((2, 3, 4))
        self.assertEqual(tuple(nested_sizable_elements(obj, seen=set())), (2, 3, 4, obj))

    def test_nested_sizable_elements_dict(self) -> None:
        obj = { 'a': 2, 'b': 3, 'c': 4 }
        self.assertEqual(tuple(nested_sizable_elements(obj, seen=set())), ('a', 2, 'b', 3, 'c', 4, obj))

    def test_nested_sizable_elements_frozenautomap(self) -> None:
        obj = FrozenAutoMap([2, 3, 4])
        self.assertEqual(tuple(nested_sizable_elements(obj, seen=set())), (obj,))


if __name__ == '__main__':
    unittest.main()
