from __future__ import annotations

import unittest
from sys import getsizeof

import frame_fixtures as ff
import numpy as np
import typing_extensions as tp
from arraymap import FrozenAutoMap  # pylint: disable=E0611

from static_frame.core.memory_measure import MaterializedArray
from static_frame.core.memory_measure import MeasureFormat
from static_frame.core.memory_measure import MemoryDisplay
from static_frame.core.memory_measure import MemoryMeasure
from static_frame.core.memory_measure import memory_total
from static_frame.test.test_case import TestCase

if tp.TYPE_CHECKING:
    TNDArrayAny = np.ndarray[tp.Any, tp.Any] #pragma: no cover
    TDtypeAny = np.dtype[tp.Any] #pragma: no cover

_iter_iterable = MemoryMeasure._iter_iterable
_iter_slots = MemoryMeasure._iter_slots
nested_sizable_elements = MemoryMeasure.nested_sizable_elements


class TestUnit(TestCase):

    #---------------------------------------------------------------------------
    # MemoryMeasure._iter_iterable

    def test_unsized_children_none(self) -> None:
        self.assertEqual(tuple(_iter_iterable(None)), ())

    def test_unsized_children_int(self) -> None:
        self.assertEqual(tuple(_iter_iterable(3)), ())

    def test_unsized_children_str(self) -> None:
        self.assertEqual(tuple(_iter_iterable('abc')), ())

    def test_unsized_children_float(self) -> None:
        self.assertEqual(tuple(_iter_iterable(4.5)), ())

    def test_unsized_children_numpy_array_int(self) -> None:
        obj = np.array([2, 3, 4], dtype=np.int64)
        self.assertEqual(tuple(_iter_iterable(obj)), ())

    def test_unsized_children_numpy_array_object(self) -> None:
        obj = np.array([2, 'a', 4], dtype=object)
        self.assertEqual(tuple(_iter_iterable(obj)), (2, 'a', 4))

    def test_unsized_children_tuple(self) -> None:
        obj = (2, 3, 4)
        self.assertEqual(tuple(_iter_iterable(obj)), (2, 3, 4))

    def test_unsized_children_tuple_nested(self) -> None:
        obj = (2, ('a', 'b', ('c', 'd')), 4)
        self.assertEqual(tuple(_iter_iterable(obj)), (2, ('a', 'b', ('c', 'd')), 4))

    def test_unsized_children_list(self) -> None:
        obj = [2, 3, 4]
        self.assertEqual(tuple(_iter_iterable(obj)), (2, 3, 4))

    def test_unsized_children_list_nested(self) -> None:
        obj = [2, ('a', 'b', ('c', 'd')), 4]
        self.assertEqual(tuple(_iter_iterable(obj)), (2, ('a', 'b', ('c', 'd')), 4))

    def test_unsized_children_set(self) -> None:
        obj = set((2, 3, 4))
        self.assertEqual(tuple(_iter_iterable(obj)), (2, 3, 4))

    def test_unsized_children_frozenset(self) -> None:
        obj = frozenset((2, 3, 4))
        self.assertEqual(tuple(_iter_iterable(obj)), (2, 3, 4))

    def test_unsized_children_dict(self) -> None:
        obj = { 'a': 2, 'b': 3, 'c': 4 }
        self.assertEqual(tuple(_iter_iterable(obj)), ('a', 2, 'b', 3, 'c', 4))

    def test_unsized_children_frozenautomap(self) -> None:
        obj = FrozenAutoMap([2, 3, 4])
        self.assertEqual(tuple(_iter_iterable(obj)), ())

    def test_unsized_children_numpy_array_object_complex_has_unique_ids(self) -> None:
        # make sure that all elements are looped through in a multi-dimensional object array
        obj = np.array([np.array([None, None, i]) for i in range(10)])
        self.assertEqual(len(set(id(el) for el in _iter_iterable(obj))), 11)

    #---------------------------------------------------------------------------
    # MemoryMeasure._iter_slots

    def test_sizable_slot_attrs_empty(self) -> None:
        class A:
            pass
        obj = A()
        self.assertEqual(tuple(_iter_slots(obj)), ())

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
        self.assertEqual(frozenset(_iter_slots(obj)), frozenset(('a', 'b', 'c', 'd', 'e')))

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
        self.assertEqual(frozenset(_iter_slots(obj)), frozenset(('a', 'b', 'e')))

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
        self.assertEqual(frozenset(_iter_slots(obj)), frozenset(('a', 'b', 'c', 'd', 'e')))

    def test_sizable_slot_attrs_inheritance_1_layer_overlapping_slots(self) -> None:
        class A:
            __slots__ = (
                'apples',
                'bananas',
                'carrots',
            )
            def __init__(self) -> None:
                self.apples = 'a'
                self.bananas = 'b'
        class B(A):
            __slots__ = ( # pylint: disable=W0244 # intentionally redefining 'carrots'
                'dumplings',
                'eggs'
            )
            def __init__(self) -> None:
                super().__init__()
                self.carrots = 'c'
                self.dumplings = 'd'
                self.eggs = 'e'
        obj = B()
        self.assertEqual(frozenset(_iter_slots(obj)), frozenset(('a', 'b', 'c', 'd', 'e')))
        self.assertEqual(len(tuple(_iter_slots(obj))), 5)

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
        sizables = frozenset(_iter_slots(obj))
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
        self.assertEqual(frozenset(_iter_slots(obj)), frozenset(('a', 'b', 'c', 'd', 'e')))

    # NOTE: From https://docs.python.org/3/reference/datamodel.html#notes-on-using-slots
    # > Multiple inheritance with multiple slotted parent classes can be used, but only one
    # >   parent is allowed to have attributes created by slots (the other bases must have empty
    # >   slot layouts) - violations raise TypeError.

    #---------------------------------------------------------------------------
    # MemoryMeasure.nested_sizable_elements

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
        obj = np.array([2, 'a', 4], dtype=object)
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

    #---------------------------------------------------------------------------
    def test_measure_format_a(self) -> None:
        empty: TNDArrayAny = np.array(())
        a1 = np.array((1, 2), dtype=np.int64)
        a2 = a1[:]

        mempty = MaterializedArray(empty, format=MeasureFormat.LOCAL_MATERIALIZED_DATA)
        ma1 = MaterializedArray(a1, format=MeasureFormat.LOCAL_MATERIALIZED_DATA)

        self.assertEqual(memory_total(mempty,
                format=MeasureFormat.REFERENCED_MATERIALIZED_DATA),
                0)
        self.assertEqual(memory_total(empty,
                format=MeasureFormat.REFERENCED_MATERIALIZED_DATA),
                0)

        self.assertEqual(memory_total(ma1,
                format=MeasureFormat.REFERENCED_MATERIALIZED_DATA),
                a1.nbytes,
                )
        self.assertEqual(memory_total(a1,
                format=MeasureFormat.REFERENCED_MATERIALIZED_DATA),
                a1.nbytes,
                )
        self.assertEqual(memory_total(a2,
                format=MeasureFormat.REFERENCED_MATERIALIZED_DATA),
                a1.nbytes,
                )

    def test_measure_format_b(self) -> None:
        empty: TNDArrayAny = np.array(())
        mempty = MaterializedArray(empty, format=MeasureFormat.REFERENCED_MATERIALIZED)

        a1 = np.array((1, 2), dtype=np.int64)
        ma1 = MaterializedArray(a1, format=MeasureFormat.REFERENCED_MATERIALIZED)

        a2 = a1[:]
        ma2 = MaterializedArray(a2, format=MeasureFormat.REFERENCED_MATERIALIZED)

        # import ipdb; ipdb.set_trace()
        self.assertEqual(memory_total(empty,
                format=MeasureFormat.REFERENCED_MATERIALIZED),
                getsizeof(mempty),
                )

        self.assertEqual(memory_total(a1,
                format=MeasureFormat.REFERENCED_MATERIALIZED),
                getsizeof(ma1)
                )

        self.assertEqual(memory_total(a2,
                format=MeasureFormat.REFERENCED_MATERIALIZED),
                getsizeof(ma2)
                )

    def test_measure_format_c(self) -> None:
        empty: TNDArrayAny = np.array(())
        a1 = np.array((1, 2), dtype=np.int64)
        a2 = a1[:]

        self.assertEqual(memory_total(empty,
                format=MeasureFormat.LOCAL),
                getsizeof(empty),
                )

        self.assertEqual(memory_total(a1,
                format=MeasureFormat.LOCAL),
                getsizeof(a1),
                )

        self.assertEqual(memory_total(a2,
                format=MeasureFormat.LOCAL),
                getsizeof(a2),
                )

    def test_measure_format_d(self) -> None:
        empty: TNDArrayAny = np.array(())
        a1 = np.array((1, 2), dtype=np.int64)
        a2 = a1[:]

        self.assertEqual(memory_total(empty,
                format=MeasureFormat.REFERENCED),
                getsizeof(empty),
                )

        self.assertEqual(memory_total(a1,
                format=MeasureFormat.REFERENCED),
                getsizeof(a1),
                )

        self.assertEqual(memory_total(a2,
                format=MeasureFormat.REFERENCED),
                getsizeof(a2) + getsizeof(a1),
                )


    def test_measure_format_e(self) -> None:
        empty: TNDArrayAny = np.array(())
        a1 = np.array((1, 2), dtype=np.int64)
        a2 = a1[:]

        self.assertEqual(memory_total(empty,
                format=MeasureFormat.LOCAL_MATERIALIZED_DATA),
                0,
                )

        self.assertEqual(memory_total(a1,
                format=MeasureFormat.LOCAL_MATERIALIZED_DATA),
                a1.nbytes,
                )

        self.assertEqual(memory_total(a2,
                format=MeasureFormat.LOCAL_MATERIALIZED_DATA),
                0,
                )

    def test_measure_format_f(self) -> None:
        empty: TNDArrayAny = np.array(())
        a1 = np.array((1, 2), dtype=np.int64)
        a2 = a1[:]

        mempty = MaterializedArray(empty, format=MeasureFormat.LOCAL_MATERIALIZED)
        ma1 = MaterializedArray(a1, format=MeasureFormat.LOCAL_MATERIALIZED)
        ma2 = MaterializedArray(a2, format=MeasureFormat.LOCAL_MATERIALIZED)

        self.assertEqual(memory_total(empty,
                format=MeasureFormat.LOCAL_MATERIALIZED),
                getsizeof(mempty),
                )

        self.assertEqual(memory_total(a1,
                format=MeasureFormat.LOCAL_MATERIALIZED),
                getsizeof(ma1),
                )

        self.assertEqual(memory_total(a2,
                format=MeasureFormat.LOCAL_MATERIALIZED),
                getsizeof(ma2),
                )

    #---------------------------------------------------------------------------

    def test_memory_display_a(self) -> None:
        f = ff.parse('s(16,8)|i(I,str)|v(str,int,float)')

        post = MemoryDisplay.from_any(f,
                (('Index', f._index), ('Columns', f._columns), ('Values', f._blocks)),
                )
        self.assertEqual(post.to_frame().loc['Total']['R'], memory_total(f, format=MeasureFormat.REFERENCED))

    def test_memory_display_b(self) -> None:
        f = ff.parse('s(16,8)|i(I,str)|v(str,int,float)').rename("test_mm")

        post = MemoryDisplay.from_any(f,
                (('Index', f._index), ('Columns', f._columns), ('Values', f._blocks)),
                )

        post_repr = repr(post)

        self.assertEqual(post._repr, repr(post))
        self.assertNotIn("test_mm", post_repr)  # name not inclucded

        from static_frame.core.util import bytes_to_size_label

        for (row, col), value in post.to_frame().iter_element_items():
            self.assertIn(row, post_repr)
            self.assertIn(col, post_repr)
            size, label = bytes_to_size_label(value).split()
            self.assertIn(size, post_repr)
            self.assertIn(label, post_repr)


if __name__ == '__main__':
    unittest.main()
