from __future__ import annotations

import unittest
from sys import getsizeof

import frame_fixtures as ff
import numpy as np
import typing_extensions as tp
from arraymap import FrozenAutoMap  # pylint: disable=E0611

from static_frame import Bus
from static_frame import Frame
from static_frame import Index
from static_frame import IndexDateGO
from static_frame import IndexGO
from static_frame import IndexHierarchy
from static_frame import Quilt
from static_frame import Series
from static_frame import StoreConfig
from static_frame import TypeBlocks
from static_frame import Yarn
from static_frame.core.memory_measure import memory_total
from static_frame.test.test_case import TestCase
from static_frame.test.test_case import temp_file


class TestUnit(TestCase):
    def test_getsizeof_total_int(self) -> None:
        obj = 2
        self.assertEqual(memory_total(obj), getsizeof(2))

    def test_getsizeof_total_set(self) -> None:
        obj = set(['a', 'b', 4])
        self.assertEqual(
            memory_total(obj),
            sum(getsizeof(e) for e in (
                'a', 'b', 4,
                set(['a', 'b', 4]),
            ))
        )

    def test_getsizeof_total_frozenset(self) -> None:
        obj = frozenset(['a', 'b', 4])
        self.assertEqual(
            memory_total(obj),
            sum(getsizeof(e) for e in (
                'a', 'b', 4,
                frozenset(['a', 'b', 4]),
            ))
        )

    def test_getsizeof_total_np_array(self) -> None:
        obj = np.arange(3)
        self.assertEqual(
            memory_total(obj),
            getsizeof(obj)
        )

    def test_getsizeof_total_np_array_slice(self) -> None:
        arr = np.arange(5)
        obj = arr[2:]
        self.assertEqual(
            memory_total(obj),
            sum(getsizeof(e) for e in (
                obj,
                obj.base,
            ))
        )

    def test_getsizeof_total_np_array_with_slice(self) -> None:
        arr = np.arange(5)
        sli = arr[2:]
        obj = [arr, sli]
        self.assertEqual(
            memory_total(obj),
            sum(getsizeof(e) for e in (
                arr,
                sli,
                obj,
            ))
        )

    def test_getsizeof_total_frozenautomap(self) -> None:
        obj = FrozenAutoMap(['a', 'b', 'c'])
        self.assertEqual(
            memory_total(obj),
            getsizeof(FrozenAutoMap(['a', 'b', 'c']))
        )

    def test_getsizeof_total_dict(self) -> None:
        obj = { 'a': 2, 'b': 3, 'c': (4, 5) }
        self.assertEqual(
            memory_total(obj),
            sum(getsizeof(e) for e in (
                'a', 2,
                'b', 3,
                'c', 4, 5, (4, 5),
                { 'a': 2, 'b': 3, 'c': (4, 5) },
            ))
        )

    def test_getsizeof_total_tuple(self) -> None:
        obj = (2, 3, 4)
        self.assertEqual(memory_total(obj), sum(getsizeof(e) for e in (
            2, 3, 4,
            (2, 3, 4),
        )))

    def test_getsizeof_total_nested_tuple(self) -> None:
        obj = (2, 'b', (2, 3))
        self.assertEqual(memory_total(obj), sum(getsizeof(e) for e in (
            2, 'b', 3,
            (2, 3),
            (2, 'b', (2, 3)),
        )))

    def test_getsizeof_total_predefined_seen(self) -> None:
        obj = (4, 5, (2, 8))
        seen = set((id(2), id(3)))
        self.assertEqual(memory_total(obj, seen=seen), sum(getsizeof(e) for e in (
            4, 5,
            8,
            (2, 8),
            (4, 5, (2, 8))
        )))

    def test_getsizeof_total_predefined_seen_all_elements(self) -> None:
        tup_a = (2, 8)
        obj = (4, 5, tup_a)
        seen = set((id(e) for e in (4, 5, 2, 8, tup_a, obj)))
        self.assertEqual(memory_total(obj, seen=seen), 0)

    def test_getsizeof_total_predefined_seen_sub_elements(self) -> None:
        tup_a = (2, 8)
        obj = (4, 5, tup_a)
        seen = set((id(e) for e in (4, 5, 2, 8, tup_a)))
        self.assertEqual(memory_total(obj, seen=seen), getsizeof(obj))

    def test_getsizeof_total_predefined_seen_base_element(self) -> None:
        tup_a = (2, 8)
        obj = (4, 5, tup_a)
        seen = set((id(obj),))
        self.assertEqual(memory_total(obj, seen=seen), 0)

    def test_getsizeof_total_larger_values_is_larger(self) -> None:
        a = ('ab', 'cd', 'ef')
        b = ('abc', 'def', 'ghi')
        self.assertLess(memory_total(a), memory_total(b))

    def test_getsizeof_total_more_values_is_larger_a(self) -> None:
        a = ('a', 'b', 'c')
        b = ('a', 'b', 'c', 'd')
        self.assertLess(memory_total(a), memory_total(b))

    def test_getsizeof_total_more_values_is_larger_b(self) -> None:
        a = ('a', 'b', 'c')
        b = 'd'
        self.assertLess(memory_total([a]), memory_total([a, b]))

    def test_getsizeof_total_more_values_is_larger_nested_a(self) -> None:
        a = ('a', (2, (8, 9), 4), 'c')
        b = ('a', (2, (8, 9, 10), 4), 'c')
        self.assertLess(memory_total(a), memory_total(b))

    def test_getsizeof_total_more_values_is_larger_nested_b(self) -> None:
        a = np.array(['a', [2, (8, 9), 4], 'c'], dtype=object)
        b = np.array(['a', [2, (8, 9, 10), 4], 'c'], dtype=object)
        self.assertLess(memory_total(a), memory_total(b))

    #---------------------------------------------------------------------------
    # Frame

    def test_getsizeof_total_frame_simple(self) -> None:
        f = ff.parse('s(3,4)')
        seen: tp.Set[int] = set()
        self.assertEqual(memory_total(f), sum((
            memory_total(f._blocks, seen=seen),
            memory_total(f._columns, seen=seen),
            memory_total(f._index, seen=seen),
            memory_total(f._name, seen=seen),
            getsizeof(f)
        )))

    def test_getsizeof_total_frame_string_index(self) -> None:
        f = ff.parse('s(3,4)|i(I,str)|c(I,str)')
        seen: tp.Set[int] = set()
        self.assertEqual(memory_total(f), sum((
            memory_total(f._blocks, seen=seen),
            memory_total(f._columns, seen=seen),
            memory_total(f._index, seen=seen),
            memory_total(f._name, seen=seen),
            getsizeof(f)
        )))

    def test_getsizeof_total_frame_object_index(self) -> None:
        f = ff.parse('s(8,12)|i(I,object)|c(I,str)')
        seen: tp.Set[int] = set()
        self.assertEqual(memory_total(f), sum((
            memory_total(f._blocks, seen=seen),
            memory_total(f._columns, seen=seen),
            memory_total(f._index, seen=seen),
            memory_total(f._name, seen=seen),
            getsizeof(f)
        )))

    def test_getsizeof_total_frame_multiple_value_types(self) -> None:
        f = ff.parse('s(8,4)|i(I,object)|c(I,str)|v(object,int,bool,str)')
        seen: tp.Set[int] = set()
        self.assertEqual(memory_total(f), sum((
            memory_total(f._blocks, seen=seen),
            memory_total(f._columns, seen=seen),
            memory_total(f._index, seen=seen),
            memory_total(f._name, seen=seen),
            getsizeof(f)
        )))

    def test_getsizeof_total_frame_he_before_hash(self) -> None:
        f = ff.parse('s(3,4)').to_frame_he()
        seen: tp.Set[int] = set()
        self.assertEqual(memory_total(f), sum((
            memory_total(f._blocks, seen=seen),
            memory_total(f._columns, seen=seen),
            memory_total(f._index, seen=seen),
            memory_total(f._name, seen=seen),
            # memory_total(f._hash, seen=seen), # not initialized yet
            getsizeof(f)
        )))

    def test_getsizeof_total_frame_he_after_hash(self) -> None:
        f = ff.parse('s(3,4)').to_frame_he()
        hash(f) # to initialize _hash
        seen: tp.Set[int] = set()
        self.assertEqual(memory_total(f), sum((
            memory_total(f._blocks, seen=seen),
            memory_total(f._columns, seen=seen),
            memory_total(f._index, seen=seen),
            memory_total(f._name, seen=seen),
            memory_total(f._hash, seen=seen),
            getsizeof(f)
        )))

    #---------------------------------------------------------------------------
    # Index

    def test_getsizeof_total_index_simple(self) -> None:
        idx = Index(('a', 'b', 'c'))
        self.assertEqual(memory_total(idx), sum(getsizeof(e) for e in (
            idx._map,
            idx._labels,
            idx._positions,
            idx._positions.base, # from PositionsAllocator.get
            idx._recache,
            idx._name,
            idx
        )))

    def test_getsizeof_total_index_object(self) -> None:
        idx = Index((1, 'b', (2, 3)))
        self.assertEqual(memory_total(idx), sum(getsizeof(e) for e in (
            idx._map,
            idx._labels,
            # _positions is an object dtype numpy array
            1, 'b',
            2, 3,
            (2, 3),
            idx._positions,
            idx._positions.base, # from PositionsAllocator.get
            idx._recache,
            idx._name,
            idx
        )))

    def test_getsizeof_total_index_loc_is_iloc(self) -> None:
        idx = Index((0, 1, 2), loc_is_iloc=True)
        self.assertEqual(memory_total(idx), sum(getsizeof(e) for e in (
            idx._map,
            idx._labels,
            idx._positions,
            idx._positions.base, # from PositionsAllocator.get
            idx._recache,
            # idx._name, # both _map and _name are None
            idx
        )))

    def test_getsizeof_total_index_empty(self) -> None:
        idx = Index(())
        self.assertEqual(memory_total(idx), sum(getsizeof(e) for e in (
            idx._map,
            idx._labels,
            idx._positions,
            idx._positions.base, # from PositionsAllocator.get
            idx._recache,
            idx._name,
            idx
        )))

    def test_getsizeof_total_index_name_adds_size(self) -> None:
        idx1 = Index(('a', 'b', 'c'))
        idx2 = idx1.rename('with_name')
        self.assertLess(memory_total(idx1), memory_total(idx2))

    def test_getsizeof_total_index_more_values_adds_size(self) -> None:
        idx1 = Index(('a', 'b', 'c'))
        idx2 = Index(('a', 'b', 'c', 'd'))
        self.assertLess(memory_total(idx1), memory_total(idx2))

    def test_getsizeof_total_index_more_nested_values_adds_size(self) -> None:
        idx1 = Index((1, 'b', (2, 3)))
        idx2 = Index((1, 'b', (2, 3, 4, 5)))
        self.assertLess(memory_total(idx1), memory_total(idx2))

    def test_getsizeof_total_index_more_doubly_nested_values_adds_size(self) -> None:
        idx1 = Index((1, 'b', ('c', (8, 9), 'd')))
        idx2 = Index((1, 'b', ('c', (8, 9, 10), 'd')))
        self.assertLess(memory_total(idx1), memory_total(idx2))

    def test_getsizeof_total_index_loc_is_iloc_reduces_size(self) -> None:
        # idx1 will be smaller since the _positions and _labels variables point to the same array
        idx1 = Index((0, 1, 2), loc_is_iloc=True)
        idx2 = Index((0, 1, 2))
        self.assertLess(memory_total(idx1), memory_total(idx2))

    def test_getsizeof_total_index_go(self) -> None:
        idx = IndexGO(('a', 'b', 'c'))
        self.assertEqual(memory_total(idx), sum(getsizeof(e) for e in (
            idx._map,
            idx._labels,
            idx._positions,
            idx._positions.base, # from PositionsAllocator.get
            idx._recache,
            idx._name,
            'a', 'b', 'c',
            idx._labels_mutable,
            idx._labels_mutable_dtype,
            idx._positions_mutable_count,
            idx
        )))

    def test_getsizeof_total_index_go_after_append(self) -> None:
        idx = IndexGO(('a', 'b', 'c'))
        idx.append('d')
        self.assertEqual(memory_total(idx), sum(getsizeof(e) for e in (
            idx._map,
            idx._labels,
            idx._positions,
            idx._positions.base, # from PositionsAllocator.get
            idx._recache,
            idx._name,
            'a', 'b', 'c', 'd',
            idx._labels_mutable,
            idx._labels_mutable_dtype,
            idx._positions_mutable_count,
            idx
        )))

    def test_getsizeof_total_index_datetime_go(self) -> None:
        idx = IndexDateGO.from_date_range('1994-01-01', '1995-01-01')
        self.assertEqual(memory_total(idx), sum(getsizeof(e) for e in (
            idx._map,
            idx._labels,
            idx._positions,
            idx._positions.base, # from PositionsAllocator.get
            idx._recache,
            idx._name,
            *idx._labels_mutable, # Note: _labels_mutable is not nested
            idx._labels_mutable,
            idx._labels_mutable_dtype,
            idx._positions_mutable_count,
            idx
        )))

    #---------------------------------------------------------------------------
    # Series

    def test_getsizeof_total_series_simple(self) -> None:
        s = Series(('a', 'b', 'c'))
        seen: tp.Set[int] = set()
        self.assertEqual(memory_total(s), sum(memory_total(e, seen=seen) for e in (
            s.values,
            s._index,
            s._name,
        )) + getsizeof(s))

    def test_getsizeof_total_series_with_index(self) -> None:
        s = Series(('a', 'b', 'c'), index=(0, 1, 2))
        seen: tp.Set[int] = set()
        self.assertEqual(memory_total(s), sum(memory_total(e, seen=seen) for e in (
            s.values,
            s._index,
            s._name,
        )) + getsizeof(s))

    def test_getsizeof_total_series_object_values(self) -> None:
        s = Series(('a', (2, (3, 4), 5), 'c'))
        seen: tp.Set[int] = set()
        self.assertEqual(memory_total(s), sum(memory_total(e, seen=seen) for e in (
            s.values,
            s._index,
            s._name,
        )) + getsizeof(s))

    def test_getsizeof_total_series_with_name_is_larger(self) -> None:
        s1 = Series(('a', 'b', 'c'))
        s2 = s1.rename('named_series')
        self.assertLess(memory_total(s1), memory_total(s2))

    def test_getsizeof_total_series_larger_series_is_larger_a(self) -> None:
        s1 = Series(('a', 'b', 'c'))
        s2 = Series(('a', 'b', 'c', 'd'))
        self.assertLess(memory_total(s1), memory_total(s2))

    def test_getsizeof_total_series_larger_series_is_larger_b(self) -> None:
        s1 = Series(('a', 'b', 'c'))
        s2 = Series(('abc', 'def', 'ghi'))
        self.assertLess(memory_total(s1), memory_total(s2))

    def test_getsizeof_total_series_larger_nested_series_is_larger(self) -> None:
        s1 = Series(('a', (2, (4, 5), 8), 'c'))
        s2 = Series(('a', (2, (4, 5, 6), 8), 'c'))
        self.assertLess(memory_total(s1), memory_total(s2))

    def test_getsizeof_total_series_larger_index_is_larger(self) -> None:
        s1 = Series(('a', 'b', 'c'), index=(0, 1, 2))
        s2 = Series(('a', 'b', 'c'), index=('abc', 'def', 'ghi'))
        self.assertLess(memory_total(s1), memory_total(s2))

    def test_getsizeof_total_series_he_before_hash(self) -> None:
        s = Series(('a', 'b', 'c')).to_series_he()
        seen: tp.Set[int] = set()
        self.assertEqual(memory_total(s), sum(memory_total(e, seen=seen) for e in (
            s.values,
            s._index,
            s._name,
            # s._hash, # not initialized yet
        )) + getsizeof(s))

    def test_getsizeof_total_series_he_after_hash(self) -> None:
        s = Series(('a', 'b', 'c')).to_series_he()
        hash(s) # to initialize _hash
        seen: tp.Set[int] = set()
        self.assertEqual(memory_total(s), sum(memory_total(e, seen=seen) for e in (
            s.values,
            s._index,
            s._name,
            s._hash,
        )) + getsizeof(s))

    #---------------------------------------------------------------------------
    # TypeBlocks

    def test_getsizeof_total_type_blocks_1d_array(self) -> None:
        a = np.array([1, 2, 3])
        tb = TypeBlocks.from_blocks(a)
        self.assertTrue(memory_total(tb), sum(getsizeof(e) for e in (
            np.array([1, 2, 3]),
            tb._blocks, # [np.array([1, 2, 3])],
            0,
            (0, 0),
            tb._index, # [(0, 0)],
            3, 1,
            # _row_dtype, # np.dtype('int64') is already included
            tb
        )))

    def test_getsizeof_total_type_blocks_list_of_1d_arrays(self) -> None:
        tb = TypeBlocks.from_blocks([
            np.array([1, 2, 3]),
            np.array([4, 5, 6])
        ])
        self.assertEqual(memory_total(tb), sum(getsizeof(e) for e in (
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            tb._blocks, # [np.array([1, 2, 3]), np.array([4, 5, 6])],
            tb._index, # [(0, 0), (1, 0)],
            tb
        )))

    def test_getsizeof_total_type_blocks_2d_array(self) -> None:
        tb = TypeBlocks.from_blocks(np.array([[1, 2, 3], [4, 5, 6]]))
        self.assertEqual(memory_total(tb), sum(getsizeof(e) for e in (
            np.array([[1, 2, 3],[4, 5, 6]]),
            tb._blocks, # [np.array([[1, 2, 3],[4, 5, 6]])],
            tb._index, # [(0, 0), (0, 1), (0, 2)],
            tb
        )))

    #---------------------------------------------------------------------------
    # IndexHierarchy

    def test_getsizeof_total_index_hierarchy_simple(self) -> None:
        idxa = Index(('a', 'b', 'c'))
        idxb = Index((1, 2, 3))
        idx = IndexHierarchy.from_product(idxa, idxb)
        seen: tp.Set[int] = set()
        self.assertEqual(memory_total(idx), sum(memory_total(e, seen=seen) for e in (
            idx._indices,
            idx._indexers,
            idx._name,
            idx._blocks,
            idx._recache,
            idx._values,
            idx._map,
            idx._index_types,
            idx._pending_extensions,
        )) + getsizeof(idx))

    #---------------------------------------------------------------------------
    # Bus

    def test_getsizeof_total_bus_simple(self) -> None:
        f1 = ff.parse('s(3,6)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        b = Bus.from_frames((f1, f2))
        seen: tp.Set[int] = set()
        self.assertEqual(memory_total(b), sum(memory_total(e, seen=seen) for e in (
            b._loaded,
            b._loaded_all,
            b._values_mutable,
            b._index,
            b._name,
            b._store,
            b._config,
            # b._last_accessed, # not initialized, not a "max_persist" bus
            b._max_persist,
        )) + getsizeof(b))

    def test_getsizeof_total_bus_maxpersist(self) -> None:
        def items() -> tp.Iterator[tp.Tuple[str, Frame]]:
            for i in range(20):
                yield str(i), Frame(np.arange(i, i+10).reshape(2, 5))

        s = Series.from_items(items(), dtype=object)
        b1 = Bus.from_series(s)

        config = StoreConfig(
                index_depth=1,
                columns_depth=1,
                include_columns=True,
                include_index=True
                )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)

            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=3)

            seen: tp.Set[int] = set()
            self.assertEqual(memory_total(b2), sum(memory_total(e, seen=seen) for e in (
                b2._loaded,
                b2._loaded_all,
                b2._values_mutable,
                b2._index,
                b2._name,
                b2._store,
                b2._config,
                b2._last_accessed,
                b2._max_persist,
            )) + getsizeof(b2))

    #---------------------------------------------------------------------------
    # Yarn

    def test_getsizeof_total_yarn_simple(self) -> None:
        f1 = ff.parse('s(4,4)|v(int,float)').rename('f1')
        f2 = ff.parse('s(4,4)|v(str)').rename('f2')
        f3 = ff.parse('s(4,4)|v(bool)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3))

        f4 = ff.parse('s(4,4)|v(int,float)').rename('f4')
        f5 = ff.parse('s(4,4)|v(str)').rename('f5')
        b2 = Bus.from_frames((f4, f5))

        y = Yarn((b1, b2))
        seen: tp.Set[int] = set()
        self.assertEqual(memory_total(y), sum(memory_total(e, seen=seen) for e in (
            y._values,
            y._hierarchy,
            y._index,
            y._indexer,
            y._deepcopy_from_bus,
        )) + getsizeof(y))

    #---------------------------------------------------------------------------
    # Quilt

    def test_getsizeof_total_quilt_simple_before_columns(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,2)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,2)').rename('f4')
        f5 = ff.parse('s(4,2)').rename('f5')
        f6 = ff.parse('s(6,2)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3), name='b1')
        b2 = Bus.from_frames((f4,), name='b2')
        b3 = Bus.from_frames((f5, f6), name='b3')

        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=False)

        q = Quilt(y1, retain_labels=True)
        seen: tp.Set[int] = set()
        self.assertEqual(memory_total(q), sum(memory_total(e, seen=seen) for e in (
            q._bus,
            q._axis,
            q._axis_hierarchy,
            q._axis_opposite,
            q._assign_axis,
            #q._columns, # not initialized until after get columns property
            #q._index, # not initialized until after get columns property
            q._retain_labels,
            q._deepcopy_from_bus,
        )) + getsizeof(q))

    def test_getsizeof_total_quilt_simple_after_columns(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,2)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,2)').rename('f4')
        f5 = ff.parse('s(4,2)').rename('f5')
        f6 = ff.parse('s(6,2)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3), name='b1')
        b2 = Bus.from_frames((f4,), name='b2')
        b3 = Bus.from_frames((f5, f6), name='b3')

        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=False)

        q = Quilt(y1, retain_labels=True)
        q.columns # pylint:disable=W0104 # force columns initialization
        seen: tp.Set[int] = set()
        self.assertEqual(memory_total(q), sum(memory_total(e, seen=seen) for e in (
            q._bus,
            q._axis,
            q._axis_hierarchy,
            q._axis_opposite,
            q._assign_axis,
            q._columns,
            q._index,
            q._retain_labels,
            q._deepcopy_from_bus,
        )) + getsizeof(q))

if __name__ == '__main__':
    unittest.main()
