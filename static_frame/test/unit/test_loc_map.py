from __future__ import annotations

from copy import deepcopy

import numpy as np
import typing_extensions as tp

from static_frame import IndexHierarchy
from static_frame.core.exception import ErrorInitIndexNonUnique
from static_frame.core.index import Index
from static_frame.core.index_datetime import IndexDate
from static_frame.core.index_hierarchy import build_indexers_from_product
from static_frame.core.loc_map import HierarchicalLocMap
from static_frame.core.loc_map import LocMap
from static_frame.core.util import DTYPE_UINT_DEFAULT
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import PositionsAllocator
from static_frame.test.test_case import TestCase

if tp.TYPE_CHECKING:
    TNDArrayAny = np.ndarray[tp.Any, tp.Any] #pragma: no cover
    TDtypeAny = np.dtype[tp.Any] #pragma: no cover


def np_arange(*args: int) -> TNDArrayAny:
    a = np.arange(*args)
    a.flags.writeable = False
    return a

class TestLocMapUnit(TestCase):

    def test_loc_map_a(self) -> None:
        idx = Index(['a', 'b', 'c'])
        post1 = LocMap.loc_to_iloc(
                label_to_pos=idx._map,
                labels=idx._labels,
                positions=idx._positions,
                key='b',
                partial_selection=False,
                )
        self.assertEqual(post1, 1)

        post2 = LocMap.loc_to_iloc(
                label_to_pos=idx._map,
                labels=idx._labels,
                positions=idx._positions,
                key=NULL_SLICE,
                partial_selection=False,
                )
        self.assertEqual(post2, NULL_SLICE)

    def test_loc_map_b(self) -> None:
        idx = Index(['a', 'b', 'c', 'd', 'e'])
        post1 = LocMap.loc_to_iloc(
                label_to_pos=idx._map,
                labels=idx._labels,
                positions=idx._positions,
                key=['b', 'd'],
                partial_selection=False,
                )
        self.assertEqual(post1.tolist(), [1, 3]) #type: ignore

    def test_loc_map_slice_a(self) -> None:
        dt64 = np.datetime64
        idx = IndexDate.from_date_range('1985-01-01', '1985-01-08')

        post1 = LocMap.loc_to_iloc(
                label_to_pos=idx._map,
                labels=idx._labels,
                positions=idx._positions,
                key=slice(dt64('1985-01-01'), dt64('1985-01-04')),
                partial_selection=False,
                )
        self.assertEqual(post1, slice(0, 4, None))

        post2 = LocMap.loc_to_iloc(
                label_to_pos=idx._map,
                labels=idx._labels,
                positions=idx._positions,
                key=slice(dt64('1985-01-01'), dt64('1985-01-04'), 2),
                partial_selection=False,
                )
        self.assertEqual(post2, slice(0, 4, 2))

    def test_loc_map_slice_b(self) -> None:
        dt64 = np.datetime64
        idx = IndexDate.from_date_range('1985-01-01', '1985-01-08')

        with self.assertRaises(RuntimeError):
            post1 = LocMap.loc_to_iloc(
                    label_to_pos=idx._map,
                    labels=idx._labels,
                    positions=idx._positions,
                    key=slice(dt64('1985-01-01'), dt64('1985-01-04'), dt64('1985-01-04')),
                    partial_selection=False,
                    )

    def test_loc_map_slice_c(self) -> None:
        dt64 = np.datetime64
        idx = IndexDate.from_date_range('1985-01-01', '1985-01-08')

        post1 = LocMap.loc_to_iloc(
                label_to_pos=idx._map,
                labels=idx._labels,
                positions=idx._positions,
                key=slice(dt64('1985-01-01'), dt64('1985-01-04')),
                partial_selection=False,
                )
        self.assertEqual(post1, slice(0, 4, None))

    def test_loc_map_slice_d(self) -> None:
        dt64 = np.datetime64
        idx = IndexDate.from_date_range('1985-01-06', '1985-04-08')

        post1 = LocMap.loc_to_iloc(
                label_to_pos=idx._map,
                labels=idx._labels,
                positions=idx._positions,
                key=slice(dt64('1985-01'), dt64('1985-03')),
                partial_selection=False,
                )
        self.assertEqual(post1, slice(0, 85, None))


class TestHierarchicalLocMapUnit(TestCase):

    #---------------------------------------------------------------------------

    def test_init_a(self) -> None:
        indices = [
                Index(np_arange(5)),
                Index(tuple('ABCDE')),
                ]
        indexers = np.array(
            [
                [3, 3, 0, 1, 4, 0, 3, 2, 2, 0],
                [4, 2, 1, 0, 3, 0, 3, 2, 0, 4],
            ]
        )

        hlmap = HierarchicalLocMap(indices=indices, indexers=indexers)

        self.assertListEqual(list(hlmap.encoded_indexer_map), [35, 19, 8, 1, 28, 0, 27, 18, 2, 32])
        self.assertFalse(hlmap.encoding_can_overflow)
        self.assertListEqual(hlmap.bit_offset_encoders.tolist(), [0, 3])

    def test_init_b(self) -> None:
        indices = [Index(()) for _ in range(4)]
        indexers: tp.List[TNDArrayAny] = [np.array(()) for _ in range(4)]

        hlmap = HierarchicalLocMap(indices=indices, indexers=indexers)

        self.assertListEqual(list(hlmap.encoded_indexer_map), [])
        self.assertFalse(hlmap.encoding_can_overflow)
        self.assertListEqual(hlmap.bit_offset_encoders.tolist(), [0, 0, 0, 0])

    def test_init_c(self) -> None:
        indices = [Index((0, 1)), Index((0, 1))]
        indexers = np.array(
            [
                [0, 0, 1, 1, 1],
                [0, 1, 0, 1, 1],
            ]
        )

        with self.assertRaises(ErrorInitIndexNonUnique):
            HierarchicalLocMap(indices=indices, indexers=indexers)

    def test_init_d(self) -> None:
        indices = [Index(tuple('ab')), Index(tuple('bc')), Index(tuple('cd'))]
        indexers = np.array(
            [
                [0, 0, 1, 1, 0],
                [0, 1, 0, 1, 1],
                [0, 1, 0, 1, 1],
                #--------------
                #a, a, b, b, a
                #b, c, b, c, c
                #c, d, c, d, d
            ]
        )

        try:
            HierarchicalLocMap(indices=indices, indexers=indexers)
        except ErrorInitIndexNonUnique as e:
            assert e.args[0] == ('a', 'c', 'd')
        else:
            assert False, 'exception not raised'

    #---------------------------------------------------------------------------

    def test_build_offsets_and_overflow_a(self) -> None:
        def check(sizes: tp.List[int], offsets: tp.List[int], overflow: bool) -> None:
            actual_offset, actual_overflow = HierarchicalLocMap.build_offsets_and_overflow(sizes)
            self.assertListEqual(actual_offset.tolist(), offsets)
            self.assertEqual(actual_overflow, overflow)

        check([17, 99], [0, 5], False)
        check([1, 1], [0, 1], False)
        check([1, 2, 4, 8, 16, 32], [0, 1, 3, 6, 10, 15], False)
        check([2**30, 2, 3, 4], [0, 31, 33, 35], False)
        check([2**40, 2**18, 15], [0, 41, 60], False)
        check([2**40, 2**18, 16], [0, 41, 60], True)

    #---------------------------------------------------------------------------

    def test_build_encoded_indexers_map_a(self) -> None:
        sizes = [188, 5, 77]
        indexers = build_indexers_from_product(sizes)

        bit_offsets, overflow = HierarchicalLocMap.build_offsets_and_overflow(sizes)

        self.assertListEqual(bit_offsets.tolist(), [0, 8, 11])
        self.assertFalse(overflow)

        result = HierarchicalLocMap.build_encoded_indexers_map(
                encoding_can_overflow=overflow,
                bit_offset_encoders=bit_offsets,
                indexers=indexers,
                )
        self.assertEqual(len(result), len(indexers[0]))

        self.assertEqual(min(result), 0)
        self.assertEqual(max(result), 156859)

        # Manually check every element to ensure it encodes to the same value
        for i, row in enumerate(np.array(indexers).T):
            encoded = np.bitwise_or.reduce(row.astype(np.uint64) << bit_offsets)
            self.assertEqual(i, result[encoded])

    def test_build_encoded_indexers_map_b(self) -> None:
        size = 2**20
        sizes = [size for _ in range(4)]

        arr = PositionsAllocator.get(size)
        indexers = np.array([arr for _ in range(4)])

        bit_offsets, overflow = HierarchicalLocMap.build_offsets_and_overflow(sizes)

        self.assertListEqual(bit_offsets.tolist(), [0, 21, 42, 63])
        self.assertTrue(overflow)

        result = HierarchicalLocMap.build_encoded_indexers_map(
                encoding_can_overflow=overflow,
                bit_offset_encoders=bit_offsets,
                indexers=indexers,
                )
        self.assertEqual(len(result), len(indexers[0]))

        self.assertEqual(min(result), 0)
        self.assertEqual(max(result), 9671401945228815945957375)

        # Manually encode the last row to ensure it matches!
        indexer = np.array([size - 1 for _ in range(4)], dtype=object)
        encoded = np.bitwise_or.reduce(indexer << bit_offsets)
        self.assertEqual(max(result), encoded)

    #---------------------------------------------------------------------------

    def test_build_key_indexers_from_key_a(self) -> None:
        ih = IndexHierarchy.from_product(range(3), range(4, 7), tuple('ABC'))

        hlmapA = ih._map
        hlmapB = deepcopy(ih._map)
        hlmapB.encoding_can_overflow = True

        def check(
                key: tuple, # type: ignore
                expected: tp.List[tp.List[int]],
                ) -> None:
            resultA = hlmapA.build_key_indexers(key, indices=ih._indices)
            self.assertEqual(resultA.dtype, np.uint64)
            self.assertListEqual(resultA.tolist(), expected)

            resultB = hlmapB.build_key_indexers(key, indices=ih._indices)
            self.assertEqual(resultB.dtype, object)
            self.assertListEqual(resultB.tolist(), expected)

        check((0, 5, 'A'), [0, 1, 0]) # type: ignore
        check((0, 5, ['A']), [[0, 1, 0]])
        check(([0, 1],  5, ['B']), [[0, 1, 1],
                                    [1, 1, 1]])
        check(([0, 1], 5, 'A'), [[0, 1, 0],
                                 [1, 1, 0]])
        check(([0, 1], [4, 5, 6], 'C'), [[0, 0, 2],
                                         [0, 1, 2],
                                         [0, 2, 2],
                                         [1, 0, 2],
                                         [1, 1, 2],
                                         [1, 2, 2]])

    #---------------------------------------------------------------------------

    def test_is_single_element_a(self) -> None:
        self.assertTrue(HierarchicalLocMap.is_single_element(None))
        self.assertTrue(HierarchicalLocMap.is_single_element(True))
        self.assertTrue(HierarchicalLocMap.is_single_element(123))
        self.assertTrue(HierarchicalLocMap.is_single_element(1.0023))
        self.assertTrue(HierarchicalLocMap.is_single_element(np.nan))
        self.assertTrue(HierarchicalLocMap.is_single_element(()))
        self.assertTrue(HierarchicalLocMap.is_single_element((1, 2, 3)))

        self.assertFalse(HierarchicalLocMap.is_single_element([False]))
        self.assertFalse(HierarchicalLocMap.is_single_element([2.3, 8878.33]))
        self.assertFalse(HierarchicalLocMap.is_single_element(np_arange(5)))

    #---------------------------------------------------------------------------

    def test_loc_to_iloc_a(self) -> None:
        indices = [
                Index(np_arange(5)),
                Index(tuple('ABCDE')),
                ]
        indexers = np.array(
            [
                [3, 3, 0, 1, 4, 0, 3, 2, 2, 0],
                [4, 2, 1, 0, 3, 0, 3, 2, 0, 4],
            ]
        )

        hlmap = HierarchicalLocMap(indices=indices, indexers=indexers)

        self.assertEqual(hlmap.loc_to_iloc((2, 'A'), indices), 8)
        self.assertEqual(hlmap.loc_to_iloc((2, ['A']), indices), [8])
        self.assertEqual(hlmap.loc_to_iloc(([2], 'A'), indices), [8])
        self.assertEqual(hlmap.loc_to_iloc(([2], ['A']), indices), [8])

        self.assertEqual(hlmap.loc_to_iloc(([0, 3], 'E'), indices), [9, 0])
        self.assertEqual(hlmap.loc_to_iloc(([0, 3], ['E']), indices), [9, 0])
        self.assertEqual(hlmap.loc_to_iloc(([3, 0], 'E'), indices), [0, 9])
        self.assertEqual(hlmap.loc_to_iloc(([3, 0], ['E']), indices), [0, 9])

        self.assertEqual(hlmap.loc_to_iloc(np.array([0, 'E'], dtype=object), indices), 9)

    def test_loc_to_iloc_b(self) -> None:
        indices = [
                Index(np_arange(5)),
                Index(tuple('ABCDE')),
                ]
        indexers = np.array(
            [
                [3, 3, 0, 1, 4, 0, 3, 2, 2, 0],
                [4, 2, 1, 0, 3, 0, 3, 2, 0, 4],
            ]
        )

        hlmap = HierarchicalLocMap(indices=indices, indexers=indexers)

        with self.assertRaises(KeyError):
            hlmap.loc_to_iloc((5, 'A'), indices)

        with self.assertRaises(KeyError):
            hlmap.loc_to_iloc((2, ['E']), indices)

        with self.assertRaises(KeyError):
            hlmap.loc_to_iloc(([0, 1, 2], ['A', 'B', 'C']), indices)

    #---------------------------------------------------------------------------

    def test_nbytes_a(self) -> None:
        indices = [
                Index(np_arange(5)),
                Index(tuple('ABCDE')),
                ]
        indexers = np.array(
            [
                [3, 3, 0, 1, 4, 0, 3, 2, 2, 0],
                [4, 2, 1, 0, 3, 0, 3, 2, 0, 4],
            ]
        )

        hlmap = HierarchicalLocMap(indices=indices, indexers=indexers)
        self.assertTrue(hlmap.nbytes > 0)

    #---------------------------------------------------------------------------

    def test_deepcopy_a(self) -> None:
        indices = [
                Index(np_arange(5)),
                Index(tuple('ABCDE')),
                ]
        indexers = np.array(
            [
                [3, 3, 0, 1, 4, 0, 3, 2, 2, 0],
                [4, 2, 1, 0, 3, 0, 3, 2, 0, 4],
            ]
        )

        hlmap = HierarchicalLocMap(indices=indices, indexers=indexers)

        hlmap_copy = deepcopy(hlmap)

        self.assertEqual(hlmap.encoding_can_overflow, hlmap_copy.encoding_can_overflow)
        self.assertListEqual(hlmap.bit_offset_encoders.tolist(), hlmap_copy.bit_offset_encoders.tolist())
        self.assertTrue((hlmap.encoded_indexer_map == hlmap_copy.encoded_indexer_map).all())

        self.assertNotEqual(id(hlmap.bit_offset_encoders), id(hlmap_copy.bit_offset_encoders))
        self.assertNotEqual(id(hlmap.encoded_indexer_map), id(hlmap_copy.encoded_indexer_map))

    #---------------------------------------------------------------------------

    def test_indexers_to_iloc_invalid_input(self) -> None:
        indices = [
                Index(np_arange(5)),
                Index(tuple('ABCDE')),
                ]
        indexers = np.array(
            [
                [3, 3, 0, 1, 4, 0, 3, 2, 2, 0],
                [4, 2, 1, 0, 3, 0, 3, 2, 0, 4],
            ]
        )

        hlmap = HierarchicalLocMap(indices=indices, indexers=indexers)

        # 1D
        with self.assertRaises(AssertionError):
            hlmap.indexers_to_iloc(np.array([0, 1, 2]))

        # Shape mismatch
        with self.assertRaises(AssertionError):
            hlmap.indexers_to_iloc(np.array([[0, 1, 2]]))

        # Invliad dtype
        with self.assertRaises(AssertionError):
            hlmap.indexers_to_iloc(np.array([[0, 1]]).astype(object))

    def test_indexers_to_iloc_a(self) -> None:
        indices = [
                Index(np_arange(5)),
                Index(tuple('ABCDE')),
                ]
        indexers = np.array(
            [
                [3, 3, 0, 1, 4, 0, 3, 2, 2, 0],
                [4, 2, 1, 0, 3, 0, 3, 2, 0, 4],
            ]
        )

        hlmap = HierarchicalLocMap(indices=indices, indexers=indexers)

        post = hlmap.indexers_to_iloc(indexers.T.astype(DTYPE_UINT_DEFAULT))
        self.assertListEqual(post, list(range(10)))

    def test_indexers_to_iloc_b(self) -> None:
        indices = [
                Index(np_arange(5)),
                Index(tuple('ABCDE')),
                ]
        indexers = np.array(
            [
                [3, 3, 0, 1, 4, 0, 3, 2, 2, 0],
                [4, 2, 1, 0, 3, 0, 3, 2, 0, 4],
            ]
        )

        hlmap = HierarchicalLocMap(indices=indices, indexers=indexers)

        subsets = [[5,2,4,1,3], [1], [9,8,7,6,5], [1,7,4,6]]

        for subset in subsets:
            post = hlmap.indexers_to_iloc(indexers.T.astype(DTYPE_UINT_DEFAULT)[subset])
            self.assertListEqual(post, subset)

    def test_indexers_to_iloc_c(self) -> None:
        indices = [
                Index(np_arange(5)),
                Index(tuple('ABCDE')),
                ]
        indexers = np.array(
            [
                [3, 3, 0, 1, 4, 0, 3, 2, 2, 0],
                [4, 2, 1, 0, 3, 0, 3, 2, 0, 4],
            ]
        )

        hlmap = HierarchicalLocMap(indices=indices, indexers=indexers)

        invalid_indexers = indexers.copy()
        invalid_indexers[0][0] = 14
        invalid_indexers[1][7] = 14
        invalid_indexers = invalid_indexers.T.astype(DTYPE_UINT_DEFAULT)

        with self.assertRaises(KeyError):
            _ = hlmap.indexers_to_iloc(invalid_indexers.copy())

        with self.assertRaises(KeyError):
            _ = hlmap.indexers_to_iloc(invalid_indexers[[0]].copy())

        with self.assertRaises(KeyError):
            _ = hlmap.indexers_to_iloc(invalid_indexers[[7]].copy())


        valid_subset = [1, 2, 3, 4, 5, 6, 8, 9]
        post = hlmap.indexers_to_iloc(invalid_indexers[valid_subset].copy())
        self.assertListEqual(post, valid_subset)

    def test_unpack_encoding_a(self) -> None:
        # Test the docstring!
        encoded_arr = np.array([36, 8, 10, 17], dtype=np.uint64)

        post = HierarchicalLocMap.unpack_encoding(
            encoded_arr=encoded_arr,
            bit_offset_encoders=np.array([0, 2, 4], dtype=np.uint64),
            encoding_can_overflow=False,
        )
        expected = np.array(
            [
                [0, 1, 2],
                [0, 2, 0],
                [2, 2, 0],
                [1, 0, 1],
            ],
            dtype=np.uint64,
        ).T

        assert (post == expected).all().all()

    def test_encoding_roundtrip(self) -> None:
        indexers = np.array(
            [
                [0, 2, 1, 1, 4, 5, 6, 7, 8, 9, 0],
                [1, 0, 2, 0, 5, 4, 5, 8, 9, 0, 7],
                [0, 9, 4, 2, 1, 3, 1, 4, 5, 6, 7],
            ],
            dtype=np.uint64,
        )
        indexers.flags.writeable = False

        bit_offset_encoders, can_overflow = HierarchicalLocMap.build_offsets_and_overflow([10, 10, 10])
        assert not can_overflow

        encodings = HierarchicalLocMap.build_encoded_indexers_map(
                encoding_can_overflow=can_overflow,
                bit_offset_encoders=bit_offset_encoders,
                indexers=indexers,
                )
        encoded_arr = np.array(list(encodings), dtype=np.uint64)

        unpacked_indexers = HierarchicalLocMap.unpack_encoding(
                encoded_arr=encoded_arr,
                bit_offset_encoders=bit_offset_encoders,
                encoding_can_overflow=can_overflow,
                )

        assert unpacked_indexers is not indexers
        assert id(unpacked_indexers) != id(indexers)
        assert (unpacked_indexers == indexers).all().all()


if __name__ == '__main__':
    import unittest
    unittest.main()
