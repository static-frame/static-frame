import unittest
import pickle
from itertools import zip_longest
import copy

import numpy as np
from arraykit import immutable_filter

from static_frame import mloc
from static_frame import TypeBlocks
from static_frame.core.exception import AxisInvalid
from static_frame.core.exception import ErrorInitTypeBlocks
from static_frame.core.index_correspondence import IndexCorrespondence
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import isna_array
from static_frame.test.test_case import skip_win
from static_frame.test.test_case import TestCase

nan = np.nan


class TestUnit(TestCase):

    def test_type_blocks_init_a(self) -> None:
        with self.assertRaises(ErrorInitTypeBlocks):
            tb1 = TypeBlocks.from_blocks((3, 4))
        with self.assertRaises(ErrorInitTypeBlocks):
            tb1 = TypeBlocks.from_blocks((np.arange(8).reshape((2, 2, 2)),))


    def test_type_blocks_a(self) -> None:

        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([1, 2, 3])
        a2 = np.array([10,50,30])
        a3 = np.array([1345,2234,3345])
        a4 = np.array([False, True, False])
        a5 = np.array([False, False, False])
        a6 = np.array(['g', 'd', 'e'])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3, a4, a5, a6))

        # can show that with tb2, a6 remains unchanged

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb3 = TypeBlocks.from_blocks((a1, a2, a3))

        # showing that slices keep the same memory location
        # self.assertTrue((tb1[0:2].mloc == tb1.mloc[:2]).all())
        # self.assertTrue((tb1.mloc[:2] == tb1.iloc[0:2, 0:2].mloc).all())

    def test_type_blocks_contiguous_pairs(self) -> None:

        a = [(0, 1), (0, 2), (2, 3), (2, 1)]
        post = list(TypeBlocks._indices_to_contiguous_pairs(a))
        self.assertEqual(post, [
                (0, slice(1, 3)),
                (2, slice(3, 4)),
                (2, slice(1, 2)),
                ])

        a = [(0, 0), (0, 1), (0, 2), (1, 4), (2, 1), (2, 3)]
        post = list(TypeBlocks._indices_to_contiguous_pairs(a))
        self.assertEqual(post, [
                (0, slice(0, 3)),
                (1, slice(4, 5)),
                (2, slice(1, 2)),
                (2, slice(3, 4)),
            ])



    def test_type_blocks_b(self) -> None:

        # given naturally of a list of rows; this corresponds to what we get with iloc, where we select a row first, then a column
        a1 = np.array([[1, 2, 3], [4, 5, 6]])
        # shape is given as rows, columns
        self.assertEqual(a1.shape, (2, 3))

        a2 = np.array([[.2, .5, .4], [.8, .6, .5]])
        a3 = np.array([['a', 'b'], ['c', 'd']])


        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertTypeBlocksArrayEqual(tb1[2], [[3], [6]])
        self.assertTypeBlocksArrayEqual(tb1[4], [[0.5], [0.6]])

        self.assertEqual(list(tb1[7].values), ['b', 'd'])

        self.assertEqual(tb1.shape, (2, 8))

        self.assertEqual(len(tb1), 2)
        self.assertEqual(tb1._row_dtype, np.object_)

        slice1 = tb1[2:5]
        self.assertEqual(slice1.shape, (2, 3))

        slice2 = tb1[0:5]
        self.assertEqual(slice2.shape, (2, 5))

        # pick columns
        slice3 = tb1[[2,6,0]]
        self.assertEqual(slice3.shape, (2, 3))

        # TODO: need to implement values

        self.assertEqual(slice3.iloc[0].values.tolist(), [[3, 'a', 1]])
        self.assertEqual(slice3.iloc[1].values.tolist(), [[6, 'c', 4]])

        ## slice refers to the same data; not sure if this is accurate test yet

        row1 = tb1.iloc[0].values
        self.assertEqual(row1.dtype, object)
        self.assertEqual(row1.shape, (1, 8))
        self.assertEqual(row1[:, :3].tolist(), [[1, 2, 3]])
        self.assertEqual(row1[:, -2:].tolist(), [['a', 'b']])

        self.assertEqual(tb1.unified, False)



    def test_type_blocks_c(self) -> None:

        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        a4 = np.array(['gd', 'cd', 'dd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3, a4))

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))

        row1 = tb1.iloc[2]

        self.assertEqual(tb1.shape, (3, 4))

        self.assertEqual(tb1.iloc[1].values.tolist(), [[2, True, 'c', 'cd']])

        self.assertEqual(tb1.iloc[0, 0:2].shape, (1, 2))
        self.assertEqual(tb1.iloc[0:2, 0:2].shape, (2, 2))



    def test_type_blocks_d(self) -> None:
        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])


        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertEqual(tb1.iloc[0:2].shape, (2, 8))
        self.assertEqual(tb1.iloc[1:3].shape, (2, 8))



    def test_type_blocks_indices_to_contiguous_pairs(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])


        tb1 = TypeBlocks.from_blocks((a1, a2, a3))
        self.assertEqual(list(tb1._key_to_block_slices(0)), [(0, 0)])
        self.assertEqual(list(tb1._key_to_block_slices(6)), [(2, 0)])
        self.assertEqual(list(tb1._key_to_block_slices([3,5,6])),
            [(1, slice(0, 1, None)), (1, slice(2, 3, None)), (2, slice(0, 1, None))]
            )

    #---------------------------------------------------------------------------

    def test_type_blocks_key_to_block_slices_a(self) -> None:
        a1 = np.array([1, 2, -1])
        a2 = np.array([[False, True], [True, True], [False, True]])
        tb1 = TypeBlocks.from_blocks((a1, a2))

        self.assertEqual(list(tb1._key_to_block_slices(None)),
                [(0, slice(0, 1, None)), (1, slice(0, 2, None))]
                )

        with self.assertRaises(NotImplementedError):
            list(tb1._key_to_block_slices('a'))


    #---------------------------------------------------------------------------

    def test_type_blocks_extract_a(self) -> None:

        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        a4 = np.array(['gd', 'cd', 'dd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3, a4))

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))

        # double point extraction goes to single elements
        self.assertEqual(tb2._extract(1, 3), True)
        self.assertEqual(tb1._extract(1, 2), 'c')

        # single row extraction
        self.assertEqual(tb1._extract(1).shape, (1, 4))
        self.assertEqual(tb1._extract(0).shape, (1, 4))
        self.assertEqual(tb2._extract(0).shape, (1, 8))

        # single column _extractions
        self.assertEqual(tb1._extract(None, 1).shape, (3, 1))
        self.assertEqual(tb2._extract(None, 1).shape, (3, 1))

        # multiple row selection
        self.assertEqual(tb2._extract([1,2],).shape, (2, 8))
        self.assertEqual(tb2._extract([0,2],).shape, (2, 8))
        self.assertEqual(tb2._extract([0,2], 6).shape, (2, 1))
        self.assertEqual(tb2._extract([0,2], [6,7]).shape, (2, 2))

        # mixed
        self.assertEqual(tb2._extract(1,).shape, (1, 8))
        self.assertEqual(tb2._extract([0,2]).shape, (2, 8))
        self.assertEqual(tb2._extract(1, 4), False)
        self.assertEqual(tb2._extract(1, 3), True)
        self.assertEqual(tb2._extract([0, 2],).shape, (2, 8))


        # slices
        self.assertEqual(tb2._extract(slice(1,3)).shape, (2, 8))
        self.assertEqual(tb2._extract(slice(1,3), slice(3,6)).shape, (2,3))
        self.assertEqual(tb2._extract(slice(1,2)).shape, (1,8))
        # a boundry over extended still gets 1
        self.assertEqual(tb2._extract(slice(2,4)).shape, (1,8))
        self.assertEqual(tb2._extract(slice(None), slice(2,4)).shape, (3, 2))
        self.assertEqual(tb1._extract(slice(2,4)).shape, (1, 4))


    def test_type_blocks_extract_b(self) -> None:
        # test case of a single unified block

        a1 = np.array([
            [1, 2, 3, -5],
            [10, 50, 30, -7],
            [1345, 2234, 3345, -200]])
        tb1 = TypeBlocks.from_blocks(a1)
        self.assertEqual(tb1.shape, (3, 4))
        self.assertEqual(len(tb1.mloc), 1)

        a1 = np.array([1,10,1345])
        a2 = np.array([2, 50, 2234])
        a3 = np.array([3, 30, 3345])
        a4 = np.array([-5, -7, -200])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3, a4))
        self.assertEqual(tb2.shape, (3, 4))
        self.assertEqual(len(tb2.mloc), 4)


        tb1_a = tb1._extract(row_key=slice(0,1))
        tb2_a = tb2._extract(row_key=slice(0,1))
        self.assertEqual(tb1_a.shape, tb2_a.shape)
        self.assertTrue((tb1_a.values == tb2_a.values).all())


        tb1_b = tb1._extract(row_key=slice(1))
        tb2_b = tb2._extract(row_key=slice(1))

        self.assertEqual(tb1_b.shape, tb2_b.shape)
        self.assertTrue((tb1_b.values == tb2_b.values).all())


        tb1_c = tb1._extract(row_key=slice(0, 2))
        tb2_c = tb2._extract(row_key=slice(0, 2))

        self.assertEqual(tb1_c.shape, tb2_c.shape)
        self.assertTrue((tb1_c.values == tb2_c.values).all())

        tb1_d = tb1._extract(row_key=slice(0, 2), column_key=3)
        tb2_d = tb2._extract(row_key=slice(0, 2), column_key=3)

        self.assertEqual(tb1_d.shape, tb2_d.shape)
        self.assertTrue((tb1_d.values == tb2_d.values).all())

        tb1_e = tb1._extract(row_key=slice(0, 2), column_key=slice(2,4))
        tb2_e = tb2._extract(row_key=slice(0, 2), column_key=slice(2,4))

        self.assertEqual(tb1_e.shape, tb2_e.shape)
        self.assertTrue((tb1_e.values == tb2_e.values).all())

        self.assertTrue(tb1._extract(row_key=2, column_key=2) ==
                tb2._extract(row_key=2, column_key=2) ==
                3345)

    def test_type_blocks_extract_c(self) -> None:
        # test negative slices

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))


        # test negative row slices
        self.assertTypeBlocksArrayEqual(
                tb1._extract(-1),
                [0, 0, 1, True, False, True, 'oe', 'od'],
                match_dtype=object
                )
        self.assertTypeBlocksArrayEqual(
                tb1._extract(slice(-2, None)),
                [[4, 5, 6, True, False, True, 'c', 'd'],
                [0, 0, 1, True, False, True, 'oe', 'od']],
                match_dtype=object
                )
        self.assertTypeBlocksArrayEqual(
                tb1._extract(slice(-3, -1)),
                [[1, 2, 3, False, False, True, 'a', 'b'],
                [4, 5, 6, True, False, True, 'c', 'd']],
                match_dtype=object
                )

        self.assertTypeBlocksArrayEqual(
                tb1._extract(slice(None), -2),
                [['a'], ['c'], ['oe']],
                match_dtype=object
                )
        self.assertTypeBlocksArrayEqual(
                tb1._extract(slice(None), slice(-6, -1)),
                [[3, False, False, True, 'a'],
                [6, True, False, True, 'c'],
                [1, True, False, True, 'oe']],
                match_dtype=object
                )

        self.assertTypeBlocksArrayEqual(
                tb1._extract(slice(None), slice(-1, -4, -1)),
                [['b', 'a', True],
                ['d', 'c', True],
                ['od', 'oe', True]],
                match_dtype=object)

    #---------------------------------------------------------------------------

    def test_type_blocks_extract_array_a(self) -> None:
        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a4 = tb1._extract_array(row_key=1)
        self.assertEqual(a4.tolist(),
                [4, 5, 6, True, False, True, 'c', 'd'])

        a5 = tb1._extract_array(column_key=5)
        self.assertEqual(a5.tolist(),
                [True, True, True])


    def test_type_blocks_extract_array_b(self) -> None:

        a1 = np.arange(10).reshape(2, 5)
        tb1 = TypeBlocks.from_blocks(a1)

        a2 = tb1._extract_array(1, 4)
        self.assertEqual(a2, 9)

        a2 = tb1._extract_array(NULL_SLICE, 4)
        self.assertEqual(a2.tolist(), [4, 9])


    def test_type_blocks_extract_array_c(self) -> None:

        a1 = np.arange(4)
        a2 = np.arange(10, 14)
        a3 = np.arange(20, 24)

        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a2 = tb1._extract_array(1, 2)
        self.assertEqual(a2, 21)

        a2 = tb1._extract_array(NULL_SLICE, 1)
        self.assertEqual(a2.tolist(), [10, 11, 12, 13])

    #---------------------------------------------------------------------------

    def test_immutable_filter(self) -> None:
        a1 = np.array([3, 4, 5])
        a2 = immutable_filter(a1)
        with self.assertRaises(ValueError):
            a2[0] = 34
        a3 = a2[:2]
        with self.assertRaises(ValueError):
            a3[0] = 34


    def test_type_blocks_static_frame(self) -> None:
        a1 = np.array([1, 2, 3], dtype=np.int64)
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertTrue(tb1.dtypes[0] == np.int64)


    def test_type_blocks_attributes(self) -> None:

        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertEqual(tb1.size, 9)
        self.assertEqual(tb2.size, 24)




    def test_type_blocks_block_pointers(self) -> None:

        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertTrue((tb1[0:2].mloc == tb1.mloc[:2]).all())
        self.assertTrue((tb1.mloc[:2] == tb1.iloc[0:2, 0:2].mloc).all())




    def test_type_blocks_unary_operator_a(self) -> None:

        a1 = np.array([1,-2,-3])
        a2 = np.array([False, True, False])
        tb1 = TypeBlocks.from_blocks((a1, a2))

        tb2 = ~tb1 # tilde
        self.assertEqual(
            (~tb1.values).tolist(),
            [[-2, -1], [1, -2], [2, -1]])

    def test_type_blocks_unary_operator_b(self) -> None:

        a1 = np.array([[1, 2, 3], [-4, 5, 6], [0,0,-1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertTypeBlocksArrayEqual(
                -tb2[0:3], #type: ignore
                [[-1, -2, -3],
                 [ 4, -5, -6],
                 [ 0,  0,  1]],
                )

        self.assertTypeBlocksArrayEqual(
                ~tb2[3:5], #type: ignore
                [[ True,  True],
                [False,  True],
                [False,  True]],
                )



    def test_type_blocks_block_compatible_a(self) -> None:

        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))

        # get slices with unified types
        tb3 = tb1[[0, 1]]
        tb4 = tb2[[2, 3]]


        self.assertTrue(tb3.block_compatible(tb4))
        self.assertTrue(tb4.block_compatible(tb3))

        self.assertFalse(tb1.block_compatible(tb2))
        self.assertFalse(tb2.block_compatible(tb1))


    def test_type_blocks_block_compatible_b(self) -> None:

        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))

        tb2a = tb2[[2,3,7]]
        self.assertTrue(tb1.block_compatible(tb2a))

    #---------------------------------------------------------------------------

    def test_type_blocks_consolidate_a(self) -> None:

        a1 = np.array([1, 2, 3])
        a2 = np.array([10,50,30])
        a3 = np.array([1345,2234,3345])

        a4 = np.array([False, True, False])
        a5 = np.array([False, False, False])

        tb1 = TypeBlocks.from_blocks((a1, a2, a3, a4, a5))
        self.assertEqual(tb1.shape, (3, 5))

        tb2 = tb1.consolidate()

        self.assertTrue([b.dtype for b in tb2._blocks], [np.int, np.bool])
        self.assertEqual(tb2.shape, (3, 5))

        # we have perfect correspondence between the two
        self.assertTrue((tb1.values == tb2.values).all())


    def test_type_blocks_consolidate_b(self) -> None:
        # if we hava part of TB consolidated, we do not reallocate


        a1 = np.array([
            [1, 2, 3],
            [10,50,30],
            [1345,2234,3345]])

        a2 = np.array([False, True, False])
        a3 = np.array([False, False, False])

        tb1 = TypeBlocks.from_blocks((a1, a2, a3))
        self.assertEqual(tb1.shape, (3, 5))
        self.assertEqual(len(tb1.mloc), 3)

        tb2 = tb1.consolidate()
        self.assertEqual(tb1.shape, (3, 5))
        self.assertEqual(len(tb2.mloc), 2)
        # the first block is the same instance
        self.assertEqual(tb1.mloc[0], tb2.mloc[0])


    def test_type_blocks_consolidate_c(self) -> None:
        blocks = [np.empty(shape=(0, 1), dtype=np.dtype('>f4')), np.empty(shape=(0, 2), dtype=np.dtype('>f4'))]

        tb1 = TypeBlocks.from_blocks(blocks)
        tb2 = tb1.consolidate()
        self.assertTrue((tb1.dtypes == tb2.dtypes).all())


    #---------------------------------------------------------------------------


    def test_type_blocks_binary_operator_a(self) -> None:

        a1 = np.array([
            [1, 2, 3, -5],
            [10, 50, 30, -7],
            [1345, 2234, 3345, -200]])
        tb1 = TypeBlocks.from_blocks(a1)

        a1 = np.array([1,10,1345])
        a2 = np.array([2, 50, 2234])
        a3 = np.array([3, 30, 3345])
        a4 = np.array([-5, -7, -200])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3, a4))

        self.assertTrue(((tb1 + tb2).values == (tb1 + tb1).values).all())

        post1 = tb1 + tb2



    def test_type_blocks_binary_operator_b(self) -> None:

        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']], dtype=object)
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))


        self.assertTypeBlocksArrayEqual(
                tb2 * 3,
                [[3, 6, 9, 0, 0, 3, 'aaa', 'bbb'],
                [12, 15, 18, 3, 0, 3, 'ccc', 'ddd'],
                [0, 0, 3, 3, 0, 3, 'oeoeoe', 'ododod']],
                match_dtype=object
                )

        self.assertTypeBlocksArrayEqual(
                tb1[:2] + 10,
                [[11, 10],
                [12, 11],
                [13, 10]],
                )

        self.assertTypeBlocksArrayEqual(
                tb1[:2] + 10,
                [[11, 10],
                [12, 11],
                [13, 10]],
                )


    def test_type_blocks_binary_operator_c(self) -> None:
        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']], dtype=object)
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))

        # these result in the same thing
        self.assertTypeBlocksArrayEqual(
                tb1[:2] * tb2[[2,3]],
                [[3, False],
                [12, True],
                [3, False]]
                )

        self.assertTypeBlocksArrayEqual(
                tb1[0:2] * tb1[0:2],
                [[1, False],
                [4, True],
                [9, False]]
                )

        self.assertTypeBlocksArrayEqual(
                tb2[:3] % 2,
                [[1, 0, 1],
                [0, 1, 0],
                [0, 0, 1]]
            )


    def test_type_blocks_binary_operator_d(self) -> None:
        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[1.5,2.6], [4.2,5.5], [0.2,0.1]])
        tb1 = TypeBlocks.from_blocks((a1, a2))

        post = tb1 * (1, 0, 2, 0, 1)
        self.assertTypeBlocksArrayEqual(post,
                [[  1. ,   0. ,   6. ,   0. ,   2.6],
                [  4. ,   0. ,  12. ,   0. ,   5.5],
                [  0. ,   0. ,   2. ,   0. ,   0.1]])


    def test_type_blocks_binary_operator_e(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        a4 = np.array([None, None, None])
        tb = TypeBlocks.from_blocks((a1, a2, a4, a3))

        post1 = [x for x in tb.element_items()]

        tb2 = TypeBlocks.from_element_items(post1, tb.shape, tb._row_dtype)
        self.assertTrue((tb.values == tb2.values).all())

        post2 = tb == tb2
        self.assertEqual(post2.values.tolist(),
                [[True, True, True, True, True, True, True, True, True], [True, True, True, True, True, True, True, True, True], [True, True, True, True, True, True, True, True, True]])

    def test_type_blocks_binary_operator_f(self) -> None:
        a1 = np.array([[1, 2, 3], [4, 5, 6]])
        a2 = np.array([[1.5,2.6], [4.2,5.5]])
        tb1 = TypeBlocks.from_blocks((a1, a2))
        tb2 = TypeBlocks.from_blocks((a1,))

        with self.assertRaises(NotImplementedError):
            _ = tb1 @ tb1

        with self.assertRaises(NotImplementedError):
            _ = tb1 + tb2

        with self.assertRaises(NotImplementedError):
            _ = tb1 + tb2.values

    # def test_type_blocks_binary_operator_e(self) -> None:

    #     a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
    #     a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
    #     tb = TypeBlocks.from_blocks((a1, a2))

    #     # this applies row-wise, NPs default:
    #     # tb * [2, 2, 2, 2, 2, 2]
    #     # if this can be assumed to be an axis 0 operation
    #     import ipdb; ipdb.set_trace()


    #---------------------------------------------------------------------------

    def test_type_blocks_extend_a(self) -> None:
        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']], dtype=object)
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))

        # mutates in place
        tb1.extend(tb2)
        self.assertEqual(tb1.shape, (3, 11))

        self.assertTypeBlocksArrayEqual(
                tb1.iloc[2],
                [3, False, 'd', 0, 0, 1, True, False, True, 'oe', 'od'],
                match_dtype=object,
                )

    def test_type_blocks_extend_b(self) -> None:
        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        tb1 = TypeBlocks.from_blocks((a1, a2))

        tb2 = TypeBlocks.from_blocks(np.array([3, 4]))

        with self.assertRaises(RuntimeError):
            tb1.extend(tb2)

    #---------------------------------------------------------------------------

    def test_type_blocks_mask_blocks_a(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        mask = TypeBlocks.from_blocks(tb1._mask_blocks(column_key=[2,3,5,6]))

        self.assertTypeBlocksArrayEqual(mask,
            [[False, False, True, True, False, True, True, False], [False, False, True, True, False, True, True, False], [False, False, True, True, False, True, True, False]]
            )

    def test_type_blocks_mask_blocks_b(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6]])
        a2 = np.array([[False, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd']])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        # show that index order is not relevant to selection
        self.assertTypeBlocksArrayEqual(
                TypeBlocks.from_blocks(tb1._mask_blocks(column_key=[0, 5, 6])),
                [[ True, False, False, False, False,  True,  True, False],
                [ True, False, False, False, False,  True,  True, False]])

        self.assertTypeBlocksArrayEqual(
                TypeBlocks.from_blocks(tb1._mask_blocks(column_key=[0, 6, 5])),
                [[ True, False, False, False, False,  True,  True, False],
                [ True, False, False, False, False,  True,  True, False]])

        self.assertTypeBlocksArrayEqual(
                TypeBlocks.from_blocks(tb1._mask_blocks(column_key=[6, 5, 0])),
                [[ True, False, False, False, False,  True,  True, False],
                [ True, False, False, False, False,  True,  True, False]])

        # with repeated values we get the same result; we are not presently filtering out duplicates, however
        self.assertTypeBlocksArrayEqual(
                TypeBlocks.from_blocks(tb1._mask_blocks(column_key=[6, 6, 5, 5, 0])),
                [[ True, False, False, False, False,  True,  True, False],
                [ True, False, False, False, False,  True,  True, False]])


    def test_type_blocks_mask_blocks_c(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6]])
        a2 = np.array([[False, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd']])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertTypeBlocksArrayEqual(
                TypeBlocks.from_blocks(tb1._mask_blocks(column_key=slice(2, None, 2))),
                [[False, False,  True, False,  True, False,  True, False],
                [False, False,  True, False,  True, False,  True, False]])

        tb2 = TypeBlocks.from_blocks(tb1._mask_blocks(column_key=slice(4, None, -1)))
        self.assertTypeBlocksArrayEqual(tb2,
                [[ True,  True,  True,  True,  True, False, False, False],
                [ True,  True,  True,  True,  True, False, False, False]])

        tb3 = TypeBlocks.from_blocks(tb1._mask_blocks(column_key=slice(4, 2, -1)))
        self.assertTypeBlocksArrayEqual(tb3,
                [[ False,  False,  False,  True,  True, False, False, False],
                [ False,  False,  False,  True,  True, False, False, False]])

        tb4 = TypeBlocks.from_blocks(tb1._mask_blocks(column_key=slice(6, None, -2)))
        self.assertTypeBlocksArrayEqual(tb4,
                [[ True, False,  True, False,  True, False,  True, False],
                [ True, False,  True, False,  True, False,  True, False]])

        tb5 = TypeBlocks.from_blocks(tb1._mask_blocks(column_key=slice(6, None, -3)))
        self.assertTypeBlocksArrayEqual(tb5,
                [[ True, False, False,  True, False, False,  True, False],
                [ True, False, False,  True, False, False,  True, False]])


    def test_type_blocks_mask_blocks_d(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6]])
        a2 = np.array([[False, False], [True, False]])
        tb1 = TypeBlocks.from_blocks((a1, a2))
        tb2 = tb1.extract_iloc_mask(1)
        self.assertEqual(tb2.values.tolist(),
                [[False, False, False, False, False],
                [True, True, True, True, True]])

    #---------------------------------------------------------------------------

    def test_type_blocks_assign_blocks_a(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        tb2 = TypeBlocks.from_blocks(tb1._assign_from_iloc_by_unit(
                column_key=[2,3,5], value=300))

        self.assertTypeBlocksArrayEqual(tb2,
            [[1, 2, 300, 300, False, 300, 'a', 'b'],
            [4, 5, 300, 300, False, 300, 'c', 'd'],
            [0, 0, 300, 300, False, 300, 'oe', 'od']], match_dtype=object)

        # blocks not mutated will be the same
        self.assertEqual(tb1.mloc[2], tb2.mloc[5])
        self.assertEqual(tb2.shapes.tolist(),
            [(3, 2), (3, 1), (3, 1), (3, 1), (3, 1), (3, 2)]
            )

    def test_type_blocks_assign_blocks_b(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        tb2 = TypeBlocks.from_blocks(tb1._assign_from_iloc_by_unit(
                column_key=slice(-3, None), value=300))

        self.assertTypeBlocksArrayEqual(tb2,
            [[1, 2, 3, False, False, 300, 300, 300],
            [4, 5, 6, True, False, 300, 300, 300],
            [0, 0, 1, True, False, 300, 300, 300]], match_dtype=object)

        # blocks not mutated will be the same
        self.assertEqual(tb1.mloc[0], tb2.mloc[0])
        self.assertEqual(tb2.shapes.tolist(),
            [(3, 3), (3, 2), (3, 1), (3, 2)]
            )

    @skip_win  # type: ignore
    def test_type_blocks_assign_blocks_c(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        tb2 = TypeBlocks.from_blocks(tb1._assign_from_iloc_by_unit(
                column_key=[0, 2, 3, 5, 7], value=300))

        self.assertTypeBlocksArrayEqual(tb2,
            [[300, 2, 300, 300, False, 300, 'a', 300],
            [300, 5, 300, 300, False, 300, 'c', 300],
            [300, 0, 300, 300, False, 300, 'oe', 300]], match_dtype=object)


        self.assertEqual(tb2.shapes.tolist(),
            [(3, 1), (3, 1), (3, 1), (3, 1), (3, 1), (3, 1), (3, 1), (3, 1)]
            )

        self.assertEqual(tb2.dtypes.tolist(),
            [np.dtype('int64'), np.dtype('int64'), np.dtype('int64'), np.dtype('int64'), np.dtype('bool'), np.dtype('int64'), np.dtype('<U2'), np.dtype('int64')]
            )


    def test_type_blocks_assign_blocks_d(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        value = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1])
        tb2 = TypeBlocks.from_blocks(tb1._assign_from_iloc_by_unit(
                row_key=[1], value=value))

        self.assertEqual(tb2.dtypes.tolist(),
                [np.dtype('float64'), np.dtype('float64'), np.dtype('float64'), np.dtype('O'), np.dtype('O'), np.dtype('O'), np.dtype('O'), np.dtype('O')])

        self.assertTypeBlocksArrayEqual(tb2,
            [[1.0, 2.0, 3.0, False, False, True, 'a', 'b'],
            [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1],
            [0.0, 0.0, 1.0, True, False, True, 'oe', 'od']], match_dtype=object)



    def test_type_blocks_assign_blocks_e(self) -> None:

        a1 = np.array([[True, True, True], [True, True, True], [True, True, True]])
        a2 = np.array([[False, False, False], [False, False, False], [False, False, False]])
        a3 = np.array([[True, True, True], [True, True, True], [True, True, True]])

        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        value = np.array([1.1, 2.1, 3.1, 4.1])

        tb2 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_unit(row_key=[1], column_key=slice(1, 5), value=value))

        self.assertEqual(tb1.shape, tb2.shape)
        self.assertEqual(tb2.dtypes.tolist(),
                [np.dtype('bool'), np.dtype('O'), np.dtype('O'), np.dtype('O'), np.dtype('O'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool')])

        self.assertEqual(tb2.iloc[1].values[0].tolist(),
                [True, 1.1, 2.1, 3.1, 4.1, False, True, True, True])



        tb3 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_unit(row_key=[1], column_key=slice(2, 6), value=value))

        self.assertEqual(tb1.shape, tb3.shape)
        self.assertEqual(tb3.dtypes.tolist(),
                [np.dtype('bool'), np.dtype('bool'), np.dtype('O'), np.dtype('O'), np.dtype('O'), np.dtype('O'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool')])

        self.assertEqual(tb3.iloc[1].values[0].tolist(),
                [True, True, 1.1, 2.1, 3.1, 4.1, True, True, True])


        tb4 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_unit(row_key=[1], column_key=slice(3, 7), value=value))

        self.assertEqual(tb1.shape, tb4.shape)
        self.assertEqual(tb4.dtypes.tolist(),
                [np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('O'), np.dtype('O'), np.dtype('O'), np.dtype('O'), np.dtype('bool'), np.dtype('bool')])

        self.assertEqual(tb4.iloc[1].values[0].tolist(),
                [True, True, True, 1.1, 2.1, 3.1, 4.1, True, True])


        tb5 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_unit(row_key=[1], column_key=slice(4, 8), value=value))

        self.assertEqual(tb1.shape, tb5.shape)
        self.assertEqual(tb5.dtypes.tolist(),
                [np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('O'), np.dtype('O'), np.dtype('O'), np.dtype('O'), np.dtype('bool')])

        self.assertEqual(tb5.iloc[1].values[0].tolist(),
                [True, True, True, False, 1.1, 2.1, 3.1, 4.1, True])


        tb6 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_unit(row_key=[1], column_key=slice(5, 9), value=value))

        self.assertEqual(tb1.shape, tb6.shape)
        self.assertEqual(tb6.dtypes.tolist(),
                [np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('O'), np.dtype('O'), np.dtype('O'), np.dtype('O'),])

        self.assertEqual(tb6.iloc[1].values[0].tolist(),
                [True, True, True, False, False, 1.1, 2.1, 3.1, 4.1])



    def test_type_blocks_assign_blocks_f(self) -> None:

        a1 = np.array([[True, True, True], [True, True, True], [True, True, True]])
        a2 = np.array([[False, False, False], [False, False, False], [False, False, False]])
        a3 = np.array([[True, True, True], [True, True, True], [True, True, True]])

        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        value = np.array([1.1, 2.1, 3.1, 4.1])

        tb2 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_unit(column_key=slice(1, 5), value=value))

        self.assertEqual(tb1.shape, tb2.shape)
        self.assertEqual(tb2.dtypes.tolist(),
                [np.dtype('bool'), np.dtype('float64'), np.dtype('float64'), np.dtype('float64'), np.dtype('float64'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool')])

        self.assertEqual(tb2.iloc[1].values[0].tolist(),
                [True, 1.1, 2.1, 3.1, 4.1, False, True, True, True])


        tb3 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_unit(column_key=slice(2, 6), value=value))

        self.assertEqual(tb1.shape, tb3.shape)
        self.assertEqual(tb3.dtypes.tolist(),
                [np.dtype('bool'), np.dtype('bool'), np.dtype('float64'), np.dtype('float64'), np.dtype('float64'), np.dtype('float64'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool')])

        self.assertEqual(tb3.iloc[1].values[0].tolist(),
                [True, True, 1.1, 2.1, 3.1, 4.1, True, True, True])


        tb4 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_unit(column_key=slice(3, 7), value=value))

        self.assertEqual(tb1.shape, tb4.shape)
        self.assertEqual(tb4.dtypes.tolist(),
                [np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('float64'), np.dtype('float64'), np.dtype('float64'), np.dtype('float64'), np.dtype('bool'), np.dtype('bool')])

        self.assertEqual(tb4.iloc[1].values[0].tolist(),
                [True, True, True, 1.1, 2.1, 3.1, 4.1, True, True])


        tb5 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_unit(column_key=slice(4, 8), value=value))

        self.assertEqual(tb1.shape, tb5.shape)
        self.assertEqual(tb5.dtypes.tolist(),
                [np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('float64'), np.dtype('float64'), np.dtype('float64'), np.dtype('float64'), np.dtype('bool')])

        self.assertEqual(tb5.iloc[1].values[0].tolist(),
                [True, True, True, False, 1.1, 2.1, 3.1, 4.1, True])


        tb6 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_unit(column_key=slice(5, 9), value=value))

        self.assertEqual(tb1.shape, tb6.shape)
        self.assertEqual(tb6.dtypes.tolist(),
                [np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('float64'), np.dtype('float64'), np.dtype('float64'), np.dtype('float64'),])

        self.assertEqual(tb6.iloc[1].values[0].tolist(),
                [True, True, True, False, False, 1.1, 2.1, 3.1, 4.1])



    def test_type_blocks_assign_blocks_g(self) -> None:

        a1 = np.array([[True, True, True], [True, True, True], [True, True, True]])
        a2 = np.array([[False, False, False], [False, False, False], [False, False, False]])
        a3 = np.array([[True, True, True], [True, True, True], [True, True, True]])

        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        value = np.array([[1.1, 2.1], [3.1, 4.1]])

        tb2 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_unit(row_key=slice(1, 3), column_key=slice(3, 5), value=value))

        self.assertEqual(tb2.dtypes.tolist(),
                [np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('O'), np.dtype('O'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool')])

        # import ipdb; ipdb.set_trace()
        self.assertEqual(tb2.shapes.tolist(),
                [(3, 3), (3, 2), (3, 1), (3, 3)])


        tb3 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_unit(row_key=slice(1, 3), column_key=slice(4, 6), value=value))

        self.assertEqual(tb3.dtypes.tolist(),
                [np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('O'), np.dtype('O'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool')])

        self.assertEqual(tb3.shapes.tolist(),
                [(3, 3), (3, 1), (3, 2), (3, 3)])


        tb4 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_unit(row_key=slice(1, 3), column_key=slice(5, 7), value=value))

        self.assertEqual(tb4.dtypes.tolist(),
                [np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('O'), np.dtype('O'), np.dtype('bool'), np.dtype('bool')])

        self.assertEqual(tb4.shapes.tolist(),
                [(3, 3), (3, 2), (3, 1), (3, 1), (3, 2)])


    def test_type_blocks_assign_blocks_h(self) -> None:

        a1 = np.array([[True, True, True], [True, True, True], [True, True, True]])
        a2 = np.array([[False, False, False], [False, False, False], [False, False, False]])
        a3 = np.array([[True, True, True], [True, True, True], [True, True, True]])

        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        value = np.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]])

        tb2 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_unit(column_key=slice(3, 5), value=value))

        self.assertEqual(tb2.dtypes.tolist(),
                [np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('float64'), np.dtype('float64'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool')])

        # import ipdb; ipdb.set_trace()
        self.assertEqual(tb2.shapes.tolist(),
                [(3, 3), (3, 2), (3, 1), (3, 3)])


        tb3 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_unit(column_key=slice(4, 6), value=value))

        self.assertEqual(tb3.dtypes.tolist(),
                [np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('float64'), np.dtype('float64'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool')])

        self.assertEqual(tb3.shapes.tolist(),
                [(3, 3), (3, 1), (3, 2), (3, 3)])


        tb4 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_unit(column_key=slice(5, 7), value=value))

        self.assertEqual(tb4.dtypes.tolist(),
                [np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('float64'), np.dtype('float64'), np.dtype('bool'), np.dtype('bool')])

        self.assertEqual(tb4.shapes.tolist(),
                [(3, 3), (3, 2), (3, 1), (3, 1), (3, 2)])



    def test_type_blocks_assign_blocks_i(self) -> None:

        a1 = np.array([[1.2], [2.1], [3.1]])
        a2 = np.array([False, True, True])
        tb1 = TypeBlocks.from_blocks((a1, a2))

        value = (20.1, 40.1)
        tb2 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_unit(column_key=0, row_key=np.array([True, True, False]), value=value))

        self.assertTypeBlocksArrayEqual(tb2,
            [[20.1, False], [40.1, True], [3.1, True]], match_dtype=object)


    def test_type_blocks_assign_blocks_j(self) -> None:

        a1 = np.array([[3], [3],])
        a2 = np.array([False, True])
        tb1 = TypeBlocks.from_blocks((a1, a2))

        targets = (np.full((2, 1), False), np.full(2, True))

        value = ('a', 'b')
        tb2 = TypeBlocks.from_blocks(
                tb1._assign_from_boolean_blocks_by_unit(
                       targets=targets,
                       value=value,
                       value_valid=targets,
                       ))
        self.assertEqual(tb2.values.tolist(),
                [[3, 'a'], [3, 'b']]
                )

        with self.assertRaises(RuntimeError):
            tb2 = TypeBlocks.from_blocks(
                    tb1._assign_from_boolean_blocks_by_unit(
                           targets=targets[:1],
                           value=value,
                           value_valid=targets,
                           ))


    def test_type_blocks_assign_blocks_k(self) -> None:

        a1 = np.arange(9).reshape(3, 3)
        tb1 = TypeBlocks.from_blocks((a1,))

        targets = np.full(tb1.shape, False)
        targets[1, 1:] = True

        value = np.full(tb1.shape, None)
        value[1, 1:] = ('a', 'b')

        tb2 = TypeBlocks.from_blocks(
                tb1._assign_from_boolean_blocks_by_unit(
                       targets=(targets,),
                       value=value,
                       value_valid=targets
                       ))

        self.assertEqual(tb2.values.tolist(),
                [[0, 1, 2], [3, 'a', 'b'], [6, 7, 8]])

    #--------------------------------------------------------------------------
    def test_type_blocks_assign_blocks_from_keys_by_blocks_a(self) -> None:

        a1 = np.array([[True, True, True], [True, True, True], [True, True, True]])
        a2 = np.array([[False, False, False], [False, False, False], [False, False, False]])
        a3 = np.array([[True, True, True], [True, True, True], [True, True, True]])

        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        values = [np.array([3, 4, 5]), np.array(['a', 'b', 'c'])]

        tb2 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_blocks(
                        row_key=None,
                        column_key=[2, 7],
                        values=values,
                        ))
        self.assertEqual([dt.kind for dt in tb2.dtypes],
                ['b', 'b', 'i', 'b', 'b', 'b', 'b', 'U', 'b'])
        self.assertEqual(tb2.iloc[1].values.tolist(),
                [[True, True, 4, False, False, False, True, 'b', True]])

        values = [np.array([3, 4, 5]),]

        tb3 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_blocks(
                        row_key=None,
                        column_key=5,
                        values=values,
                        ))
        self.assertEqual([dt.kind for dt in tb3.dtypes],
                ['b', 'b', 'b', 'b', 'b', 'i', 'b', 'b', 'b'])
        self.assertEqual(tb3.iloc[1].values.tolist(),
                [[True, True, True, False, False, 4, True, True, True]])


        values = [np.array([[3, 3], [4, 4], [5, 5]]),]

        tb4 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_blocks(
                        row_key=None,
                        column_key=slice(2, 4),
                        values=values,
                        ))
        self.assertEqual([dt.kind for dt in tb4.dtypes],
                ['b', 'b', 'i', 'i', 'b', 'b', 'b', 'b', 'b'])
        self.assertEqual(tb4.iloc[1].values.tolist(),
                [[True, True, 4, 4, False, False, True, True, True]])


        values = [np.array([[3, 3], [4, 4], [5, 5]]),
                np.array(['a', 'b', 'c'])]

        tb5 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_blocks(
                        row_key=None,
                        column_key=[2, 4, 7],
                        values=values,
                        ))

        self.assertEqual([dt.kind for dt in tb5.dtypes],
                ['b', 'b', 'i', 'b', 'i', 'b', 'b', 'U', 'b'])
        self.assertEqual(tb5.iloc[1].values.tolist(),
                [[True, True, 4, False, 4, False, True, 'b', True]])


    def test_type_blocks_assign_blocks_from_keys_by_blocks_b(self) -> None:

        a1 = np.array([[True, True, True], [True, True, True], [True, True, True]])
        a2 = np.array([[False, False, False], [False, False, False], [False, False, False]])
        a3 = np.array([[True, True, True], [True, True, True], [True, True, True]])

        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        values = [np.array([3, 4]), np.array(['a', 'b'])]

        tb2 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_blocks(
                        row_key=slice(1, None),
                        column_key=[2, 7],
                        values=values,
                        ))

        self.assertEqual([dt.kind for dt in tb2.dtypes],
                ['b', 'b', 'O', 'b', 'b', 'b', 'b', 'O', 'b'])
        self.assertEqual(tb2.values.tolist(),
                [[True, True, True, False, False, False, True, True, True], [True, True, 3, False, False, False, True, 'a', True], [True, True, 4, False, False, False, True, 'b', True]])

        tb3 = TypeBlocks.from_blocks(
                tb1._assign_from_iloc_by_blocks(
                        row_key=[0, 2],
                        column_key=[2, 7],
                        values=values,
                        ))

        self.assertEqual([dt.kind for dt in tb3.dtypes],
                ['b', 'b', 'O', 'b', 'b', 'b', 'b', 'O', 'b'])
        self.assertEqual(tb3.values.tolist(),
                [[True, True, 3, False, False, False, True, 'a', True], [True, True, True, False, False, False, True, True, True], [True, True, 4, False, False, False, True, 'b', True]]
                )




    #--------------------------------------------------------------------------

    def test_type_blocks_assign_blocks_value_arrays_a(self) -> None:

        a1 = np.array([[3, 20], [4, 80],])
        a2 = np.array([False, True])
        a3 = np.array([['df', 'er'], ['fd', 'ij'],])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        targets = (np.array([[False, True], [False, True]]),
                np.full(2, True),
                np.array([[False, True], [True, False]]))

        values = (np.array([None, None]),
                np.array([100, 200]),
                np.array([True, False]),
                np.array([100, 200]),
                np.array([500, 700]),
                )
        tb2 = TypeBlocks.from_blocks(
                tb1._assign_from_boolean_blocks_by_blocks(
                       targets=targets,
                       values=values
                       ))

        self.assertEqual([dt.kind for dt in tb2.dtypes],
                ['i', 'i', 'b', 'O', 'O'])
        self.assertEqual(tb2.values.tolist(),
                [[3, 100, True, 'df', 500], [4, 200, False, 200, 'ij']]
                )

    def test_type_blocks_assign_blocks_value_arrays_b(self) -> None:

        a1 = np.array([4, 30])
        a2 = np.array([False, True])
        a3 = np.array([False, True])
        a4 = np.array(['df', 'er'])
        a5 = np.array([None, np.nan])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3, a4, a5))

        targets = (np.array([False, True]),
                np.array([True, False]),
                np.array([True, False]),
                np.array([True, True]),
                np.array([False, True]),
                )
        values = (np.array([0, 1.5]),
                np.array([100, 200]),
                np.array([True, False]),
                np.array(['fooo', 'bar']),
                np.array([500, 700]),
                )
        tb2 = TypeBlocks.from_blocks(
                tb1._assign_from_boolean_blocks_by_blocks(
                       targets=targets,
                       values=values
                       ))

        self.assertEqual([dt.kind for dt in tb2.dtypes],
                ['f', 'O', 'b', 'U', 'O'])
        self.assertEqual(tb2.values.tolist(),
                [[4.0, 100, True, 'fooo', None], [1.5, True, True, 'bar', 700]]
                )

    def test_type_blocks_assign_blocks_value_arrays_c(self) -> None:

        a1 = np.array([[3, 20, -5], [4, 80, -20],])
        a2 = np.array([['df', 'er', 'er'], ['fd', 'ij', 'we'],])
        tb1 = TypeBlocks.from_blocks((a1, a2))

        targets = (np.array([[False, True, False], [False, True, True]]),
                np.array([[False, True, False], [False, True, True]]))

        values = (np.array([None, None]),
                np.array([True, False]),
                np.array([1.5, 2.5]),
                np.array([100, 200]),
                np.array([500, 700]),
                np.array([None, False]),
                )
        tb2 = TypeBlocks.from_blocks(
                tb1._assign_from_boolean_blocks_by_blocks(
                       targets=targets,
                       values=values
                       ))

        self.assertEqual([dt.kind for dt in tb2.dtypes],
                ['i', 'b', 'f', 'U', 'i', 'O'])
        self.assertEqual(tb2.values.tolist(),
                [[3, True, -5.0, 'df', 500, 'er'], [4, False, 2.5, 'fd', 700, False]]
                )

    #--------------------------------------------------------------------------
    def test_type_blocks_group_a(self) -> None:

        a1 = np.array([
                [1, 2, 3,4],
                [4,2,6,3],
                [0, 0, 1,2],
                [0, 0, 1,1]
                ])
        a2 = np.array([[False, False, True],
                [False, False, True],
                [True, False, True],
                [True, False, True]])

        tb1 = TypeBlocks.from_blocks((a1, a2))

        # return rows, by columns key 1
        groups = list(tb1.group(axis=0, key=1))

        self.assertEqual(len(groups), 2)


        group, selection, subtb = groups[0]
        self.assertEqual(group, 0)
        self.assertEqual(subtb.values.tolist(),
                [[0, 0, 1, 2, True, False, True], [0, 0, 1, 1, True, False, True]])


        group, selection, subtb = groups[1]
        self.assertEqual(group, 2)
        self.assertEqual(subtb.values.tolist(),
                [[1, 2, 3, 4, False, False, True], [4, 2, 6, 3, False, False, True]])


    def test_type_blocks_group_b(self) -> None:

        a1 = np.array([
                [1, 2, 3,4],
                [4,2,6,3],
                [0, 0, 1,2],
                [0, 0, 1,1]
                ])
        a2 = np.array([[False, False, True],
                [False, False, True],
                [True, False, True],
                [True, False, True]])

        tb1 = TypeBlocks.from_blocks((a1, a2))

        # return rows, by columns key [4, 5]
        groups = list(tb1.group(axis=0, key=[4, 5]))

        self.assertEqual(len(groups), 2)


        group, selection, subtb = groups[0]
        self.assertEqual(group, (False, False))
        self.assertEqual(subtb.values.tolist(),
                [[1, 2, 3, 4, False, False, True], [4, 2, 6, 3, False, False, True]]
                )

        group, selection, subtb = groups[1]
        self.assertEqual(group, (True, False))
        self.assertEqual(subtb.values.tolist(),
                [[0, 0, 1, 2, True, False, True], [0, 0, 1, 1, True, False, True]])


    def test_type_blocks_transpose_a(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        tb2 = tb1.transpose()

        self.assertEqual(tb2.values.tolist(),
                [[1, 4, 0], [2, 5, 0], [3, 6, 1], [False, True, True], [False, False, False], [True, True, True], ['a', 'c', 'oe'], ['b', 'd', 'od']])

        self.assertEqual(tb1.transpose().transpose().values.tolist(),
                tb1.values.tolist())



    def test_type_blocks_display_a(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb = TypeBlocks.from_blocks((a1, a2, a3))

        disp = tb.display()
        self.assertEqual(len(disp), 5)


    #---------------------------------------------------------------------------

    def test_type_blocks_axis_values_a(self) -> None:
        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertEqual(
            list(a.tolist() for a in tb.axis_values(1)),
            [[1, 2, 3, False, False, True, 'a', 'b'], [4, 5, 6, True, False, True, 'c', 'd'], [0, 0, 1, True, False, True, 'oe', 'od']]
            )

        self.assertEqual(list(a.tolist() for a in tb.axis_values(0)),
            [[1, 4, 0], [2, 5, 0], [3, 6, 1], [False, True, True], [False, False, False], [True, True, True], ['a', 'c', 'oe'], ['b', 'd', 'od']]
            )

        # we are iterating over slices so we get views of columns without copying
        self.assertEqual(tb.mloc[0], mloc(next(tb.axis_values(0))))


    def test_type_blocks_axis_values_b(self) -> None:
        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertEqual(
                [a.tolist() for a in tb.axis_values(axis=0, reverse=True)],
                [['b', 'd', 'od'], ['a', 'c', 'oe'], [True, True, True], [False, False, False], [False, True, True], [3, 6, 1], [2, 5, 0], [1, 4, 0]])
        self.assertEqual([a.tolist() for a in tb.axis_values(axis=0, reverse=False)],
                [[1, 4, 0], [2, 5, 0], [3, 6, 1], [False, True, True], [False, False, False], [True, True, True], ['a', 'c', 'oe'], ['b', 'd', 'od']])


    def test_type_blocks_axis_values_c(self) -> None:
        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb = TypeBlocks.from_blocks((a1, a2, a3))

        with self.assertRaises(AxisInvalid):
            _ = next(tb.axis_values(-1))


    #---------------------------------------------------------------------------
    def test_type_blocks_extract_iloc_mask_a(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertEqual(tb.extract_iloc_mask((slice(None), [4, 5])).values.tolist(),
                [[False, False, False, False, True, True, False, False], [False, False, False, False, True, True, False, False], [False, False, False, False, True, True, False, False]])

        self.assertEqual(tb.extract_iloc_mask(([0,2], slice(None))).values.tolist(),
                [[True, True, True, True, True, True, True, True], [False, False, False, False, False, False, False, False], [True, True, True, True, True, True, True, True]]
                )

        self.assertEqual(tb.extract_iloc_mask(([0,2], [3,7])).values.tolist(),
                [[False, False, False, True, False, False, False, True], [False, False, False, False, False, False, False, False], [False, False, False, True, False, False, False, True]]
                )

        self.assertEqual(tb.extract_iloc_mask((slice(1, None), slice(4, None))).values.tolist(),
                [[False, False, False, False, False, False, False, False], [False, False, False, False, True, True, True, True], [False, False, False, False, True, True, True, True]])




    def test_type_blocks_extract_iloc_assign_a(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb = TypeBlocks.from_blocks((a1, a2, a3))


        self.assertEqual(tb.extract_iloc_assign_by_unit((1, None), 600).values.tolist(),
                [[1, 2, 3, False, False, True, 'a', 'b'],
                [600, 600, 600, 600, 600, 600, 600, 600],
                [0, 0, 1, True, False, True, 'oe', 'od']])


    def test_type_blocks_extract_iloc_assign_b(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertEqual(tb.extract_iloc_assign_by_unit((1, 5), 20).values.tolist(),
                [[1, 2, 3, False, False, True, 'a', 'b'],
                [4, 5, 6, True, False, 20, 'c', 'd'],
                [0, 0, 1, True, False, True, 'oe', 'od']])


    def test_type_blocks_extract_iloc_assign_c(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertEqual(tb.extract_iloc_assign_by_unit((slice(2), slice(5)), 'X').values.tolist(),
                [['X', 'X', 'X', 'X', 'X', True, 'a', 'b'],
                ['X', 'X', 'X', 'X', 'X', True, 'c', 'd'],
                [0, 0, 1, True, False, True, 'oe', 'od']]
                )


    def test_type_blocks_extract_iloc_assign_d(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertEqual(tb.extract_iloc_assign_by_unit(([0,1], [1,4,7]), -5).values.tolist(),
                [[1, -5, 3, False, -5, True, 'a', -5],
                [4, -5, 6, True, -5, True, 'c', -5],
                [0, 0, 1, True, False, True, 'oe', 'od']])


    def test_type_blocks_extract_iloc_assign_e(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertEqual(
                tb.extract_iloc_assign_by_unit((1, slice(4)), (-1, -2, -3, -4)).values.tolist(),
                [[1, 2, 3, False, False, True, 'a', 'b'],
                [-1, -2, -3, -4, False, True, 'c', 'd'],
                [0, 0, 1, True, False, True, 'oe', 'od']])


    def test_type_blocks_extract_iloc_assign_f(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertEqual(
                tb.extract_iloc_assign_by_unit((2, slice(3,7)), (-1, -2, -3, -4)).values.tolist(),
                [[1, 2, 3, False, False, True, 'a', 'b'],
                [4, 5, 6, True, False, True, 'c', 'd'],
                [0, 0, 1, -1, -2, -3, -4, 'od']])


    def test_type_blocks_extract_iloc_assign_g(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertEqual(
                tb.extract_iloc_assign_by_unit((0, slice(4,8)), (-1, -2, -3, -4)).values.tolist(),
                [[1, 2, 3, False, -1, -2, -3, -4],
                [4, 5, 6, True, False, True, 'c', 'd'],
                [0, 0, 1, True, False, True, 'oe', 'od']])


    #---------------------------------------------------------------------------

    def test_type_blocks_elements_items_a(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        a4 = np.array([None, None, None])
        tb = TypeBlocks.from_blocks((a1, a2, a4, a3))

        post = [x for x in tb.element_items()]

        self.assertEqual(post,
                [((0, 0), 1), ((0, 1), 2), ((0, 2), 3), ((0, 3), False), ((0, 4), False), ((0, 5), True), ((0, 6), None), ((0, 7), 'a'), ((0, 8), 'b'), ((1, 0), 4), ((1, 1), 5), ((1, 2), 6), ((1, 3), True), ((1, 4), False), ((1, 5), True), ((1, 6), None), ((1, 7), 'c'), ((1, 8), 'd'), ((2, 0), 0), ((2, 1), 0), ((2, 2), 1), ((2, 3), True), ((2, 4), False), ((2, 5), True), ((2, 6), None), ((2, 7), 'oe'), ((2, 8), 'od')]
                )

        tb2 = TypeBlocks.from_element_items(post, tb.shape, tb._row_dtype)
        self.assertTrue((tb.values == tb2.values).all())

    def test_type_blocks_elements_items_b(self) -> None:
        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([None, None, None])
        tb = TypeBlocks.from_blocks((a1, a2))

        post = [x for x in tb.element_items(axis=1)]

        self.assertEqual(post,
                [((0, 0), 1), ((1, 0), 4), ((2, 0), 0), ((0, 1), 2), ((1, 1), 5), ((2, 1), 0), ((0, 2), 3), ((1, 2), 6), ((2, 2), 1), ((0, 3), None), ((1, 3), None), ((2, 3), None)]
)


    #---------------------------------------------------------------------------
    def test_type_blocks_reblock_signature_a(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]], dtype=np.int64)
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        a4 = np.array([None, None, None])
        tb = TypeBlocks.from_blocks((a1, a2, a4, a3))

        dtype = np.dtype
        self.assertEqual(
                list(tb._reblock_signature()),
                [(dtype('int64'), 3), (dtype('bool'), 3), (dtype('O'), 1), (dtype('<U2'), 2)])

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]], dtype=np.int64)
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        a4 = np.array([None, None, None])
        tb = TypeBlocks.from_blocks((a1, a2, a3, a4))

        self.assertEqual(
                list(tb._reblock_signature()),
                [(dtype('int64'), 3), (dtype('bool'), 3), (dtype('<U2'), 2), (dtype('O'), 1)])


    #---------------------------------------------------------------------------

    def test_type_blocks_copy_a(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        a4 = np.array([None, None, None])
        tb1 = TypeBlocks.from_blocks((a1, a2, a4, a3))

        tb2 = tb1.copy()
        tb1.append(np.array((1, 2, 3)))

        self.assertEqual(tb2.shape, (3, 9))
        self.assertEqual(tb1.shape, (3, 10))

        self.assertEqual(tb1.iloc[2].values.tolist(),
                [[0, 0, 1, True, False, True, None, 'oe', 'od', 3]])

        self.assertEqual(tb2.iloc[2].values.tolist(),
                [[0, 0, 1, True, False, True, None, 'oe', 'od']])


    def test_type_blocks_copy_b(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        tb2 = copy.copy(tb1)

        self.assertEqual([id(a) for a in tb1._blocks], [id(a) for a in tb2._blocks])


    #---------------------------------------------------------------------------
    def test_type_blocks_isna_a(self) -> None:

        a1 = np.array([[1, 2, 3], [4, np.nan, 6], [0, 0, 1]], dtype=object)
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        a4 = np.array([None, None, None])
        tb1 = TypeBlocks.from_blocks((a1, a2, a4, a3))

        self.assertEqual(tb1.isna().values.tolist(),
                [[False, False, False, False, False, False, True, False, False], [False, True, False, False, False, False, True, False, False], [False, False, False, False, False, False, True, False, False]])

        self.assertEqual(tb1.notna().values.tolist(),
                [[True, True, True, True, True, True, False, True, True], [True, False, True, True, True, True, False, True, True], [True, True, True, True, True, True, False, True, True]])

    #---------------------------------------------------------------------------

    def test_type_blocks_clip_a(self) -> None:

        a1 = np.array([[-10, 2], [30, 6], [1, 200]], dtype=float)
        a2 = np.array([[False, False], [True, False], [True, False]])
        tb1 = TypeBlocks.from_blocks((a1, a2))

        tb2 = tb1.clip(1, 4)
        self.assertEqual([dt.kind for dt in tb2.dtypes],
                ['f', 'f', 'i', 'i'])
        self.assertEqual(tb2.values.tolist(),
                [[1.0, 2.0, 1.0, 1.0],
                [4.0, 4.0, 1.0, 1.0],
                [1.0, 4.0, 1.0, 1.0]])

    def test_type_blocks_clip_b(self) -> None:

        a1 = np.array([[-10, 2], [30, 6], [1, 200]], dtype=float)
        a2 = np.array([[False, False], [True, False], [True, False]])
        tb1 = TypeBlocks.from_blocks((a1, a2))

        tb2 = tb1.clip(np.full(tb1.shape, 1), np.full(tb1.shape, 4))

        self.assertEqual([dt.kind for dt in tb2.dtypes],
                ['f', 'f', 'i', 'i'])
        self.assertEqual(tb2.values.tolist(),
                [[1.0, 2.0, 1.0, 1.0],
                [4.0, 4.0, 1.0, 1.0],
                [1.0, 4.0, 1.0, 1.0]])

    def test_type_blocks_clip_c(self) -> None:

        a1 = np.array([[-10, 2], [30, 6], [1, 200]], dtype=float)
        a2 = np.array([[False, False], [True, False], [True, False]])
        tb1 = TypeBlocks.from_blocks((a1, a2))

        lb = (np.array((2, 2, -1)),
                np.array((2, 2, -1)),
                np.array((2, 2, -1)),
                np.array((2, 2, -1)))

        tb2 = tb1.clip(lb, np.full(tb1.shape, 4))
        self.assertEqual([dt.kind for dt in tb2.dtypes],
                ['f', 'f', 'i', 'i'])

        self.assertEqual(tb2.values.tolist(),
                [[2.0, 2.0, 2.0, 2.0],
                [4.0, 4.0, 2.0, 2.0],
                [1.0, 4.0, 1.0, 0.0]])


    def test_type_blocks_clip_d(self) -> None:

        a1 = np.array([[10, 2], [10, 2], [10, 2]], dtype=float)
        a2 = np.array([[True, False], [True, False], [True, False]])
        tb1 = TypeBlocks.from_blocks((a1, a2))


        ub = (np.array((8, 6, 8)),
                np.array((2, 0, 2)),
                np.array((2, 0, 2)),
                np.array((0, 1, 0)))

        tb2 = tb1.clip(None, ub)
        self.assertEqual([dt.kind for dt in tb2.dtypes],
                ['f', 'f', 'i', 'i'])

        self.assertEqual(tb2.values.tolist(),
                [[8.0, 2.0, 1.0, 0.0],
                [6.0, 0.0, 0.0, 0.0],
                [8.0, 2.0, 1.0, 0.0]])


    def test_type_blocks_clip_e(self) -> None:

        a1 = np.array([[10, 2, 10, 2], [10, 2, 10, 2], [10, 2, 10, 2]], dtype=float)
        tb1 = TypeBlocks.from_blocks((a1,))

        ub = (np.full((3, 2), 5), np.full((3, 2), 0))
        tb2 = tb1.clip(None, ub)
        self.assertEqual(tb2.values.tolist(),
                [[5.0, 2.0, 0.0, 0.0],
                [5.0, 2.0, 0.0, 0.0],
                [5.0, 2.0, 0.0, 0.0]])

        a1 = np.array([[10, 2, 10], [10, 2, 10], [10, 2, 10]], dtype=float)
        a2 = np.array([2, 2, 2])
        tb3 = TypeBlocks.from_blocks((a1, a2))

        tb4 = tb3.clip(None, ub)
        self.assertEqual(tb4.values.tolist(),
                [[5.0, 2.0, 0.0, 0.0],
                [5.0, 2.0, 0.0, 0.0],
                [5.0, 2.0, 0.0, 0.0]])

    def test_type_blocks_clip_f(self) -> None:

        a1 = np.array([[10, 2, 10, 2], [10, 2, 10, 2], [10, 2, 10, 2]], dtype=float)
        tb1 = TypeBlocks.from_blocks((a1,))

        ub = (np.full((3, 1), 5), np.full((3, 3), 0))
        tb2 = tb1.clip(None, ub)

        self.assertEqual(tb2.values.tolist(),
                [[5.0, 0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0, 0.0]])

        a1 = np.array([[10, 2, 10], [10, 2, 10], [10, 2, 10]], dtype=float)
        a2 = np.array([2, 2, 2])
        tb3 = TypeBlocks.from_blocks((a1, a2))

        tb4 = tb3.clip(None, ub)
        self.assertEqual(tb4.values.tolist(),
                [[5.0, 0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0, 0.0]])



    #---------------------------------------------------------------------------

    def test_type_blocks_dropna_to_slices(self) -> None:

        a1 = np.array([
                [1,np.nan,3, 4],
                [4, np.nan, 6, 2],
                [np.nan, np.nan, np.nan, np.nan]
                ], dtype=object)
        a2 = np.array([
                [1,np.nan,3, 4],
                [4, np.nan, 6, 2],
                [np.nan, np.nan, np.nan, np.nan]
                ], dtype=object)

        tb1 = TypeBlocks.from_blocks((a1, a2))

        row_key, column_key = tb1.drop_missing_to_keep_locations(axis=1, func=isna_array)
        assert column_key is not None

        self.assertEqual(column_key.tolist(),
                [True, False, True, True, True, False, True, True])
        self.assertEqual(row_key, None)

        row_key, column_key = tb1.drop_missing_to_keep_locations(axis=0, func=isna_array)
        assert row_key is not None
        self.assertEqual(row_key.tolist(),
                [True, True, False])
        self.assertEqual(column_key, None)


    def test_type_blocks_fillna_a(self) -> None:

        a1 = np.array([
                [1,np.nan,3, 4],
                [4, np.nan, 6, 2],
                [np.nan, np.nan, np.nan, np.nan]
                ], dtype=float)
        a2 = np.array([
                [1,np.nan,3, 4],
                [4, np.nan, 6, 2],
                [np.nan, np.nan, np.nan, np.nan]
                ], dtype=object)

        tb1 = TypeBlocks.from_blocks((a1, a2))
        tb2 = tb1.fill_missing_by_unit(0, func=isna_array)
        self.assertEqual([b.dtype for b in tb2._blocks],
                [np.dtype('float64'), np.dtype('O')])
        self.assertEqual(tb2.isna().values.any(), False)

        tb3 = tb1.fill_missing_by_unit(None, func=isna_array)
        self.assertEqual([b.dtype for b in tb3._blocks],
                [np.dtype('O'), np.dtype('O')])
        # we ahve Nones, which are na
        self.assertEqual(tb3.isna().values.any(), True)



    def test_type_blocks_fillna_trailing_a(self) -> None:

        for axis in (0, 1):
            for arrays in self.get_arrays_b():
                tb = TypeBlocks.from_blocks(arrays)
                post = tb.fillna_trailing(-1, axis=axis)
                self.assertEqual(tb.shape, post.shape)


    def test_type_blocks_fillna_trailing_b(self) -> None:

        a1 = np.array([
                [nan, nan,3, 4],
                [nan, nan, 6, nan],
                [5, nan, nan, nan]
                ], dtype=float)
        a2 = np.array([nan, nan, nan], dtype=object)

        tb1 = TypeBlocks.from_blocks((a1, a2))

        tb2 = tb1.fillna_trailing(0, axis=0)

        self.assertAlmostEqualValues(
                list(tb2.values.flat),
                [nan, 0.0, 3.0, 4.0, 0, nan, 0.0, 6.0, 0.0, 0, 5.0, 0.0, 0.0, 0.0, 0])


    def test_type_blocks_fillna_trailing_c(self) -> None:

        a2 = np.array([
                [None, None, None, None],
                [None, 1, None, 6],
                [None, 5, None, None]
                ], dtype=object)
        a1 = np.array([None, None, None], dtype=object)
        a3 = np.array([
                [None, 4],
                [None, 1],
                [None, 5]
                ], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))
        tb2 = tb1.fillna_trailing(0, axis=1)

        # no change as no leading values are NaN
        self.assertEqual(tb1.values.tolist(), tb2.values.tolist())


    def test_type_blocks_fillna_trailing_d(self) -> None:

        a2 = np.array([
                [None, None, None, None],
                [None, 1, None, 6],
                [None, 5, None, None]
                ], dtype=object)
        a1 = np.array([None, None, None], dtype=object)
        a3 = np.array([
                [None, None],
                [None, 1],
                [None, 5]
                ], dtype=object)
        tb1 = TypeBlocks.from_blocks((a3, a2, a1))
        tb2 = tb1.fillna_trailing(0, axis=1)

        self.assertEqual(tb2.values.tolist(),
                [[0, 0, 0, 0, 0, 0, 0],
                [None, 1, None, 1, None, 6, 0],
                [None, 5, None, 5, 0, 0, 0]])


    def test_type_blocks_fillna_trailing_e(self) -> None:

        a1 = np.array([None, None, None], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1,))
        with self.assertRaises(NotImplementedError):
            tb1.fillna_trailing(value=3, axis=2)


    def test_type_blocks_fillna_leading_a(self) -> None:

        for axis in (0, 1):
            for arrays in self.get_arrays_b():
                tb = TypeBlocks.from_blocks(arrays)
                post = tb.fillna_leading(-1, axis=axis)
                self.assertEqual(tb.shape, post.shape)


    def test_type_blocks_fillna_leading_b(self) -> None:

        a1 = np.array([
                [nan, nan,3, 4],
                [nan, nan, 6, nan],
                [5, nan, nan, nan]
                ], dtype=float)
        a2 = np.array([nan, nan, nan], dtype=object)

        tb1 = TypeBlocks.from_blocks((a1, a2))

        tb2 = tb1.fillna_leading(0)

        self.assertAlmostEqualValues(list(tb2.values.flat),
                [0.0, 0.0, 3.0, 4.0, 0, 0.0, 0.0, 6.0, nan, 0, 5.0, 0.0, nan, nan, 0])



    def test_type_blocks_fillna_leading_c(self) -> None:

        a2 = np.array([
                [None, None, 3, 4],
                [1, None, None, 6],
                [5, None, None, None]
                ], dtype=object)
        a1 = np.array([1, None, None], dtype=object)

        tb1 = TypeBlocks.from_blocks((a2, a1))

        tb2 = tb1.fillna_leading(-1, axis=1)
        self.assertEqual(tb2.values.tolist(),
                [[-1, -1, 3, 4, 1],
                [1, None, None, 6, None],
                [5, None, None, None, None]])

    def test_type_blocks_fillna_leading_d(self) -> None:

        a2 = np.array([
                [None, None, 3, 4],
                [None, None, None, 6],
                [None, None, None, None]
                ], dtype=object)
        a1 = np.array([1, None, None], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2))
        tb2 = tb1.fillna_leading(0, axis=1)

        self.assertEqual(tb2.values.tolist(),
                [[1, None, None, 3, 4],
                [0, 0, 0, 0, 6],
                [0, 0, 0, 0, 0]])

    def test_type_blocks_fillna_leading_e(self) -> None:

        a2 = np.array([
                [None, None, None, None],
                [None, 1, None, 6],
                [None, 5, None, None]
                ], dtype=object)
        a1 = np.array([None, None, None], dtype=object)
        a3 = np.array([
                [None, 4],
                [None, 1],
                [None, 5]
                ], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))
        tb2 = tb1.fillna_leading(0, axis=1)

        self.assertEqual(tb2.values.tolist(),
                [[0, 0, 0, 0, 0, 0, 4],
                [0, 0, 1, None, 6, None, 1],
                [0, 0, 5, None, None, None, 5]])


    def test_type_blocks_fillna_leading_f(self) -> None:

        a1 = np.array([None, None, None], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1,))
        with self.assertRaises(NotImplementedError):
            tb1.fillna_leading(value=3, axis=2)


    def test_type_blocks_fillna_leading_g(self) -> None:

        a1 = np.array([None, None, None], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1,))
        with self.assertRaises(RuntimeError):
            tb1.fillna_leading(value=np.array((3, 4)), axis=1)
        with self.assertRaises(RuntimeError):
            tb1.fillna_leading(value=np.array((3, 4)), axis=0)


    #---------------------------------------------------------------------------

    def test_type_blocks_fillna_forward_a(self) -> None:

        for axis in (0, 1):
            for arrays in self.get_arrays_b():
                tb = TypeBlocks.from_blocks(arrays)
                post1 = tb.fillna_forward(axis=axis)
                self.assertEqual(tb.shape, post1.shape)

                post2 = tb.fillna_backward(axis=axis)
                self.assertEqual(tb.shape, post2.shape)



    def test_type_blocks_fillna_forward_b(self) -> None:

        a1 = np.array([
                [nan, nan,3, 4],
                [nan, nan, 6, nan],
                [5, nan, nan, nan]
                ], dtype=float)
        a2 = np.array([nan, nan, nan], dtype=object)

        tb1 = TypeBlocks.from_blocks((a1, a2))
        tb2 = tb1.fillna_forward()

        self.assertEqual(
                tb2.fill_missing_by_unit(0, func=isna_array).values.tolist(),
                [[0.0, 0.0, 3.0, 4.0, 0],
                [0.0, 0.0, 6.0, 4.0, 0],
                [5.0, 0.0, 6.0, 4.0, 0]]
                )

        tb3 = tb1.fillna_backward()
        self.assertEqual(tb3.fill_missing_by_unit(0, func=isna_array).values.tolist(),
                [[5.0, 0.0, 3.0, 4.0, 0],
                [5.0, 0.0, 6.0, 0.0, 0],
                [5.0, 0.0, 0.0, 0.0, 0]]
                )

    def test_type_blocks_fillna_forward_c(self) -> None:

        a1 = np.array([
                [nan, nan,3, 4],
                [nan, nan, 6, nan],
                [5, nan, nan, nan]
                ], dtype=float)
        a2 = np.array([nan, nan, nan], dtype=object)

        tb1 = TypeBlocks.from_blocks((a1, a2))
        tb2 = tb1.fillna_forward(axis=1)

        self.assertEqual(
            tb2.fill_missing_by_unit(0, func=isna_array).values.tolist(),
            [[0.0, 0.0, 3.0, 4.0, 4.0], [0.0, 0.0, 6.0, 6.0, 6.0], [5.0, 5.0, 5.0, 5.0, 5.0]]
        )

        tb3 = tb1.fillna_backward(axis=1)
        self.assertEqual(
            tb3.fill_missing_by_unit(0, func=isna_array).values.tolist(),
            [[3.0, 3.0, 3.0, 4.0, 0], [6.0, 6.0, 6.0, 0, 0], [5.0, 0, 0, 0, 0]]
            )

    def test_type_blocks_fillna_forward_d(self) -> None:

        a1 = np.array([
                [None, 10, None],
                [None, 88, None],
                [None, 40, None]
                ], dtype=object)
        a2 = np.array([None, None, None], dtype=object)
        a3 = np.array([543, 601, 234], dtype=object)

        tb1 = TypeBlocks.from_blocks((a3, a2, a1))
        tb2 = tb1.fillna_forward(axis=1)

        self.assertEqual(tb2.values.tolist(),
                [[543, 543, 543, 10, 10],
                [601, 601, 601, 88, 88],
                [234, 234, 234, 40, 40]]
                )

        tb3 = tb1.fillna_backward(axis=1)
        self.assertEqual(tb3.values.tolist(),
                [[543, 10, 10, 10, None],
                [601, 88, 88, 88, None],
                [234, 40, 40, 40, None]]
                )

    def test_type_blocks_fillna_forward_e(self) -> None:

        a1 = np.array([None, None, None], dtype=object)
        a2 = np.array([None, 8, None], dtype=object)
        a3 = np.array([543, 601, 234], dtype=object)
        a4 = np.array([30, None, 74], dtype=object)

        tb1 = TypeBlocks.from_blocks((a2, a1, a1, a4, a1, a3, a1))

        # axis 1 tests
        self.assertEqual(tb1.fillna_forward(axis=1).values.tolist(),
                [[None, None, None, 30, 30, 543, 543],
                [8, 8, 8, 8, 8, 601, 601],
                [None, None, None, 74, 74, 234, 234]])

        self.assertEqual(tb1.fillna_backward(axis=1).values.tolist(),
                [[30, 30, 30, 30, 543, 543, None],
                [8, 601, 601, 601, 601, 601, None],
                [74, 74, 74, 74, 234, 234, None]]
                )


    def test_type_blocks_fillna_forward_f(self) -> None:

        a1 = np.array([None, None, 40], dtype=object)
        a2 = np.array([None, 8, None], dtype=object)
        a3 = np.array([543, 601, 234], dtype=object)
        a4 = np.array([30, None, 74], dtype=object)

        tb1 = TypeBlocks.from_blocks((a2, a1, a4, a3))

        self.assertEqual(
                tb1.fillna_forward().values.tolist(),
                [[None, None, 30, 543],
                [8, None, 30, 601],
                [8, 40, 74, 234]]
                )
        self.assertEqual(tb1.fillna_backward().values.tolist(),
                [[8, 40, 30, 543],
                [8, 40, 74, 601],
                [None, 40, 74, 234]]
                )




    def test_type_blocks_fillna_forward_g(self) -> None:

        a1 = np.array([None, None, None], dtype=object)
        a2 = np.array([None, 8, None], dtype=object)
        a3 = np.array([543, 601, 234], dtype=object)
        a4 = np.array([30, None, 74], dtype=object)

        tb1 = TypeBlocks.from_blocks((a2, a1, a1, a4, a1, a1, a3, a1))

        self.assertEqual(tb1.fillna_forward(limit=1, axis=1).values.tolist(),
            [[None, None, None, 30, 30, None, 543, 543], [8, 8, None, None, None, None, 601, 601], [None, None, None, 74, 74, None, 234, 234]]
            )

        self.assertEqual(tb1.fillna_forward(limit=2, axis=1).values.tolist(),
            [[None, None, None, 30, 30, 30, 543, 543], [8, 8, 8, None, None, None, 601, 601], [None, None, None, 74, 74, 74, 234, 234]])


        self.assertEqual(tb1.fillna_forward(limit=3, axis=1).values.tolist(),
            [[None, None, None, 30, 30, 30, 543, 543], [8, 8, 8, 8, None, None, 601, 601], [None, None, None, 74, 74, 74, 234, 234]]
            )

        self.assertEqual(tb1.fillna_backward(limit=1, axis=1).values.tolist(),
            [[None, None, 30, 30, None, 543, 543, None], [8, None, None, None, None, 601, 601, None], [None, None, 74, 74, None, 234, 234, None]])



    def test_type_blocks_fillna_forward_h(self) -> None:

        a1 = np.array([
                [None, None, 10, None],
                [None, None, 88, None],
                [None, None, 40, None]
                ], dtype=object)
        a2 = np.array([
                [None, 3, None, None],
                [None, None, 4, None],
                [3, None, None, None]
                ], dtype=object)
        a3 = np.array([None, None, None], dtype=object)
        a4 = np.array([543, 601, 234], dtype=object)

        tb1 = TypeBlocks.from_blocks((a4, a3, a2, a1))

        self.assertEqual(
                tb1.fillna_forward(axis=1, limit=1).values.tolist(),
                [[543, 543, None, 3, 3, None, None, None, 10, 10], [601, 601, None, None, 4, 4, None, None, 88, 88], [234, 234, 3, 3, None, None, None, None, 40, 40]])

        self.assertEqual(
                tb1.fillna_forward(axis=1, limit=3).values.tolist(),
                [[543, 543, 543, 3, 3, 3, 3, None, 10, 10], [601, 601, 601, 601, 4, 4, 4, 4, 88, 88], [234, 234, 3, 3, 3, 3, None, None, 40, 40]])


    def test_type_blocks_fillna_forward_i(self) -> None:

        a1 = np.array([
                [None, None, 10, None],
                [23, None, 88, None],
                [None, None, 40, None]
                ], dtype=object)
        a2 = np.array([
                [None, 3, None, None],
                [None, None, 4, None],
                [3, None, None, None]
                ], dtype=object)
        a3 = np.array([None, None, None], dtype=object)
        a4 = np.array([543, 601, 234], dtype=object)

        tb1 = TypeBlocks.from_blocks((a4, a3, a2, a1))

        self.assertEqual(
                tb1.fillna_backward(axis=1, limit=1).values.tolist(),
            	[[543, None, 3, 3, None, None, None, 10, 10, None], [601, None, None, 4, 4, 23, 23, 88, 88, None], [234, 3, 3, None, None, None, None, 40, 40, None]]
                )

        self.assertEqual(
                tb1.fillna_backward(axis=1, limit=3).values.tolist(),
                [[543, 3, 3, 3, None, 10, 10, 10, 10, None], [601, 4, 4, 4, 4, 23, 23, 88, 88, None], [234, 3, 3, None, None, 40, 40, 40, 40, None]]
                )

        self.assertEqual(
                tb1.fillna_backward(axis=1, limit=2).values.tolist(),
                [[543, 3, 3, 3, None, None, 10, 10, 10, None], [601, None, 4, 4, 4, 23, 23, 88, 88, None], [234, 3, 3, None, None, None, 40, 40, 40, None]])



    def test_type_blocks_fillna_forward_j(self) -> None:


        a2 = np.array([None, None, None])
        a3 = np.array([543, 601, 234])
        tb1 = TypeBlocks.from_blocks((a3, a2,))

        with self.assertRaises(AxisInvalid):
            tb1.fillna_forward(axis=3)

        with self.assertRaises(AxisInvalid):
            tb1.fillna_backward(axis=3)

    #---------------------------------------------------------------------------
    def test_type_blocks_from_none_a(self) -> None:

        a1 = np.array([[1, 2, 3], [4, np.nan, 6], [0, 0, 1]], dtype=object)
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        a4 = np.array([None, None, None])

        tb1 = TypeBlocks.from_zero_size_shape((3, 0))
        tb1.append(a1)
        self.assertEqual(tb1.shape, (3, 3))
        tb1.append(a4)
        self.assertEqual(tb1.shape, (3, 4))

        tb1 = TypeBlocks.from_zero_size_shape((3, 0))
        tb1.append(a4)
        self.assertEqual(tb1.shape, (3, 1))
        tb1.append(a1)
        self.assertEqual(tb1.shape, (3, 4))

    def test_type_blocks_from_none_b(self) -> None:

        with self.assertRaises(RuntimeError):
            tb1 = TypeBlocks.from_zero_size_shape((1, 5))

        with self.assertRaises(RuntimeError):
            tb1 = TypeBlocks.from_zero_size_shape((5, 1))

    def test_type_blocks_from_none_c(self) -> None:

        for shape in ((0, 3), (3, 0), (0, 0)):
            tb1 = TypeBlocks.from_zero_size_shape(shape)
            self.assertEqual(tb1.shape, shape)
            self.assertEqual(tb1.values.shape, shape)
            self.assertEqual(tb1.size, 0)
            self.assertEqual(tb1.nbytes, 0)
            self.assertEqual(len(tb1), tb1.shape[0])


    def test_type_blocks_datetime64_a(self) -> None:

        d = np.datetime64
        a1 = np.array([d('2018-01-01'), d('2018-01-02'), d('2018-01-03')])
        a2 = np.array([d('2017-01-01'), d('2017-01-02'), d('2017-01-03')])
        a3 = np.array([d('2016-01-01'), d('2016-01-02'), d('2018-01-03')])


        tb1 = TypeBlocks.from_zero_size_shape((3, 0))
        tb1.append(a1)
        tb1.append(a2)
        tb1.append(a3)

        self.assertEqual(list(tb1._reblock_signature()),
            [(np.dtype('<M8[D]'), 3)])

        tb2 = tb1.consolidate()

        self.assertEqual(list(tb2._reblock_signature()),
            [(np.dtype('<M8[D]'), 3)])

        self.assertEqual(len(tb2._blocks), 1)


    def test_type_blocks_resize_blocks_a(self) -> None:

        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        index_ic = IndexCorrespondence(has_common=True,
                is_subset=True,
                iloc_src=np.array((1, 2)),
                iloc_dst=np.array((0, 2)),
                size=2)

        tb2 = TypeBlocks.from_blocks(tb1.resize_blocks(index_ic=index_ic, columns_ic=None, fill_value=None))
        self.assertEqual(tb2.shape, (2, 3))


    def test_type_blocks_resize_blocks_b(self) -> None:

        a1 = np.arange(6).reshape(3, 2)
        # [[0, 1],
        #  [2, 3],
        #  [4, 5]]
        tb1 = TypeBlocks.from_blocks((a1))

        # reverse rows
        index_ic = IndexCorrespondence(has_common=True,
                is_subset=True,
                iloc_src=np.array((2, 1, 0)),
                iloc_dst=np.array((0, 1, 2)),
                size=3)

        # only column 2
        columns_ic = IndexCorrespondence(has_common=True,
                is_subset=True,
                iloc_src=np.array((1)),
                iloc_dst=np.array((0)),
                size=1)

        result = tb1.resize_blocks(index_ic=index_ic, columns_ic=columns_ic, fill_value=None)
        expected = [np.array([[5], [3], [1]])]
        for r,e in zip_longest(result, expected):
            self.assertTrue(np.array_equal(r, e))

    def test_type_blocks_resize_blocks_c(self) -> None:

        a1 = np.arange(6)
        tb1 = TypeBlocks.from_blocks((a1))
        # only first and last value
        index_ic = IndexCorrespondence(has_common=True,
                is_subset=True,
                iloc_src=np.array((0,5)),
                iloc_dst=np.array((0, 1)),
                size=2)

        # keep col
        columns_ic = IndexCorrespondence(has_common=True,
                is_subset=True,
                iloc_src=np.array((0)),
                iloc_dst=np.array((0)),
                size=1)

        result = tb1.resize_blocks(index_ic=index_ic, columns_ic=columns_ic, fill_value=None)
        expected = [np.array([0,5])]
        for r,e in zip_longest(result, expected):
            self.assertTrue(np.array_equal(r, e))

    def test_type_blocks_resize_blocks_d(self) -> None:

        a1 = np.arange(6)
        tb1 = TypeBlocks.from_blocks((a1))

        # no change
        columns_ic = IndexCorrespondence(has_common=True,
                is_subset=True,
                iloc_src=np.array((0)),
                iloc_dst=np.array((0)),
                size=1)

        result = tb1.resize_blocks(index_ic=None, columns_ic=columns_ic, fill_value=None)
        expected = [np.array([0,1,2,3,4,5])]
        for r,e in zip_longest(result, expected):
            self.assertTrue(np.array_equal(r, e))

    def test_type_blocks_resize_blocks_e(self) -> None:

        a1 = np.arange(6)
        a2 = np.arange(12).reshape(6,2)
        tb1 = TypeBlocks.from_blocks((a1, a2))
        # only first and last value
        index_ic = IndexCorrespondence(has_common=True,
                is_subset=True,
                iloc_src=np.array((0, 5)),
                iloc_dst=np.array((0, 1)),
                size=2)

        # keep all cols
        columns_ic = IndexCorrespondence(has_common=True,
                is_subset=True,
                iloc_src=np.array((0,1,2)),
                iloc_dst=np.array((0,1,2)),
                size=3)

        result = tb1.resize_blocks(index_ic=index_ic, columns_ic=columns_ic, fill_value=None)
        expected = [np.array([0, 5]), np.array([ 0, 10]), np.array([ 1, 11])]
        # [[0,  0,  1],
        #  [5, 10, 11]]
        for r,e in zip_longest(result, expected):
            self.assertTrue(np.array_equal(r, e))

    def test_type_blocks_astype_a(self) -> None:
        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])

        tb1 = TypeBlocks.from_blocks((a1, a2))

        tb2 = TypeBlocks.from_blocks(tb1._astype_blocks(slice(0, 2), bool))

        self.assertTypeBlocksArrayEqual(tb2,
                [[True, True, 3, False, False, True],
                [True, True, 6, True, False, True],
                [False, False, 1, True, False, True]])

        tb3 = TypeBlocks.from_blocks(tb1._astype_blocks(slice(1, 3), bool))
        self.assertTypeBlocksArrayEqual(tb3,
                [[1, True, True, False, False, True],
                [4, True, True, True, False, True],
                [0, False, True, True, False, True]]
                )

        tb4 = TypeBlocks.from_blocks(tb1._astype_blocks([0, 2, 4], bool))
        self.assertTypeBlocksArrayEqual(tb4,
                [[True, 2, True, False, False, True],
                [True, 5, True, True, False, True],
                [False, 0, True, True, False, True]]
                )

        tb5 = TypeBlocks.from_blocks(tb1._astype_blocks([4, 2, 0], bool))
        self.assertTypeBlocksArrayEqual(tb4,
                [[True, 2, True, False, False, True],
                [True, 5, True, True, False, True],
                [False, 0, True, True, False, True]]
                )

        tb6 = TypeBlocks.from_blocks(tb1._astype_blocks(4, int))
        self.assertTypeBlocksArrayEqual(tb6,
                [[1, 2, 3, False, 0, True],
                [4, 5, 6, True, 0, True],
                [0, 0, 1, True, 0, True]]
                )


    def test_type_blocks_astype_b(self) -> None:

        a1 = np.array([1, 2, 3])
        a2 = np.array([4, 5, 6])
        a3 = np.array([False, False, True])
        a4 = np.array([True, False, True])


        tb1 = TypeBlocks.from_blocks((a1, a2, a3, a4))

        tb2 = TypeBlocks.from_blocks(tb1._astype_blocks(slice(2, None), int))

        self.assertTypeBlocksArrayEqual(tb2,
                [[1, 4, 0, 1],
                [2, 5, 0, 0],
                [3, 6, 1, 1]]
                )

        tb3 = TypeBlocks.from_blocks(tb1._astype_blocks([0, 1], bool))
        self.assertTypeBlocksArrayEqual(tb3,
                [[True, True, False, True],
                [True, True, False, False],
                [True, True, True, True]]
                )


    def test_type_blocks_astype_c(self) -> None:

        a1 = np.array([1, 2, 3])
        a2 = np.array([4, 5, 6])
        a3 = np.array([False, False, True])
        a4 = np.array([True, False, True])


        tb1 = TypeBlocks.from_blocks((a1, a2, a3, a4))

        tb2 = TypeBlocks.from_blocks(tb1._astype_blocks_from_dtypes(
                {0: str, 2: str})
                )
        self.assertEqual([d.kind for d in tb2.dtypes],
                ['U', 'i', 'U', 'b'])
        self.assertEqual(tb2.shapes.tolist(),
                [(3,), (3,), (3,), (3,)])

        tb3 = TypeBlocks.from_blocks(tb1._astype_blocks_from_dtypes(
                (str, None, str, None))
                )
        self.assertEqual([d.kind for d in tb3.dtypes],
                ['U', 'i', 'U', 'b'])


    def test_type_blocks_astype_d(self) -> None:

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])

        tb1 = TypeBlocks.from_blocks((a1, a2))

        tb2 = TypeBlocks.from_blocks(tb1._astype_blocks_from_dtypes(
                {2: str, 3: str, 5: str})
                )
        self.assertEqual([d.kind for d in tb2.dtypes],
                ['i', 'i', 'U', 'U', 'b', 'U'])
        self.assertEqual(tb2.shapes.tolist(),
                [(3, 2), (3, 1), (3, 1), (3, 1), (3, 1)])

        tb3 = TypeBlocks.from_blocks(tb1._astype_blocks_from_dtypes(str))

        self.assertEqual([d.kind for d in tb3.dtypes],
                ['U', 'U', 'U', 'U', 'U', 'U'])
        self.assertEqual(tb3.shapes.tolist(),
                [(3, 3), (3, 3)])

        # import ipdb; ipdb.set_trace()


    #---------------------------------------------------------------------------

    def test_type_blocks_drop_blocks_a(self) -> None:
        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])

        tb1 = TypeBlocks.from_blocks((a1, a2))


        tb2 = TypeBlocks.from_blocks(tb1._drop_blocks(column_key=slice(0, 2)))
        self.assertEqual(tb2.shape, (3, 4))
        self.assertTrue(tb1.mloc[1] == tb2.mloc[1])

        tb4 = TypeBlocks.from_blocks(tb1._drop_blocks(column_key=[0, 2, 4]))
        self.assertTypeBlocksArrayEqual(tb4,
                [[2, False, True],
                [5, True, True],
                [0, True, True]]
                )

        self.assertTypeBlocksArrayEqual(tb2,
                [[3, False, False, True],
                [6, True, False, True],
                [1, True, False, True]])

        tb3 = TypeBlocks.from_blocks(tb1._drop_blocks(column_key=slice(1, 3)))
        self.assertTypeBlocksArrayEqual(tb3,
                [[1, False, False, True],
                [4, True, False, True],
                [0, True, False, True]]
                )


        tb5 = TypeBlocks.from_blocks(tb1._drop_blocks(column_key=[4, 2, 0]))
        self.assertTypeBlocksArrayEqual(tb4,
                [[2, False, True],
                [5, True, True],
                [0, True, True]]
                )

        tb6 = TypeBlocks.from_blocks(tb1._drop_blocks(column_key=4))
        self.assertTypeBlocksArrayEqual(tb6,
                [[1, 2, 3, False, True],
                [4, 5, 6, True, True],
                [0, 0, 1, True, True]]
                )
        self.assertTrue(tb1.mloc[0] == tb6.mloc[0])


    def test_type_blocks_drop_blocks_b(self) -> None:

        a1 = np.array([1, 2, 3])
        a2 = np.array([4, 5, 6])
        a3 = np.array([False, False, True])
        a4 = np.array([True, False, True])


        tb1 = TypeBlocks.from_blocks((a1, a2, a3, a4))

        tb2 = TypeBlocks.from_blocks(tb1._drop_blocks(column_key=slice(2, None)))

        self.assertTypeBlocksArrayEqual(tb2,
                [[1, 4],
                [2, 5],
                [3, 6]]
                )

        tb3 = TypeBlocks.from_blocks(tb1._drop_blocks(column_key=[0, 1]))

        self.assertTypeBlocksArrayEqual(tb3,
                [[False, True],
                [False, False],
                [True, True]]
                )


    def test_type_blocks_drop_blocks_c(self) -> None:
        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])

        tb1 = TypeBlocks.from_blocks((a1, a2))

        tb2 = TypeBlocks.from_blocks(tb1._drop_blocks(row_key=-1,
                column_key=slice(0, 2)))

        self.assertTypeBlocksArrayEqual(tb2,
                [[3, False, False, True],
                [6, True, False, True]])

        tb3 = TypeBlocks.from_blocks(tb1._drop_blocks(row_key=[0,2],
                column_key=slice(1, 3)))

        self.assertTypeBlocksArrayEqual(tb3,
                [[4, True, False, True]]
                )


    def test_type_blocks_drop_blocks_d(self) -> None:

        a1 = np.array([1, 2, 3])
        a2 = np.array([4, 5, 6])
        a3 = np.array([False, False, True])
        a4 = np.array([True, False, True])


        tb1 = TypeBlocks.from_blocks((a1, a2, a3, a4))

        tb2 = TypeBlocks.from_blocks(tb1._drop_blocks(row_key=1, column_key=[0, 3]))

        self.assertTypeBlocksArrayEqual(tb2,
                [[4, False], [6, True]]
                )

        tb3 = TypeBlocks.from_blocks(tb1._drop_blocks(row_key=1))

        self.assertTypeBlocksArrayEqual(tb3,
                [[1, 4, False, True],
                [3, 6, True, True]]
                )


    def test_type_blocks_drop_blocks_e(self) -> None:


        for arrays in self.get_arrays_a():
            tb1 = TypeBlocks.from_blocks(arrays)

            for i in range(tb1.shape[1]):
                tb2 = TypeBlocks.from_blocks(tb1._drop_blocks(column_key=i))
                self.assertTrue(tb2.shape == (3, tb1.shape[1] - 1))



    def test_type_blocks_drop_blocks_f(self) -> None:
        a1 = np.array([[1], [5], [0]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])

        tb1 = TypeBlocks.from_blocks((a1, a2))
        self.assertEqual(tb1.shape, (3, 4))

        tb2 = TypeBlocks.from_blocks(tb1._drop_blocks(column_key=slice(0, 2)))
        self.assertEqual(tb2.shape, (3, 2))
        self.assertTrue((tb1[2:].values == tb2.values).all())


    def test_type_blocks_drop_blocks_g(self) -> None:
        a1 = np.array([[1], [5], [0]])
        a2 = np.array([[2], [6], [10]])
        a3 = np.array([[3], [7], [11]])
        a4 = np.array([[4], [8], [2]])

        tb1 = TypeBlocks.from_blocks((a1, a2, a3, a4))
        self.assertEqual(tb1.shape, (3, 4))

        tb2 = TypeBlocks.from_blocks(tb1._drop_blocks(column_key=slice(0, 2)))
        self.assertEqual(tb2.shape, (3, 2))
        self.assertTrue((tb1[2:].values == tb2.values).all())



    def test_type_blocks_drop_blocks_h(self) -> None:
        a1 = np.array([[False]])
        tb1 = TypeBlocks.from_blocks(a1)
        self.assertEqual(tb1.shape, (1, 1))
        self.assertEqual(tb1.drop(0).shape, (0, 1))
        self.assertEqual(tb1.drop((None, 0)).shape, (1, 0))

        # after no rows remain, trying to drop more rows
        tb2 = tb1.drop(0)
        with self.assertRaises(IndexError):
            tb2.drop(0) # raise from NumPy

        tb3 = tb1.drop((None, 0))

        # after no columns remain, tyring to drop more should raise an exception
        with self.assertRaises(IndexError):
            tb3.drop((None, 0))


    def test_type_blocks_pickle_a(self) -> None:

        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        pbytes = pickle.dumps(tb1)
        tb2 = pickle.loads(pbytes)

        self.assertEqual([b.flags.writeable for b in tb2._blocks],
                [False, False, False]
                )


    #---------------------------------------------------------------------------
    def test_type_blocks_roll_blocks_a(self) -> None:

        a1 = np.array([[1, 2, 3], [4, -1, 6], [0, 0, 1]], dtype=object)
        a2 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        a3 = np.array([None, None, None])

        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertTypeBlocksArrayEqual(
                TypeBlocks.from_blocks(tb1._shift_blocks(1, 1, wrap=True)),
                [[None, 0, 0, 1, 'oe', 'od'],
                [None, 1, 2, 3, 'a', 'b'],
                [None, 4, -1, 6, 'c', 'd']]
                )

        self.assertTypeBlocksArrayEqual(
                TypeBlocks.from_blocks(tb1._shift_blocks(-1, -1, wrap=True)),
                [[-1, 6, 'c', 'd', None, 4],
                [0, 1, 'oe', 'od', None, 0],
                [2, 3, 'a', 'b', None, 1]]
                )

        self.assertTypeBlocksArrayEqual(
                TypeBlocks.from_blocks(tb1._shift_blocks(-2, 2, wrap=True)),
                [['od', None, 0, 0, 1, 'oe'],
                ['b', None, 1, 2, 3, 'a'],
                ['d', None, 4, -1, 6, 'c']]
                )


    def test_type_blocks_roll_blocks_b(self) -> None:

        a1 = np.array([[1, 2, 3], [4, -1, 6], [0, 0, 1]], dtype=object)
        a2 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        a3 = np.array([None, None, None])

        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        # import ipdb; ipdb.set_trace()
        self.assertTypeBlocksArrayEqual(
                TypeBlocks.from_blocks(tb1._shift_blocks(1, 1, wrap=False,fill_value='x')),
                [['x', 'x', 'x', 'x', 'x', 'x'],
                ['x', 1, 2, 3, 'a', 'b'],
                ['x', 4, -1, 6, 'c', 'd']],
                match_dtype=object
                )

        self.assertTypeBlocksArrayEqual(
                TypeBlocks.from_blocks(tb1._shift_blocks(2,
                        -2,
                        wrap=False,
                        fill_value=10)),
                [[10, 10, 10, 10, 10, 10],
                [10, 10, 10, 10, 10, 10],
                [3, 'a', 'b', None, 10, 10]],
                match_dtype=object
                )

    def test_type_blocks_roll_blocks_c(self) -> None:

        a1 = np.array([1, 2, 3])
        a2 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb1 = TypeBlocks.from_blocks((a1, a2))

        self.assertEqual(
                TypeBlocks.from_blocks(tb1._shift_blocks(0, 0, True)).values.tolist(),
                [[1, 'a', 'b'], [2, 'c', 'd'], [3, 'oe', 'od']]
                )

    def test_type_blocks_roll_blocks_d(self) -> None:

        a1 = np.array([1, 2, 3])
        a2 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb1 = TypeBlocks.from_blocks((a1, a2))

        self.assertEqual(
                TypeBlocks.from_blocks(tb1._shift_blocks(0, 0, False)).values.tolist(),
                [[1, 'a', 'b'], [2, 'c', 'd'], [3, 'oe', 'od']]
                )


    #---------------------------------------------------------------------------
    def test_type_blocks_from_blocks_a(self) -> None:

        a1 = np.full((3, 0), False)
        a2 = np.full((3, 4), 'x')
        tb = TypeBlocks.from_blocks((a1, a2))
        self.assertEqual(tb.shape, (3, 4))
        self.assertEqual(tb.values.tolist(),
            [['x', 'x', 'x', 'x'],
            ['x', 'x', 'x', 'x'],
            ['x', 'x', 'x', 'x']])

        a3 = next(tb.axis_values(0))
        self.assertEqual(a3.tolist(),
            ['x', 'x', 'x']
            )

    def test_type_blocks_from_blocks_b(self) -> None:

        a1 = np.full((3, 0), False)
        a2 = np.full((3, 2), 'x')
        a3 = np.full((3, 0), False)
        a4 = np.full((3, 2), 'y')
        a5 = np.full((3, 0), False)

        tb = TypeBlocks.from_blocks((a1, a2, a3, a4, a5))
        self.assertEqual(tb.shape, (3, 4))
        self.assertEqual(tb.values.tolist(),
            [['x', 'x', 'y', 'y'],
            ['x', 'x', 'y', 'y'],
            ['x', 'x', 'y', 'y']])

        a3 = next(tb.axis_values(0, reverse=True))
        self.assertEqual(a3.tolist(),
            ['y', 'y', 'y']
            )


    def test_type_blocks_from_blocks_c(self) -> None:

        a1 = np.full((0, 2), False)
        a2 = np.full((0, 1), 'x')

        tb = TypeBlocks.from_blocks((a1, a2))
        self.assertEqual(tb.shape, (0, 3))
        self.assertEqual(len(tb), 0)

        self.assertEqual(tb.dtypes.tolist(),
            [np.dtype('bool'), np.dtype('bool'), np.dtype('<U1')])

        tb.append(np.empty((0, 2)))
        self.assertEqual(tb.shape, (0, 5))
        self.assertEqual(len(tb), 0)

        with self.assertRaises(RuntimeError):
            tb.append(np.empty((3, 0)))

        # import ipdb; ipdb.set_trace()


    def test_type_blocks_from_blocks_d(self) -> None:

        a1 = np.full(0, False)
        a2 = np.full(0, 'x')

        # if the row lentgh is defined as 0, adding 1D arrays, even if empty, count as adding a column, as a 1D array is by definition shape (0, 1)
        tb = TypeBlocks.from_blocks((a1, a2))
        self.assertEqual(tb.shape, (0, 2))
        self.assertEqual(len(tb), 0)
        self.assertEqual(len(tb.shapes), 2)

        tb.append(np.empty((0, 2)))
        self.assertEqual(tb.shape, (0, 4))
        self.assertEqual(len(tb.shapes), 3)

        tb.append(np.empty((0, 0)))
        self.assertEqual(tb.shape, (0, 4))
        self.assertEqual(len(tb.shapes), 3)

        tb.append(np.empty(0))
        self.assertEqual(tb.shape, (0, 5))
        self.assertEqual(len(tb.shapes), 4)


    def test_type_blocks_append_a(self) -> None:
        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        tb1 = TypeBlocks.from_blocks((a1, a2))
        self.assertTrue(tb1.shape, (3, 2))

        tb1.append(np.array((3,5,4)))
        self.assertTrue(tb1.shape, (3, 3))

        tb1.append(np.array([(3,5),(4,6),(5,10)]))
        self.assertTrue(tb1.shape, (3, 5))

        self.assertEqual(tb1.iloc[0].values.tolist(), [[1, False, 3, 3, 5]])
        self.assertEqual(tb1.iloc[1].values.tolist(), [[2, True, 5, 4, 6]])
        self.assertEqual(tb1.iloc[:, 3].values.tolist(), [[3], [4], [5]])


    def test_type_blocks_append_b(self) -> None:

        a1 = np.full((2, 3), False)

        tb = TypeBlocks.from_blocks(a1)
        tb.append(np.empty((2, 0)))

        self.assertEqual(tb.shape, (2, 3))
        # array was not added
        self.assertEqual(len(tb.shapes), 1)


    def test_type_blocks_append_c(self) -> None:

        a1 = np.full((0, 3), False)

        tb = TypeBlocks.from_blocks(a1)
        tb.append(np.empty((0, 0)))

        self.assertEqual(tb.shape, (0, 3))
        # array was not added
        self.assertEqual(len(tb.shapes), 1)



    def test_type_blocks_append_d(self) -> None:

        tb = TypeBlocks.from_zero_size_shape((0, 0))
        tb.append(np.empty((0, 0)))

        self.assertEqual(tb.shape, (0, 0))
        # array was not added
        self.assertEqual(len(tb.shapes), 0)


    def test_type_blocks_append_e(self) -> None:

        # given a zero row TB, appending a zero length array could mean changing the shape, as the row aligns
        tb = TypeBlocks.from_zero_size_shape((0, 0))
        tb.append(np.empty(0))

        self.assertEqual(tb.shape, (0, 1))
        # array was not added
        self.assertEqual(len(tb.shapes), 1)




    def test_type_blocks_extract_bloc_assign_a(self) -> None:

        a1 = np.array([[1, 2, 3], [4, -5, 6], [0, 0, 1]])
        a2 = np.array([1.5, 5.2, 5.5])
        a3 = np.array([[False, False, True], [True, False, True], [True, False, True]])

        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        coords = ((1, 1), (2, 4), (0, 3))

        targets = np.full(tb1.shape, False)
        for coord in coords:
            targets[coord] = True

        tb2 = tb1.extract_bloc_assign_by_unit(targets, None)
        self.assertEqual(tb2.values.tolist(),
                [[1, 2, 3, None, False, False, True], [4, None, 6, 5.2, True, False, True], [0, 0, 1, 5.5, None, False, True]]
                )



    def test_type_blocks_extract_bloc_assign_b(self) -> None:

        a1 = np.array([[1, 2, 3], [4, -5, 6], [0, 0, 1]])
        a2 = np.array([1.5, 5.2, 5.5])
        a3 = np.array([[False, False, True], [True, False, True], [True, False, True]])

        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        coords = ((1, 1), (0, 4), (2, 5), (0, 3), (1, 3), (1, 6 ))

        targets = np.full(tb1.shape, False)
        for coord in coords:
            targets[coord] = True

        values = np.arange(np.prod(tb1.shape)).reshape(tb1.shape) * -100
        tb2 = tb1.extract_bloc_assign_by_unit(targets, values)

        self.assertEqual(tb2.values.tolist(),
            [[1, 2, 3, -300.0, -400, False, True], [4, -800, 6, -1000.0, True, False, -1300], [0, 0, 1, 5.5, True, -1900, True]]
            )

    #---------------------------------------------------------------------------
    def test_type_blocks_round_a(self) -> None:

        a1 = np.full(4, .33333, )
        a2 = np.full((4, 2), .88888, )
        a3 = np.full(4, .55555)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        tb2 = round(tb1, 3) #type: ignore
        self.assertEqual(
                tb2.values.tolist(),
                [[0.333, 0.889, 0.889, 0.556], [0.333, 0.889, 0.889, 0.556], [0.333, 0.889, 0.889, 0.556], [0.333, 0.889, 0.889, 0.556]]
                )
        tb3 = round(tb1, 1) #type: ignore
        self.assertEqual(
                tb3.values.tolist(),
                [[0.3, 0.9, 0.9, 0.6], [0.3, 0.9, 0.9, 0.6], [0.3, 0.9, 0.9, 0.6], [0.3, 0.9, 0.9, 0.6]]
                )


    #---------------------------------------------------------------------------

    def test_type_blocks_ufunc_blocks_a(self) -> None:

        a1 = np.arange(8).reshape(2, 4)
        tb1 = TypeBlocks.from_blocks(a1)

        ufunc = lambda x: x * 2
        tb2 = TypeBlocks.from_blocks(tb1._ufunc_blocks(NULL_SLICE, ufunc))

        self.assertEqual(tb2.values.tolist(),
                [[0, 2, 4, 6], [8, 10, 12, 14]])

        tb3 = TypeBlocks.from_blocks(tb1._ufunc_blocks(1, ufunc))
        self.assertEqual(tb3.values.tolist(),
                [[0, 2, 2, 3], [4, 10, 6, 7]])


    def test_type_blocks_ufunc_blocks_b(self) -> None:

        a1 = np.arange(3)
        a2 = np.arange(3)
        a3 = np.arange(3)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        ufunc = lambda x: x * 2
        tb2 = TypeBlocks.from_blocks(tb1._ufunc_blocks(NULL_SLICE, ufunc))

        self.assertEqual(tb2.values.tolist(),
                [[0, 0, 0], [2, 2, 2], [4, 4, 4]])

        tb3 = TypeBlocks.from_blocks(tb1._ufunc_blocks(1, ufunc))
        self.assertEqual(tb3.values.tolist(),
                [[0, 0, 0], [1, 2, 1], [2, 4, 2]])


    def test_type_blocks_ufunc_blocks_c(self) -> None:

        a1 = np.arange(3)
        a2 = np.arange(6).reshape(3, 2)
        a3 = np.arange(3)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        ufunc = lambda x: x * 2
        tb2 = TypeBlocks.from_blocks(tb1._ufunc_blocks(NULL_SLICE, ufunc))
        self.assertEqual(tb2.values.tolist(),
                [[0, 0, 2, 0], [2, 4, 6, 2], [4, 8, 10, 4]])

        tb3 = TypeBlocks.from_blocks(tb1._ufunc_blocks(slice(2,4), ufunc))
        self.assertEqual(tb3.values.tolist(),
                [[0, 0, 2, 0], [1, 2, 6, 2], [2, 4, 10, 4]])


    #---------------------------------------------------------------------------
    def test_type_blocks_getitem_a(self) -> None:

        a1 = np.arange(3)
        a2 = np.arange(10, 16).reshape(3, 2)
        a3 = np.arange(20, 23)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertEqual(tb1[2].values.tolist(),
                [[11], [13], [15]])

        with self.assertRaises(KeyError):
            _ = tb1[2, 2]

        self.assertEqual(
                tb1[2:].values.tolist(),
                [[11, 20], [13, 21], [15, 22]]
                )

    #---------------------------------------------------------------------------
    def test_type_blocks_equals_a(self) -> None:

        a1 = np.array([[1, 10], [2, 20], [3, 30]])
        a2 = np.array([False, True, False])
        tb1 = TypeBlocks.from_blocks((a1, a2))

        a3 = np.array([1, 2, 3])
        a4 = np.array([10, 20, 30])
        a5 = np.array([False, True, False])
        tb2 = TypeBlocks.from_blocks((a3, a4, a5))

        self.assertTrue(tb1.equals(tb2))
        self.assertTrue(tb2.equals(tb1))
        self.assertTrue(tb1.equals(tb1))

        a6 = np.array([1, 2, 3])
        a7 = np.array([10, 21, 30])
        a8 = np.array([False, True, False])
        tb3 = TypeBlocks.from_blocks((a6, a7, a8))

        self.assertFalse(tb1.equals(tb3))
        self.assertFalse(tb3.equals(tb1))


    def test_type_blocks_equals_b(self) -> None:

        a1 = np.array([[1, 10], [2, 20], [3, 30]])
        a2 = np.array([False, True, False])
        tb1 = TypeBlocks.from_blocks((a1, a2))

        self.assertFalse(tb1.equals('a'))
        self.assertFalse(tb1.equals(1))
        self.assertFalse(tb1.equals([1, 10]))
        self.assertFalse(tb1.equals(np.arange(3)))
        self.assertFalse(tb1.equals(None))
        self.assertFalse(tb1.equals(dict(a=30, b=40)))


    def test_type_blocks_equals_c(self) -> None:

        a1 = np.array([False, True, False])
        a2 = np.array([10, 20, 30], dtype=np.int64)
        tb1 = TypeBlocks.from_blocks((a1, a2))

        a3 = np.array([False, True, False])
        a4 = np.array([10, 20, 30], dtype=np.int32)
        tb2 = TypeBlocks.from_blocks((a3, a4))

        self.assertFalse(tb1.equals(tb2, compare_dtype=True))
        self.assertTrue(tb1.equals(tb2, compare_dtype=False))



    def test_type_blocks_equals_d(self) -> None:

        a1 = np.array([False, True, False])
        a2 = np.array([10, 20, 30], dtype=np.int64)
        tb1 = TypeBlocks.from_blocks((a1, a2))
        tb2 = TypeBlocks.from_blocks((a1,))

        self.assertFalse(tb1.equals(tb1.values, compare_class=True))
        # difference by shape
        self.assertFalse(tb1.equals(tb2))


    def test_type_blocks_equals_e(self) -> None:

        a1 = np.array([False, True, False])
        a2 = np.array(['2020-01-01', '2020-01-01', '2021-01-01'], dtype=np.datetime64)
        a3 = np.array(['2020', '2020', '2021'], dtype=np.datetime64)

        tb1 = TypeBlocks.from_blocks((a1, a2))
        tb2 = TypeBlocks.from_blocks((a1, a3))

        # import ipdb; ipdb.set_trace()
        self.assertFalse(tb1.equals(tb2))




    #---------------------------------------------------------------------------
    def test_type_blocks_ufunc_binary_operator_a(self) -> None:
        a1 = np.array([10, 20, 30])
        a2 = np.arange(10, 16).reshape(3, 2)
        a3 = np.array([False, True, False])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        tb2 = tb1._ufunc_binary_operator(
                operator=lambda x, y: x * y,
                other = np.array([1, 0, 1]),
                axis=1,
                )
        self.assertEqual(tb2.shapes.tolist(), [(3,), (3,), (3,), (3,)])
        self.assertEqual(tb2.values.tolist(),
                [[10, 10, 11, 0], [0, 0, 0, 0], [30, 14, 15, 0]]
                )

        with self.assertRaises(NotImplementedError):
            tb2 = tb1._ufunc_binary_operator(
                    operator=lambda x, y: x * y,
                    other = np.array([1, 0, 1]),
                    axis=0,
                    )

    #---------------------------------------------------------------------------
    def test_type_blocks_deepcopy_a(self) -> None:
        a1 = np.array([10, 20, 30])
        a2 = np.arange(10, 16).reshape(3, 2)
        a3 = np.array([False, True, False])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        tb2 = copy.deepcopy(tb1)
        self.assertEqual(tb1.shape, tb2.shape)
        self.assertNotEqual([id(a) for a in tb1._blocks], [id(a) for a in tb2._blocks])

    def test_type_blocks_deepcopy_b(self) -> None:
        a1 = np.array([10, 20, 30])
        a1.flags.writeable = False

        tb1 = TypeBlocks.from_blocks((a1, a1, a1))

        tb2 = copy.deepcopy(tb1)

        self.assertTrue(id(tb2._blocks[0]) != id(tb1._blocks[0]))

        # usage of memo dict means we reuse the same array once we have one copy
        self.assertTrue(id(tb2._blocks[0]) == id(tb2._blocks[1]))
        self.assertTrue(id(tb2._blocks[0]) == id(tb2._blocks[2]))

    def test_type_blocks_deepcopy_c(self) -> None:
        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        tb2 = copy.deepcopy(tb1)

        self.assertTrue(tb1.shape == tb2.shape)
        self.assertTrue(id(tb1._blocks[2]) != id(tb2._blocks[2]))




if __name__ == '__main__':
    unittest.main()
