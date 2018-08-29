import unittest


import numpy as np

import static_frame as sf
# assuming located in the same directory
from static_frame import Index
from static_frame import IndexGO
from static_frame import Series
from static_frame import Frame
from static_frame import FrameGO
from static_frame import TypeBlocks
from static_frame import Display
from static_frame import mloc
from static_frame import DisplayConfig

from static_frame.core.util import immutable_filter

from static_frame.test.test_case import TestCase


nan = np.nan


class TestUnit(TestCase):

    def test_type_blocks_a(self):

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

    def test_type_blocks_contiguous_pairs(self):

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



    def test_type_blocks_b(self):

        # given naturally of a list of rows; this corresponds to what we get with iloc, where we select a row first, then a column
        a1 = np.array([[1, 2, 3], [4, 5, 6]])
        # shape is given as rows, columns
        self.assertEqual(a1.shape, (2, 3))

        a2 = np.array([[.2, .5, .4], [.8, .6, .5]])
        a3 = np.array([['a', 'b'], ['c', 'd']])


        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertTypeBlocksArrayEqual(tb1[2], [3, 6])
        self.assertTypeBlocksArrayEqual(tb1[4], [0.5, 0.6])

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

        self.assertEqual(slice3.iloc[0].values.tolist(), [3, 'a', 1])
        self.assertEqual(slice3.iloc[1].values.tolist(), [6, 'c', 4])

        ## slice refers to the same data; not sure if this is accurate test yet

        row1 = tb1.iloc[0].values
        self.assertEqual(row1.dtype, object)
        self.assertEqual(len(row1), 8)
        self.assertEqual(list(row1[:3]), [1, 2, 3])
        self.assertEqual(list(row1[-2:]), ['a', 'b'])

        self.assertEqual(tb1.unified, False)



    def test_type_blocks_c(self):

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

        self.assertEqual(tb1.iloc[1].values.tolist(), [2, True, 'c', 'cd'])
        #tb1.iloc[0:2]

        #tb1.iloc[0:2, 0:2]

        #tb1.iloc[0,2]

        #tb1.iloc[0, 0:2]

        self.assertEqual(tb1.iloc[0, 0:2].shape, (1, 2))
        self.assertEqual(tb1.iloc[0:2, 0:2].shape, (2, 2))



    def test_type_blocks_d(self):
        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])


        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertEqual(tb1.iloc[0:2].shape, (2, 8))
        self.assertEqual(tb1.iloc[1:3].shape, (2, 8))

        #tb1.iloc[0, 1:5]


    def test_type_blocks_indices_to_contiguous_pairs(self):

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])


        tb1 = TypeBlocks.from_blocks((a1, a2, a3))
        self.assertEqual(list(tb1._key_to_block_slices(0)), [(0, 0)])
        self.assertEqual(list(tb1._key_to_block_slices(6)), [(2, 0)])
        self.assertEqual(list(tb1._key_to_block_slices([3,5,6])),
            [(1, slice(0, 1, None)), (1, slice(2, 3, None)), (2, slice(0, 1, None))]
            )

        # for rows, all areg grouped by 0
        #self.assertEqual(list(tb1._key_to_block_slices(1, axis=0)), [(0, 1)])
        #self.assertEqual(list(tb1._key_to_block_slices((0,2), axis=0)),
            #[(0, slice(0, 1, None)), (0, slice(2, 3, None))]
            #)
        #self.assertEqual(list(tb1._key_to_block_slices((0,1), axis=0)),
            #[(0, slice(0, 2, None))]
            #)





    def test_type_blocks_extract_a(self):

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


    def test_type_blocks_extract_b(self):
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

    def test_type_blocks_extract_c(self):
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
                ['a', 'c', 'oe'],
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


    def test_type_blocks_extract_array_a(self):
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


    def test_immutable_filter(self):
        a1 = np.array([3, 4, 5])
        a2 = immutable_filter(a1)
        with self.assertRaises(ValueError):
            a2[0] = 34
        a3 = a2[:2]
        with self.assertRaises(ValueError):
            a3[0] = 34


    def test_type_blocks_static_frame(self):
        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertTrue(tb1.dtypes[0] == np.int64)


    def test_type_blocks_attributes(self):

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




    def test_type_blocks_block_pointers(self):

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


    def test_type_blocks_append(self):
        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        tb1 = TypeBlocks.from_blocks((a1, a2))
        self.assertTrue(tb1.shape, (3, 2))

        tb1.append(np.array((3,5,4)))
        self.assertTrue(tb1.shape, (3, 3))

        tb1.append(np.array([(3,5),(4,6),(5,10)]))
        self.assertTrue(tb1.shape, (3, 5))

        self.assertEqual(tb1.iloc[0].values.tolist(), [1, False, 3, 3, 5])
        self.assertEqual(tb1.iloc[1].values.tolist(), [2, True, 5, 4, 6])
        self.assertEqual(tb1.iloc[:, 3].values.tolist(), [3, 4, 5])




    def test_type_blocks_unary_operator_a(self):

        a1 = np.array([1,-2,-3])
        a2 = np.array([False, True, False])
        tb1 = TypeBlocks.from_blocks((a1, a2))

        tb2 = ~tb1 # tilde
        self.assertEqual(
            (~tb1.values).tolist(),
            [[-2, -1], [1, -2], [2, -1]])

    def test_type_blocks_unary_operator_b(self):

        a1 = np.array([[1, 2, 3], [-4, 5, 6], [0,0,-1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertTypeBlocksArrayEqual(
                -tb2[0:3],
                [[-1, -2, -3],
                 [ 4, -5, -6],
                 [ 0,  0,  1]],
                )

        self.assertTypeBlocksArrayEqual(
                ~tb2[3:5],
                [[ True,  True],
                [False,  True],
                [False,  True]],
                )



    def test_type_blocks_block_compatible_a(self):

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


    #@unittest.skip('to fix')
    def test_type_blocks_block_compatible_b(self):

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



    def test_type_blocks_consolidate_a(self):

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


    def test_type_blocks_consolidate_b(self):
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




    def test_type_blocks_binary_operator_a(self):

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



    def test_type_blocks_binary_operator_b(self):

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


    def test_type_blocks_binary_operator_c(self):
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


    def test_type_blocks_binary_operator_d(self):
        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[1.5,2.6], [4.2,5.5], [0.2,0.1]])
        tb1 = TypeBlocks.from_blocks((a1, a2))

        post = tb1 * (1, 0, 2, 0, 1)
        self.assertTypeBlocksArrayEqual(post,
                [[  1. ,   0. ,   6. ,   0. ,   2.6],
                [  4. ,   0. ,  12. ,   0. ,   5.5],
                [  0. ,   0. ,   2. ,   0. ,   0.1]])



    def test_type_blocks_extend_a(self):
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



    def test_type_blocks_mask_blocks_a(self):
        # test negative slices

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        mask = TypeBlocks.from_blocks(tb1._mask_blocks(column_key=[2,3,5,6]))

        self.assertTypeBlocksArrayEqual(mask,
            [[False, False, True, True, False, True, True, False], [False, False, True, True, False, True, True, False], [False, False, True, True, False, True, True, False]]
            )



    def test_type_blocks_assign_blocks_a(self):
        # test negative slices

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        tb2 = TypeBlocks.from_blocks(tb1._assign_blocks_from_keys(column_key=[2,3,5], value=300))

        self.assertTypeBlocksArrayEqual(tb2,
            [[1, 2, 300, 300, False, 300, 'a', 'b'],
            [4, 5, 300, 300, False, 300, 'c', 'd'],
            [0, 0, 300, 300, False, 300, 'oe', 'od']], match_dtype=object)

        # blocks not mutated will be the same
        self.assertEqual(tb1.mloc[2], tb2.mloc[2])

    def test_type_blocks_group_a(self):

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


    def test_type_blocks_group_b(self):

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
        self.assertEqual(group.tolist(), [False, False])
        self.assertEqual(subtb.values.tolist(),
                [[1, 2, 3, 4, False, False, True], [4, 2, 6, 3, False, False, True]]
                )

        group, selection, subtb = groups[1]
        self.assertEqual(group.tolist(), [True, False])
        self.assertEqual(subtb.values.tolist(),
                [[0, 0, 1, 2, True, False, True], [0, 0, 1, 1, True, False, True]])
        # TODO: add more tests here


    def test_type_blocks_transpose_a(self):

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        tb2 = tb1.transpose()

        self.assertEqual(tb2.values.tolist(),
                [[1, 4, 0], [2, 5, 0], [3, 6, 1], [False, True, True], [False, False, False], [True, True, True], ['a', 'c', 'oe'], ['b', 'd', 'od']])

        self.assertEqual(tb1.transpose().transpose().values.tolist(),
                tb1.values.tolist())



    def test_type_blocks_display_a(self):

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb = TypeBlocks.from_blocks((a1, a2, a3))

        disp = tb.display()
        self.assertEqual(len(disp), 5)
        # self.assertEqual(list(disp),
        #     [['<TypeBlocks> ', '                  ', '          '],
        #     ['1 2 3        ', 'False False  True ', "'a' 'b'   "],
        #     ['4 5 6        ', ' True False  True ', "'c' 'd'   "],
        #     ['0 0 1        ', ' True False  True ', "'oe' 'od' "],
        #     ['int64        ', 'bool              ', '<U2       ']])


    def test_type_blocks_axis_values_a(self):
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


    def test_type_blocks_axis_values_b(self):
        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertEqual(
                [a.tolist() for a in tb.axis_values(axis=0, reverse=True)],
                [['b', 'd', 'od'], ['a', 'c', 'oe'], [True, True, True], [False, False, False], [False, True, True], [3, 6, 1], [2, 5, 0], [1, 4, 0]])
        self.assertEqual([a.tolist() for a in tb.axis_values(axis=0, reverse=False)],
                [[1, 4, 0], [2, 5, 0], [3, 6, 1], [False, True, True], [False, False, False], [True, True, True], ['a', 'c', 'oe'], ['b', 'd', 'od']])


    def test_type_blocks_extract_iloc_mask_a(self):

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


    def test_type_blocks_extract_iloc_assign_a(self):

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb = TypeBlocks.from_blocks((a1, a2, a3))


        self.assertEqual(tb.extract_iloc_assign(1, 600).values.tolist(),
                [[1, 2, 3, False, False, True, 'a', 'b'], [600, 600, 600, 600, 600, 600, 600, 600], [0, 0, 1, True, False, True, 'oe', 'od']])

        self.assertEqual(tb.extract_iloc_assign((1, 5), 20).values.tolist(),
                [[1, 2, 3, False, False, True, 'a', 'b'], [4, 5, 6, True, False, 20, 'c', 'd'], [0, 0, 1, True, False, True, 'oe', 'od']])

        self.assertEqual(tb.extract_iloc_assign((slice(2), slice(5)), 'X').values.tolist(),
                [['X', 'X', 'X', 'X', 'X', True, 'a', 'b'], ['X', 'X', 'X', 'X', 'X', True, 'c', 'd'], [0, 0, 1, True, False, True, 'oe', 'od']]
                )

        self.assertEqual(tb.extract_iloc_assign(([0,1], [1,4,7]), -5).values.tolist(),
                [[1, -5, 3, False, -5, True, 'a', -5], [4, -5, 6, True, -5, True, 'c', -5], [0, 0, 1, True, False, True, 'oe', 'od']])


        self.assertEqual(
                tb.extract_iloc_assign((1, slice(4)), (-1, -2, -3, -4)).values.tolist(),
                [[1, 2, 3, False, False, True, 'a', 'b'], [-1, -2, -3, -4, False, True, 'c', 'd'], [0, 0, 1, True, False, True, 'oe', 'od']])

        self.assertEqual(
                tb.extract_iloc_assign((2, slice(3,7)), (-1, -2, -3, -4)).values.tolist(),
                [[1, 2, 3, False, False, True, 'a', 'b'], [4, 5, 6, True, False, True, 'c', 'd'], [0, 0, 1, -1, -2, -3, -4, 'od']])
        self.assertEqual(
                tb.extract_iloc_assign((0, slice(4,8)), (-1, -2, -3, -4)).values.tolist(),
                [[1, 2, 3, False, -1, -2, -3, -4], [4, 5, 6, True, False, True, 'c', 'd'], [0, 0, 1, True, False, True, 'oe', 'od']])


    def test_type_blocks_elements_items_a(self):

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


    def test_type_blocks_reblock_signature_a(self):

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        a4 = np.array([None, None, None])
        tb = TypeBlocks.from_blocks((a1, a2, a4, a3))

        dtype = np.dtype
        self.assertEqual(
                list(tb._reblock_signature()),
                [(dtype('int64'), 3), (dtype('bool'), 3), (dtype('O'), 1), (dtype('<U2'), 2)])

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        a4 = np.array([None, None, None])
        tb = TypeBlocks.from_blocks((a1, a2, a3, a4))

        self.assertEqual(
                list(tb._reblock_signature()),
                [(dtype('int64'), 3), (dtype('bool'), 3), (dtype('<U2'), 2), (dtype('O'), 1)])



#     @unittest.skip('implement operators for same sized but differently typed blocks')
    def test_type_blocks_binary_operator_e(self):

        a1 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        a4 = np.array([None, None, None])
        tb = TypeBlocks.from_blocks((a1, a2, a4, a3))

        post = [x for x in tb.element_items()]

        tb2 = TypeBlocks.from_element_items(post, tb.shape, tb._row_dtype)
        self.assertTrue((tb.values == tb2.values).all())

        post = tb == tb2
        self.assertEqual(post.values.tolist(),
                [[True, True, True, True, True, True, True, True, True], [True, True, True, True, True, True, True, True, True], [True, True, True, True, True, True, True, True, True]])


    def test_type_blocks_copy_a(self):

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
                [0, 0, 1, True, False, True, None, 'oe', 'od', 3])

        self.assertEqual(tb2.iloc[2].values.tolist(),
                [0, 0, 1, True, False, True, None, 'oe', 'od'])




    def test_type_blocks_isna_a(self):

        a1 = np.array([[1, 2, 3], [4, np.nan, 6], [0, 0, 1]], dtype=object)
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        a4 = np.array([None, None, None])
        tb1 = TypeBlocks.from_blocks((a1, a2, a4, a3))

        self.assertEqual(tb1.isna().values.tolist(),
                [[False, False, False, False, False, False, True, False, False], [False, True, False, False, False, False, True, False, False], [False, False, False, False, False, False, True, False, False]])

        self.assertEqual(tb1.notna().values.tolist(),
                [[True, True, True, True, True, True, False, True, True], [True, False, True, True, True, True, False, True, True], [True, True, True, True, True, True, False, True, True]])


    def test_type_blocks_dropna_to_slices(self):

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

        row_key, column_key = tb1.dropna_to_keep_locations(axis=1)

        self.assertEqual(column_key.tolist(),
                [True, False, True, True, True, False, True, True])
        self.assertEqual(row_key, None)

        row_key, column_key = tb1.dropna_to_keep_locations(axis=0)
        self.assertEqual(row_key.tolist(),
                [True, True, False])

        self.assertEqual(column_key, None)


    def test_type_blocks_fillna_a(self):

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
        tb2 = tb1.fillna(0)
        self.assertEqual([b.dtype for b in tb2._blocks],
                [np.dtype('float64'), np.dtype('O')])
        self.assertEqual(tb2.isna().values.any(), False)

        tb3 = tb1.fillna(None)
        self.assertEqual([b.dtype for b in tb3._blocks],
                [np.dtype('O'), np.dtype('O')])
        # we ahve Nones, which are na
        self.assertEqual(tb3.isna().values.any(), True)


    def test_type_blocks_from_none_a(self):

        a1 = np.array([[1, 2, 3], [4, np.nan, 6], [0, 0, 1]], dtype=object)
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        a4 = np.array([None, None, None])

        tb1 = TypeBlocks.from_none()
        tb1.append(a1)
        self.assertEqual(tb1.shape, (3, 3))
        tb1.append(a4)
        self.assertEqual(tb1.shape, (3, 4))

        tb1 = TypeBlocks.from_none()
        tb1.append(a4)
        self.assertEqual(tb1.shape, (3, 1))
        tb1.append(a1)
        self.assertEqual(tb1.shape, (3, 4))


    def test_type_blocks_datetime64_a(self):

        d = np.datetime64
        a1 = np.array([d('2018-01-01'), d('2018-01-02'), d('2018-01-03')])
        a2 = np.array([d('2017-01-01'), d('2017-01-02'), d('2017-01-03')])
        a3 = np.array([d('2016-01-01'), d('2016-01-02'), d('2018-01-03')])


        tb1 = TypeBlocks.from_none()
        tb1.append(a1)
        tb1.append(a2)
        tb1.append(a3)

        self.assertEqual(list(tb1._reblock_signature()),
            [(np.dtype('<M8[D]'), 3)])

        tb2 = tb1.consolidate()

        self.assertEqual(list(tb2._reblock_signature()),
            [(np.dtype('<M8[D]'), 3)])

        self.assertEqual(len(tb2._blocks), 1)


if __name__ == '__main__':
    unittest.main()
