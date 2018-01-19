import itertools
from itertools import zip_longest
import unittest
from collections import OrderedDict
from io import StringIO
from io import BytesIO


import pandas as pd
import numpy as np


# assuming located in the same directory
from static_frame import Index
from static_frame import IndexGrowOnly
from static_frame import Series
from static_frame import Frame
from static_frame import TypeBlocks
from static_frame import Display
from static_frame import mloc

nan = np.nan

class TestUnit(unittest.TestCase):

    def setUp(self):
        pass

    def assertTypeBlocksArrayEqual(self, tb: TypeBlocks, match, match_dtype=None):
        '''
        Args:
            tb: a TypeBlocks instance
            match: can be anything that can be used to create an array.
        '''
        # could use np.testing
        if not isinstance(match, np.ndarray):
            match = np.array(match, dtype=match_dtype)
        self.assertTrue((tb.values == match).all())


    def assertAlmostEqualItems(self, pairs1, pairs2):
        for (k1, v1), (k2, v2) in zip_longest(pairs1, pairs2):
            self.assertEqual(k1, k2)

            if isinstance(v1, float) and np.isnan(v1) and isinstance(v2, float) and np.isnan(v2):
                continue

            self.assertEqual(v1, v2)


    #---------------------------------------------------------------------------
    # type blocks


    def test_usage_examples(self):

        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([1,2,3])
        a2 = np.array([10,50,30])
        a3 = np.array([1345,2234,3345])
        a4 = np.array([False, True, False])
        a5 = np.array([False, False, False])
        a6 = np.array(['g', 'd', 'e'])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3, a4, a5, a6))

        # can show that with tb2, a6 remains unchanged

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb3 = TypeBlocks.from_blocks((a1, a2, a3))

        # showing that slices keep the same memory location
        # self.assertTrue((tb1[0:2].mloc == tb1.mloc[:2]).all())
        # self.assertTrue((tb1.mloc[:2] == tb1.iloc[0:2, 0:2].mloc).all())

    def test_contiguous_pairs(self):

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



    def test_sf_type_blocks_a(self):

        # given naturally of a list of rows; this corresponds to what we get with iloc, where we select a row first, then a column
        a1 = np.array([[1,2,3], [4,5,6]])
        # shape is given as rows, columns
        self.assertEqual(a1.shape, (2, 3))
        # which is the same as pandsa
        self.assertEqual(pd.DataFrame(a1).shape, (2, 3))
        self.assertEqual(len(pd.DataFrame(a1).columns), 3)


        a2 = np.array([[.2, .5, .4], [.8, .6, .5]])
        a3 = np.array([['a', 'b'], ['c', 'd']])


        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertTypeBlocksArrayEqual(tb1[2], [3, 6])
        self.assertTypeBlocksArrayEqual(tb1[4], [0.5, 0.6])

        self.assertEqual(list(tb1[7].values), ['b', 'd'])

        self.assertEqual(tb1.shape, (2, 8))

        self.assertEqual(len(tb1), 2)
        self.assertEqual(tb1._row_dtype, object)

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



    def test_sf_type_blocks_b(self):

        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        a4 = np.array(['gd', 'cd', 'dd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3, a4))

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
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



    def test_sf_type_blocks_c(self):
        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])


        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertEqual(tb1.iloc[0:2].shape, (2, 8))
        self.assertEqual(tb1.iloc[1:3].shape, (2, 8))

        #tb1.iloc[0, 1:5]


    def test_indices_to_contiguous_pairs(self):

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])


        tb1 = TypeBlocks.from_blocks((a1, a2, a3))
        self.assertEqual(list(tb1._key_to_block_slices(0)), [(0, 0)])
        self.assertEqual(list(tb1._key_to_block_slices(6)), [(2, 0)])
        self.assertEqual(list(tb1._key_to_block_slices((3,5,6))),
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





    def test_extract_a(self):

        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        a4 = np.array(['gd', 'cd', 'dd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3, a4))

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
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
        self.assertEqual(tb2._extract((1,2),).shape, (2, 8))
        self.assertEqual(tb2._extract((0,2),).shape, (2, 8))
        self.assertEqual(tb2._extract((0,2), 6).shape, (2, 1))
        self.assertEqual(tb2._extract((0,2), (6,7)).shape, (2, 2))

        # mixed
        self.assertEqual(tb2._extract(1,).shape, (1, 8))
        self.assertEqual(tb2._extract((0,2)).shape, (2, 8))
        self.assertEqual(tb2._extract(1, 4), False)
        self.assertEqual(tb2._extract(1, 3), True)
        self.assertEqual(tb2._extract((0, 2),).shape, (2, 8))


        # slices
        self.assertEqual(tb2._extract(slice(1,3)).shape, (2, 8))
        self.assertEqual(tb2._extract(slice(1,3), slice(3,6)).shape, (2,3))
        self.assertEqual(tb2._extract(slice(1,2)).shape, (1,8))
        # a boundry over extended still gets 1
        self.assertEqual(tb2._extract(slice(2,4)).shape, (1,8))
        self.assertEqual(tb2._extract(slice(None), slice(2,4)).shape, (3, 2))
        self.assertEqual(tb1._extract(slice(2,4)).shape, (1, 4))


    def test_extract_b(self):
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

    def test_extract_c(self):
        # test negative slices

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
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





    def test_immutable_filter(self):
        a1 = np.array([3, 4, 5])
        a2 = TypeBlocks.immutable_filter(a1)
        with self.assertRaises(ValueError):
            a2[0] = 34
        a3 = a2[:2]
        with self.assertRaises(ValueError):
            a3[0] = 34


    def test_type_blocks_static_frame(self):
        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertTrue(tb1.dtypes[0] == np.int64)


    def test_type_blocks_attributes(self):

        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertEqual(tb1.size, 9)
        self.assertEqual(tb2.size, 24)




    def test_type_blocks_block_pointers(self):

        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertTrue((tb1[0:2].mloc == tb1.mloc[:2]).all())
        self.assertTrue((tb1.mloc[:2] == tb1.iloc[0:2, 0:2].mloc).all())


    def test_type_blocks_append(self):
        a1 = np.array([1,2,3])
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

    def test_unary_operator_a(self):

        a1 = np.array([1,-2,-3])
        a2 = np.array([False, True, False])
        tb1 = TypeBlocks.from_blocks((a1, a2))

        tb2 = ~tb1 # tilde
        self.assertEqual(
            (~tb1.values).tolist(),
            [[-2, -1], [1, -2], [2, -1]])

    def test_unary_operator_b(self):

        a1 = np.array([[1,2,3], [-4,5,6], [0,0,-1]])
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



    def test_block_compatible_a(self):

        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
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
    def test_block_compatible_b(self):

        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))

        tb2a = tb2[[2,3,7]]
        self.assertTrue(tb1.block_compatible(tb2a))



    def test_consolidate_a(self):

        a1 = np.array([1,2,3])
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


    def test_consolidate_b(self):
        # if we hava part of TB consolidated, we do not reallocate


        a1 = np.array([
            [1,2,3],
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




    def test_binary_operator_a(self):

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



    def test_binary_operator_b(self):

        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
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


    def test_extend_a(self):
        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
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


    def test_binary_operator_c(self):
        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
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


        #-----------------------------------------------------------------------
        # index tests


    # def test_sf_index_a(self):
    #     idx = Index(['a', 'b', 'c'])

    #     # this gets a slice
    #     self.assertEqual(idx[1:], [1, 2])

    #     self.assertEqual(idx['b'], 1)

    #     self.assertEqual(idx[[False, True, True]], [1, 2])

    def test_index_loc_to_iloc_a(self):

        idx = Index(('a', 'b', 'c', 'd'))

        self.assertEqual(
                idx.loc_to_iloc(np.array([True, False, True, False])).tolist(),
                [0, 2])

        self.assertEqual(idx.loc_to_iloc(slice('c',)), slice(None, 3, None))
        self.assertEqual(idx.loc_to_iloc(slice('b','d')), slice(1, 4, None))
        self.assertEqual(idx.loc_to_iloc('d'), 3)


    def test_index_mloc_a(self):
        idx = Index(('a', 'b', 'c', 'd'))
        self.assertTrue(idx.mloc == idx[:2].mloc)


    def test_index_unique(self):

        with self.assertRaises(KeyError):
            idx = Index(('a', 'b', 'c', 'a'))
        with self.assertRaises(KeyError):
            idx = IndexGrowOnly(('a', 'b', 'c', 'a'))

        with self.assertRaises(KeyError):
            idx = Index(['a', 'a'])
        with self.assertRaises(KeyError):
            idx = IndexGrowOnly(['a', 'a'])

        with self.assertRaises(KeyError):
            idx = Index(np.array([True, False, True], dtype=bool))
        with self.assertRaises(KeyError):
            idx = IndexGrowOnly(np.array([True, False, True], dtype=bool))

        # acceptable but not advisiable
        idx = Index([0, '0'])


    def test_index_creation_a(self):
        idx = Index(('a', 'b', 'c', 'd'))

        #idx2 = idx['b':'d']

        self.assertEqual(idx.values.tolist(), ['a', 'b', 'c', 'd'])

        self.assertEqual(idx[2:].values.tolist(), ['c', 'd'])

        self.assertEqual(idx.loc['b':].values.tolist(), ['b', 'c', 'd'])

        self.assertEqual(idx.loc['b':'d'].values.tolist(), ['b', 'c', 'd'])

        self.assertEqual(idx.loc_to_iloc(('b', 'b', 'c')), [1, 1, 2])

        self.assertEqual(idx.loc['c'].values.tolist(), ['c'])



        idxgo = IndexGrowOnly(('a', 'b', 'c', 'd'))
        self.assertEqual(idxgo.values.tolist(), ['a', 'b', 'c', 'd'])

        idxgo.append('e')
        self.assertEqual(idxgo.values.tolist(), ['a', 'b', 'c', 'd', 'e'])

        idxgo.extend(('f', 'g'))
        self.assertEqual(idxgo.values.tolist(), ['a', 'b', 'c', 'd', 'e', 'f', 'g'])




    def test_index_unary_operators_a(self):
        idx = Index((20, 30, 40, 50))

        invert_idx = -idx
        self.assertEqual(invert_idx.tolist(),
                [-20, -30, -40, -50],)

        # this is strange but consistent with NP
        not_idx = ~idx
        self.assertEqual(not_idx.tolist(),
                [-21, -31, -41, -51],)

    def test_index_binary_operators_a(self):
        idx = Index((20, 30, 40, 50))

        self.assertEqual((idx + 2).tolist(),
                [22, 32, 42, 52])
        self.assertEqual((2 + idx).tolist(),
                [22, 32, 42, 52])
        self.assertEqual((idx * 2).tolist(),
                [40, 60, 80, 100])
        self.assertEqual((2 * idx).tolist(),
                [40, 60, 80, 100])
        self.assertEqual((idx - 2).tolist(),
                [18, 28, 38, 48])
        self.assertEqual(
                (2 - idx).tolist(),
                [-18, -28, -38, -48])


    def test_index_binary_operators_b(self):
        '''Both opperands are Index instances
        '''
        idx1 = Index((20, 30, 40, 50))
        idx2 = Index((20, 3, 4, 5))

        self.assertEqual((idx1 == idx2).tolist(), [True, False, False, False])




    def test_series_index_reassign_a(self):

        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        s1.index = ('x', 'y', 'z', 'q')

        self.assertEqual(list(s1.items()),
            [('x', 0), ('y', 1), ('z', 2), ('q', 3)])

        with self.assertRaises(Exception):
            s1.index = ('b','c','d')

    def test_series_slice_a(self):
        # create a series from a single value
        # s0 = Series(3, index=('a',))

        # generator based construction of values and index
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        # self.assertEqual(s1['b'], 1)
        # self.assertEqual(s1['d'], 3)

        s2 = s1['a':'c'] # with Pandas this is inclusive
        self.assertEqual(s2.values.tolist(), [0, 1, 2])
        self.assertTrue(s2['b'] == s1['b'])

        s3 = s1['c':]
        self.assertEqual(s3.values.tolist(), [2, 3])
        self.assertTrue(s3['d'] == s1['d'])

        self.assertEqual(s1['b':'d'].values.tolist(), [1, 2, 3])

        self.assertEqual(s1[['a', 'c']].values.tolist(), [0, 2])


    def test_series_intersection_a(self):
        # create a series from a single value
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        s3 = s1['c':]
        self.assertEqual(s1.index.intersection(s3.index).values.tolist(),
            ['c', 'd'])


    def test_series_intersection_b(self):
        # create a series from a single value
        idxa = IndexGrowOnly(('a', 'b', 'c'))
        idxb = IndexGrowOnly(('b', 'c', 'd'))

        self.assertEqual(idxa.intersection(idxb).values.tolist(),
            ['b', 'c'])

        self.assertEqual(idxa.union(idxb).values.tolist(),
            ['a', 'b', 'c', 'd'])




    def test_series_binary_operator_a(self):
        '''Test binary operators where one operand is a numeric.
        '''
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        self.assertEqual(list((s1 * 3).items()),
                [('a', 0), ('b', 3), ('c', 6), ('d', 9)])

        self.assertEqual(list((s1 / .5).items()),
                [('a', 0.0), ('b', 2.0), ('c', 4.0), ('d', 6.0)])

        self.assertEqual(list((s1 ** 3).items()),
                [('a', 0), ('b', 1), ('c', 8), ('d', 27)])


    def test_series_binary_operator_b(self):
        '''Test binary operators with Series of same index
        '''
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        s2 = Series((x * 2 for x in range(4)), index=('a', 'b', 'c', 'd'))

        self.assertEqual(list((s1 + s2).items()),
                [('a', 0), ('b', 3), ('c', 6), ('d', 9)])

        self.assertEqual(list((s1 * s2).items()),
                [('a', 0), ('b', 2), ('c', 8), ('d', 18)])




    def test_series_binary_operator_c(self):
        '''Test binary operators with Series of different index
        '''
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        s2 = Series((x * 2 for x in range(4)), index=('c', 'd', 'e', 'f'))

        self.assertAlmostEqualItems(list((s1 * s2).items()),
                [('a', nan), ('b', nan), ('c', 0), ('d', 6), ('e', nan), ('f', nan)]
                )


    def test_series_reindex_a(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        s2 = s1.reindex(('c', 'd', 'a'))
        self.assertEqual(list(s2.items()), [('c', 2), ('d', 3), ('a', 0)])

        s3 = s1.reindex(['a','b'])
        self.assertEqual(list(s3.items()), [('a', 0), ('b', 1)])


        # an int-valued array is hard to provide missing values for

        s4 = s1.reindex(['b', 'q', 'g', 'a'], fill_value=None)
        self.assertEqual(list(s4.items()),
                [('b', 1), ('q', None), ('g', None), ('a', 0)])

        # by default this gets float because filltype is nan by default
        s5 = s1.reindex(['b', 'q', 'g', 'a'])
        self.assertAlmostEqualItems(list(s5.items()),
                [('b', 1), ('q', nan), ('g', nan), ('a', 0)])


    def test_series_isnull(self):

        s1 = Series((234.3, 3.2, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s2 = Series((234.3, None, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s3 = Series((234.3, 5, 6.4, -234.3), index=('a', 'b', 'c', 'd'))
        s4 = Series((234.3, None, None, None), index=('a', 'b', 'c', 'd'))
        s5 = Series(('p', 'q', 'e', 'g'), index=('a', 'b', 'c', 'd'))
        s6 = Series((False, True, False, True), index=('a', 'b', 'c', 'd'))

        self.assertEqual(list(s1.isnull().items()),
                [('a', False), ('b', False), ('c', False), ('d', True)]
                )
        self.assertEqual(list(s2.isnull().items()),
                [('a', False), ('b', True), ('c', False), ('d', True)])

        self.assertEqual(list(s3.isnull().items()),
                [('a', False), ('b', False), ('c', False), ('d', False)])

        self.assertEqual(list(s4.isnull().items()),
                [('a', False), ('b', True), ('c', True), ('d', True)])

        # those that are always false
        self.assertEqual(list(s5.isnull().items()),
                [('a', False), ('b', False), ('c', False), ('d', False)])

        self.assertEqual(list(s6.isnull().items()),
                [('a', False), ('b', False), ('c', False), ('d', False)])



    def test_series_notnull(self):

        s1 = Series((234.3, 3.2, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s2 = Series((234.3, None, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s3 = Series((234.3, 5, 6.4, -234.3), index=('a', 'b', 'c', 'd'))
        s4 = Series((234.3, None, None, None), index=('a', 'b', 'c', 'd'))
        s5 = Series(('p', 'q', 'e', 'g'), index=('a', 'b', 'c', 'd'))
        s6 = Series((False, True, False, True), index=('a', 'b', 'c', 'd'))

        self.assertEqual(list(s1.notnull().items()),
                [('a', True), ('b', True), ('c', True), ('d', False)]
                )
        self.assertEqual(list(s2.notnull().items()),
                [('a', True), ('b', False), ('c', True), ('d', False)])

        self.assertEqual(list(s3.notnull().items()),
                [('a', True), ('b', True), ('c', True), ('d', True)])

        self.assertEqual(list(s4.notnull().items()),
                [('a', True), ('b', False), ('c', False), ('d', False)])

        # those that are always false
        self.assertEqual(list(s5.notnull().items()),
                [('a', True), ('b', True), ('c', True), ('d', True)])

        self.assertEqual(list(s6.notnull().items()),
                [('a', True), ('b', True), ('c', True), ('d', True)])


    def test_series_dropna(self):

        s1 = Series((234.3, 3.2, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s2 = Series((234.3, None, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s3 = Series((234.3, 5, 6.4, -234.3), index=('a', 'b', 'c', 'd'))
        s4 = Series((234.3, None, None, None), index=('a', 'b', 'c', 'd'))
        s5 = Series(('p', 'q', 'e', 'g'), index=('a', 'b', 'c', 'd'))
        s6 = Series((False, True, False, True), index=('a', 'b', 'c', 'd'))


        self.assertEqual(list(s2.dropna().items()),
                [('a', 234.3), ('c', 6.4)])


    def test_series_fillna_a(self):

        s1 = Series((234.3, 3.2, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s2 = Series((234.3, None, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s3 = Series((234.3, 5, 6.4, -234.3), index=('a', 'b', 'c', 'd'))
        s4 = Series((234.3, None, None, None), index=('a', 'b', 'c', 'd'))
        s5 = Series(('p', 'q', 'e', 'g'), index=('a', 'b', 'c', 'd'))
        s6 = Series((False, True, False, True), index=('a', 'b', 'c', 'd'))
        s7 = Series((10, 20, 30, 40), index=('a', 'b', 'c', 'd'))
        s8 = Series((234.3, None, 6.4, np.nan, 'q'), index=('a', 'b', 'c', 'd', 'e'))


        self.assertEqual(s1.fillna(0.0).values.tolist(),
                [234.3, 3.2, 6.4, 0.0])

        self.assertEqual(s1.fillna(-1).values.tolist(),
                [234.3, 3.2, 6.4, -1.0])

        # given a float array, inserting None, None is casted to nan
        self.assertEqual(s1.fillna(None).values.tolist(),
                [234.3, 3.2, 6.4, None])

        post = s1.fillna('wer')
        self.assertEqual(post.dtype, object)
        self.assertEqual(post.values.tolist(),
                [234.3, 3.2, 6.4, 'wer'])


        post = s7.fillna(None)
        self.assertEqual(post.dtype, int)

    def test_display_a(self):

        d = Display.from_values(np.array([[1, 2], [3, 4]]), 'header')

        self.assertEqual(list(d),
            [['header '], ['1 2    '], ['3 4    '], ['int64  ']])


    def test_type_blocks_display_a(self):

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb = TypeBlocks.from_blocks((a1, a2, a3))

        disp = tb.display()
        self.assertEqual(len(disp), 5)
        self.assertEqual(list(disp),
            [['<TypeBlocks> ', '                  ', '          '],
            ['1 2 3        ', 'False False  True ', "'a' 'b'   "],
            ['4 5 6        ', ' True False  True ', "'c' 'd'   "],
            ['0 0 1        ', ' True False  True ', "'oe' 'od' "],
            ['int64        ', 'bool              ', '<U2       ']])


    def test_type_blocks_axis_values_a(self):
        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertEqual(
            list(a.tolist() for a in tb.axis_values(0)),
            [[1, 2, 3, False, False, True, 'a', 'b'], [4, 5, 6, True, False, True, 'c', 'd'], [0, 0, 1, True, False, True, 'oe', 'od']]
            )

        self.assertEqual(list(a.tolist() for a in tb.axis_values(1)),
            [[1, 4, 0], [2, 5, 0], [3, 6, 1], [False, True, True], [False, False, False], [True, True, True], ['a', 'c', 'oe'], ['b', 'd', 'od']]
            )

        # we are iterating over slices so we get views of columns without copying
        self.assertEqual(tb.mloc[0], mloc(next(tb.axis_values(1))))

    def test_frame_init_a(self):

        frame = Frame(dict(a=[3,4,5], b=[6,3,2]))
        self.assertEqual(list(frame.display()),
            [['<Frame>         ', '      ', '      ', '    '],
            ['<IndexGrowOnly> ', 'a     ', 'b     ', '<U1 '],
            ['<Index>         ', '      ', '      ', '    '],
            ['0               ', '3     ', '6     ', '    '],
            ['1               ', '4     ', '3     ', '    '],
            ['2               ', '5     ', '2     ', '    '],
            ['int64           ', 'int64 ', 'int64 ', '    ']]
            )

        frame = Frame(OrderedDict((('b', [6,3,2]), ('a', [3,4,5]))))
        self.assertEqual(list(frame.display()),
            [['<Frame>         ', '      ', '      ', '    '],
            ['<IndexGrowOnly> ', 'b     ', 'a     ', '<U1 '],
            ['<Index>         ', '      ', '      ', '    '],
            ['0               ', '6     ', '3     ', '    '],
            ['1               ', '3     ', '4     ', '    '],
            ['2               ', '2     ', '5     ', '    '],
            ['int64           ', 'int64 ', 'int64 ', '    ']]
            )


    def test_frame_getitem_a(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        # # show that block hav ebeen consolidated
        # self.assertEqual(len(f1._blocks._blocks), 3)

        # s1 = f1['s']
        # self.assertTrue((s1.index == f1.index).all())

        # # we have not copied the index array
        # self.assertEqual(mloc(f1.index.values), mloc(s1.index.values))

        f2 = f1['r':]
        self.assertEqual(f2.columns.values.tolist(), ['r', 's', 't'])
        self.assertTrue((f2.index == f1.index).all())
        self.assertEqual(mloc(f2.index.values), mloc(f1.index.values))




    def test_frame_from_csv_a(self):
        filelike = BytesIO(b'''count,number,weight,scalar,color,active
0,4,234.5,5.3,'red',False
30,50,9.234,5.434,'blue',True''')
        f1 = Frame.from_csv(filelike)

        self.assertEqual(f1.columns.values.tolist(),
                ['count', 'number', 'weight', 'scalar', 'color', 'active'])


    def test_series_mask_a(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        self.assertEqual(
                s1.mask.loc[['b', 'd']].assign(3000).values.tolist(),
                [0, 3000, 2, 3000])



    def test_frame_length_a(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        self.assertEqual(len(f1), 2)



    def test_frame_iloc_a(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        self.assertEqual((f1.iloc[0].values == f1.loc['x'].values).all(), True)
        self.assertEqual((f1.iloc[1].values == f1.loc['y'].values).all(), True)



    def test_frame_iter_a(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        self.assertEqual((f1.keys() == f1.columns).all(), True)
        self.assertEqual([x for x in f1.columns], ['p', 'q', 'r', 's', 't'])
        self.assertEqual([x for x in f1], ['p', 'q', 'r', 's', 't'])


    def test_index_contains_a(self):

        index = Index(('a', 'b', 'c'))
        self.assertTrue('a' in index)
        self.assertTrue('d' not in index)


    def test_index_grow_only_a(self):

        index = IndexGrowOnly(('a', 'b', 'c'))
        index.append('d')
        self.assertEqual(index.loc_to_iloc('d'), 3)
        # import ipdb; ipdb.set_trace()

        index.extend(('e', 'f'))
        self.assertEqual(index.loc_to_iloc('e'), 4)
        self.assertEqual(index.loc_to_iloc('f'), 5)


    def test_seies_loc_extract_a(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        # TODO: raaise exectin when doing a loc that Pandas reindexes



    def test_frame_setitem_a(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        f1['a'] = (False, True)
        self.assertEqual(f1['a'].values.tolist(), [False, True])

        # test index alginment
        f1['b'] = Series((3,2,5), index=('y', 'x', 'g'))
        self.assertEqual(f1['b'].values.tolist(), [2, 3])

        f1['c'] = Series((300,200,500), index=('y', 'j', 'k'))
        self.assertAlmostEqualItems(f1['c'].items(), [('x', nan), ('y', 300)])

    def test_frame_extend_columns_a(self):
        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        columns = OrderedDict(
            (('c', np.array([0, -1])), ('d', np.array([3, 5]))))

        f1.extend_columns(columns.keys(), columns.values())

        self.assertEqual(f1.columns.values.tolist(),
                ['p', 'q', 'r', 's', 't', 'c', 'd'])

        self.assertTypeBlocksArrayEqual(f1._blocks,
                [[1, 2, 'a', False, True, 0, 3],
                [30, 50, 'b', True, False, -1, 5]],
                match_dtype=object)

    def test_frame_extend_blocks_a(self):
        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        blocks = ([[50, 40], [30, 20]], [[50, 40], [30, 20]])
        columns = ('a', 'b', 'c', 'd')
        f1.extend_blocks(columns, blocks)

        self.assertEqual(f1.columns.values.tolist(),
                ['p', 'q', 'r', 's', 't', 'a', 'b', 'c', 'd'])

        self.assertEqual(f1.values.tolist(),
                [[1, 2, 'a', False, True, 50, 40, 50, 40],
                [30, 50, 'b', True, False, 30, 20, 30, 20]]
                )

if __name__ == '__main__':
    unittest.main()
    # t = TestUnit()
    # t.test_frame_iloc_a()
