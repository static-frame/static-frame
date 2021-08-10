
import unittest

import numpy as np

from static_frame.test.test_case import TestCase
from static_frame.core.rank import rank_1d
from static_frame.core.rank import rank_2d

from static_frame.core.rank import RankMethod

class TestUnit(TestCase):

    def test_rank_ordinal_a(self) -> None:
        self.assertEqual(
                rank_1d(np.array([3, 2, 6, 20]), RankMethod.ORDINAL).tolist(),
                [1, 0, 2, 3]
                )

        self.assertEqual(
                rank_1d(np.array([3, 2, 6, 20]), RankMethod.ORDINAL, start=1).tolist(),
                [2, 1, 3, 4]
                )


        a2 = rank_1d(np.array([3, 2, 6, 20]), RankMethod.ORDINAL, ascending=False)
        self.assertEqual(a2.tolist(),
                [2, 3, 1, 0]
                )

    def test_rank_ordinal_b(self) -> None:

        a1 = rank_1d(np.array([0, 2, 3, 2]), 'ordinal', start=1)
        self.assertEqual(a1.tolist(),
                [1, 2, 4, 3]
                )

        a2 = rank_1d(np.array([0, 2, 3, 2]), 'ordinal', start=0)
        self.assertEqual(a2.tolist(),
                [0, 1, 3, 2]
                )

        a3 = rank_1d(np.array([0, 2, 3, 2]), 'ordinal', ascending=False)
        self.assertEqual(a3.tolist(),
                [3, 2, 0, 1]
                )

    def test_rank_ordinal_c(self) -> None:
        a1 = rank_1d(np.array([8, 15, 7, 2, 20, 4, 20, 7, 15, 15]), 'ordinal', start=1)
        self.assertEqual(a1.tolist(),
                [5, 6, 3, 1, 9, 2, 10, 4, 7, 8]
                )
        #scipy: [5, 6, 3, 1, 9, 2, 10, 4, 7, 8]

    def test_rank_ordinal_d(self) -> None:
        a1 = rank_1d(np.array([17, 10, 3, -4, 9, 3, -12, 18, 3, 0, 17]), 'ordinal', start=1)
        self.assertEqual(a1.tolist(),
                [9, 8, 4, 2, 7, 5, 1, 11, 6, 3, 10]
                )
        #scipy: [9, 8, 4, 2, 7, 5, 1, 11, 6, 3, 10]







    def test_rank_average_a(self) -> None:

        a1 = rank_1d(np.array([0, 2, 3, 2]), 'mean', ascending=True)
        self.assertEqual(a1.tolist(),
                [0.0, 1.5, 3.0, 1.5]
                )

        a2 = rank_1d(np.array([0, 2, 3, 2]), 'mean', start=1)
        self.assertEqual(a2.tolist(),
                [1.0, 2.5, 4.0, 2.5]
                )

        a3 = rank_1d(np.array([0, 2, 3, 2]), 'mean', start=1, ascending=False)
        self.assertEqual(a3.tolist(),
                [4.0, 2.5, 1.0, 2.5]
                )


    def test_rank_average_b(self) -> None:

        a1 = rank_1d(np.array([0, 2, 5, 2, 2, 2]), 'mean', ascending=True)
        self.assertEqual(a1.tolist(),
                [0.0, 2.5, 5.0, 2.5, 2.5, 2.5]
                )

        a1 = rank_1d(np.array([0, 2, 5, 2, 2, 2]), 'mean', ascending=True, start=1)
        self.assertEqual(a1.tolist(),
                [1.0, 3.5, 6.0, 3.5, 3.5, 3.5]
                )
        #scipy: [1.0, 3.5, 6.0, 3.5, 3.5, 3.5]

        # import ipdb; ipdb.set_trace()
        a2 = rank_1d(np.array([0, 2, 3, 2]), 'mean', start=1)
        self.assertEqual(a2.tolist(),
                [1.0, 2.5, 4.0, 2.5]
                )

        a3 = rank_1d(np.array([0, 2, 3, 2]), 'mean', start=1, ascending=False)
        self.assertEqual(a3.tolist(),
                [4.0, 2.5, 1.0, 2.5]
                )

    def test_rank_average_c(self) -> None:
        a1 = rank_1d(np.array([8, 15, 7, 2, 20, 4, 20, 7, 15, 15]), 'mean', start=1)
        self.assertEqual(a1.tolist(),
                [5.0, 7.0, 3.5, 1.0, 9.5, 2.0, 9.5, 3.5, 7.0, 7.0]
                )
        #scipy: [5.0, 7.0, 3.5, 1.0, 9.5, 2.0, 9.5, 3.5, 7.0, 7.0]

    def test_rank_average_d(self) -> None:
        a1 = rank_1d(np.array([17, 10, 3, -4, 9, 3, -12, 18, 3, 0, 17]), 'mean', start=1)
        self.assertEqual(a1.tolist(),
                [9.5, 8.0, 5.0, 2.0, 7.0, 5.0, 1.0, 11.0, 5.0, 3.0, 9.5]
                )
        #scipy: [9.5, 8.0, 5.0, 2.0, 7.0, 5.0, 1.0, 11.0, 5.0, 3.0, 9.5






    def test_rank_min_a(self) -> None:

        a1 = rank_1d(np.array([0, 2, 3, 2]), 'min', start=1)
        self.assertEqual(a1.tolist(),
                [1, 2, 4, 2]
                )
        a2 = rank_1d(np.array([0, 2, 3, 2]), 'min', start=0)
        self.assertEqual(a2.tolist(),
                [0, 1, 3, 1]
                )

        a3 = rank_1d(np.array([0, 2, 3, 2]), 'min', ascending=False)
        self.assertEqual(a3.tolist(),
                [3, 1, 0, 1]
                )

    def test_rank_min_b(self) -> None:
        a1 = rank_1d(np.array([8, 15, 7, 2, 20, 4, 20, 7, 15, 15]), 'min', start=1)
        self.assertEqual(a1.tolist(),
                [5, 6, 3, 1, 9, 2, 9, 3, 6, 6]
                )
        #scipy: [5, 6, 3, 1, 9, 2, 9, 3, 6, 6]

    def test_rank_min_c(self) -> None:
        a1 = rank_1d(np.array([17, 10, 3, -4, 9, 3, -12, 18, 3, 0, 17]), 'min', start=1)
        self.assertEqual(a1.tolist(),
                [9, 8, 4, 2, 7, 4, 1, 11, 4, 3, 9]
                )

        #scipy: [9, 8, 4, 2, 7, 4, 1, 11, 4, 3, 9]






    def test_rank_max_a(self) -> None:

        a1 = rank_1d(np.array([0, 2, 3, 2]), 'max', start=1)
        # import ipdb; ipdb.set_trace()

        self.assertEqual(a1.tolist(),
                [1, 3, 4, 3]
                )
        a2 = rank_1d(np.array([0, 2, 3, 2]), 'max', start=0)
        self.assertEqual(a2.tolist(),
                [0, 2, 3, 2]
                )

        a2 = rank_1d(np.array([0, 2, 3, 2]), 'max', ascending=False)
        self.assertEqual(a2.tolist(),
                [3, 2, 0, 2]
                )

    def test_rank_max_b(self) -> None:
        a1 = rank_1d(np.array([8, 15, 7, 2, 20, 4, 20, 7, 15, 15]), 'max', start=1)
        self.assertEqual(a1.tolist(),
                [5, 8, 4, 1, 10, 2, 10, 4, 8, 8]
                )
        #scipy: [5, 8, 4, 1, 10, 2, 10, 4, 8, 8]

    def test_rank_max_c(self) -> None:
        a1 = rank_1d(np.array([17, 10, 3, -4, 9, 3, -12, 18, 3, 0, 17]), 'max', start=1)
        self.assertEqual(a1.tolist(),
                [10, 8, 6, 2, 7, 6, 1, 11, 6, 3, 10]
                )
        #scipy: [10, 8, 6, 2, 7, 6, 1, 11, 6, 3, 10]




    def test_rank_dense_a(self) -> None:

        a1 = rank_1d(np.array([0, 2, 3, 2]), 'dense', start=1)
        self.assertEqual(a1.tolist(),
                [1, 2, 3, 2]
                )

        a2 = rank_1d(np.array([0, 2, 3, 2]), 'dense', start=0)
        self.assertEqual(a2.tolist(),
                [0, 1, 2, 1]
                )

        a3 = rank_1d(np.array([0, 2, 3, 2]), 'dense', ascending=False)
        self.assertEqual(a3.tolist(),
                [2, 1, 0, 1]
                )

    def test_rank_dense_b(self) -> None:
        a1 = rank_1d(np.array([8, 15, 7, 2, 20, 4, 20, 7, 15, 15]), 'dense', start=1)
        self.assertEqual(a1.tolist(),
                [4, 5, 3, 1, 6, 2, 6, 3, 5, 5]
                )
        #scipy: [4, 5, 3, 1, 6, 2, 6, 3, 5, 5]

    def test_rank_dense_c(self) -> None:
        a1 = rank_1d(np.array([17, 10, 3, -4, 9, 3, -12, 18, 3, 0, 17]), 'dense', start=1)
        self.assertEqual(a1.tolist(),
                [7, 6, 4, 2, 5, 4, 1, 8, 4, 3, 7]
                )
        #scipy: [7, 6, 4, 2, 5, 4, 1, 8, 4, 3, 7]





    def test_rank_2d_a(self) -> None:
        a1 = np.array([10, 3, -4, 9, 3, -12, 18, 3, 0, 17]).reshape(5,2)
        self.assertEqual(
            rank_2d(a1, axis=0, method='ordinal', start=1).tolist(),
            [[4, 2], [1, 4], [3, 1], [5, 3], [2, 5]]
        )
        self.assertEqual(
            rank_2d(a1, axis=1, method='ordinal', start=1).tolist(),
            [[2, 1], [1, 2], [2, 1], [2, 1], [1, 2]]
        )

    def test_rank_2d_b(self) -> None:
        a1 = np.array([10, 3, -4, 9, 3, -12, 18, 3, 0, 17]).reshape(5,2)
        self.assertEqual(
            rank_2d(a1, axis=0, method='mean', start=1).tolist(),
            [[4.0, 2.5], [1.0, 4.0], [3.0, 1.0], [5.0, 2.5], [2.0, 5.0]]
        )
        self.assertEqual(
            rank_2d(a1, axis=1, method='mean', start=1).tolist(),
            [[2.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 1.0], [1.0, 2.0]]
        )

    def test_rank_2d_c(self) -> None:
        a1 = np.array([10, 3, -4, 9, 3, -12, 18, 3, 0, 17]).reshape(5,2)
        self.assertEqual(
            rank_2d(a1, axis=0, method='min', start=1).tolist(),
            [[4, 2], [1, 4], [3, 1], [5, 2], [2, 5]]
        )
        self.assertEqual(
            rank_2d(a1, axis=1, method='min', start=1).tolist(),
            [[2, 1], [1, 2], [2, 1], [2, 1], [1, 2]]
        )

    def test_rank_2d_d(self) -> None:
        a1 = np.array([10, 3, -4, 9, 3, -12, 18, 3, 0, 17]).reshape(5,2)
        self.assertEqual(
            rank_2d(a1, axis=0, method='max', start=1).tolist(),
            [[4, 3], [1, 4], [3, 1], [5, 3], [2, 5]]
        )
        self.assertEqual(
            rank_2d(a1, axis=1, method='max', start=1).tolist(),
            [[2, 1], [1, 2], [2, 1], [2, 1], [1, 2]]
        )


    def test_rank_2d_e(self) -> None:
        a1 = np.array([10, 3, -4, 9, 3, -12, 18, 3, 0, 17]).reshape(5,2)
        self.assertEqual(
            rank_2d(a1, axis=0, method='dense', start=1).tolist(),
            [[4, 2], [1, 3], [3, 1], [5, 2], [2, 4]]
        )
        self.assertEqual(
            rank_2d(a1, axis=1, method='dense', start=1).tolist(),
            [[2, 1], [1, 2], [2, 1], [2, 1], [1, 2]]
        )



    #---------------------------------------------------------------------------
    def test_rank_1d_pair_a(self) -> None:
        self.assertEqual(
                rank_1d(np.array([0, 0]), 'mean').tolist(),
                [0.5, 0.5]
                )

        self.assertEqual(
                rank_1d(np.array([0, 0]), 'min').tolist(),
                [0, 0]
                )

        self.assertEqual(
                rank_1d(np.array([0, 0]), 'max').tolist(),
                [1, 1]
                )

        self.assertEqual(
                rank_1d(np.array([0, 0]), 'dense').tolist(),
                [0, 0]
                )

        self.assertEqual(
                rank_1d(np.array([0, 0]), 'ordinal').tolist(),
                [0, 1]
                )

    def test_rank_1d_pair_b(self) -> None:
        self.assertEqual(
                rank_1d(np.array([0, 0, 1]), 'mean').tolist(),
                [0.5, 0.5, 2]
                )

        self.assertEqual(
                rank_1d(np.array([0, 0, 1]), 'min').tolist(),
                [0, 0, 2]
                )

        self.assertEqual(
                rank_1d(np.array([0, 0, 1]), 'max').tolist(),
                [1, 1, 2]
                )

        self.assertEqual(
                rank_1d(np.array([0, 0, 1]), 'dense').tolist(),
                [0, 0, 1]
                )

        self.assertEqual(
                rank_1d(np.array([0, 0, 1]), 'ordinal').tolist(),
                [0, 1, 2]
                )






if __name__ == '__main__':
    unittest.main()



# the algo
# [8, 15, 7, 2, 20, 4, 20, 7, 15, 15]
# >>> a = np.array([8, 15, 7, 2, 20, 4, 20, 7, 15, 15])
# >>> a
# array([ 8, 15,  7,  2, 20,  4, 20,  7, 15, 15])


# get the index needed in each position to sort this array

# >>> index_sorted = np.argsort(a, kind='merge')
# >>> index_sorted
# array([3, 5, 2, 7, 0, 1, 8, 9, 4, 6])

# get index 3, then 5. notice 1,8,9 (15) and 4,6 (20)



# use those indices to select from a contiguous range

# >>> np.arange(a.size)
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# >>> ordinal = np.empty(a.size, dtype=int)
# >>> ordinal[index_sorted] = np.arange(a.size)
# >>> ordinal
# array([4, 5, 2, 0, 8, 1, 9, 3, 6, 7])

# not the same as this!!
# >>> np.arange(a.size)[index_sorted]

# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) range
# array([3, 5, 2, 7, 0, 1, 8, 9, 4, 6]) index_sorted
# array([4, 5, 2, 0, 8, 1, 9, 3, 6, 7]) result

# use the indices in index_sorted to position the values in the range; 0 index_sorted says put 4 in the zeroth position; 1 in index_sorted says to put 5 in the second position

# we basically unsort the range to correspond to the indices sorted



# find duplicates by using the sorted values to sort array, and then shift compare

# >>> a_sorted = a[index_sorted]
# >>> a_sorted
# array([ 2,  4,  7,  7,  8, 15, 15, 15, 20, 20])

# >>> is_unique = np.full(size, True, dtype=bool)
# >>> is_unique[1:] = a_sorted[1:] != a_sorted[:-1]

# >>> is_unique
# array([ True,  True,  True, False,  True,  True, False, False,  True,
#        False])


# get the dense rank by performing cum sum of Booleans and unsorting back to orignal values

# >>> is_unique.cumsum()
# array([1, 2, 3, 3, 4, 5, 5, 5, 6, 6])
# >>> dense = is_unique.cumsum()[ordinal]
# >>> dense
# array([4, 5, 3, 1, 6, 2, 6, 3, 5, 5])

# notice that this rank naturally starts at 1


# get the index positions of unique values

# >>> unique_pos = np.nonzero(is_unique)[0]
# >>> unique_pos
# array([0, 1, 2, 4, 5, 8])


# get an array of the equal to unique values + 1, set last value to max possible value

# >>> size_unique = len(unique_pos)
# >>> count = np.empty(size_unique + 1)
# >>> count[:size_unique] = unique_pos
# >>> count[size_unique] = len(a)
# >>> count
# array([ 0,  1,  2,  4,  5,  8, 10])

# notice that the missing indices are values that repeated in a_sorted, plus one value at the length of a
# >>> a_sorted
# array([ 2,  4,  7,  7,  8, 15, 15, 15, 20, 20])



# get the max rank: whenever there are ties, take the max value

# >>> count
# array([ 0,  1,  2,  4,  5,  8, 10])
# >>> dense
# array([4, 5, 3, 1, 6, 2, 6, 3, 5, 5])
# >>> count[dense]
# array([ 5,  8,  4,  1, 10,  2, 10,  4,  8,  8])

# position values from count by the index and order of dense; so first position is the index 4 (5), second position is index 5 (8). this gets the max rank for each tie.

# as values come from count, but we select with dense, we will never select 0; we can shift by 1 to start at zero

# >>> count[dense] - 1
# array([4, 7, 3, 0, 9, 1, 9, 3, 7, 7])



# get the min rank: whenever there are ties, take the min value

# >>> count
# array([ 0,  1,  2,  4,  5,  8, 10])
# >>> dense
# array([4, 5, 3, 1, 6, 2, 6, 3, 5, 5])
# >>> count[dense - 1]
# array([4, 5, 2, 0, 8, 1, 8, 2, 5, 5])

# as dense starts at 1, we can shift it down; this means we select from count the value on the left side of a boundary


# finally, to geth average rank, we simple take the mean of rank for min and max

# >>> .5 * ((count[dense] - 1) + count[dense - 1])
# array([4. , 6. , 2.5, 0. , 8.5, 1. , 8.5, 2.5, 6. , 6. ])


# finally invert if ascending, and shift for start



