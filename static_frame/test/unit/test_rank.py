
import unittest

import numpy as np

from static_frame.test.test_case import TestCase
from static_frame.core.rank import rank_1d
from static_frame.core.rank import RankMethod

class TestUnit(TestCase):

    def test_rank_a(self) -> None:
        self.assertEqual(
                rank_1d(np.array([3, 2, 6, 20]), RankMethod.ORDINAL).tolist(),
                [1, 0, 2, 3]
                )

        a2 = rank_1d(np.array([3, 2, 6, 20]), RankMethod.ORDINAL, ascending=False)
        self.assertEqual(a2.tolist(),
                [2, 3, 1, 0]
                )


        # a3 = rank_1d(np.array([0, 2, 3, 2]), 'average', ascending=False)
        # self.assertEqual(a3.tolist(),
        #         [4.0, 5.0, 3.0, 2.0, 1.0]
        #         )



if __name__ == '__main__':
    unittest.main()


