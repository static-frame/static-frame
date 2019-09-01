import unittest

import numpy as np  # type: ignore


from static_frame.core.container_util import index_from_optional_constructor
from static_frame.core.container_util import matmul

from static_frame import Series
from static_frame import Frame

from static_frame import Index
from static_frame import IndexGO
# from static_frame import IndexDate
# from static_frame import IndexHierarchy
# from static_frame import Series
# from static_frame import Frame
# from static_frame import IndexYearMonth
# from static_frame import IndexYear
from static_frame import IndexSecond
# from static_frame import IndexMillisecond

from static_frame.test.test_case import TestCase



class TestUnit(TestCase):


    def test_index_from_optional_constructor_a(self) -> None:
        idx1 = index_from_optional_constructor([1, 3, 4],
                default_constructor=Index)
        self.assertEqual(idx1.__class__, Index)

        # given a mutable index and an immutable default, get immutable version
        idx2 = index_from_optional_constructor(IndexGO((1, 3, 4)),
                default_constructor=Index)
        self.assertEqual(idx2.__class__, Index)

        # given a mutable index and an immutable default, get immutable version
        idx3 = index_from_optional_constructor(IndexGO((1, 3, 4)),
                default_constructor=IndexGO)
        self.assertEqual(idx3.__class__, IndexGO)

        # given a mutable index and an immutable default, get immutable version
        idx4 = index_from_optional_constructor(
                IndexSecond((1, 3, 4)),
                default_constructor=Index)
        self.assertEqual(idx4.__class__, IndexSecond)


    def test_matmul_a(self) -> None:

        f1 = Frame.from_items((('a', (1, 2, 3)), ('b', (3, 4, 5))),
                index=('x', 'y', 'z'))

        self.assertEqual(
                matmul(f1, [4, 3]).to_pairs(),
                (('x', 13), ('y', 20), ('z', 27))
                )

        self.assertEqual(
                matmul(f1, np.array([4, 3])).to_pairs(),
                (('x', 13), ('y', 20), ('z', 27))
                )


        self.assertEqual(
                matmul(f1, [3, 4]).to_pairs(),
                (('x', 15), ('y', 22), ('z', 29))
                )

        self.assertEqual(
                matmul(f1, np.array([3, 4])).to_pairs(),
                (('x', 15), ('y', 22), ('z', 29))
                )


    def test_matmul_b(self) -> None:

        f1 = Frame.from_items((('a', (1, 2, 3)), ('b', (3, 4, 5))),
                index=('x', 'y', 'z'))

        # get an auto incremented integer columns
        self.assertEqual(
            matmul(f1, np.arange(10).reshape(2, 5)).to_pairs(0),
            ((0, (('x', 15), ('y', 20), ('z', 25))), (1, (('x', 19), ('y', 26), ('z', 33))), (2, (('x', 23), ('y', 32), ('z', 41))), (3, (('x', 27), ('y', 38), ('z', 49))), (4, (('x', 31), ('y', 44), ('z', 57))))
            )




    def test_matmul_c(self) -> None:

        f1 = Frame.from_items((('a', (1, 2, 3)), ('b', (3, 4, 5))),
                index=('x', 'y', 'z'))

        s1 = Series((3, 4, 2), index=('x', 'y', 'z'))

        self.assertEqual(
            matmul(s1, f1).to_pairs(),
            (('a', 17), ('b', 35))
            )


if __name__ == '__main__':
    unittest.main()
