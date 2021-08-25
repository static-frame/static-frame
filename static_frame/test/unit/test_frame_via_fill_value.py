

import unittest

import frame_fixtures as ff
import numpy as np

from static_frame import Frame
from static_frame import FrameGO
from static_frame import Series
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    def test_frame_via_fill_value_a(self) -> None:

        f1 = ff.parse('s(3,3)|c(I,str)|i(I,str)|v(int)')
        f2 = ff.parse('s(2,2)|c(I,str)|i(I,str)|v(int)')

        f3 = f1.via_fill_value(0) + f2
        self.assertEqual(f3.to_pairs(),
                (('zUvW', (('zUvW', 30205), ('zZbu', -3648), ('ztsv', 91301))), ('zZbu', (('zUvW', 84967), ('zZbu', -176034), ('ztsv', 185734))), ('ztsv', (('zUvW', 5729), ('zZbu', 324394), ('ztsv', -82314))))
                )


        f4 = f1.via_fill_value(0) + f2.iloc[0]
        self.assertEqual(f4.to_pairs(),
                (('zUvW', (('zZbu', -3648), ('ztsv', 91301), ('zUvW', 30205))), ('zZbu', (('zZbu', -176034), ('ztsv', 4850), ('zUvW', -3050))), ('ztsv', (('zZbu', 324394), ('ztsv', 121040), ('zUvW', 167926))))
                )


    def test_frame_via_fill_value_b(self) -> None:

        f1 = ff.parse('s(3,3)|c(I,str)|i(I,str)|v(int)')
        f2 = ff.parse('s(2,2)|c(I,str)|i(I,int)|v(int)') % 3

        f3 =  f1.via_T.via_fill_value(1) * f2.iloc[0]

        self.assertEqual(f3.to_pairs(),
                (('zZbu', (('zUvW', 84967), ('zZbu', 0), ('ztsv', 185734))), ('ztsv', (('zUvW', 5729), ('zZbu', 0), ('ztsv', -82314))), ('zUvW', (('zUvW', 30205), ('zZbu', 0), ('ztsv', 182602))))
                )

        f4 =  f1.via_fill_value(1).via_T * f2.iloc[0]

        self.assertEqual(f4.to_pairs(),
                (('zZbu', (('zUvW', 84967), ('zZbu', 0), ('ztsv', 185734))), ('ztsv', (('zUvW', 5729), ('zZbu', 0), ('ztsv', -82314))), ('zUvW', (('zUvW', 30205), ('zZbu', 0), ('ztsv', 182602))))
                )


    def test_frame_via_fill_value_c(self) -> None:
        f1 = Frame(np.arange(20).reshape(4, 5), index=tuple('abcd'))
        f2 = f1.via_T.via_fill_value(0) * Series((0, 2), index=tuple('bc'))
        self.assertEqual(f2.to_pairs(),
                ((0, (('a', 0), ('b', 0), ('c', 20), ('d', 0))), (1, (('a', 0), ('b', 0), ('c', 22), ('d', 0))), (2, (('a', 0), ('b', 0), ('c', 24), ('d', 0))), (3, (('a', 0), ('b', 0), ('c', 26), ('d', 0))), (4, (('a', 0), ('b', 0), ('c', 28), ('d', 0)))))

    def test_frame_via_fill_value_d(self) -> None:
        f1 = Frame(np.arange(20).reshape(4, 5), index=tuple('abcd'))
        with self.assertRaises(RuntimeError):
            f2 = f1.via_T.via_fill_value(0) * Series((0, 2), index=tuple('bc')).via_fill_value(1)

    def test_frame_via_fill_value_e(self) -> None:

        f1 = FrameGO(index=range(5))
        f1.via_fill_value(0)['a'] = Series([10,20], index=(2,4))
        f1.via_fill_value(-1)['b'] = Series([10,20], index=(0,1))

        self.assertEqual(f1.to_pairs(),
                (('a', ((0, 0), (1, 0), (2, 10), (3, 0), (4, 20))), ('b', ((0, 10), (1, 20), (2, -1), (3, -1), (4, -1))))
                )

        f2 = Frame(index=range(5))
        with self.assertRaises(TypeError):
            f2.via_fill_value(0)['a'] = range(5) # type: ignore #pylint: disable=E1137




if __name__ == '__main__':
    unittest.main()

