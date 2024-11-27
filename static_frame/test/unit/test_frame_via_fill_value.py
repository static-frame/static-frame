from __future__ import annotations

import frame_fixtures as ff
import numpy as np

from static_frame import Frame
from static_frame import FrameGO
from static_frame import HLoc
from static_frame import IndexHierarchy
from static_frame import IndexYearMonth
from static_frame import IndexYearMonthGO
from static_frame import Series
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    def test_frame_via_fill_value_a1(self) -> None:

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

    def test_frame_via_fill_value_a2(self) -> None:

        f1 = ff.parse('s(3,3)|c(I,str)|i(I,str)|v(int64)')
        f2 = ff.parse('s(2,2)|c(I,str)|i(I,str)|v(int64)')

        f3 = f1.via_fill_value({'zZbu': 0, 'ztsv': 1, 'zUvW': 1}) * f2
        self.assertEqual(f3.to_pairs(),
                (('zUvW', (('zUvW', 30205), ('zZbu', -3648), ('ztsv', 91301))), ('zZbu', (('zUvW', 0), ('zZbu', 7746992289), ('ztsv', 8624279689))), ('ztsv', (('zUvW', 5729), ('zZbu', 26307866809), ('ztsv', 1693898649))))
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

    def test_frame_via_fill_value_loc_a(self) -> None:

        f1 = Frame(np.arange(12).reshape(4, 3), index=tuple('abcd'), columns=tuple('xyz'), name='foo')
        f2 = f1.via_fill_value(-1).loc['c':, ['w', 'x']]
        self.assertEqual(f2.to_pairs(),
                (('w', (('c', -1), ('d', -1))), ('x', (('c', 6), ('d', 9))))
                )

    def test_frame_via_fill_value_loc_b(self) -> None:

        f1 = Frame(np.arange(12).reshape(4, 3), index=tuple('abcd'), columns=tuple('xyz'), name='foo')
        self.assertEqual(f1.via_fill_value(-1).loc['a', 'z'], 2)
        self.assertEqual(f1.via_fill_value(-1).loc['a', 'w'], -1)
        self.assertEqual(f1.name, 'foo')

    def test_frame_via_fill_value_loc_c(self) -> None:

        f1 = Frame(np.arange(12).reshape(4, 3), index=tuple('abcd'), columns=tuple('xyz'), name='foo')
        s1 = f1.via_fill_value(-1).loc['b', ['w', 'y', 'z']]
        self.assertEqual(s1.to_pairs(),
                (('w', -1), ('y', 4), ('z', 5)))
        self.assertEqual(s1.name, 'b')

    def test_frame_via_fill_value_loc_d(self) -> None:

        f1 = Frame(np.arange(12).reshape(4, 3), index=tuple('abcd'), columns=tuple('xyz'))
        s1 = f1.via_fill_value(-1).loc['q', ['w', 'y', 'z']]
        self.assertEqual(s1.to_pairs(),
                (('w', -1), ('y', -1), ('z', -1)))
        self.assertEqual(s1.name, 'q')

    def test_frame_via_fill_value_loc_e1(self) -> None:

        f1 = Frame(np.arange(12).reshape(4, 3), index=tuple('abcd'), columns=tuple('xyz'))
        s1 = f1.via_fill_value(-1)['y']
        self.assertEqual(s1.to_pairs(),
                (('a', 1), ('b', 4), ('c', 7), ('d', 10))
                )

        f2 = f1.via_fill_value(-1)[['y', 'w']]
        self.assertEqual(f2.to_pairs(),
                (('y', (('a', 1), ('b', 4), ('c', 7), ('d', 10))), ('w', (('a', -1), ('b', -1), ('c', -1), ('d', -1))))
                )

    def test_frame_via_fill_value_loc_f(self) -> None:

        f1 = Frame(np.arange(12).reshape(4, 3), index=tuple('abcd'), columns=tuple('xyz'))
        f2 = f1.via_fill_value(-1).loc[['b', 'e'], ['y', 'q']]

        self.assertEqual(f2.to_pairs(),
                (('y', (('b', 4), ('e', -1))), ('q', (('b', -1), ('e', -1))))
                )

    def test_frame_via_fill_value_loc_g(self) -> None:

        f1 = Frame(np.arange(12).reshape(4, 3), index=tuple('abcd'), columns=tuple('xyz'))

        f2 = f1.via_fill_value(-1).loc[['d', 'e']]
        self.assertEqual(f2.to_pairs(),
                (('x', (('d', 9), ('e', -1))), ('y', (('d', 10), ('e', -1))), ('z', (('d', 11), ('e', -1))))
                )
    def test_frame_via_fill_value_loc_h(self) -> None:

        f1 = Frame(np.arange(12).reshape(4, 3), index=tuple('abcd'), columns=tuple('xyz'))
        f2 = f1.via_fill_value(-1).loc[['d', 'e'], 'w']
        self.assertEqual(f2.to_pairs(),
                (('d', -1), ('e', -1))
                )

    #---------------------------------------------------------------------------
    def test_frame_via_fill_value_loc_i(self) -> None:
        f1 = Frame.from_element('a', index=[1, 2, 3], columns=['a', 'b', 'c'])
        f2 = f1.via_fill_value('').loc[[3, 4], ['d', 'b']]
        self.assertEqual(f2.to_pairs(),
            (('d', ((3, ''), (4, ''))), ('b', ((3, 'a'), (4, ''))))
            )

    def test_frame_via_fill_value_loc_j(self) -> None:
        f1 = Frame.from_element('a', index=[1, 2, 3], columns=['a', 'b'])
        f2 = f1.via_fill_value('').loc[[4], ['a', 'b']]
        self.assertEqual(f2.to_pairs(), (('a', ((4, ''),)), ('b', ((4, ''),))))

        f3 = f1.via_fill_value('').loc[[4, 5], ['a', 'b']]
        self.assertEqual(f3.to_pairs(),
                (('a', ((4, ''), (5, ''))), ('b', ((4, ''), (5, ''))))
                )

    def test_frame_via_fill_value_loc_k(self) -> None:
        f1 = Frame.from_element('a', index=[1, 2, 3], columns=['a', 'b'])
        f2 = f1.via_fill_value('').loc[[1, 2], ['d']]
        self.assertEqual(f2.to_pairs(), (('d', ((1, ''), (2, ''))),))

    def test_frame_via_fill_value_loc_l(self) -> None:
        f1 = Frame.from_element('a', index=[1, 2, 3], columns=['a', 'b']
                ).rename(index='y', columns='x')
        f2 = f1.via_fill_value('').loc[[1, 2], ['d', 'a']]
        self.assertEqual(f2.index.name, 'y')
        self.assertEqual(f2.columns.name, 'x')

        f3 = f1.via_fill_value('').loc[3, ['d', 'a']]
        self.assertEqual(f3.index.name, 'x')
        self.assertEqual(f3.to_pairs(), (('d', ''), ('a', 'a')))

        f4 = f1.via_fill_value('').loc[[1, 4], 'd']
        self.assertEqual(f4.index.name, 'y')
        self.assertEqual(f4.to_pairs(), ((1, ''), (4, '')))

    def test_frame_via_fill_value_loc_m(self) -> None:
        f1 = Frame.from_element('a', index=[1, 2, 3], columns=IndexYearMonth(['2024-01', '1954-03'])).rename(index='y', columns='x')
        f2 = f1.via_fill_value('').loc[[1, 2], ['1954-03', '3000-01']]
        self.assertEqual(f2.to_pairs(),
                ((np.datetime64('1954-03'), ((1, 'a'), (2, 'a'))), (np.datetime64('3000-01'), ((1, ''), (2, '')))))

    def test_frame_via_fill_value_loc_n(self) -> None:
        f1 = FrameGO.from_element('a', index=[1, 2, 3], columns=IndexYearMonth(['2024-01', '1954-03'])).rename(index='y', columns='x')
        f2 = f1.via_fill_value('').loc[[1, 2], ['1954-03', '3000-01']]
        self.assertEqual(f2.to_pairs(),
                ((np.datetime64('1954-03'), ((1, 'a'), (2, 'a'))), (np.datetime64('3000-01'), ((1, ''), (2, '')))))
        self.assertEqual(f2.columns.__class__, IndexYearMonthGO)


    def test_frame_via_fill_value_loc_o(self) -> None:

        f1 = Frame.from_element(0, index=IndexHierarchy.from_product(('a', 'b'), (True, False)), columns=IndexHierarchy.from_product(('x', 'y'), (True, False)))
        f2 = f1.via_fill_value(-1).loc[list(f1.index) + [('c', True)], list(f1.columns) + [('z', False)]]
        self.assertIs(f2.columns.__class__, IndexHierarchy)
        self.assertIs(f2.index.__class__, IndexHierarchy)
        self.assertEqual(f2.to_pairs(),
            ((('x', True),
              ((('a', True), 0),
               (('a', False), 0),
               (('b', True), 0),
               (('b', False), 0),
               (('c', True), -1))),
             (('x', False),
              ((('a', True), 0),
               (('a', False), 0),
               (('b', True), 0),
               (('b', False), 0),
               (('c', True), -1))),
             (('y', True),
              ((('a', True), 0),
               (('a', False), 0),
               (('b', True), 0),
               (('b', False), 0),
               (('c', True), -1))),
             (('y', False),
              ((('a', True), 0),
               (('a', False), 0),
               (('b', True), 0),
               (('b', False), 0),
               (('c', True), -1))),
             (('z', False),
              ((('a', True), -1),
               (('a', False), -1),
               (('b', True), -1),
               (('b', False), -1),
               (('c', True), -1))))
            )

    #---------------------------------------------------------------------------
    def test_frame_via_fill_value_getitem_a(self) -> None:
        label = (1, 'a')
        labels = [label]
        f1 = Frame.from_element(0, index=['x'], columns=IndexHierarchy.from_labels(labels))
        s1 = f1.via_fill_value(-1)[label]
        self.assertEqual(s1.name, (1, 'a'))
        self.assertEqual(s1.to_pairs(), (('x', 0),))

        # get item selection with a list of labels fails only with an IndexHierarchy
        f2 = f1.via_fill_value(-1)[labels]
        self.assertEqual(f2.to_pairs(), (((1, 'a'), (('x', 0),)),))
        self.assertIs(f2.columns.__class__, IndexHierarchy)

        f3 = f1.via_fill_value(-1)[labels + [(1, 'b'), (2, 'a')]]
        self.assertIs(f3.columns.__class__, IndexHierarchy)
        self.assertEqual(f3.to_pairs(),
            (((1, 'a'), (('x', 0),)), ((1, 'b'), (('x', -1),)), ((2, 'a'), (('x', -1),))))


    #---------------------------------------------------------------------------
    def test_frame_via_fill_value_hloc_a(self) -> None:

        f1 = Frame.from_element(1, index=IndexHierarchy.from_labels([[1, 2], [3, 4]]), columns=['a', 'b'])

        f2 = f1.via_fill_value(0).loc[HLoc[1, 2]]
        self.assertIs(f2.__class__, Series)
        self.assertEqual(f2.to_pairs(), (('a', 1), ('b', 1)))

        f3 = f1.via_fill_value(0).loc[HLoc[1, :]]
        self.assertEqual(f3.to_pairs(),
            ((np.str_('a'), (((np.int64(1), np.int64(2)), np.int64(1)),)), (np.str_('b'), (((np.int64(1), np.int64(2)), np.int64(1)),)))
            )
        self.assertIs(f3.__class__, Frame)

    def test_frame_via_fill_value_hloc_b(self) -> None:

        f1 = Frame.from_element(1, index=IndexHierarchy.from_labels([[1, 2], [3, 4], [10, 20]]), columns=['a', 'b'])
        f2 = f1.via_fill_value(0).loc[HLoc[[1, 3]], ['a', 'b', 'c']]
        self.assertIs(f2.__class__, Frame)
        self.assertEqual(f2.to_pairs(),
            ((np.str_('a'), (((np.int64(1), np.int64(2)), np.int64(1)), ((np.int64(3), np.int64(4)), np.int64(1)))), (np.str_('b'), (((np.int64(1), np.int64(2)), np.int64(1)), ((np.int64(3), np.int64(4)), np.int64(1)))), (np.str_('c'), (((np.int64(1), np.int64(2)), np.int64(0)), ((np.int64(3), np.int64(4)), np.int64(0))))))
        # import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    import unittest
    unittest.main()
