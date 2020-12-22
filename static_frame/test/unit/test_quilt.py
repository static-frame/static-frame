
import unittest

import frame_fixtures as ff

from static_frame.test.test_case import TestCase
from static_frame.core.quilt import Quilt
from static_frame.core.quilt import AxisMap
from static_frame.core.index import Index
from static_frame.core.hloc import HLoc
from static_frame.core.display_config import DisplayConfig


class TestUnit(TestCase):

    def test_axis_map_a(self) -> None:

        components = dict(
                x=Index(('a', 'b', 'c')),
                y=Index(('a', 'b', 'c')),
                )

        am = AxisMap.from_tree(components) #type: ignore
        self.assertEqual(am.to_pairs(),
                ((('x', 'a'), 'x'), (('x', 'b'), 'x'), (('x', 'c'), 'x'), (('y', 'a'), 'y'), (('y', 'b'), 'y'), (('y', 'c'), 'y')))

    def test_quilt_display_a(self) -> None:

        dc = DisplayConfig(type_show=False)

        f1 = ff.parse('s(10,4)|v(int)|i(I,str)|c(I,str)').rename('foo')
        q1 = Quilt.from_frame(f1, chunksize=2, retain_bus_labels=False)
        self.assertEqual(
                q1.display(dc).to_rows(),
                f1.display(dc).to_rows())

    def test_quilt_values_a(self) -> None:
        f1 = ff.parse('s(6,4)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=2, retain_bus_labels=False)
        self.assertEqual(q1.values.tolist(),
                [[-88017, 162197, -3648, 129017], [92867, -41157, 91301, 35021], [84967, 5729, 30205, 166924], [13448, -168387, 54020, 122246], [175579, 140627, 129017, 197228], [58768, 66269, 35021, 105269]])


    def test_quilt_nbytes_a(self) -> None:

        dc = DisplayConfig(type_show=False)

        f1 = ff.parse('s(10,4)|v(int)|i(I,str)|c(I,str)').rename('foo')
        q1 = Quilt.from_frame(f1, chunksize=2, retain_bus_labels=False)
        self.assertEqual(q1.nbytes, f1.nbytes)

    #---------------------------------------------------------------------------
    def test_quilt_from_frame_a(self) -> None:

        f1 = ff.parse('s(100,4)|v(int)|i(I,str)|c(I,str)').rename('foo')

        q1 = Quilt.from_frame(f1, chunksize=10, retain_bus_labels=False)

        # import ipdb; ipdb.set_trace()

        self.assertEqual(q1.name, 'foo')
        self.assertEqual(q1.rename('bar').name, 'bar')
        self.assertTrue(repr(q1).startswith('<Quilt: foo'))

        post = AxisMap.from_bus(q1._bus, q1._axis)
        self.assertEqual(len(post), 100)

        s1 = q1['ztsv']
        self.assertEqual(s1.shape, (100,))
        self.assertTrue(s1['zwVN'] == f1.loc['zwVN', 'ztsv'])

        f1 = q1['zUvW':] #type: ignore
        self.assertEqual(f1.shape, (100, 2))
        self.assertEqual(f1.columns.values.tolist(), ['zUvW', 'zkuW'])

        f2 = q1[['zZbu', 'zkuW']]
        self.assertEqual(f2.shape, (100, 2))
        self.assertEqual(f2.columns.values.tolist(), ['zZbu', 'zkuW'])

        f3 = q1.loc['zQuq':, 'zUvW':] #type: ignore
        self.assertEqual(f3.shape, (6, 2))


    def test_quilt_from_frame_b(self) -> None:

        f1 = ff.parse('s(4,100)|v(int)|i(I,str)|c(I,str)')

        q1 = Quilt.from_frame(f1, chunksize=10, axis=1, retain_bus_labels=False)

        post = AxisMap.from_bus(q1._bus, q1._axis)
        self.assertEqual(len(post), 100)

        # import ipdb; ipdb.set_trace()


    #---------------------------------------------------------------------------
    def test_quilt_retain_labels_a(self) -> None:

        dc = DisplayConfig(type_show=False)

        f1 = ff.parse('s(10,4)|v(int)|i(I,str)|c(I,str)').rename('foo')
        q1 = Quilt.from_frame(f1, chunksize=2, retain_bus_labels=False)
        self.assertEqual(q1.index.depth, 1)
        f2 = q1.loc['zkuW':'z2Oo'] #type: ignore
        self.assertEqual(f2.index.depth, 1)
        self.assertEqual(f2.to_pairs(0),
                (('zZbu', (('zkuW', 13448), ('zmVj', 175579), ('z2Oo', 58768))), ('ztsv', (('zkuW', -168387), ('zmVj', 140627), ('z2Oo', 66269))), ('zUvW', (('zkuW', 54020), ('zmVj', 129017), ('z2Oo', 35021))), ('zkuW', (('zkuW', 122246), ('zmVj', 197228), ('z2Oo', 105269))))
                )

        q2 = Quilt.from_frame(f1, chunksize=2, retain_bus_labels=True)
        self.assertEqual(q2.index.depth, 2)
        # import ipdb; ipdb.set_trace()
        f3 = q2.loc[HLoc['zUvW':'z5l6']] #type: ignore
        self.assertEqual(f3.index.depth, 2)
        self.assertEqual(f3.to_pairs(0),
                (('zZbu', ((('zUvW', 'zUvW'), 84967), (('zUvW', 'zkuW'), 13448), (('zmVj', 'zmVj'), 175579), (('zmVj', 'z2Oo'), 58768), (('z5l6', 'z5l6'), 146284), (('z5l6', 'zCE3'), 170440))), ('ztsv', ((('zUvW', 'zUvW'), 5729), (('zUvW', 'zkuW'), -168387), (('zmVj', 'zmVj'), 140627), (('zmVj', 'z2Oo'), 66269), (('z5l6', 'z5l6'), -171231), (('z5l6', 'zCE3'), -38997))), ('zUvW', ((('zUvW', 'zUvW'), 30205), (('zUvW', 'zkuW'), 54020), (('zmVj', 'zmVj'), 129017), (('zmVj', 'z2Oo'), 35021), (('z5l6', 'z5l6'), 166924), (('z5l6', 'zCE3'), 122246))), ('zkuW', ((('zUvW', 'zUvW'), 166924), (('zUvW', 'zkuW'), 122246), (('zmVj', 'zmVj'), 197228), (('zmVj', 'z2Oo'), 105269), (('z5l6', 'z5l6'), 119909), (('z5l6', 'zCE3'), 194224))))
                )


        # import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    unittest.main()


