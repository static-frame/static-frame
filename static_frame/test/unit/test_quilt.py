
import unittest

import frame_fixtures as ff

from static_frame.test.test_case import TestCase
from static_frame.core.quilt import Quilt
from static_frame.core.quilt import AxisMap

class TestUnit(TestCase):

    def test_quilt_from_frame_a(self) -> None:

        f1 = ff.parse('s(100,4)|v(int)|i(I,str)|c(I,str)').rename('foo')

        q1 = Quilt.from_frame(f1, chunksize=10)

        self.assertEqual(q1.name, 'foo')
        self.assertEqual(q1.rename('bar').name, 'bar')
        self.assertTrue(repr(q1).startswith('<Quilt: foo'))

        post = AxisMap.from_bus(q1._bus, q1._axis)
        self.assertEqual(len(post), 100)

        s1 = q1['ztsv']
        self.assertEqual(s1.shape, (100,))
        self.assertTrue(s1['zwVN'] == f1.loc['zwVN', 'ztsv'])

        f1 = q1['zUvW':]
        self.assertEqual(f1.shape, (100, 2))
        self.assertEqual(f1.columns.values.tolist(), ['zUvW', 'zkuW'])

        f2 = q1[['zZbu', 'zkuW']]
        self.assertEqual(f2.shape, (100, 2))
        self.assertEqual(f2.columns.values.tolist(), ['zZbu', 'zkuW'])

        # this does not trim upper region
        f3 = q1.loc['zQuq':, 'zUvW':]
        # import ipdb; ipdb.set_trace()

    def test_quilt_from_frame_b(self) -> None:

        f1 = ff.parse('s(4,100)|v(int)|i(I,str)|c(I,str)')

        q1 = Quilt.from_frame(f1, chunksize=10, axis=1)

        post = AxisMap.from_bus(q1._bus, q1._axis)
        self.assertEqual(len(post), 100)

        # import ipdb; ipdb.set_trace()



if __name__ == '__main__':
    unittest.main()


