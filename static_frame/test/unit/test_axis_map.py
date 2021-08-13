

import unittest

import frame_fixtures as ff

from static_frame.test.test_case import TestCase
from static_frame.core.bus import Bus
from static_frame.core.exception import ErrorInitQuilt
from static_frame.core.exception import AxisInvalid

from static_frame.core.axis_map import AxisMap
from static_frame.core.index import Index


class TestUnit(TestCase):


    def test_axis_map_a(self) -> None:

        components = dict(
                x=Index(('a', 'b', 'c')),
                y=Index(('a', 'b', 'c')),
                )

        am = AxisMap.get_axis_series(components) #type: ignore
        self.assertEqual(am.to_pairs(),
                ((('x', 'a'), 'x'), (('x', 'b'), 'x'), (('x', 'c'), 'x'), (('y', 'a'), 'y'), (('y', 'b'), 'y'), (('y', 'c'), 'y')))


    def test_axis_map_b(self) -> None:

        f1 = ff.parse('s(4,4)|v(int,float)').rename('f1')
        f2 = ff.parse('s(4,4)|v(str)').rename('f2')
        f3 = ff.parse('s(4,4)|v(bool)').rename('f3')

        b1 = Bus.from_frames((f1, f2, f3))

        with self.assertRaises(AxisInvalid):
            AxisMap.from_bus(b1, deepcopy_from_bus=False, axis=3, init_exception_cls=ErrorInitQuilt)


if __name__ == '__main__':
    unittest.main()

