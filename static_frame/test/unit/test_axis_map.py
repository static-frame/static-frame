

import unittest

import frame_fixtures as ff

from static_frame.test.test_case import TestCase
from static_frame.core.bus import Bus
from static_frame.core.exception import ErrorInitQuilt
from static_frame.core.exception import AxisInvalid
from static_frame.core.exception import ErrorInitYarn

from static_frame.core.axis_map import bus_to_hierarchy
from static_frame.core.axis_map import buses_to_hierarchy

# from static_frame.core.index import Index


class TestUnit(TestCase):



    def test_axis_map_b(self) -> None:

        f1 = ff.parse('s(4,4)|v(int,float)').rename('f1')
        f2 = ff.parse('s(4,4)|v(str)').rename('f2')
        f3 = ff.parse('s(4,4)|v(bool)').rename('f3')

        b1 = Bus.from_frames((f1, f2, f3))

        with self.assertRaises(AxisInvalid):
            bus_to_hierarchy(b1, deepcopy_from_bus=False, axis=3, init_exception_cls=ErrorInitQuilt)


    def test_index_map_a(self) -> None:

        f1 = ff.parse('s(4,4)|v(int,float)').rename('f1')
        f2 = ff.parse('s(4,4)|v(str)').rename('f2')
        f3 = ff.parse('s(4,4)|v(bool)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3), name='a')

        f4 = ff.parse('s(4,4)|v(int,float)').rename('f4')
        f5 = ff.parse('s(4,4)|v(str)').rename('f5')
        b2 = Bus.from_frames((f4, f5), name='b')

        post = buses_to_hierarchy((b1, b2), (b1.name, b2.name), deepcopy_from_bus=False, init_exception_cls=ErrorInitYarn)

        self.assertEqual(post.values.tolist(),
                [['a', 'f1'], ['a', 'f2'], ['a', 'f3'], ['b', 'f4'], ['b', 'f5']])


if __name__ == '__main__':
    unittest.main()

