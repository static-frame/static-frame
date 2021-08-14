

import unittest

import frame_fixtures as ff

from static_frame.test.test_case import TestCase
from static_frame.core.bus import Bus
# from static_frame.core.exception import ErrorInitQuilt
# from static_frame.core.exception import AxisInvalid
# from static_frame.core.exception import ErrorInitYarn

# from static_frame.core.axis_map import AxisMap
# from static_frame.core.index import Index
# from static_frame.core.axis_map import IndexMap

from static_frame.core.yarn import Yarn

class TestUnit(TestCase):


    def test_yarn_from_buses_a(self) -> None:

        f1 = ff.parse('s(4,4)|v(int,float)').rename('f1')
        f2 = ff.parse('s(4,4)|v(str)').rename('f2')
        f3 = ff.parse('s(4,4)|v(bool)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3), name='a')

        f4 = ff.parse('s(4,4)|v(int,float)').rename('f4')
        f5 = ff.parse('s(4,4)|v(str)').rename('f5')
        b2 = Bus.from_frames((f4, f5), name='b')


        y1 = Yarn.from_buses((b1, b2), retain_labels=True)
        self.assertEqual(len(y1), 5)
        self.assertEqual(y1.index.shape, (5, 2))

        y2 = Yarn.from_buses((b1, b2), retain_labels=False)
        self.assertEqual(len(y2), 5)
        self.assertEqual(y2.index.shape, (5,))





if __name__ == '__main__':
    unittest.main()

