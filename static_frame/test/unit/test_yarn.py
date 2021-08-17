

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
from static_frame.test.test_case import temp_file


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
        self.assertEqual(y1.shape, (5,))
        self.assertEqual(y1.size, 5)
        self.assertEqual(y1.dtype, object)
        self.assertEqual(y1.ndim, 1)

        y1[('a', 'f2'):]

        y2 = Yarn.from_buses((b1, b2), retain_labels=False)
        self.assertEqual(len(y2), 5)
        self.assertEqual(y2.index.shape, (5,))
        self.assertEqual(y1.shape, (5,))
        self.assertEqual(y1.size, 5)
        self.assertEqual(y1.dtype, object)
        self.assertEqual(y1.ndim, 1)


    def test_yarn_from_buses_b(self) -> None:

        f1 = ff.parse('s(4,4)|v(int,float)').rename('f1')
        f2 = ff.parse('s(4,4)|v(str)').rename('f2')
        f3 = ff.parse('s(4,4)|v(bool)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3), name='a')

        f4 = ff.parse('s(4,4)|v(int,float)').rename('f4')
        f5 = ff.parse('s(4,4)|v(str)').rename('f5')
        b2 = Bus.from_frames((f4, f5), name='b')

        f6 = ff.parse('s(2,4)|v(int,float)').rename('f6')
        f7 = ff.parse('s(4,2)|v(str)').rename('f7')
        b3 = Bus.from_frames((f6, f7), name='c')

        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=True)


        # import ipdb; ipdb.set_trace()

    def test_yarn_max_persist(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')


        b1 = Bus.from_frames((f1, f2, f3))
        b2 = Bus.from_frames((f4, f5, f6))

        with temp_file('.zip') as fp1, temp_file('.zip') as fp2:
            b1.to_zip_pickle(fp1)
            b2.to_zip_pickle(fp2)


            bus_a = Bus.from_zip_pickle(fp1, max_persist=1).rename('a')
            bus_b = Bus.from_zip_pickle(fp2, max_persist=1).rename('b')

            y1 = Yarn.from_buses((bus_a, bus_b), retain_labels=False)
            self.assertEqual(y1.nbytes, 0)
            self.assertEqual(y1.status['loaded'].sum(), 0)

            import ipdb; ipdb.set_trace()
            self.assertEqual(y1['f2'].shape, (4, 5))
            self.assertEqual(y1['f6'].shape, (6, 4))
            self.assertEqual(y1.nbytes, 352)
            self.assertEqual(y1.status['loaded'].sum(), 2)

            import ipdb; ipdb.set_trace()

            pass



if __name__ == '__main__':
    unittest.main()

