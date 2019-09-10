import unittest

from static_frame.core.frame import Frame
from static_frame.core.bus import Bus

from static_frame.test.test_case import TestCase
# from static_frame.test.test_case import skip_win
# from static_frame.core.exception import ErrorInitFrame


class TestUnit(TestCase):

    def test_bus_init_a(self):

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='foo')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='bar')

        b1 = Bus.from_frames(f1, f2)

        self.assertEqual(b1.keys().values.tolist(),
                ['foo', 'bar'])


if __name__ == '__main__':
    unittest.main()
