import unittest
# from io import StringIO

from static_frame.core.frame import Frame
from static_frame.core.bus import Bus
from static_frame.core.bus import StoreZip

from static_frame.test.test_case import TestCase
# from static_frame.test.test_case import skip_win
# from static_frame.core.exception import ErrorInitFrame


class TestUnit(TestCase):

    @unittest.skip('temp')
    def test_bus_init_a(self) -> None:

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

        b1.to_zip('/tmp/a_test.zip')

        b2 = Bus.from_zip('/tmp/a_test.zip')

        f3 = b2['bar']
        f4 = b2['foo']
        # import ipdb; ipdb.set_trace()

        # TODO: how to use StringIO
        zs = StoreZip('/tmp/test.zip')
        zs.write(b1.items())

        f3 = zs.read('foo')
        self.assertEqual(
            f3.to_pairs(0),
            (('a', (('x', 1), ('y', 2))), ('b', (('x', 3), ('y', 4))))
        )

if __name__ == '__main__':
    unittest.main()
