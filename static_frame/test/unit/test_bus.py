import unittest
# from io import StringIO

from static_frame.core.frame import Frame
from static_frame.core.bus import Bus
from static_frame.core.series import Series

from static_frame.core.store import StoreZipTSV

from static_frame.test.test_case import TestCase
from static_frame.test.test_case import temp_file

# from static_frame.test.test_case import skip_win
from static_frame.core.exception import ErrorInitBus


class TestUnit(TestCase):

    def test_bus_init_a(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='foo')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='bar')

        b1 = Bus.from_frames((f1, f2))

        self.assertEqual(b1.keys().values.tolist(),
                ['foo', 'bar'])


        with temp_file('.tsv') as fp:
            b1.to_zip_tsv(fp)
            b2 = Bus.from_zip_tsv(fp)

            f3 = b2['bar']
            f4 = b2['foo']
            # import ipdb; ipdb.set_trace()
            zs = StoreZipTSV(fp)
            zs.write(b1.items())

            f3 = zs.read('foo')
            self.assertEqual(
                f3.to_pairs(0),
                (('a', (('x', 1), ('y', 2))), ('b', (('x', 3), ('y', 4))))
            )

    def test_bus_init_b(self) -> None:

        with self.assertRaises(ErrorInitBus):
            Bus(Series([1, 2, 3]))

        with self.assertRaises(ErrorInitBus):
            Bus(Series([3, 4], dtype=object))


if __name__ == '__main__':
    unittest.main()
