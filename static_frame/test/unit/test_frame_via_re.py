import unittest
import re

import frame_fixtures as ff
# import numpy as np

# from static_frame import Frame
# from static_frame import Series
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    def test_frame_via_re_a(self) -> None:
        f1 = ff.parse('s(3,3)|c(I,str)|i(I,str)|v(int)')

        self.assertEqual(
            f1.via_re('8[68]').search().to_pairs(),
            (('zZbu', (('zZbu', True), ('ztsv', True), ('zUvW', False))), ('ztsv', (('zZbu', False), ('ztsv', False), ('zUvW', False))), ('zUvW', (('zZbu', False), ('ztsv', False), ('zUvW', False))))
            )

        self.assertEqual(
            f1.via_re('9').search(endpos=2).to_pairs(),
            (('zZbu', (('zZbu', False), ('ztsv', True), ('zUvW', False))), ('ztsv', (('zZbu', False), ('ztsv', False), ('zUvW', False))), ('zUvW', (('zZbu', False), ('ztsv', True), ('zUvW', False))))
            )

    def test_frame_via_re_b(self) -> None:
        f1 = ff.parse('s(2,5)|c(I,str)|i(I,str)|v(int,bool,bool,float,str)')

        self.assertEqual(
            f1.via_re('[a.-]').search().to_pairs(),
                (('zZbu', (('zZbu', True), ('ztsv', False))), ('ztsv', (('zZbu', True), ('ztsv', True))), ('zUvW', (('zZbu', False), ('ztsv', True))), ('zkuW', (('zZbu', True), ('ztsv', True))), ('zmVj', (('zZbu', False), ('ztsv', False))))
            )

        self.assertEqual(
            f1.via_re('f', re.I).match(endpos=2).to_pairs(),
            (('zZbu', (('zZbu', False), ('ztsv', False))), ('ztsv', (('zZbu', True), ('ztsv', True))), ('zUvW', (('zZbu', False), ('ztsv', True))), ('zkuW', (('zZbu', False), ('ztsv', False))), ('zmVj', (('zZbu', False), ('ztsv', False))))
            )

        self.assertEqual(
            f1.via_re('z[5h][5h]i', re.I).fullmatch().to_pairs(),
            (('zZbu', (('zZbu', False), ('ztsv', False))), ('ztsv', (('zZbu', False), ('ztsv', False))), ('zUvW', (('zZbu', False), ('ztsv', False))), ('zkuW', (('zZbu', False), ('ztsv', False))), ('zmVj', (('zZbu', False), ('ztsv', True)))))

    def test_frame_via_re_c(self) -> None:
        f1 = ff.parse('s(2,5)|c(I,str)|i(I,str)|v(int,bool,bool,float,str)')

        self.assertEqual(f1.via_re('[a.r]').split().to_pairs(),
            (('zZbu', (('zZbu', ('-88017',)), ('ztsv', ('92867',)))), ('ztsv', (('zZbu', ('F', 'lse')), ('ztsv', ('F', 'lse')))), ('zUvW', (('zZbu', ('T', 'ue')), ('ztsv', ('F', 'lse')))), ('zkuW', (('zZbu', ('1080', '4')), ('ztsv', ('2580', '34')))), ('zmVj', (('zZbu', ('zDVQ',)), ('ztsv', ('z5hI',)))))
            )

    def test_frame_via_re_d(self) -> None:
        f1 = ff.parse('s(2,5)|c(I,str)|i(I,str)|v(int,bool,bool,float,str)')

        self.assertEqual(
            f1.via_re('[a0z]').findall().to_pairs(),
            (('zZbu', (('zZbu', ('0',)), ('ztsv', ()))), ('ztsv', (('zZbu', ('a',)), ('ztsv', ('a',)))), ('zUvW', (('zZbu', ()), ('ztsv', ('a',)))), ('zkuW', (('zZbu', ('0', '0')), ('ztsv', ('0',)))), ('zmVj', (('zZbu', ('z',)), ('ztsv', ('z',)))))
            )

        self.assertEqual(
            f1.via_re('[a0z]').findall(endpos=1).to_pairs(),
            (('zZbu', (('zZbu', ()), ('ztsv', ()))), ('ztsv', (('zZbu', ()), ('ztsv', ()))), ('zUvW', (('zZbu', ()), ('ztsv', ()))), ('zkuW', (('zZbu', ()), ('ztsv', ()))), ('zmVj', (('zZbu', ('z',)), ('ztsv', ('z',)))))
            )

    def test_frame_via_re_e(self) -> None:
        f1 = ff.parse('s(2,5)|c(I,str)|i(I,str)|v(int,bool,bool,float,str)')

        self.assertEqual(f1[f1.columns.via_re('[uU][vW]').search()].to_pairs(),
            (('zUvW', (('zZbu', True), ('ztsv', False))), ('zkuW', (('zZbu', 1080.4), ('ztsv', 2580.34))))
            )

    def test_frame_via_re_sub_a(self) -> None:
        f1 = ff.parse('s(2,5)|c(I,str)|i(I,str)|v(int,bool,bool,float,str)')

        self.assertEqual(
            f1.via_re('[za0.2]').sub('--').to_pairs(),
            (('zZbu', (('zZbu', '-88--17'), ('ztsv', '9--867'))), ('ztsv', (('zZbu', 'F--lse'), ('ztsv', 'F--lse'))), ('zUvW', (('zZbu', 'True'), ('ztsv', 'F--lse'))), ('zkuW', (('zZbu', '1--8----4'), ('ztsv', '--58----34'))), ('zmVj', (('zZbu', '--DVQ'), ('ztsv', '--5hI'))))
            )

        self.assertEqual(f1.via_re('[za0.2]').sub('--', count=1).to_pairs(),
            (('zZbu', (('zZbu', '-88--17'), ('ztsv', '9--867'))), ('ztsv', (('zZbu', 'F--lse'), ('ztsv', 'F--lse'))), ('zUvW', (('zZbu', 'True'), ('ztsv', 'F--lse'))), ('zkuW', (('zZbu', '1--80.4'), ('ztsv', '--580.34'))), ('zmVj', (('zZbu', '--DVQ'), ('ztsv', '--5hI'))))
            )

    def test_frame_via_re_subn_a(self) -> None:
        f1 = ff.parse('s(2,5)|c(I,str)|i(I,str)|v(int,bool,bool,float,str)')

        self.assertEqual(
            f1.via_re('[e8]').subn('*').to_pairs(),
            (('zZbu', (('zZbu', ('-**017', 2)), ('ztsv', ('92*67', 1)))), ('ztsv', (('zZbu', ('Fals*', 1)), ('ztsv', ('Fals*', 1)))), ('zUvW', (('zZbu', ('Tru*', 1)), ('ztsv', ('Fals*', 1)))), ('zkuW', (('zZbu', ('10*0.4', 1)), ('ztsv', ('25*0.34', 1)))), ('zmVj', (('zZbu', ('zDVQ', 0)), ('ztsv', ('z5hI', 0)))))
            )


if __name__ == '__main__':
    unittest.main()

