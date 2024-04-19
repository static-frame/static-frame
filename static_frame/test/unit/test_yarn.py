from __future__ import annotations

import datetime
from hashlib import sha256

import frame_fixtures as ff
import numpy as np

from static_frame import HLoc
from static_frame import ILoc
from static_frame.core.bus import Bus
from static_frame.core.display_config import DisplayConfig
from static_frame.core.exception import ErrorInitYarn
from static_frame.core.exception import RelabelInvalid
from static_frame.core.frame import Frame
from static_frame.core.index import Index
from static_frame.core.index_auto import IndexAutoFactory
from static_frame.core.index_datetime import IndexDate
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.series import Series
from static_frame.core.yarn import Yarn
from static_frame.test.test_case import TestCase
from static_frame.test.test_case import temp_file


class TestUnit(TestCase):

    #---------------------------------------------------------------------------

    def test_yarn_init_a(self) -> None:

        with self.assertRaises(ErrorInitYarn):
            Yarn(np.array([3, 4]))

    def test_yarn_init_b(self) -> None:

        f1 = ff.parse('s(4,4)|v(int,float)').rename('f1')
        f2 = ff.parse('s(4,4)|v(str)').rename('f2')
        f3 = ff.parse('s(4,4)|v(bool)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3))

        f4 = ff.parse('s(4,4)|v(int,float)').rename('f4')
        f5 = ff.parse('s(4,4)|v(str)').rename('f5')
        b2 = Bus.from_frames((f4, f5))

        y1 = Yarn((b1, b2), index=tuple('abcde'))
        self.assertEqual(y1.index.values.tolist(), list('abcde'))
        self.assertEqual(y1[['a', 'c', 'e']].shape, (3,))

        y2 = Yarn((b1, b2))
        self.assertEqual(y2.index.values.tolist(), list(range(5)))
        self.assertEqual(y2[2:].shape, (3,))

        y3 = Yarn((b2,), index=('2021-01-01', '2021-02-15'), index_constructor=IndexDate)
        self.assertEqual(y3.index.__class__, IndexDate)
        self.assertEqual(y3.index.values.tolist(), [datetime.date(2021, 1, 1), datetime.date(2021, 2, 15)])

        with self.assertRaises(ErrorInitYarn):
            y4 = Yarn((b2,), index=range(5))

    def test_yarn_init_c(self) -> None:

        with self.assertRaises(ErrorInitYarn):
            Yarn((ff.parse('s(2,2)'),))

        with self.assertRaises(ErrorInitYarn):
            Yarn(Series((ff.parse('s(2,2)'),), dtype=object))

    def test_yarn_init_d(self) -> None:

        with self.assertRaises(ErrorInitYarn):
            Yarn(Series(np.array((False, True))))

    def test_yarn_init_e1(self) -> None:
        b1 = Bus.from_frames((ff.parse("s(3,3)").rename("f1"),)).rename('x')
        b2 = Bus.from_frames((ff.parse("s(3,4)").rename("f2"),)).rename('x')
        b3 = Bus.from_frames((ff.parse("s(3,2)").rename("f3"),)).rename('x')
        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=True)
        self.assertEqual(y1.index.values.tolist(),
            [['x', 'f1'], ['x', 'f2'], ['x', 'f3']]
            )
        self.assertEqual(y1[('x', 'f3')].shape, (3, 2))

        y2 = y1[('x', 'f2'):]
        self.assertEqual(y2.index.values.tolist(),
            [['x', 'f2'], ['x', 'f3']]
            )


    def test_yarn_init_e2(self) -> None:
        b1 = Bus.from_frames((ff.parse("s(3,3)").rename("f1"),))
        b2 = Bus.from_frames((ff.parse("s(3,4)").rename("f2"),))
        b3 = Bus.from_frames((ff.parse("s(3,2)").rename("f3"),))
        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=False)
        self.assertEqual(y1.index.values.tolist(), ['f1', 'f2', 'f3'])
        self.assertEqual(y1['f3'].shape, (3, 2))

    def test_yarn_init_f(self) -> None:
        b1 = Bus.from_frames((ff.parse("s(3,3)").rename("f1"),))
        b2 = Bus.from_frames((ff.parse("s(3,4)").rename("f2"),))

        with self.assertRaises(ErrorInitYarn):
            y1 = Yarn((b1, b2), indexer=np.array([2, 1, 4]))




    #---------------------------------------------------------------------------

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

        y3 = y1[('a', 'f2'):] #type: ignore
        self.assertEqual(y3.shape, (4,))

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
        self.assertEqual(y1.shape, (7,))
        self.assertEqual(len(y1), 7)

    #---------------------------------------------------------------------------

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

            self.assertEqual(y1['f2'].shape, (4, 5))
            self.assertEqual(y1['f6'].shape, (6, 4))
            self.assertEqual(y1.nbytes, 352)
            self.assertEqual(y1.status['loaded'].sum(), 2)

            self.assertEqual(y1.shapes.to_pairs(),
                    (('f1', None), ('f2', (4, 5)), ('f3', None), ('f4', None), ('f5', None), ('f6', (6, 4)))
                    )

            self.assertEqual(y1.mloc.isna().sum(), 4)
            self.assertEqual((y1.dtypes == float).sum().sum(), 9)

    #---------------------------------------------------------------------------

    def test_yarn_from_concat_a(self) -> None:
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

        y1 = Yarn.from_concat((Yarn.from_buses((b1,), retain_labels=True), Yarn.from_buses((b2, b3), retain_labels=True)))
        self.assertEqual(y1.shape, (7,))
        self.assertEqual(y1.index.values.tolist(),
                [['a', 'f1'], ['a', 'f2'], ['a', 'f3'], ['b', 'f4'], ['b', 'f5'], ['c', 'f6'], ['c', 'f7']]
                )

    def test_yarn_from_concat_b(self) -> None:
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

            y1 = Yarn.from_concat((Yarn.from_buses((bus_a,), retain_labels=True), Yarn.from_buses((bus_b,), retain_labels=True)))

            y2 = Yarn.from_concat((y1, y1), index=IndexAutoFactory)

            self.assertEqual(y2[3].shape, (2, 8))
            self.assertEqual(y2[0].shape, (4, 2))
            self.assertEqual(y2[5].shape, (6, 4))

            y3 = y2.iloc[4:]
            self.assertEqual(y3.shape, (8,))

    def test_yarn_from_concat_c(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')

        with self.assertRaises(NotImplementedError):
            Yarn.from_concat((f1, f2))

    def test_yarn_from_concat_d(self) -> None:
        f1 = ff.parse('s(4,4)|v(int,float)').rename('f1')
        f2 = ff.parse('s(4,4)|v(str)').rename('f2')
        f3 = ff.parse('s(4,4)|v(bool)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3))

        f4 = ff.parse('s(4,4)|v(int,float)').rename('f4')
        f5 = ff.parse('s(4,4)|v(str)').rename('f5')
        b2 = Bus.from_frames((f4, f5))

        f6 = ff.parse('s(2,4)|v(int,float)').rename('f6')
        f7 = ff.parse('s(4,2)|v(str)').rename('f7')
        b3 = Bus.from_frames((f6, f7))


        y1 = Yarn.from_buses((b1,), retain_labels=False)
        y2 = Yarn.from_buses((b2, b3), retain_labels=False)

        y3 = y1[['f2']]
        y4 = y2[['f7']]

        y5 = Yarn.from_concat((y3, y4))
        self.assertEqual(y5.shape, (2,))
        self.assertEqual(y5.index.values.tolist(), ['f2', 'f7'])
        self.assertEqual(len(y5._values), 3) # all bus are retained
        self.assertEqual(len(y5._hierarchy), 7) # concat of all found


    def test_yarn_from_concat_e(self) -> None:
        f1 = ff.parse('s(4,4)|v(int,float)').rename('f1')
        f2 = ff.parse('s(4,4)|v(str)').rename('f2')
        f3 = ff.parse('s(4,4)|v(bool)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3))

        f4 = ff.parse('s(4,4)|v(int,float)').rename('f4')
        f5 = ff.parse('s(4,4)|v(str)').rename('f5')
        b2 = Bus.from_frames((f4, f5))

        f6 = ff.parse('s(2,4)|v(int,float)').rename('f6')
        f7 = ff.parse('s(4,2)|v(str)').rename('f7')
        b3 = Bus.from_frames((f6, f7))

        f8 = ff.parse('s(2,4)|v(int,float)').rename('f8')
        f9 = ff.parse('s(4,2)|v(str)').rename('f9')
        f10 = ff.parse('s(4,2)|v(str)').rename('f10')
        b4 = Bus.from_frames((f8, f9, f10))

        f11 = ff.parse('s(4,4)|v(int,float)').rename('f11')
        f12 = ff.parse('s(4,4)|v(str)').rename('f12')
        b5 = Bus.from_frames((f11, f12))

        f13 = ff.parse('s(2,4)|v(int,float)').rename('f13')
        f14 = ff.parse('s(4,2)|v(str)').rename('f14')
        b6 = Bus.from_frames((f13, f14))

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)
        y2 = Yarn.from_buses((b3, b4), retain_labels=False)
        y3 = Yarn.from_buses((b5, b6), retain_labels=False)

        yc1 = y1[['f2', 'f4', 'f1']]
        yc2 = y2[['f7', 'f6']]
        yc3 = y3[['f14', 'f12']]

        yp = Yarn.from_concat((yc1, yc2, yc3))

        self.assertEqual(yp.shape, (7,))
        self.assertEqual(yp.index.values.tolist(), ['f2', 'f4', 'f1', 'f7', 'f6', 'f14', 'f12'])
        self.assertEqual(len(yp._values), 6) # all bus are retained
        self.assertEqual(len(yp._hierarchy), 14) # concat of all found


    def test_yarn_from_concat_f(self) -> None:
        f1 = ff.parse('s(4,4)|v(int,float)').rename('f1')
        f2 = ff.parse('s(4,4)|v(str)').rename('f2')
        f3 = ff.parse('s(4,4)|v(bool)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3))

        f4 = ff.parse('s(4,4)|v(int,float)').rename('f4')
        f5 = ff.parse('s(4,4)|v(str)').rename('f5')
        b2 = Bus.from_frames((f4, f5))

        f6 = ff.parse('s(2,4)|v(int,float)').rename('f6')
        f7 = ff.parse('s(4,2)|v(str)').rename('f7')
        b3 = Bus.from_frames((f6, f7))

        f8 = ff.parse('s(2,4)|v(int,float)').rename('f8')
        f9 = ff.parse('s(4,2)|v(str)').rename('f9')
        f10 = ff.parse('s(4,2)|v(str)').rename('f10')
        b4 = Bus.from_frames((f8, f9, f10))

        f11 = ff.parse('s(4,4)|v(int,float)').rename('f11')
        f12 = ff.parse('s(4,4)|v(str)').rename('f12')
        b5 = Bus.from_frames((f11, f12))

        f13 = ff.parse('s(2,4)|v(int,float)').rename('f13')
        f14 = ff.parse('s(4,2)|v(str)').rename('f14')
        b6 = Bus.from_frames((f13, f14))

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)
        y2 = Yarn.from_buses((b3, b4), retain_labels=False)
        y3 = Yarn.from_buses((b5, b6), retain_labels=False)

        yc1 = y1['f3': 'f5']
        yc2 = y2['f9':]
        yc3 = y3[['f14', 'f11']]

        yp = Yarn.from_concat((yc1, yc2, yc3))

        self.assertEqual(yp.shape, (7,))
        self.assertEqual(yp.index.values.tolist(), ['f3', 'f4', 'f5', 'f9', 'f10', 'f14', 'f11'])
        self.assertEqual(len(yp._values), 6) # all bus are retained
        self.assertEqual(len(yp._hierarchy), 14) # concat of all found

    #---------------------------------------------------------------------------

    def test_yarn_reversed_a(self) -> None:
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

        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=False)
        self.assertEqual(tuple(reversed(y1)), ('f7', 'f6', 'f5', 'f4', 'f3', 'f2', 'f1'))

    #---------------------------------------------------------------------------

    def test_yarn_size_a(self) -> None:
        f1 = ff.parse('s(4,4)|v(int,float)').rename('f1')
        f2 = ff.parse('s(4,4)|v(str)').rename('f2')
        f3 = ff.parse('s(4,4)|v(bool)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3), name='a')

        f4 = ff.parse('s(4,4)|v(int,float)').rename('f4')
        f5 = ff.parse('s(4,4)|v(str)').rename('f5')
        b2 = Bus.from_frames((f4, f5), name='b')

        y1 = Yarn.from_buses((b1, b2), retain_labels=False, name='foo')
        self.assertEqual(y1.size, 5)

    #---------------------------------------------------------------------------

    def test_yarn_index_a(self) -> None:
        f1 = ff.parse('s(4,4)|v(int,float)').rename('f1')
        f2 = ff.parse('s(4,4)|v(str)').rename('f2')
        f3 = ff.parse('s(4,4)|v(bool)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3), name='a')

        f4 = ff.parse('s(4,4)|v(int,float)').rename('f4')
        f5 = ff.parse('s(4,4)|v(str)').rename('f5')
        b2 = Bus.from_frames((f4, f5), name='b')

        y1 = Yarn.from_buses((b1, b2), retain_labels=False, name='foo')
        self.assertEqual(y1.index.values.tolist(), ['f1', 'f2', 'f3', 'f4', 'f5'])

    #---------------------------------------------------------------------------

    def test_yarn_rename_a(self) -> None:
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

        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=False, name='foo')
        self.assertEqual(y1.name, 'foo')
        y2 = y1.rename('bar')
        self.assertEqual(y2.name, 'bar')

    #---------------------------------------------------------------------------

    def test_yarn_loc_a(self) -> None:
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

        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=False, name='foo')
        self.assertEqual(y1.loc['f4'].shape, (4, 4))
        self.assertEqual(y1.loc['f4':].shape, (4,)) #type: ignore
        self.assertEqual(y1.loc[['f2', 'f7']].shape, (2,))
        self.assertEqual(y1.loc[y1.index.via_str.startswith('f3')].shape, (1,))

    def test_yarn_loc_b(self) -> None:
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

        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=False, name='foo')
        y2 = y1.loc[y1.index.via_re('[26]').search()]
        self.assertEqual(y2.index.values.tolist(), ['f2', 'f6'])
        self.assertEqual(y2.shapes.to_pairs(),
                (('f2', (4, 4)), ('f6', (2, 4))))
        self.assertEqual(y2.name, 'foo')

    def test_yarn_loc_c(self) -> None:
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

        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=True, name='foo')
        y2 = y1.loc[[('a', 'f3'), ('b', 'f5'), ('c', 'f6')]]
        self.assertEqual(y2.shapes.to_pairs(),
                ((('a', 'f3'), (4, 4)), (('b', 'f5'), (4, 4)), (('c', 'f6'), (2, 4))))

    def test_yarn_loc_d(self) -> None:
        f1 = ff.parse('s(4,4)|v(int,float)').rename('f1')
        f2 = ff.parse('s(4,3)|v(str)').rename('f2')
        f3 = ff.parse('s(4,5)|v(bool)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3), name='a')

        f4 = ff.parse('s(3,4)|v(int,float)').rename('f4')
        f5 = ff.parse('s(3,2)|v(str)').rename('f5')
        b2 = Bus.from_frames((f4, f5), name='b')

        f6 = ff.parse('s(2,4)|v(int,float)').rename('f6')
        f7 = ff.parse('s(2,8)|v(str)').rename('f7')
        b3 = Bus.from_frames((f6, f7), name='c')

        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=False, name='foo')
        y2 = y1.loc[['f7', 'f3']]
        self.assertEqual(y2.shapes.to_pairs(),
                (('f7', (2, 8)), ('f3', (4, 5)))
                )

        y3 = y1.loc[['f1', 'f7', 'f3']]
        self.assertEqual(y3.shapes.to_pairs(),
                (('f1', (4, 4)), ('f7', (2, 8)), ('f3', (4, 5)))
                )

    def test_yarn_loc_e(self) -> None:
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

        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=False, name='foo')
        y2 = y1['f2':'f6'] #type: ignore
        self.assertEqual(y2.shapes.to_pairs(), #type: ignore
                (('f2', (4, 4)), ('f3', (4, 4)), ('f4', (4, 4)), ('f5', (4, 4)), ('f6', (2, 4))))
        self.assertEqual(y2['f5'].to_pairs(), #type: ignore
                ((0, ((0, 'zjZQ'), (1, 'zO5l'), (2, 'zEdH'), (3, 'zB7E'))), (1, ((0, 'zaji'), (1, 'zJnC'), (2, 'zDdR'), (3, 'zuVU'))), (2, ((0, 'ztsv'), (1, 'zUvW'), (2, 'zkuW'), (3, 'zmVj'))), (3, ((0, 'z2Oo'), (1, 'z5l6'), (2, 'zCE3'), (3, 'zr4u')))))

    def test_yarn_loc_f(self) -> None:
        f1 = ff.parse('s(4,4)|v(int,float)').rename('f1')
        f2 = ff.parse('s(3,4)|v(str)').rename('f2')
        f3 = ff.parse('s(4,5)|v(bool)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3), name='a')

        f4 = ff.parse('s(4,8)|v(int,float)').rename('f4')
        f5 = ff.parse('s(2,3)|v(str)').rename('f5')
        b2 = Bus.from_frames((f4, f5), name='b')

        f6 = ff.parse('s(2,4)|v(int,float)').rename('f6')
        f7 = ff.parse('s(4,2)|v(str)').rename('f7')
        b3 = Bus.from_frames((f6, f7), name='c')

        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=True, name='foo')
        y2 = y1.loc[[ILoc[2], ('c', 'f6')]]
        self.assertEqual(y2.shapes.to_pairs(),
                ((('a', 'f3'), (4, 5)), (('c', 'f6'), (2, 4))))

        y3 = y1.loc[HLoc[['a', 'c']]]
        self.assertEqual(y3.shapes.to_pairs(),
                ((('a', 'f1'), (4, 4)), (('a', 'f2'), (3, 4)), (('a', 'f3'), (4, 5)), (('c', 'f6'), (2, 4)), (('c', 'f7'), (4, 2))))

    def test_yarn_loc_g(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')

        b1 = Bus.from_frames((f1, f2))
        b2 = Bus.from_frames((f3, f4))

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)

        y2 = y1[['f3', 'f1', 'f4']]
        self.assertEqual(y2.index.values.tolist(), ['f3', 'f1', 'f4'])
        self.assertEqual(y2.index.values.tolist(), [f.name for f in y2.values])

        y3 = y2[['f1', 'f3', 'f4']]
        self.assertEqual(y3.index.values.tolist(), ['f1', 'f3', 'f4'])
        self.assertEqual(y3.index.values.tolist(), [f.name for f in y3.values])

    def test_yarn_loc_h(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')

        b1 = Bus.from_frames((f1, f2))
        b2 = Bus.from_frames((f3, f4))

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)

        y2 = y1[['f1', 'f2']]
        self.assertEqual(y2.index.values.tolist(), ['f1', 'f2'])
        self.assertEqual([f.name for f in y2.values], ['f1', 'f2'])
        self.assertIsNone(y2._values[1]) # removed Bus
        # we keep an old hierarchy
        self.assertTrue(y2._hierarchy.equals(y1._hierarchy))

    #---------------------------------------------------------------------------

    def test_yarn_iloc_a(self) -> None:
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

        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=False, name='foo')
        self.assertEqual(y1.iloc[3].shape, (4, 4))
        self.assertEqual(y1.iloc[3:].shape, (4,))
        self.assertEqual(y1.iloc[[1, 6]].shape, (2,))
        self.assertEqual(y1.iloc[y1.index.via_str.startswith('f3')].shape, (1,))

    #---------------------------------------------------------------------------

    def test_yarn_keys_a(self) -> None:
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

        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=False, name='foo')
        self.assertEqual(tuple(y1.keys()), ('f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7'))

    #---------------------------------------------------------------------------

    def test_yarn_iter_a(self) -> None:
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

        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=False, name='foo')
        self.assertEqual(next(iter(y1)), 'f1')

    #---------------------------------------------------------------------------

    def test_yarn_contains_a(self) -> None:
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

        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=False, name='foo')
        self.assertTrue('f6' in y1)

    def test_yarn_contains_b(self) -> None:
        f1 = ff.parse('s(4,4)|v(int,float)').rename('f1')
        f2 = ff.parse('s(4,4)|v(str)').rename('f2')
        f3 = ff.parse('s(4,4)|v(bool)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3))

        f4 = ff.parse('s(4,4)|v(int,float)').rename('f4')
        f5 = ff.parse('s(4,4)|v(str)').rename('f5')
        b2 = Bus.from_frames((f4, f5))

        f6 = ff.parse('s(2,4)|v(int,float)').rename('f6')
        f7 = ff.parse('s(4,2)|v(str)').rename('f7')
        b3 = Bus.from_frames((f6, f7))

        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=False, name='foo')
        y2 = y1[['f6', 'f4']]
        self.assertTrue('f6' in y2) # pylint: disable=E1135
        self.assertTrue('f4' in y2) # pylint: disable=E1135
        self.assertFalse('f3' in y2) # pylint: disable=E1135

    #---------------------------------------------------------------------------

    def test_yarn_get_a(self) -> None:
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

        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=False, name='foo')
        self.assertTrue(y1.get('f2').equals(f2))
        self.assertEqual(y1.get('f99'), None)

    #---------------------------------------------------------------------------

    def test_yarn_head_a(self) -> None:
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

        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=False, name='foo')
        self.assertEqual(y1.head(2).shape, (2,))
        self.assertEqual(tuple(y1.head(2).keys()), ('f1', 'f2'))

    def test_yarn_head_b(self) -> None:
        f1 = ff.parse('s(4,4)|v(int,float)').rename('f1')
        f2 = ff.parse('s(4,3)|v(str)').rename('f2')
        f3 = ff.parse('s(4,2)|v(bool)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3))

        f4 = ff.parse('s(3,4)|v(int,float)').rename('f4')
        f5 = ff.parse('s(3,5)|v(str)').rename('f5')
        b2 = Bus.from_frames((f4, f5))

        f6 = ff.parse('s(2,4)|v(int,float)').rename('f6')
        f7 = ff.parse('s(2,2)|v(str)').rename('f7')
        b3 = Bus.from_frames((f6, f7))

        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=False, name='foo')
        y2 = y1[['f7', 'f1', 'f6']]
        self.assertEqual(y2.index.values.tolist(), ['f7', 'f1', 'f6'])
        self.assertEqual([f.name for f in y2.values], ['f7', 'f1', 'f6'])

        y3 = y2.head(2)
        self.assertEqual(y3.shape, (2,))
        self.assertEqual(tuple(y3.keys()), ('f7', 'f1'))

    #---------------------------------------------------------------------------

    def test_yarn_tail_a(self) -> None:
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

        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=False, name='foo')
        self.assertEqual(y1.tail(2).shape, (2,))
        self.assertEqual(tuple(y1.tail(2).keys()), ('f6', 'f7'))

    #---------------------------------------------------------------------------

    def test_yarn_items_a(self) -> None:
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

            labels = []
            for label, frame in y1.items():
                self.assertTrue(frame.__class__ is Frame)
                labels.append(label)

            self.assertEqual(labels, list(y1.index))
            self.assertEqual(y1.status['loaded'].sum(), 2)
            self.assertEqual(y1.status.loc[y1.status['loaded']].index.values.tolist(),
                ['f3', 'f6'])

    #---------------------------------------------------------------------------

    def test_yarn_values_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3), name='a')
        b2 = Bus.from_frames((f4, f5, f6), name='b')

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)
        s1 = y1.values
        self.assertEqual(len(s1), len(y1))
        self.assertEqual([f.shape for f in y1.values],
            [(4, 2), (4, 5), (2, 2), (2, 8), (4, 4), (6, 4)]
            )

    #---------------------------------------------------------------------------

    def test_yarn_display_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3), name='a')
        b2 = Bus.from_frames((f4, f5, f6), name='b')

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)

        d = y1.display(DisplayConfig(type_show=True, type_color=False))
        self.assertEqual(d.to_rows(),
            ['<Yarn>', '<Index>', 'f1      Frame', 'f2      Frame', 'f3      Frame', 'f4      Frame', 'f5      Frame', 'f6      Frame', '<<U2>   <object>'])

    #---------------------------------------------------------------------------

    def test_yarn_mloc_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')

        b1 = Bus.from_frames((f1, f2), name='a')
        b2 = Bus.from_frames((f3, f4), name='b')

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)
        self.assertEqual(y1.mloc.shape, (4,))

    def test_yarn_mloc_b(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')

        b1 = Bus.from_frames((f1, f2))
        b2 = Bus.from_frames((f3, f4))

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)
        y2 = y1[['f4', 'f2']]
        self.assertEqual(y2.mloc.shape, (2,))

    def test_yarn_mloc_c(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')

        b1 = Bus.from_frames((f1, f2))
        b2 = Bus.from_frames((f3, f4))

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)
        y2 = y1[['f3']]
        self.assertEqual(y2.mloc.shape, (1,))



    #---------------------------------------------------------------------------

    def test_yarn_dtypes_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')

        b1 = Bus.from_frames((f1, f2), name='a')
        b2 = Bus.from_frames((f3, f4), name='b')

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)
        self.assertEqual(y1.dtypes.shape, (4, 8))

    def test_yarn_dtypes_b(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')

        b1 = Bus.from_frames((f1, f2), name='a')
        b2 = Bus.from_frames((f3, f4), name='b')

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)
        y2 = y1[['f4', 'f1']]
        self.assertEqual(y2.dtypes.shape, (2, 8))


    #---------------------------------------------------------------------------

    def test_yarn_shapes_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')

        b1 = Bus.from_frames((f1, f2), name='a')
        b2 = Bus.from_frames((f3, f4), name='b')

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)
        self.assertEqual(y1.shapes.to_pairs(),
                (('f1', (4, 2)), ('f2', (4, 5)), ('f3', (2, 2)), ('f4', (2, 8)))
                )

    #---------------------------------------------------------------------------

    def test_yarn_items_b(self) -> None:
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

            s1 = y1.to_series()

            self.assertEqual(
                [(label, f.shape) for label, f in s1.items()],
                [('f1', (4, 2)), ('f2', (4, 5)), ('f3', (2, 2)), ('f4', (2, 8)), ('f5', (4, 4)), ('f6', (6, 4))]
                )

    #---------------------------------------------------------------------------

    def test_yarn_equals_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')

        b1 = Bus.from_frames((f1, f2), name='a')
        b2 = Bus.from_frames((f3, f4), name='b')

        y1 = Yarn.from_buses((b1, b2), retain_labels=False, name='foo')
        y2 = Yarn.from_buses((b1, b2), retain_labels=False, name='bar')

        self.assertTrue(y1.equals(y1))
        self.assertFalse(y1.equals(f1))
        self.assertFalse(y1.equals(f1, compare_class=True))

        self.assertTrue(y1.equals(y2))
        self.assertFalse(y1.equals(y2, compare_name=True))

    def test_yarn_equals_b(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')

        b1 = Bus.from_frames((f1, f2), name='a')
        b2 = Bus.from_frames((f3, f4), name='b')

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)
        y2 = Yarn.from_buses((b1,), retain_labels=False)

        # fails on length
        self.assertFalse(y1.equals(y2))

    def test_yarn_equals_c(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4a = ff.parse('s(2,8)').rename('f4a')
        f4b = ff.parse('s(2,8)').rename('f4b')

        b1 = Bus.from_frames((f1, f2), name='a')
        b2a = Bus.from_frames((f3, f4a), name='b')
        b2b = Bus.from_frames((f3, f4b), name='b')

        y1 = Yarn.from_buses((b1, b2a), retain_labels=False)
        y2 = Yarn.from_buses((b1, b2b), retain_labels=False)

        # fails on index
        self.assertFalse(y1.equals(y2))

    def test_yarn_equals_d(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4a = ff.parse('s(2,8)').rename('f4')
        f4b = ff.parse('s(2,8)|v(str)').rename('f4')

        b1 = Bus.from_frames((f1, f2), name='a')
        b2a = Bus.from_frames((f3, f4a), name='b')
        b2b = Bus.from_frames((f3, f4b), name='b')

        y1 = Yarn.from_buses((b1, b2a), retain_labels=False)
        y2 = Yarn.from_buses((b1, b2b), retain_labels=False)

        # fails on index
        self.assertFalse(y1.equals(y2))

    def test_yarn_equals_e(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3), name='a')
        b2 = Bus.from_frames((f4, f5, f6), name='b')

        y1 = Yarn.from_buses((b1, b2), retain_labels=True)

        with temp_file('.zip') as fp1, temp_file('.zip') as fp2:
            b1.to_zip_pickle(fp1)
            b2.to_zip_pickle(fp2)

            bus_a = Bus.from_zip_pickle(fp1, max_persist=1).rename('a')
            bus_b = Bus.from_zip_pickle(fp2, max_persist=1).rename('b')

            y2 = Yarn.from_buses((bus_a, bus_b), retain_labels=True)
            self.assertEqual(y2.status['loaded'].sum(), 0)

            self.assertTrue(y1.equals(y2))
            self.assertEqual(y2.status['loaded'].sum(), 2)

    #---------------------------------------------------------------------------

    def test_yarn_to_zip_pickle_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3))
        b2 = Bus.from_frames((f4, f5, f6))

        with temp_file('.zip') as fp1, temp_file('.zip') as fp2, temp_file('.zip') as fp3:
            b1.to_zip_pickle(fp1)
            b2.to_zip_pickle(fp2)

            bus_a = Bus.from_zip_pickle(fp1, max_persist=1).rename('a')
            bus_b = Bus.from_zip_pickle(fp2, max_persist=1).rename('b')

            y1 = Yarn.from_buses((bus_a, bus_b), retain_labels=False)
            y1.to_zip_pickle(fp3)

            b3 = Bus.from_zip_pickle(fp3)
            self.assertTrue(b3.index.equals(y1.index))

    #---------------------------------------------------------------------------

    def test_yarn_iter_element_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3), name='b1')
        b2 = Bus.from_frames((f4, f5, f6), name='b2')
        y1 = Yarn.from_buses((b1, b2), retain_labels=False)

        s1 = y1.iter_element().apply(lambda f: f.shape)
        self.assertEqual(s1.to_pairs(),
                (('f1', (4, 2)), ('f2', (4, 5)), ('f3', (2, 2)), ('f4', (2, 8)), ('f5', (4, 4)), ('f6', (6, 4))))

        self.assertEqual([f.name for f in y1.iter_element() if f.shape[0] > 2], #type: ignore
                ['f1', 'f2', 'f5', 'f6'],
                )

    def test_yarn_iter_element_b(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3), name='b1')
        b2 = Bus.from_frames((f4, f5, f6), name='b2')
        y1 = Yarn.from_buses((b1, b2), retain_labels=False)

        y2 = y1[['f5', 'f1', 'f4']]
        post = list(y2.iter_element())
        self.assertEqual(len(post), 3)

    #---------------------------------------------------------------------------

    def test_yarn_iter_element_items_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3), name='b1')
        b2 = Bus.from_frames((f4, f5, f6), name='b2')
        y1 = Yarn.from_buses((b1, b2), retain_labels=False)

        s1 = y1.iter_element_items().apply(lambda label, f: (label, f.shape[0], f.shape[1]))

        self.assertEqual(s1.to_pairs(),
                (('f1', ('f1', 4, 2)), ('f2', ('f2', 4, 5)), ('f3', ('f3', 2, 2)), ('f4', ('f4', 2, 8)), ('f5', ('f5', 4, 4)), ('f6', ('f6', 6, 4)))
                )

        self.assertEqual([label for label, f in y1.iter_element_items() if f.shape[0] > 2],
                ['f1', 'f2', 'f5', 'f6'],
                )

    #---------------------------------------------------------------------------

    def test_yarn_drop_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3), name='b1')
        b2 = Bus.from_frames((f4,), name='b2')
        b3 = Bus.from_frames((f5, f6), name='b3')

        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=False)

        y2 = y1.drop['f3':'f5'] #type: ignore
        self.assertEqual([(f.name, f.shape) for f in y2.values],
                [('f1', (4, 2)), ('f2', (4, 5)), ('f6', (6, 4))]
                )

        y3 = y1.drop[y1.index.isin(('f1', 'f6'))]
        self.assertEqual([(f.name, f.shape) for f in y3.values],
                [('f2', (4, 5)), ('f3', (2, 2)), ('f4', (2, 8)), ('f5', (4, 4))]
                )

        y4 = y1.drop['f4']
        self.assertEqual([(f.name, f.shape) for f in y4.values],
                [('f1', (4, 2)), ('f2', (4, 5)), ('f3', (2, 2)), ('f5', (4, 4)), ('f6', (6, 4))]
                )

        y5 = y1.drop[['f4', 'f5', 'f6']]
        self.assertEqual([(f.name, f.shape) for f in y5.values],
                [('f1', (4, 2)), ('f2', (4, 5)), ('f3', (2, 2))]
                )

    def test_yarn_drop_b(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3), name='b1')
        b2 = Bus.from_frames((f4,), name='b2')
        b3 = Bus.from_frames((f5, f6), name='b3')

        y1 = Yarn.from_buses((b1, b2, b3), retain_labels=False)

        y2 = y1.drop.iloc[2: 5]
        self.assertEqual([(f.name, f.shape) for f in y2.values],
                [('f1', (4, 2)), ('f2', (4, 5)), ('f6', (6, 4))]
                )

        y3 = y1.drop.iloc[np.array([True, False, False, False, False, True])]
        self.assertEqual([(f.name, f.shape) for f in y3.values],
                [('f2', (4, 5)), ('f3', (2, 2)), ('f4', (2, 8)), ('f5', (4, 4))]
                )

        y4 = y1.drop.iloc[3]
        self.assertEqual([(f.name, f.shape) for f in y4.values],
                [('f1', (4, 2)), ('f2', (4, 5)), ('f3', (2, 2)), ('f5', (4, 4)), ('f6', (6, 4))]
                )

        y5 = y1.drop.iloc[[3, 4, 5]]
        self.assertEqual([(f.name, f.shape) for f in y5.values],
                [('f1', (4, 2)), ('f2', (4, 5)), ('f3', (2, 2))]
                )

    #---------------------------------------------------------------------------

    def test_yarn_unpersist_a(self) -> None:
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
            self.assertEqual(len(tuple(y1.items())), 6)

            self.assertEqual(y1.status['loaded'].sum(), 2)
            y1.unpersist()

            self.assertEqual(y1.status['loaded'].sum(), 0)
            self.assertEqual(len(tuple(y1.items())), 6)
            self.assertEqual(y1.status['loaded'].sum(), 2)

    #---------------------------------------------------------------------------

    def test_yarn_relabel_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3))
        b2 = Bus.from_frames((f4,))
        b3 = Bus.from_frames((f5, f6))

        y1 = Yarn((b1, b2, b3))

        self.assertEqual(
                y1.relabel(lambda x: f'--{x}--').loc['--4--'].shape, (4, 4)
                )

        # None is a no-op
        self.assertEqual(
                y1.relabel(None).loc[4].shape, (4, 4)
                )

        with self.assertRaises(RelabelInvalid):
            y1.relabel({3,4,5})

        self.assertEqual(
                y1.relabel(tuple('abcdef'))['d':].status['shape'].to_pairs(), #type: ignore
                (('d', (2, 8)), ('e', (4, 4)), ('f', (6, 4)))
                )


        y2 = Yarn((b1, b2, b3), index=tuple('abcdef'))
        self.assertEqual(y2.index.values.tolist(), ['a', 'b', 'c', 'd', 'e', 'f'])
        self.assertEqual(y2.relabel(IndexAutoFactory).index.values.tolist(), [0, 1, 2, 3, 4, 5])

    #---------------------------------------------------------------------------

    def test_yarn_relabel_flat_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3))
        b2 = Bus.from_frames((f4,))
        b3 = Bus.from_frames((f5, f6))

        y1 = Yarn((b1, b2, b3), index=IndexHierarchy.from_product(('a', 'b'), (1, 2, 3)))

        self.assertEqual(
                y1.relabel_flat()[('a', 3):].status['shape'].to_pairs(), # type: ignore
                ((('a', 3), (2, 2)), (('b', 1), (2, 8)), (('b', 2), (4, 4)), (('b', 3), (6, 4)))
                )

    def test_yarn_relabel_flat_b(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')

        b1 = Bus.from_frames((f1, f2, f3))

        y1 = Yarn((b1,))
        with self.assertRaises(RuntimeError):
            y1.relabel_flat()

    #---------------------------------------------------------------------------

    def test_yarn_relabel_level_add_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3))
        b2 = Bus.from_frames((f4,))
        b3 = Bus.from_frames((f5, f6))

        y1 = Yarn((b1, b2, b3), index=IndexHierarchy.from_product(('a', 'b'), (1, 2, 3)))

        self.assertEqual(
                y1.relabel_level_add('c').iloc[4:].status['shape'].to_pairs(),
                ((('c', 'b', 2), (4, 4)), (('c', 'b', 3), (6, 4)))
                )

    #---------------------------------------------------------------------------

    def test_yarn_relabel_level_drop_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3))
        b2 = Bus.from_frames((f4,))
        b3 = Bus.from_frames((f5, f6))

        y1 = Yarn((b1, b2, b3), index=IndexHierarchy.from_product(('a', 'b'), (1, 2, 3)))

        self.assertEqual(
                y1.iloc[[0,2,4]].relabel_level_drop(1).status['shape'].to_pairs(),
                ((1, (4, 2)), (3, (2, 2)), (2, (4, 4)))
                )

    def test_yarn_relabel_level_drop_b(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')

        b1 = Bus.from_frames((f1, f2, f3))

        y1 = Yarn((b1,))
        with self.assertRaises(RuntimeError):
            y1.relabel_level_drop()

    #---------------------------------------------------------------------------

    def test_yarn_rehierarch_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3))
        b2 = Bus.from_frames((f4,))
        b3 = Bus.from_frames((f5, f6))

        y1 = Yarn((b1, b2, b3), index=IndexHierarchy.from_product(('a', 'b'), (1, 2, 3)))
        self.assertEqual(
                y1.iloc[[0,2,4]].rehierarch((1, 0)).status['shape'].to_pairs(),
                (((1, 'a'), (4, 2)), ((3, 'a'), (2, 2)), ((2, 'b'), (4, 4)))
                )

    def test_yarn_rehierarch_b(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')

        b1 = Bus.from_frames((f1, f2, f3))

        y1 = Yarn((b1,))
        with self.assertRaises(RuntimeError):
            y1.rehierarch((1,0))

    #---------------------------------------------------------------------------

    def test_yarn_to_signature_bytes_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3))
        b2 = Bus.from_frames((f4,))
        b3 = Bus.from_frames((f5, f6))

        a1 = np.array((1, 2, 3), dtype=np.int64)
        a1.flags.writeable = False
        y1 = Yarn((b1, b2, b3), index=IndexHierarchy.from_product(('a', 'b'), a1))

        bytes1 = y1._to_signature_bytes(include_name=False)
        self.assertEqual(sha256(bytes1).hexdigest(),
            '4f080dceb07b959487a84134447e0e838754ac45246975080b9fc8bc85829bc6')

    def test_yarn_to_signature_bytes_b(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3))
        b2 = Bus.from_frames((f4,))
        b3 = Bus.from_frames((f5, f6))

        a1 = np.array((1, 2, 3), dtype=np.int64)
        a1.flags.writeable = False
        y1 = Yarn((b1, b2, b3), index=IndexHierarchy.from_product(('a', 'b'), a1))
        bytes1 = y1._to_signature_bytes(include_name=False)

        y2 = Yarn((b1, b2, b3))
        bytes2 = y2._to_signature_bytes(include_name=False)

        self.assertNotEqual(sha256(bytes1).hexdigest(), sha256(bytes2).hexdigest())

    def test_yarn_to_signature_bytes_c(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3))
        b2 = Bus.from_frames((f4,))
        b3 = Bus.from_frames((f5, f6))
        b4 = Bus.from_frames((f5,))

        y1 = Yarn((b1, b2, b3))
        bytes1 = y1._to_signature_bytes(include_name=False)

        y2 = Yarn((b1, b2, b4))
        bytes2 = y2._to_signature_bytes(include_name=False)

        self.assertNotEqual(sha256(bytes1).hexdigest(), sha256(bytes2).hexdigest())

    def test_yarn_to_signature_bytes_d(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')

        b1 = Bus.from_frames((f1, f2, f3))
        y1 = Yarn.from_buses((b1,), retain_labels=False)

        # if not comparing class, the bytes of a Yarn and Bus will be the same as only index and contained frame are read
        bytes1 = b1._to_signature_bytes(include_name=False, include_class=False)
        bytes2 = y1._to_signature_bytes(include_name=False, include_class=False)

        self.assertEqual(sha256(bytes1).hexdigest(), sha256(bytes2).hexdigest())

    def test_yarn_via_hashlib_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3))
        b2 = Bus.from_frames((f4,))
        b3 = Bus.from_frames((f5, f6))

        y1 = Yarn((b1, b2, b3))
        digest = y1.via_hashlib(include_name=False).sha256().hexdigest()
        self.assertEqual(digest, 'c966f3bfa983b696da219838197f627135b3405dc2f50c0ab29c02ed74ac3bd1')

    #---------------------------------------------------------------------------

    def test_yarn_sort_index_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')

        b1 = Bus.from_frames((f4, f2))
        b2 = Bus.from_frames((f1, f3))

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)

        self.assertEqual(y1.shapes.to_pairs(),
            (('f4', (2, 8)), ('f2', (4, 5)), ('f1', (4, 2)), ('f3', (2, 2))))

        y2 = y1.sort_index()
        self.assertEqual(y2.shapes.to_pairs(),
            (('f1', (4, 2)), ('f2', (4, 5)), ('f3', (2, 2)), ('f4', (2, 8))))

    #---------------------------------------------------------------------------

    def test_yarn_sort_values_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')

        b1 = Bus.from_frames((f1, f2))
        b2 = Bus.from_frames((f3, f4))

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)
        y2 = y1.sort_values(key=lambda y: np.array([f.size for f in y.iter_element()]))

        self.assertEqual(y2.index.values.tolist(), ['f3', 'f1', 'f4', 'f2'])
        self.assertEqual([f.name for f in y2.iter_element()], ['f3', 'f1', 'f4', 'f2'])

    def test_yarn_sort_values_b(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')

        b1 = Bus.from_frames((f1, f2))
        b2 = Bus.from_frames((f3, f4))

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)

        with self.assertRaises(RuntimeError):
            _ = y1.sort_values(key=lambda y: np.array([f.size for f in y.iter_element()]), ascending=[True, False])

        y2 = y1.sort_values(key=lambda y: np.array([f.size for f in y.iter_element()]), ascending=False)

        self.assertEqual(y2.index.values.tolist(), ['f2', 'f4', 'f1', 'f3'])
        self.assertEqual([f.name for f in y2.iter_element()], ['f2', 'f4', 'f1', 'f3'])

    def test_yarn_sort_values_c(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')

        b1 = Bus.from_frames((f1, f2))
        b2 = Bus.from_frames((f3, f4))

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)

        y2 = y1[['f4', 'f3']]

        y3 = y2.sort_values(key=lambda y: np.array([f.size for f in y.iter_element()]))

        self.assertEqual(y3.index.values.tolist(), ['f3', 'f4'])
        self.assertEqual([f.name for f in y3.iter_element()], ['f3', 'f4'])

    #---------------------------------------------------------------------------

    def test_yarn_roll_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')

        b1 = Bus.from_frames((f1, f2))
        b2 = Bus.from_frames((f3, f4))

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)
        y2 = y1.roll(2)

        self.assertEqual(
                [(l, f.name) for l, f in y2.items()],
                [('f1', 'f3'), ('f2', 'f4'), ('f3', 'f1'), ('f4', 'f2')],
                )

        y3 = y1.roll(4)
        self.assertTrue(y3.equals(y1))

    def test_yarn_roll_b(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')

        b1 = Bus.from_frames((f1, f2))
        b2 = Bus.from_frames((f3, f4))

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)
        y2 = y1.roll(2, include_index=True)

        self.assertEqual(
                [(l, f.name) for l, f in y2.items()],
                [('f3', 'f3'), ('f4', 'f4'), ('f1', 'f1'), ('f2', 'f2')],
                )

    def test_yarn_shift_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')

        b1 = Bus.from_frames((f1, f2))
        b2 = Bus.from_frames((f3, f4))

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)
        with self.assertRaises(NotImplementedError):
            _ = y1.shift(2, fill_value=None)

    #---------------------------------------------------------------------------

    def test_yarn_reindex_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')

        b1 = Bus.from_frames((f1, f2))
        b2 = Bus.from_frames((f3, f4))

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)
        y2 = y1.reindex(['f3', 'f1', 'f2'])
        self.assertEqual(y2.index.values.tolist(), ['f3', 'f1', 'f2'])
        self.assertEqual([f.name for f in y2.iter_element()], ['f3', 'f1', 'f2'])

    def test_yarn_reindex_b(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')

        b1 = Bus.from_frames((f1, f2))
        b2 = Bus.from_frames((f3, f4))

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)
        with self.assertRaises(NotImplementedError):
            _ = y1.reindex(['f3', 'f1', 'f2', 'f8'])

    def test_yarn_reindex_c(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')

        b1 = Bus.from_frames((f1, f2))
        b2 = Bus.from_frames((f3, f4))

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)
        y2 = y1.reindex(['f3', 'f1', 'f2', 'f4'])
        self.assertEqual(y2.index.values.tolist(), ['f3', 'f1', 'f2', 'f4'])
        self.assertEqual([f.name for f in y2.iter_element()], ['f3', 'f1', 'f2', 'f4'])

        y3 = y2[['f4', 'f3']]
        y4 = y3.reindex(('f3',))

        self.assertEqual(y4.index.values.tolist(), ['f3'])
        self.assertEqual([f.name for f in y4.iter_element()], ['f3'])

        # y4 still retains the original hierarchyh
        self.assertIs(y1._hierarchy, y4._hierarchy)

    def test_yarn_reindex_d(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')

        b1 = Bus.from_frames((f1, f2))
        b2 = Bus.from_frames((f3, f4))

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)

        idx = Index(['f3', 'f1', 'f2'])
        y2 = y1.reindex(idx, own_index=True)
        self.assertTrue(y2.index is idx)

        y3 = y2.reindex(y2.index, own_index=True)
        self.assertTrue(y3.index is idx)


    def test_yarn_reindex_e(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')

        b1 = Bus.from_frames((f1, f2))
        b2 = Bus.from_frames((f3, f4))

        y1 = Yarn.from_buses((b1, b2), retain_labels=False)

        y2 = y1.reindex(())
        self.assertEqual(len(y2), 0)
        self.assertEqual(y2._values.tolist(), [None, None])
        self.assertEqual(len(y2._hierarchy), 4)


if __name__ == '__main__':
    import unittest
    unittest.main()
