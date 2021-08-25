

import unittest

import frame_fixtures as ff
import numpy as np

from static_frame.test.test_case import TestCase
from static_frame.core.bus import Bus
# from static_frame.core.exception import ErrorInitQuilt
# from static_frame.core.exception import AxisInvalid
# from static_frame.core.exception import ErrorInitYarn

# from static_frame.core.axis_map import AxisMap
# from static_frame.core.index import Index
# from static_frame.core.axis_map import IndexMap

from static_frame.core.display_config import DisplayConfig
from static_frame.core.yarn import Yarn
from static_frame.core.frame import Frame
from static_frame.test.test_case import temp_file
from static_frame.core.exception import ErrorInitYarn
from static_frame.core.exception import ErrorInitIndex
from static_frame import ILoc
from static_frame import HLoc

class TestUnit(TestCase):

    #---------------------------------------------------------------------------
    def test_yarn_init_a(self) -> None:

        with self.assertRaises(ErrorInitYarn):
            Yarn(np.array([3, 4]), retain_labels=True)


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

        y1 = Yarn.from_concat((b1, b2, b3), retain_labels=False)
        self.assertEqual(y1.shape, (7,))

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

            y1 = Yarn.from_concat((bus_a, bus_b), retain_labels=False)
            self.assertEqual(y1.shape, (6,))
            self.assertEqual(y1.status['loaded'].sum(), 0)

            from static_frame import IndexAutoFactory
            y2 = Yarn.from_concat((y1, y1), retain_labels=True, index=IndexAutoFactory)
            self.assertEqual(y2[(1, 'f4')].shape, (2, 8))
            self.assertEqual(y2[(2, 'f1')].shape, (4, 2))
            self.assertEqual(y2[(3, 'f6')].shape, (6, 4))

            y3 = y2.iloc[4:]
            self.assertEqual(y3.shape, (8,))


    def test_yarn_from_concat_c(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')

        with self.assertRaises(NotImplementedError):
            Yarn.from_concat((f1, f2), retain_labels=True)

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
        y2 = y1.loc[['f7', 'f3']]
        self.assertEqual(y2.shapes.to_pairs(),
                (('f7', (4, 2)), ('f3', (4, 4)))
                )
        with self.assertRaises(ErrorInitIndex):
            y2 = y1.loc[['f1', 'f7', 'f3']]

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
        self.assertEqual(y2.shapes.to_pairs(),
                (('f2', (4, 4)), ('f3', (4, 4)), ('f4', (4, 4)), ('f5', (4, 4)), ('f6', (2, 4))))
        self.assertEqual(y2['f5'].to_pairs(),
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



if __name__ == '__main__':
    unittest.main()

