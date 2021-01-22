
import unittest

import frame_fixtures as ff
import numpy as np

from static_frame.test.test_case import TestCase
from static_frame.core.quilt import Quilt
from static_frame.core.quilt import AxisMap
from static_frame.core.index import Index
from static_frame.core.hloc import HLoc
from static_frame.core.display_config import DisplayConfig
from static_frame.core.index import ILoc
from static_frame.core.frame import Frame
from static_frame.core.bus import Bus
from static_frame.core.batch import Batch
from static_frame.core.store import StoreConfig

from static_frame.test.test_case import temp_file
from static_frame.core.exception import ErrorInitQuilt
from static_frame.core.exception import ErrorInitIndexNonUnique


class TestUnit(TestCase):

    def test_axis_map_a(self) -> None:

        components = dict(
                x=Index(('a', 'b', 'c')),
                y=Index(('a', 'b', 'c')),
                )

        am = AxisMap.get_axis_series(components) #type: ignore
        self.assertEqual(am.to_pairs(),
                ((('x', 'a'), 'x'), (('x', 'b'), 'x'), (('x', 'c'), 'x'), (('y', 'a'), 'y'), (('y', 'b'), 'y'), (('y', 'c'), 'y')))

    #---------------------------------------------------------------------------
    def test_quilt_init_a(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(a=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f1, f2, f3))
        # columns are aligned
        q1 = Quilt(b1, retain_labels=True, axis=0)
        self.assertEqual(q1.shape, (7, 2))

        # index is not aligned
        q2 = Quilt(b1, retain_labels=True, axis=1)
        with self.assertRaises(ErrorInitQuilt):
            self.assertEqual(q2.shape, (7, 2))

    def test_quilt_init_b(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), c=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(a=(2,3), b=(4,6)),
                index=('x', 'y'),
                name='f2')
        f3 = Frame.from_dict(
                dict(c=(10,20), b=(50,60)),
                index=('x', 'y'),
                name='f3')

        b1 = Bus.from_frames((f1, f2, f3))
        # columns are not aligned
        q1 = Quilt(b1, retain_labels=True, axis=0)
        with self.assertRaises(ErrorInitQuilt):
            self.assertEqual(q1.shape, (7, 2))

        # index is aligned
        q2 = Quilt(b1, retain_labels=True, axis=1)
        self.assertEqual(q2.shape, (2, 6))

        # must retain labels for non-unique axis
        q3 = Quilt(b1, retain_labels=False, axis=1)
        with self.assertRaises(ErrorInitIndexNonUnique):
            self.assertEqual(q3.shape, (7, 2))


    #---------------------------------------------------------------------------
    def test_quilt_display_a(self) -> None:

        dc = DisplayConfig(type_show=False)

        f1 = ff.parse('s(10,4)|v(int)|i(I,str)|c(I,str)').rename('foo')
        q1 = Quilt.from_frame(f1, chunksize=2, retain_labels=False)
        self.assertEqual(
                q1.display(dc).to_rows(),
                f1.display(dc).to_rows())

    def test_quilt_values_a(self) -> None:
        f1 = ff.parse('s(6,4)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=2, retain_labels=False)
        self.assertEqual(q1.values.tolist(),
                [[-88017, 162197, -3648, 129017], [92867, -41157, 91301, 35021], [84967, 5729, 30205, 166924], [13448, -168387, 54020, 122246], [175579, 140627, 129017, 197228], [58768, 66269, 35021, 105269]])


    def test_quilt_nbytes_a(self) -> None:

        dc = DisplayConfig(type_show=False)

        f1 = ff.parse('s(10,4)|v(int)|i(I,str)|c(I,str)').rename('foo')
        q1 = Quilt.from_frame(f1, chunksize=2, retain_labels=False)
        self.assertEqual(q1.nbytes, f1.nbytes)

    #---------------------------------------------------------------------------
    def test_quilt_from_frame_a(self) -> None:

        f1 = ff.parse('s(100,4)|v(int)|i(I,str)|c(I,str)').rename('foo')

        q1 = Quilt.from_frame(f1, chunksize=10, retain_labels=False)

        self.assertEqual(q1.name, 'foo')
        self.assertEqual(q1.rename('bar').name, 'bar')
        self.assertTrue(repr(q1).startswith('<Quilt: foo'))

        post, opp = AxisMap.from_bus(q1._bus, q1._axis, deepcopy_from_bus=True)
        self.assertEqual(len(post), 100)
        self.assertEqual(len(opp), 4)

        s1 = q1['ztsv']
        self.assertEqual(s1.shape, (100,))
        self.assertTrue(s1['zwVN'] == f1.loc['zwVN', 'ztsv'])

        f1 = q1['zUvW':] #type: ignore
        self.assertEqual(f1.shape, (100, 2))
        self.assertEqual(f1.columns.values.tolist(), ['zUvW', 'zkuW'])

        f2 = q1[['zZbu', 'zkuW']]
        self.assertEqual(f2.shape, (100, 2))
        self.assertEqual(f2.columns.values.tolist(), ['zZbu', 'zkuW'])

        f3 = q1.loc['zQuq':, 'zUvW':] #type: ignore
        self.assertEqual(f3.shape, (6, 2))


    def test_quilt_from_frame_b(self) -> None:

        f1 = ff.parse('s(4,100)|v(int)|i(I,str)|c(I,str)')

        q1 = Quilt.from_frame(f1, chunksize=10, axis=1, retain_labels=False)

        post, opp = AxisMap.from_bus(q1._bus, q1._axis, deepcopy_from_bus=False)
        self.assertEqual(len(post), 100)
        self.assertEqual(len(opp), 4)

    def test_quilt_from_frame_c(self) -> None:

        f1 = ff.parse('s(100,4)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=10, axis=1, retain_labels=False)
        self.assertEqual(q1.shape, (100, 4))
        self.assertEqual(len(q1._bus), 1)


    #---------------------------------------------------------------------------

    def test_quilt_extract_a1(self) -> None:

        f1 = ff.parse('s(4,100)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=10, axis=1, retain_labels=False)
        self.assertEqual(q1.shape, (4, 100))
        self.assertEqual(len(q1._bus), 10)
        self.assertEqual(q1['zkuW':'zTSt'].shape, (4, 95)) #type: ignore
        self.assertEqual(q1.loc[ILoc[-2:], 'zaji': 'zsa5'].shape, (2, 17)) #type: ignore


    def test_quilt_extract_a2(self) -> None:

        f1 = ff.parse('s(4,100)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=10, axis=1, retain_labels=False, deepcopy_from_bus=True)
        self.assertEqual(q1.shape, (4, 100))
        self.assertEqual(len(q1._bus), 10)
        self.assertEqual(q1['zkuW':'zTSt'].shape, (4, 95)) #type: ignore
        self.assertEqual(q1.loc[ILoc[-2:], 'zaji': 'zsa5'].shape, (2, 17)) #type: ignore



    def test_quilt_extract_b1(self) -> None:

        f1 = ff.parse('s(4,10)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=3, axis=1, retain_labels=True)
        self.assertEqual(q1.shape, (4, 10))
        self.assertEqual(len(q1._bus), 4)

        f1 = q1.loc[ILoc[-2:], HLoc[['zkuW', 'z5l6']]]
        self.assertEqual(f1.shape, (2, 6))
        self.assertEqual(f1.to_pairs(0),
                ((('zkuW', 'zkuW'), (('zUvW', 166924), ('zkuW', 122246))), (('zkuW', 'zmVj'), (('zUvW', 170440), ('zkuW', 32395))), (('zkuW', 'z2Oo'), (('zUvW', 175579), ('zkuW', 58768))), (('z5l6', 'z5l6'), (('zUvW', 32395), ('zkuW', 137759))), (('z5l6', 'zCE3'), (('zUvW', 172142), ('zkuW', -154686))), (('z5l6', 'zr4u'), (('zUvW', -31776), ('zkuW', 102088))))
                )

        s1 = q1.loc['zUvW', HLoc[['zkuW', 'z5l6']]]
        self.assertEqual(s1.shape, (6,))
        self.assertEqual(s1.to_pairs(),
                ((('zkuW', 'zkuW'), 166924), (('zkuW', 'zmVj'), 170440), (('zkuW', 'z2Oo'), 175579), (('z5l6', 'z5l6'), 32395), (('z5l6', 'zCE3'), 172142), (('z5l6', 'zr4u'), -31776)))

        s2 = q1.loc[:, ('z5l6', 'z5l6')]
        self.assertEqual(s2.shape, (4,))
        self.assertEqual(s2.name, ('z5l6', 'z5l6'))
        self.assertEqual(s2.to_pairs(),
                (('zZbu', 146284), ('ztsv', 170440), ('zUvW', 32395), ('zkuW', 137759))
                )

    def test_quilt_extract_b2(self) -> None:

        f1 = ff.parse('s(4,10)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=3, axis=1, retain_labels=True, deepcopy_from_bus=True)
        self.assertEqual(q1.shape, (4, 10))
        self.assertEqual(len(q1._bus), 4)

        f1 = q1.loc[ILoc[-2:], HLoc[['zkuW', 'z5l6']]]
        self.assertEqual(f1.shape, (2, 6))
        self.assertEqual(f1.to_pairs(0),
                ((('zkuW', 'zkuW'), (('zUvW', 166924), ('zkuW', 122246))), (('zkuW', 'zmVj'), (('zUvW', 170440), ('zkuW', 32395))), (('zkuW', 'z2Oo'), (('zUvW', 175579), ('zkuW', 58768))), (('z5l6', 'z5l6'), (('zUvW', 32395), ('zkuW', 137759))), (('z5l6', 'zCE3'), (('zUvW', 172142), ('zkuW', -154686))), (('z5l6', 'zr4u'), (('zUvW', -31776), ('zkuW', 102088))))
                )

        s1 = q1.loc['zUvW', HLoc[['zkuW', 'z5l6']]]
        self.assertEqual(s1.shape, (6,))
        self.assertEqual(s1.to_pairs(),
                ((('zkuW', 'zkuW'), 166924), (('zkuW', 'zmVj'), 170440), (('zkuW', 'z2Oo'), 175579), (('z5l6', 'z5l6'), 32395), (('z5l6', 'zCE3'), 172142), (('z5l6', 'zr4u'), -31776)))

        s2 = q1.loc[:, ('z5l6', 'z5l6')]
        self.assertEqual(s2.shape, (4,))
        self.assertEqual(s2.name, ('z5l6', 'z5l6'))
        self.assertEqual(s2.to_pairs(),
                (('zZbu', 146284), ('ztsv', 170440), ('zUvW', 32395), ('zkuW', 137759))
                )



    def test_quilt_extract_c1(self) -> None:

        f1 = ff.parse('s(20,4)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=5, axis=0, retain_labels=True)
        self.assertEqual(q1.shape, (20, 4))
        self.assertEqual(len(q1._bus), 4)

        # returns a Series
        self.assertEqual(q1['zUvW'].shape, (20,))
        self.assertEqual(q1['zUvW'].index.depth, 2)
        self.assertEqual(q1.loc[('zOyq', 'zIA5'), :].shape, (4,))
        self.assertEqual(q1.loc[:, 'ztsv'].shape, (20,))

        # return an element
        self.assertEqual(q1.loc[('zOyq', 'zIA5'), 'zkuW'], 92867)


    def test_quilt_extract_c2(self) -> None:

        f1 = ff.parse('s(20,4)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=5, axis=0, retain_labels=True, deepcopy_from_bus=True)
        self.assertEqual(q1.shape, (20, 4))
        self.assertEqual(len(q1._bus), 4)

        # returns a Series
        self.assertEqual(q1['zUvW'].shape, (20,))
        self.assertEqual(q1['zUvW'].index.depth, 2)
        self.assertEqual(q1.loc[('zOyq', 'zIA5'), :].shape, (4,))
        self.assertEqual(q1.loc[:, 'ztsv'].shape, (20,))

        # return an element
        self.assertEqual(q1.loc[('zOyq', 'zIA5'), 'zkuW'], 92867)


    def test_quilt_extract_d1(self) -> None:

        f1 = ff.parse('s(4,20)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=5, axis=1, retain_labels=True)
        self.assertEqual(q1.shape, (4, 20))
        self.assertEqual(len(q1._bus), 4)

        self.assertEqual(q1.loc['zUvW'].shape, (20,))
        self.assertEqual(q1.loc['zUvW'].name, 'zUvW')

    def test_quilt_extract_d2(self) -> None:

        f1 = ff.parse('s(4,20)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=5, axis=1, retain_labels=True, deepcopy_from_bus=True)
        self.assertEqual(q1.shape, (4, 20))
        self.assertEqual(len(q1._bus), 4)

        self.assertEqual(q1.loc['zUvW'].shape, (20,))
        self.assertEqual(q1.loc['zUvW'].name, 'zUvW')

    def test_quilt_extract_e1(self) -> None:

        f1 = ff.parse('s(20,4)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=5, axis=0, retain_labels=False)

        self.assertEqual(q1.loc['zO5l', 'zkuW'], 146284)

        f2 = q1.loc[['zmVj', 'zOyq', 'zmhG', 'zwIp']]
        self.assertTrue(f2.equals(f1.loc[['zmVj', 'zOyq', 'zmhG', 'zwIp']]))

        s1 = q1.loc[['zmVj', 'zOyq', 'zmhG', 'zwIp'], 'zkuW']
        self.assertTrue(s1.equals(f1.loc[['zmVj', 'zOyq', 'zmhG', 'zwIp'], 'zkuW']))
        self.assertEqual(s1.name, 'zkuW')
        self.assertEqual(s1.shape, (4,))

    def test_quilt_extract_e2(self) -> None:

        f1 = ff.parse('s(20,4)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=5, axis=0, retain_labels=False, deepcopy_from_bus=True)

        self.assertEqual(q1.loc['zO5l', 'zkuW'], 146284)

        f2 = q1.loc[['zmVj', 'zOyq', 'zmhG', 'zwIp']]
        self.assertTrue(f2.equals(f1.loc[['zmVj', 'zOyq', 'zmhG', 'zwIp']]))

        s1 = q1.loc[['zmVj', 'zOyq', 'zmhG', 'zwIp'], 'zkuW']
        self.assertTrue(s1.equals(f1.loc[['zmVj', 'zOyq', 'zmhG', 'zwIp'], 'zkuW']))
        self.assertEqual(s1.name, 'zkuW')
        self.assertEqual(s1.shape, (4,))


    def test_quilt_extract_f1(self) -> None:

        f1 = ff.parse('s(4,20)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=5, axis=1, retain_labels=False)

        f2 = q1.loc[:, ['zUvW', 'zB7E', 'zwIp']]
        self.assertTrue(f2.equals(f1.loc[:, ['zUvW', 'zB7E', 'zwIp']]))

        self.assertEqual(q1.loc['zkuW', 'zwIp'], -112188)


    def test_quilt_extract_f2(self) -> None:

        f1 = ff.parse('s(4,20)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=5, axis=1, retain_labels=False, deepcopy_from_bus=True)

        f2 = q1.loc[:, ['zUvW', 'zB7E', 'zwIp']]
        self.assertTrue(f2.equals(f1.loc[:, ['zUvW', 'zB7E', 'zwIp']]))

        self.assertEqual(q1.loc['zkuW', 'zwIp'], -112188)

    def test_quilt_extract_g1(self) -> None:
        from string import ascii_lowercase
        config = StoreConfig(include_index=True, index_depth=1)

        with temp_file('.zip') as fp:

            items = ((ascii_lowercase[i], Frame(np.arange(2_000).reshape(1_000, 2), columns=tuple('xy'))) for i in range(4))

            Batch(items).to_zip_pickle(fp, config=config)

            q1 = Quilt.from_zip_pickle(fp, max_persist=1, retain_labels=True, config=config)
            self.assertEqual(q1.shape, (4_000, 2))
            post = q1.iloc[2_000:2_010, 1:]
            self.assertEqual(post.shape, (10, 1))
            self.assertEqual(set(post.index.values_at_depth(0)), {'c'})

    def test_quilt_extract_g2(self) -> None:
        from string import ascii_lowercase
        config = StoreConfig(include_index=True, index_depth=1)

        with temp_file('.zip') as fp:

            items = ((ascii_lowercase[i], Frame(np.arange(2_000).reshape(1_000, 2), columns=tuple('xy'))) for i in range(4))

            Batch(items).to_zip_pickle(fp, config=config)

            q1 = Quilt.from_zip_pickle(fp, max_persist=1, retain_labels=True, config=config, deepcopy_from_bus=True)
            self.assertEqual(q1.shape, (4_000, 2))
            post = q1.iloc[2_000:2_010, 1:]
            self.assertEqual(post.shape, (10, 1))
            self.assertEqual(set(post.index.values_at_depth(0)), {'c'})



    #---------------------------------------------------------------------------
    def test_quilt_extract_array_a1(self) -> None:

        f1 = ff.parse('s(20,4)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=5, axis=0, retain_labels=True)

        a1 = q1._extract_array(None, 2)
        self.assertEqual(a1.tolist(),
                [-3648, 91301, 30205, 54020, 129017, 35021, 166924, 122246, 197228, 105269, 119909, 194224, 172133, 96520, -88017, 92867, 84967, 13448, 175579, 58768])

        self.assertEqual(q1._extract_array(3).tolist(), [13448, -168387, 54020, 122246])

        self.assertEqual(q1._extract_array([1, 8, 9, 10, 17]).tolist(),
                [[92867, -41157, 91301, 35021], [32395, 17698, 197228, 172133], [137759, -24849, 105269, 96520], [-62964, -30183, 119909, -88017], [130010, 81275, 13448, 170440]]
                )

        self.assertEqual(q1._extract_array([1, 9, 17], [0, 3]).tolist(),
                [[92867, 35021], [137759, 96520], [130010, 170440]])

        self.assertEqual(q1._extract_array(slice(4,8), 2).tolist(),
                [129017, 35021, 166924, 122246])

        self.assertEqual(q1._extract_array(slice(4,8), slice(1,3)).tolist(),
                [[140627, 129017], [66269, 35021], [-171231, 166924], [-38997, 122246]])

        self.assertEqual(q1._extract_array(2, 2), 30205)

    def test_quilt_extract_array_a2(self) -> None:

        f1 = ff.parse('s(20,4)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=5, axis=0, retain_labels=True, deepcopy_from_bus=True)

        a1 = q1._extract_array(None, 2)
        self.assertEqual(a1.tolist(),
                [-3648, 91301, 30205, 54020, 129017, 35021, 166924, 122246, 197228, 105269, 119909, 194224, 172133, 96520, -88017, 92867, 84967, 13448, 175579, 58768])

        self.assertEqual(q1._extract_array(3).tolist(), [13448, -168387, 54020, 122246])

        self.assertEqual(q1._extract_array([1, 8, 9, 10, 17]).tolist(),
                [[92867, -41157, 91301, 35021], [32395, 17698, 197228, 172133], [137759, -24849, 105269, 96520], [-62964, -30183, 119909, -88017], [130010, 81275, 13448, 170440]]
                )

        self.assertEqual(q1._extract_array([1, 9, 17], [0, 3]).tolist(),
                [[92867, 35021], [137759, 96520], [130010, 170440]])

        self.assertEqual(q1._extract_array(slice(4,8), 2).tolist(),
                [129017, 35021, 166924, 122246])

        self.assertEqual(q1._extract_array(slice(4,8), slice(1,3)).tolist(),
                [[140627, 129017], [66269, 35021], [-171231, 166924], [-38997, 122246]])

        self.assertEqual(q1._extract_array(2, 2), 30205)

    def test_quilt_extract_array_b1(self) -> None:

        f1 = ff.parse('s(4,20)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=5, axis=1, retain_labels=True)

        a1 = q1._extract_array(None, 2)
        self.assertEqual(a1.tolist(), [-3648, 91301, 30205, 54020])

        self.assertEqual(q1._extract_array(2).tolist(), [84967, 5729, 30205, 166924, 170440, 175579, 32395, 172142, -31776, -97851, -12447, 119909, 172142, 35684, 170440, 316, 81275, 81275, 96640, -110091])

        self.assertEqual(q1._extract_array(slice(0,2), 2).tolist(), [-3648, 91301])

        self.assertEqual(q1._extract_array(slice(1,3), slice(4,6)).tolist(),
                [[146284, 13448], [170440, 175579]])

        self.assertEqual(q1._extract_array([0, 3], [1, 14, 15]).tolist(),
                [[162197, 58768, 10240], [-168387, 32395, -170415]]
                )

        self.assertEqual(q1._extract_array(2, 2), 30205)
        self.assertEqual(q1._extract_array(-1, -1), -112188)


    def test_quilt_extract_array_b2(self) -> None:

        f1 = ff.parse('s(4,20)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=5, axis=1, retain_labels=True, deepcopy_from_bus=True)

        a1 = q1._extract_array(None, 2)
        self.assertEqual(a1.tolist(), [-3648, 91301, 30205, 54020])

        self.assertEqual(q1._extract_array(2).tolist(), [84967, 5729, 30205, 166924, 170440, 175579, 32395, 172142, -31776, -97851, -12447, 119909, 172142, 35684, 170440, 316, 81275, 81275, 96640, -110091])

        self.assertEqual(q1._extract_array(slice(0,2), 2).tolist(), [-3648, 91301])

        self.assertEqual(q1._extract_array(slice(1,3), slice(4,6)).tolist(),
                [[146284, 13448], [170440, 175579]])

        self.assertEqual(q1._extract_array([0, 3], [1, 14, 15]).tolist(),
                [[162197, 58768, 10240], [-168387, 32395, -170415]]
                )

        self.assertEqual(q1._extract_array(2, 2), 30205)
        self.assertEqual(q1._extract_array(-1, -1), -112188)

    def test_quilt_extract_array_b3(self) -> None:

        f1 = ff.parse('s(4,4)|v(int)|i(I,str)|c(I,str)')

        q1 = Quilt.from_frame(f1, chunksize=4, axis=1, retain_labels=False, deepcopy_from_bus=False)
        a1_id_in_bus = id(q1._bus._series.values[0].values)
        a1_id_via_quilt = id(q1._extract_array())
        self.assertEqual(a1_id_in_bus, a1_id_via_quilt)

        q2 = Quilt.from_frame(f1, chunksize=4, axis=1, retain_labels=False, deepcopy_from_bus=True)
        a2_id_in_bus = id(q2._bus._series.values[0].values)
        a2_id_via_quilt = id(q2._extract_array())
        self.assertNotEqual(a2_id_in_bus, a2_id_via_quilt)



    #---------------------------------------------------------------------------
    def test_quilt_retain_labels_a(self) -> None:

        dc = DisplayConfig(type_show=False)

        f1 = ff.parse('s(10,4)|v(int)|i(I,str)|c(I,str)').rename('foo')
        q1 = Quilt.from_frame(f1, chunksize=2, retain_labels=False)
        self.assertEqual(q1.index.depth, 1)
        f2 = q1.loc['zkuW':'z2Oo'] #type: ignore
        self.assertEqual(f2.index.depth, 1)
        self.assertEqual(f2.to_pairs(0),
                (('zZbu', (('zkuW', 13448), ('zmVj', 175579), ('z2Oo', 58768))), ('ztsv', (('zkuW', -168387), ('zmVj', 140627), ('z2Oo', 66269))), ('zUvW', (('zkuW', 54020), ('zmVj', 129017), ('z2Oo', 35021))), ('zkuW', (('zkuW', 122246), ('zmVj', 197228), ('z2Oo', 105269))))
                )

        q2 = Quilt.from_frame(f1, chunksize=2, retain_labels=True)
        self.assertEqual(q2.index.depth, 2)

        f3 = q2.loc[HLoc['zUvW':'z5l6']] #type: ignore
        self.assertEqual(f3.index.depth, 2)
        self.assertEqual(f3.to_pairs(0),
                (('zZbu', ((('zUvW', 'zUvW'), 84967), (('zUvW', 'zkuW'), 13448), (('zmVj', 'zmVj'), 175579), (('zmVj', 'z2Oo'), 58768), (('z5l6', 'z5l6'), 146284), (('z5l6', 'zCE3'), 170440))), ('ztsv', ((('zUvW', 'zUvW'), 5729), (('zUvW', 'zkuW'), -168387), (('zmVj', 'zmVj'), 140627), (('zmVj', 'z2Oo'), 66269), (('z5l6', 'z5l6'), -171231), (('z5l6', 'zCE3'), -38997))), ('zUvW', ((('zUvW', 'zUvW'), 30205), (('zUvW', 'zkuW'), 54020), (('zmVj', 'zmVj'), 129017), (('zmVj', 'z2Oo'), 35021), (('z5l6', 'z5l6'), 166924), (('z5l6', 'zCE3'), 122246))), ('zkuW', ((('zUvW', 'zUvW'), 166924), (('zUvW', 'zkuW'), 122246), (('zmVj', 'zmVj'), 197228), (('zmVj', 'z2Oo'), 105269), (('z5l6', 'z5l6'), 119909), (('z5l6', 'zCE3'), 194224))))
                )


    #---------------------------------------------------------------------------
    def test_quilt_items_store_a(self) -> None:

        f1 = ff.parse('s(10,4)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=2, retain_labels=False)

        self.assertEqual(len(tuple(q1._items_store())), 5)

    #---------------------------------------------------------------------------
    def test_quilt_keys_a(self) -> None:

        f1 = ff.parse('s(4,4)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=2, axis=1, retain_labels=False)
        self.assertEqual(list(q1.keys()), ['zZbu', 'ztsv', 'zUvW', 'zkuW'])

    def test_quilt_iter_a(self) -> None:

        f1 = ff.parse('s(4,4)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=2, axis=1, retain_labels=False)
        self.assertEqual(list(q1), ['zZbu', 'ztsv', 'zUvW', 'zkuW']) #type: ignore

    def test_quilt_contains_a(self) -> None:

        f1 = ff.parse('s(4,4)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=2, axis=1, retain_labels=False)
        self.assertTrue('zZbu' in q1)

    def test_quilt_get_a(self) -> None:

        f1 = ff.parse('s(4,4)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=2, axis=1, retain_labels=False)
        self.assertEqual(q1.get('zZbu').shape, (4,)) #type: ignore
        self.assertEqual(q1.get(''), None)


    #---------------------------------------------------------------------------
    def test_quilt_store_client_mixin_a(self) -> None:

        # indexes are heterogenous but columns are not
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(a=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f1, f2, f3))

        with temp_file('.zip') as fp:

            b1.to_zip_pickle(fp)

            q1 = Quilt.from_zip_pickle(fp, max_persist=1, retain_labels=True)

            self.assertEqual(q1.loc[:, :].to_pairs(0),
                    (('a', ((('f1', 'x'), 1), (('f1', 'y'), 2), (('f2', 'x'), 1), (('f2', 'y'), 2), (('f2', 'z'), 3), (('f3', 'p'), 10), (('f3', 'q'), 20))), ('b', ((('f1', 'x'), 3), (('f1', 'y'), 4), (('f2', 'x'), 4), (('f2', 'y'), 5), (('f2', 'z'), 6), (('f3', 'p'), 50), (('f3', 'q'), 60)))))


    #---------------------------------------------------------------------------

    def test_quilt_iter_array_a1(self) -> None:

        # indexes are hetergenous but columns are not
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(a=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f1, f2, f3))
        q1 = Quilt(b1, retain_labels=True)

        arrays = tuple(q1.iter_array(1))
        self.assertEqual(len(arrays), 7)
        self.assertEqual(arrays[0].tolist(), [1, 3])
        self.assertEqual(arrays[-1].tolist(), [20, 60])

        arrays = tuple(q1.iter_array_items(1))
        self.assertEqual(len(arrays), 7)
        self.assertEqual(arrays[0][0], ('f1', 'x'))
        self.assertEqual(arrays[0][1].tolist(), [1, 3])

        self.assertEqual(arrays[-1][0], ('f3', 'q'))
        self.assertEqual(arrays[-1][1].tolist(), [20, 60])

        with self.assertRaises(NotImplementedError):
            _ = tuple(q1.iter_array(0))

        with self.assertRaises(NotImplementedError):
            _ = tuple(q1.iter_array_items(0))


    def test_quilt_iter_array_a2(self) -> None:

        # indexes are hetergenous but columns are not
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(a=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f1, f2, f3))
        q1 = Quilt(b1, retain_labels=True, deepcopy_from_bus=True)

        arrays = tuple(q1.iter_array(1))
        self.assertEqual(len(arrays), 7)
        self.assertEqual(arrays[0].tolist(), [1, 3])
        self.assertEqual(arrays[-1].tolist(), [20, 60])

        arrays = tuple(q1.iter_array_items(1))
        self.assertEqual(len(arrays), 7)
        self.assertEqual(arrays[0][0], ('f1', 'x'))
        self.assertEqual(arrays[0][1].tolist(), [1, 3])

        self.assertEqual(arrays[-1][0], ('f3', 'q'))
        self.assertEqual(arrays[-1][1].tolist(), [20, 60])

        with self.assertRaises(NotImplementedError):
            _ = tuple(q1.iter_array(0))

        with self.assertRaises(NotImplementedError):
            _ = tuple(q1.iter_array_items(0))



    def test_quilt_iter_array_b1(self) -> None:

        f1 = ff.parse('s(2,6)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=2, axis=1, retain_labels=False)

        arrays = tuple(q1.iter_array_items(0))
        self.assertEqual(len(arrays), 6)
        self.assertEqual(arrays[0][0], 'zZbu')
        self.assertEqual(arrays[0][1].tolist(), [-88017, 92867])

        self.assertEqual(len(arrays), 6)
        self.assertEqual(arrays[-1][0], 'z2Oo')
        self.assertEqual(arrays[-1][1].tolist(), [84967, 13448])

        with self.assertRaises(NotImplementedError):
            _ = tuple(q1.iter_array(1))

        with self.assertRaises(NotImplementedError):
            _ = tuple(q1.iter_array_items(1))



    def test_quilt_iter_array_b2(self) -> None:

        f1 = ff.parse('s(2,6)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=2, axis=1, retain_labels=False, deepcopy_from_bus=True)

        arrays = tuple(q1.iter_array_items(0))
        self.assertEqual(len(arrays), 6)
        self.assertEqual(arrays[0][0], 'zZbu')
        self.assertEqual(arrays[0][1].tolist(), [-88017, 92867])

        self.assertEqual(len(arrays), 6)
        self.assertEqual(arrays[-1][0], 'z2Oo')
        self.assertEqual(arrays[-1][1].tolist(), [84967, 13448])

        with self.assertRaises(NotImplementedError):
            _ = tuple(q1.iter_array(1))

        with self.assertRaises(NotImplementedError):
            _ = tuple(q1.iter_array_items(1))


    def test_quilt_iter_array_b3(self) -> None:

        f1 = ff.parse('s(2,6)|v(int,str)|i(I,str)|c(I,str)')

        q2 = Quilt.from_frame(f1, chunksize=3, axis=1, retain_labels=False, deepcopy_from_bus=False)
        a2_src_id = id(q2._bus._series.values[0]._extract_array(None, 0))
        a2_dst_id = id(next(iter(q2.iter_array(0))))
        self.assertTrue(a2_src_id == a2_dst_id)

        q1 = Quilt.from_frame(f1, chunksize=3, axis=1, retain_labels=False, deepcopy_from_bus=True)
        a1_src_id = id(q1._bus._series.values[0]._extract_array(None, 0))
        a1_dst_id = id(next(iter(q1.iter_array(0))))
        self.assertTrue(a1_src_id != a1_dst_id)

    #---------------------------------------------------------------------------

    def test_quilt_iter_series_a1(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(a=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f1, f2, f3))
        q1 = Quilt(b1, retain_labels=True)

        series = tuple(q1.iter_series(axis=1))
        s1 = series[0]
        self.assertEqual(s1.name, ('f1', 'x'))
        self.assertEqual(s1.to_pairs(), (('a', 1), ('b', 3)))
        s2 = series[-1]
        self.assertEqual(s2.name, ('f3', 'q'))
        self.assertEqual(s2.to_pairs(), (('a', 20), ('b', 60)))

        with self.assertRaises(NotImplementedError):
            _ = tuple(q1.iter_series(0))

        with self.assertRaises(NotImplementedError):
            _ = tuple(q1.iter_series_items(0))

        with self.assertRaises(NotImplementedError):
            _ = tuple(q1.items())

    def test_quilt_iter_series_a2(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(a=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f1, f2, f3))
        q1 = Quilt(b1, retain_labels=True, deepcopy_from_bus=True)

        series = tuple(q1.iter_series(axis=1))
        s1 = series[0]
        self.assertEqual(s1.name, ('f1', 'x'))
        self.assertEqual(s1.to_pairs(), (('a', 1), ('b', 3)))
        s2 = series[-1]
        self.assertEqual(s2.name, ('f3', 'q'))
        self.assertEqual(s2.to_pairs(), (('a', 20), ('b', 60)))

        with self.assertRaises(NotImplementedError):
            _ = tuple(q1.iter_series(0))

        with self.assertRaises(NotImplementedError):
            _ = tuple(q1.iter_series_items(0))

        with self.assertRaises(NotImplementedError):
            _ = tuple(q1.items())


    #---------------------------------------------------------------------------
    def test_quilt_iter_tuple_a1(self) -> None:

        f1 = ff.parse('s(2,6)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=2, axis=1, retain_labels=False)
        post = tuple(q1.iter_tuple()) # iter columns
        self.assertEqual(len(post), 6)
        self.assertEqual(post[0]._fields, ('zZbu', 'ztsv'))

        with self.assertRaises(NotImplementedError):
            _ = tuple(q1.iter_tuple(1))

    def test_quilt_iter_tuple_a2(self) -> None:

        f1 = ff.parse('s(2,6)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=2, axis=1, retain_labels=False, deepcopy_from_bus=True)
        post = tuple(q1.iter_tuple()) # iter columns
        self.assertEqual(len(post), 6)
        self.assertEqual(post[0]._fields, ('zZbu', 'ztsv'))

        with self.assertRaises(NotImplementedError):
            _ = tuple(q1.iter_tuple(1))


    def test_quilt_iter_tuple_b1(self) -> None:

        f1 = ff.parse('s(2,6)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=2, axis=1, retain_labels=False)
        post = tuple(q1.iter_tuple_items()) # iter columns
        self.assertEqual(len(post), 6)
        self.assertEqual(post[0][0], 'zZbu')
        self.assertEqual(post[0][1]._fields, ('zZbu', 'ztsv'))

        with self.assertRaises(NotImplementedError):
            _ = tuple(q1.iter_tuple_items(1))

    def test_quilt_iter_tuple_b2(self) -> None:

        f1 = ff.parse('s(2,6)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=2, axis=1, retain_labels=False, deepcopy_from_bus=True)
        post = tuple(q1.iter_tuple_items()) # iter columns
        self.assertEqual(len(post), 6)
        self.assertEqual(post[0][0], 'zZbu')
        self.assertEqual(post[0][1]._fields, ('zZbu', 'ztsv'))

        with self.assertRaises(NotImplementedError):
            _ = tuple(q1.iter_tuple_items(1))

    #---------------------------------------------------------------------------
    def test_quilt_to_frame_a1(self) -> None:

        f1 = ff.parse('s(4,4)|v(int)|i(I,str)|c(I,str)')

        q1 = Quilt.from_frame(f1, chunksize=2, axis=1, retain_labels=False)
        self.assertTrue(q1.to_frame().equals(f1))

        q2 = Quilt.from_frame(f1, chunksize=2, axis=0, retain_labels=False)
        self.assertTrue(q2.to_frame().equals(f1))

    def test_quilt_to_frame_a2(self) -> None:

        f1 = ff.parse('s(4,4)|v(int)|i(I,str)|c(I,str)')

        q1 = Quilt.from_frame(f1, chunksize=2, axis=1, retain_labels=False, deepcopy_from_bus=True)
        self.assertTrue(q1.to_frame().equals(f1))

        q2 = Quilt.from_frame(f1, chunksize=2, axis=0, retain_labels=False, deepcopy_from_bus=True)
        self.assertTrue(q2.to_frame().equals(f1))

    #---------------------------------------------------------------------------
    def test_quilt_iter_window_a1(self) -> None:

        f1 = ff.parse('s(20,2)|v(int)|i(I,str)|c(I,str)')

        q1 = Quilt.from_frame(f1, chunksize=4, axis=0, retain_labels=False)
        self.assertTrue(len(q1._bus), 5)

        post1 = tuple(q1.iter_window(size=5))
        self.assertEqual(len(post1), 16)
        self.assertEqual(post1[-1].shape, (5, 2))

        f2 = Batch(q1.iter_window_items(size=5)).mean().to_frame()
        self.assertEqual(f2.to_pairs(0),
                (('zZbu', (('zmVj', 55768.8), ('z2Oo', 85125.8), ('z5l6', 95809.2), ('zCE3', 112903.8), ('zr4u', 116693.2), ('zYVB', 109129.2), ('zOyq', 84782.8), ('zIA5', 89954.4), ('zGDJ', 24929.2), ('zmhG', 43655.2), ('zo2Q', 28049.0), ('zjZQ', 21071.6), ('zO5l', 20315.6), ('zEdH', 77254.8), ('zB7E', 21935.2), ('zwIp', -21497.8))), ('ztsv', (('zmVj', 19801.8), ('z2Oo', 616.2), ('z5l6', -25398.6), ('zCE3', -34343.8), ('zr4u', 2873.2), ('zYVB', -30222.0), ('zOyq', -49512.4), ('zIA5', -3803.0), ('zGDJ', -15530.0), ('zmhG', 19152.0), ('zo2Q', 59952.2), ('zjZQ', 63255.8), ('zO5l', 57918.2), ('zEdH', 93699.6), ('zB7E', 56150.2), ('zwIp', 17830.4))))
                )

    def test_quilt_iter_window_a2(self) -> None:

        f1 = ff.parse('s(20,2)|v(int)|i(I,str)|c(I,str)')

        q1 = Quilt.from_frame(f1, chunksize=4, axis=0, retain_labels=False, deepcopy_from_bus=True)
        self.assertTrue(len(q1._bus), 5)

        post1 = tuple(q1.iter_window(size=5))
        self.assertEqual(len(post1), 16)
        self.assertEqual(post1[-1].shape, (5, 2))

        f2 = Batch(q1.iter_window_items(size=5)).mean().to_frame()
        self.assertEqual(f2.to_pairs(0),
                (('zZbu', (('zmVj', 55768.8), ('z2Oo', 85125.8), ('z5l6', 95809.2), ('zCE3', 112903.8), ('zr4u', 116693.2), ('zYVB', 109129.2), ('zOyq', 84782.8), ('zIA5', 89954.4), ('zGDJ', 24929.2), ('zmhG', 43655.2), ('zo2Q', 28049.0), ('zjZQ', 21071.6), ('zO5l', 20315.6), ('zEdH', 77254.8), ('zB7E', 21935.2), ('zwIp', -21497.8))), ('ztsv', (('zmVj', 19801.8), ('z2Oo', 616.2), ('z5l6', -25398.6), ('zCE3', -34343.8), ('zr4u', 2873.2), ('zYVB', -30222.0), ('zOyq', -49512.4), ('zIA5', -3803.0), ('zGDJ', -15530.0), ('zmhG', 19152.0), ('zo2Q', 59952.2), ('zjZQ', 63255.8), ('zO5l', 57918.2), ('zEdH', 93699.6), ('zB7E', 56150.2), ('zwIp', 17830.4))))
                )

    def test_quilt_iter_window_b1(self) -> None:
        from string import ascii_lowercase
        # indexes are heterogenous but columns are not
        def get_frame(scale: int = 1) -> Frame:
            return Frame(np.arange(12).reshape(4, 3) * scale, columns=('x', 'y', 'z'))

        config = StoreConfig(include_index=True, index_depth=1)
        with temp_file('.zip') as fp:

            items = ((ascii_lowercase[i], get_frame(scale=i)) for i in range(20))
            Batch(items).to_zip_parquet(fp, config=config)

            # aggregate index is not unique so must retain outer labels
            q1 = Quilt.from_zip_parquet(fp, max_persist=1, retain_labels=True, config=config)
            self.assertEqual(q1.status['loaded'].sum(), 0)

            s1 = q1['y'] # extract and consolidate a column
            self.assertEqual(s1.shape, (80,))
            self.assertEqual(q1.status['loaded'].sum(), 1)

            # extract a region using a loc selection
            s2 = q1.loc[HLoc['h':'m'], ['x', 'z']].sum() #type: ignore
            self.assertEqual(s2.to_pairs(), (('x', 1026), ('z', 1482)))

            # iterate over all rows and apply a function
            s3 = q1.iter_series(axis=1).apply(lambda s: s.mean())
            self.assertEqual(s3.shape, (80,))
            self.assertEqual(q1.status['loaded'].sum(), 1)

            # take a rolling mean of size six
            f1 = Batch(q1.iter_window_items(size=6)).mean().to_frame()
            self.assertEqual(f1.shape, (75, 3))
            self.assertEqual(q1.status['loaded'].sum(), 1)



    def test_quilt_iter_window_b2(self) -> None:
        from string import ascii_lowercase
        # indexes are heterogenous but columns are not
        def get_frame(scale: int = 1) -> Frame:
            return Frame(np.arange(12).reshape(4, 3) * scale, columns=('x', 'y', 'z'))

        config = StoreConfig(include_index=True, index_depth=1)
        with temp_file('.zip') as fp:

            items = ((ascii_lowercase[i], get_frame(scale=i)) for i in range(20))
            Batch(items).to_zip_parquet(fp, config=config)

            # aggregate index is not unique so must retain outer labels
            q1 = Quilt.from_zip_parquet(fp,
                    max_persist=1,
                    retain_labels=True,
                    config=config,
                    deepcopy_from_bus=True,
                    )
            self.assertEqual(q1.status['loaded'].sum(), 0)

            s1 = q1['y'] # extract and consolidate a column
            self.assertEqual(s1.shape, (80,))
            self.assertEqual(q1.status['loaded'].sum(), 1)

            # extract a region using a loc selection
            s2 = q1.loc[HLoc['h':'m'], ['x', 'z']].sum() #type: ignore
            self.assertEqual(s2.to_pairs(), (('x', 1026), ('z', 1482)))

            # iterate over all rows and apply a function
            s3 = q1.iter_series(axis=1).apply(lambda s: s.mean())
            self.assertEqual(s3.shape, (80,))
            self.assertEqual(q1.status['loaded'].sum(), 1)

            # take a rolling mean of size six
            f1 = Batch(q1.iter_window_items(size=6)).mean().to_frame()
            self.assertEqual(f1.shape, (75, 3))
            self.assertEqual(q1.status['loaded'].sum(), 1)

    #---------------------------------------------------------------------------
    def test_quilt_iter_window_array_b1(self) -> None:

        f1 = ff.parse('s(20,2)|v(int)|i(I,str)|c(I,str)')

        q1 = Quilt.from_frame(f1, chunksize=4, axis=0, retain_labels=False)
        self.assertTrue(len(q1._bus), 5)

        s1 = q1.iter_window_array(size=5, step=4).apply(lambda a: a.sum())
        self.assertEqual(s1.to_pairs(),
                (('zmVj', 377853), ('zr4u', 597832), ('zGDJ', 46996), ('zO5l', 391169)))

    def test_quilt_iter_window_array_b2(self) -> None:

        f1 = ff.parse('s(20,2)|v(int)|i(I,str)|c(I,str)')

        q1 = Quilt.from_frame(f1, chunksize=4, axis=0, retain_labels=False, deepcopy_from_bus=True)
        self.assertTrue(len(q1._bus), 5)

        s1 = q1.iter_window_array(size=5, step=4).apply(lambda a: a.sum())
        self.assertEqual(s1.to_pairs(),
                (('zmVj', 377853), ('zr4u', 597832), ('zGDJ', 46996), ('zO5l', 391169)))



if __name__ == '__main__':
    unittest.main()


