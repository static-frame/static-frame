
import unittest

import frame_fixtures as ff

from static_frame.test.test_case import TestCase
from static_frame.core.quilt import Quilt
from static_frame.core.quilt import AxisMap
from static_frame.core.index import Index
from static_frame.core.hloc import HLoc
from static_frame.core.display_config import DisplayConfig
from static_frame.core.index import ILoc
from static_frame.core.frame import Frame
from static_frame.core.bus import Bus
from static_frame.test.test_case import temp_file


class TestUnit(TestCase):

    def test_axis_map_a(self) -> None:

        components = dict(
                x=Index(('a', 'b', 'c')),
                y=Index(('a', 'b', 'c')),
                )

        am = AxisMap.from_tree(components) #type: ignore
        self.assertEqual(am.to_pairs(),
                ((('x', 'a'), 'x'), (('x', 'b'), 'x'), (('x', 'c'), 'x'), (('y', 'a'), 'y'), (('y', 'b'), 'y'), (('y', 'c'), 'y')))

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

        # import ipdb; ipdb.set_trace()

        self.assertEqual(q1.name, 'foo')
        self.assertEqual(q1.rename('bar').name, 'bar')
        self.assertTrue(repr(q1).startswith('<Quilt: foo'))

        post = AxisMap.from_bus(q1._bus, q1._axis)
        self.assertEqual(len(post), 100)

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

        post = AxisMap.from_bus(q1._bus, q1._axis)
        self.assertEqual(len(post), 100)


    def test_quilt_from_frame_c(self) -> None:

        f1 = ff.parse('s(100,4)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=10, axis=1, retain_labels=False)
        self.assertEqual(q1.shape, (100, 4))
        self.assertEqual(len(q1._bus), 1)


    #---------------------------------------------------------------------------

    def test_quilt_extract_a(self) -> None:

        f1 = ff.parse('s(4,100)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=10, axis=1, retain_labels=False)
        self.assertEqual(q1.shape, (4, 100))
        self.assertEqual(len(q1._bus), 10)
        self.assertEqual(q1['zkuW':'zTSt'].shape, (4, 95)) #type: ignore
        self.assertEqual(q1.loc[ILoc[-2:], 'zaji': 'zsa5'].shape, (2, 17)) #type: ignore


    def test_quilt_extract_b(self) -> None:

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
        self.assertEqual(s1.to_pairs(), #type: ignore
                ((('zkuW', 'zkuW'), 166924), (('zkuW', 'zmVj'), 170440), (('zkuW', 'z2Oo'), 175579), (('z5l6', 'z5l6'), 32395), (('z5l6', 'zCE3'), 172142), (('z5l6', 'zr4u'), -31776)))

        s2 = q1.loc[:, ('z5l6', 'z5l6')]
        self.assertEqual(s2.shape, (4,))
        self.assertEqual(s2.name, ('z5l6', 'z5l6'))
        self.assertEqual(s2.to_pairs(), #type: ignore
                (('zZbu', 146284), ('ztsv', 170440), ('zUvW', 32395), ('zkuW', 137759))
                )


    def test_quilt_extract_c(self) -> None:

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


    def test_quilt_extract_d(self) -> None:

        f1 = ff.parse('s(4,20)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=5, axis=1, retain_labels=True)
        self.assertEqual(q1.shape, (4, 20))
        self.assertEqual(len(q1._bus), 4)

        self.assertEqual(q1.loc['zUvW'].shape, (20,))
        self.assertEqual(q1.loc['zUvW'].name, 'zUvW')


    def test_quilt_extract_e(self) -> None:

        f1 = ff.parse('s(20,4)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=5, axis=0, retain_labels=False)

        self.assertEqual(q1.loc['zO5l', 'zkuW'], 146284)

        f2 = q1.loc[['zmVj', 'zOyq', 'zmhG', 'zwIp']]
        self.assertTrue(f2.equals(f1.loc[['zmVj', 'zOyq', 'zmhG', 'zwIp']]))

        s1 = q1.loc[['zmVj', 'zOyq', 'zmhG', 'zwIp'], 'zkuW']
        self.assertTrue(s1.equals(f1.loc[['zmVj', 'zOyq', 'zmhG', 'zwIp'], 'zkuW']))
        self.assertEqual(s1.name, 'zkuW')
        self.assertEqual(s1.shape, (4,))


    def test_quilt_extract_f(self) -> None:

        f1 = ff.parse('s(4,20)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=5, axis=1, retain_labels=False)

        f2 = q1.loc[:, ['zUvW', 'zB7E', 'zwIp']]
        self.assertTrue(f2.equals(f1.loc[:, ['zUvW', 'zB7E', 'zwIp']]))

        self.assertEqual(q1.loc['zkuW', 'zwIp'], -112188)

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
        # import ipdb; ipdb.set_trace()
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
        self.assertEqual(list(q1), ['zZbu', 'ztsv', 'zUvW', 'zkuW'])

    def test_quilt_contains_a(self) -> None:

        f1 = ff.parse('s(4,4)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=2, axis=1, retain_labels=False)
        self.assertTrue('zZbu' in q1)

    def test_quilt_get_a(self) -> None:

        f1 = ff.parse('s(4,4)|v(int)|i(I,str)|c(I,str)')
        q1 = Quilt.from_frame(f1, chunksize=2, axis=1, retain_labels=False)
        self.assertEqual(q1.get('zZbu').shape, (4,))
        self.assertEqual(q1.get(''), None)


    #---------------------------------------------------------------------------
    def test_quilt_store_client_mixin_a(self) -> None:

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

        with temp_file('.zip') as fp:

            b1.to_zip_pickle(fp)

            q1 = Quilt.from_zip_pickle(fp, max_persist=1, retain_labels=True)

            self.assertEqual(q1.loc[:, :].to_pairs(0),
                    (('a', ((('f1', 'x'), 1), (('f1', 'y'), 2), (('f2', 'x'), 1), (('f2', 'y'), 2), (('f2', 'z'), 3), (('f3', 'p'), 10), (('f3', 'q'), 20))), ('b', ((('f1', 'x'), 3), (('f1', 'y'), 4), (('f2', 'x'), 4), (('f2', 'y'), 5), (('f2', 'z'), 6), (('f3', 'p'), 50), (('f3', 'q'), 60)))))


    #---------------------------------------------------------------------------

    def test_quilt_iter_array_a(self) -> None:

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


    def test_quilt_iter_array_b(self) -> None:

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


if __name__ == '__main__':
    unittest.main()


