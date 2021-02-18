import unittest
from datetime import date
from datetime import datetime
# from io import StringIO
import typing as tp
import numpy as np

from static_frame.core.frame import Frame
from static_frame.core.bus import Bus
from static_frame.core.bus import FrameDeferred

from static_frame.core.series import Series
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.store_zip import StoreZipTSV

from static_frame.core.store import StoreConfigMap
from static_frame.core.store import StoreConfig
from static_frame.core.display_config import DisplayConfig
from static_frame.core.hloc import HLoc

from static_frame.test.test_case import TestCase
from static_frame.test.test_case import temp_file
from static_frame.test.test_case import skip_win


from static_frame.core.exception import ErrorInitBus
from static_frame.core.exception import StoreFileMutation


class TestUnit(TestCase):

    def test_frame_deferred_a(self) -> None:

        self.assertEqual(str(FrameDeferred), '<FrameDeferred>')

    def test_bus_slotted_a(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='foo')

        b1 = Bus.from_frames((f1,))

        with self.assertRaises(AttributeError):
            b1.g = 30 # type: ignore #pylint: disable=E0237
        with self.assertRaises(AttributeError):
            b1.__dict__ #pylint: disable=W0104

    #---------------------------------------------------------------------------
    def test_bus_init_a(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='foo')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='bar')

        config = StoreConfigMap.from_config(StoreConfig(index_depth=1))
        b1 = Bus.from_frames((f1, f2), config=config)

        self.assertEqual(b1.keys().values.tolist(),
                ['foo', 'bar'])


        with temp_file('.zip') as fp:
            b1.to_zip_tsv(fp)
            b2 = Bus.from_zip_tsv(fp)

            f3 = b2['bar']
            f4 = b2['foo']
            # import ipdb; ipdb.set_trace()
            zs = StoreZipTSV(fp)
            zs.write(b1.items())

            # how to show that this derived getitem has derived type?
            f3 = zs.read('foo', config=config['foo'])
            self.assertEqual(
                f3.to_pairs(0),
                (('a', (('x', 1), ('y', 2))), ('b', (('x', 3), ('y', 4))))
            )

    def test_bus_init_b(self) -> None:

        with self.assertRaises(ErrorInitBus):
            Bus(Series([1, 2, 3]))

        with self.assertRaises(ErrorInitBus):
            Bus(Series([3, 4], dtype=object))

        with self.assertRaises(ErrorInitBus):
            Bus(Series([3, 4], index=('a', 'b'), dtype=object))


    def test_bus_init_c(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='foo')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='bar')

        config = StoreConfigMap.from_config(StoreConfig(index_depth=1))
        b1 = Bus.from_frames((f1, f2), config=config)

        self.assertEqual(b1.keys().values.tolist(),
                ['foo', 'bar'])

        with temp_file('.zip') as fp:
            b1.to_zip_csv(fp)
            b2 = Bus.from_zip_csv(fp, config=config)

            f1_loaded = b2['foo']
            f2_loaded = b2['bar']

            self.assertEqualFrames(f1, f1_loaded)
            self.assertEqualFrames(f2, f2_loaded)

    #---------------------------------------------------------------------------
    def test_bus_rename_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')

        b1 = Bus.from_frames((f1, f2,), name='foo')
        self.assertEqual(b1.name, 'foo')
        b2 = b1.rename('bar')
        self.assertTrue(b2.__class__ is Bus)
        self.assertEqual(b2.name, 'bar')

    #---------------------------------------------------------------------------
    def test_bus_from_items_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                )
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                )

        b1 = Bus.from_items((('a', f1), ('b', f2)))
        self.assertEqual(b1.index.values.tolist(), ['a', 'b'])


    #---------------------------------------------------------------------------
    def test_bus_shapes_a(self) -> None:
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

            b2 = Bus.from_zip_pickle(fp)

            f2_loaded = b2['f2']

            self.assertEqual(b2.shapes.to_pairs(),
                    (('f1', None), ('f2', (3, 2)), ('f3', None)))

            f3_loaded = b2['f3']

            self.assertEqual(b2.shapes.to_pairs(),
                    (('f1', None), ('f2', (3, 2)), ('f3', (2, 2 )))
                    )

    @skip_win # type: ignore
    def test_bus_nbytes_a(self) -> None:
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
            b2 = Bus.from_zip_pickle(fp)

            f2_loaded = b2['f2']

            self.assertEqual(b2.nbytes, 48)

            f3_loaded = b2['f3']

            self.assertEqual(b2.nbytes, 80)

            f1_loaded = b2['f1']

            self.assertEqual(b2.nbytes, 112)


    @skip_win # type: ignore
    def test_bus_dtypes_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f1, f2, f3))

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp)

            self.assertEqual(b2.dtypes.to_pairs(0), ())

            f2_loaded = b2['f2']

            self.assertEqual(b2.dtypes.to_pairs(0),
                    (('c', (('f1', None), ('f2', np.dtype('int64')), ('f3', None))), ('b', (('f1', None), ('f2', np.dtype('int64')), ('f3', None))))
            )

            f3_loaded = b2['f3']

            self.assertEqual(b2.dtypes.to_pairs(0),
                    (('b', (('f1', None), ('f2', np.dtype('int64')), ('f3', np.dtype('int64')))), ('c', (('f1', None), ('f2', np.dtype('int64')), ('f3', None))), ('d', (('f1', None), ('f2', None), ('f3', np.dtype('int64')))))
                    )


    @skip_win # type: ignore
    def test_bus_status_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f1, f2, f3))

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp)

            status = b2.status
            self.assertEqual(status.shape, (3, 4))
            # force load all
            tuple(b2.items())

            self.assertEqual(
                    b2.status.to_pairs(0),                                                           (('loaded', (('f1', True), ('f2', True), ('f3', True))), ('size', (('f1', 4.0), ('f2', 6.0), ('f3', 4.0))), ('nbytes', (('f1', 32.0), ('f2', 48.0), ('f3', 32.0))),('shape', (('f1', (2, 2)), ('f2', (3, 2)), ('f3', (2, 2)))))
            )


    def test_bus_keys_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')
        f4 = Frame.from_dict(
                dict(q=(None,None), r=(np.nan,np.nan)),
                index=(1000, 1001),
                name='f4')

        b1 = Bus.from_frames((f1, f2, f3, f4))

        self.assertEqual(b1.keys().values.tolist(), ['f1', 'f2', 'f3', 'f4'])
        self.assertEqual(b1.values[2].name, 'f3')

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp)
            self.assertFalse(b2._loaded_all)

            self.assertEqual(b2.keys().values.tolist(), ['f1', 'f2', 'f3', 'f4'])
            self.assertFalse(b2._loaded.any())
            # accessing values forces loading all
            self.assertEqual(b2.values[2].name, 'f3')
            self.assertTrue(b2._loaded_all)


    def test_bus_reversed_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f1, f2, f3))
        self.assertEqual(list(reversed(b1)), ['f3', 'f2', 'f1'])


    def test_bus_display_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f1, f2, f3))
        self.assertEqual(
                b1.display(config=DisplayConfig(type_color=False)).to_rows(),
                ['<Bus>',
                '<Index>',
                'f1      Frame',
                'f2      Frame',
                'f3      Frame',
                '<<U2>   <object>'])

        rows1 = b1.display(config=DisplayConfig(
                type_color=False,
                type_show=False)).to_rows()
        self.assertEqual(rows1, ['f1 Frame', 'f2 Frame', 'f3 Frame'])

        rows2 = b1.display(config=DisplayConfig(
                type_color=False,
                type_show=False,
                include_index=False)).to_rows()
        self.assertEqual(rows2, ['Frame', 'Frame', 'Frame'])



    #---------------------------------------------------------------------------
    def test_bus_iloc_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f1, f2, f3))

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp)

            self.assertEqual(
                    b2.iloc[[0, 2]].status['loaded'].to_pairs(),
                    (('f1', True), ('f3', True))
                    )

    def test_bus_iloc_b(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')

        b1 = Bus.from_frames((f1,))
        f2 = b1.iloc[0]
        self.assertTrue(f1 is f2)

    def test_bus_loc_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')

        b1 = Bus.from_frames((f1,))
        f2 = b1.loc['f1']
        self.assertTrue(f1 is f2)

    def test_bus_loc_b(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f1, f2, f3))
        b2 = b1.loc['f2':] #type: ignore
        self.assertEqual(len(b2), 2)
        self.assertEqual(b2.index.values.tolist(), ['f2', 'f3'])


    def test_bus_getitem_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f1, f2, f3))

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp)

            self.assertEqual(b2['f2':].status['loaded'].to_pairs(), #type: ignore
                    (('f2', True), ('f3', True))
                    )

    #---------------------------------------------------------------------------
    def test_bus_to_xlsx_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        config = StoreConfigMap.from_config(
                StoreConfig(
                        index_depth=1,
                        columns_depth=1,
                        include_columns=True,
                        include_index=True
                        ))
        b1 = Bus.from_frames((f1, f2, f3), config=config)

        with temp_file('.xlsx') as fp:
            b1.to_xlsx(fp)

            b2 = Bus.from_xlsx(fp, config=config)
            tuple(b2.items()) # force loading all

        for frame in (f1, f2, f3):
            self.assertEqualFrames(frame, b2[frame.name])


    def test_bus_to_xlsx_b(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2,3)),
                index=('x', 'y', 'z'),
                name='f1')
        f2 = Frame.from_dict(
                dict(A=(10,20,30)),
                index=('q', 'r', 's'),
                name='f2')

        config = StoreConfig(include_index=True, index_depth=1)
        b1 = Bus.from_frames((f1, f2), config=config)

        with temp_file('.xlsx') as fp:
            b1.to_xlsx(fp)

            b2 = Bus.from_xlsx(fp, config=config)
            tuple(b2.items()) # force loading all

        for frame in (f1, f2):
            self.assertEqualFrames(frame, b2[frame.name])



    def test_bus_to_xlsx_c(self) -> None:
        '''
        Test manipulating a file behind the Bus.
        '''
        f1 = Frame.from_dict(
                dict(a=(1,2,3)),
                index=('x', 'y', 'z'),
                name='f1')

        f2 = Frame.from_dict(
                dict(x=(10,20,30)),
                index=('q', 'r', 's'),
                name='f2')

        b1 = Bus.from_frames((f1,),)

        with temp_file('.xlsx') as fp:

            b1.to_xlsx(fp)

            b2 = Bus.from_xlsx(fp)

            f2.to_xlsx(fp)

            with self.assertRaises(StoreFileMutation):
                tuple(b2.items())


    def test_bus_to_xlsx_d(self) -> None:
        '''
        Test manipulating a file behind the Bus.
        '''
        f1 = Frame.from_dict(
                dict(a=(1,2,3)),
                index=('x', 'y', 'z'),
                name='f1')

        b1 = Bus.from_frames((f1,),)

        with temp_file('.xlsx') as fp:

            b1.to_xlsx(fp)

            b2 = Bus.from_xlsx(fp)

        with self.assertRaises(StoreFileMutation):
            tuple(b2.items())


    def test_bus_to_xlsx_e(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2,3)),
                index=('x', 'y', 'z'),
                name='f1')
        f2 = Frame.from_dict(
                dict(A=(10,20,30)),
                index=('q', 'r', 's'),
                name='f2')

        b1 = Bus.from_frames((f1, f2))

        with temp_file('.xlsx') as fp:
            b1.to_xlsx(fp)

            config = StoreConfig(include_index=True, index_depth=1)
            b2 = Bus.from_xlsx(fp, config=config)
            tuple(b2.items()) # force loading all

        for frame in (f1, f2):
            self.assertTrue(frame.equals(b2[frame.name]))


    def test_bus_to_xlsx_f(self) -> None:
        f = Frame.from_records([
                [np.datetime64('1983-02-20 05:34:18.763'), np.datetime64('2020-08-01')],
                [np.datetime64('1975-03-20 05:20:18.001'), np.datetime64('2020-07-31')]
                ],
                columns=(date(2020, 7, 31), date(2020, 8, 1)),
                index=(datetime(2020, 7, 31, 14, 20, 8), datetime(2017, 4, 28, 2, 30, 2)),
                name='frame')
        b1 = Bus.from_frames([f])

        with temp_file('.xlsx') as fp:
            b1.to_xlsx(fp)

            config = StoreConfig(include_index=True, index_depth=1)
            b2 = Bus.from_xlsx(fp, config=config)
            tuple(b2.items()) # force loading all

        self.assertEqual(b2['frame'].index.values.tolist(),
                [datetime(2020, 7, 31, 14, 20, 8),
                datetime(2017, 4, 28, 2, 30, 2)])

        self.assertEqual(b2['frame'].index.values.tolist(),
                [datetime(2020, 7, 31, 14, 20, 8),
                datetime(2017, 4, 28, 2, 30, 2)])

        self.assertEqual(b2['frame'].values.tolist(),
                [[datetime(1983, 2, 20, 5, 34, 18, 763000), datetime(2020, 8, 1, 0, 0)], [datetime(1975, 3, 20, 5, 20, 18, 1000), datetime(2020, 7, 31, 0, 0)]]
                )

    def test_bus_to_xlsx_g(self) -> None:

        dt64 = np.datetime64

        f1 = Frame.from_dict(
                dict(a=(1,2,3)),
                index=('x', 'y', 'z'),
                name=dt64('2019-12-31'))
        f2 = Frame.from_dict(
                dict(A=(10,20,30)),
                index=('q', 'r', 's'),
                name=dt64('2020-01-01'))

        config = StoreConfig(include_index=True,
                index_depth=1,
                label_encoder=str,
                label_decoder=dt64,
                )
        b1 = Bus.from_frames((f1, f2), config=config)

        with temp_file('.xlsx') as fp:
            b1.to_xlsx(fp, config=config)

            b2 = Bus.from_xlsx(fp, config=config)
            tuple(b2.items()) # force loading all

        for frame in (f1, f2):
            self.assertEqualFrames(frame, b2[frame.name])

    #---------------------------------------------------------------------------
    def test_bus_to_sqlite_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        frames = (f1, f2, f3)
        config = StoreConfigMap.from_frames(frames)
        b1 = Bus.from_frames(frames, config=config)

        with temp_file('.sqlite') as fp:
            b1.to_sqlite(fp)
            b2 = Bus.from_sqlite(fp, config=config)
            tuple(b2.items()) # force loading all

        for frame in frames:
            self.assertEqualFrames(frame, b2[frame.name])


    def test_bus_to_sqlite_b(self) -> None:
        '''
        Test manipulating a file behind the Bus.
        '''
        f1 = Frame.from_dict(
                dict(a=(1,2,3)),
                index=('x', 'y', 'z'),
                name='f1')

        b1 = Bus.from_frames((f1,),)

        with temp_file('.db') as fp:

            b1.to_sqlite(fp)

            b2 = Bus.from_sqlite(fp)

        with self.assertRaises(StoreFileMutation):
            tuple(b2.items())


    def test_bus_to_sqlite_c(self) -> None:

        dt64 = np.datetime64

        f1 = Frame.from_dict(
                dict(a=(1,2,3)),
                index=('x', 'y', 'z'),
                name=dt64('2019-12-31'))
        f2 = Frame.from_dict(
                dict(A=(10,20,30)),
                index=('q', 'r', 's'),
                name=dt64('2020-01-01'))

        config = StoreConfig(include_index=True,
                index_depth=1,
                label_encoder=str,
                label_decoder=dt64,
                )
        b1 = Bus.from_frames((f1, f2), config=config)

        with temp_file('.db') as fp:
            b1.to_sqlite(fp, config=config)

            b2 = Bus.from_sqlite(fp, config=config)
            tuple(b2.items()) # force loading all
            self.assertEqual(b2.index.dtype.kind, 'M')


        for frame in (f1, f2):
            self.assertEqualFrames(frame, b2[frame.name])



    #---------------------------------------------------------------------------
    def test_bus_to_hdf5_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        frames = (f1, f2, f3)
        config = StoreConfigMap.from_frames(frames)
        b1 = Bus.from_frames(frames, config=config)

        with temp_file('.h5') as fp:
            b1.to_hdf5(fp)
            b2 = Bus.from_hdf5(fp, config=config)
            tuple(b2.items()) # force loading all

        for frame in frames:
            self.assertEqualFrames(frame, b2[frame.name])


    def test_bus_to_hdf5_b(self) -> None:
        '''
        Test manipulating a file behind the Bus.
        '''
        f1 = Frame.from_dict(
                dict(a=(1,2,3)),
                index=('x', 'y', 'z'),
                name='f1')

        b1 = Bus.from_frames((f1,),)

        with temp_file('.h5') as fp:

            b1.to_hdf5(fp)

            b2 = Bus.from_hdf5(fp)

        with self.assertRaises(StoreFileMutation):
            tuple(b2.items())

    def test_bus_to_hdf5_c(self) -> None:
        dt64 = np.datetime64

        f1 = Frame.from_dict(
                dict(a=(1,2,3)),
                index=('x', 'y', 'z'),
                name=dt64('2019-12-31'))
        f2 = Frame.from_dict(
                dict(A=(10,20,30)),
                index=('q', 'r', 's'),
                name=dt64('2020-01-01'))

        config = StoreConfig(include_index=True,
                index_depth=1,
                label_encoder=str,
                label_decoder=dt64,
                )
        b1 = Bus.from_frames((f1, f2), config=config)

        with temp_file('.h5') as fp:
            b1.to_hdf5(fp, config=config)

            b2 = Bus.from_hdf5(fp, config=config)
            tuple(b2.items()) # force loading all
            self.assertEqual(b2.index.dtype.kind, 'M')


        for frame in (f1, f2):
            self.assertEqualFrames(frame, b2[frame.name])

    #---------------------------------------------------------------------------


    def test_bus_equals_a(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        f4 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f5 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f6 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        f7 = Frame.from_dict(
                dict(d=(10,20), b=(50,61)),
                index=('p', 'q'),
                name='f3')


        b1 = Bus.from_frames((f1, f2, f3))
        b2 = Bus.from_frames((f1, f2, f3))
        b3 = Bus.from_frames((f4, f5, f6))
        b4 = Bus.from_frames((f4, f5, f7))
        b5 = Bus.from_frames((f4, f5))
        b6 = Bus.from_frames((f3, f2, f1))

        self.assertTrue(b1.equals(b2))
        self.assertTrue(b1.equals(b3))

        self.assertFalse(b1.equals(b4))
        self.assertFalse(b1.equals(b5))
        self.assertFalse(b1.equals(b6))

    def test_bus_equals_b(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f1, f2, f3))
        self.assertTrue(b1.equals(b1))

        class BusDerived(Bus):
            pass

        b2 = BusDerived.from_frames((f1, f2, f3))

        self.assertFalse(b1.equals(b2, compare_class=True))
        self.assertTrue(b1.equals(b2, compare_class=False))
        self.assertFalse(b1.equals('foo', compare_class=False))


    def test_bus_equals_c(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')

        b1 = Bus.from_frames((f1, f2), name='foo')
        self.assertEqual(b1.name, 'foo')

        b2 = Bus.from_frames((f1, f2), name='bar')
        self.assertEqual(b2.name, 'bar')

        self.assertTrue(b1.equals(b2))
        self.assertFalse(b1.equals(b2, compare_name=True))


    #---------------------------------------------------------------------------


    def test_bus_interface_b(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='foo')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='bar')

        b1 = Bus.from_frames((f1, f2))
        post1 = b1.interface
        self.assertTrue(isinstance(post1, Frame))
        self.assertTrue(post1.shape, (41, 3))

        post2 = Bus.interface
        self.assertTrue(isinstance(post2, Frame))
        self.assertTrue(post2.shape, (41, 3)) #type: ignore


    #---------------------------------------------------------------------------

    def test_bus_mloc_a(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2,3)),
                index=('x', 'y', 'z'),
                name='f1')
        b1 = Bus.from_frames((f1,),)

        with temp_file('.db') as fp:

            b1.to_sqlite(fp)
            b2 = Bus.from_sqlite(fp)

            mloc = b2.mloc

            self.assertEqual(mloc.to_pairs(),
                    (('f1', None),))


    def test_bus_mloc_b(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2,3)),
                index=('x', 'y', 'z'),
                name='f1')

        f2 = Frame.from_dict(
                dict(x=(10,20,30)),
                index=('q', 'r', 's'),
                name='f2')

        config = StoreConfigMap.from_config(StoreConfig(index_depth=1))
        b1 = Bus.from_frames((f1, f2), config=config)

        with temp_file('.xlsx') as fp:
            b1.to_xlsx(fp)
            b2 = Bus.from_xlsx(fp, config=config)

            # only show memory locations for loaded Frames
            self.assertTrue(b2.iloc[1].equals(f2))
            self.assertEqual((b2.mloc == None).to_pairs(), #pylint: disable=C0121
                    (('f1', True), ('f2', False)))

    def test_bus_mloc_c(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f1, f2, f3))

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp)

            f2_loaded = b2['f2']

            mloc1 = b2.mloc

            f3_loaded = b2['f3']
            f1_loaded = b2['f1']

            self.assertEqual(mloc1['f2'], b2.mloc.loc['f2'])

    #---------------------------------------------------------------------------
    def test_bus_update_series_cache_iloc(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='foo')

        config = StoreConfigMap.from_config(StoreConfig(index_depth=1))

        # simulating a Bus with a FrameDefferred but no Store, just for testing
        s1 = Series((f1, FrameDeferred), index=('p', 'q'))
        b1 = Bus(s1, config=config)
        self.assertFalse(b1._loaded_all)

        with self.assertRaises(RuntimeError):
            b1._update_series_cache_iloc(1)

    #---------------------------------------------------------------------------
    def test_bus_extract_loc_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='foo')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='bar')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')


        ih = IndexHierarchy.from_labels((('a', 1), ('b', 2), ('b', 1)))
        s1 = Series((f1, f2, f3), index=ih, dtype=object)

        # do not support IndexHierarchy, as lables are tuples, not strings
        from ast import literal_eval
        config = StoreConfig(label_encoder=str, label_decoder=literal_eval)
        b1 = Bus(s1)
        b2 = b1[HLoc[:, 1]]
        self.assertEqual(b2.shape, (2,))
        self.assertEqual(b2.index.values.tolist(),
                [['a', 1], ['b', 1]])

        with temp_file('.zip') as fp:

            with self.assertRaises(RuntimeError):
                b1.to_zip_pickle(fp)

            b1.to_zip_pickle(fp, config=config)

            # NOTE: this comes back as an Index of tuples
            b3 = Bus.from_zip_pickle(fp, config=config)

            self.assertEqual(b3.index.values.tolist(),
                    [('a', 1), ('b', 2), ('b', 1)])


    #---------------------------------------------------------------------------
    def test_bus_values_a(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2,3)),
                index=('x', 'y', 'z'),
                name='f1')
        f2 = Frame.from_dict(
                dict(A=(10,20,30)),
                index=('q', 'r', 's'),
                name='f2')

        b1 = Bus.from_frames((f1, f2))
        post = b1.values.tolist()
        self.assertTrue(f1.equals(post[0]))
        self.assertTrue(f2.equals(post[1]))

    #---------------------------------------------------------------------------
    def test_bus_to_parquet_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        config = StoreConfig(
                index_depth=1,
                columns_depth=1,
                include_columns=True,
                include_index=True
                )
        b1 = Bus.from_frames((f1, f2, f3), config=config)

        with temp_file('.zip') as fp:
            b1.to_zip_parquet(fp)

            b2 = Bus.from_zip_parquet(fp, config=config)
            tuple(b2.items()) # force loading all

        for frame in (f1, f2, f3):
            # parquet brings in characters as objects, thus forcing different dtypes
            self.assertEqualFrames(frame, b2[frame.name], compare_dtype=False)


    def test_bus_to_parquet_b(self) -> None:

        dt64 = np.datetime64

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name=dt64('2020-01-01'))
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name=dt64('1932-12-17'))
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name=dt64('1950-04-23'))

        config = StoreConfig(
                index_depth=1,
                columns_depth=1,
                include_columns=True,
                include_index=True,
                label_encoder=str,
                label_decoder=dt64
                )

        b1 = Bus.from_frames((f1, f2, f3), config=config)
        self.assertEqual(b1.index.dtype.kind, 'M')

        with temp_file('.zip') as fp:
            b1.to_zip_parquet(fp, config=config)

            b2 = Bus.from_zip_parquet(fp, config=config)
            self.assertEqual(b2.index.dtype.kind, 'M')

            key = dt64('2020-01-01')
            self.assertEqualFrames(b1[key], b2[key], compare_dtype=False)

    #---------------------------------------------------------------------------
    def test_bus_max_persist_a(self) -> None:
        def items() -> tp.Iterator[tp.Tuple[str, Frame]]:
            for i in range(20):
                yield str(i), Frame(np.arange(i, i+10).reshape(2, 5))

        s = Series.from_items(items(), dtype=object)
        b1 = Bus(s)

        config = StoreConfig(
                index_depth=1,
                columns_depth=1,
                include_columns=True,
                include_index=True
                )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)

            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=3)
            for i in b2.index:
                _ = b2[i]
                self.assertTrue(b2._loaded.sum() <= 3)

            # after iteration only the last three are loaded
            self.assertEqual(b2._loaded.tolist(),
                    [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True])


    def test_bus_max_persist_b(self) -> None:
        def items() -> tp.Iterator[tp.Tuple[str, Frame]]:
            for i in range(20):
                yield str(i), Frame(np.arange(i, i+10).reshape(2, 5))

        s = Series.from_items(items(), dtype=object)
        b1 = Bus(s)

        config = StoreConfig(
                index_depth=1,
                columns_depth=1,
                include_columns=True,
                include_index=True
                )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)

            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=1)
            b3 = b2.iloc[10:]
            self.assertEqual(b3._loaded.sum(), 1)
            # only the last one is loasded
            self.assertEqual(b3._loaded.tolist(),
                    [False, False, False, False, False, False, False, False, False, True]
                    )
            self.assertEqual(b3.iloc[0].sum().sum(), 145)
            self.assertEqual(b3._loaded.tolist(),
                    [True, False, False, False, False, False, False, False, False, False]
                    )
            self.assertEqual(b3.iloc[4].sum().sum(), 185)
            self.assertEqual(b3._loaded.tolist(),
                    [False, False, False, False, True, False, False, False, False, False]
                    )

    def test_bus_max_persist_c(self) -> None:
        def items() -> tp.Iterator[tp.Tuple[str, Frame]]:
            for i in range(4):
                yield str(i), Frame(np.arange(i, i+10).reshape(2, 5))

        s = Series.from_items(items(), dtype=object)
        b1 = Bus(s)

        config = StoreConfig(
                index_depth=1,
                columns_depth=1,
                include_columns=True,
                include_index=True
                )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)

            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=4)

            for _ in b2.items():
                pass
            self.assertTrue(b2._loaded.all())

            b3 = Bus.from_zip_pickle(fp, config=config, max_persist=3)

            _ = b3.iloc[[0, 2, 3]]
            self.assertEqual(b3._loaded.tolist(),
                    [True, False, True, True])

            _ = b3.iloc[[0, 1, 3]]
            self.assertEqual(b3._loaded.tolist(),
                    [True, True, False, True])

            _ = b3.iloc[[1, 2, 3]]
            self.assertEqual(b3._loaded.tolist(),
                    [False, True, True, True])

            _ = b3.iloc[[0, 1, 2]]
            self.assertEqual(b3._loaded.tolist(),
                    [True, True, True, False])

            _ = b3.iloc[[0, 2, 3]]
            self.assertEqual(b3._loaded.tolist(),
                    [True, False, True, True])



    def test_bus_max_persist_d(self) -> None:
        def items() -> tp.Iterator[tp.Tuple[str, Frame]]:
            for i in range(5):
                yield str(i), Frame(np.arange(i, i+10).reshape(2, 5))

        s = Series.from_items(items(), dtype=object)
        b1 = Bus(s)

        config = StoreConfig(
                index_depth=1,
                columns_depth=1,
                include_columns=True,
                include_index=True
                )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)

            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=3)

            _ = b2.iloc[[0, 2, 4]]
            self.assertEqual(b2._loaded.tolist(),
                    [True, False, True, False, True])

            _ = b2.iloc[[1, 2, 3]]
            self.assertEqual(b2._loaded.tolist(),
                    [False, True, True, True, False])

            _ = b2.iloc[4]
            self.assertEqual(b2._loaded.tolist(),
                    [False, False, True, True, True])

            _ = b2.iloc[0]
            self.assertEqual(b2._loaded.tolist(),
                    [True, False, False, True, True])

            _ = b2.iloc[[2, 3, 4]]
            self.assertEqual(b2._loaded.tolist(),
                    [False, False, True, True, True])

            _ = b2.iloc[[0, 1]]
            self.assertEqual(b2._loaded.tolist(),
                    [True, True, False, False, True])

            _ = b2.iloc[0]
            self.assertEqual(b2._loaded.tolist(),
                    [True, True, False, False, True])

            _ = b2.iloc[[3, 4]]
            self.assertEqual(b2._loaded.tolist(),
                    [True, False, False, True, True])


    def test_bus_max_persist_e(self) -> None:
        def items() -> tp.Iterator[tp.Tuple[str, Frame]]:
            for i in range(4):
                yield str(i), Frame(np.arange(i, i+10).reshape(2, 5))

        s = Series.from_items(items(), dtype=object)
        b1 = Bus(s)

        config = StoreConfig(
                index_depth=1,
                columns_depth=1,
                include_columns=True,
                include_index=True
                )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=4)

            _ = b2.iloc[[0, 1]]
            _ = b2.iloc[[2, 3]]
            self.assertTrue(b2._loaded_all)

            _ = b2.iloc[[1, 0]]
            self.assertEqual(list(b2._last_accessed.keys()),
                    ['2', '3', '1', '0'])

            _ = b2.iloc[3]
            self.assertEqual(list(b2._last_accessed.keys()),
                    ['2', '1', '0', '3'])

            _ = b2.iloc[:3]
            self.assertEqual(list(b2._last_accessed.keys()),
                    ['3', '0', '1', '2'])



    def test_bus_max_persist_f(self) -> None:
        def items() -> tp.Iterator[tp.Tuple[str, Frame]]:
            for i in range(5):
                yield str(i), Frame(np.arange(i, i+10).reshape(2, 5))

        s = Series.from_items(items(), dtype=object)
        b1 = Bus(s)

        config = StoreConfig(
                index_depth=1,
                columns_depth=1,
                include_columns=True,
                include_index=True
                )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)

            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=1)

            # insure that items delivers Frame, not FrameDeferred
            post = tuple(b2.items())
            self.assertEqual(len(post), 5)
            self.assertTrue(all(isinstance(f, Frame) for _, f in post))
            self.assertEqual(b2._loaded.tolist(),
                    [False, False, False, False, True])


    def test_bus_max_persist_g(self) -> None:
        def items() -> tp.Iterator[tp.Tuple[str, Frame]]:
            for i in range(5):
                yield str(i), Frame(np.arange(i, i+10).reshape(2, 5))

        s = Series.from_items(items(), dtype=object)
        b1 = Bus(s)

        config = StoreConfig(
                index_depth=1,
                columns_depth=1,
                include_columns=True,
                include_index=True
                )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)

            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=1)
            # insure that items delivers Frame, not FrameDeferred
            post = b2.values
            self.assertEqual(len(post), 5)
            self.assertTrue(all(isinstance(f, Frame) for f in post))
            self.assertEqual(b2._loaded.tolist(),
                    [False, False, False, False, True])



    #---------------------------------------------------------------------------
    def test_bus_sort_index_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f3, f2, f1))
        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp)
            b3 = b2.sort_index()
            self.assertEqual(b2.index.values.tolist(),
                    ['f3', 'f2', 'f1'])
            self.assertEqual(b3.index.values.tolist(),
                    ['f1', 'f2', 'f3'])

            self.assertTrue(isinstance(b3, Bus))

            f4 = b3['f2']
            self.assertEqual(f4.to_pairs(0),
                    (('c', (('x', 1), ('y', 2), ('z', 3))), ('b', (('x', 4), ('y', 5), ('z', 6)))))

            f5 = b3['f1']
            self.assertEqual(f5.to_pairs(),
                    (('a', (('x', 1), ('y', 2))), ('b', (('x', 3), ('y', 4)))))

    #---------------------------------------------------------------------------
    def test_bus_sort_values_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(30000,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(1,2), b=(5,6)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f3, f2, f1))
        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp)
            b3 = b2.sort_values(key=lambda b: b.iter_element().apply(lambda f: f.sum().sum()))

            self.assertEqual(b3.index.values.tolist(),
                ['f3', 'f2', 'f1']
                )

    #---------------------------------------------------------------------------
    def test_bus_drop_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f3, f2, f1))
        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp)

            b3 = b2.drop[['f3', 'f1']]
            self.assertEqual(b3.status['loaded'].sum(), 0)
            self.assertEqual(b3['f2'].to_pairs(),
                    (('c', (('x', 1), ('y', 2), ('z', 3))), ('b', (('x', 4), ('y', 5), ('z', 6))))
                    )

    #---------------------------------------------------------------------------
    def test_bus_from_dict_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')

        b1 = Bus.from_dict(dict(a=f1, b=f2))
        self.assertEqual(b1.index.values.tolist(),
                ['a', 'b'])

    #---------------------------------------------------------------------------
    def test_bus_iter_element_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')

        b1 = Bus.from_dict(dict(a=f1, b=f2))
        post = tuple(b1.iter_element())
        self.assertEqual(len(post), 2)
        self.assertEqual(post[1].to_pairs(),
            (('c', (('x', 1), ('y', 2), ('z', 3))), ('b', (('x', 4), ('y', 5), ('z', 6))))
            )

    #---------------------------------------------------------------------------
    def test_bus_reindex_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f3, f2, f1))
        b2 = b1.reindex(('f1', 'f2', 'f4'), fill_value=f3)
        self.assertTrue(b2['f4'].equals(f3))
        self.assertTrue(b2.__class__ is Bus)

    #---------------------------------------------------------------------------
    def test_bus_relabel_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f3, f2, f1))
        b2 = b1.relabel(('a', 'b', 'c'))

        self.assertTrue(b2['a'].equals(f3))
        self.assertTrue(b2.__class__ is Bus)

    #---------------------------------------------------------------------------
    def test_bus_relabel_flat_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f3, f2, f1))
        b2 = b1.relabel_level_add('X')
        self.assertEqual(b2.relabel_flat().index.values.tolist(),
                [('X', 'f3'), ('X', 'f2'), ('X', 'f1')])

        self.assertTrue(b2[('X', 'f2')].equals(f2))
        self.assertTrue(b2.__class__ is Bus)

    #---------------------------------------------------------------------------
    def test_bus_relabel_level_drop_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f3, f2, f1))
        b2 = b1.relabel_level_add('X')
        b3 = b2.relabel_level_drop(1)
        self.assertTrue(b3.__class__ is Bus)
        self.assertTrue(b3.index.equals(b1.index))

    #---------------------------------------------------------------------------
    def test_bus_relabel_rehierarch_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f3, f2, f1))
        b2 = b1.relabel_level_add('X')
        b3 = b2.rehierarch([1, 0])
        self.assertEqual(b3.index.values.tolist(),
                [['f3', 'X'], ['f2', 'X'], ['f1', 'X']])

    #---------------------------------------------------------------------------
    def test_bus_roll_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f3, f2, f1))
        b2 = b1.roll(2, include_index=True)
        self.assertEqual(b2.index.values.tolist(),
                ['f2', 'f1', 'f3']
                )
        self.assertTrue(b2['f2'].equals(b1['f2']))



    #---------------------------------------------------------------------------
    def test_bus_shift_a(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='f1')
        f2 = Frame.from_dict(
                dict(c=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f3 = Frame.from_dict(
                dict(d=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='f3')

        b1 = Bus.from_frames((f1, f2, f3))
        b2 = b1.shift(2, fill_value=f1)
        self.assertTrue(b2['f3'].equals(b1['f1']))
        self.assertTrue(b2['f2'].equals(b1['f1']))
        self.assertTrue(b2['f1'].equals(b1['f1']))





if __name__ == '__main__':

    unittest.main()

