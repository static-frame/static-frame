from __future__ import annotations

import ast
import os
import pickle
from ast import literal_eval
from datetime import date, datetime
from hashlib import sha256

import frame_fixtures as ff
import numpy as np
import typing_extensions as tp

from static_frame.core.batch import Batch
from static_frame.core.bus import Bus, FrameDeferred
from static_frame.core.display_config import DisplayConfig
from static_frame.core.exception import (
    ErrorInitBus,
    ErrorInitIndexNonUnique,
    ErrorNPYEncode,
    ImmutableTypeError,
    StoreFileMutation,
)
from static_frame.core.frame import Frame
from static_frame.core.hloc import HLoc
from static_frame.core.index_auto import IndexAutoConstructorFactory, IndexAutoFactory
from static_frame.core.index_datetime import IndexDate, IndexYearMonth
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.series import Series
from static_frame.core.store_config import StoreConfig, StoreConfigMap
from static_frame.core.store_zip import StoreZipTSV
from static_frame.test.test_case import TestCase, skip_win, temp_file


class TestUnit(TestCase):
    def test_frame_deferred_a(self) -> None:
        self.assertEqual(str(FrameDeferred), '<FrameDeferred>')

    def test_bus_slotted_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='foo')

        b1 = Bus.from_frames((f1,))

        with self.assertRaises(AttributeError):
            b1.g = 30  # type: ignore
        with self.assertRaises(AttributeError):
            b1.__dict__

    # ---------------------------------------------------------------------------

    def test_bus_init_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='foo')
        f2 = Frame.from_dict(
            dict(a=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='bar'
        )

        config = StoreConfigMap.from_config(StoreConfig(index_depth=1))
        b1 = Bus.from_frames((f1, f2), config=config)

        self.assertEqual(b1.keys().values.tolist(), ['foo', 'bar'])

        with temp_file('.zip') as fp:
            b1.to_zip_tsv(fp)
            b2 = Bus.from_zip_tsv(fp)

            f3 = b2['bar']
            f4 = b2['foo']
            zs = StoreZipTSV(fp)
            zs.write(b1.items())

            # how to show that this derived getitem has derived type?
            f3 = zs.read('foo', config=config['foo'])
            self.assertEqual(
                f3.to_pairs(),  # type: ignore
                (('a', (('x', 1), ('y', 2))), ('b', (('x', 3), ('y', 4)))),
            )

    def test_bus_init_b(self) -> None:
        with self.assertRaises(ErrorInitBus):
            Bus.from_series(Series([1, 2, 3]))

        with self.assertRaises(ErrorInitBus):
            Bus.from_series(Series([3, 4], dtype=object))

        with self.assertRaises(ErrorInitBus):
            Bus.from_series(Series([3, 4], index=('a', 'b'), dtype=object))

    def test_bus_init_c(self) -> None:
        f1 = Frame.from_dict(
            dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='foo', dtypes=np.int64
        )
        f2 = Frame.from_dict(
            dict(a=(1, 2, 3), b=(4, 5, 6)),
            index=('x', 'y', 'z'),
            name='bar',
            dtypes=np.int64,
        )

        config = StoreConfigMap.from_config(StoreConfig(index_depth=1))
        b1 = Bus.from_frames((f1, f2), config=config)

        self.assertEqual(b1.keys().values.tolist(), ['foo', 'bar'])

        with temp_file('.zip') as fp:
            b1.to_zip_csv(fp)
            b2 = Bus.from_zip_csv(fp, config=config)

            f1_loaded = b2['foo']
            f2_loaded = b2['bar']

            self.assertEqualFrames(f1, f1_loaded)
            self.assertEqualFrames(f2, f2_loaded)

    def test_bus_init_d(self) -> None:
        f1 = ff.parse('s(2,2)|c(I,str)|v(int64)')
        f2 = ff.parse('s(2,2)|c(I,str)|v(bool)')

        b1 = Bus((f1, f2), index=('a', 'b'))
        self.assertEqual(
            b1.shapes.to_pairs(),
            (('a', (2, 2)), ('b', (2, 2))),
        )

        b2 = Bus(
            (f1, f2),
            index=IndexDate(('2021-01-01', '1542-01-22')),
            own_index=True,
        )
        self.assertEqual(b2.keys().values.tolist(), [date(2021, 1, 1), date(1542, 1, 22)])

    def test_bus_init_e(self) -> None:
        with self.assertRaises(ErrorInitBus):
            b1 = Bus(np.arange(0, 2), index=('a', 'b'))

    def test_bus_init_f(self) -> None:
        with self.assertRaises(ErrorInitBus):
            _ = Bus(None, index=('a', 'b', 'c'))

    def test_bus_init_g(self) -> None:
        f1 = ff.parse('s(2,2)|c(I,str)|v(int64)')
        f2 = ff.parse('s(2,2)|c(I,str)|v(bool)')

        with self.assertRaises(ErrorInitBus):
            _ = Bus((f1, f2), index=('a', 'b'), own_data=True)

    def test_bus_init_h(self) -> None:
        f1 = ff.parse('s(2,2)|c(I,str)|v(int64)')

        with self.assertRaises(ErrorInitBus):
            b1 = Bus((f for f in (f1,)), index=('a', 'b'))

    def test_bus_init_i(self) -> None:
        f1 = ff.parse('s(2,2)|c(I,str)|v(int64)')
        f2 = ff.parse('s(2,2)|c(I,str)|v(bool)')

        with self.assertRaises(ErrorInitBus):
            b1 = Bus((f1, f2), index=('a', 'b'), max_persist=0)

    # ---------------------------------------------------------------------------

    def test_bus_from_frames_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='foo')
        f2 = Frame.from_dict(
            dict(a=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='foo'
        )

        with self.assertRaises(ErrorInitIndexNonUnique):
            _ = Bus.from_frames((f1, f2))

    # ---------------------------------------------------------------------------

    def test_bus_from_series_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        s1 = Series((f1, f2), index=('a', 'b'))
        b1 = Bus.from_series(s1, own_data=True)
        self.assertEqual(
            b1.status['shape'].to_pairs(),
            (('a', (4, 2)), ('b', (4, 5))),
        )

    # ---------------------------------------------------------------------------

    def test_bus_rename_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(a=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )

        b1 = Bus.from_frames(
            (
                f1,
                f2,
            ),
            name='foo',
        )
        self.assertEqual(b1.name, 'foo')
        b2 = b1.rename('bar')
        self.assertTrue(b2.__class__ is Bus)
        self.assertEqual(b2.name, 'bar')

    # ---------------------------------------------------------------------------

    def test_bus_from_items_a(self) -> None:
        f1 = Frame.from_dict(
            dict(a=(1, 2), b=(3, 4)),
            index=('x', 'y'),
        )
        f2 = Frame.from_dict(
            dict(a=(1, 2, 3), b=(4, 5, 6)),
            index=('x', 'y', 'z'),
        )

        b1 = Bus.from_items((('a', f1), ('b', f2)))
        self.assertEqual(b1.index.values.tolist(), ['a', 'b'])

    # ---------------------------------------------------------------------------

    def test_bus_shapes_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(a=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(a=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f1, f2, f3))

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)

            b2 = Bus.from_zip_pickle(fp)

            f2_loaded = b2['f2']

            self.assertEqual(
                b2.shapes.to_pairs(), (('f1', None), ('f2', (3, 2)), ('f3', None))
            )

            f3_loaded = b2['f3']

            self.assertEqual(
                b2.shapes.to_pairs(), (('f1', None), ('f2', (3, 2)), ('f3', (2, 2)))
            )

    @skip_win
    def test_bus_nbytes_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(a=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(a=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

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

    @skip_win
    def test_bus_dtypes_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f1, f2, f3))

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp)

            self.assertEqual(b2.dtypes.to_pairs(), ())

            f2_loaded = b2['f2']

            self.assertEqual(
                b2.dtypes.to_pairs(),
                (
                    ('c', (('f1', None), ('f2', np.dtype('int64')), ('f3', None))),
                    ('b', (('f1', None), ('f2', np.dtype('int64')), ('f3', None))),
                ),
            )

            f3_loaded = b2['f3']

            self.assertEqual(
                b2.dtypes.to_pairs(),
                (
                    (
                        'b',
                        (
                            ('f1', None),
                            ('f2', np.dtype('int64')),
                            ('f3', np.dtype('int64')),
                        ),
                    ),
                    ('c', (('f1', None), ('f2', np.dtype('int64')), ('f3', None))),
                    ('d', (('f1', None), ('f2', None), ('f3', np.dtype('int64')))),
                ),
            )

    @skip_win
    def test_bus_status_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f1, f2, f3))

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp)

            status = b2.status
            self.assertEqual(status.shape, (3, 4))
            # force load all
            tuple(b2.items())

            self.assertEqual(
                b2.status.to_pairs(),
                (
                    ('loaded', (('f1', True), ('f2', True), ('f3', True))),
                    ('size', (('f1', 4.0), ('f2', 6.0), ('f3', 4.0))),
                    ('nbytes', (('f1', 32.0), ('f2', 48.0), ('f3', 32.0))),
                    ('shape', (('f1', (2, 2)), ('f2', (3, 2)), ('f3', (2, 2)))),
                ),
            )

    def test_bus_keys_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')
        f4 = Frame.from_dict(
            dict(q=(None, None), r=(np.nan, np.nan)), index=(1000, 1001), name='f4'
        )

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

    def test_bus_iter_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        b1 = Bus.from_frames((f1, f2))
        biter = iter(b1)
        self.assertEqual(next(biter), 'f1')
        self.assertEqual(next(biter), 'f2')

    def test_bus_reversed_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f1, f2, f3))
        self.assertEqual(list(reversed(b1)), ['f3', 'f2', 'f1'])

    def test_bus_display_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f1, f2, f3))
        self.assertEqual(
            b1.display(DisplayConfig(type_color=False)).to_rows(),
            [
                '<Bus>',
                '<Index>',
                'f1      Frame',
                'f2      Frame',
                'f3      Frame',
                '<<U2>   <object>',
            ],
        )

        rows1 = b1.display(DisplayConfig(type_color=False, type_show=False)).to_rows()
        self.assertEqual(rows1, ['f1 Frame', 'f2 Frame', 'f3 Frame'])

        rows2 = b1.display(
            DisplayConfig(type_color=False, type_show=False, include_index=False)
        ).to_rows()
        self.assertEqual(rows2, ['Frame', 'Frame', 'Frame'])

    # ---------------------------------------------------------------------------

    def test_bus_iloc_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f1, f2, f3))

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp)

            self.assertEqual(
                b2.iloc[[0, 2]].status['loaded'].to_pairs(),
                (('f1', False), ('f3', False)),
            )

    def test_bus_iloc_b(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')

        b1 = Bus.from_frames((f1,))
        f2 = b1.iloc[0]
        self.assertTrue(f1 is f2)

    def test_bus_loc_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')

        b1 = Bus.from_frames((f1,))
        f2 = b1.loc['f1']
        self.assertTrue(f1 is f2)

    def test_bus_loc_b(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f1, f2, f3))
        b2 = b1.loc['f2':]  # type: ignore
        self.assertEqual(len(b2), 2)
        self.assertEqual(b2.index.values.tolist(), ['f2', 'f3'])

    def test_bus_getitem_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f1, f2, f3))

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp)

            self.assertEqual(
                b2['f2':].status['loaded'].to_pairs(),  # type: ignore
                (('f2', False), ('f3', False)),
            )

    # ---------------------------------------------------------------------------

    def test_bus_to_xlsx_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        config = StoreConfigMap.from_config(
            StoreConfig(
                index_depth=1, columns_depth=1, include_columns=True, include_index=True
            )
        )
        b1 = Bus.from_frames((f1, f2, f3), config=config)

        with temp_file('.xlsx') as fp:
            b1.to_xlsx(fp)

            b2 = Bus.from_xlsx(fp, config=config)
            tuple(b2.items())  # force loading all

        for frame in (f1, f2, f3):
            self.assertEqualFrames(frame, b2[frame.name])

    def test_bus_to_xlsx_b(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2, 3)), index=('x', 'y', 'z'), name='f1')
        f2 = Frame.from_dict(dict(A=(10, 20, 30)), index=('q', 'r', 's'), name='f2')

        config = StoreConfig(include_index=True, index_depth=1)
        b1 = Bus.from_frames((f1, f2), config=config)

        with temp_file('.xlsx') as fp:
            b1.to_xlsx(fp)

            b2 = Bus.from_xlsx(fp, config=config)
            tuple(b2.items())  # force loading all

        for frame in (f1, f2):
            self.assertEqualFrames(frame, b2[frame.name])

    def test_bus_to_xlsx_c(self) -> None:
        """
        Test manipulating a file behind the Bus.
        """
        f1 = Frame.from_dict(dict(a=(1, 2, 3)), index=('x', 'y', 'z'), name='f1')

        f2 = Frame.from_dict(dict(x=(10, 20, 30)), index=('q', 'r', 's'), name='f2')

        b1 = Bus.from_frames(
            (f1,),
        )

        with temp_file('.xlsx') as fp:
            b1.to_xlsx(fp)

            b2 = Bus.from_xlsx(fp)

            f2.to_xlsx(fp)

            with self.assertRaises((StoreFileMutation, KeyError)):
                tuple(b2.items())

    def test_bus_to_xlsx_d(self) -> None:
        """
        Test manipulating a file behind the Bus.
        """
        f1 = Frame.from_dict(dict(a=(1, 2, 3)), index=('x', 'y', 'z'), name='f1')

        b1 = Bus.from_frames(
            (f1,),
        )

        with temp_file('.xlsx') as fp:
            b1.to_xlsx(fp)

            b2 = Bus.from_xlsx(fp)

        with self.assertRaises(StoreFileMutation):
            tuple(b2.items())

    def test_bus_to_xlsx_e(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2, 3)), index=('x', 'y', 'z'), name='f1')
        f2 = Frame.from_dict(dict(A=(10, 20, 30)), index=('q', 'r', 's'), name='f2')

        b1 = Bus.from_frames((f1, f2))

        with temp_file('.xlsx') as fp:
            b1.to_xlsx(fp)

            config = StoreConfig(include_index=True, index_depth=1)
            b2 = Bus.from_xlsx(fp, config=config)
            tuple(b2.items())  # force loading all

        for frame in (f1, f2):
            self.assertTrue(frame.equals(b2[frame.name]))

    def test_bus_to_xlsx_f(self) -> None:
        f = Frame.from_records(
            [
                [np.datetime64('1983-02-20 05:34:18.763'), np.datetime64('2020-08-01')],
                [np.datetime64('1975-03-20 05:20:18.001'), np.datetime64('2020-07-31')],
            ],
            columns=(date(2020, 7, 31), date(2020, 8, 1)),
            index=(datetime(2020, 7, 31, 14, 20, 8), datetime(2017, 4, 28, 2, 30, 2)),
            name='frame',
        )
        b1 = Bus.from_frames([f])

        with temp_file('.xlsx') as fp:
            b1.to_xlsx(fp)

            config = StoreConfig(include_index=True, index_depth=1)
            b2 = Bus.from_xlsx(fp, config=config)
            tuple(b2.items())  # force loading all

        self.assertEqual(
            b2['frame'].index.values.tolist(),
            [datetime(2020, 7, 31, 14, 20, 8), datetime(2017, 4, 28, 2, 30, 2)],
        )

        self.assertEqual(
            b2['frame'].index.values.tolist(),
            [datetime(2020, 7, 31, 14, 20, 8), datetime(2017, 4, 28, 2, 30, 2)],
        )

        self.assertEqual(
            b2['frame'].values.tolist(),
            [
                [datetime(1983, 2, 20, 5, 34, 18, 763000), datetime(2020, 8, 1, 0, 0)],
                [datetime(1975, 3, 20, 5, 20, 18, 1000), datetime(2020, 7, 31, 0, 0)],
            ],
        )

    def test_bus_to_xlsx_g(self) -> None:
        dt64 = np.datetime64

        f1 = Frame.from_dict(
            dict(a=(1, 2, 3)), index=('x', 'y', 'z'), name=dt64('2019-12-31')
        )
        f2 = Frame.from_dict(
            dict(A=(10, 20, 30)), index=('q', 'r', 's'), name=dt64('2020-01-01')
        )

        config = StoreConfig(
            include_index=True,
            index_depth=1,
            label_encoder=str,
            label_decoder=dt64,
        )
        b1 = Bus.from_frames((f1, f2), config=config, index_constructor=IndexDate)

        with temp_file('.xlsx') as fp:
            b1.to_xlsx(fp, config=config)

            b2 = Bus.from_xlsx(fp, config=config, index_constructor=IndexDate)
            tuple(b2.items())  # force loading all

        for frame in (f1, f2):
            self.assertEqualFrames(frame, b2[frame.name])

    # ---------------------------------------------------------------------------

    def test_bus_to_sqlite_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        frames = (f1, f2, f3)
        config = StoreConfigMap.from_frames(frames)
        b1 = Bus.from_frames(frames, config=config)

        with temp_file('.sqlite') as fp:
            b1.to_sqlite(fp)
            b2 = Bus.from_sqlite(fp, config=config)
            tuple(b2.items())  # force loading all

        for frame in frames:
            self.assertEqualFrames(frame, b2[frame.name])

    @skip_win
    def test_bus_to_sqlite_b(self) -> None:
        """
        Test manipulating a file behind the Bus.
        """
        f1 = Frame.from_dict(dict(a=(1, 2, 3)), index=('x', 'y', 'z'), name='f1')

        b1 = Bus.from_frames(
            (f1,),
        )

        with temp_file('.db') as fp:
            b1.to_sqlite(fp)
            b2 = Bus.from_sqlite(fp)

        with self.assertRaises(StoreFileMutation):
            tuple(b2.items())

    def test_bus_to_sqlite_c(self) -> None:
        dt64 = np.datetime64

        f1 = Frame.from_dict(
            dict(a=(1, 2, 3)), index=('x', 'y', 'z'), name=dt64('2019-12-31')
        )
        f2 = Frame.from_dict(
            dict(A=(10, 20, 30)), index=('q', 'r', 's'), name=dt64('2020-01-01')
        )

        config = StoreConfig(
            include_index=True,
            index_depth=1,
            label_encoder=str,
            label_decoder=dt64,
        )
        b1 = Bus.from_frames((f1, f2), config=config, index_constructor=IndexDate)

        with temp_file('.db') as fp:
            b1.to_sqlite(fp, config=config)

            b2 = Bus.from_sqlite(fp, config=config, index_constructor=IndexDate)
            tuple(b2.items())  # force loading all
            self.assertEqual(b2.index.dtype.kind, 'M')  # type: ignore

        for frame in (f1, f2):
            self.assertEqualFrames(frame, b2[frame.name])

    # ---------------------------------------------------------------------------

    def test_bus_equals_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        f4 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f5 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f6 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        f7 = Frame.from_dict(dict(d=(10, 20), b=(50, 61)), index=('p', 'q'), name='f3')

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
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f1, f2, f3))
        self.assertTrue(b1.equals(b1))

        class BusDerived(Bus):
            pass

        b2 = BusDerived.from_frames((f1, f2, f3))

        self.assertFalse(b1.equals(b2, compare_class=True))
        self.assertTrue(b1.equals(b2, compare_class=False))
        self.assertFalse(b1.equals('foo', compare_class=False))

    def test_bus_equals_c(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )

        b1 = Bus.from_frames((f1, f2), name='foo')
        self.assertEqual(b1.name, 'foo')

        b2 = Bus.from_frames((f1, f2), name='bar')
        self.assertEqual(b2.name, 'bar')

        self.assertTrue(b1.equals(b2))
        self.assertFalse(b1.equals(b2, compare_name=True))

    # ---------------------------------------------------------------------------

    def test_bus_interface_b(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='foo')
        f2 = Frame.from_dict(
            dict(a=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='bar'
        )

        b1 = Bus.from_frames((f1, f2))
        post1 = b1.interface
        self.assertTrue(isinstance(post1, Frame))
        self.assertTrue(post1.shape, (41, 3))

        post2 = Bus.interface
        self.assertTrue(isinstance(post2, Frame))
        self.assertTrue(post2.shape, (41, 3))  # type: ignore

    # ---------------------------------------------------------------------------

    def test_bus_mloc_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2, 3)), index=('x', 'y', 'z'), name='f1')
        b1 = Bus.from_frames(
            (f1,),
        )

        with temp_file('.db') as fp:
            b1.to_sqlite(fp)
            b2 = Bus.from_sqlite(fp)

            mloc = b2.mloc

            self.assertEqual(mloc.to_pairs(), (('f1', None),))

    def test_bus_mloc_b(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2, 3)), index=('x', 'y', 'z'), name='f1')

        f2 = Frame.from_dict(dict(x=(10, 20, 30)), index=('q', 'r', 's'), name='f2')

        config = StoreConfigMap.from_config(StoreConfig(index_depth=1))
        b1 = Bus.from_frames((f1, f2), config=config)

        with temp_file('.xlsx') as fp:
            b1.to_xlsx(fp)
            b2 = Bus.from_xlsx(fp, config=config)

            # only show memory locations for loaded Frames
            self.assertTrue(b2.iloc[1].equals(f2))
            self.assertEqual((b2.mloc == None).to_pairs(), (('f1', True), ('f2', False)))

    def test_bus_mloc_c(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f1, f2, f3))

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp)

            f2_loaded = b2['f2']

            mloc1 = b2.mloc

            f3_loaded = b2['f3']
            f1_loaded = b2['f1']

            self.assertEqual(mloc1['f2'], b2.mloc.loc['f2'])

    # ---------------------------------------------------------------------------

    def test_bus_extract_loc_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='foo')
        f2 = Frame.from_dict(
            dict(a=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='bar'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        ih = IndexHierarchy.from_labels((('a', 1), ('b', 2), ('b', 1)))
        s1 = Series((f1, f2, f3), index=ih, dtype=object)

        # do not support IndexHierarchy, as lables are tuples, not strings
        config = StoreConfig(label_encoder=str, label_decoder=literal_eval)
        b1 = Bus.from_series(s1)
        b2 = b1[HLoc[:, 1]]
        self.assertEqual(b2.shape, (2,))
        self.assertEqual(b2.index.values.tolist(), [['a', 1], ['b', 1]])

        with temp_file('.zip') as fp:
            with self.assertRaises(RuntimeError):
                b1.to_zip_pickle(fp)

            b1.to_zip_pickle(fp, config=config)
            # NOTE: this comes back as an Index of tuples
            b3 = Bus.from_zip_pickle(fp, config=config)

            self.assertEqual(b3.index.values.tolist(), [('a', 1), ('b', 2), ('b', 1)])

    # ---------------------------------------------------------------------------

    def test_bus_values_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2, 3)), index=('x', 'y', 'z'), name='f1')
        f2 = Frame.from_dict(dict(A=(10, 20, 30)), index=('q', 'r', 's'), name='f2')

        b1 = Bus.from_frames((f1, f2))
        post = b1.values.tolist()
        self.assertTrue(f1.equals(post[0]))
        self.assertTrue(f2.equals(post[1]))

    # ---------------------------------------------------------------------------

    def test_bus_to_parquet_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )
        b1 = Bus.from_frames((f1, f2, f3), config=config)

        with temp_file('.zip') as fp:
            b1.to_zip_parquet(fp)

            b2 = Bus.from_zip_parquet(fp, config=config)
            tuple(b2.items())  # force loading all

        for frame in (f1, f2, f3):
            # parquet brings in characters as objects, thus forcing different dtypes
            self.assertEqualFrames(frame, b2[frame.name], compare_dtype=False)

    def test_bus_to_parquet_b(self) -> None:
        dt64 = np.datetime64

        f1 = Frame.from_dict(
            dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name=dt64('2020-01-01')
        )
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)),
            index=('x', 'y', 'z'),
            name=dt64('1932-12-17'),
        )
        f3 = Frame.from_dict(
            dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name=dt64('1950-04-23')
        )

        config = StoreConfig(
            index_depth=1,
            columns_depth=1,
            include_columns=True,
            include_index=True,
            label_encoder=str,
            label_decoder=dt64,
        )

        b1 = Bus.from_frames((f1, f2, f3), config=config, index_constructor=IndexDate)
        self.assertEqual(b1.index.dtype.kind, 'M')  # type: ignore

        with temp_file('.zip') as fp:
            b1.to_zip_parquet(fp, config=config)

            b2 = Bus.from_zip_parquet(fp, config=config, index_constructor=IndexDate)
            self.assertEqual(b2.index.dtype.kind, 'M')  # type: ignore

            key = dt64('2020-01-01')
            self.assertEqualFrames(b1[key], b2[key], compare_dtype=False)

    def test_bus_to_parquet_c(self) -> None:
        f1 = ff.parse('s(4,4)|i(ID,dtD)|v(int64)').rename('a')
        f2 = ff.parse('s(4,4)|i(ID,dtD)|v(int64)').rename('b')

        config = StoreConfig(
            index_depth=1,
            include_index=True,
            index_constructors=IndexDate,
            columns_depth=1,
            include_columns=True,
        )
        b1 = Bus.from_frames((f1, f2))
        with temp_file('.zip') as fp:
            b1.to_zip_parquet(fp, config=config)
            b2 = Bus.from_zip_parquet(fp, config=config)
            self.assertIs(b2['a'].index.__class__, IndexDate)
            self.assertIs(b2['b'].index.__class__, IndexDate)

    # ---------------------------------------------------------------------------

    def test_bus_max_persist_a(self) -> None:
        def items() -> tp.Iterator[tp.Tuple[str, Frame]]:
            for i in range(20):
                yield str(i), Frame(np.arange(i, i + 10).reshape(2, 5))

        s = Series.from_items(items(), dtype=object)
        b1 = Bus.from_series(s)

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)

            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=3)
            for i in b2.index:
                _ = b2[i]
                self.assertTrue(b2._loaded.sum() <= 3)

            # after iteration only the last three are loaded
            self.assertEqual(
                b2._loaded.tolist(),
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                ],
            )

    def test_bus_max_persist_b(self) -> None:
        def items() -> tp.Iterator[tp.Tuple[str, Frame]]:
            for i in range(20):
                yield str(i), Frame(np.arange(i, i + 10).reshape(2, 5))

        s = Series.from_items(items(), dtype=object)
        b1 = Bus.from_series(s)

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)

            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=1)
            b3 = b2.iloc[10:]
            self.assertEqual(b3._loaded.sum(), 0)
            # only the last one is loasded
            self.assertEqual(
                b3._loaded.tolist(),
                [False, False, False, False, False, False, False, False, False, False],
            )
            self.assertEqual(b3.iloc[0].sum().sum(), 145)  # type: ignore
            self.assertEqual(
                b3._loaded.tolist(),
                [True, False, False, False, False, False, False, False, False, False],
            )
            self.assertEqual(b3.iloc[4].sum().sum(), 185)  # type: ignore
            self.assertEqual(
                b3._loaded.tolist(),
                [False, False, False, False, True, False, False, False, False, False],
            )

    def test_bus_max_persist_c(self) -> None:
        def items() -> tp.Iterator[tp.Tuple[str, Frame]]:
            for i in range(4):
                yield str(i), Frame(np.arange(i, i + 10).reshape(2, 5))

        s = Series.from_items(items(), dtype=object)
        b1 = Bus.from_series(s)

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)

            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=4)

            for _ in b2.items():
                pass
            self.assertTrue(b2._loaded.all())

            b3 = Bus.from_zip_pickle(fp, config=config, max_persist=3)

            _ = b3.iloc[[0, 2, 3]]
            self.assertEqual(b3._loaded.any(), False)

            _ = b3.iloc[[0, 1, 3]]
            self.assertEqual(b3._loaded.any(), False)

            _ = b3.iloc[[1, 2, 3]]
            self.assertEqual(b3._loaded.any(), False)

            _ = b3.iloc[[0, 1, 2]]
            self.assertEqual(b3._loaded.any(), False)

    def test_bus_max_persist_d(self) -> None:
        def items() -> tp.Iterator[tp.Tuple[str, Frame]]:
            for i in range(5):
                yield str(i), Frame(np.arange(i, i + 10).reshape(2, 5))

        s = Series.from_items(items(), dtype=object)
        b1 = Bus.from_series(s)

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)

            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=3)

            _ = b2.iloc[[0, 2, 4]]
            self.assertEqual(b2._loaded.any(), False)

            _ = b2.iloc[[1, 2, 3]]
            self.assertEqual(b2._loaded.any(), False)

            _ = b2.iloc[4]
            self.assertEqual(b2._loaded.tolist(), [False, False, False, False, True])

            _ = b2.iloc[0]
            self.assertEqual(b2._loaded.tolist(), [True, False, False, False, True])

            _ = b2.iloc[[2, 3, 4]]
            self.assertEqual(b2._loaded.tolist(), [True, False, False, False, True])

            _ = b2.iloc[[0, 1]]
            self.assertEqual(b2._loaded.tolist(), [True, False, False, False, True])

            _ = b2.iloc[0]
            self.assertEqual(b2._loaded.tolist(), [True, False, False, False, True])

            _ = b2.iloc[[3, 4]]
            self.assertEqual(b2._loaded.tolist(), [True, False, False, False, True])

    def test_bus_max_persist_e(self) -> None:
        def items() -> tp.Iterator[tp.Tuple[str, Frame]]:
            for i in range(4):
                yield str(i), Frame(np.arange(i, i + 10).reshape(2, 5))

        s = Series.from_items(items(), dtype=object)
        b1 = Bus.from_series(s)

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=4)

            _ = b2.iloc[[0, 1]]
            _ = b2.iloc[[2, 3]]
            self.assertFalse(b2._loaded_all)

            _ = b2.iloc[[1, 0]]
            self.assertEqual(list(b2._last_loaded.keys()), [])

            _ = b2.iloc[3]
            self.assertEqual(list(b2._last_loaded.keys()), ['3'])

            _ = b2.iloc[:3]
            self.assertEqual(list(b2._last_loaded.keys()), ['3'])

            b3 = b2.iloc[:3]
            _ = list(b3.values)
            self.assertEqual(list(b3._last_loaded.keys()), ['0', '1', '2'])

    def test_bus_max_persist_f(self) -> None:
        def items() -> tp.Iterator[tp.Tuple[str, Frame]]:
            for i in range(5):
                yield str(i), Frame(np.arange(i, i + 10).reshape(2, 5))

        s = Series.from_items(items(), dtype=object)
        b1 = Bus.from_series(s)

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)

            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=1)

            # insure that items delivers Frame, not FrameDeferred
            post = tuple(b2.items())
            self.assertEqual(len(post), 5)
            self.assertTrue(all(isinstance(f, Frame) for _, f in post))
            self.assertEqual(b2._loaded.tolist(), [False, False, False, False, True])

    def test_bus_max_persist_g(self) -> None:
        def items() -> tp.Iterator[tp.Tuple[str, Frame]]:
            for i in range(5):
                yield str(i), Frame(np.arange(i, i + 10).reshape(2, 5))

        s = Series.from_items(items(), dtype=object)
        b1 = Bus.from_series(s)

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)

            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=1)
            # insure that items delivers Frame, not FrameDeferred
            post = b2.values
            self.assertEqual(len(post), 5)
            self.assertTrue(all(isinstance(f, Frame) for f in post))
            self.assertEqual(b2._loaded.tolist(), [False, False, False, False, True])

    def test_bus_max_persist_h(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        s1 = Series((f1, f2, f3), index=('a', 'b', 'c'))
        with self.assertRaises(ErrorInitBus):
            # max_persist cannot be less than the number of already loaded Frames
            Bus.from_series(s1, max_persist=2)

    def test_bus_max_persist_i(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=2)
            # NOTE: this type of selection forces _store_reader to use read_many at size of max_persist
            b3 = b2[['f1', 'f2', 'f3', 'f4', 'f5']]
            self.assertEqual(
                b3.status.loc[b3.status['loaded'], 'shape'].to_pairs(),
                (),
            )

    def test_bus_max_persist_j(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=2)
            # NOTE: this type of selection forces _store_reader to use read_many at size of max_persist
            a1 = b2.values
            self.assertNotEqual(id(a1), id(b2._values_mutable))
            self.assertEqual(b2.status['loaded'].sum(), 2)
            self.assertTrue(all(f.__class__ is Frame for f in a1))

    def test_bus_max_persist_k1(self) -> None:
        b1 = Bus.from_frames(
            [
                Frame(
                    np.arange(9).reshape(3, 3) * i,
                    index=range(3),
                    name=f'f{i}',
                )
                for i in range(1, 4)
            ],
        )
        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)

            b2 = Bus.from_zip_npz(fp, max_persist=2)
            _ = b2.iloc[1]
            _ = b2.iloc[2]
            self.assertEqual(
                b2.status.index[b2.status['loaded']].tolist(),
                ['f2', 'f3'],
            )
            b3 = b2.iloc[:2]
            self.assertEqual(
                b2.status.index[b2.status['loaded']].tolist(),
                ['f2', 'f3'],
            )
            # sliced bus retains previous loaded, which was only f2 an f3; and now is only f2
            self.assertEqual(
                b3.status.index[b3.status['loaded']].tolist(),
                ['f2'],
            )

    def test_bus_max_persist_k2(self) -> None:
        b1 = Bus.from_frames(
            [
                Frame(
                    np.arange(9).reshape(3, 3) * i,
                    index=range(3),
                    name=f'f{i}',
                )
                for i in range(1, 4)
            ],
        )
        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)

            b2 = Bus.from_zip_npz(fp, max_persist=2)
            self.assertEqual(len(list(b2.items())), 3)
            self.assertEqual(
                b2.status.index[b2.status['loaded']].tolist(),
                ['f2', 'f3'],
            )

    def test_bus_max_persist_l(self) -> None:
        b1 = Bus.from_frames(
            [
                Frame(
                    np.arange(9).reshape(3, 3) * i,
                    index=range(3),
                    name=f'f{i}',
                )
                for i in range(1, 7)
            ],
        )
        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)
            b2 = Bus.from_zip_npz(fp, max_persist=2)
            _ = b2.iloc[1]
            _ = b2.iloc[5]
            b3 = b2.iloc[2:]
            self.assertEqual(len(b3), 4)
            self.assertEqual(
                b3.status.index[b3.status['loaded']].tolist(),
                ['f6'],
            )

    def test_bus_max_persist_m(self) -> None:
        b1 = Bus.from_frames(
            [
                Frame(
                    np.arange(9).reshape(3, 3) * i,
                    index=range(3),
                    name=('a', i),
                )
                for i in range(1, 7)
            ],
            index_constructor=IndexHierarchy.from_labels,
        )

        config = StoreConfig(
            label_encoder=str,
            label_decoder=ast.literal_eval,
        )

        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp, config=config)
            b2 = Bus.from_zip_npz(
                fp,
                max_persist=2,
                config=config,
                index_constructor=IndexHierarchy.from_labels,
            )
            _ = b2.iloc[1]
            _ = b2.iloc[5]
            self.assertEqual(
                list(b2.status.index[b2.status['loaded']]),
                [('a', 2), ('a', 6)],
            )

            b3 = b2.loc[('a', 3) :]
            self.assertEqual(len(b3), 4)
            self.assertEqual(
                list(b3.status.index[b3.status['loaded']]),
                [('a', 6)],
            )

    def test_bus_max_persist_n(self) -> None:
        f1 = Frame.from_dict(
            dict(a=(1, 2), b=(3, 4)),
            index=('x', 'y'),
            name='f1',
        )
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)),
            index=('x', 'y', 'z'),
            name='f2',
        )
        f3 = Frame.from_dict(
            dict(d=(10, 20), b=(50, 60)),
            index=('p', 'q'),
            name='f3',
        )
        f4 = Frame.from_dict(
            dict(q=(None, None), r=(np.nan, np.nan)),
            index=(1000, 1001),
            name='f4',
        )

        with temp_file('.zip') as fp:
            Batch.from_frames((f1, f2, f3, f4)).to_zip_pickle(fp)
            b1 = Bus.from_zip_pickle(fp, max_persist=2)
            for _ in range(4):
                _ = b1[:]

            b2 = Bus.from_zip_pickle(fp, max_persist=2)
            for _ in range(4):
                _ = b2.iloc[list(range(len(b2)))]

            b3 = Bus.from_zip_pickle(fp, max_persist=2)
            for _ in range(4):
                _ = b3[b3.index]

    def test_bus_max_persist_o1(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=1)
            post = b2.values
            assert all(isinstance(x, Frame) for x in post)
            self.assertEqual(
                b2._loaded.tolist(), [False, False, False, False, False, True]
            )
            self.assertEqual(list(b2._last_loaded.keys()), ['f6'])

    def test_bus_max_persist_o2(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=6)
            post = b2.values
            assert all(isinstance(x, Frame) for x in post)
            assert b2._loaded.all()
            self.assertEqual(
                list(b2._last_loaded.keys()), ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
            )
            self.assertTrue(b2._loaded_all)

    def test_bus_max_persist_o3(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=6)
            _ = b2['f1']
            _ = b2['f2']
            _ = b2['f3']
            _ = b2['f4']
            _ = b2['f5']
            _ = b2['f6']
            assert b2._loaded.all()
            self.assertEqual(
                list(b2._last_loaded.keys()), ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
            )
            self.assertTrue(b2._loaded_all)

    def test_bus_max_persist_o4(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=2)
            _ = b2['f1']
            _ = b2['f2']
            self.assertEqual(list(b2._last_loaded.keys()), ['f1', 'f2'])
            _ = b2['f3']
            self.assertEqual(list(b2._last_loaded.keys()), ['f2', 'f3'])
            _ = b2['f4']
            self.assertEqual(list(b2._last_loaded.keys()), ['f3', 'f4'])
            _ = b2['f5']
            self.assertEqual(list(b2._last_loaded.keys()), ['f4', 'f5'])
            self.assertFalse(b2._loaded_all)

    def test_bus_max_persist_p1(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3))

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=3)
            post1 = b2.items()
            assert all(isinstance(x, Frame) for _, x in post1)
            self.assertTrue(b2._loaded.all())
            self.assertTrue(b2._loaded_all)

            post2 = b2.items()  # hits _loaded_all
            assert all(isinstance(x, Frame) for _, x in post2)

    def test_bus_max_persist_p2(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        ih = IndexHierarchy.from_labels((('a', 1), ('b', 2), ('b', 1)))
        s1 = Series((f1, f2, f3), index=ih, dtype=object)
        b1 = Bus.from_series(s1)
        config = StoreConfig(label_encoder=str, label_decoder=literal_eval)

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp, config=config)
            b2 = Bus.from_zip_pickle(
                fp,
                config=config,
                max_persist=2,
                index_constructor=IndexHierarchy.from_labels,
            )
            post1 = b2.items()
            assert all(isinstance(x, Frame) for _, x in post1)
            self.assertFalse(b2._loaded.all())
            self.assertEqual(list(b2._last_loaded.keys()), [('b', 2), ('b', 1)])

    def test_bus_max_persist_q1(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))
        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=4)
            _ = b2['f3']
            _ = b2['f2']
            self.assertEqual(list(b2._last_loaded.keys()), ['f3', 'f2'])

            post = list(b2.items())
            self.assertEqual(len(post), len(b2))
            self.assertEqual(list(b2._last_loaded.keys()), ['f1', 'f4', 'f5', 'f6'])
            self.assertFalse(b2._loaded_all)

    def test_bus_max_persist_q2(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')
        f7 = ff.parse('s(5,4)').rename('f7')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6, f7))
        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=4)
            _ = b2['f3']
            _ = b2['f2']
            self.assertEqual(list(b2._last_loaded.keys()), ['f3', 'f2'])

            post = list(b2.items())
            self.assertEqual(
                [f.name for _, f in post], ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']
            )
            self.assertEqual(len(post), len(b2))
            self.assertEqual(list(b2._last_loaded.keys()), ['f4', 'f5', 'f6', 'f7'])
            self.assertFalse(b2._loaded_all)

    def test_bus_max_persist_r1(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')
        f7 = ff.parse('s(2,4)').rename('f7')
        f8 = ff.parse('s(5,4)').rename('f8')
        f9 = ff.parse('s(2,7)').rename('f9')
        f10 = ff.parse('s(8,2)').rename('f10')
        f11 = ff.parse('s(4,9)').rename('f11')
        f12 = ff.parse('s(4,6)').rename('f12')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12))

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)

            b1.to_zip_npz(fp)
            b2 = Bus.from_zip_npz(fp, max_persist=3)

            for iloc in [2, 3]:
                self.assertIsInstance(b2.iloc[iloc], Frame)

            items = list(b2.items())

    # ---------------------------------------------------------------------------
    def test_bus_persistant_a1(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3))

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp, config=config)
            f = b2.loc['f2']
            self.assertFalse(b2._loaded.all())
            self.assertFalse(b2._loaded_all)
            f = b2.loc['f2']  # load check on previously loaded

    def test_bus_persistant_a2(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        ih = IndexHierarchy.from_labels((('a', 1), ('b', 2), ('b', 1)))
        s1 = Series((f1, f2, f3), index=ih, dtype=object)
        b1 = Bus.from_series(s1)
        config = StoreConfig(label_encoder=str, label_decoder=literal_eval)

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp, config=config)
            b2 = Bus.from_zip_pickle(
                fp, config=config, index_constructor=IndexHierarchy.from_labels
            )
            post1 = b2.items()
            assert all(isinstance(x, Frame) for _, x in post1)
            self.assertTrue(b2._loaded.all())

    def test_bus_persistant_b1(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')
        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp, config=config)

            f = b2['f3']
            values = b2.values
            self.assertEqual(len(values), len(b2))

    def test_bus_persistant_b2(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')
        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp, config=config)

            _ = b2['f3']
            _ = b2['f1']
            _ = b2['f6']
            self.assertEqual(b2._loaded.tolist(), [True, False, True, False, False, True])

            values = list(b2.items())
            self.assertEqual(len(values), len(b2))

    # ---------------------------------------------------------------------------
    def test_bus_sort_index_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f3, f2, f1))
        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp)
            b3 = b2.sort_index()
            self.assertEqual(b2.index.values.tolist(), ['f3', 'f2', 'f1'])
            self.assertEqual(b3.index.values.tolist(), ['f1', 'f2', 'f3'])

            self.assertTrue(isinstance(b3, Bus))

            f4 = b3['f2']
            self.assertEqual(
                f4.to_pairs(),  # type: ignore
                (
                    ('c', (('x', 1), ('y', 2), ('z', 3))),
                    ('b', (('x', 4), ('y', 5), ('z', 6))),
                ),
            )

            f5 = b3['f1']
            self.assertEqual(
                f5.to_pairs(),  # type: ignore
                (('a', (('x', 1), ('y', 2))), ('b', (('x', 3), ('y', 4)))),
            )

    # ---------------------------------------------------------------------------

    def test_bus_sort_values_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(30000, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(1, 2), b=(5, 6)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f3, f2, f1))
        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp)

            b3 = b2.sort_values(
                key=lambda b: b.iter_element().apply(lambda f: f.sum().sum())
            )
            self.assertEqual(b3.index.values.tolist(), ['f3', 'f2', 'f1'])

    # ---------------------------------------------------------------------------

    def test_bus_iter_element_apply_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(30000, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(1, 2), b=(5, 6)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f3, f2, f1))
        b2 = b1.iter_element().apply(lambda f: f.sum().sum())

        self.assertEqual(
            b2.to_pairs(),  # type: ignore
            (('f3', 14), ('f2', 21), ('f1', 30007)),
        )

        self.assertEqual(id(b1.index), id(b2.index))

    # ---------------------------------------------------------------------------

    def test_bus_iter_element_reduce_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(30000, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(a=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(a=(1, 2), b=(5, 6)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f3, f2, f1))
        f4 = (
            b1.iter_element()
            .reduce.from_label_pair_map(
                {
                    ('b', 'b-min'): np.min,
                    ('b', 'b-max'): np.max,
                    ('a', 'a-mean'): np.mean,
                }
            )
            .to_frame()
        )
        self.assertEqual(
            f4.to_pairs(),
            (
                ('b-min', (('f3', 5), ('f2', 4), ('f1', 4))),
                ('b-max', (('f3', 6), ('f2', 6), ('f1', 30000))),
                ('a-mean', (('f3', 1.5), ('f2', 2.0), ('f1', 1.5))),
            ),
        )

    def test_bus_iter_element_reduce_b(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(30000, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(a=(1, 2, 3), c=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(a=(1, 2), b=(5, 6)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f3, f2, f1))
        f4 = (
            b1.iter_element()
            .reduce.from_label_pair_map(
                {
                    ('b', 'b-min'): np.min,
                    ('b', 'b-max'): np.max,
                    ('a', 'a-mean'): np.mean,
                },
                fill_value=-1,
            )
            .to_frame()
        )
        self.assertEqual(
            f4.to_pairs(),
            (
                ('b-min', (('f3', 5), ('f2', -1), ('f1', 4))),
                ('b-max', (('f3', 6), ('f2', -1), ('f1', 30000))),
                ('a-mean', (('f3', 1.5), ('f2', 2.0), ('f1', 1.5))),
            ),
        )

    def test_bus_iter_element_reduce_c(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(30000, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(a=(1, 2, 3), c=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(c=(1, 2), b=(5, 6)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f3, f2, f1))
        f4 = (
            b1.iter_element()
            .reduce.from_label_pair_map(
                {
                    ('a', 'b-min'): np.min,
                    ('b', 'b-max'): np.max,
                    ('c', 'c-mean'): np.mean,
                },
                fill_value=-1,
            )
            .to_frame()
        )
        self.assertEqual(
            f4.to_pairs(),
            (
                ('b-min', (('f3', -1), ('f2', 1), ('f1', 1))),
                ('b-max', (('f3', 6), ('f2', -1), ('f1', 30000))),
                ('c-mean', (('f3', 1.5), ('f2', 5.0), ('f1', -1.0))),
            ),
        )

    def test_bus_iter_element_reduce_d1(self) -> None:
        f1 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('x')
            .relabel(columns=('a', 'b', 'c'))
        )
        f2 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('y')
            .relabel(columns=('a', 'b', 'c'))
            * 2
        )
        f3 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('z')
            .relabel(columns=('a', 'b', 'c'))
            * -10
        )
        b = Bus.from_frames((f1, f2, f3))

        f1 = b.iter_element().reduce.from_label_map({'a': np.sum, 'b': np.min}).to_frame()
        self.assertEqual(
            f1.to_pairs(),
            (
                ('a', (('x', 89817), ('y', 179634), ('z', -898170))),
                ('b', (('x', -41157), ('y', -82314), ('z', -1621970))),
            ),
        )

    def test_bus_iter_element_reduce_d2(self) -> None:
        f1 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('x')
            .relabel(columns=('a', 'b', 'c'))
        )
        f2 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('y')
            .relabel(columns=('a', 'b', 'c'))
            * 2
        )
        f3 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('z')
            .relabel(columns=('a', 'b', 'c'))
            * -10
        )
        b = Bus.from_frames((f1, f2, f3))

        f4 = (
            b.iter_element()
            .reduce.from_label_pair_map(
                {('a', 'a-sum'): np.sum, ('b', 'b-min'): np.min, ('b', 'b-max'): np.max}
            )
            .to_frame()
        )
        self.assertEqual(
            f4.to_pairs(),
            (
                ('a-sum', (('x', 89817), ('y', 179634), ('z', -898170))),
                ('b-min', (('x', -41157), ('y', -82314), ('z', -1621970))),
                ('b-max', (('x', 162197), ('y', 324394), ('z', 411570))),
            ),
        )

    def test_bus_iter_element_reduce_d3(self) -> None:
        f1 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('x')
            .relabel(columns=('a', 'b', 'c'))
        )
        f2 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('y')
            .relabel(columns=('a', 'b', 'c'))
            * 2
        )
        f3 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('z')
            .relabel(columns=('a', 'b', 'c'))
            * -10
        )
        b = Bus.from_frames((f1, f2, f3))
        f4 = b.iter_element().reduce.from_map_func(np.sum).to_frame()
        self.assertEqual(
            f4.to_pairs(),
            (
                ('a', (('x', 89817), ('y', 179634), ('z', -898170))),
                ('b', (('x', 126769), ('y', 253538), ('z', -1267690))),
                ('c', (('x', 117858), ('y', 235716), ('z', -1178580))),
            ),
        )

    def test_bus_iter_element_reduce_d4(self) -> None:
        f1 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('x')
            .relabel(columns=('a', 'b', 'c'))
        )
        f2 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('y')
            .relabel(columns=('a', 'b', 'c'))
            * 2
        )
        f3 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('z')
            .relabel(columns=('a', 'b', 'c'))
            * -10
        )
        b = Bus.from_frames((f1, f2, f3))

        f4 = (
            b.iter_element()
            .reduce.from_func(lambda f: f.iloc[1:, 1:])
            .to_frame(index=IndexAutoFactory)
        )
        self.assertEqual(
            f4.to_pairs(),
            (
                (
                    'b',
                    (
                        (0, -41157),
                        (1, 5729),
                        (2, -82314),
                        (3, 11458),
                        (4, 411570),
                        (5, -57290),
                    ),
                ),
                (
                    'c',
                    (
                        (0, 91301),
                        (1, 30205),
                        (2, 182602),
                        (3, 60410),
                        (4, -913010),
                        (5, -302050),
                    ),
                ),
            ),
        )

    def test_bus_iter_element_reduce_e1(self) -> None:
        f1 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('x')
            .relabel(columns=('a', 'b', 'c'))
        )
        f2 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('y')
            .relabel(columns=('a', 'b', 'c'))
            * 2
        )
        f3 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('z')
            .relabel(columns=('a', 'b', 'c'))
            * -10
        )
        b = Bus.from_frames((f1, f2, f3))
        f4 = (
            b.iter_element_items()
            .reduce.from_label_map(dict(a=lambda l, v: np.sum(v), c=lambda l, v: l))
            .to_frame()
        )
        self.assertEqual(
            f4.to_pairs(),
            (
                ('a', (('x', 89817), ('y', 179634), ('z', -898170))),
                ('c', (('x', 'x'), ('y', 'y'), ('z', 'z'))),
            ),
        )

    def test_bus_iter_element_reduce_e2(self) -> None:
        f1 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('x')
            .relabel(columns=('x', 'b', 'c'))
        )
        f2 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('y')
            .relabel(columns=('a', 'x', 'c'))
            * 2
        )
        f3 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('z')
            .relabel(columns=('a', 'b', 'x'))
            * -10
        )
        b = Bus.from_frames((f1, f2, f3))
        f4 = (
            b.iter_element_items()
            .reduce.from_label_map(
                dict(a=lambda l, v: np.sum(v), c=lambda l, v: l), fill_value=-1
            )
            .to_frame()
        )
        self.assertEqual(
            f4.to_pairs(),
            (
                ('a', (('x', -1), ('y', 179634), ('z', -898170))),
                ('c', (('x', 'x'), ('y', 'y'), ('z', -1))),
            ),
        )

    def test_bus_iter_element_reduce_f1(self) -> None:
        f1 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('x')
            .relabel(columns=('x', 'b', 'c'))
        )
        f2 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('y')
            .relabel(columns=('a', 'x', 'c'))
            * 2
        )
        f3 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('z')
            .relabel(columns=('a', 'b', 'x'))
            * -10
        )
        b = Bus.from_frames((f1, f2, f3))
        post = list(
            b.iter_element()
            .reduce.from_label_map(dict(a=np.sum, c=np.sum), fill_value=-1)
            .values()
        )
        self.assertEqual(post[0].to_pairs(), (('a', -1), ('c', 117858)))
        self.assertEqual(post[1].to_pairs(), (('a', 179634), ('c', 235716)))

    def test_bus_iter_element_reduce_f2(self) -> None:
        f1 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('x')
            .relabel(columns=('x', 'b', 'c'))
        )
        f2 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('y')
            .relabel(columns=('a', 'x', 'c'))
            * 2
        )
        f3 = (
            ff.parse('s(3,3)|i(I,str)|v(int64)')
            .rename('z')
            .relabel(columns=('a', 'b', 'x'))
            * -10
        )
        b = Bus.from_frames((f1, f2, f3))
        post = list(
            b.iter_element_items()
            .reduce.from_label_map(
                dict(a=lambda l, v: np.sum(v), c=lambda l, v: l), fill_value=''
            )
            .values()
        )
        self.assertEqual(post[0].to_pairs(), (('a', ''), ('c', 'x')))
        self.assertEqual(post[1].to_pairs(), (('a', 179634), ('c', 'y')))

    def test_bus_iter_element_reduce_g(self) -> None:
        b = Bus.from_frames(
            (
                Frame(
                    np.arange(6).reshape(3, 2),
                    index=('p', 'q', 'r'),
                    columns=('a', 'b'),
                    name='x',
                ),
                Frame(
                    (np.arange(6).reshape(3, 2) % 2).astype(bool),
                    index=('p', 'q', 'r'),
                    columns=('c', 'd'),
                    name='y',
                ),
                Frame(
                    np.arange(40, 46).reshape(3, 2),
                    index=('p', 'q', 'r'),
                    columns=('a', 'b'),
                    name='v',
                ),
                Frame(
                    (np.arange(6).reshape(3, 2) % 3).astype(bool),
                    index=('p', 'q', 'r'),
                    columns=('c', 'd'),
                    name='w',
                ),
            ),
            name='k',
        )

        post = (
            b.iter_element_items()
            .reduce.from_map_func(lambda s: np.min(s), fill_value=0)
            .to_frame()
        )
        self.assertEqual(
            post.to_pairs(),
            (
                (
                    np.str_('a'),
                    (
                        (np.str_('x'), np.int64(0)),
                        (np.str_('y'), np.int64(0)),
                        (np.str_('v'), np.int64(40)),
                        (np.str_('w'), np.int64(0)),
                    ),
                ),
                (
                    np.str_('b'),
                    (
                        (np.str_('x'), np.int64(1)),
                        (np.str_('y'), np.int64(0)),
                        (np.str_('v'), np.int64(41)),
                        (np.str_('w'), np.int64(0)),
                    ),
                ),
                (
                    np.str_('c'),
                    (
                        (np.str_('x'), 0),
                        (np.str_('y'), False),
                        (np.str_('v'), 0),
                        (np.str_('w'), False),
                    ),
                ),
                (
                    np.str_('d'),
                    (
                        (np.str_('x'), 0),
                        (np.str_('y'), True),
                        (np.str_('v'), 0),
                        (np.str_('w'), False),
                    ),
                ),
            ),
        )

    # ---------------------------------------------------------------------------

    def test_bus_drop_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f3, f2, f1))
        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp)

            b3 = b2.drop[['f3', 'f1']]
            self.assertEqual(b3.status['loaded'].sum(), 0)
            self.assertEqual(
                b3['f2'].to_pairs(),
                (
                    ('c', (('x', 1), ('y', 2), ('z', 3))),
                    ('b', (('x', 4), ('y', 5), ('z', 6))),
                ),
            )

    # ---------------------------------------------------------------------------

    def test_bus_from_dict_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )

        b1 = Bus.from_dict(dict(a=f1, b=f2))
        self.assertEqual(b1.index.values.tolist(), ['a', 'b'])

    def test_bus_from_dict_b(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )

        b1 = Bus.from_dict(
            {'2021-01': f1, '2021-06': f2}, index_constructor=IndexYearMonth
        )
        self.assertIs(b1.index.__class__, IndexYearMonth)

    # ---------------------------------------------------------------------------

    def test_bus_iter_element_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )

        b1 = Bus.from_dict(dict(a=f1, b=f2))
        post = tuple(b1.iter_element())
        self.assertEqual(len(post), 2)
        self.assertEqual(
            post[1].to_pairs(),
            (
                ('c', (('x', 1), ('y', 2), ('z', 3))),
                ('b', (('x', 4), ('y', 5), ('z', 6))),
            ),
        )

    def test_bus_iter_element_b(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))
        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=2)

            frames = list(b2.iter_element())
            self.assertTrue(all(x.__class__ is Frame for x in frames))
            self.assertEqual(len(frames), 6)
            self.assertEqual(b2.status['loaded'].sum(), 2)

    def test_bus_iter_element_c(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))
        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=1)

            frames = list(b2.iter_element())
            self.assertTrue(all(x.__class__ is Frame for x in frames))
            self.assertEqual(len(frames), 6)
            self.assertEqual(b2.status['loaded'].sum(), 1)

    def test_bus_iter_element_d(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))
        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp, config=config)

            frames = list(b2.iter_element())
            self.assertTrue(all(x.__class__ is Frame for x in frames))
            self.assertEqual(len(frames), 6)
            self.assertEqual(b2.status['loaded'].sum(), 6)

    # ---------------------------------------------------------------------------

    def test_bus_iter_element_items_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )

        b1 = Bus.from_dict(dict(a=f1, b=f2))
        post = tuple(b1.iter_element_items())
        self.assertEqual(len(post), 2)
        self.assertEqual([pair[0] for pair in post], ['a', 'b'])
        self.assertTrue(post[0][1].equals(f1))
        self.assertTrue(post[1][1].equals(f2))

    # ---------------------------------------------------------------------------

    def test_bus_reindex_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f3, f2, f1))
        b2 = b1.reindex(('f1', 'f2', 'f4'), fill_value=f3)
        self.assertTrue(b2['f4'].equals(f3))
        self.assertTrue(b2.__class__ is Bus)

    def test_bus_reindex_b(self) -> None:
        def items() -> tp.Iterator[tp.Tuple[str, Frame]]:
            for i in range(5):
                yield 'abcde'[i], Frame(np.arange(i, i + 10).reshape(2, 5))

        s = Series.from_items(items(), dtype=object)
        b1 = Bus.from_series(s)

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            # set max_persist to size to test when fully loaded with max_persist
            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=1)

            b3 = b2.reindex(('c', 'd', 'q'), fill_value=Frame())
            f1 = b3['c']
            # this would fail if the Store was still associated with the Bus
            f2 = b3['q']

            self.assertEqual(b3._max_persist, None)
            self.assertEqual(b3._store, None)
            self.assertEqual(b2._max_persist, 1)

    # ---------------------------------------------------------------------------

    def test_bus_relabel_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f3, f2, f1))
        b2 = b1.relabel(('a', 'b', 'c'))

        self.assertTrue(b2['a'].equals(f3))
        self.assertTrue(b2.__class__ is Bus)

    # ---------------------------------------------------------------------------

    def test_bus_relabel_flat_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f3, f2, f1))
        b2 = b1.relabel_level_add('X')
        self.assertEqual(
            b2.relabel_flat().index.values.tolist(),
            [('X', 'f3'), ('X', 'f2'), ('X', 'f1')],
        )

        self.assertTrue(b2[('X', 'f2')].equals(f2))
        self.assertTrue(b2.__class__ is Bus)

    # ---------------------------------------------------------------------------

    def test_bus_relabel_level_drop_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f3, f2, f1))
        b2 = b1.relabel_level_add('X')
        b3 = b2.relabel_level_drop(1)
        self.assertTrue(b3.__class__ is Bus)
        self.assertTrue(b3.index.equals(b1.index))

    # ---------------------------------------------------------------------------

    def test_bus_relabel_rehierarch_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f3, f2, f1))
        b2 = b1.relabel_level_add('X')
        b3 = b2.rehierarch([1, 0])
        self.assertEqual(
            b3.index.values.tolist(), [['f3', 'X'], ['f2', 'X'], ['f1', 'X']]
        )

    # ---------------------------------------------------------------------------

    def test_bus_roll_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f3, f2, f1))
        b2 = b1.roll(2, include_index=True)
        self.assertEqual(b2.index.values.tolist(), ['f2', 'f1', 'f3'])
        self.assertTrue(b2['f2'].equals(b1['f2']))

    # ---------------------------------------------------------------------------

    def test_bus_shift_a(self) -> None:
        f1 = Frame.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='f1')
        f2 = Frame.from_dict(
            dict(c=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='f2'
        )
        f3 = Frame.from_dict(dict(d=(10, 20), b=(50, 60)), index=('p', 'q'), name='f3')

        b1 = Bus.from_frames((f1, f2, f3))
        b2 = b1.shift(2, fill_value=f1)
        self.assertTrue(b2['f3'].equals(b1['f1']))
        self.assertTrue(b2['f2'].equals(b1['f1']))
        self.assertTrue(b2['f1'].equals(b1['f1']))

    # ---------------------------------------------------------------------------

    def test_bus_from_concat_a(self) -> None:
        def items() -> tp.Iterator[tp.Tuple[str, Frame]]:
            for i in range(5):
                yield str(i), Frame(np.arange(i, i + 10).reshape(2, 5))

        s = Series.from_items(items(), dtype=object)
        b1 = Bus.from_series(s)

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=1)

            b3 = b1.relabel(('a', 'b', 'c', 'd', 'e'))

            # fully loaded in memory
            b4 = Bus.from_concat((b2, b3))

            self.assertEqual(
                b4.status['loaded'].to_pairs(),
                (
                    ('0', True),
                    ('1', True),
                    ('2', True),
                    ('3', True),
                    ('4', True),
                    ('a', True),
                    ('b', True),
                    ('c', True),
                    ('d', True),
                    ('e', True),
                ),
            )
            self.assertEqual(b2.status['loaded'].sum(), 1)

            b5 = Bus.from_concat((b2, b3), name='foo', index=IndexAutoFactory)

            self.assertEqual(
                b5.status['loaded'].to_pairs(),
                (
                    (0, True),
                    (1, True),
                    (2, True),
                    (3, True),
                    (4, True),
                    (5, True),
                    (6, True),
                    (7, True),
                    (8, True),
                    (9, True),
                ),
            )
            self.assertEqual(b5.name, 'foo')

    def test_bus_from_concat_b(self) -> None:
        def items() -> tp.Iterator[tp.Tuple[str, Frame]]:
            for i in range(5):
                yield str(i), Frame(np.arange(i, i + 10).reshape(2, 5))

        s = Series.from_items(items(), dtype=object)
        b1 = Bus.from_series(s)

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            # set max_persist to size to test when fully loaded with max_persist
            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=5)

            b3 = b1.relabel(('a', 'b', 'c', 'd', 'e'))

            # fully loaded in memory
            b4 = Bus.from_concat((b2, b3))

            self.assertEqual(
                b4.status['loaded'].to_pairs(),
                (
                    ('0', True),
                    ('1', True),
                    ('2', True),
                    ('3', True),
                    ('4', True),
                    ('a', True),
                    ('b', True),
                    ('c', True),
                    ('d', True),
                    ('e', True),
                ),
            )
            self.assertEqual(b2.status['loaded'].sum(), 5)

    def test_bus_from_concat_c(self) -> None:
        def items() -> tp.Iterator[tp.Tuple[str, Frame]]:
            for i in range(5):
                yield str(i), Frame(np.arange(i, i + 10).reshape(2, 5))

        s = Series.from_items(items(), dtype=object)
        b1 = Bus.from_series(s)

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            # set max_persist to size to test when fully loaded with max_persist
            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=1)

            s1 = b2.to_series()
            self.assertEqual(b2.status['loaded'].sum(), 1)

            self.assertEqual(
                [f.shape for f in s1.values], [(2, 5), (2, 5), (2, 5), (2, 5), (2, 5)]
            )

    # ---------------------------------------------------------------------------

    def test_bus_contains_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        b1 = Bus.from_frames((f1, f2))

        self.assertTrue('f1' in b1)
        self.assertFalse('f3' in b1)

    # ---------------------------------------------------------------------------

    def test_bus_get_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        b1 = Bus.from_frames((f1, f2))

        self.assertTrue(b1.get('f1').equals(f1))
        self.assertEqual(b1.get('f3'), None)

    # ---------------------------------------------------------------------------

    def test_bus_head_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))

        self.assertEqual(b1.head().index.values.tolist(), ['f1', 'f2', 'f3', 'f4', 'f5'])

    # ---------------------------------------------------------------------------

    def test_bus_tail_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))

        self.assertEqual(b1.tail().index.values.tolist(), ['f2', 'f3', 'f4', 'f5', 'f6'])

    # ---------------------------------------------------------------------------

    def test_bus_unpersist_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            # set max_persist to size to test when fully loaded with max_persist
            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=3)
            _ = b2['f2']
            _ = b2['f3']
            _ = b2['f4']

            self.assertEqual(b2.status['loaded'].sum(), 3)
            b2.unpersist()

            self.assertEqual(b2.status['loaded'].sum(), 0)
            self.assertEqual(b2['f6'].shape, (6, 4))
            self.assertEqual(b2.status['loaded'].sum(), 1)

    def test_bus_unpersist_b(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))
        b1.unpersist()

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            # set max_persist to size to test when fully loaded with max_persist
            b2 = Bus.from_zip_pickle(fp, config=config, max_persist=2)
            list(b2.items())
            list(b2.items())
            self.assertEqual(b2.status['loaded'].sum(), 2)

            b2.unpersist()
            self.assertEqual(b2.status['loaded'].sum(), 0)

    def test_bus_unpersist_c(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))
        b1.unpersist()

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            # set max_persist to size to test when fully loaded with max_persist
            b2 = Bus.from_zip_pickle(fp, config=config)
            list(b2.items())
            self.assertEqual(b2.status['loaded'].sum(), 6)

            b2.unpersist()
            self.assertEqual(b2.status['loaded'].sum(), 0)

            b3 = b2[['f2', 'f5', 'f6']]
            self.assertEqual(b3.status['loaded'].sum(), 0)
            self.assertEqual(b2.status['loaded'].sum(), 0)

            b2.unpersist()
            self.assertEqual(b2.status['loaded'].sum(), 0)

    # ---------------------------------------------------------------------------

    def test_bus_npz_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))

        config = StoreConfig()

        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)
            # set max_persist to size to test when fully loaded with max_persist
            b2 = Bus.from_zip_npz(fp, config=config, max_persist=3)
            b3 = b2['f2':]
            self.assertEqual(
                b3['f5'].to_pairs(),  # type: ignore
                (
                    (0, ((0, 1930.4), (1, -1760.34), (2, 1857.34), (3, 1699.34))),
                    (1, ((0, -610.8), (1, 3243.94), (2, -823.14), (3, 114.58))),
                    (2, ((0, 694.3), (1, -72.96), (2, 1826.02), (3, 604.1))),
                    (3, ((0, 1080.4), (1, 2580.34), (2, 700.42), (3, 3338.48))),
                ),
            )

    def test_bus_npz_b(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3').astype(object)

        b1 = Bus.from_frames((f1, f2, f3))
        config = StoreConfig()

        with temp_file('.zip') as fp:
            with self.assertRaises(ErrorNPYEncode):
                b1.to_zip_npz(fp)
            self.assertFalse(os.path.exists(fp))

    def test_bus_npz_c(self) -> None:
        frame = Frame(
            np.random.normal(size=(2, 2)),
            columns=IndexAutoFactory,
            index=IndexAutoFactory,
            name=np.datetime64('2000-01-01'),
        )

        b1 = Bus.from_frames(
            (frame,),
            index_constructor=IndexDate,
        )

        config = StoreConfig(
            label_encoder=str,
            label_decoder=np.datetime64,
        )

        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp, config=config)
            b2 = Bus.from_zip_npz(
                fp, config=config, index_constructor=IndexAutoConstructorFactory
            )
            self.assertEqual(frame.name, b2.iloc[0].name)
            self.assertEqual(frame.shape, b2.iloc[0].shape)

    # ---------------------------------------------------------------------------
    def test_bus_npy_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))

        config = StoreConfig()

        with temp_file('.zip') as fp:
            b1.to_zip_npy(fp)
            # set max_persist to size to test when fully loaded with max_persist
            b2 = Bus.from_zip_npy(fp, config=config, max_persist=3)
            b3 = b2['f2':]
            self.assertEqual(
                b3['f5'].to_pairs(),  # type: ignore
                (
                    (0, ((0, 1930.4), (1, -1760.34), (2, 1857.34), (3, 1699.34))),
                    (1, ((0, -610.8), (1, 3243.94), (2, -823.14), (3, 114.58))),
                    (2, ((0, 694.3), (1, -72.96), (2, 1826.02), (3, 604.1))),
                    (3, ((0, 1080.4), (1, 2580.34), (2, 700.42), (3, 3338.48))),
                ),
            )

    def test_bus_npy_b(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3').astype(object)

        b1 = Bus.from_frames((f1, f2, f3))
        config = StoreConfig()

        with temp_file('.zip') as fp:
            with self.assertRaises(ErrorNPYEncode):
                b1.to_zip_npy(fp)
            self.assertFalse(os.path.exists(fp))

    # ---------------------------------------------------------------------------

    def test_bus_to_signature_bytes_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')

        b1 = Bus.from_frames((f1, f2, f3))
        bytes1 = b1._to_signature_bytes(include_name=False)
        self.assertEqual(
            sha256(bytes1).hexdigest(),
            '29a271e0d800ecaa673c7deded9dd7e8166cc746963c1717298e6af9e4189f23',
        )

        b2 = Bus.from_frames((f1, f2))
        bytes2 = b2._to_signature_bytes(include_name=False)
        self.assertNotEqual(sha256(bytes1).hexdigest(), sha256(bytes2).hexdigest())

    def test_bus_to_signature_bytes_b(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,2)').rename('f4')

        b1 = Bus.from_frames((f1, f2, f3))
        bytes1 = b1._to_signature_bytes(include_name=False)

        b2 = Bus.from_frames((f1, f2, f4))
        bytes2 = b2._to_signature_bytes(include_name=False)
        self.assertNotEqual(sha256(bytes1).hexdigest(), sha256(bytes2).hexdigest())

    def test_bus_via_hashlib_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')

        b1 = Bus.from_frames((f1, f2, f3))
        d = b1.via_hashlib(include_name=False).sha256().hexdigest()
        self.assertEqual(
            d, '29a271e0d800ecaa673c7deded9dd7e8166cc746963c1717298e6af9e4189f23'
        )

    # ---------------------------------------------------------------------------

    def test_bus_store_pickle_roundtrip(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(2,2)').rename('f2')

        b1 = Bus.from_frames((f1, f2))

        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)
            b2 = Bus.from_zip_npz(fp, max_persist=1)

            assert not b2._store._weak_cache  # type: ignore

            f1_r = b2.iloc[0]
            f2_r = b2.iloc[1]

            assert b2.iloc[1] is f2_r

            assert f1_r in b2._store._weak_cache.values()  # type: ignore
            assert b2.iloc[0] is f1_r

            b3 = pickle.loads(pickle.dumps(b2))

            assert not b3._store._weak_cache
            assert b3.iloc[0].equals(f1_r)

    # ---------------------------------------------------------------------------
    def test_bus_immutable_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(2,2)').rename('f2')
        b1 = Bus.from_frames((f1, f2))
        with self.assertRaises(ImmutableTypeError):
            b1['f1'] = f2

    def test_bus_immutable_b(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(2,2)').rename('f2')
        b1 = Bus.from_frames((f1, f2))
        with self.assertRaises(ImmutableTypeError):
            b1.loc['f1'] = f2

    def test_bus_immutable_c(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(2,2)').rename('f2')
        b1 = Bus.from_frames((f1, f2))
        with self.assertRaises(ImmutableTypeError):
            b1.iloc['f1'] = f2

    # ---------------------------------------------------------------------------

    def test_bus_persist_a1(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))
        config = StoreConfig()

        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)
            # set max_persist to size to test when fully loaded with max_persist
            b2 = Bus.from_zip_npz(fp, config=config)
            self.assertEqual(b2.status['loaded'].sum(), 0)
            b2._update_mutable_persistant_many(slice(None, None))
            self.assertEqual(b2.status['loaded'].sum(), 6)

    def test_bus_persist_a2(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))
        config = StoreConfig()

        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)
            # set max_persist to size to test when fully loaded with max_persist
            b2 = Bus.from_zip_npz(fp, config=config)
            self.assertEqual(b2.status['loaded'].sum(), 0)
            _ = b2['f2']
            _ = b2['f6']
            self.assertEqual(b2.status['loaded'].sum(), 2)

            b2._update_mutable_persistant_many(slice(None, None))
            self.assertEqual(b2.status['loaded'].sum(), 6)

    def test_bus_persist_a3(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))
        config = StoreConfig()

        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)
            # set max_persist to size to test when fully loaded with max_persist
            b2 = Bus.from_zip_npz(fp, config=config)

            b2._update_mutable_persistant_many(slice(0, 4))
            self.assertFalse(b2._loaded_all)
            b2._update_mutable_persistant_many([5, 4, 3, 2])
            self.assertTrue(b2._loaded_all)

    def test_bus_persist_a4(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))
        config = StoreConfig()

        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)
            # set max_persist to size to test when fully loaded with max_persist
            b2 = Bus.from_zip_npz(fp, config=config)

            b2._update_mutable_persistant_many([0, 2, 3, 5])
            self.assertFalse(b2._loaded_all)
            b2._update_mutable_persistant_many(slice(0, 3))
            self.assertFalse(b2._loaded_all)
            b2._update_mutable_persistant_many(slice(3, 6))
            self.assertTrue(b2._loaded_all)

    def test_bus_persist_b1(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))
        config = StoreConfig()

        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)
            # set max_persist to size to test when fully loaded with max_persist
            b2 = Bus.from_zip_npz(fp, config=config, max_persist=2)
            self.assertEqual(b2.status['loaded'].sum(), 0)

            b2._update_mutable_max_persist_many(slice(None, None))
            self.assertEqual(
                dict(b2.status['loaded']),
                {
                    'f1': False,
                    'f2': False,
                    'f3': False,
                    'f4': False,
                    'f5': True,
                    'f6': True,
                },
            )

            b2._update_mutable_max_persist_many(slice(None, None))
            self.assertEqual(
                dict(b2.status['loaded']),
                {
                    'f1': False,
                    'f2': False,
                    'f3': False,
                    'f4': False,
                    'f5': True,
                    'f6': True,
                },
            )

    def test_bus_persist_b2(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))
        config = StoreConfig()

        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)
            # set max_persist to size to test when fully loaded with max_persist
            b2 = Bus.from_zip_npz(fp, config=config, max_persist=2)
            self.assertEqual(b2.status['loaded'].sum(), 0)

            b2._update_mutable_max_persist_many([1, 2, 3])
            self.assertEqual(
                dict(b2.status['loaded']),
                {
                    'f1': False,
                    'f2': False,
                    'f3': True,
                    'f4': True,
                    'f5': False,
                    'f6': False,
                },
            )

            b2._update_mutable_max_persist_many([1, 2, 3, 5])
            self.assertEqual(
                dict(b2.status['loaded']),
                {
                    'f1': False,
                    'f2': False,
                    'f3': False,
                    'f4': True,
                    'f5': False,
                    'f6': True,
                },
            )

            b2._update_mutable_max_persist_many([1, 2, 3, 5])
            self.assertEqual(
                dict(b2.status['loaded']),
                {
                    'f1': False,
                    'f2': False,
                    'f3': False,
                    'f4': True,
                    'f5': False,
                    'f6': True,
                },
            )

    def test_bus_persist_b3(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))
        config = StoreConfig()

        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)
            # set max_persist to size to test when fully loaded with max_persist
            b2 = Bus.from_zip_npz(fp, config=config, max_persist=6)

            b2._update_mutable_max_persist_many([0, 2, 3, 5])
            self.assertFalse(b2._loaded_all)
            b2._update_mutable_max_persist_many(slice(0, 3))
            self.assertFalse(b2._loaded_all)
            b2._update_mutable_max_persist_many(slice(3, 6))
            self.assertTrue(b2._loaded_all)

    def test_bus_persist_b4(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))
        config = StoreConfig()

        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)
            # set max_persist to size to test when fully loaded with max_persist
            for max_persist in range(1, 6):  # keep less than all
                b2 = Bus.from_zip_npz(fp, config=config, max_persist=max_persist)

                b2._update_mutable_max_persist_many([0, 2, 3, 5])
                self.assertFalse(b2._loaded_all)
                self.assertTrue(b2.status['loaded'].sum() <= max_persist)

                b2._update_mutable_max_persist_many(slice(0, 3))
                self.assertFalse(b2._loaded_all)
                self.assertTrue(b2.status['loaded'].sum() <= max_persist)

                b2._update_mutable_max_persist_many(slice(3, 6))
                self.assertFalse(b2._loaded_all)
                self.assertTrue(b2.status['loaded'].sum() <= max_persist)

    def test_bus_persist_c1(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))
        config = StoreConfig()

        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)
            b2 = Bus.from_zip_npz(fp, config=config)
            b2.persist()
            self.assertTrue(b2._loaded_all)
            self.assertEqual(b2.status['loaded'].sum(), len(b1))

    def test_bus_persist_c2(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))
        config = StoreConfig()

        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)
            b2 = Bus.from_zip_npz(fp, config=config)
            b2.persist[:]
            self.assertTrue(b2._loaded_all)
            self.assertEqual(b2.status['loaded'].sum(), len(b1))

    def test_bus_persist_c3(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))
        config = StoreConfig()

        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)
            b2 = Bus.from_zip_npz(fp, config=config)
            b2.persist.iloc[:]
            self.assertTrue(b2._loaded_all)
            self.assertEqual(b2.status['loaded'].sum(), len(b1))

    def test_bus_persist_c4(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))
        config = StoreConfig()

        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)
            b2 = Bus.from_zip_npz(fp, config=config)
            b2.persist.loc[['f1', 'f6']]
            self.assertFalse(b2._loaded_all)
            self.assertEqual(b2.status['loaded'].sum(), 2)

            b2.persist.loc['f3':]
            self.assertFalse(b2._loaded_all)
            self.assertEqual(b2.status['loaded'].sum(), 5)

    def test_bus_persist_d1(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))
        config = StoreConfig()

        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)
            b2 = Bus.from_zip_npz(fp, config=config, max_persist=2)
            b2.persist.loc[['f1', 'f6']]
            self.assertFalse(b2._loaded_all)
            self.assertEqual(
                dict(b2.status['loaded']),
                {
                    'f1': True,
                    'f2': False,
                    'f3': False,
                    'f4': False,
                    'f5': False,
                    'f6': True,
                },
            )
            b2.persist.loc['f3':]
            self.assertEqual(
                dict(b2.status['loaded']),
                {
                    'f1': False,
                    'f2': False,
                    'f3': False,
                    'f4': False,
                    'f5': True,
                    'f6': True,
                },
            )

    def test_bus_persist_d2(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))
        config = StoreConfig()

        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)
            b2 = Bus.from_zip_npz(fp, config=config, max_persist=1)
            b2.persist.loc[['f1', 'f6', 'f2']]
            self.assertFalse(b2._loaded_all)
            self.assertEqual(
                dict(b2.status['loaded']),
                {
                    'f1': False,
                    'f2': False,
                    'f3': False,
                    'f4': False,
                    'f5': False,
                    'f6': True,
                },
            )
            b2.persist.loc[:]
            self.assertEqual(
                dict(b2.status['loaded']),
                {
                    'f1': False,
                    'f2': False,
                    'f3': False,
                    'f4': False,
                    'f5': False,
                    'f6': True,
                },
            )
            b2.persist.loc['f2']
            self.assertEqual(
                dict(b2.status['loaded']),
                {
                    'f1': False,
                    'f2': True,
                    'f3': False,
                    'f4': False,
                    'f5': False,
                    'f6': False,
                },
            )

    def test_bus_persist_d3(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,8)').rename('f4')
        f5 = ff.parse('s(4,4)').rename('f5')
        f6 = ff.parse('s(6,4)').rename('f6')

        b1 = Bus.from_frames((f1, f2, f3, f4, f5, f6))
        config = StoreConfig()

        with temp_file('.zip') as fp:
            b1.to_zip_npz(fp)
            b2 = Bus.from_zip_npz(fp, config=config, max_persist=2)
            b2.persist.loc[['f2', 'f6']]
            self.assertEqual(
                dict(b2.status['loaded']),
                {
                    'f1': False,
                    'f2': True,
                    'f3': False,
                    'f4': False,
                    'f5': False,
                    'f6': True,
                },
            )
            b2.persist.loc[['f1', 'f3']]
            self.assertEqual(
                dict(b2.status['loaded']),
                {
                    'f1': True,
                    'f2': False,
                    'f3': True,
                    'f4': False,
                    'f5': False,
                    'f6': False,
                },
            )
            b2.persist.loc[['f2', 'f1']]
            self.assertEqual(
                dict(b2.status['loaded']),
                {
                    'f1': True,
                    'f2': True,
                    'f3': False,
                    'f4': False,
                    'f5': False,
                    'f6': False,
                },
            )
            b2.persist()
            self.assertEqual(
                dict(b2.status['loaded']),
                {
                    'f1': False,
                    'f2': False,
                    'f3': False,
                    'f4': False,
                    'f5': True,
                    'f6': True,
                },
            )

    def test_bus_max_persist_e1(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        ih = IndexHierarchy.from_labels((('a', 1), ('b', 2), ('b', 1)))
        s1 = Series((f1, f2, f3), index=ih, dtype=object)
        b1 = Bus.from_series(s1)
        config = StoreConfig(label_encoder=str, label_decoder=literal_eval)

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp, config=config)
            b2 = Bus.from_zip_pickle(
                fp,
                config=config,
                max_persist=2,
                index_constructor=IndexHierarchy.from_labels,
            )

            b2.persist.loc[[('a', 1), ('b', 1)]]

            self.assertEqual(
                dict(b2.status['loaded']),
                {('a', 1): True, ('b', 1): True, ('b', 2): False},
            )
            b2.persist()
            self.assertEqual(
                dict(b2.status['loaded']),
                {('a', 1): False, ('b', 1): True, ('b', 2): True},
            )

    def test_bus_max_persist_e2(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        f4 = ff.parse('s(2,5)').rename('f4')
        ih = IndexHierarchy.from_labels((('a', 1), ('b', 1), ('b', 2), ('c', 0)))
        s1 = Series((f1, f2, f3, f4), index=ih, dtype=object)
        b1 = Bus.from_series(s1)
        config = StoreConfig(label_encoder=str, label_decoder=literal_eval)

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp, config=config)
            b2 = Bus.from_zip_pickle(
                fp, config=config, index_constructor=IndexHierarchy.from_labels
            )

            b2.persist.loc[('b', 2) :]

            self.assertEqual(
                dict(b2.status['loaded']),
                {('a', 1): False, ('b', 1): False, ('b', 2): True, ('c', 0): True},
            )
            b2.persist[:]
            self.assertEqual(
                dict(b2.status['loaded']),
                {('a', 1): True, ('b', 1): True, ('b', 2): True, ('c', 0): True},
            )

    # ---------------------------------------------------------------------------
    def test_bus_inventory_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3))

        config = StoreConfig(
            index_depth=1, columns_depth=1, include_columns=True, include_index=True
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            self.assertEqual(b1.inventory.shape, (1, 3))
            b2 = Bus.from_zip_pickle(fp, config=config)
            self.assertEqual(b2.inventory.shape, (1, 3))
            self.assertEqual(b2.inventory.index.values.tolist(), [None])

            b3 = b2.rename('foo')
            self.assertEqual(b3.inventory.index.values.tolist(), ['foo'])

    # ---------------------------------------------------------------------------
    def test_bus_read_frame_filter_a(self) -> None:
        f1 = ff.parse('s(4,8)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(6,8)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3))

        config = StoreConfig(
            index_depth=1,
            columns_depth=1,
            include_columns=True,
            include_index=True,
            read_frame_filter=lambda l, f: f.iloc[:2, :1],
        )

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)

            b2 = Bus.from_zip_pickle(fp, config=config)
            self.assertEqual(
                [f.shape for f in b2.iter_element()], [(2, 1), (2, 1), (2, 1)]
            )

            b3 = Bus.from_zip_pickle(fp)
            self.assertEqual(
                [f.shape for f in b3.iter_element()], [(4, 8), (4, 5), (6, 8)]
            )

    # ---------------------------------------------------------------------------
    def test_bus_copy_a(self) -> None:
        f1 = ff.parse('s(4,2)').rename('f1')
        f2 = ff.parse('s(4,5)').rename('f2')
        f3 = ff.parse('s(2,2)').rename('f3')
        b1 = Bus.from_frames((f1, f2, f3))

        with temp_file('.zip') as fp:
            b1.to_zip_pickle(fp)
            b2 = Bus.from_zip_pickle(fp)
            b2.persist()
            self.assertTrue(b2._loaded_all)

            b3 = b2.copy()
            self.assertEqual(b3._loaded.sum(), 3)
            b3.unpersist()
            self.assertEqual(b3._loaded.sum(), 0)

            self.assertTrue(b2._loaded_all)
