import unittest
# from io import StringIO

from static_frame.core.frame import Frame
# from static_frame.core.bus import Bus
# from static_frame.core.series import Series

from static_frame.core.store import StoreConfigConstructor
from static_frame.core.store import StoreConfigExporter
from static_frame.core.store import StoreConfigConstructorMap
from static_frame.core.store import StoreConfigExporterMap

from static_frame.core.store_zip import StoreZipTSV
from static_frame.core.store_zip import StoreZipCSV
from static_frame.core.store_zip import StoreZipPickle

from static_frame.test.test_case import TestCase
from static_frame.test.test_case import temp_file

# from static_frame.test.test_case import skip_win
from static_frame.core.exception import ErrorInitStore
from static_frame.core.exception import ErrorInitStoreConfig


class TestUnit(TestCase):


    #---------------------------------------------------------------------------

    def test_store_config_map_a(self) -> None:

        sc1 = StoreConfigConstructor(index_depth=3, columns_depth=3)
        sc1m = StoreConfigConstructorMap.from_config(sc1)
        self.assertEqual(sc1m['a'].index_depth, 3)
        self.assertEqual(sc1m['b'].index_depth, 3)

        sc2 = StoreConfigExporter(include_index=False)
        sc2m = StoreConfigExporterMap.from_config(sc2)
        self.assertEqual(sc2m['a'].include_index, False)
        self.assertEqual(sc2m['b'].include_index, False)

        with self.assertRaises(ErrorInitStoreConfig):
            sc3m = StoreConfigExporterMap.from_config(sc1)


    def test_store_config_map_b(self) -> None:

        maps = {'a': StoreConfigConstructor(index_depth=2),
                'b': StoreConfigConstructor(index_depth=3)}
        sc1m = StoreConfigConstructorMap(maps)
        self.assertEqual(sc1m['a'].index_depth, 2)
        self.assertEqual(sc1m['b'].index_depth, 3)
        self.assertEqual(sc1m['c'].index_depth, 0)

    def test_store_config_map_c(self) -> None:
        sc1 = StoreConfigConstructor(index_depth=3, columns_depth=3)
        maps = {'a': StoreConfigConstructor(index_depth=2),
                'b': StoreConfigConstructor(index_depth=3)}
        sc1m = StoreConfigConstructorMap(maps)

        sc2m = StoreConfigConstructorMap.from_initializer(sc1)
        self.assertEqual(sc2m['a'].index_depth, 3)

        sc3m = StoreConfigConstructorMap.from_initializer(sc1m)
        self.assertEqual(sc3m['a'].index_depth, 2)
        self.assertEqual(sc3m['b'].index_depth, 3)

        sc4m = StoreConfigConstructorMap.from_initializer(maps)
        self.assertEqual(sc4m['a'].index_depth, 2)
        self.assertEqual(sc4m['b'].index_depth, 3)


    #---------------------------------------------------------------------------

    def test_store_init_a(self) -> None:
        with self.assertRaises(ErrorInitStore):
            StoreZipTSV('test.txt') # must be a zip


    def test_store_zip_tsv_a(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='foo')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='bar')
        f3 = Frame.from_dict(
                dict(a=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='baz')

        with temp_file('.zip') as fp:

            st = StoreZipTSV(fp)
            st.write((f.name, f) for f in (f1, f2, f3))

            labels = tuple(st.labels(strip_ext=False))
            self.assertEqual(labels, ('foo.txt', 'bar.txt', 'baz.txt'))

            for label, frame in ((f.name, f) for f in (f1, f2, f3)):
                frame_stored = st.read(label)
                self.assertEqual(frame_stored.shape, frame.shape)
                self.assertTrue((frame_stored == frame).all().all())
                self.assertEqual(frame.to_pairs(0), frame_stored.to_pairs(0))


    def test_store_zip_csv_a(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='foo')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='bar')
        f3 = Frame.from_dict(
                dict(a=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='baz')

        with temp_file('.zip') as fp:

            st = StoreZipCSV(fp)
            st.write((f.name, f) for f in (f1, f2, f3))

            labels = tuple(st.labels(strip_ext=False))
            self.assertEqual(labels, ('foo.csv', 'bar.csv', 'baz.csv'))

            for label, frame in ((f.name, f) for f in (f1, f2, f3)):
                frame_stored = st.read(label)
                self.assertEqual(frame_stored.shape, frame.shape)
                self.assertTrue((frame_stored == frame).all().all())
                self.assertEqual(frame.to_pairs(0), frame_stored.to_pairs(0))



    def test_store_zip_pickle_a(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='foo')
        f2 = Frame.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='bar')
        f3 = Frame.from_dict(
                dict(a=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='baz')

        with temp_file('.zip') as fp:

            st = StoreZipPickle(fp)
            st.write((f.name, f) for f in (f1, f2, f3))

            labels = tuple(st.labels(strip_ext=False))
            self.assertEqual(labels, ('foo.pickle', 'bar.pickle', 'baz.pickle'))

            for label, frame in ((f.name, f) for f in (f1, f2, f3)):
                frame_stored = st.read(label)
                self.assertEqual(frame_stored.shape, frame.shape)
                self.assertTrue((frame_stored == frame).all().all())
                self.assertEqual(frame.to_pairs(0), frame_stored.to_pairs(0))


if __name__ == '__main__':
    unittest.main()
