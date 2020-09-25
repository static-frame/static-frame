import unittest
# from io import StringIO

from static_frame.core.frame import Frame
from static_frame.core.frame import FrameGO
# from static_frame.core.bus import Bus
# from static_frame.core.series import Series

from static_frame.core.store import StoreConfig
from static_frame.core.store import StoreConfigMap

from static_frame.core.store_zip import StoreZipTSV
from static_frame.core.store_zip import StoreZipCSV
from static_frame.core.store_zip import StoreZipPickle
from static_frame.core.store_zip import StoreZipParquet

from static_frame.test.test_case import TestCase
from static_frame.test.test_case import temp_file

# from static_frame.test.test_case import skip_win
from static_frame.core.exception import ErrorInitStore
# from static_frame.core.exception import ErrorInitStoreConfig


class TestUnit(TestCase):

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

            config = StoreConfig(index_depth=1)

            for label, frame in ((f.name, f) for f in (f1, f2, f3)):
                frame_stored = st.read(label, config=config)
                self.assertEqual(frame_stored.shape, frame.shape)
                self.assertTrue((frame_stored == frame).all().all())
                self.assertEqual(frame.to_pairs(0), frame_stored.to_pairs(0))

                frame_stored_2 = st.read(label, config=config, container_type=FrameGO)
                self.assertEqual(frame_stored_2.__class__, FrameGO)
                self.assertEqual(frame_stored_2.shape, frame.shape)

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

            config = StoreConfig(index_depth=1)

            for label, frame in ((f.name, f) for f in (f1, f2, f3)):
                frame_stored = st.read(label, config=config)
                self.assertEqual(frame_stored.shape, frame.shape)
                self.assertTrue((frame_stored == frame).all().all())
                self.assertEqual(frame.to_pairs(0), frame_stored.to_pairs(0))


    def test_store_zip_csv_b(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='foo')

        with temp_file('.zip') as fp:

            st = StoreZipCSV(fp)
            st.write((f.name, f) for f in (f1,))

            with self.assertRaises(ErrorInitStore):
                # config is required
                _ = st.read(f1.name)



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

                frame_stored_2 = st.read(label, container_type=FrameGO)
                self.assertEqual(frame_stored_2.__class__, FrameGO)
                self.assertEqual(frame_stored_2.shape, frame.shape)


    def test_store_zip_pickle_b(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='foo')

        config = StoreConfig(index_depth=1, include_index=True)
        config_map = StoreConfigMap.from_config(config)

        with temp_file('.zip') as fp:

            st = StoreZipPickle(fp)
            st.write(((f1.name, f1),))

            frame_stored = st.read(f1.name)
            self.assertEqual(frame_stored.shape, f1.shape)

    def test_store_zip_pickle_c(self) -> None:

        f1 = FrameGO.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='foo')

        config = StoreConfig(index_depth=1, include_index=True)
        config_map = StoreConfigMap.from_config(config)

        with temp_file('.zip') as fp:

            st = StoreZipPickle(fp)

            # raise if trying to store a FrameGO
            with self.assertRaises(NotImplementedError):
                st.write(((f1.name, f1),))


    def test_store_zip_parquet_a(self) -> None:

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

        config = StoreConfig(index_depth=1, include_index=True, columns_depth=1)

        with temp_file('.zip') as fp:

            st = StoreZipParquet(fp)
            st.write((f.name, f) for f in (f1, f2, f3))

            f1_post = st.read('foo', config=config)
            self.assertTrue(f1.equals(f1_post, compare_name=True, compare_class=True))

            f2_post = st.read('bar', config=config)
            self.assertTrue(f2.equals(f2_post, compare_name=True, compare_class=True))

            f3_post = st.read('baz', config=config)
            self.assertTrue(f3.equals(f3_post, compare_name=True, compare_class=True))



if __name__ == '__main__':
    unittest.main()


