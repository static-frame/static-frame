import unittest
import typing as tp

import frame_fixtures as ff

from static_frame.core.frame import Frame
from static_frame.core.frame import FrameGO
from static_frame.core.frame import FrameHE
# from static_frame.core.bus import Bus
# from static_frame.core.series import Series
from static_frame.core.index_datetime import IndexDate

from static_frame.core.store import StoreConfig
from static_frame.core.store import StoreConfigMap

from static_frame.core.store_zip import _StoreZip
from static_frame.core.store_zip import StoreZipTSV
from static_frame.core.store_zip import StoreZipCSV
from static_frame.core.store_zip import StoreZipPickle
from static_frame.core.store_zip import StoreZipParquet
from static_frame.core.store_zip import StoreZipNPZ

from static_frame.test.test_case import TestCase
from static_frame.test.test_case import temp_file

# from static_frame.test.test_case import skip_win
from static_frame.core.exception import ErrorInitStore
# from static_frame.core.exception import ErrorInitStoreConfig

def get_test_framesA(container_type: tp.Type[Frame] = Frame) -> tp.Tuple[Frame, Frame, Frame]:
    return (
            container_type.from_dict(
                dict(a=(1,2), b=(3,4)),
                index=('x', 'y'),
                name='foo'),
            container_type.from_dict(
                dict(a=(1,2,3), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='bar'),
            container_type.from_dict(
                dict(a=(10,20), b=(50,60)),
                index=('p', 'q'),
                name='baz')
            )

def get_test_framesB() -> tp.Tuple[Frame, Frame]:
    return (
            ff.parse('s(4,4)|i(ID,dtD)|v(int)').rename('a'),
            ff.parse('s(4,4)|i(ID,dtD)|v(int)').rename('b'),
            )


class TestUnit(TestCase):
    #---------------------------------------------------------------------------

    def test_store_init_a(self) -> None:
        with self.assertRaises(ErrorInitStore):
            StoreZipTSV('test.txt') # must be a zip

    def test_store_base_class_init(self) -> None:
        with self.assertRaises(NotImplementedError):
            _StoreZip._container_type_to_constructor(None) # type: ignore

        with self.assertRaises(NotImplementedError):
            _StoreZip._build_frame(
                    src=bytes(),
                    name=None,
                    config=StoreConfig(),
                    constructor=lambda x: Frame(),
            )

    def test_store_zip_tsv_a(self) -> None:

        f1, f2, f3 = get_test_framesA()

        with temp_file('.zip') as fp:

            st = StoreZipTSV(fp)
            st.write((f.name, f) for f in (f1, f2, f3))

            labels = tuple(st.labels(strip_ext=False))
            self.assertEqual(labels, ('foo.txt', 'bar.txt', 'baz.txt'))

            for label, frame in ((f.name, f) for f in (f1, f2, f3)):
                for read_max_workers in (None, 1, 2):
                    config = StoreConfig(index_depth=1, read_max_workers=read_max_workers)
                    frame_stored = st.read(label, config=config)
                    self.assertEqual(frame_stored.shape, frame.shape)
                    self.assertTrue((frame_stored == frame).all().all())
                    self.assertEqual(frame.to_pairs(0), frame_stored.to_pairs(0))

                    frame_stored_2 = st.read(label, config=config, container_type=FrameGO)
                    self.assertEqual(frame_stored_2.__class__, FrameGO)
                    self.assertEqual(frame_stored_2.shape, frame.shape)

    def test_store_zip_csv_a(self) -> None:

        f1, f2, f3 = get_test_framesA()

        with temp_file('.zip') as fp:

            st = StoreZipCSV(fp)
            st.write((f.name, f) for f in (f1, f2, f3))

            labels = tuple(st.labels(strip_ext=False))
            self.assertEqual(labels, ('foo.csv', 'bar.csv', 'baz.csv'))

            for label, frame in ((f.name, f) for f in (f1, f2, f3)):
                for read_max_workers in (1, 2):
                    config = StoreConfig(index_depth=1, read_max_workers=1)
                    frame_stored = st.read(label, config=config)
                    self.assertEqual(frame_stored.shape, frame.shape)
                    self.assertTrue((frame_stored == frame).all().all())
                    self.assertEqual(frame.to_pairs(0), frame_stored.to_pairs(0))

    def test_store_zip_csv_b(self) -> None:

        f1, *_ = get_test_framesA()

        with temp_file('.zip') as fp:

            st = StoreZipCSV(fp)
            st.write((f.name, f) for f in (f1,))

            # this now uses a default config
            f = st.read(f1.name)
            self.assertEqual(f.to_pairs(), (('__index0__', ((0, 'x'), (1, 'y'))), ('a', ((0, 1), (1, 2))), ('b', ((0, 3), (1, 4)))))

    #---------------------------------------------------------------------------
    def test_store_zip_pickle_a(self) -> None:

        f1, f2, f3 = get_test_framesA()

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

                frame_stored_3 = st.read(label, container_type=FrameHE)
                self.assertEqual(frame_stored_3.__class__, FrameHE)
                self.assertEqual(frame_stored_3.shape, frame.shape)

    def test_store_zip_pickle_b(self) -> None:

        f1, *_ = get_test_framesA()

        with temp_file('.zip') as fp:

            st = StoreZipPickle(fp)
            st.write(((f1.name, f1),))

            frame_stored = st.read(f1.name)
            self.assertEqual(frame_stored.shape, f1.shape)

    def test_store_zip_pickle_c(self) -> None:

        f1, *_ = get_test_framesA(FrameGO)

        with temp_file('.zip') as fp:
            st = StoreZipPickle(fp)
            st.write(((f1.name, f1),))

            frame_stored = st.read(f1.name)
            self.assertEqual(frame_stored.shape, f1.shape)
            self.assertTrue(frame_stored.__class__ is Frame)

    def test_store_zip_pickle_d(self) -> None:

        f1, *_ = get_test_framesA()

        with temp_file('.zip') as fp:
            for read_max_workers in (1, 2):
                config = StoreConfig(
                        index_depth=1,
                        include_index=True,
                        label_encoder=lambda x: x.upper(), #type: ignore
                        label_decoder=lambda x: x.lower(),
                        read_max_workers=read_max_workers,
                )

                st = StoreZipPickle(fp)
                st.write(((f1.name, f1),), config=config)

                frame_stored = st.read(f1.name, config=config)

                self.assertEqual(tuple(st.labels()), ('FOO',))
                self.assertEqual(tuple(st.labels(config=config)), ('foo',))

    def test_store_zip_pickle_e(self) -> None:

        f1, f2, f3 = get_test_framesA(FrameGO)

        with temp_file('.zip') as fp:
            st = StoreZipPickle(fp)
            st.write((f.name, f) for f in (f1, f2, f3))

            post = tuple(st.read_many(('baz', 'bar', 'foo'), container_type=Frame))
            self.assertEqual(len(post), 3)
            self.assertEqual(post[0].name, 'baz')
            self.assertEqual(post[1].name, 'bar')
            self.assertEqual(post[2].name, 'foo')

            self.assertTrue(post[0].__class__ is Frame)
            self.assertTrue(post[1].__class__ is Frame)
            self.assertTrue(post[2].__class__ is Frame)

    #---------------------------------------------------------------------------

    def test_store_zip_parquet_a(self) -> None:

        f1, f2, f3 = get_test_framesA()

        with temp_file('.zip') as fp:
            for read_max_workers in (1, 2):
                config = StoreConfig(index_depth=1, include_index=True, columns_depth=1, read_max_workers=read_max_workers)

                st = StoreZipParquet(fp)
                st.write((f.name, f) for f in (f1, f2, f3))

                f1_post = st.read('foo', config=config)
                self.assertTrue(f1.equals(f1_post, compare_name=True, compare_class=True))

                f2_post = st.read('bar', config=config)
                self.assertTrue(f2.equals(f2_post, compare_name=True, compare_class=True))

                f3_post = st.read('baz', config=config)
                self.assertTrue(f3.equals(f3_post, compare_name=True, compare_class=True))

    def test_store_zip_parquet_b(self) -> None:

        f1, f2, f3 = get_test_framesA()

        with temp_file('.zip') as fp:
            for read_max_workers in (1, 2):
                config = StoreConfig(index_depth=1, include_index=True, columns_depth=1, read_max_workers=read_max_workers)
                st = StoreZipParquet(fp)
                st.write((f.name, f) for f in (f1, f2, f3))

                post = tuple(st.read_many(('baz', 'bar', 'foo'), config=config))
                self.assertEqual(len(post), 3)
                self.assertEqual(post[0].name, 'baz')
                self.assertEqual(post[1].name, 'bar')
                self.assertEqual(post[2].name, 'foo')

    def test_store_zip_parquet_c(self) -> None:

        f1, f2 = get_test_framesB()

        config = StoreConfig(
                index_depth=1,
                include_index=True,
                index_constructors=IndexDate,
                columns_depth=1,
                include_columns=True,
        )

        with temp_file('.zip') as fp:
            st = StoreZipParquet(fp)
            st.write(((f.name, f) for f in (f1, f2)), config=config)

            post = tuple(st.read_many(('a', 'b'),
                    container_type=Frame,
                    config=config,
                    ))

            self.assertIs(post[0].index.__class__, IndexDate)
            self.assertIs(post[1].index.__class__, IndexDate)

    def test_store_read_many_single_thread_weak_cache(self) -> None:

        f1, f2, f3 = get_test_framesA()

        with temp_file('.zip') as fp:

            st = StoreZipTSV(fp)
            st.write((f.name, f) for f in (f1, f2, f3))

            kwargs = dict(
                    config_map=StoreConfigMap.from_initializer(StoreConfig(index_depth=1)),
                    constructor=st._container_type_to_constructor(Frame),
                    container_type=Frame
                    )

            labels = tuple(st.labels(strip_ext=False))
            self.assertEqual(labels, ('foo.txt', 'bar.txt', 'baz.txt'))

            self.assertEqual(0, len(list(st._weak_cache)))

            # Result is not held onto!
            next(st._read_many_single_thread(('foo',), **kwargs))

            self.assertEqual(0, len(list(st._weak_cache)))

            # Result IS held onto!
            frame = next(st._read_many_single_thread(('foo',), **kwargs))

            self.assertEqual(1, len(list(st._weak_cache)))

            # Reference in our weak_cache _is_ `frame`
            self.assertIs(frame, st._weak_cache['foo'])
            del frame

            # Reference is gone now!
            self.assertEqual(0, len(list(st._weak_cache)))

    def test_store_read_many_weak_cache_a(self) -> None:

        def gen_test_frames() -> tp.Iterator[tp.Tuple[str, Frame]]:
            f1, f2, f3 = get_test_framesA()
            yield from ((f.name, f) for f in (f1, f2, f3))

        for read_max_workers in (None, 1):
            with temp_file('.zip') as fp:

                st = StoreZipTSV(fp)
                st.write(gen_test_frames())

                kwargs = dict(
                        config=StoreConfig(index_depth=1, read_max_workers=read_max_workers),
                        container_type=Frame,
                        )

                labels = tuple(st.labels(strip_ext=True))
                self.assertEqual(labels, ('foo', 'bar', 'baz'))

                self.assertEqual(0, len(list(st._weak_cache)))

                # Go through the pass where there are no cache hits!
                # Don't hold onto the result!
                list(st.read_many(labels, **kwargs))
                self.assertEqual(0, len(list(st._weak_cache)))

                # Hold onto all results
                result = list(st.read_many(labels, **kwargs))
                self.assertEqual(3, len(result))
                self.assertEqual(3, len(list(st._weak_cache)))

                del result
                self.assertEqual(0, len(list(st._weak_cache)))

                [frame] = list(st.read_many(("foo",), **kwargs))
                self.assertIs(frame, st._weak_cache['foo'])

                # Go through pass where there are some cache hits!
                # Don't hold onto the result!
                list(st.read_many(labels, **kwargs))
                self.assertEqual(1, len(list(st._weak_cache)))

                # Hold onto all results
                result = list(st.read_many(labels, **kwargs))
                self.assertEqual(3, len(result))
                self.assertEqual(3, len(list(st._weak_cache)))

                # Go through pass where all labels are in the cache
                result2 = list(st.read_many(labels, **kwargs))
                self.assertEqual(len(result), len(result2))
                for f1, f2 in zip(result, result2):
                    self.assertIs(f1, f2)


class TestUnitMultiProcess(TestCase):

    def run_assertions(self, klass: tp.Type[_StoreZip]) -> None:
        f1, f2, f3 = get_test_framesA()
        with temp_file('.zip') as fp:
            for max_workers in range(1, 6):
                for chunksize in (1, 2, 3):
                    config = StoreConfig(
                            index_depth=1,
                            include_index=True,
                            columns_depth=1,
                            write_max_workers=max_workers,
                            write_chunksize=chunksize,
                    )
                    st = klass(fp)
                    st.write(((f.name, f) for f in (f1, f2, f3)), config=config)

                    post = tuple(st.read_many(('baz', 'bar', 'foo'), config=config))
                    self.assertEqual(len(post), 3)
                    self.assertEqual(post[0].name, 'baz')
                    self.assertEqual(post[1].name, 'bar')
                    self.assertEqual(post[2].name, 'foo')

    def test_store_zip_tsv_mp(self) -> None:
        self.run_assertions(StoreZipTSV)

    def test_store_zip_csv_mp(self) -> None:
        self.run_assertions(StoreZipCSV)

    def test_store_zip_pickle_mp(self) -> None:
        self.run_assertions(StoreZipPickle)

    def test_store_zip_parquet_mp(self) -> None:
        self.run_assertions(StoreZipParquet)

    def test_store_zip_npz_mp(self) -> None:
        self.run_assertions(StoreZipNPZ)

    #---------------------------------------------------------------------------

    def test_store_zip_npz_a(self) -> None:

        f1, f2 = get_test_framesB()

        config = StoreConfig()

        with temp_file('.zip') as fp:
            st = StoreZipNPZ(fp)
            st.write(((f.name, f) for f in (f1, f2)), config=config)

            post = tuple(st.read_many(('a', 'b'),
                    container_type=Frame,
                    config=config,
                    ))

            self.assertIs(post[0].index.__class__, IndexDate)
            self.assertIs(post[1].index.__class__, IndexDate)


if __name__ == '__main__':
    unittest.main()
