from __future__ import annotations

import io

import frame_fixtures as ff
import typing_extensions as tp

from static_frame.core.exception import ErrorInitStore
from static_frame.core.frame import Frame, FrameGO, FrameHE
from static_frame.core.index_datetime import IndexDate
from static_frame.core.store_config import (
    StoreConfig,
    StoreConfigCSV,
    StoreConfigMap,
    StoreConfigNPY,
    StoreConfigNPZ,
    StoreConfigParquet,
    StoreConfigPickle,
    StoreConfigTSV,
)
from static_frame.core.store_zip import (
    StoreZipCSV,
    StoreZipNPY,
    StoreZipNPZ,
    StoreZipParquet,
    StoreZipPickle,
    StoreZipTSV,
    _StoreZip,
    bytes_io_to_str_io,
)
from static_frame.test.test_case import TestCase, temp_file


def get_test_framesA(
    container_type: tp.Type[Frame] = Frame,
) -> tp.Tuple[Frame, Frame, Frame]:
    return (
        container_type.from_dict(dict(a=(1, 2), b=(3, 4)), index=('x', 'y'), name='foo'),
        container_type.from_dict(
            dict(a=(1, 2, 3), b=(4, 5, 6)), index=('x', 'y', 'z'), name='bar'
        ),
        container_type.from_dict(
            dict(a=(10, 20), b=(50, 60)), index=('p', 'q'), name='baz'
        ),
    )


def get_test_framesB() -> tp.Tuple[Frame, Frame]:
    return (
        ff.parse('s(4,4)|i(ID,dtD)|v(int)').rename('a'),
        ff.parse('s(4,4)|i(ID,dtD)|v(int)').rename('b'),
    )


class TestUnit(TestCase):
    # ---------------------------------------------------------------------------

    def test_store_init_a(self) -> None:
        with self.assertRaises(ErrorInitStore):
            StoreZipTSV('test.txt')  # must be a zip

    def test_store_base_class_init(self) -> None:
        with self.assertRaises(NotImplementedError):
            _StoreZip._build_frame(
                src=bytes(),
                label=None,
                config=StoreConfig(),
            )

    def test_store_zip_tsv_a(self) -> None:
        f1, f2, f3 = get_test_framesA()

        with temp_file('.zip') as fp:
            for read_max_workers in (None, 1, 2):
                st = StoreZipTSV(
                    fp,
                    config=StoreConfigTSV(
                        index_depth=1, read_max_workers=read_max_workers
                    ),
                )
                st.write((f.name, f) for f in (f1, f2, f3))

                labels = tuple(st.labels(strip_ext=False))
                self.assertEqual(labels, ('foo.txt', 'bar.txt', 'baz.txt'))

                for label, frame in ((f.name, f) for f in (f1, f2, f3)):
                    frame_stored = st.read(label)
                    self.assertEqual(frame_stored.shape, frame.shape)
                    self.assertTrue((frame_stored == frame).all().all())
                    self.assertEqual(frame.to_pairs(), frame_stored.to_pairs())

                    frame_stored_2 = st.read(label)
                    self.assertEqual(frame_stored_2.shape, frame.shape)

    def test_store_zip_csv_a(self) -> None:
        f1, f2, f3 = get_test_framesA()

        with temp_file('.zip') as fp:
            for read_max_workers in (1, 2):
                st = StoreZipCSV(
                    fp,
                    config=StoreConfigCSV(
                        index_depth=1, read_max_workers=read_max_workers
                    ),
                )
                st.write((f.name, f) for f in (f1, f2, f3))

                labels = tuple(st.labels(strip_ext=False))
                self.assertEqual(labels, ('foo.csv', 'bar.csv', 'baz.csv'))

                for label, frame in ((f.name, f) for f in (f1, f2, f3)):
                    frame_stored = st.read(label)
                    self.assertEqual(frame_stored.shape, frame.shape)
                    self.assertTrue((frame_stored == frame).all().all())
                    self.assertEqual(frame.to_pairs(), frame_stored.to_pairs())

    def test_store_zip_csv_b(self) -> None:
        f1, *_ = get_test_framesA()

        with temp_file('.zip') as fp:
            st = StoreZipCSV(fp)
            st.write((f.name, f) for f in (f1,))

            # this now uses a default config
            f = st.read(f1.name)
            self.assertEqual(
                f.to_pairs(),
                (
                    ('__index0__', ((0, 'x'), (1, 'y'))),
                    ('a', ((0, 1), (1, 2))),
                    ('b', ((0, 3), (1, 4))),
                ),
            )

    # ---------------------------------------------------------------------------
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
                self.assertEqual(frame.to_pairs(), frame_stored.to_pairs())

                frame_stored_2 = st.read(label)
                self.assertEqual(frame_stored_2.shape, frame.shape)

                frame_stored_3 = st.read(label)
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
                st1 = StoreZipPickle(
                    fp,
                    config=StoreConfigPickle(
                        label_encoder=lambda x: x.upper(),  # type: ignore
                        label_decoder=lambda x: x.lower(),
                        read_max_workers=read_max_workers,
                    ),
                )
                st1.write(((f1.name, f1),))

                st2 = StoreZipPickle(fp)

                self.assertEqual(tuple(st1.labels()), ('foo',))
                self.assertEqual(tuple(st2.labels()), ('FOO',))

    def test_store_zip_pickle_e(self) -> None:
        f1, f2, f3 = get_test_framesA(FrameGO)

        with temp_file('.zip') as fp:
            st = StoreZipPickle(fp)
            st.write((f.name, f) for f in (f1, f2, f3))

            post = tuple(st.read_many(('baz', 'bar', 'foo')))
            self.assertEqual(len(post), 3)
            self.assertEqual(post[0].name, 'baz')
            self.assertEqual(post[1].name, 'bar')
            self.assertEqual(post[2].name, 'foo')

            self.assertTrue(post[0].__class__ is Frame)
            self.assertTrue(post[1].__class__ is Frame)
            self.assertTrue(post[2].__class__ is Frame)

    # ---------------------------------------------------------------------------

    def test_store_zip_parquet_a(self) -> None:
        f1, f2, f3 = get_test_framesA()

        with temp_file('.zip') as fp:
            for read_max_workers in (1, 2):
                st = StoreZipParquet(
                    fp,
                    config=StoreConfigParquet(
                        index_depth=1,
                        include_index=True,
                        columns_depth=1,
                        read_max_workers=read_max_workers,
                    ),
                )

                st.write((f.name, f) for f in (f1, f2, f3))

                f1_post = st.read('foo')
                self.assertTrue(f1.equals(f1_post, compare_name=True, compare_class=True))

                f2_post = st.read('bar')
                self.assertTrue(f2.equals(f2_post, compare_name=True, compare_class=True))

                f3_post = st.read('baz')
                self.assertTrue(f3.equals(f3_post, compare_name=True, compare_class=True))

    def test_store_zip_parquet_b(self) -> None:
        f1, f2, f3 = get_test_framesA()

        with temp_file('.zip') as fp:
            for read_max_workers in (1, 2):
                st = StoreZipParquet(
                    fp,
                    config=StoreConfigParquet(
                        index_depth=1,
                        include_index=True,
                        columns_depth=1,
                        read_max_workers=read_max_workers,
                    ),
                )
                st.write((f.name, f) for f in (f1, f2, f3))

                post = tuple(st.read_many(('baz', 'bar', 'foo')))
                self.assertEqual(len(post), 3)
                self.assertEqual(post[0].name, 'baz')
                self.assertEqual(post[1].name, 'bar')
                self.assertEqual(post[2].name, 'foo')

    def test_store_zip_parquet_c(self) -> None:
        f1, f2 = get_test_framesB()

        config = StoreConfigParquet(
            index_depth=1,
            include_index=True,
            index_constructors=IndexDate,
            columns_depth=1,
            include_columns=True,
        )

        with temp_file('.zip') as fp:
            st = StoreZipParquet(fp, config=config)
            st.write(((f.name, f) for f in (f1, f2)))

            post = tuple(st.read_many(('a', 'b')))

            self.assertIs(post[0].index.__class__, IndexDate)
            self.assertIs(post[1].index.__class__, IndexDate)

    def test_store_read_many_single_thread_weak_cache(self) -> None:
        f1, f2, f3 = get_test_framesA()

        with temp_file('.zip') as fp:
            st = StoreZipTSV(
                fp,
                config=StoreConfigMap.from_initializer(StoreConfigTSV(index_depth=1)),
            )
            st.write((f.name, f) for f in (f1, f2, f3))

            labels = tuple(st.labels(strip_ext=False))
            self.assertEqual(labels, ('foo.txt', 'bar.txt', 'baz.txt'))

            self.assertEqual(0, len(list(st._weak_cache)))

            # Result is not held onto!
            next(st._read_many_single_thread(('foo',)))

            self.assertEqual(0, len(list(st._weak_cache)))

            # Result IS held onto!
            frame = next(st._read_many_single_thread(('foo',)))

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
                st = StoreZipTSV(
                    fp,
                    config=StoreConfigTSV(
                        index_depth=1, read_max_workers=read_max_workers
                    ),
                )
                st.write(gen_test_frames())

                labels = tuple(st.labels(strip_ext=True))
                self.assertEqual(labels, ('foo', 'bar', 'baz'))

                self.assertEqual(0, len(list(st._weak_cache)))

                # Go through the pass where there are no cache hits!
                # Don't hold onto the result!
                list(st.read_many(labels))
                self.assertEqual(0, len(list(st._weak_cache)))

                # Hold onto all results
                result = list(st.read_many(labels))
                self.assertEqual(3, len(result))
                self.assertEqual(3, len(list(st._weak_cache)))

                del result
                self.assertEqual(0, len(list(st._weak_cache)))

                [frame] = list(st.read_many(('foo',)))
                self.assertIs(frame, st._weak_cache['foo'])

                # Go through pass where there are some cache hits!
                # Don't hold onto the result!
                list(st.read_many(labels))
                self.assertEqual(1, len(list(st._weak_cache)))

                # Hold onto all results
                result = list(st.read_many(labels))
                self.assertEqual(3, len(result))
                self.assertEqual(3, len(list(st._weak_cache)))

                # Go through pass where all labels are in the cache
                result2 = list(st.read_many(labels))
                self.assertEqual(len(result), len(result2))
                for f1, f2 in zip(result, result2):
                    self.assertIs(f1, f2)

    def test_bytes_io_to_str_io_a(self) -> None:
        bio = io.BytesIO()
        with bytes_io_to_str_io(bio) as tw:
            tw.write('Hello')
            tw.write('World')

        self.assertEqual(bio.getvalue(), b'HelloWorld')

    def test_bytes_io_to_str_io_b(self) -> None:
        with temp_file() as fp:
            with open(fp, 'wb') as f:
                with bytes_io_to_str_io(f) as tw:
                    tw.write('Hello')
                    tw.write('World')

            with open(fp, 'rb') as f:
                self.assertEqual(f.read(), b'HelloWorld')


class TestUnitMultiProcess(TestCase):
    def run_assertions(self, klass: tp.Type[_StoreZip[StoreConfig]]) -> None:
        f1, f2, f3 = get_test_framesA()

        if klass in (StoreZipTSV, StoreZipCSV, StoreZipParquet):
            kwargs = dict(index_depth=1, include_index=True, columns_depth=1)
        else:
            kwargs = dict()

        with temp_file('.zip') as fp:
            for max_workers in range(1, 6):
                for chunksize in (1, 2, 3):
                    st = klass(
                        fp,
                        config=klass._STORE_CONFIG_CLASS(
                            write_max_workers=max_workers,
                            write_chunksize=chunksize,
                            **kwargs,
                        ),
                    )
                    st.write(((f.name, f) for f in (f1, f2, f3)))

                    post = tuple(st.read_many(('baz', 'bar', 'foo')))
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

    # ---------------------------------------------------------------------------

    def test_store_zip_npz_a(self) -> None:
        f1, f2 = get_test_framesB()

        to_write = [
            (f1.name, f1),
            (f2.name, f2),
            ('unnamed', f2.rename(None)),
        ]

        config = StoreConfigNPZ()

        with temp_file('.zip') as fp:
            st = StoreZipNPZ(fp, config=config)
            st.write(to_write)

            post = tuple(st.read_many(('a', 'b', 'unnamed')))

            self.assertIs(post[0].index.__class__, IndexDate)
            self.assertIs(post[1].index.__class__, IndexDate)
            self.assertIs(post[2].index.__class__, IndexDate)

            self.assertEqual(post[0].name, 'a')
            self.assertEqual(post[1].name, 'b')
            self.assertEqual(post[2].name, 'unnamed')

    # ---------------------------------------------------------------------------
    def test_store_zip_npy_a(self) -> None:
        f1 = ff.parse('s(4,6)|v(int,int,bool)|i(I,str)|c(I,str)').rename('a')
        f2 = ff.parse('s(4,8)|v(bool,str,float)|i(I,str)|c(I,str)').rename('b')
        f3 = ff.parse('s(4,7)|v(str)|i(I,str)|c(I,str)').rename('c')

        config = StoreConfigNPY()

        with temp_file('.zip') as fp:
            st = StoreZipNPY(fp, config=config)
            st.write(((f.name, f) for f in (f1, f2, f3)))

            self.assertEqual(tuple(st.labels()), ('a', 'b', 'c'))

            self.assertTrue(f1.equals(st.read('a')))
            self.assertTrue(f2.equals(st.read('b')))
            self.assertTrue(f3.equals(st.read('c')))

    def test_store_zip_npy_b(self) -> None:
        f1 = ff.parse('s(4,6)|v(int,int,bool)|i(I,str)|c(I,str)').rename('a')
        f2 = ff.parse('s(4,8)|v(bool,str,float)|i(I,str)|c(I,str)').rename('b')
        f3 = ff.parse('s(4,7)|v(str)|i(I,str)|c(I,str)').rename('c')

        config = StoreConfigNPY()

        with temp_file('.zip') as fp:
            st = StoreZipNPY(fp, config=config)
            st.write(((f.name, f) for f in (f1, f2, f3)))

            f4 = st.read('a')
            self.assertTrue(f1.equals(f4))
            self.assertTrue('a' in st._weak_cache)
            self.assertIs(f4, st.read('a'))

            post = tuple(st.read_many(('a', 'b', 'c')))
            self.assertEqual(len(post), 3)

    # ---------------------------------------------------------------------------
    def test_store_zip_npz_frame_filter_a(self) -> None:
        f1 = ff.parse('s(4,6)|v(int,int,bool)|i(I,str)|c(I,str)').rename('a')
        f2 = ff.parse('s(4,6)|v(bool,str,float)|i(I,str)|c(I,str)').rename('b')
        f3 = ff.parse('s(4,6)|v(str)|i(I,str)|c(I,str)').rename('c')

        config = StoreConfigNPZ(read_frame_filter=lambda l, f: f.iloc[:2, :2])

        with temp_file('.zip') as fp:
            st1 = StoreZipNPZ(fp, config=config)
            st1.write(((f.name, f) for f in (f1, f2, f3)))

            st2 = StoreZipNPZ(fp, config=config)
            post1 = [st2.read(l).shape for l in ('a', 'b', 'c')]
            self.assertEqual(post1, [(2, 2), (2, 2), (2, 2)])

    def test_store_zip_npz_frame_filter_b(self) -> None:
        f1 = ff.parse('s(4,6)|v(int,int,bool)|i(I,str)|c(I,str)').rename('a')
        f2 = ff.parse('s(4,6)|v(bool,str,float)|i(I,str)|c(I,str)').rename('b')
        f3 = ff.parse('s(4,6)|v(str)|i(I,str)|c(I,str)').rename('c')

        config = StoreConfigNPZ(
            read_frame_filter=lambda l, f: f.iloc[:2, :2], read_max_workers=3
        )

        with temp_file('.zip') as fp:
            st1 = StoreZipNPZ(fp)
            st1.write(((f.name, f) for f in (f1, f2, f3)))

            st2 = StoreZipNPZ(fp, config=config)
            post1 = [st2.read(l).shape for l in ('a', 'b', 'c')]
            self.assertEqual(post1, [(2, 2), (2, 2), (2, 2)])

    # ---------------------------------------------------------------------------
    def test_store_zip_npy_frame_filter_a(self) -> None:
        f1 = ff.parse('s(4,6)|v(int,int,bool)|i(I,str)|c(I,str)').rename('a')
        f2 = ff.parse('s(4,6)|v(bool,str,float)|i(I,str)|c(I,str)').rename('b')
        f3 = ff.parse('s(4,6)|v(str)|i(I,str)|c(I,str)').rename('c')

        config = StoreConfigNPY(read_frame_filter=lambda l, f: f.iloc[:2, :2])

        with temp_file('.zip') as fp:
            st1 = StoreZipNPY(fp)
            st1.write(((f.name, f) for f in (f1, f2, f3)))

            st2 = StoreZipNPY(fp, config=config)
            post1 = [st2.read(l).shape for l in ('a', 'b', 'c')]
            self.assertEqual(post1, [(2, 2), (2, 2), (2, 2)])

    def test_store_zip_npy_frame_filter_b(self) -> None:
        f1 = ff.parse('s(4,6)|v(int,int,bool)|i(I,str)|c(I,str)').rename('a')
        f2 = ff.parse('s(4,6)|v(bool,str,float)|i(I,str)|c(I,str)').rename('b')
        f3 = ff.parse('s(4,6)|v(str)|i(I,str)|c(I,str)').rename('c')

        def read_frame_filter(l, f):
            if l in ('a', 'c'):
                return f.iloc[:2, :3]
            return f

        config = StoreConfigNPY(read_frame_filter=read_frame_filter)

        with temp_file('.zip') as fp:
            st1 = StoreZipNPY(fp)
            st1.write(((f.name, f) for f in (f1, f2, f3)))

            st2 = StoreZipNPY(fp, config=config)
            post1 = [st2.read(l).shape for l in ('a', 'b', 'c')]
            self.assertEqual(post1, [(2, 3), (4, 6), (2, 3)])


if __name__ == '__main__':
    import unittest

    unittest.main()
