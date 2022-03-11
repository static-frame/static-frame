import os
from tempfile import TemporaryDirectory
from io import UnsupportedOperation

import numpy as np
from numpy.lib.format import write_array # type: ignore
import frame_fixtures as ff

from static_frame.core.frame import Frame
from static_frame.core.index import Index
from static_frame.core.archive_npy import NPYConverter
from static_frame.core.archive_npy import ArchiveDirectory
from static_frame.core.archive_npy import ArchiveZip
from static_frame.core.archive_npy import NPZ
from static_frame.core.archive_npy import NPY

from static_frame.core.exception import ErrorNPYDecode
from static_frame.core.exception import ErrorNPYEncode
from static_frame.core.exception import AxisInvalid

from static_frame.test.test_case import temp_file
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    #---------------------------------------------------------------------------
    def test_to_npy_a(self) -> None:
        a1 = np.arange(20)

        with temp_file('.npy') as fp:
            with open(fp, 'wb') as f:
                NPYConverter.to_npy(f, a1)

            a2 = np.load(fp)
            self.assertTrue((a1 == a2).all())

    def test_to_npy_b(self) -> None:
        a1 = np.array([None, 'foo', 3], dtype=object)

        with temp_file('.npy') as fp:
            with open(fp, 'wb') as f:
                with self.assertRaises(ErrorNPYEncode):
                    NPYConverter.to_npy(f, a1)

    def test_to_npy_c(self) -> None:
        a1 = np.arange(12).reshape(2, 3, 2)
        with temp_file('.npy') as fp:
            with open(fp, 'wb') as f:
                with self.assertRaises(ErrorNPYEncode):
                    NPYConverter.to_npy(f, a1)

    def test_to_npy_d(self) -> None:
        a1 = np.arange(12).reshape(2,6).T

        with temp_file('.npy') as fp:
            with open(fp, 'wb') as f:
                NPYConverter.to_npy(f, a1)
            with open(fp, 'rb') as f:
                a2, _ = NPYConverter.from_npy(f, {})

            self.assertTrue((a1 == a2).all())

    def test_to_npy_e(self) -> None:
        a1 = np.arange(4)
        with temp_file('.npy') as fp:

            with open(fp, 'wb') as f:
                NPYConverter.to_npy(f, a1)
            # ensure compatibility with numpy loaders
            a2 = np.load(fp)
            self.assertTrue((a1 == a2).all())

    def test_to_npy_f(self) -> None:
        a1 = np.array([('Rex', 9, 81.0), ('Fido', 3, 27.0)],
                dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')]
                )
        with temp_file('.npy') as fp:
            with open(fp, 'wb') as f:
                with self.assertRaises(ErrorNPYEncode):
                    NPYConverter.to_npy(f, a1)

    def test_from_npy_a(self) -> None:
        a1 = np.arange(20)

        with temp_file('.npy') as fp:
            with open(fp, 'wb') as f:
                NPYConverter.to_npy(f, a1)
            with open(fp, 'rb') as f:
                a2, _ = NPYConverter.from_npy(f, {})
            self.assertTrue((a1 == a2).all())

    def test_from_npy_b(self) -> None:
        a1 = np.arange(100).reshape(5, 20)

        with temp_file('.npy') as fp:
            with open(fp, 'wb') as f:
                NPYConverter.to_npy(f, a1)
            with open(fp, 'rb') as f:
                a2, _ = NPYConverter.from_npy(f, {})

            self.assertTrue(a1.shape == a2.shape)
            self.assertTrue((a1 == a2).all())

    def test_from_npy_c(self) -> None:
        with temp_file('.npy') as fp:
            with open(fp, 'wb') as f:
                f.write(b'foo')

            with open(fp, 'rb') as f:
                # invaliud header raises
                with self.assertRaises(ErrorNPYDecode):
                    a2, _ = NPYConverter.from_npy(f, {})

    def test_from_npy_d(self) -> None:
        a1 = np.arange(12).reshape(2, 3, 2)

        with temp_file('.npy') as fp:
            with open(fp, 'wb') as f:
                write_array(f, a1, version=(1, 0))

            with open(fp, 'rb') as f:
                # invalid shape
                with self.assertRaises(ErrorNPYDecode):
                    a2, _ = NPYConverter.from_npy(f, {})

    def test_from_npy_e(self) -> None:
        a1 = np.array([2, 3, 4])

        with temp_file('.npy') as fp:
            with open(fp, 'wb') as f:
                write_array(f, a1, version=(3, 0))

            with open(fp, 'rb') as f:
                # invlid header; only version 1,0 is supported
                with self.assertRaises(ErrorNPYDecode):
                    a2, _ = NPYConverter.from_npy(f, {})

    def test_from_npy_f(self) -> None:
        a1 = np.array([None, 'foo', 3], dtype=object)

        with temp_file('.npy') as fp:
            with open(fp, 'wb') as f:
                write_array(f, a1, version=(1, 0))

            with open(fp, 'rb') as f:
                # invalid object dtype
                with self.assertRaises(ErrorNPYDecode):
                    a2, _ = NPYConverter.from_npy(f, {})

    def test_from_npy_g(self) -> None:
        a1 = np.array([2, 3, 4])

        with temp_file('.npy') as fp:
            with open(fp, 'wb') as f:
                NPYConverter.to_npy(f, a1)
            with open(fp, 'rb') as f:
                a2, _ = NPYConverter.from_npy(f, {}, memory_map=True)
                self.assertEqual(a2.tolist(), [2, 3, 4])

    def test_from_npy_h(self) -> None:
        with temp_file('.npy') as fp:
            with open(fp, 'wb') as f:
                f.write(b'foo')

            with open(fp, 'rb') as f:
                # invaliud header raises
                with self.assertRaises(ErrorNPYDecode):
                    a2, _ = NPYConverter.header_from_npy(f, {})

    #---------------------------------------------------------------------------

    def test_archive_zip_a(self) -> None:
        with temp_file('.zip') as fp:
            with self.assertRaises(RuntimeError):
                _ = ArchiveZip(fp, writeable=True, memory_map=True)

    def test_archive_directory_a(self) -> None:
        with temp_file('.npy') as fp:
            with self.assertRaises(RuntimeError):
                ArchiveDirectory(fp, writeable=False, memory_map=False)

    def test_archive_directory_b(self) -> None:
        with TemporaryDirectory() as fp:
            os.rmdir(fp)
            # creates directory
            ad = ArchiveDirectory(fp, writeable=True, memory_map=False)

    def test_archive_directory_c(self) -> None:
        with TemporaryDirectory() as fp:
            os.rmdir(fp)
            # reading from a non-existant directory
            with self.assertRaises(RuntimeError):
                ad = ArchiveDirectory(fp, writeable=False, memory_map=False)
            os.mkdir(fp) # restore the directory for context manager

    def test_archive_directory_d(self) -> None:
        with TemporaryDirectory() as fp:
            a1 = np.arange(10)
            ad1 = ArchiveDirectory(fp, writeable=True, memory_map=False)
            ad1.write_array('a1.npy', a1)

            ad2 = ArchiveDirectory(fp, writeable=False, memory_map=False)
            a2 = ad2.read_array('a1.npy')
            self.assertTrue((a1 == a2).all())

    #---------------------------------------------------------------------------
    def test_archive_components_npz_write_arrays_a(self) -> None:
        with temp_file('.zip') as fp:

            a1 = np.arange(12).reshape(3, 4)
            NPZ(fp, 'w').from_arrays(blocks=(a1,))

            f = Frame.from_npz(fp)
            self.assertEqual(f.values.tolist(), a1.tolist())
            self.assertIs(f.index._map, None)
            self.assertIs(f.columns._map, None)

    def test_archive_components_npz_write_arrays_b(self) -> None:
        with temp_file('.zip') as fp:

            a1 = np.arange(12).reshape(3, 4)
            a2 = np.array([3, 4])

            with self.assertRaises(RuntimeError):
                NPZ(fp, 'w').from_arrays(blocks=(a1, a2), axis=1)

            with self.assertRaises(RuntimeError):
                NPZ(fp, 'w').from_arrays(blocks=(a2, a1), axis=1)

    def test_archive_components_npz_write_arrays_c(self) -> None:
        with temp_file('.zip') as fp:
            a1 = np.arange(12).reshape(3, 4)
            index = Index((10, 20, 30))
            NPZ(fp, 'w').from_arrays(blocks=(a1,), index=index)

    def test_archive_components_npz_write_arrays_d(self) -> None:
        with temp_file('.zip') as fp:
            from static_frame.core.index_datetime import IndexYear

            a1 = np.arange(12).reshape(3, 4)
            a2 = np.array(['a', 'b', 'c'])
            a3 = np.array([True, False, True])

            index = np.array(['2021', '2022', '1542'], dtype='datetime64[Y]')
            NPZ(fp, 'w').from_arrays(blocks=(a1, a2, a3), index=index, axis=1)
            f = Frame.from_npz(fp)
            self.assertIs(f.index.__class__, IndexYear)
            self.assertEqual([dt.kind for dt in f.dtypes.values],
                    ['i', 'i', 'i', 'i', 'U', 'b'])

    def test_archive_components_npz_write_arrays_e(self) -> None:
        with temp_file('.zip') as fp:

            a1 = np.arange(12).reshape(3, 4)
            a2 = np.array([3, 4])

            with self.assertRaises(AxisInvalid):
                NPZ(fp, 'w').from_arrays(blocks=(a1, a2), axis=3)

    def test_archive_components_npz_write_arrays_f(self) -> None:
        a1 = np.arange(12).reshape(3, 4)
        a2 = np.array([10, 20, 30, 40]).reshape(1, 4)
        a3 = np.arange(8).reshape(2, 4)

        with temp_file('.zip') as fp:

            NPZ(fp, 'w').from_arrays(blocks=(a1, a2, a3), axis=0)
            f = Frame.from_npz(fp)
            self.assertEqual(f.shape, (6, 4))

    def test_archive_components_npz_write_arrays_g(self) -> None:

        a1 = np.arange(12).reshape(3, 4)
        a2 = np.array(['a', 'b', 'c'])
        a3 = np.array([True, False, True])

        with temp_file('.zip') as fp:
            index = Index((10, 20, 30), name='foo')
            NPZ(fp, 'w').from_arrays(blocks=(a1, a2, a3), index=index, name='bar', axis=1)

            f = Frame.from_npz(fp)
            self.assertEqual(f.to_pairs(),
                    ((0, ((10, 0), (20, 4), (30, 8))), (1, ((10, 1), (20, 5), (30, 9))), (2, ((10, 2), (20, 6), (30, 10))), (3, ((10, 3), (20, 7), (30, 11))), (4, ((10, 'a'), (20, 'b'), (30, 'c'))), (5, ((10, True), (20, False), (30, True))))
                    )
            self.assertEqual(f.name, 'bar')
            self.assertEqual(f.index.name, 'foo')

    def test_archive_components_npz_write_arrays_h(self) -> None:

        a1 = np.arange(12).reshape(3, 4)
        a2 = np.array(['a', 'b', 'c'])
        a3 = np.array([True, False, True])

        with temp_file('.zip') as fp:
            columns=Index(('a', 'b', 'c', 'd', 'e', 'f'), name='foo')
            NPZ(fp, 'w').from_arrays(blocks=(a1, a2, a3), columns=columns, name='bar', axis=1)

            f = Frame.from_npz(fp)
            self.assertEqual(f.to_pairs(),
                    (('a', ((0, 0), (1, 4), (2, 8))), ('b', ((0, 1), (1, 5), (2, 9))), ('c', ((0, 2), (1, 6), (2, 10))), ('d', ((0, 3), (1, 7), (2, 11))), ('e', ((0, 'a'), (1, 'b'), (2, 'c'))), ('f', ((0, True), (1, False), (2, True))))
                    )
            self.assertEqual(f.name, 'bar')
            self.assertEqual(f.columns.name, 'foo')

    def test_archive_components_npz_write_arrays_i(self) -> None:
        with temp_file('.zip') as fp:

            a1 = np.arange(12).reshape(3, 4)
            a2 = np.array([3, 4, 5])

            with self.assertRaises(RuntimeError):
                NPZ(fp, 'w').from_arrays(blocks=(a1, a2), axis=1, index=(3, 4, 5))

    def test_archive_components_npz_write_arrays_j(self) -> None:
        with temp_file('.zip') as fp:
            a1 = np.arange(12).reshape(3, 4)
            with self.assertRaises(RuntimeError):
                NPZ(fp, 'foo').from_arrays(blocks=(a1,))

    #-----------------------------------------------------------------------------

    def test_archive_components_npy_write_arrays_h(self) -> None:

        a1 = np.arange(12).reshape(3, 4)
        a2 = np.array(['a', 'b', 'c'])
        a3 = np.array([True, False, True])

        with TemporaryDirectory() as fp:
            columns=Index(('a', 'b', 'c', 'd', 'e', 'f'), name='foo')
            NPY(fp, 'w').from_arrays(blocks=(a1, a2, a3), columns=columns, name='bar', axis=1)

            f = Frame.from_npy(fp)
            self.assertEqual(f.to_pairs(),
                    (('a', ((0, 0), (1, 4), (2, 8))), ('b', ((0, 1), (1, 5), (2, 9))), ('c', ((0, 2), (1, 6), (2, 10))), ('d', ((0, 3), (1, 7), (2, 11))), ('e', ((0, 'a'), (1, 'b'), (2, 'c'))), ('f', ((0, True), (1, False), (2, True))))
                    )
            self.assertEqual(f.name, 'bar')
            self.assertEqual(f.columns.name, 'foo')

    def test_archive_components_npy_write_arrays_i(self) -> None:

        a1 = np.arange(12).reshape(3, 4)
        a2 = np.array(['a', 'b', 'c'])
        a3 = np.array([True, False, True])

        with TemporaryDirectory() as fp:
            columns=('a', 'b', 'c', 'd', 'e', 'f')
            with self.assertRaises(RuntimeError):
                NPY(fp, 'w').from_arrays(blocks=(a1, a2, a3), columns=columns, name='bar')

    def test_archive_components_npy_write_arrays_j(self) -> None:

        a1 = np.arange(12).reshape(3, 4)
        a2 = np.array(['a', 'b', 'c'])
        a3 = np.array([True, False, True])

        with TemporaryDirectory() as fp:
            columns=np.arange(6).astype('datetime64[D]')
            NPY(fp, 'w').from_arrays(blocks=(a1, a2, a3), columns=columns, name='bar', axis=1)
            f = Frame.from_npy(fp)
            dt64 = np.datetime64
            self.assertEqual(f.to_pairs(),
                    ((dt64('1970-01-01'), ((0, 0), (1, 4), (2, 8))), (dt64('1970-01-02'), ((0, 1), (1, 5), (2, 9))), (dt64('1970-01-03'), ((0, 2), (1, 6), (2, 10))), (dt64('1970-01-04'), ((0, 3), (1, 7), (2, 11))), (dt64('1970-01-05'), ((0, 'a'), (1, 'b'), (2, 'c'))), (dt64('1970-01-06'), ((0, True), (1, False), (2, True))))
                    )

    def test_archive_components_npy_write_arrays_k(self) -> None:

        a1 = np.arange(12).reshape(3, 4)
        a2 = np.array(['a', 'b', 'c'])
        a3 = np.array([True, False, True])

        with TemporaryDirectory() as fp:
            with self.assertRaises(UnsupportedOperation):
                NPY(fp, 'r').from_arrays(blocks=(a1, a2, a3))

    #-----------------------------------------------------------------------------

    def test_archive_components_npz_from_frames_a(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').relabel(index=('a', 'b'))
        f2 = ff.parse('s(2,2)|v(int)').relabel(index=('c', 'd'))

        with TemporaryDirectory() as fp:
            NPY(fp, 'w').from_frames(frames=(f1, f2), axis=0)

            f = Frame.from_npy(fp)
            self.assertEqual(f.to_pairs(),
                    ((0, (('a', -88017), ('b', 92867), ('c', -88017), ('d', 92867))), (1, (('a', 162197), ('b', -41157), ('c', 162197), ('d', -41157))))
                    )

    def test_archive_components_npz_from_frames_b(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').relabel(index=('a', 'b'))
        f2 = ff.parse('s(2,2)|v(int)').relabel(index=('c', 'd'))

        with TemporaryDirectory() as fp:
            NPY(fp, 'w').from_frames(frames=(f1, f2), axis=0, include_index=False)

            f = Frame.from_npy(fp)
            self.assertEqual(f.to_pairs(),
                    ((0, ((0, -88017), (1, 92867), (2, -88017), (3, 92867))), (1, ((0, 162197), (1, -41157), (2, 162197), (3, -41157))))
                    )

    def test_archive_components_npz_from_frames_c(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').relabel(columns=('a', 'b'))
        f2 = ff.parse('s(2,2)|v(int)').relabel(columns=('c', 'd'))

        with TemporaryDirectory() as fp:
            NPY(fp, 'w').from_frames(frames=(f1, f2), axis=1)

            f = Frame.from_npy(fp)
            self.assertEqual(f.to_pairs(),
                    (('a', ((0, -88017), (1, 92867))), ('b', ((0, 162197), (1, -41157))), ('c', ((0, -88017), (1, 92867))), ('d', ((0, 162197), (1, -41157))))
                    )

    def test_archive_components_npz_from_frames_d(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').relabel(columns=('a', 'b'))
        f2 = ff.parse('s(2,2)|v(int)').relabel(columns=('c', 'd'))

        with TemporaryDirectory() as fp:
            NPY(fp, 'w').from_frames(frames=(f1, f2), axis=1, include_columns=False)

            f = Frame.from_npy(fp)
            self.assertEqual(f.to_pairs(),
                    ((0, ((0, -88017), (1, 92867))), (1, ((0, 162197), (1, -41157))), (2, ((0, -88017), (1, 92867))), (3, ((0, 162197), (1, -41157))))
                    )

    def test_archive_components_npz_from_frames_e(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)')
        f2 = ff.parse('s(2,2)|v(int)')

        with TemporaryDirectory() as fp:
            with self.assertRaises(RuntimeError):
                NPY(fp, 'w').from_frames(frames=(f1, f2), axis=0)

    def test_archive_components_npz_from_frames_f(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)')
        f2 = ff.parse('s(2,2)|v(int)')

        with TemporaryDirectory() as fp:
            with self.assertRaises(RuntimeError):
                NPY(fp, 'w').from_frames(frames=(f1, f2), axis=1)

    def test_archive_components_npz_from_frames_g(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').relabel(columns=('a', 'b'))
        f2 = ff.parse('s(2,2)|v(int)').relabel(columns=('c', 'd'))

        with TemporaryDirectory() as fp:
            with self.assertRaises(RuntimeError):
                NPY(fp, 'w').from_frames(frames=(f1, f2), axis=1, include_index=False)

    def test_archive_components_npz_from_frames_h(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').relabel(index=('a', 'b'))
        f2 = ff.parse('s(2,2)|v(int)').relabel(index=('c', 'd'))

        with TemporaryDirectory() as fp:
            with self.assertRaises(RuntimeError):
                NPY(fp, 'w').from_frames(frames=(f1, f2), axis=0, include_columns=False)

    def test_archive_components_npz_from_frames_i(self) -> None:
        f1 = ff.parse('s(2,2)|v(float)').relabel(index=('a', 'b'))
        f2 = ff.parse('s(2,2)|v(float)').relabel(index=('b', 'c'))

        with TemporaryDirectory() as fp:
            NPY(fp, 'w').from_frames(frames=(f1, f2), axis=1, include_columns=False)
            f = Frame.from_npy(fp).fillna(0)
            self.assertEqual(f.to_pairs(),
                    ((0, (('a', 1930.4), ('b', -1760.34), ('c', 0))), (1, (('a', -610.8), ('b', 3243.94), ('c', 0))), (2, (('a', 0), ('b', 1930.4), ('c', -1760.34))), (3, (('a', 0), ('b', -610.8), ('c', 3243.94))))
                    )

    def test_archive_components_npz_from_frames_j(self) -> None:
        f1 = ff.parse('s(2,2)|v(float)').relabel(columns=('a', 'b'))
        f2 = ff.parse('s(2,2)|v(float)').relabel(columns=('b', 'c'))

        with TemporaryDirectory() as fp:
            NPY(fp, 'w').from_frames(frames=(f1, f2), axis=0, include_index=False)
            f = Frame.from_npy(fp).fillna(0)
            self.assertEqual(f.to_pairs(),
                    (('a', ((0, 1930.4), (1, -1760.34), (2, 0.0), (3, 0.0))), ('b', ((0, -610.8), (1, 3243.94), (2, 1930.4), (3, -1760.34))), ('c', ((0, 0.0), (1, 0.0), (2, -610.8), (3, 3243.94))))
                    )

    def test_archive_components_npz_from_frames_k(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').relabel(index=('a', 'b'))
        f2 = ff.parse('s(2,2)|v(int)').relabel(index=('c', 'd'))

        with TemporaryDirectory() as fp:
            with self.assertRaises(RuntimeError):
                NPY(fp, 'w').from_frames(frames=(f1, f2), axis=3)

    def test_archive_components_npz_from_frames_l(self) -> None:
        f1 = ff.parse('s(2,2)|v(float)').relabel(columns=('a', 'b'))
        f2 = ff.parse('s(2,2)|v(float)').relabel(columns=('b', 'c'))

        with TemporaryDirectory() as fp:
            with NPY(fp, 'w') as npy:
                npy.from_frames(frames=(f1, f2), axis=0, include_index=False)
                f = Frame.from_npy(fp).fillna(0)
                self.assertEqual(f.to_pairs(),
                        (('a', ((0, 1930.4), (1, -1760.34), (2, 0.0), (3, 0.0))), ('b', ((0, -610.8), (1, 3243.94), (2, 1930.4), (3, -1760.34))), ('c', ((0, 0.0), (1, 0.0), (2, -610.8), (3, 3243.94))))
                        )

    def test_archive_components_npz_from_frames_m(self) -> None:
        f1 = ff.parse('s(2,2)|v(int)').relabel(index=('a', 'b'))
        f2 = ff.parse('s(2,2)|v(int)').relabel(index=('c', 'd'))

        with TemporaryDirectory() as fp:
            with self.assertRaises(UnsupportedOperation):
                NPY(fp, 'r').from_frames(frames=(f1, f2), axis=3)

    #-----------------------------------------------------------------------------

    def test_archive_components_npy_contents_a(self) -> None:
        f1 = ff.parse('s(2,4)|v(int,str,bool,bool)').relabel(index=('a', 'b'))

        with TemporaryDirectory() as fp:
            f1.to_npy(fp)

            npy = NPY(fp, 'r')
            post = npy.contents
            self.assertEqual(post.shape, (5, 4))
            self.assertEqual(post['size'].sum(), npy.nbytes)

    def test_archive_components_npy_contents_b(self) -> None:
        f1 = ff.parse('s(2,4)|v(int,str,bool,bool)').relabel(index=('a', 'b'))

        with TemporaryDirectory() as fp:
            f1.to_npy(fp)
            with self.assertRaises(UnsupportedOperation):
                _ = NPY(fp, 'w').contents

    def test_archive_components_npz_contents_a(self) -> None:
        f1 = ff.parse('s(2,4)|v(int,str,bool,bool)').relabel(index=('a', 'b'))

        with temp_file('.zip') as fp:
            f1.to_npz(fp)
            post = NPZ(fp).contents
            self.assertEqual(post.shape, (5, 4))
            self.assertTrue(post['size'].sum() > 0)

    def test_archive_components_npy_nbytes_a(self) -> None:
        f1 = ff.parse('s(2,4)|v(int,str,bool,bool)').relabel(index=('a', 'b'))

        with TemporaryDirectory() as fp:
            f1.to_npy(fp)
            npy = NPY(fp, 'r')
            self.assertEqual(npy.contents['size'].sum(), npy.nbytes)

    def test_archive_components_npy_nbytes_b(self) -> None:
        f1 = ff.parse('s(2,4)|v(int,str,bool,bool)').relabel(index=('a', 'b'))

        with TemporaryDirectory() as fp:
            f1.to_npy(fp)
            npy = NPY(fp, 'w')
            with self.assertRaises(UnsupportedOperation):
                _ = npy.nbytes


if __name__ == '__main__':
    import unittest
    unittest.main()
