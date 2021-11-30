
import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
from numpy.lib.format import write_array # type: ignore

from static_frame.core.archive_npy import NPYConverter
from static_frame.core.archive_npy import ArchiveDirectory
from static_frame.core.archive_npy import ArchiveZip

from static_frame.core.exception import ErrorNPYDecode
from static_frame.core.exception import ErrorNPYEncode

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

if __name__ == '__main__':
    unittest.main()
