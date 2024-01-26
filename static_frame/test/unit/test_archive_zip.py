import io
from pathlib import Path
from zipfile import ZIP_DEFLATED
from zipfile import BadZipFile
from zipfile import ZipFile

import numpy as np

# from static_frame.core.archive_zip import ZipFilePartRO
from static_frame.core.archive_zip import ZipFileRO
from static_frame.core.archive_zip import ZipInfoRO
from static_frame.core.archive_zip import zip_namelist
from static_frame.core.frame import Frame
from static_frame.test.test_case import TestCase
from static_frame.test.test_case import temp_file


class TestUnit(TestCase):

    #---------------------------------------------------------------------------
    def test_zip_file_ro_a(self) -> None:
        f = Frame(np.arange(20))

        with temp_file('.zip') as fp:
            f.to_npz(fp)

            with ZipFileRO(fp) as zf:
                self.assertTrue(repr(zf).startswith('<ZipFileRO'))
                self.assertEqual(zf.namelist(), ['__blocks_0__.npy', '__meta__.json'])

                self.assertEqual(len(zf.infolist()), 2)
                self.assertTrue(all(isinstance(zinfo, ZipInfoRO) for zinfo in zf.infolist()))

                zinfo = zf.getinfo('__meta__.json')
                self.assertEqual(zinfo.filename, '__meta__.json')

                self.assertEqual(
                    zf.read('__meta__.json'),
                    b'{"__names__": [null, null, null], "__types__": ["Index", "Index"], "__depths__": [1, 1, 1]}'
                    )

                with zf.open('__meta__.json') as zfpart:
                    self.assertTrue(zfpart.seekable())
                    self.assertEqual(zfpart.read(),
                            b'{"__names__": [null, null, null], "__types__": ["Index", "Index"], "__depths__": [1, 1, 1]}'
                            )

                with zf.open('__meta__.json') as zfpart:
                    self.assertTrue(zfpart.seekable())
                    self.assertEqual(zfpart.read(),
                            b'{"__names__": [null, null, null], "__types__": ["Index", "Index"], "__depths__": [1, 1, 1]}'
                            )

    def test_zip_file_ro_b(self) -> None:
        with temp_file('.zip') as fp:
            with ZipFile(fp, 'w') as zf:
                zf.writestr('foo', b'0')
            # using a Path for init
            with ZipFileRO(Path(fp)) as zfro:
                self.assertEqual(len(zfro), 1)


    #---------------------------------------------------------------------------
    def test_zip_file_ro_close_a(self) -> None:
        with temp_file('.zip') as fp:
            zf = ZipFile(fp, 'w') #pylint: disable=R1732
            zf.writestr(str('foo'), b'0')
            zf.close()

            zfro = ZipFileRO(fp)
            zfro.close()

            self.assertTrue(repr(zfro).endswith('[closed]>'))
            with self.assertRaises(ValueError):
                zfro.open('foo')

            with self.assertRaises(ValueError):
                zfro.open('foo')

            self.assertEqual(zfro.close(), None)

    def test_zip_file_ro_close_b(self) -> None:
        with temp_file('.zip') as fp:
            zf = ZipFile(fp, 'w') #pylint: disable=R1732
            zf.writestr(str('foo'), b'0')
            zf.close()

            with ZipFileRO(fp) as zfro:
                self.assertEqual(len(zfro), 1)
                zfpart = zfro.open('foo')
                zfpart.close()

                with self.assertRaises(ValueError):
                    zfpart.seekable()

                with self.assertRaises(ValueError):
                    zfpart.seek(0)

                with self.assertRaises(ValueError):
                    zfpart.read(1)

                buffer = io.BytesIO()
                with self.assertRaises(ValueError):
                    zfpart.readinto(buffer)

                with self.assertRaises(NotImplementedError):
                    zfpart.write(b'x')

    #---------------------------------------------------------------------------
    def test_zip_file_ro_repr_a(self) -> None:
        with temp_file('.zip') as fp:
            with ZipFile(fp, 'w') as zf:
                zf.writestr('foo', b'0')

            with open(fp, 'rb') as f:
                with ZipFileRO(f) as zfro:
                    self.assertTrue(str(zfro).startswith('<ZipFileRO file=<'))

    #---------------------------------------------------------------------------
    def test_zip_file_ro_seek_a(self) -> None:
        with temp_file('.zip') as fp:
            with ZipFile(fp, 'w') as zf:
                zf.writestr('foo', b'0')

            with ZipFileRO(fp) as zfro:
                zfpart = zfro.open('foo')
                p = zfpart.tell()
                self.assertEqual(zfpart.seek(1, 1), p + 1)

                with self.assertRaises(NotImplementedError):
                    zfpart.seek(1, 0)

                with self.assertRaises(NotImplementedError):
                    zfpart.seek(1, 2)

    #---------------------------------------------------------------------------
    def test_zip_file_ro_zip64_a(self) -> None:
        # try to force a zip64 by exceeding 65,535 file limit

        count = 65_536 # limit is  65,535

        with temp_file('.zip') as fp:
            with ZipFile(fp, 'w') as zf:
                for i in range(count):
                    zf.writestr(str(i), b'0')

            with ZipFileRO(fp) as zfro:
                self.assertEqual(len(zfro), count)

    #---------------------------------------------------------------------------
    def test_zip_file_ro_comment_a(self) -> None:
        with temp_file('.zip') as fp:
            with ZipFile(fp, 'w') as zf:
                zf.writestr(str('foo'), b'0')
                zf.comment = b'bar'
            with ZipFileRO(fp) as zfro:
                self.assertEqual(len(zfro), 1)

    def test_zip_file_ro_bad_zip_a(self) -> None:
        with temp_file('.zip') as fp:
            with ZipFile(fp, 'w') as zf:
                zf.writestr(str('foo'), b'0')

            with open(fp, 'rb') as f:
                data = f.read()

            # mutate last byte
            data = data[:-5]
            with open(fp, 'wb') as f:
                f.write(data)

            with self.assertRaises(BadZipFile):
                ZipFileRO(fp)

    def test_zip_file_ro_bad_zip_b(self) -> None:
        with temp_file('.zip') as fp:
            with ZipFile(fp, 'w', compression=ZIP_DEFLATED) as zf:
                zf.writestr(str('foo'), b'0')
                zf.comment = b'bar'

            with self.assertRaises(BadZipFile):
                with ZipFileRO(fp) as zfro:
                    pass


    #---------------------------------------------------------------------------
    def test_zip_namelist_a(self) -> None:

        with temp_file('.zip') as fp:
            with ZipFile(fp, 'w') as zf:
                for i in range(4):
                    zf.writestr(str(i), b'0')

            self.assertEqual(list(zip_namelist(fp)), ['0', '1', '2', '3'])

    def test_zip_namelist_b(self) -> None:

        with temp_file('.zip') as fp:
            with ZipFile(fp, 'w', compression=ZIP_DEFLATED) as zf:
                for i in range(4):
                    zf.writestr(str(i), b'0')

            self.assertEqual(list(zip_namelist(fp)), ['0', '1', '2', '3'])

