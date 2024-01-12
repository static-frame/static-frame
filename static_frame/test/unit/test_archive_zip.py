import numpy as np

from static_frame.core.archive_zip import ZipFilePartRO
from static_frame.core.archive_zip import ZipFileRO
from static_frame.core.archive_zip import ZipInfoRO
from static_frame.core.frame import Frame
from static_frame.test.test_case import TestCase
from static_frame.test.test_case import temp_file


class TestUnit(TestCase):

    #---------------------------------------------------------------------------
    def test_to_npy_a(self) -> None:
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







