import numpy as np

from static_frame.core.archive_zip import ZipFileRO
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
                print(zf)
                print(zf.infolist())
                print(zf.namelist())
                f_part = zf.open('__blocks_0__.npy')
                print(f_part.read())



