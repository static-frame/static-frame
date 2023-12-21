import numpy as np
import typing_extensions as tp  # pylint: disable=W0611
from hypothesis import given

from static_frame.core.archive_npy import NPYConverter
from static_frame.core.frame import Frame
from static_frame.core.index_datetime import IndexDate
from static_frame.core.util import DTYPE_INEXACT_KINDS
from static_frame.core.util import DTYPE_NAT_KINDS
from static_frame.core.util import WarningsSilent
from static_frame.test.property import strategies as sfst
from static_frame.test.test_case import TestCase
from static_frame.test.test_case import temp_file

if tp.TYPE_CHECKING:
    from static_frame.core.archive_npy import HeaderDecodeCacheType  # pylint: disable=W0611 #pragma: no cover

class TestUnit(TestCase):

    @given(sfst.get_frame(dtype_group=sfst.DTGroup.ALL_NO_OBJECT,
            index_dtype_group=sfst.DTGroup.BASIC,
            columns_dtype_group=sfst.DTGroup.BASIC,
            ))
    def test_frame_to_npz_a(self, f1: Frame) -> None:
        # if f1.columns.dtype.kind != 'O' and f1.index.dtype.kind != 'O':
        with temp_file('.npz') as fp:
            f1.to_npz(fp)
            f2 = Frame.from_npz(fp)
            self.assertTrue(f1.equals(f2, compare_name=True, compare_dtype=True, compare_class=True))

    @given(sfst.get_frame(dtype_group=sfst.DTGroup.ALL_NO_OBJECT,
            index_cls=IndexDate,
            index_dtype_group=sfst.DTGroup.DATE,
            columns_cls=IndexDate,
            columns_dtype_group=sfst.DTGroup.DATE,
            ))
    def test_frame_to_npz_b(self, f1: Frame) -> None:
        # if f1.columns.dtype.kind != 'O' and f1.index.dtype.kind != 'O':
        with temp_file('.npz') as fp:
            f1.to_npz(fp)
            f2 = Frame.from_npz(fp)
            self.assertTrue(f1.equals(f2, compare_name=True, compare_dtype=True, compare_class=True))

    @given(sfst.get_array_1d2d(dtype_group=sfst.DTGroup.ALL_NO_OBJECT))
    def test_frame_to_npy_a(self, a1: Frame) -> None:

        header_decode_cache: HeaderDecodeCacheType = {}

        with temp_file('.npy') as fp, WarningsSilent():
            with open(fp, 'wb') as f:
                NPYConverter.to_npy(f, a1)

            # check compatibility with built-in NPY reading
            a2 = np.load(fp)
            if a2.dtype.kind in DTYPE_INEXACT_KINDS:
                self.assertAlmostEqualArray(a1, a2)
            elif a2.dtype.kind in DTYPE_NAT_KINDS:
                assert a1.shape == a2.shape
                for v1, v2 in zip(a1.flat, a2.flat):
                    self.assertEqualWithNaN(v1, v2)
            else:
                self.assertTrue((a1 == a2).all())
            self.assertTrue(a1.shape == a2.shape)

            with open(fp, 'rb') as f:
                a3, _ = NPYConverter.from_npy(f, header_decode_cache)
                if a3.dtype.kind in DTYPE_INEXACT_KINDS:
                    self.assertAlmostEqualArray(a1, a3)
                elif a3.dtype.kind in DTYPE_NAT_KINDS:
                    assert a3.shape == a1.shape
                    for v1, v2 in zip(a3.flat, a1.flat):
                        self.assertEqualWithNaN(v1, v2)
                else:
                    self.assertTrue((a1 == a3).all())
                self.assertTrue(a1.shape == a3.shape)

    @given(sfst.get_array_1d2d(dtype_group=sfst.DTGroup.ALL_NO_OBJECT))
    def test_frame_to_npy_b(self, a1: np.ndarray) -> None:

        header_decode_cache: HeaderDecodeCacheType = {}

        with temp_file('.npy') as fp, WarningsSilent():
            with open(fp, 'wb') as f:
                NPYConverter.to_npy(f, a1)

            with open(fp, 'rb') as f:
                a2, mm = NPYConverter.from_npy(f,
                        header_decode_cache,
                        memory_map=True,
                        )
                if a2.dtype.kind in DTYPE_INEXACT_KINDS:
                    self.assertAlmostEqualArray(a1, a2)
                elif a2.dtype.kind in DTYPE_NAT_KINDS:
                    assert a1.shape == a2.shape
                    for v1, v2 in zip(a1.flat, a2.flat):
                        self.assertEqualWithNaN(v1, v2)
                else:
                    self.assertTrue((a1 == a2).all())
                self.assertTrue(a1.shape == a2.shape)


if __name__ == '__main__':
    import unittest
    unittest.main()
