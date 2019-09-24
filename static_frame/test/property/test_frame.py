
import typing as tp
import unittest

import numpy as np  # type: ignore

# from hypothesis import strategies as st
from hypothesis import given  # type: ignore


from static_frame.core.frame import Frame
from static_frame.core.frame import FrameGO
from static_frame.core.series import Series

from static_frame.test.property import strategies as sfst

from static_frame.test.test_case import TestCase

from static_frame import Frame



class TestUnit(TestCase):



    @given(sfst.get_frame_or_frame_go())  # type: ignore
    def test_basic_attributes(self, f1: Frame) -> None:

        self.assertEqual(len(f1.dtypes), f1.shape[1])
        # self.assertEqual(f1.shape, f1.shape)
        self.assertEqual(f1.ndim, 2)
        # self.assertEqual(f1.unified, len(f1.mloc) <= 1)

        if f1.shape[0] > 0 and f1.shape[1] > 0:
            self.assertTrue(f1.size > 0)
            self.assertTrue(f1.nbytes > 0)
        else:
            self.assertTrue(f1.size == 0)
            self.assertTrue(f1.nbytes == 0)


    #---------------------------------------------------------------------------

    @given(sfst.get_frame_go(), sfst.get_label())  # type: ignore
    def test_frame_go_setitem(self, f1: Frame, label: tp.Hashable) -> None:

        shape = f1.shape
        f1['foo'] = label # type: ignore
        self.assertEqual(shape[1] + 1, f1.shape[1])


    @given(sfst.get_arrays_2d_aligned_rows(min_size=2, max_size=2))  # type: ignore
    def test_frame_go_extend(self, arrays: tp.Sequence[np.ndarray]) -> None:
        f1 = FrameGO(arrays[0], columns=self.get_letters(arrays[0].shape[1]))
        shape = f1.shape
        f2 = Frame(arrays[1])
        f1.extend(f2)
        self.assertEqual(f1.shape[1], shape[1] + f2.shape[1])


    @given(sfst.get_arrays_2d_aligned_rows(min_size=3))  # type: ignore
    def test_frame_go_extend_items(self, arrays: tp.Sequence[np.ndarray]) -> None:
        frame_array = arrays[0]
        # just take first columm form 2d arrays
        series_arrays = [a[:, 0] for a in arrays[1:]]

        f1 = FrameGO(frame_array)
        shape = f1.shape

        letters = self.get_letters(len(series_arrays))

        def items() -> tp.Iterator[tp.Tuple[tp.Hashable, Series]]:
            for idx, label in enumerate(letters):
                s = Series(series_arrays[idx], index=f1.index)
                yield label, s

        f1.extend_items(items())

        self.assertEqual(f1.shape[1], shape[1] + len(series_arrays))




if __name__ == '__main__':
    unittest.main()
