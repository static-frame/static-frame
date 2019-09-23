
import typing as tp
import unittest

import numpy as np  # type: ignore

from hypothesis import strategies as st
from hypothesis import given  # type: ignore

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


    @given(sfst.get_frame_go(), sfst.get_label())  # type: ignore
    def test_frame_go_setitem(self, f1: Frame, label: tp.Hashable) -> None:

        shape = f1.shape
        f1['foo'] = label
        self.assertEqual(shape[1] + 1, f1.shape[1])
        # import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    unittest.main()
