
import unittest

import numpy as np

from hypothesis import strategies as st
from hypothesis import given
from hypothesis import example
from hypothesis import reproduce_failure

from static_frame.test.property.strategies import get_shape_1d2d
from static_frame.test.property.strategies import get_array_1d
from static_frame.test.property.strategies import get_array_1d2d
from static_frame.test.property.strategies import get_dtype_pairs

from static_frame.test.property.strategies import get_dtype
from static_frame.test.property.strategies import get_dtypes
from static_frame.test.property.strategies import get_label
from static_frame.test.property.strategies import get_value
from static_frame.test.property.strategies import get_labels
from static_frame.test.property.strategies import get_arrays_2d_aligned_columns
from static_frame.test.property.strategies import get_arrays_2d_aligned_rows

from static_frame.test.test_case import TestCase
from static_frame.core import util


class TestUnit(TestCase):


    @given(get_array_1d2d())
    def test_mloc(self, array):

        x = util.mloc(array)
        self.assertTrue(isinstance(x, int))


    @given(get_dtype_pairs())
    def test_resolve_dtype(self, dtype_pair):

        x = util.resolve_dtype(*dtype_pair)
        self.assertTrue(isinstance(x, np.dtype))

    @given(get_dtypes(min_size=1))
    def test_resolve_dtype_iter(self, dtypes):

        x = util.resolve_dtype_iter(dtypes)
        self.assertTrue(isinstance(x, np.dtype))

    @given(get_labels(min_size=1))
    def test_resolve_type_object_iter(self, objects):

        x = util.resolve_type_object_iter(objects)
        self.assertTrue(x in (None, str, object))

    @given(get_arrays_2d_aligned_columns())
    def test_concat_resolved_axis_0(self, arrays):
        array = util.concat_resolved(arrays, axis=0)
        self.assertEqual(array.ndim, 2)
        self.assertEqual(array.dtype, util.resolve_dtype_iter((x.dtype for x in arrays)))

    @given(get_arrays_2d_aligned_rows())
    def test_concat_resolved_axis_1(self, arrays):
        array = util.concat_resolved(arrays, axis=1)
        self.assertEqual(array.ndim, 2)
        self.assertEqual(array.dtype, util.resolve_dtype_iter((x.dtype for x in arrays)))

    @given(get_dtype(), get_shape_1d2d(), get_value())
    def test_full_or_fill(self, dtype, shape, value):
        array = util.full_for_fill(dtype, shape, fill_value=value)
        self.assertTrue(array.shape == shape)
        if isinstance(value, (float, complex)) and np.isnan(value):
            pass
        else:
            self.assertTrue(value in array)

    @given(get_dtype())
    def test_dtype_to_na(self, dtype):
        post = util.dtype_to_na(dtype)
        self.assertTrue(post in {0, False, None, '', np.nan, util.NAT})


if __name__ == '__main__':
    unittest.main()
