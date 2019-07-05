
import typing as tp
import unittest

import numpy as np  # type: ignore

from hypothesis import strategies as st
from hypothesis import given  # type: ignore
from hypothesis import example  # type: ignore
from hypothesis import reproduce_failure  # type: ignore

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


    @given(get_array_1d2d())  # type: ignore
    def test_mloc(self, array: np.ndarray) -> None:

        x = util.mloc(array)
        self.assertTrue(isinstance(x, int))


    @given(get_dtype_pairs())  # type: ignore
    def test_resolve_dtype(self, dtype_pair: tp.Tuple[np.dtype, np.dtype]) -> None:

        x = util.resolve_dtype(*dtype_pair)
        self.assertTrue(isinstance(x, np.dtype))

    @given(get_dtypes(min_size=1))  # type: ignore
    def test_resolve_dtype_iter(self, dtypes: tp.Iterable[np.dtype]) -> None:

        x = util.resolve_dtype_iter(dtypes)
        self.assertTrue(isinstance(x, np.dtype))

    @given(get_labels(min_size=1))  # type: ignore
    def test_resolve_type_object_iter(self, objects: tp.Iterable[object]) -> None:

        x = util.resolve_type_object_iter(objects)
        self.assertTrue(x in (None, str, object))

    @given(get_arrays_2d_aligned_columns())  # type: ignore
    def test_concat_resolved_axis_0(self, arrays: tp.List[np.ndarray]) -> None:
        array = util.concat_resolved(arrays, axis=0)
        self.assertEqual(array.ndim, 2)
        self.assertEqual(array.dtype, util.resolve_dtype_iter((x.dtype for x in arrays)))

    @given(get_arrays_2d_aligned_rows())  # type: ignore
    def test_concat_resolved_axis_1(self, arrays: tp.List[np.ndarray]) -> None:
        array = util.concat_resolved(arrays, axis=1)
        self.assertEqual(array.ndim, 2)
        self.assertEqual(array.dtype, util.resolve_dtype_iter((x.dtype for x in arrays)))

    @given(get_dtype(), get_shape_1d2d(), get_value())  # type: ignore
    def test_full_or_fill(self,
            dtype: np.dtype,
            shape: tp.Union[tp.Tuple[int], tp.Tuple[int, int]],
            value: object) -> None:
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
