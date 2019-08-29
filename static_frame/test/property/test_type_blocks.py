
import typing as tp
import unittest

import numpy as np  # type: ignore

from hypothesis import strategies as st
from hypothesis import given  # type: ignore

from static_frame.test.property import strategies as sfst

from static_frame.test.test_case import TestCase

from static_frame import TypeBlocks
# from static_frame import Index
# from static_frame import IndexHierarchy
# from static_frame import Series
# from static_frame import Frame


class TestUnit(TestCase):


    @given(st.lists(sfst.get_shape_2d(), min_size=1), sfst.get_labels(min_size=1)) # type: ignore
    def test_from_element_items(self,
            shapes: tp.List[tp.Tuple[int, int]],
            labels: tp.Sequence[tp.Hashable]
            ) -> None:

        # use shapes to get coordinates, where the max shape + 1 is the final shape
        shape = tuple(np.array(shapes).max(axis=0) + 1)

        def values() -> tp.Iterator[tp.Tuple[tp.Tuple[int, int], tp.Hashable]]:
            for idx, coord in enumerate(shapes):
                yield coord, labels[idx % len(labels)]

        post = TypeBlocks.from_element_items(values(), shape=shape, dtype=object)
        self.assertEqual(post.shape, shape)


    @given(st.integers(max_value=sfst.MAX_COLUMNS)) # type: ignore
    def test_from_zero_size_shape(self, value: int) -> None:

        for shape in ((0, value), (value, 0)):
            post = TypeBlocks.from_zero_size_shape(shape=shape)
            self.assertEqual(post.shape, shape)


    @given(sfst.get_type_blocks())  # type: ignore
    def test_basic_attributes(self, tb: TypeBlocks) -> None:
        self.assertEqual(len(tb.dtypes), tb.shape[1])
        self.assertEqual(len(tb.shapes), len(tb.mloc))
        self.assertEqual(tb.copy().shape, tb.shape)
        self.assertEqual(tb.ndim, 2)
        self.assertEqual(tb.unified, len(tb.mloc) <= 1)

        if tb.shape[0] > 0 and tb.shape[1] > 0:
            self.assertTrue(tb.size > 0)
            self.assertTrue(tb.nbytes > 0)
        else:
            self.assertTrue(tb.size == 0)
            self.assertTrue(tb.nbytes == 0)



    @given(sfst.get_type_blocks())  # type: ignore
    def test_values(self, tb: TypeBlocks) -> None:
        values = tb.values
        self.assertEqual(values.shape, tb.shape)
        self.assertEqual(values.dtype, tb._row_dtype)


    @given(sfst.get_type_blocks())  # type: ignore
    def test_axis_values(self, tb: TypeBlocks) -> None:
        # this test found a flaw in axis_values when dealing with axis 1 and unified,  1D type blocks
        for axis in (0, 1):
            for reverse in (True, False):
                post = tuple(tb.axis_values(axis=axis, reverse=reverse))
                for idx, array in enumerate(post):
                    self.assertTrue(len(array) == tb.shape[axis])
                    if axis == 0 and not reverse: # colums
                        self.assertTrue(array.dtype == tb.dtypes[idx])
                    elif axis == 0 and reverse: # colums
                        self.assertTrue(array.dtype == tb.dtypes[tb.shape[1] - 1 - idx])
                    else:
                        # NOTE: only checking kinde because found cases where byte-order deviates
                        self.assertTrue(array.dtype.kind == tb._row_dtype.kind)


    @given(sfst.get_type_blocks())  # type: ignore
    def test_element_items(self, tb: TypeBlocks) -> None:
        # NOTE: this found a flaw in _extract_iloc where we tried to optimize selection with a unified array
        count = 0
        for k, v in tb.element_items():
            count += 1
            v_extract = tb.iloc[k]
            self.assertEqualWithNaN(v, v_extract)
        self.assertEqual(count, tb.size)

    @given(sfst.get_type_blocks())  # type: ignore
    def test_reblock_signature(self, tb: TypeBlocks) -> None:
        post = tuple(tb._reblock_signature())
        unique_dtypes = np.unique(tb.dtypes)
        # the reblock signature must be have at least as many entries as types
        self.assertTrue(len(post) >= len(unique_dtypes))
        # sum of column widths is qual to columns in shape
        self.assertTrue(sum(p[1] for p in post), tb.shape[1])


    @given(sfst.get_type_blocks(), sfst.get_type_blocks())  # type: ignore
    def test_block_compatible(self, tb1: TypeBlocks, tb2: TypeBlocks) -> None:

        for axis in (None, 0, 1):
            post1 = tb1.block_compatible(tb2, axis)
            post2 = tb2.block_compatible(tb1, axis)
            # either direction gets the same result
            self.assertTrue(post1 == post2)
            # if the shapes are different, they cannot be block compatible
            if axis is None and tb1.shape != tb2.shape:
                self.assertFalse(post1)


    @given(sfst.get_type_blocks(), sfst.get_type_blocks())  # type: ignore
    def test_reblock_compatible(self, tb1: TypeBlocks, tb2: TypeBlocks) -> None:

        post1 = tb1.reblock_compatible(tb2)
        post2 = tb2.reblock_compatible(tb1)
        # either direction gets the same result
        self.assertTrue(post1 == post2)
        # if the shapes are different, they cannot be block compatible
        if tb1.shape[1] != tb2.shape[1]:
            self.assertFalse(post1)
        if post1:
            self.assertTrue(tb1.shape[1] == tb2.shape[1])
            self.assertTrue(len(tb1.shapes) == len(tb2.shapes))

    # TODO: _concatenate_blocks



    #---------------------------------------------------------------------------
    # last two methods

    @given(sfst.get_type_blocks_aligned_array())  # type: ignore
    def test_append(self, tb_aligned_array: tp.Tuple[TypeBlocks, np.ndarray]) -> None:
        tb, aa = tb_aligned_array
        shape_original = tb.shape
        tb.append(aa)
        if aa.ndim == 1:
            self.assertEqual(tb.shape[1], shape_original[1] + 1)
        else:
            self.assertEqual(tb.shape[1], shape_original[1] + aa.shape[1])

    @given(sfst.get_type_blocks_aligned_type_blocks(min_size=2, max_size=2))  # type: ignore
    def test_extend(self, tbs: tp.Sequence[TypeBlocks]) -> None:
        front = tbs[0]
        back = tbs[1]
        shape_original = front.shape
        # extend with type blocks
        front.extend(back)
        self.assertEqual(front.shape,
                (shape_original[0], shape_original[1] + back.shape[1]))

        # extend with iterable of arrays
        front.extend(back._blocks)
        self.assertEqual(front.shape,
                (shape_original[0], shape_original[1] + back.shape[1] * 2))


if __name__ == '__main__':
    unittest.main()
