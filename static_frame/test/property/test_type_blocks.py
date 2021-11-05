
import typing as tp
import unittest

import numpy as np

from hypothesis import strategies as st
from hypothesis import given

from static_frame.test.property import strategies as sfst

from static_frame.test.test_case import TestCase

from static_frame import TypeBlocks
# from static_frame import Index
# from static_frame import IndexHierarchy
# from static_frame import Series
# from static_frame import Frame


class TestUnit(TestCase):


    @given(st.lists(sfst.get_shape_2d(), min_size=1), sfst.get_labels(min_size=1))
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


    @given(st.integers(max_value=sfst.MAX_COLUMNS))
    def test_from_zero_size_shape(self, value: int) -> None:

        for shape in ((0, value), (value, 0)):
            post = TypeBlocks.from_zero_size_shape(shape=shape)
            self.assertEqual(post.shape, shape)


    @given(sfst.get_type_blocks())
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



    @given(sfst.get_type_blocks())
    def test_values(self, tb: TypeBlocks) -> None:
        values = tb.values
        self.assertEqual(values.shape, tb.shape)
        self.assertEqual(values.dtype, tb._row_dtype)


    @given(sfst.get_type_blocks())
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


    @given(sfst.get_type_blocks())
    def test_element_items(self, tb: TypeBlocks) -> None:
        # NOTE: this found a flaw in _extract_iloc where we tried to optimize selection with a unified array
        count = 0
        for k, v in tb.element_items():
            count += 1
            v_extract = tb.iloc[k]
            self.assertEqualWithNaN(v, v_extract)
        self.assertEqual(count, tb.size)

    @given(sfst.get_type_blocks())
    def test_reblock_signature(self, tb: TypeBlocks) -> None:
        post = tuple(tb._reblock_signature())
        unique_dtypes = np.unique(tb.dtypes)
        # the reblock signature must be have at least as many entries as types
        self.assertTrue(len(post) >= len(unique_dtypes))
        # sum of column widths is qual to columns in shape
        self.assertTrue(sum(p[1] for p in post), tb.shape[1])


    @given(sfst.get_type_blocks(), sfst.get_type_blocks())
    def test_block_compatible(self, tb1: TypeBlocks, tb2: TypeBlocks) -> None:

        for axis in (None, 0, 1):
            post1 = tb1.block_compatible(tb2, axis)
            post2 = tb2.block_compatible(tb1, axis)
            # either direction gets the same result
            self.assertTrue(post1 == post2)
            # if the shapes are different, they cannot be block compatible
            if axis is None and tb1.shape != tb2.shape:
                self.assertFalse(post1)


    @given(sfst.get_type_blocks(), sfst.get_type_blocks())
    def test_reblock_compatible(self, tb1: TypeBlocks, tb2: TypeBlocks) -> None:

        post1 = tb1.reblock_compatible(tb2)
        post2 = tb2.reblock_compatible(tb1)
        # either direction gets the same result
        self.assertTrue(post1 == post2)
        # if the shapes are different, they cannot be block compatible
        if tb1.shape[1] != tb2.shape[1]:
            self.assertFalse(post1)

    @unittest.skip('pending')
    def test_concatenate_blocks(self) -> None:
        pass

    @given(sfst.get_type_blocks())
    def test_consolidate_blocks(self, tb: TypeBlocks) -> None:

        tb_post = TypeBlocks.from_blocks(tb.consolidate_blocks(tb._blocks))
        self.assertEqual(tb_post.shape, tb.shape)
        self.assertTrue((tb_post.dtypes == tb.dtypes).all())

    @given(sfst.get_type_blocks())
    def test_reblock(self, tb: TypeBlocks) -> None:
        tb_post = TypeBlocks.from_blocks(tb._reblock())
        self.assertEqual(tb_post.shape, tb.shape)
        self.assertTrue((tb_post.dtypes == tb.dtypes).all())

    @given(sfst.get_type_blocks())
    def test_consolidate(self, tb: TypeBlocks) -> None:
        tb_post = tb.consolidate()
        self.assertEqual(tb_post.shape, tb.shape)
        self.assertTrue((tb_post.dtypes == tb.dtypes).all())

    @unittest.skip('pending')
    def test_resize_blocks(self) -> None:
        pass

    @unittest.skip('pending')
    def test_group(self) -> None:
        pass

    @unittest.skip('pending')
    def test_ufunc_axis_skipna(self) -> None:
        pass

    @given(sfst.get_type_blocks())
    def test_display(self, tb: TypeBlocks) -> None:
        post = tb.display()
        self.assertTrue(len(post) > 0)

    @unittest.skip('pending')
    def test_cols_to_slice(self) -> None:
        pass

    @unittest.skip('pending')
    def test_indices_to_contiguous_pairs(self) -> None:
        pass

    @unittest.skip('pending')
    def test_key_to_block_slices(self) -> None:
        pass

    @unittest.skip('pending')
    def test_mask_blocks(self) -> None:
        pass

    @unittest.skip('pending')
    def test_astype_blocks(self) -> None:
        pass

    @unittest.skip('pending')
    def test_shift_blocks(self) -> None:
        pass

    @given(sfst.get_type_blocks())
    def test_assign_blocks_from_keys(self, tb1: TypeBlocks) -> None:

        # assigning a single value from a list of column keys
        for i in range(tb1.shape[1]):
            tb2 = TypeBlocks.from_blocks(tb1._assign_from_iloc_by_unit(
                    column_key=[i], value=300))
            self.assertTrue(tb1.shape == tb2.shape)
            # no more than one type should be changed
            self.assertTrue((tb1.dtypes != tb2.dtypes).sum() <= 1)

        # assigning a single value from a list of row keys
        for i in range(tb1.shape[0]):
            tb3 = TypeBlocks.from_blocks(tb1._assign_from_iloc_by_unit(
                    row_key=[i], value=300))
            self.assertTrue(tb1.shape == tb3.shape)
            self.assertTrue(tb3.iloc[i, 0] == 300)

        # column slices to the end
        for i in range(tb1.shape[1]):
            tb4 = TypeBlocks.from_blocks(tb1._assign_from_iloc_by_unit(
                    column_key=slice(i, None), value=300))
            self.assertTrue(tb1.shape == tb4.shape)
            # we have as many or more blocks
            self.assertTrue(len(tb4.shapes) >= len(tb1.shapes))


    @unittest.skip('pending')
    def test_assign_blocks_from_boolean_blocks(self) -> None:
        pass

    @unittest.skip('pending')
    def test_slice_blocks(self) -> None:
        pass

    @unittest.skip('pending')
    def test_extract_array(self) -> None:
        pass

    @unittest.skip('pending')
    def test_extract(self) -> None:
        pass

    @unittest.skip('pending')
    def test_extract_iloc(self) -> None:
        pass

    @unittest.skip('pending')
    def test_extract_iloc_mask(self) -> None:
        pass

    @unittest.skip('pending')
    def test_extract_iloc_assign(self) -> None:
        pass

    @given(sfst.get_type_blocks(min_rows=1, min_columns=1))
    def test_drop(self, tb: TypeBlocks) -> None:

        for row in range(tb.shape[0]):
            tb_post1 = tb.drop(row)
            self.assertTrue(tb_post1.shape[0] == tb.shape[0] - 1)

        if tb.shape[0] > 2:
            for start in range(1, tb.shape[0]):
                tb_post2 = tb.drop(slice(start, None))
                self.assertTrue(tb_post2.shape[0] == start)

        for col in range(tb.shape[1]):
            tb_post3 = tb.drop((None, col))
            self.assertTrue(tb_post3.shape[1] == tb.shape[1] - 1)

        if tb.shape[1] > 2:
            for start in range(1, tb.shape[1]):
                tb_post4 = tb.drop((None, slice(start, None)))
                self.assertTrue(tb_post4.shape[1] == start)



    @unittest.skip('pending')
    def test_getitem(self) -> None:
        pass

    @unittest.skip('pending')
    def test_ufunc_unary_operator(self) -> None:
        pass

    @unittest.skip('pending')
    def test_block_shape_slices(self) -> None:
        pass

    @unittest.skip('pending')
    def test_ufunc_binary_operator(self) -> None:
        pass

    @unittest.skip('pending')
    def test_transpose(self) -> None:
        pass

    @unittest.skip('pending')
    def test_isna(self) -> None:
        pass

    @unittest.skip('pending')
    def test_notna(self) -> None:
        pass

    @unittest.skip('pending')
    def test_fillna_leading(self) -> None:
        pass

    @unittest.skip('pending')
    def test_fillna_trailing(self) -> None:
        pass

    @unittest.skip('pending')
    def test_fillna_forward(self) -> None:
        pass

    @unittest.skip('pending')
    def test_fillna_backward(self) -> None:
        pass

    @unittest.skip('pending')
    def test_dropna_to_keep_locations(self) -> None:
        pass

    @unittest.skip('pending')
    def test_fillna(self) -> None:
        pass


    @given(sfst.get_type_blocks_aligned_array())
    def test_append(self, tb_aligned_array: tp.Tuple[TypeBlocks, np.ndarray]) -> None:
        tb, aa = tb_aligned_array
        shape_original = tb.shape
        tb.append(aa)
        if aa.ndim == 1:
            self.assertEqual(tb.shape[1], shape_original[1] + 1)
        else:
            self.assertEqual(tb.shape[1], shape_original[1] + aa.shape[1])

    @given(sfst.get_type_blocks_aligned_type_blocks(min_size=2, max_size=2))
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
