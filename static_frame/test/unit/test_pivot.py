from __future__ import annotations

from unittest import mock

import frame_fixtures as ff
import numpy as np

import static_frame.core.pivot as pivot_module
from static_frame.core.frame import Frame
from static_frame.core.index import Index
from static_frame.core.index_hierarchy import IndexHierarchy
from static_frame.core.pivot import (
    _REDUCERS_BINCOUNT,
    BR_MEAN,
    BR_NANMEAN,
    BR_NANSUM,
    BR_SUM,
    pivot_group_reduce_1d,
    pivot_items_to_block,
    pivot_items_to_frame,
    pivot_records_group,
)
from static_frame.test.test_case import TestCase

# from static_frame.core.pivot import pivot_records_items


class TestUnit(TestCase):
    def test_pivot_records_group_single_field(self) -> None:
        # group column 0 (values 0,0,1,1,2), extract data column 1 per group
        f = Frame.from_records(
            [(0, 10), (0, 11), (1, 20), (1, 21), (2, 30)],
            columns=('g', 'a'),
        )
        post = list(pivot_records_group(f._blocks, 0, (1,), 'mergesort'))
        labels = [label for label, _ in post]
        self.assertEqual(labels, [0, 1, 2])
        # each group yields a list with one array (data field 1)
        self.assertEqual([fv[0].tolist() for _, fv in post], [[10, 11], [20, 21], [30]])

    def test_pivot_records_group_multi_field_and_tuple_label(self) -> None:
        # two group columns -> tuple labels; two data columns
        f = Frame.from_records(
            [
                ('x', 0, 10, 1.0),
                ('x', 0, 11, 2.0),
                ('x', 1, 20, 3.0),
                ('y', 0, 30, 4.0),
            ],
            columns=('g0', 'g1', 'a', 'b'),
        )
        post = list(pivot_records_group(f._blocks, [0, 1], (2, 3), 'mergesort'))
        labels = [label for label, _ in post]
        self.assertEqual(labels, [('x', 0), ('x', 1), ('y', 0)])
        # first group has two rows across both data fields
        first_a, first_b = post[0][1]
        self.assertEqual(first_a.tolist(), [10, 11])
        self.assertEqual(first_b.tolist(), [1.0, 2.0])

    def test_pivot_records_group_unsortable_fallback(self) -> None:
        # heterogeneous object group key is not sortable: match-based fallback,
        # preserving first-appearance ordering
        f = Frame.from_records(
            [('x', 1), (2, 2), ('x', 3), (2, 4)],
            columns=('g', 'a'),
        )
        post = list(pivot_records_group(f._blocks, 0, (1,), 'mergesort'))
        labels = [label for label, _ in post]
        self.assertEqual(labels, ['x', 2])
        self.assertEqual([fv[0].tolist() for _, fv in post], [[1, 3], [2, 4]])

    def test_pivot_records_group_empty(self) -> None:
        f = Frame.from_records([(0, 1)], columns=('g', 'a'))
        empty = f._blocks._extract(row_key=slice(0, 0))
        self.assertEqual(list(pivot_records_group(empty, 0, (1,), 'mergesort')), [])

    # ---------------------------------------------------------------------------

    def test_pivot_group_reduce_1d_int_key(self) -> None:
        key = np.array([2, 0, 2, 1, 0])
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        labels, (out,) = pivot_group_reduce_1d(key, (data,), BR_NANSUM, (None,))
        self.assertEqual(labels.tolist(), [0, 1, 2])  # sorted unique keys
        self.assertEqual(out.tolist(), [70.0, 40.0, 40.0])  # 20+50, 40, 10+30
        self.assertFalse(out.flags.writeable)

    def test_pivot_group_reduce_1d_bool_key(self) -> None:
        key = np.array([True, False, True, False])
        data = np.array([1, 2, 3, 4])
        labels, (out,) = pivot_group_reduce_1d(key, (data,), BR_SUM, (None,))
        self.assertEqual(labels.tolist(), [False, True])
        self.assertEqual(out.tolist(), [6, 4])  # False: 2+4, True: 1+3

    def test_pivot_group_reduce_1d_nansum_skips_nan(self) -> None:
        key = np.array([0, 0, 1])
        data = np.array([1.0, np.nan, 5.0])
        _, (out,) = pivot_group_reduce_1d(key, (data,), BR_NANSUM, (None,))
        self.assertEqual(out.tolist(), [1.0, 5.0])  # NaN skipped

    def test_pivot_group_reduce_1d_multi_field(self) -> None:
        key = np.array([1, 0, 1])
        d0 = np.array([10.0, 20.0, 30.0])
        d1 = np.array([1.0, 2.0, 3.0])
        labels, (o0, o1) = pivot_group_reduce_1d(key, (d0, d1), BR_NANSUM, (None, None))
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertEqual(o0.tolist(), [20.0, 40.0])
        self.assertEqual(o1.tolist(), [2.0, 4.0])

    def test_pivot_group_reduce_1d_not_applicable(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        # unsupported key dtype (float) -> None
        self.assertIsNone(
            pivot_group_reduce_1d(np.array([0.5, 1.5, 0.5]), (data,), BR_NANSUM, (None,))
        )
        # key range too sparse for a dense bincount -> None
        self.assertIsNone(
            pivot_group_reduce_1d(
                np.array([0, 10_000_000, 1]), (data,), BR_NANSUM, (None,)
            )
        )
        # non-numeric data column -> None
        self.assertIsNone(
            pivot_group_reduce_1d(
                np.array([0, 1, 0]),
                (np.array(['x', 'y', 'z']),),
                BR_NANSUM,
                (None,),
            )
        )

    def test_pivot_group_reduce_1d_str_key(self) -> None:
        # string keys are factorized to dense codes; labels come back sorted
        key = np.array(['b', 'a', 'b', 'c', 'a'])
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        labels, (out,) = pivot_group_reduce_1d(key, (data,), BR_SUM, (None,))
        self.assertEqual(labels.tolist(), ['a', 'b', 'c'])  # sorted unique keys
        self.assertEqual(out.tolist(), [70.0, 40.0, 40.0])  # a:20+50, b:10+30, c:40
        self.assertFalse(out.flags.writeable)

    def test_pivot_group_reduce_1d_bytes_key_mean(self) -> None:
        # np.bytes_ ('S') keys are supported the same way as unicode
        key = np.array([b'x', b'y', b'x', b'y'])
        data = np.array([2.0, 10.0, 4.0, 20.0])
        labels, (out,) = pivot_group_reduce_1d(key, (data,), BR_MEAN, (None,))
        self.assertEqual(labels.tolist(), [b'x', b'y'])
        self.assertEqual(out.tolist(), [3.0, 15.0])  # x:(2+4)/2, y:(10+20)/2

    def test_pivot_str_key_fast_path_matches_general(self) -> None:
        # the string bincount fast path must be identical to the general grouping
        # path in both values and (sorted) index order
        groups = np.array(['alpha', 'beta', 'gamma', 'delta'])
        idx = np.array([0, 1, 2, 3, 0, 2, 1, 3, 0, 1])
        f = Frame.from_fields(
            (groups[idx], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            columns=('g', 'v'),
        )
        for func in (np.sum, np.mean, np.nansum, np.nanmean):
            fast = f.pivot('g', data_fields='v', func=func)
            # clearing the reducer registry forces the general grouping fallback
            with mock.patch.object(pivot_module, '_REDUCERS_BINCOUNT', {}):
                general = f.pivot('g', data_fields='v', func=func)
            self.assertEqual(fast.index.values.tolist(), general.index.values.tolist())
            self.assertTrue(np.allclose(fast.values, general.values, equal_nan=True))

    def test_pivot_group_reduce_1d_mean(self) -> None:
        key = np.array([0, 0, 1, 1])
        data = np.array([2.0, 4.0, 10.0, 20.0])
        _, (out,) = pivot_group_reduce_1d(key, (data,), BR_MEAN, (None,))
        self.assertEqual(out.tolist(), [3.0, 15.0])  # (2+4)/2, (10+20)/2

    def test_pivot_group_reduce_1d_nanmean(self) -> None:
        key = np.array([0, 0, 0, 1])
        data = np.array([2.0, np.nan, 4.0, 9.0])
        _, (out,) = pivot_group_reduce_1d(key, (data,), BR_NANMEAN, (None,))
        # group 0: mean of [2, 4] (NaN excluded from both sum and count) = 3.0
        self.assertEqual(out.tolist(), [3.0, 9.0])

    def test_pivot_group_reduce_1d_nanmean_all_nan_group(self) -> None:
        key = np.array([0, 0, 1])
        data = np.array([np.nan, np.nan, 5.0])
        _, (out,) = pivot_group_reduce_1d(key, (data,), BR_NANMEAN, (None,))
        self.assertTrue(np.isnan(out[0]))  # all-NaN group -> NaN
        self.assertEqual(out[1], 5.0)

    def test_pivot_builtin_sum_reducer_mapping(self) -> None:
        # the builtin sum routes through the same bincount fast path as np.sum
        self.assertEqual(_REDUCERS_BINCOUNT.get(sum), BR_SUM)

    def test_pivot_builtin_sum_matches_np_sum(self) -> None:
        # user-supplied builtin sum takes the fast path and agrees with np.sum,
        # for both float data (NaN propagates) and integer data
        f = Frame.from_records(
            [(0, 10.0), (1, 20.0), (0, 5.0), (1, 2.0), (0, np.nan)],
            columns=('g', 'v'),
        )
        p_builtin = f.pivot('g', data_fields='v', func=sum)
        p_np = f.pivot('g', data_fields='v', func=np.sum)
        self.assertEqual(p_builtin.index.values.tolist(), p_np.index.values.tolist())
        self.assertTrue(np.array_equal(p_builtin.values, p_np.values, equal_nan=True))
        self.assertEqual(p_builtin.dtypes.values.tolist(), [np.dtype(np.float64)])
        # group 0 contains a NaN -> non-nan-aware sum propagates NaN
        self.assertTrue(np.isnan(p_builtin.loc[0, 'v']))
        self.assertEqual(p_builtin.loc[1, 'v'], 22.0)

        fi = Frame.from_records(
            [(0, 1), (1, 2), (0, 3), (1, 4)],
            columns=('g', 'v'),
        )
        pi = fi.pivot('g', data_fields='v', func=sum)
        self.assertEqual(pi.dtypes.values.tolist(), [np.dtype(np.int64)])
        self.assertEqual(pi.to_pairs(), (('v', ((0, 4), (1, 6))),))

    def test_pivot_group_reduce_1d_int_dtype_cast(self) -> None:
        key = np.array([0, 0, 1])
        data = np.array([1, 2, 3])  # integer data
        _, (out,) = pivot_group_reduce_1d(key, (data,), BR_NANSUM, (np.dtype(np.int64),))
        self.assertEqual(out.dtype, np.dtype(np.int64))  # cast back to int
        self.assertEqual(out.tolist(), [3, 3])

    def test_pivot_items_to_block_a(self) -> None:
        f = ff.parse('s(6,4)|v(int)').assign[0](range(6))
        group_fields_iloc = [0]
        index_outer = Index(f[0].values.tolist())

        post = pivot_items_to_block(
            blocks=f._blocks,
            group_fields_iloc=group_fields_iloc,
            group_depth=1,
            data_field_iloc=3,
            func_single=None,
            dtype=np.dtype(int),
            fill_value=0,
            fill_value_dtype=np.dtype(int),
            index_outer=index_outer,
            kind='mergesort',
        )
        self.assertEqual(post.tolist(), [129017, 35021, 166924, 122246, 197228, 105269])

    def test_pivot_items_to_frame_a(self) -> None:
        f = ff.parse('s(6,4)|v(int)').assign[0](range(6))

        post = pivot_items_to_frame(
            blocks=f._blocks,
            group_fields_iloc=[0],
            group_depth=1,
            data_field_iloc=3,
            func_single=lambda x: str(x) if x % 2 else sum(x),
            frame_cls=Frame,
            name='foo',
            dtype=None,
            index_constructor=Index,
            columns_constructor=Index,
            kind='mergesort',
        )
        self.assertEqual(
            post.to_pairs(),
            (
                (
                    'foo',
                    (
                        (0, '[129017]'),
                        (1, '[35021]'),
                        (2, 166924),
                        (3, 122246),
                        (4, 197228),
                        (5, '[105269]'),
                    ),
                ),
            ),
        )

    def test_pivot_items_to_frame_b(self) -> None:
        f = ff.parse('s(6,4)|v(int)').assign[0](range(6))
        post = pivot_items_to_frame(
            blocks=f._blocks,
            group_fields_iloc=[0, 1],
            group_depth=2,
            data_field_iloc=3,
            func_single=None,
            frame_cls=Frame,
            name='foo',
            dtype=np.dtype(int),
            index_constructor=IndexHierarchy.from_labels,
            columns_constructor=Index,
            kind='mergesort',
        )
        self.assertEqual(
            post.to_pairs(),
            (
                (
                    'foo',
                    (
                        ((0, 162197), 129017),
                        ((1, -41157), 35021),
                        ((2, 5729), 166924),
                        ((3, -168387), 122246),
                        ((4, 140627), 197228),
                        ((5, 66269), 105269),
                    ),
                ),
            ),
        )

    def test_pivot_core_a(self) -> None:
        frame = (
            ff.parse('s(20,4)|v(int)')
            .assign[0]
            .apply(lambda s: s % 4)
            .assign[1]
            .apply(lambda s: s % 3)
        )

        # by default we get a tuple index
        post1 = frame.pivot([0, 1])
        self.assertEqual(post1.index.name, (0, 1))
        self.assertIs(post1.index.__class__, Index)
        self.assertTrue(
            post1.to_pairs(),
            (
                (
                    2,
                    (
                        ((0, 0), 463099),
                        ((0, 1), -88017),
                        ((0, 2), 35021),
                        ((1, 0), 92867),
                        ((1, 2), 96520),
                        ((2, 0), 172133),
                        ((2, 1), 279191),
                        ((2, 2), 13448),
                        ((3, 0), 255338),
                        ((3, 1), 372807),
                        ((3, 2), 155574),
                    ),
                ),
                (
                    3,
                    (
                        ((0, 0), 348362),
                        ((0, 1), 175579),
                        ((0, 2), 105269),
                        ((1, 0), 58768),
                        ((1, 2), 13448),
                        ((2, 0), 84967),
                        ((2, 1), 239151),
                        ((2, 2), 170440),
                        ((3, 0), 269300),
                        ((3, 1), 204528),
                        ((3, 2), 493169),
                    ),
                ),
            ),
        )

        # can provide index constructor
        post2 = frame.pivot([0, 1], index_constructor=IndexHierarchy.from_labels)
        self.assertEqual(post2.index.name, (0, 1))
        self.assertIs(post2.index.__class__, IndexHierarchy)
        self.assertTrue(
            post2.to_pairs(),
            (
                (
                    2,
                    (
                        ((0, 0), 463099),
                        ((0, 1), -88017),
                        ((0, 2), 35021),
                        ((1, 0), 92867),
                        ((1, 2), 96520),
                        ((2, 0), 172133),
                        ((2, 1), 279191),
                        ((2, 2), 13448),
                        ((3, 0), 255338),
                        ((3, 1), 372807),
                        ((3, 2), 155574),
                    ),
                ),
                (
                    3,
                    (
                        ((0, 0), 348362),
                        ((0, 1), 175579),
                        ((0, 2), 105269),
                        ((1, 0), 58768),
                        ((1, 2), 13448),
                        ((2, 0), 84967),
                        ((2, 1), 239151),
                        ((2, 2), 170440),
                        ((3, 0), 269300),
                        ((3, 1), 204528),
                        ((3, 2), 493169),
                    ),
                ),
            ),
        )

    def test_pivot_core_b(self) -> None:
        f1 = Frame.from_records(
            [['a', 10, 1.0, 2.0], ['a', 11, 3.5, 4.5], ['b', 11, 8.2, 7.3]]
        )
        f2 = f1.pivot(index_fields=0, columns_fields=1, fill_value=-1, func=None)
        self.assertEqual(
            f2.to_pairs(),
            (
                ((10, 2), (('a', 1.0), ('b', -1))),
                ((10, 3), (('a', 2.0), ('b', -1))),
                ((11, 2), (('a', 3.5), ('b', 8.2))),
                ((11, 3), (('a', 4.5), ('b', 7.3))),
            ),
        )

    def test_pivot_ih1(self) -> None:
        f1 = Frame.from_records(
            [
                ['coA', '2025-12-31', 'metricA', 12.34],
                ['coA', '2026-01-01', 'metricA', 23.45],
                ['coB', '2026-01-01', 'metricA', 34.56],
                ['coB', '2026-01-01', 'metricB', 45.67],
                ['coC', '2025-12-31', 'metricC', 56.78],
                ['coC', '2025-12-31', 'metricD', np.nan],
                ['coD', '2025-12-30', 'metricC', 67.89],
                ['coD', '2025-12-31', 'metricC', 78.90],
            ],
            dtypes=[str, 'datetime64[D]', str, float],
            columns=['company', 'date', 'metric', 'value'],
        )
        f2 = f1.pivot(
            index_fields=['company', 'date'],
            columns_fields='metric',
            data_fields='value',
            func=None,
        )
        self.assertIs(f2.index.__class__, IndexHierarchy)
        self.assertEqual(f2.shape, (6, 4))

    def test_pivot_ih2(self) -> None:
        f1 = Frame.from_records(
            [
                ['coA', 'x', 'metricA', 12.34],
                ['coA', 'y', 'metricA', 23.45],
                ['coB', 'y', 'metricA', 34.56],
                ['coB', 'y', 'metricB', 45.67],
                ['coC', 'x', 'metricC', 56.78],
                ['coC', 'x', 'metricD', np.nan],
                ['coD', 'z', 'metricC', 67.89],
                ['coD', 'x', 'metricC', 78.90],
            ],
            dtypes=[str, str, str, float],
            columns=['company', 'attr', 'metric', 'value'],
        )
        f2 = f1.pivot(
            index_fields=['company', 'attr'],
            columns_fields='metric',
            data_fields='value',
            func=None,
        )
        self.assertIs(f2.index.__class__, IndexHierarchy)
        self.assertEqual(f2.shape, (6, 4))


if __name__ == '__main__':
    import unittest

    unittest.main()
