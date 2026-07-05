from __future__ import annotations

import numpy as np

from static_frame.core.rank import RankMethod, rank_1d, rank_2d
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):
    def test_rank_method_a(self) -> None:
        with self.assertRaises(NotImplementedError):
            rank_1d(np.array([3, 2, 6, 20]), None)  # type: ignore

    def test_rank_method_b(self) -> None:
        post = rank_1d(np.array([]), 'ordinal')  # type: ignore
        self.assertEqual(len(post), 0)
        self.assertEqual(post.dtype, np.int64)

    def test_rank_method_c(self) -> None:
        post = rank_1d(np.array([]), 'mean')  # type: ignore
        self.assertEqual(len(post), 0)
        self.assertEqual(post.dtype, np.float64)

    def test_rank_ordinal_a(self) -> None:
        self.assertEqual(
            rank_1d(np.array([3, 2, 6, 20]), RankMethod.ORDINAL).tolist(), [1, 0, 2, 3]
        )

        self.assertEqual(
            rank_1d(np.array([3, 2, 6, 20]), RankMethod.ORDINAL, start=1).tolist(),
            [2, 1, 3, 4],
        )

        a2 = rank_1d(np.array([3, 2, 6, 20]), RankMethod.ORDINAL, ascending=False)
        self.assertEqual(a2.tolist(), [2, 3, 1, 0])

    def test_rank_ordinal_b(self) -> None:
        a1 = rank_1d(np.array([0, 2, 3, 2]), 'ordinal', start=1)
        self.assertEqual(a1.tolist(), [1, 2, 4, 3])

        a2 = rank_1d(np.array([0, 2, 3, 2]), 'ordinal', start=0)
        self.assertEqual(a2.tolist(), [0, 1, 3, 2])

        a3 = rank_1d(np.array([0, 2, 3, 2]), 'ordinal', ascending=False)
        self.assertEqual(a3.tolist(), [3, 2, 0, 1])

    def test_rank_ordinal_c(self) -> None:
        a1 = rank_1d(np.array([8, 15, 7, 2, 20, 4, 20, 7, 15, 15]), 'ordinal', start=1)
        self.assertEqual(a1.tolist(), [5, 6, 3, 1, 9, 2, 10, 4, 7, 8])
        # scipy: [5, 6, 3, 1, 9, 2, 10, 4, 7, 8]

    def test_rank_ordinal_d(self) -> None:
        a1 = rank_1d(
            np.array([17, 10, 3, -4, 9, 3, -12, 18, 3, 0, 17]), 'ordinal', start=1
        )
        self.assertEqual(a1.tolist(), [9, 8, 4, 2, 7, 5, 1, 11, 6, 3, 10])
        # scipy: [9, 8, 4, 2, 7, 5, 1, 11, 6, 3, 10]

    def test_rank_average_a(self) -> None:
        a1 = rank_1d(np.array([0, 2, 3, 2]), 'mean', ascending=True)
        self.assertEqual(a1.tolist(), [0.0, 1.5, 3.0, 1.5])

        a2 = rank_1d(np.array([0, 2, 3, 2]), 'mean', start=1)
        self.assertEqual(a2.tolist(), [1.0, 2.5, 4.0, 2.5])

        a3 = rank_1d(np.array([0, 2, 3, 2]), 'mean', start=1, ascending=False)
        self.assertEqual(a3.tolist(), [4.0, 2.5, 1.0, 2.5])

    def test_rank_average_b(self) -> None:
        a1 = rank_1d(np.array([0, 2, 5, 2, 2, 2]), 'mean', ascending=True)
        self.assertEqual(a1.tolist(), [0.0, 2.5, 5.0, 2.5, 2.5, 2.5])

        a1 = rank_1d(np.array([0, 2, 5, 2, 2, 2]), 'mean', ascending=True, start=1)
        self.assertEqual(a1.tolist(), [1.0, 3.5, 6.0, 3.5, 3.5, 3.5])
        # scipy: [1.0, 3.5, 6.0, 3.5, 3.5, 3.5]

        a2 = rank_1d(np.array([0, 2, 3, 2]), 'mean', start=1)
        self.assertEqual(a2.tolist(), [1.0, 2.5, 4.0, 2.5])

        a3 = rank_1d(np.array([0, 2, 3, 2]), 'mean', start=1, ascending=False)
        self.assertEqual(a3.tolist(), [4.0, 2.5, 1.0, 2.5])

    def test_rank_average_c(self) -> None:
        a1 = rank_1d(np.array([8, 15, 7, 2, 20, 4, 20, 7, 15, 15]), 'mean', start=1)
        self.assertEqual(a1.tolist(), [5.0, 7.0, 3.5, 1.0, 9.5, 2.0, 9.5, 3.5, 7.0, 7.0])
        # scipy: [5.0, 7.0, 3.5, 1.0, 9.5, 2.0, 9.5, 3.5, 7.0, 7.0]

    def test_rank_average_d(self) -> None:
        a1 = rank_1d(np.array([17, 10, 3, -4, 9, 3, -12, 18, 3, 0, 17]), 'mean', start=1)
        self.assertEqual(
            a1.tolist(), [9.5, 8.0, 5.0, 2.0, 7.0, 5.0, 1.0, 11.0, 5.0, 3.0, 9.5]
        )
        # scipy: [9.5, 8.0, 5.0, 2.0, 7.0, 5.0, 1.0, 11.0, 5.0, 3.0, 9.5

    def test_rank_min_a(self) -> None:
        a1 = rank_1d(np.array([0, 2, 3, 2]), 'min', start=1)
        self.assertEqual(a1.tolist(), [1, 2, 4, 2])
        a2 = rank_1d(np.array([0, 2, 3, 2]), 'min', start=0)
        self.assertEqual(a2.tolist(), [0, 1, 3, 1])

        a3 = rank_1d(np.array([0, 2, 3, 2]), 'min', ascending=False)
        self.assertEqual(a3.tolist(), [3, 1, 0, 1])

    def test_rank_min_b(self) -> None:
        a1 = rank_1d(np.array([8, 15, 7, 2, 20, 4, 20, 7, 15, 15]), 'min', start=1)
        self.assertEqual(a1.tolist(), [5, 6, 3, 1, 9, 2, 9, 3, 6, 6])
        # scipy: [5, 6, 3, 1, 9, 2, 9, 3, 6, 6]

    def test_rank_min_c(self) -> None:
        a1 = rank_1d(np.array([17, 10, 3, -4, 9, 3, -12, 18, 3, 0, 17]), 'min', start=1)
        self.assertEqual(a1.tolist(), [9, 8, 4, 2, 7, 4, 1, 11, 4, 3, 9])

        # scipy: [9, 8, 4, 2, 7, 4, 1, 11, 4, 3, 9]

    def test_rank_max_a(self) -> None:
        a1 = rank_1d(np.array([0, 2, 3, 2]), 'max', start=1)

        self.assertEqual(a1.tolist(), [1, 3, 4, 3])
        a2 = rank_1d(np.array([0, 2, 3, 2]), 'max', start=0)
        self.assertEqual(a2.tolist(), [0, 2, 3, 2])

        a2 = rank_1d(np.array([0, 2, 3, 2]), 'max', ascending=False)
        self.assertEqual(a2.tolist(), [3, 2, 0, 2])

    def test_rank_max_b(self) -> None:
        a1 = rank_1d(np.array([8, 15, 7, 2, 20, 4, 20, 7, 15, 15]), 'max', start=1)
        self.assertEqual(a1.tolist(), [5, 8, 4, 1, 10, 2, 10, 4, 8, 8])
        # scipy: [5, 8, 4, 1, 10, 2, 10, 4, 8, 8]

    def test_rank_max_c(self) -> None:
        a1 = rank_1d(np.array([17, 10, 3, -4, 9, 3, -12, 18, 3, 0, 17]), 'max', start=1)
        self.assertEqual(a1.tolist(), [10, 8, 6, 2, 7, 6, 1, 11, 6, 3, 10])
        # scipy: [10, 8, 6, 2, 7, 6, 1, 11, 6, 3, 10]

    def test_rank_dense_a(self) -> None:
        a1 = rank_1d(np.array([0, 2, 3, 2]), 'dense', start=1)
        self.assertEqual(a1.tolist(), [1, 2, 3, 2])

        a2 = rank_1d(np.array([0, 2, 3, 2]), 'dense', start=0)
        self.assertEqual(a2.tolist(), [0, 1, 2, 1])

        a3 = rank_1d(np.array([0, 2, 3, 2]), 'dense', ascending=False)
        self.assertEqual(a3.tolist(), [2, 1, 0, 1])

    def test_rank_dense_b(self) -> None:
        a1 = rank_1d(np.array([8, 15, 7, 2, 20, 4, 20, 7, 15, 15]), 'dense', start=1)
        self.assertEqual(a1.tolist(), [4, 5, 3, 1, 6, 2, 6, 3, 5, 5])
        # scipy: [4, 5, 3, 1, 6, 2, 6, 3, 5, 5]

    def test_rank_dense_c(self) -> None:
        a1 = rank_1d(np.array([17, 10, 3, -4, 9, 3, -12, 18, 3, 0, 17]), 'dense', start=1)
        self.assertEqual(a1.tolist(), [7, 6, 4, 2, 5, 4, 1, 8, 4, 3, 7])
        # scipy: [7, 6, 4, 2, 5, 4, 1, 8, 4, 3, 7]

    def test_rank_2d_a(self) -> None:
        a1 = np.array([10, 3, -4, 9, 3, -12, 18, 3, 0, 17]).reshape(5, 2)
        self.assertEqual(
            rank_2d(a1, axis=0, method='ordinal', start=1).tolist(),
            [[4, 2], [1, 4], [3, 1], [5, 3], [2, 5]],
        )
        self.assertEqual(
            rank_2d(a1, axis=1, method='ordinal', start=1).tolist(),
            [[2, 1], [1, 2], [2, 1], [2, 1], [1, 2]],
        )

    def test_rank_2d_b(self) -> None:
        a1 = np.array([10, 3, -4, 9, 3, -12, 18, 3, 0, 17]).reshape(5, 2)
        self.assertEqual(
            rank_2d(a1, axis=0, method='mean', start=1).tolist(),
            [[4.0, 2.5], [1.0, 4.0], [3.0, 1.0], [5.0, 2.5], [2.0, 5.0]],
        )
        self.assertEqual(
            rank_2d(a1, axis=1, method='mean', start=1).tolist(),
            [[2.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 1.0], [1.0, 2.0]],
        )

    def test_rank_2d_c(self) -> None:
        a1 = np.array([10, 3, -4, 9, 3, -12, 18, 3, 0, 17]).reshape(5, 2)
        self.assertEqual(
            rank_2d(a1, axis=0, method='min', start=1).tolist(),
            [[4, 2], [1, 4], [3, 1], [5, 2], [2, 5]],
        )
        self.assertEqual(
            rank_2d(a1, axis=1, method='min', start=1).tolist(),
            [[2, 1], [1, 2], [2, 1], [2, 1], [1, 2]],
        )

    def test_rank_2d_d(self) -> None:
        a1 = np.array([10, 3, -4, 9, 3, -12, 18, 3, 0, 17]).reshape(5, 2)
        self.assertEqual(
            rank_2d(a1, axis=0, method='max', start=1).tolist(),
            [[4, 3], [1, 4], [3, 1], [5, 3], [2, 5]],
        )
        self.assertEqual(
            rank_2d(a1, axis=1, method='max', start=1).tolist(),
            [[2, 1], [1, 2], [2, 1], [2, 1], [1, 2]],
        )

    def test_rank_2d_e(self) -> None:
        a1 = np.array([10, 3, -4, 9, 3, -12, 18, 3, 0, 17]).reshape(5, 2)
        self.assertEqual(
            rank_2d(a1, axis=0, method='dense', start=1).tolist(),
            [[4, 2], [1, 3], [3, 1], [5, 2], [2, 4]],
        )
        self.assertEqual(
            rank_2d(a1, axis=1, method='dense', start=1).tolist(),
            [[2, 1], [1, 2], [2, 1], [2, 1], [1, 2]],
        )

    # ---------------------------------------------------------------------------

    def test_rank_1d_pair_a(self) -> None:
        self.assertEqual(rank_1d(np.array([0, 0]), 'mean').tolist(), [0.5, 0.5])

        self.assertEqual(rank_1d(np.array([0, 0]), 'min').tolist(), [0, 0])

        self.assertEqual(rank_1d(np.array([0, 0]), 'max').tolist(), [1, 1])

        self.assertEqual(rank_1d(np.array([0, 0]), 'dense').tolist(), [0, 0])

        self.assertEqual(rank_1d(np.array([0, 0]), 'ordinal').tolist(), [0, 1])

    def test_rank_1d_pair_b(self) -> None:
        self.assertEqual(rank_1d(np.array([0, 0, 1]), 'mean').tolist(), [0.5, 0.5, 2])
        self.assertEqual(rank_1d(np.array([0, 0, 1]), 'mean').dtype.kind, 'f')

        self.assertEqual(rank_1d(np.array([0, 0, 1]), 'min').tolist(), [0, 0, 2])
        self.assertEqual(rank_1d(np.array([0, 0, 1]), 'min').dtype.kind, 'i')

        self.assertEqual(rank_1d(np.array([0, 0, 1]), 'max').tolist(), [1, 1, 2])
        self.assertEqual(rank_1d(np.array([0, 0, 1]), 'max').dtype.kind, 'i')

        self.assertEqual(rank_1d(np.array([0, 0, 1]), 'dense').tolist(), [0, 0, 1])
        self.assertEqual(rank_1d(np.array([0, 0, 1]), 'dense').dtype.kind, 'i')

        self.assertEqual(rank_1d(np.array([0, 0, 1]), 'ordinal').tolist(), [0, 1, 2])
        self.assertEqual(rank_1d(np.array([0, 0, 1]), 'ordinal').dtype.kind, 'i')

    def test_rank_1d_factorize_str(self) -> None:
        # string keys go through the hash-factorize fast path
        a1 = np.array(['b', 'a', 'b', 'c', 'a'])
        self.assertEqual(rank_1d(a1, 'dense').tolist(), [1, 0, 1, 2, 0])
        self.assertEqual(rank_1d(a1, 'min').tolist(), [2, 0, 2, 4, 0])
        self.assertEqual(rank_1d(a1, 'max').tolist(), [3, 1, 3, 4, 1])
        self.assertEqual(rank_1d(a1, 'mean').tolist(), [2.5, 0.5, 2.5, 4.0, 0.5])
        self.assertEqual(rank_1d(a1, 'ordinal').tolist(), [2, 0, 3, 4, 1])

    def test_rank_1d_factorize_bool(self) -> None:
        a1 = np.array([True, False, True, False])
        self.assertEqual(rank_1d(a1, 'dense').tolist(), [1, 0, 1, 0])
        self.assertEqual(rank_1d(a1, 'min').tolist(), [2, 0, 2, 0])

    def test_rank_1d_factorize_matches_reference(self) -> None:
        # the NaN-free fast path must be byte-identical to the argsort reference
        # across every method / direction / start
        def argsort_rank(
            arr: np.ndarray, m: RankMethod, ascending: bool, start: int
        ) -> np.ndarray:
            idx = np.argsort(arr, kind='stable')
            size = len(arr)
            ordv = np.empty(size, dtype=np.int64)
            ordv[idx] = np.arange(size)
            if m == RankMethod.ORDINAL:
                r = ordv
                rmax: object = size - 1
            else:
                asorted = arr[idx]
                uniq = np.full(size, True)
                uniq[1:] = asorted[1:] != asorted[:-1]
                dense = uniq.cumsum()[ordv]
                if m == RankMethod.DENSE:
                    r = dense - 1
                else:
                    up = np.flatnonzero(uniq)
                    cnt = np.empty(len(up) + 1, dtype=np.int64)
                    cnt[:-1] = up
                    cnt[-1] = size
                    # under descending, min/max selection swaps before the flip
                    if (m == RankMethod.MAX and ascending) or (
                        m == RankMethod.MIN and not ascending
                    ):
                        r = cnt[dense] - 1
                    elif (m == RankMethod.MIN and ascending) or (
                        m == RankMethod.MAX and not ascending
                    ):
                        r = cnt[dense - 1]
                    else:
                        r = 0.5 * ((cnt[dense] - 1) + cnt[dense - 1])
                rmax = r.max()
            if not ascending:
                r = rmax - r
            if start != 0:
                r = r + start
            return r

        rng = np.random.default_rng(3)
        cases = (
            rng.integers(0, 6, 50),
            np.array(list('abcde'))[rng.integers(0, 5, 50)],
            rng.integers(0, 2, 50).astype(bool),
            np.round(rng.random(50), 1),  # float, no NaN
        )
        for arr in cases:
            for m in RankMethod:
                for ascending in (True, False):
                    for start in (0, 1):
                        post = rank_1d(arr, m, ascending, start)
                        ref = argsort_rank(arr, m, ascending, start)
                        self.assertEqual(post.tolist(), ref.tolist())
                        self.assertEqual(post.dtype, ref.dtype)
                        self.assertFalse(post.flags.writeable)

    def test_rank_1d_float_nan_fallback(self) -> None:
        # float with NaN must stay on the sort path (each NaN distinct)
        a1 = np.array([2.0, np.nan, 1.0, np.nan])
        # NaNs sort last and each gets a distinct ordinal position
        self.assertEqual(rank_1d(a1, 'ordinal').tolist(), [1, 2, 0, 3])
        # min rank keeps NaNs distinct (not collapsed into one group)
        self.assertEqual(rank_1d(a1, 'min').tolist(), [1, 2, 0, 3])


if __name__ == '__main__':
    import unittest

    unittest.main()
