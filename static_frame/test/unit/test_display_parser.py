from __future__ import annotations

import unittest

import numpy as np

from static_frame import (
    Frame,
    FrameGO,
    Index,
    IndexDate,
    IndexHierarchy,
    IndexSecond,
    IndexYear,
    Series,
)
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):
    # ---------------------------------------------------------------------------
    # Frame.from_display tests

    def test_frame_from_display_a1(self) -> None:
        """Basic Frame with string index."""
        f1 = Frame.from_records(
            (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
            ),
            columns=('p', 'q', 'r', 's', 't'),
            index=('w', 'x', 'y'),
        )
        f2 = Frame.from_display(repr(f1))
        self.assertTrue(
            f1.equals(f2, compare_class=True, compare_dtype=True, compare_name=True)
        )

    def test_frame_from_display_a2(self) -> None:
        """Basic Frame with string index."""
        msg = """


<Frame>
<Index> p       q       r     s      t      <<U1>
<Index>
w       2       2       a     False  False
x       30      34      b     True   False
y       2       95      c     False  False
<<U1>   <int64> <int64> <<U1> <bool> <bool>

"""

        f = Frame.from_display(msg)
        self.assertEqual(
            f.to_pairs(),
            (
                (
                    np.str_('p'),
                    (
                        (np.str_('w'), np.int64(2)),
                        (np.str_('x'), np.int64(30)),
                        (np.str_('y'), np.int64(2)),
                    ),
                ),
                (
                    np.str_('q'),
                    (
                        (np.str_('w'), np.int64(2)),
                        (np.str_('x'), np.int64(34)),
                        (np.str_('y'), np.int64(95)),
                    ),
                ),
                (
                    np.str_('r'),
                    (
                        (np.str_('w'), np.str_('a')),
                        (np.str_('x'), np.str_('b')),
                        (np.str_('y'), np.str_('c')),
                    ),
                ),
                (
                    np.str_('s'),
                    (
                        (np.str_('w'), np.False_),
                        (np.str_('x'), np.True_),
                        (np.str_('y'), np.False_),
                    ),
                ),
                (
                    np.str_('t'),
                    (
                        (np.str_('w'), np.False_),
                        (np.str_('x'), np.False_),
                        (np.str_('y'), np.False_),
                    ),
                ),
            ),
        )

    def test_frame_from_display_b(self) -> None:
        """Frame with integer (auto) index."""
        f1 = Frame.from_records(
            ((1, 2.5), (3, 4.5)),
            columns=('x', 'y'),
        )
        f2 = Frame.from_display(repr(f1))
        self.assertTrue(
            f1.equals(f2, compare_class=True, compare_dtype=True, compare_name=True)
        )

    def test_frame_from_display_c(self) -> None:
        """Named Frame."""
        f1 = Frame.from_records(
            ((1, 2), (3, 4)),
            columns=('a', 'b'),
            name='myframe',
        )
        f2 = Frame.from_display(repr(f1))
        self.assertTrue(
            f1.equals(f2, compare_class=True, compare_dtype=True, compare_name=True)
        )
        self.assertEqual(f2.name, 'myframe')

    def test_frame_from_display_d(self) -> None:
        """Frame with various dtypes."""
        f1 = Frame.from_records(
            ((1, 1.1, True, 'a'), (2, 2.2, False, 'bb'), (3, 3.3, True, 'ccc')),
            columns=('i', 'f', 'b', 's'),
        )
        f2 = Frame.from_display(repr(f1))
        self.assertTrue(
            f1.equals(f2, compare_class=True, compare_dtype=True, compare_name=True)
        )

    def test_frame_from_display_e(self) -> None:
        """Frame with IndexHierarchy (depth-2) row index."""
        f1 = Frame.from_records(
            ((2, 2, 'a'), (30, 34, 'b'), (2, 95, 'c'), (300, -4, 'x')),
            columns=('p', 'q', 'r'),
            index=IndexHierarchy.from_labels((('A', 1), ('A', 2), ('B', 1), ('B', 2))),
        )
        f2 = Frame.from_display(repr(f1))
        self.assertTrue(
            f1.equals(f2, compare_class=True, compare_dtype=True, compare_name=True)
        )

    def test_frame_from_display_f(self) -> None:
        """Frame with IndexHierarchy (depth-3) row index."""
        f1 = Frame.from_records(
            ((1, 2), (3, 4), (5, 6), (7, 8)),
            columns=('a', 'b'),
            index=IndexHierarchy.from_labels(
                (('X', 'A', 1), ('X', 'A', 2), ('X', 'B', 1), ('Y', 'A', 1))
            ),
        )
        f2 = Frame.from_display(repr(f1))
        self.assertTrue(
            f1.equals(f2, compare_class=True, compare_dtype=True, compare_name=True)
        )

    def test_frame_from_display_g(self) -> None:
        """Frame with IndexHierarchy column index."""
        f1 = Frame.from_records(
            ((1, 2, 3), (4, 5, 6)),
            columns=IndexHierarchy.from_labels((('a', 1), ('b', 2), ('c', 3))),
            index=('x', 'y'),
        )
        f2 = Frame.from_display(repr(f1))
        self.assertTrue(
            f1.equals(f2, compare_class=True, compare_dtype=True, compare_name=True)
        )

    def test_frame_from_display_h(self) -> None:
        """Frame with integer column labels."""
        f1 = Frame.from_records(
            ((1, 2), (3, 4)),
            columns=(10, 20),
        )
        f2 = Frame.from_display(repr(f1))
        self.assertTrue(
            f1.equals(f2, compare_class=True, compare_dtype=True, compare_name=True)
        )
        self.assertEqual(f2.columns.values.tolist(), [10, 20])

    def test_frame_from_display_i(self) -> None:
        """Frame with named Index."""
        idx = Index(range(3), name='myidx')
        f1 = Frame.from_records(
            [(1, 2), (3, 4), (5, 6)],
            columns=('a', 'b'),
            index=idx,
        )
        f2 = Frame.from_display(repr(f1))
        self.assertTrue(
            f1.equals(f2, compare_class=True, compare_dtype=True, compare_name=True)
        )
        self.assertEqual(f2.index.name, 'myidx')

    def test_frame_from_display_j(self) -> None:
        """Frame with named IndexHierarchy row index."""
        ih = IndexHierarchy.from_labels([('A', 1), ('B', 2)], name='myhier')
        f1 = Frame.from_records(
            [(1, 2), (3, 4)],
            columns=('a', 'b'),
            index=ih,
            own_index=True,
        )
        f2 = Frame.from_display(repr(f1))
        self.assertTrue(
            f1.equals(f2, compare_class=True, compare_dtype=True, compare_name=True)
        )
        self.assertEqual(f2.index.name, 'myhier')

    def test_frame_from_display_k1(self) -> None:
        """Frame with NaN float values."""
        f1 = Frame.from_dict(
            {
                'a': [1.0, np.nan, 3.0],
                'b': [np.nan, 2.0, np.nan],
            }
        )
        f2 = Frame.from_display(repr(f1))
        self.assertTrue(
            f1.equals(f2, compare_class=True, compare_dtype=True, compare_name=True)
        )

    def test_frame_from_display_k2(self) -> None:
        """Frame with NaN float values."""
        msg = """
<Frame>
<Index> a         b         <<U1>
<Index>
0       1.0       nan
1       nan       2.0
2       3.0       nan
<int64> <float64> <float64>
"""
        f1 = Frame.from_dict(
            {
                'a': [1.0, np.nan, 3.0],
                'b': [np.nan, 2.0, np.nan],
            }
        )
        f = Frame.from_display(msg)
        self.assertEqual(
            f.fillna(-1).to_pairs(),
            (
                (
                    np.str_('a'),
                    (
                        (np.int64(0), np.float64(1.0)),
                        (np.int64(1), np.float64(-1.0)),
                        (np.int64(2), np.float64(3.0)),
                    ),
                ),
                (
                    np.str_('b'),
                    (
                        (np.int64(0), np.float64(-1.0)),
                        (np.int64(1), np.float64(2.0)),
                        (np.int64(2), np.float64(-1.0)),
                    ),
                ),
            ),
        )

    def test_frame_from_display_l(self) -> None:
        """Frame with string values containing internal spaces."""
        f1 = Frame.from_records(
            [('hello world', 1), ('foo bar', 2)],
            columns=('text', 'num'),
        )
        f2 = Frame.from_display(repr(f1))
        self.assertTrue(
            f1.equals(f2, compare_class=True, compare_dtype=True, compare_name=True)
        )
        self.assertEqual(f2['text'].values.tolist(), ['hello world', 'foo bar'])

    def test_frame_from_display_m(self) -> None:
        """Frame with DateIndex."""
        f1 = Frame.from_records(
            [(1, 2.0), (3, 4.0)],
            columns=('a', 'b'),
            index=IndexDate(('2021-01-01', '2021-01-02')),
        )
        f2 = Frame.from_display(repr(f1))
        self.assertTrue(
            f1.equals(f2, compare_class=True, compare_dtype=True, compare_name=True)
        )
        self.assertIsInstance(f2.index, IndexDate)

    def test_frame_from_display_n(self) -> None:
        """Manually ANSI-escaped repr is still parsed correctly."""
        f1 = Frame.from_records(
            ((1, 2), (3, 4)),
            columns=('a', 'b'),
        )
        # Inject ANSI escape codes the same way the terminal renderer does
        plain = repr(f1)
        ansi_repr = '\x1b[38;5;243m<Frame>\x1b[0m\n' + '\n'.join(plain.split('\n')[1:])
        f2 = Frame.from_display(ansi_repr)
        self.assertTrue(
            f1.equals(f2, compare_class=True, compare_dtype=True, compare_name=True)
        )

    def test_frame_from_display_o(self) -> None:
        """FrameGO.from_display returns a FrameGO."""
        f1 = Frame.from_records(
            ((1, 2), (3, 4)),
            columns=('a', 'b'),
        )
        f2 = FrameGO.from_display(repr(f1))
        self.assertIsInstance(f2, FrameGO)
        self.assertEqual(f2.to_frame().equals(f1), True)

    # ---------------------------------------------------------------------------
    # Series.from_display tests

    def test_series_from_display_a(self) -> None:
        """Series with string index and integer values."""
        s1 = Series([1, 2, 3], index=['a', 'b', 'c'], name='test')
        s2 = Series.from_display(repr(s1))
        self.assertTrue(s1.equals(s2))

    def test_series_from_display_b(self) -> None:
        """Unnamed Series."""
        s1 = Series([1, 2, 3], index=['a', 'b', 'c'])
        s2 = Series.from_display(repr(s1))
        self.assertTrue(
            s1.equals(s2, compare_class=True, compare_dtype=True, compare_name=True)
        )
        self.assertIsNone(s2.name)

    def test_series_from_display_c(self) -> None:
        """Series with float values including NaN."""
        s1 = Series([1.0, np.nan, 3.0], index=['a', 'b', 'c'], name='f')
        s2 = Series.from_display(repr(s1))
        self.assertTrue(
            s1.equals(s2, compare_class=True, compare_dtype=True, compare_name=True)
        )

    def test_series_from_display_d(self) -> None:
        """Series with boolean values."""
        s1 = Series([True, False, True], index=['a', 'b', 'c'])
        s2 = Series.from_display(repr(s1))
        self.assertTrue(
            s1.equals(s2, compare_class=True, compare_dtype=True, compare_name=True)
        )

    def test_series_from_display_e(self) -> None:
        """Series with IndexHierarchy (depth-2) index."""
        s1 = Series(
            [1, 2, 3, 4],
            index=IndexHierarchy.from_labels([('A', 1), ('A', 2), ('B', 1), ('B', 2)]),
            name='test',
        )
        s2 = Series.from_display(repr(s1))
        self.assertTrue(
            s1.equals(s2, compare_class=True, compare_dtype=True, compare_name=True)
        )

    def test_series_from_display_f(self) -> None:
        """Series with IndexDate."""
        s1 = Series([1.0, 2.0], index=IndexDate(('2021-01-01', '2021-01-02')))
        s2 = Series.from_display(repr(s1))
        self.assertTrue(
            s1.equals(s2, compare_class=True, compare_dtype=True, compare_name=True)
        )
        self.assertIsInstance(s2.index, IndexDate)

    def test_series_from_display_g(self) -> None:
        """Series with IndexYear."""
        s1 = Series([100, 200], index=IndexYear(('2020', '2021')), name='revenue')
        s2 = Series.from_display(repr(s1))
        self.assertTrue(
            s1.equals(s2, compare_class=True, compare_dtype=True, compare_name=True)
        )
        self.assertIsInstance(s2.index, IndexYear)

    def test_series_from_display_h(self) -> None:
        """Series with named index."""
        s1 = Series([1, 2, 3], index=Index(['x', 'y', 'z'], name='letters'))
        s2 = Series.from_display(repr(s1))
        self.assertTrue(
            s1.equals(s2, compare_class=True, compare_dtype=True, compare_name=True)
        )
        self.assertEqual(s2.index.name, 'letters')

    def test_series_from_display_i(self) -> None:
        """Manually ANSI-escaped repr is still parsed correctly."""
        s1 = Series([1, 2, 3], index=['a', 'b', 'c'], name='test')
        plain = repr(s1)
        # Inject ANSI escape codes the same way the terminal renderer does
        ansi_repr = '\x1b[38;5;243m<Series: test>\x1b[0m\n' + '\n'.join(
            plain.split('\n')[1:]
        )
        s2 = Series.from_display(ansi_repr)
        self.assertTrue(
            s1.equals(s2, compare_class=True, compare_dtype=True, compare_name=True)
        )


if __name__ == '__main__':
    unittest.main()
