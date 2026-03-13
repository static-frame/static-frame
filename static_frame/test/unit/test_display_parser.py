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
from static_frame.core.display_parser import (
    build_columns,
    display_parse_frame,
    display_parse_series,
    extract_cell,
    extract_column_header_data,
    find_dtype_positions,
    find_index_depth,
    find_standalone_index_line,
    make_array,
    parse_header_line,
)
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):
    def test_extract_cell(self) -> None:
        self.assertEqual(extract_cell('', 10, 12), '')

    def test_make_array(self) -> None:
        a1 = make_array(
            ['None', 'True', 'False', '1.2', '80', 'foo', 'bar'], dtype=np.dtype(object)
        )
        self.assertEqual(a1.tolist(), [None, True, False, 1.2, 80, 'foo', 'bar'])

    def test_parse_header_line(self) -> None:
        with self.assertRaises(ValueError):
            parse_header_line('>M<')

    def test_find_standalone_index_line(self) -> None:
        with self.assertRaises(ValueError):
            find_standalone_index_line(['', '', ''])

    def test_find_dytpe_positions(self) -> None:
        x = find_dtype_positions('<<U1>   <int64> <int64> <<U1> <bool> <bool>')
        self.assertEqual(
            x,
            [
                (0, '<<U1>'),
                (8, '<int64>'),
                (16, '<int64>'),
                (24, '<<U1>'),
                (30, '<bool>'),
                (37, '<bool>'),
            ],
        )

    def test_find_index_depth_a(self) -> None:
        # find_index_depth()
        ch = '<Index> p       q       r     s      t      <<U1>'
        dp = [
            (0, '<<U1>'),
            (8, '<int64>'),
            (16, '<int64>'),
            (24, '<<U1>'),
            (30, '<bool>'),
            (37, '<bool>'),
        ]
        self.assertEqual(find_index_depth(ch, dp), 1)

    def test_find_index_depth_b(self) -> None:
        # <Frame>
        # <Index>                  x       y       <<U1>
        # <IndexHierarchy>
        # a                1       0       0
        # a                2       0       0
        # b                1       0       0
        # b                2       0       0
        # <<U1>            <int64> <int64> <int64>

        ch = '<Index>                  x       y       <<U1>'
        dp = find_dtype_positions('<<U1>            <int64> <int64> <int64>')
        self.assertEqual(find_index_depth(ch, dp), 2)

    def test_find_index_depth_c(self) -> None:
        # <Frame>
        # <Index>                  <<U1>
        # <IndexHierarchy>
        # a                1
        # a                2
        # b                1
        # b                2
        # <<U1>            <int64>

        ch = '<Index>                  <<U1>'
        dp = find_dtype_positions('<<U1>            <int64>')
        self.assertEqual(find_index_depth(ch, dp), 2)

    def test_extract_column_header_data_a(self) -> None:
        # <Frame>
        # <Index>                  x       y       <<U1>
        # <IndexHierarchy>
        # a                1       0       0
        # a                2       0       0
        # b                1       0       0
        # b                2       0       0
        # <<U1>            <int64> <int64> <int64>

        ch = '<Index>                  x       y       <<U1>'
        dp = find_dtype_positions('<<U1>            <int64> <int64> <int64>')
        idxd = find_index_depth(ch, dp)  # 2
        post = extract_column_header_data([ch], dp, idxd)
        self.assertEqual(post, [(['x', 'y'], '<<U1>')])

    def test_build_columns_a(self) -> None:
        post = build_columns([(['x', 'y'], '<<U1>')])
        self.assertIs(post.__class__, Index)
        self.assertEqual(post.values.tolist(), ['x', 'y'])

    def test_build_columns_b(self) -> None:
        post = build_columns([])
        self.assertIs(post.__class__, Index)
        self.assertEqual(post.values.tolist(), [])

    def test_display_parse_frame_a(self) -> None:
        with self.assertRaises(ValueError):
            display_parse_frame('     ')

    def test_display_parse_frame_b(self) -> None:
        msg = """
        <Frame>
        <Index> a         b         <<U1>
        <Index>
        0       1.0       nan
        1       nan       2.0
        2       3.0       nan
        """
        with self.assertRaises(ValueError):
            display_parse_frame(msg)

    def test_display_parse_series_a(self) -> None:
        with self.assertRaises(ValueError):
            display_parse_series('     ')

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

    def test_frame_from_display_p(self) -> None:
        # >>> sf.Frame.from_element(0, index=('a', 'b'), columns=sf.IndexYear(['2022', '1913'])
        msg = """
<Frame>
<IndexYear> 2022    1913    <datetime64[Y]>
<Index>
a           0       0
b           0       0
<<U1>       <int64> <int64>
"""
        f = Frame.from_display(msg)

    def test_frame_empty_a(self) -> None:
        msg = """<Frame>
        <Index> <int64>
        <Index>
        <int64>"""
        f = Frame.from_display(msg)
        self.assertEqual(f.shape, (0, 0))

    def test_frame_empty_b(self) -> None:
        msg = """<Frame>
<Index> <<U1>
<Index>
<<U1>"""
        f = Frame.from_display(msg)
        self.assertEqual(f.shape, (0, 0))
        self.assertEqual(f.columns.dtype.kind, 'U')
        self.assertEqual(f.index.dtype.kind, 'U')

    def test_frame_ih_a(self) -> None:
        f1 = Frame.from_element(
            0, index=IndexHierarchy.from_labels([('a', 1, False)]), columns=(('a',))
        )
        f2 = Frame.from_display(repr(f1))
        self.assertTrue(
            f1.equals(f2, compare_class=True, compare_dtype=True, compare_name=True)
        )

    def test_frame_ih_b(self) -> None:
        # shape (0, 1)
        msg = """
<Frame>
<Index>                         a       <<U1>
<IndexHierarchy>
<<U1>            <int64> <bool> <int64>
"""
        f = Frame.from_display(msg)
        self.assertEqual(f.shape, (0, 1))
        self.assertEqual([dt.kind for dt in f.index.dtypes.values], ['U', 'i', 'b'])

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

    def test_series_from_display_j1(self) -> None:
        msg = """
        <Series>
        <IndexHierarchy>
        a                0       False  a
        <<U1>            <int64> <bool> <<U1>
        """
        s = Series.from_display(msg)
        self.assertEqual(s.shape, (1,))
        self.assertEqual(
            s.to_pairs(), (((np.str_('a'), np.int64(0), np.False_), np.str_('a')),)
        )
        self.assertEqual([dt.kind for dt in s.index.dtypes.values], ['U', 'i', 'b'])

    def test_series_from_display_j2(self) -> None:
        msg = """
        <Series>
        <IndexHierarchy>
        <<U1>            <int64> <bool> <<U1>
        """
        s = Series.from_display(msg)
        self.assertEqual(s.shape, (0,))
        self.assertEqual(s.to_pairs(), ())
        self.assertEqual([dt.kind for dt in s.index.dtypes.values], ['U', 'i', 'b'])


if __name__ == '__main__':
    unittest.main()
