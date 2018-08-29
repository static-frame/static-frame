from itertools import zip_longest
from itertools import combinations
import unittest
from collections import OrderedDict
from io import StringIO
import string
import hashlib

import numpy as np

from static_frame.test.test_case import TestCase

import static_frame as sf
# assuming located in the same directory
from static_frame import Index
from static_frame import IndexGO
from static_frame import Series
from static_frame import Frame
from static_frame import FrameGO
from static_frame import TypeBlocks
from static_frame import Display
from static_frame import mloc
from static_frame import DisplayConfig

from static_frame.core.util import _isna
from static_frame.core.util import _resolve_dtype
from static_frame.core.util import _resolve_dtype_iter
from static_frame.core.util import _array_to_duplicated
from static_frame.core.util import _array_set_ufunc_many


from static_frame.core.operator_delegate import _all
from static_frame.core.operator_delegate import _any
from static_frame.core.operator_delegate import _nanall
from static_frame.core.operator_delegate import _nanany

nan = np.nan

LONG_SAMPLE_STR = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'


class TestUnit(TestCase):

    def test_demo(self):
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        value1 = Series((100, 200, 300), index=('s', 'u', 't'))

        value2 = Frame.from_records(((20, 21, 22), (23, 24, 25)),
                index=('x', 'y'),
                columns=('s', 't', 'w'))

        f2 = FrameGO([
                [np.nan, 2, 3, 0],
                [3, 4, np.nan, 1],
                [0, 1, 2, 3]],
                columns=list('ABCD'))

        f3 = FrameGO([
                [np.nan, 2, 3, 0],
                [3, 4, np.nan, 1],
                [0, 1, 2, 3]],
                columns=list('EFGH'))




    #---------------------------------------------------------------------------
    # display tests

    def test_display_config_a(self):
        config = DisplayConfig.from_default(type_color=False)
        d = Display.from_values(np.array([[1, 2], [3, 4]]), 'header', config=config)
        self.assertEqual(d.to_rows(),
                ['header', '1 2', '3 4', '<int64>'])


    def test_display_config_b(self):
        post = sf.DisplayConfig.from_default(cell_align_left=False)

        self.assertEqual(
                sorted(post.to_dict().items()),
                [('cell_align_left', False), ('cell_max_width', 20), ('display_columns', 12), ('display_rows', 36), ('type_color', False), ('type_delimiter', '<>'), ('type_show', True)])


    def test_display_config_c(self):
        config_right = sf.DisplayConfig.from_default(cell_align_left=False)
        config_left = sf.DisplayConfig.from_default(cell_align_left=True)

        msg = config_right.to_json()
        # self.assertEqual(msg,
        #         '{"type_show": true, "type_delimiter": "<>", "cell_align_left": false, "type_color": false, "cell_max_width": 20, "display_rows": null, "display_columns": null}')
        # d = DisplayConfig.from_json(msg)


    def test_display_cell_align_left_a(self):
        config_right = sf.DisplayConfig.from_default(cell_align_left=False)
        config_left = sf.DisplayConfig.from_default(cell_align_left=True)

        index = Index((x for x in 'abc'))

        self.assertEqual(index.display(config=config_left).to_rows(),
                ['<Index>', 'a', 'b', 'c', '<object>'])

        self.assertEqual(
                index.display(config=config_right).to_rows(),
                [' <Index>', '       a', '       b', '       c', '<object>'])




    def test_display_cell_align_left_b(self):
        config_right = sf.DisplayConfig.from_default(cell_align_left=False)
        config_left = sf.DisplayConfig.from_default(cell_align_left=True)

        s = Series(range(3), index=('a', 'b', 'c'))

        self.assertEqual(s.display(config_right).to_rows(),
                ['<Index> <Series>',
                '      a        0',
                '      b        1',
                '      c        2',
                '  <<U1>  <int64>'])

        self.assertEqual(s.display(config_left).to_rows(),
                ['<Index> <Series>',
                'a       0',
                'b       1',
                'c       2',
                '<<U1>   <int64>'])


        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y'))

        self.assertEqual(f1.display(config_left).to_rows(),
                ['<Frame>',
                '<Index> p       q       r     s      t      <<U1>',
                '<Index>',
                'w       2       2       a     False  False',
                'x       30      34      b     True   False',
                'y       2       95      c     False  False',
                '<<U1>   <int64> <int64> <<U1> <bool> <bool>'])

        self.assertEqual(f1.display(config_right).to_rows(),
                ['<Frame>',
                '<Index>       p       q     r      s      t <<U1>',
                '<Index>',
                '      w       2       2     a  False  False',
                '      x      30      34     b   True  False',
                '      y       2      95     c  False  False',
                '  <<U1> <int64> <int64> <<U1> <bool> <bool>'])


    def test_display_type_show_a(self):
        config_type_show_true = sf.DisplayConfig.from_default(type_show=True)
        config_type_show_false = sf.DisplayConfig.from_default(type_show=False)

        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y'))

        self.assertEqual(f1.display(config_type_show_false).to_rows(),
                ['<Frame>',
                '<Index> p  q  r s     t',
                '<Index>',
                'w       2  2  a False False',
                'x       30 34 b True  False',
                'y       2  95 c False False',
                ''])

        self.assertEqual(f1.display(config_type_show_true).to_rows(),
                ['<Frame>',
                '<Index> p       q       r     s      t      <<U1>',
                '<Index>',
                'w       2       2       a     False  False',
                'x       30      34      b     True   False',
                'y       2       95      c     False  False',
                '<<U1>   <int64> <int64> <<U1> <bool> <bool>'])



    def test_display_cell_fill_width_a(self):

        config_width_12 = sf.DisplayConfig.from_default(cell_max_width=12)
        config_width_6 = sf.DisplayConfig.from_default(cell_max_width=6)

        def chunks(size, count):
            pos = 0
            for _ in range(count):
                yield LONG_SAMPLE_STR[pos: pos + size]
                pos = pos + size

        s = Series(chunks(20, 3), index=('a', 'b', 'c'))


        self.assertEqual(s.display(config=config_width_12).to_rows(),
                ['<Index> <Series>',
                'a       Lorem ips...',
                'b       t amet, c...',
                'c       adipiscin...',
                '<<U1>   <<U20>'])

        self.assertEqual(s.display(config=config_width_6).to_rows(),
                ['<In... <Se...',
                'a      Lor...',
                'b      t a...',
                'c      adi...',
                '<<U1>  <<U20>']
)

        row_count = 2
        index = [str(chr(x)) for x in range(97, 97+row_count)]
        f = FrameGO(index=index)
        for i in range(4):
            chunker = iter(chunks(10, row_count))
            s = Series((x for x in chunker), index=index)
            f[i] = s

        self.assertEqual(f.display().to_rows(),
                ['<FrameGO>',
                '<IndexGO> 0          1          2          3          <int64>',
                '<Index>',
                'a         Lorem ipsu Lorem ipsu Lorem ipsu Lorem ipsu',
                'b         m dolor si m dolor si m dolor si m dolor si',
                '<<U1>     <<U10>     <<U10>     <<U10>     <<U10>'])

        self.assertEqual(f.display(config=config_width_6).to_rows(),
                ['<Fr...',
                '<In... 0      1      2      3      <in...',
                '<In...',
                'a      Lor... Lor... Lor... Lor...',
                'b      m d... m d... m d... m d...',
                '<<U1>  <<U10> <<U10> <<U10> <<U10>']
                )


    def test_display_display_rows_a(self):

        config_rows_12 = sf.DisplayConfig.from_default(display_rows=12)
        config_rows_7 = sf.DisplayConfig.from_default(display_rows=7)

        index = list(''.join(x) for x in combinations(string.ascii_lowercase, 2))
        s = Series(range(len(index)), index=index)

        self.assertEqual(s.display(config_rows_12).to_rows(),
                ['<Index> <Series>',
                'ab      0',
                'ac      1',
                'ad      2',
                'ae      3',
                '...     ...',
                'wz      321',
                'xy      322',
                'xz      323',
                'yz      324',
                '<<U2>   <int64>'])

        self.assertEqual(s.display(config_rows_7).to_rows(),
                ['<Index> <Series>',
                'ab      0',
                'ac      1',
                '...     ...',
                'xz      323',
                'yz      324',
                '<<U2>   <int64>'])



    def test_display_display_columns_a(self):

        config_columns_8 = sf.DisplayConfig.from_default(display_columns=8)
        config_columns_5 = sf.DisplayConfig.from_default(display_columns=5)

        columns = list(''.join(x) for x in combinations(string.ascii_lowercase, 2))
        f = FrameGO(index=range(4))
        for i, col in enumerate(columns):
            f[col] = Series(i, index=range(4))

        self.assertEqual(
                f.display(config_columns_8).to_rows(),
                ['<FrameGO>',
                '<IndexGO> ab      ac      ... xz      yz      <<U2>',
                '<Index>                   ...',
                '0         0       1       ... 323     324',
                '1         0       1       ... 323     324',
                '2         0       1       ... 323     324',
                '3         0       1       ... 323     324',
                '<int64>   <int64> <int64> ... <int64> <int64>']
                )

        self.assertEqual(
                f.display(config_columns_5).to_rows(),
                ['<FrameGO>',
                '<IndexGO> ab      ... yz      <<U2>',
                '<Index>           ...',
                '0         0       ... 324',
                '1         0       ... 324',
                '2         0       ... 324',
                '3         0       ... 324',
                '<int64>   <int64> ... <int64>'])




    def test_display_display_columns_b(self):

        config_columns_4 = sf.DisplayConfig.from_default(display_columns=4)
        config_columns_5 = sf.DisplayConfig.from_default(display_columns=5)

        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                )

        f = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y'))

        self.assertEqual(
                f.display(config_columns_4).to_rows(),
                ['<Frame>',
                '<Index> p       ... t      <<U1>',
                '<Index>         ...',
                'w       2       ... False',
                'x       30      ... False',
                'y       2       ... False',
                '<<U1>   <int64> ... <bool>'])

        # at one point the columns woiuld be truncated shorter than the frame when the max xolumns was the same
        self.assertEqual(
                f.display(config_columns_5).to_rows(),
                ['<Frame>',
                '<Index> p       q       r     s      t      <<U1>',
                '<Index>',
                'w       2       2       a     False  False',
                'x       30      34      b     True   False',
                'y       2       95      c     False  False',
                '<<U1>   <int64> <int64> <<U1> <bool> <bool>']
                )


    def test_display_truncate_a(self):

        config_rows_12_cols_8 = sf.DisplayConfig.from_default(display_rows=12, display_columns=8)
        config_rows_7_cols_5 = sf.DisplayConfig.from_default(display_rows=7, display_columns=5)


        size = 10000
        columns = 100
        a1 = (np.arange(size * columns)).reshape((size, columns)) * .001
        # insert random nan in very other columns
        for col in range(0, 100, 2):
            a1[:100, col] = np.nan

        index = (hashlib.sha224(str(x).encode('utf-8')).hexdigest()
                for x in range(size))
        columns = (hashlib.sha224(str(x).encode('utf-8')).hexdigest()
                for x in range(columns))

        f = Frame(a1, index=index, columns=columns)

        self.assertEqual(
                len(f.display(config_rows_12_cols_8).to_rows()), 13)

        self.assertEqual(
                len(f.display(config_rows_7_cols_5).to_rows()), 9)


    #---------------------------------------------------------------------------
    # test series

    def test_series_init_a(self):
        s1 = Series(np.nan, index=('a', 'b', 'c', 'd'))

        self.assertTrue(s1.dtype == float)
        self.assertTrue(len(s1) == 4)

        s2 = Series(False, index=('a', 'b', 'c', 'd'))

        self.assertTrue(s2.dtype == bool)
        self.assertTrue(len(s2) == 4)

        s3 = Series(None, index=('a', 'b', 'c', 'd'))

        self.assertTrue(s3.dtype == object)
        self.assertTrue(len(s3) == 4)


    def test_series_init_b(self):
        s1 = Series(['a', 'b', 'c', 'd'], index=('a', 'b', 'c', 'd'))
        self.assertEqual(s1.to_pairs(),
                (('a', 'a'), ('b', 'b'), ('c', 'c'), ('d', 'd')))

        # testing direct specification of string type
        s2 = Series(['a', 'b', 'c', 'd'], index=('a', 'b', 'c', 'd'), dtype=str)
        self.assertEqual(s2.to_pairs(),
                (('a', 'a'), ('b', 'b'), ('c', 'c'), ('d', 'd')))

    def test_series_init_c(self):
        s1 = Series(dict(a=1, b=4), dtype=int)
        self.assertEqual(s1.to_pairs(),
                (('a', 1), ('b', 4)))

        s1 = Series(dict(b=4, a=1), dtype=int)
        self.assertEqual(s1.to_pairs(),
                (('a', 1), ('b', 4)))

        s1 = Series(OrderedDict([('b', 4), ('a', 1)]), dtype=int)
        self.assertEqual(s1.to_pairs(),
                (('b', 4), ('a', 1)))



    def test_series_slice_a(self):
        # create a series from a single value
        # s0 = Series(3, index=('a',))

        # generator based construction of values and index
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        # self.assertEqual(s1['b'], 1)
        # self.assertEqual(s1['d'], 3)

        s2 = s1['a':'c'] # with Pandas this is inclusive
        self.assertEqual(s2.values.tolist(), [0, 1, 2])
        self.assertTrue(s2['b'] == s1['b'])

        s3 = s1['c':]
        self.assertEqual(s3.values.tolist(), [2, 3])
        self.assertTrue(s3['d'] == s1['d'])

        self.assertEqual(s1['b':'d'].values.tolist(), [1, 2, 3])

        self.assertEqual(s1[['a', 'c']].values.tolist(), [0, 2])


    def test_series_keys_a(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        self.assertEqual(list(s1.keys()), ['a', 'b', 'c', 'd'])

    def test_series_iter_a(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        self.assertEqual(list(s1), ['a', 'b', 'c', 'd'])

    def test_series_items_a(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        self.assertEqual(list(s1.items()), [('a', 0), ('b', 1), ('c', 2), ('d', 3)])


    def test_series_intersection_a(self):
        # create a series from a single value
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        s3 = s1['c':]
        self.assertEqual(s1.index.intersection(s3.index).values.tolist(),
            ['c', 'd'])


    def test_series_intersection_b(self):
        # create a series from a single value
        idxa = IndexGO(('a', 'b', 'c'))
        idxb = IndexGO(('b', 'c', 'd'))

        self.assertEqual(idxa.intersection(idxb).values.tolist(),
            ['b', 'c'])

        self.assertEqual(idxa.union(idxb).values.tolist(),
            ['a', 'b', 'c', 'd'])




    def test_series_binary_operator_a(self):
        '''Test binary operators where one operand is a numeric.
        '''
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        self.assertEqual(list((s1 * 3).items()),
                [('a', 0), ('b', 3), ('c', 6), ('d', 9)])

        self.assertEqual(list((s1 / .5).items()),
                [('a', 0.0), ('b', 2.0), ('c', 4.0), ('d', 6.0)])

        self.assertEqual(list((s1 ** 3).items()),
                [('a', 0), ('b', 1), ('c', 8), ('d', 27)])


    def test_series_binary_operator_b(self):
        '''Test binary operators with Series of same index
        '''
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        s2 = Series((x * 2 for x in range(4)), index=('a', 'b', 'c', 'd'))

        self.assertEqual(list((s1 + s2).items()),
                [('a', 0), ('b', 3), ('c', 6), ('d', 9)])

        self.assertEqual(list((s1 * s2).items()),
                [('a', 0), ('b', 2), ('c', 8), ('d', 18)])




    def test_series_binary_operator_c(self):
        '''Test binary operators with Series of different index
        '''
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        s2 = Series((x * 2 for x in range(4)), index=('c', 'd', 'e', 'f'))

        self.assertAlmostEqualItems(list((s1 * s2).items()),
                [('a', nan), ('b', nan), ('c', 0), ('d', 6), ('e', nan), ('f', nan)]
                )


    def test_series_reindex_a(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        s2 = s1.reindex(('c', 'd', 'a'))
        self.assertEqual(list(s2.items()), [('c', 2), ('d', 3), ('a', 0)])

        s3 = s1.reindex(['a','b'])
        self.assertEqual(list(s3.items()), [('a', 0), ('b', 1)])


        # an int-valued array is hard to provide missing values for

        s4 = s1.reindex(['b', 'q', 'g', 'a'], fill_value=None)
        self.assertEqual(list(s4.items()),
                [('b', 1), ('q', None), ('g', None), ('a', 0)])

        # by default this gets float because filltype is nan by default
        s5 = s1.reindex(['b', 'q', 'g', 'a'])
        self.assertAlmostEqualItems(list(s5.items()),
                [('b', 1), ('q', nan), ('g', nan), ('a', 0)])


    def test_series_isnull_a(self):

        s1 = Series((234.3, 3.2, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s2 = Series((234.3, None, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s3 = Series((234.3, 5, 6.4, -234.3), index=('a', 'b', 'c', 'd'))
        s4 = Series((234.3, None, None, None), index=('a', 'b', 'c', 'd'))
        s5 = Series(('p', 'q', 'e', 'g'), index=('a', 'b', 'c', 'd'))
        s6 = Series((False, True, False, True), index=('a', 'b', 'c', 'd'))

        self.assertEqual(list(s1.isna().items()),
                [('a', False), ('b', False), ('c', False), ('d', True)]
                )
        self.assertEqual(list(s2.isna().items()),
                [('a', False), ('b', True), ('c', False), ('d', True)])

        self.assertEqual(list(s3.isna().items()),
                [('a', False), ('b', False), ('c', False), ('d', False)])

        self.assertEqual(list(s4.isna().items()),
                [('a', False), ('b', True), ('c', True), ('d', True)])

        # those that are always false
        self.assertEqual(list(s5.isna().items()),
                [('a', False), ('b', False), ('c', False), ('d', False)])

        self.assertEqual(list(s6.isna().items()),
                [('a', False), ('b', False), ('c', False), ('d', False)])



    def test_series_isnull_b(self):

        # NOTE: this is a problematic case as it as a string with numerics and None
        s1 = Series((234.3, 'a', None, 6.4, np.nan), index=('a', 'b', 'c', 'd', 'e'))

        self.assertEqual(list(s1.isna().items()),
                [('a', False), ('b', False), ('c', True), ('d', False), ('e', True)]
                )

    def test_series_notnull(self):

        s1 = Series((234.3, 3.2, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s2 = Series((234.3, None, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s3 = Series((234.3, 5, 6.4, -234.3), index=('a', 'b', 'c', 'd'))
        s4 = Series((234.3, None, None, None), index=('a', 'b', 'c', 'd'))
        s5 = Series(('p', 'q', 'e', 'g'), index=('a', 'b', 'c', 'd'))
        s6 = Series((False, True, False, True), index=('a', 'b', 'c', 'd'))

        self.assertEqual(list(s1.notna().items()),
                [('a', True), ('b', True), ('c', True), ('d', False)]
                )
        self.assertEqual(list(s2.notna().items()),
                [('a', True), ('b', False), ('c', True), ('d', False)])

        self.assertEqual(list(s3.notna().items()),
                [('a', True), ('b', True), ('c', True), ('d', True)])

        self.assertEqual(list(s4.notna().items()),
                [('a', True), ('b', False), ('c', False), ('d', False)])

        # those that are always false
        self.assertEqual(list(s5.notna().items()),
                [('a', True), ('b', True), ('c', True), ('d', True)])

        self.assertEqual(list(s6.notna().items()),
                [('a', True), ('b', True), ('c', True), ('d', True)])


    def test_series_dropna(self):

        s1 = Series((234.3, 3.2, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s2 = Series((234.3, None, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s3 = Series((234.3, 5, 6.4, -234.3), index=('a', 'b', 'c', 'd'))
        s4 = Series((234.3, None, None, None), index=('a', 'b', 'c', 'd'))
        s5 = Series(('p', 'q', 'e', 'g'), index=('a', 'b', 'c', 'd'))
        s6 = Series((False, True, False, True), index=('a', 'b', 'c', 'd'))


        self.assertEqual(list(s2.dropna().items()),
                [('a', 234.3), ('c', 6.4)])


    def test_series_fillna_a(self):

        s1 = Series((234.3, 3.2, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s2 = Series((234.3, None, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s3 = Series((234.3, 5, 6.4, -234.3), index=('a', 'b', 'c', 'd'))
        s4 = Series((234.3, None, None, None), index=('a', 'b', 'c', 'd'))
        s5 = Series(('p', 'q', 'e', 'g'), index=('a', 'b', 'c', 'd'))
        s6 = Series((False, True, False, True), index=('a', 'b', 'c', 'd'))
        s7 = Series((10, 20, 30, 40), index=('a', 'b', 'c', 'd'))
        s8 = Series((234.3, None, 6.4, np.nan, 'q'), index=('a', 'b', 'c', 'd', 'e'))


        self.assertEqual(s1.fillna(0.0).values.tolist(),
                [234.3, 3.2, 6.4, 0.0])

        self.assertEqual(s1.fillna(-1).values.tolist(),
                [234.3, 3.2, 6.4, -1.0])

        # given a float array, inserting None, None is casted to nan
        self.assertEqual(s1.fillna(None).values.tolist(),
                [234.3, 3.2, 6.4, None])

        post = s1.fillna('wer')
        self.assertEqual(post.dtype, object)
        self.assertEqual(post.values.tolist(),
                [234.3, 3.2, 6.4, 'wer'])


        post = s7.fillna(None)
        self.assertEqual(post.dtype, int)


    def test_series_from_pairs_a(self):

        def gen():
            r1 = range(10)
            r2 = iter(range(10, 20))
            for x in r1:
                yield x, next(r2)

        s1 = Series.from_items(gen())
        self.assertEqual(s1.loc[7:9].values.tolist(), [17, 18, 19])

        # NOTE: ordere here is unstable until python 3.6
        s2 = Series.from_items(dict(a=30, b=40, c=50).items())
        self.assertEqual(s2['c'], 50)
        self.assertEqual(s2['b'], 40)
        self.assertEqual(s2['a'], 30)


    def test_series_contains_a(self):

        s1 = Series.from_items(zip(('a', 'b', 'c'), (10, 20, 30)))
        self.assertTrue('b' in s1)
        self.assertTrue('c' in s1)
        self.assertTrue('a' in s1)

        self.assertFalse('d' in s1)
        self.assertFalse('' in s1)


    def test_series_sum_a(self):

        s1 = Series.from_items(zip(('a', 'b', 'c'), (10, 20, 30)))
        self.assertEqual(s1.sum(), 60)

        s1 = Series.from_items(zip(('a', 'b', 'c', 'd'), (10, 20, 30, np.nan)))
        self.assertEqual(s1.sum(), 60)

        s1 = Series.from_items(zip(('a', 'b', 'c', 'd'), (10, 20, 30, None)))
        self.assertEqual(s1.sum(), 60)


    def test_series_mask_a(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        self.assertEqual(
                s1.mask.loc[['b', 'd']].values.tolist(),
                [False, True, False, True])
        self.assertEqual(s1.mask.iloc[1:].values.tolist(),
                [False, True, True, True])

        self.assertEqual(s1.masked_array.loc[['b', 'd']].sum(), 2)
        self.assertEqual(s1.masked_array.loc[['a', 'b']].sum(), 5)



    def test_series_assign_a(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))


        self.assertEqual(
                s1.assign.loc[['b', 'd']](3000).values.tolist(),
                [0, 3000, 2, 3000])

        self.assertEqual(
                s1.assign['b':](300).values.tolist(),
                [0, 300, 300, 300])


    def test_series_assign_b(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        self.assertEqual(list(s1.isin([2]).items()),
                [('a', False), ('b', False), ('c', True), ('d', False)])

        self.assertEqual(list(s1.isin({2, 3}).items()),
                [('a', False), ('b', False), ('c', True), ('d', True)])

        self.assertEqual(list(s1.isin(range(2, 4)).items()),
                [('a', False), ('b', False), ('c', True), ('d', True)])


    def test_series_assign_c(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        self.assertEqual(s1.assign.loc['c':](0).to_pairs(),
                (('a', 0), ('b', 1), ('c', 0), ('d', 0))
                )
        self.assertEqual(s1.assign.loc['c':]((20, 30)).to_pairs(),
                (('a', 0), ('b', 1), ('c', 20), ('d', 30)))

        self.assertEqual(s1.assign['c':](s1['c':] * 10).to_pairs(),
                (('a', 0), ('b', 1), ('c', 20), ('d', 30)))

        self.assertEqual(s1.assign['c':](Series({'d':40, 'c':60})).to_pairs(),
                (('a', 0), ('b', 1), ('c', 60), ('d', 40)))


    def test_series_assign_d(self):
        s1 = Series(tuple('pqrs'), index=('a', 'b', 'c', 'd'))
        s2 = s1.assign['b'](None)
        self.assertEqual(s2.to_pairs(),
                (('a', 'p'), ('b', None), ('c', 'r'), ('d', 's')))
        self.assertEqual(s1.assign['b':](None).to_pairs(),
                (('a', 'p'), ('b', None), ('c', None), ('d', None)))


    def test_series_loc_extract_a(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        # TODO: raaise exectin when doing a loc that Pandas reindexes



    def test_series_group_a(self):

        s1 = Series((0, 1, 0, 1), index=('a', 'b', 'c', 'd'))

        groups = tuple(s1.iter_group_items())

        self.assertEqual([g[0] for g in groups], [0, 1])

        self.assertEqual([g[1].to_pairs() for g in groups],
                [(('a', 0), ('c', 0)), (('b', 1), ('d', 1))])

    def test_series_group_b(self):

        s1 = Series(('foo', 'bar', 'foo', 20, 20),
                index=('a', 'b', 'c', 'd', 'e'),
                dtype=object)

        groups = tuple(s1.iter_group_items())


        self.assertEqual([g[0] for g in groups],
                [20, 'bar', 'foo'])
        self.assertEqual([g[1].to_pairs() for g in groups],
                [(('d', 20), ('e', 20)), (('b', 'bar'),), (('a', 'foo'), ('c', 'foo'))])


    def test_series_group_c(self):

        s1 = Series((10, 10, 10, 20, 20),
                index=('a', 'b', 'c', 'd', 'e'),
                dtype=object)

        groups = tuple(s1.iter_group())
        self.assertEqual([g.sum() for g in groups], [30, 40])

        self.assertEqual(
                s1.iter_group().apply(np.sum).to_pairs(),
                ((10, 30), (20, 40)))

        self.assertEqual(
                s1.iter_group_items().apply(lambda g, s: (g * s).values.tolist()).to_pairs(),
                ((10, [100, 100, 100]), (20, [400, 400])))



    def test_series_iter_element_a(self):

        s1 = Series((10, 3, 15, 21, 28),
                index=('a', 'b', 'c', 'd', 'e'),
                dtype=object)

        self.assertEqual([x for x in s1.iter_element()], [10, 3, 15, 21, 28])

        self.assertEqual([x for x in s1.iter_element_items()],
                        [('a', 10), ('b', 3), ('c', 15), ('d', 21), ('e', 28)])

        self.assertEqual(s1.iter_element().apply(lambda x: x * 20).to_pairs(),
                (('a', 200), ('b', 60), ('c', 300), ('d', 420), ('e', 560)))

        self.assertEqual(
                s1.iter_element_items().apply(lambda k, v: v * 20 if k == 'b' else 0).to_pairs(),
                (('a', 0), ('b', 60), ('c', 0), ('d', 0), ('e', 0)))



    def test_series_sort_index_a(self):

        s1 = Series((10, 3, 28, 21, 15),
                index=('a', 'c', 'b', 'e', 'd'),
                dtype=object)

        self.assertEqual(s1.sort_index().to_pairs(),
                (('a', 10), ('b', 28), ('c', 3), ('d', 15), ('e', 21)))

        self.assertEqual(s1.sort_values().to_pairs(),
                (('c', 3), ('a', 10), ('d', 15), ('e', 21), ('b', 28)))


    def test_series_relabel_a(self):

        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        s2 = s1.relabel({'b': 'bbb'})
        self.assertEqual(s2.to_pairs(),
                (('a', 0), ('bbb', 1), ('c', 2), ('d', 3)))

        self.assertEqual(mloc(s2.values), mloc(s1.values))


    def test_series_relabel_b(self):

        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        s2 = s1.relabel({'a':'x', 'b':'y', 'c':'z', 'd':'q'})

        self.assertEqual(list(s2.items()),
            [('x', 0), ('y', 1), ('z', 2), ('q', 3)])

    def test_series_get_a(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        self.assertEqual(s1.get('q'), None)
        self.assertEqual(s1.get('a'), 0)
        self.assertEqual(s1.get('f', -1), -1)


    def test_series_all_a(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        self.assertEqual(s1.all(), False)
        self.assertEqual(s1.any(), True)


    def test_series_all_b(self):
        s1 = Series([True, True, np.nan, True], index=('a', 'b', 'c', 'd'), dtype=object)

        self.assertEqual(s1.all(skipna=False), True)
        self.assertEqual(s1.all(skipna=True), False)
        self.assertEqual(s1.any(), True)


    def test_series_unique_a(self):
        s1 = Series([10, 10, 2, 2], index=('a', 'b', 'c', 'd'), dtype=int)

        self.assertEqual(s1.unique().tolist(), [2, 10])

        s2 = Series(['b', 'b', 'c', 'c'], index=('a', 'b', 'c', 'd'), dtype=object)
        self.assertEqual(s2.unique().tolist(), ['b', 'c'])


    def test_series_unique_a(self):
        s1 = Series([10, 10, 2, 2], index=('a', 'b', 'c', 'd'), dtype=int)

        self.assertEqual(s1.unique().tolist(), [2, 10])

        s2 = Series(['b', 'b', 'c', 'c'], index=('a', 'b', 'c', 'd'), dtype=object)
        self.assertEqual(s2.unique().tolist(), ['b', 'c'])



    def test_series_duplicated_a(self):
        s1 = Series([1, 10, 10, 5, 2, 2],
                index=('a', 'b', 'c', 'd', 'e', 'f'), dtype=int)

        # this is showing all duplicates, not just the first-found
        self.assertEqual(s1.duplicated().to_pairs(),
                (('a', False), ('b', True), ('c', True), ('d', False), ('e', True), ('f', True)))

        self.assertEqual(s1.duplicated(exclude_first=True).to_pairs(),
                (('a', False), ('b', False), ('c', True), ('d', False), ('e', False), ('f', True)))

        self.assertEqual(s1.duplicated(exclude_last=True).to_pairs(),
                (('a', False), ('b', True), ('c', False), ('d', False), ('e', True), ('f', False)))


    def test_series_duplicated_b(self):
        s1 = Series([5, 3, 3, 3, 7, 2, 2, 2, 1],
                index=('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'), dtype=int)

        # this is showing all duplicates, not just the first-found
        self.assertEqual(s1.duplicated().to_pairs(),
                (('a', False), ('b', True), ('c', True),
                ('d', True), ('e', False), ('f', True),
                ('g', True), ('h', True), ('i', False),
                ))

        self.assertEqual(s1.duplicated(exclude_first=True).to_pairs(),
                (('a', False), ('b', False), ('c', True),
                ('d', True), ('e', False), ('f', False),
                ('g', True), ('h', True), ('i', False),
                ))

        self.assertEqual(s1.duplicated(exclude_last=True).to_pairs(),
                (('a', False), ('b', True), ('c', True),
                ('d', False), ('e', False), ('f', True),
                ('g', True), ('h', False), ('i', False),
                ))


    def test_series_drop_duplicated_a(self):
        s1 = Series([5, 3, 3, 3, 7, 2, 2, 2, 1],
                index=('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'), dtype=int)

        self.assertEqual(s1.drop_duplicated().to_pairs(),
                (('a', 5), ('e', 7), ('i', 1)))

        self.assertEqual(s1.drop_duplicated(exclude_first=True).to_pairs(),
                (('a', 5), ('b', 3), ('e', 7), ('f', 2), ('i', 1))
                )


if __name__ == '__main__':
    unittest.main()
