import itertools
from itertools import zip_longest
from itertools import combinations
import unittest
from collections import OrderedDict
from io import StringIO
from io import BytesIO
import string
import hashlib

import numpy as np

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

from static_frame import _isna
from static_frame import _resolve_dtype
from static_frame import _resolve_dtype_iter
from static_frame import _ufunc_logical_skipna
from static_frame import _array_to_duplicated
from static_frame import _array_set_ufunc_many


nan = np.nan

LONG_SAMPLE_STR = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'


class TestUnit(unittest.TestCase):

    def setUp(self):
        pass

    def assertTypeBlocksArrayEqual(self, tb: TypeBlocks, match, match_dtype=None):
        '''
        Args:
            tb: a TypeBlocks instance
            match: can be anything that can be used to create an array.
        '''
        # could use np.testing
        if not isinstance(match, np.ndarray):
            match = np.array(match, dtype=match_dtype)
        self.assertTrue((tb.values == match).all())


    def assertAlmostEqualItems(self, pairs1, pairs2):
        for (k1, v1), (k2, v2) in zip_longest(pairs1, pairs2):
            self.assertEqual(k1, k2)

            if isinstance(v1, float) and np.isnan(v1) and isinstance(v2, float) and np.isnan(v2):
                continue

            self.assertEqual(v1, v2)


    def assertAlmostEqualFramePairs(self, pairs1, pairs2):
        '''
        For comparing nested tuples returned by Frame.to_pairs()
        '''
        for (k1, v1), (k2, v2) in zip_longest(pairs1, pairs2):
            self.assertEqual(k1, k2)
            self.assertAlmostEqualItems(v1, v2)

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

        value2 = Frame.from_records(((20, 21, 22),(23, 24, 25)),
                index=('x', 'y'),
                columns=('s', 't', 'w'))

        f2 = FrameGO([
                [np.nan, 2, 3, 0],
                [3, 4, np.nan, 1],
                [0 ,1, 2, 3]],
                columns=list('ABCD'))

        f3 = FrameGO([
                [np.nan, 2, 3, 0],
                [3, 4, np.nan, 1],
                [0 ,1, 2, 3]],
                columns=list('EFGH'))

    #---------------------------------------------------------------------------
    # module level resources

    def test_resolve_dtype_a(self):

        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        a4 = np.array([2.3, 3.2])
        a5 = np.array(['test', 'test again'], dtype='S')
        a6 = np.array([2.3,5.4], dtype='float32')

        self.assertEqual(_resolve_dtype(a1.dtype, a1.dtype), a1.dtype)

        self.assertEqual(_resolve_dtype(a1.dtype, a2.dtype), np.object_)
        self.assertEqual(_resolve_dtype(a2.dtype, a3.dtype), np.object_)
        self.assertEqual(_resolve_dtype(a2.dtype, a4.dtype), np.object_)
        self.assertEqual(_resolve_dtype(a3.dtype, a4.dtype), np.object_)
        self.assertEqual(_resolve_dtype(a3.dtype, a6.dtype), np.object_)

        self.assertEqual(_resolve_dtype(a1.dtype, a4.dtype), np.float64)
        self.assertEqual(_resolve_dtype(a1.dtype, a6.dtype), np.float64)
        self.assertEqual(_resolve_dtype(a4.dtype, a6.dtype), np.float64)

    def test_resolve_dtype_iter_a(self):

        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        a4 = np.array([2.3, 3.2])
        a5 = np.array(['test', 'test again'], dtype='S')
        a6 = np.array([2.3,5.4], dtype='float32')

        self.assertEqual(_resolve_dtype_iter((a1.dtype, a1.dtype)), a1.dtype)
        self.assertEqual(_resolve_dtype_iter((a2.dtype, a2.dtype)), a2.dtype)

        # boolean with mixed types
        self.assertEqual(_resolve_dtype_iter((a2.dtype, a2.dtype, a3.dtype)), np.object_)
        self.assertEqual(_resolve_dtype_iter((a2.dtype, a2.dtype, a5.dtype)), np.object_)
        self.assertEqual(_resolve_dtype_iter((a2.dtype, a2.dtype, a6.dtype)), np.object_)

        # numerical types go to float64
        self.assertEqual(_resolve_dtype_iter((a1.dtype, a4.dtype, a6.dtype)), np.float64)

        # add in bool or str, goes to object
        self.assertEqual(_resolve_dtype_iter((a1.dtype, a4.dtype, a6.dtype, a2.dtype)), np.object_)
        self.assertEqual(_resolve_dtype_iter((a1.dtype, a4.dtype, a6.dtype, a5.dtype)), np.object_)

        # mixed strings go to the largest
        self.assertEqual(_resolve_dtype_iter((a3.dtype, a5.dtype)), np.dtype('<U10'))


    def test_isna_array_a(self):

        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        a4 = np.array([2.3, 3.2])
        a5 = np.array(['test', 'test again'], dtype='S')
        a6 = np.array([2.3, 5.4], dtype='float32')

        self.assertEqual(_isna(a1).tolist(), [False, False, False])
        self.assertEqual(_isna(a2).tolist(), [False, False, False])
        self.assertEqual(_isna(a3).tolist(), [False, False, False])
        self.assertEqual(_isna(a4).tolist(), [False, False])
        self.assertEqual(_isna(a5).tolist(), [False, False])
        self.assertEqual(_isna(a6).tolist(), [False, False])

        a1 = np.array([1, 2, 3, None])
        a2 = np.array([False, True, False, None])
        a3 = np.array(['b', 'c', 'd', None])
        a4 = np.array([2.3, 3.2, None])
        a5 = np.array(['test', 'test again', None])
        a6 = np.array([2.3, 5.4, None])

        self.assertEqual(_isna(a1).tolist(), [False, False, False, True])
        self.assertEqual(_isna(a2).tolist(), [False, False, False, True])
        self.assertEqual(_isna(a3).tolist(), [False, False, False, True])
        self.assertEqual(_isna(a4).tolist(), [False, False, True])
        self.assertEqual(_isna(a5).tolist(), [False, False, True])
        self.assertEqual(_isna(a6).tolist(), [False, False, True])

        a1 = np.array([1, 2, 3, np.nan])
        a2 = np.array([False, True, False, np.nan])
        a3 = np.array(['b', 'c', 'd', np.nan], dtype=object)
        a4 = np.array([2.3, 3.2, np.nan], dtype=object)
        a5 = np.array(['test', 'test again', np.nan], dtype=object)
        a6 = np.array([2.3, 5.4, np.nan], dtype='float32')

        self.assertEqual(_isna(a1).tolist(), [False, False, False, True])
        self.assertEqual(_isna(a2).tolist(), [False, False, False, True])
        self.assertEqual(_isna(a3).tolist(), [False, False, False, True])
        self.assertEqual(_isna(a4).tolist(), [False, False, True])
        self.assertEqual(_isna(a5).tolist(), [False, False, True])
        self.assertEqual(_isna(a6).tolist(), [False, False, True])


    def test_isna_array_b(self):

        a1 = np.array([[1, 2], [3, 4]])
        a2 = np.array([[False, True, False], [False, True, False]])
        a3 = np.array([['b', 'c', 'd'], ['b', 'c', 'd']])
        a4 = np.array([[2.3, 3.2, np.nan], [2.3, 3.2, np.nan]])
        a5 = np.array([['test', 'test again', np.nan],
                ['test', 'test again', np.nan]], dtype=object)
        a6 = np.array([[2.3, 5.4, np.nan], [2.3, 5.4, np.nan]], dtype='float32')

        self.assertEqual(_isna(a1).tolist(),
                [[False, False], [False, False]])

        self.assertEqual(_isna(a2).tolist(),
                [[False, False, False], [False, False, False]])

        self.assertEqual(_isna(a3).tolist(),
                [[False, False, False], [False, False, False]])

        self.assertEqual(_isna(a4).tolist(),
                [[False, False, True], [False, False, True]])

        self.assertEqual(_isna(a5).tolist(),
                [[False, False, True], [False, False, True]])

        self.assertEqual(_isna(a6).tolist(),
                [[False, False, True], [False, False, True]])


    def test_ufunc_logical_skipna_a(self):

        # empty arrays
        a1 = np.array([], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False), True)

        a1 = np.array([], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.any, skipna=False), False)


        # float arrays 1d
        a1 = np.array([2.4, 5.4], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=True), True)

        a1 = np.array([2.4, 0], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False), False)

        a1 = np.array([0, np.nan, 0], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.any, skipna=True), False)

        a1 = np.array([0, np.nan, 0], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.any, skipna=False), True)


        # float arrays 2d
        a1 = np.array([[2.4, 5.4, 3.2], [2.4, 5.4, 3.2]], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False, axis=0).tolist(),
                [True, True, True])


        a1 = np.array([[2.4, 5.4, 3.2], [2.4, 5.4, 3.2]], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False, axis=1).tolist(),
                [True, True])

        a1 = np.array([[2.4, 5.4, 0], [2.4, 5.4, 3.2]], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False, axis=0).tolist(),
                [True, True, False])

        a1 = np.array([[2.4, 5.4, 0], [2.4, 5.4, 3.2]], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False, axis=1).tolist(),
                [False, True])


        # object arrays
        a1 = np.array([[2.4, 5.4, 0], [2.4, None, 3.2]], dtype=object)

        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False, axis=1).tolist(),
                [False, False])
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False, axis=0).tolist(),
                [True, False, False])


        a1 = np.array([[2.4, 5.4, 0], [2.4, np.nan, 3.2]], dtype=object)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False, axis=0).tolist(),
                [True, True, False])


    def test_ufunc_logical_skipna_b(self):
        # object arrays

        a1 = np.array([['sdf', '', 'wer'], [True, False, True]], dtype=object)

        self.assertEqual(
                _ufunc_logical_skipna(a1, np.all, skipna=False, axis=0).tolist(),
                [True, False, True]
                )
        self.assertEqual(
                _ufunc_logical_skipna(a1, np.all, skipna=False, axis=1).tolist(),
                [False, False]
                )


        # string arrays
        a1 = np.array(['sdf', ''], dtype=str)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False, axis=0), False)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=True, axis=0), False)


        a1 = np.array([['sdf', '', 'wer'], ['sdf', '', 'wer']], dtype=str)
        self.assertEqual(
                _ufunc_logical_skipna(a1, np.all, skipna=False, axis=0).tolist(),
                [True,  False,  True])

        self.assertEqual(
                _ufunc_logical_skipna(a1, np.all, skipna=False, axis=1).tolist(),
                [False, False])

        self.assertEqual(
                _ufunc_logical_skipna(a1, np.any, skipna=False, axis=0).tolist(),
                [True,  False,  True])

        self.assertEqual(
                _ufunc_logical_skipna(a1, np.any, skipna=False, axis=1).tolist(),
                [True, True])


    def test_array_to_duplicated_a(self):
        a = _array_to_duplicated(
                np.array([0,1,2,2,1,4,5,3,4,5,5,6]),
                exclude_first=False,
                exclude_last=False
                )
        self.assertEqual(a.tolist(),
                [False, True, True, True, True, True, True, False, True, True, True, False])

        a = _array_to_duplicated(
                np.array([0,1,2,2,1,4,5,3,4,5,5,6]),
                exclude_first=True,
                exclude_last=False
                )
        self.assertEqual(a.tolist(),
                [False, False, False, True, True, False, False, False, True, True, True, False])


    def test_array_to_duplicated_b(self):
        a = np.array([[50, 50, 32, 17, 17], [2,2,1,3,3]])
        # find duplicate rows
        post = _array_to_duplicated(a, axis=0)
        self.assertEqual(post.tolist(),
                [False, False])

        post = _array_to_duplicated(a, axis=1)
        self.assertEqual(post.tolist(),
                [True, True, False, True, True])

        post = _array_to_duplicated(a, axis=1, exclude_first=True)
        self.assertEqual(post.tolist(),
                [False, True, False, False, True])


    def test_array_set_ufunc_many_a(self):
        a1 = np.array([3, 2, 1])
        a2 = np.array([3, 2, 1])
        a3 = np.array([3, 2, 1])
        a4 = np.array([3, 2, 1])

        post = _array_set_ufunc_many((a1, a2, a3, a4), ufunc=np.intersect1d)
        self.assertEqual(post.tolist(), [3, 2, 1])

        post = _array_set_ufunc_many((a1, a2, a3, a4), ufunc=np.union1d)
        self.assertEqual(post.tolist(), [3, 2, 1])

    def test_array_set_ufunc_many_a(self):
        a1 = np.array([3, 2, 1])
        a2 = np.array([3, 2])
        a3 = np.array([5, 3, 2, 1])
        a4 = np.array([2])

        post = _array_set_ufunc_many((a1, a2, a3, a4), ufunc=np.intersect1d)
        self.assertEqual(post.tolist(), [2])

        post = _array_set_ufunc_many((a1, a2, a3, a4), ufunc=np.union1d)
        self.assertEqual(post.tolist(), [1, 2, 3, 5])


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
    # type blocks

    def test_type_blocks_a(self):

        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([1,2,3])
        a2 = np.array([10,50,30])
        a3 = np.array([1345,2234,3345])
        a4 = np.array([False, True, False])
        a5 = np.array([False, False, False])
        a6 = np.array(['g', 'd', 'e'])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3, a4, a5, a6))

        # can show that with tb2, a6 remains unchanged

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb3 = TypeBlocks.from_blocks((a1, a2, a3))

        # showing that slices keep the same memory location
        # self.assertTrue((tb1[0:2].mloc == tb1.mloc[:2]).all())
        # self.assertTrue((tb1.mloc[:2] == tb1.iloc[0:2, 0:2].mloc).all())

    def test_type_blocks_contiguous_pairs(self):

        a = [(0, 1), (0, 2), (2, 3), (2, 1)]
        post = list(TypeBlocks._indices_to_contiguous_pairs(a))
        self.assertEqual(post, [
                (0, slice(1, 3)),
                (2, slice(3, 4)),
                (2, slice(1, 2)),
                ])

        a = [(0, 0), (0, 1), (0, 2), (1, 4), (2, 1), (2, 3)]
        post = list(TypeBlocks._indices_to_contiguous_pairs(a))
        self.assertEqual(post, [
                (0, slice(0, 3)),
                (1, slice(4, 5)),
                (2, slice(1, 2)),
                (2, slice(3, 4)),
            ])



    def test_type_blocks_b(self):

        # given naturally of a list of rows; this corresponds to what we get with iloc, where we select a row first, then a column
        a1 = np.array([[1,2,3], [4,5,6]])
        # shape is given as rows, columns
        self.assertEqual(a1.shape, (2, 3))

        a2 = np.array([[.2, .5, .4], [.8, .6, .5]])
        a3 = np.array([['a', 'b'], ['c', 'd']])


        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertTypeBlocksArrayEqual(tb1[2], [3, 6])
        self.assertTypeBlocksArrayEqual(tb1[4], [0.5, 0.6])

        self.assertEqual(list(tb1[7].values), ['b', 'd'])

        self.assertEqual(tb1.shape, (2, 8))

        self.assertEqual(len(tb1), 2)
        self.assertEqual(tb1._row_dtype, np.object_)

        slice1 = tb1[2:5]
        self.assertEqual(slice1.shape, (2, 3))

        slice2 = tb1[0:5]
        self.assertEqual(slice2.shape, (2, 5))

        # pick columns
        slice3 = tb1[[2,6,0]]
        self.assertEqual(slice3.shape, (2, 3))

        # TODO: need to implement values

        self.assertEqual(slice3.iloc[0].values.tolist(), [3, 'a', 1])
        self.assertEqual(slice3.iloc[1].values.tolist(), [6, 'c', 4])

        ## slice refers to the same data; not sure if this is accurate test yet

        row1 = tb1.iloc[0].values
        self.assertEqual(row1.dtype, object)
        self.assertEqual(len(row1), 8)
        self.assertEqual(list(row1[:3]), [1, 2, 3])
        self.assertEqual(list(row1[-2:]), ['a', 'b'])

        self.assertEqual(tb1.unified, False)



    def test_type_blocks_c(self):

        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        a4 = np.array(['gd', 'cd', 'dd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3, a4))

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))

        row1 = tb1.iloc[2]

        self.assertEqual(tb1.shape, (3, 4))

        self.assertEqual(tb1.iloc[1].values.tolist(), [2, True, 'c', 'cd'])
        #tb1.iloc[0:2]

        #tb1.iloc[0:2, 0:2]

        #tb1.iloc[0,2]

        #tb1.iloc[0, 0:2]

        self.assertEqual(tb1.iloc[0, 0:2].shape, (1, 2))
        self.assertEqual(tb1.iloc[0:2, 0:2].shape, (2, 2))



    def test_type_blocks_d(self):
        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])


        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertEqual(tb1.iloc[0:2].shape, (2, 8))
        self.assertEqual(tb1.iloc[1:3].shape, (2, 8))

        #tb1.iloc[0, 1:5]


    def test_type_blocks_indices_to_contiguous_pairs(self):

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])


        tb1 = TypeBlocks.from_blocks((a1, a2, a3))
        self.assertEqual(list(tb1._key_to_block_slices(0)), [(0, 0)])
        self.assertEqual(list(tb1._key_to_block_slices(6)), [(2, 0)])
        self.assertEqual(list(tb1._key_to_block_slices([3,5,6])),
            [(1, slice(0, 1, None)), (1, slice(2, 3, None)), (2, slice(0, 1, None))]
            )

        # for rows, all areg grouped by 0
        #self.assertEqual(list(tb1._key_to_block_slices(1, axis=0)), [(0, 1)])
        #self.assertEqual(list(tb1._key_to_block_slices((0,2), axis=0)),
            #[(0, slice(0, 1, None)), (0, slice(2, 3, None))]
            #)
        #self.assertEqual(list(tb1._key_to_block_slices((0,1), axis=0)),
            #[(0, slice(0, 2, None))]
            #)





    def test_type_blocks_extract_a(self):

        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        a4 = np.array(['gd', 'cd', 'dd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3, a4))

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))

        # double point extraction goes to single elements
        self.assertEqual(tb2._extract(1, 3), True)
        self.assertEqual(tb1._extract(1, 2), 'c')

        # single row extraction
        self.assertEqual(tb1._extract(1).shape, (1, 4))
        self.assertEqual(tb1._extract(0).shape, (1, 4))
        self.assertEqual(tb2._extract(0).shape, (1, 8))

        # single column _extractions
        self.assertEqual(tb1._extract(None, 1).shape, (3, 1))
        self.assertEqual(tb2._extract(None, 1).shape, (3, 1))

        # multiple row selection
        self.assertEqual(tb2._extract([1,2],).shape, (2, 8))
        self.assertEqual(tb2._extract([0,2],).shape, (2, 8))
        self.assertEqual(tb2._extract([0,2], 6).shape, (2, 1))
        self.assertEqual(tb2._extract([0,2], [6,7]).shape, (2, 2))

        # mixed
        self.assertEqual(tb2._extract(1,).shape, (1, 8))
        self.assertEqual(tb2._extract([0,2]).shape, (2, 8))
        self.assertEqual(tb2._extract(1, 4), False)
        self.assertEqual(tb2._extract(1, 3), True)
        self.assertEqual(tb2._extract([0, 2],).shape, (2, 8))


        # slices
        self.assertEqual(tb2._extract(slice(1,3)).shape, (2, 8))
        self.assertEqual(tb2._extract(slice(1,3), slice(3,6)).shape, (2,3))
        self.assertEqual(tb2._extract(slice(1,2)).shape, (1,8))
        # a boundry over extended still gets 1
        self.assertEqual(tb2._extract(slice(2,4)).shape, (1,8))
        self.assertEqual(tb2._extract(slice(None), slice(2,4)).shape, (3, 2))
        self.assertEqual(tb1._extract(slice(2,4)).shape, (1, 4))


    def test_type_blocks_extract_b(self):
        # test case of a single unified block

        a1 = np.array([
            [1, 2, 3, -5],
            [10, 50, 30, -7],
            [1345, 2234, 3345, -200]])
        tb1 = TypeBlocks.from_blocks(a1)
        self.assertEqual(tb1.shape, (3, 4))
        self.assertEqual(len(tb1.mloc), 1)

        a1 = np.array([1,10,1345])
        a2 = np.array([2, 50, 2234])
        a3 = np.array([3, 30, 3345])
        a4 = np.array([-5, -7, -200])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3, a4))
        self.assertEqual(tb2.shape, (3, 4))
        self.assertEqual(len(tb2.mloc), 4)


        tb1_a = tb1._extract(row_key=slice(0,1))
        tb2_a = tb2._extract(row_key=slice(0,1))
        self.assertEqual(tb1_a.shape, tb2_a.shape)
        self.assertTrue((tb1_a.values == tb2_a.values).all())


        tb1_b = tb1._extract(row_key=slice(1))
        tb2_b = tb2._extract(row_key=slice(1))

        self.assertEqual(tb1_b.shape, tb2_b.shape)
        self.assertTrue((tb1_b.values == tb2_b.values).all())


        tb1_c = tb1._extract(row_key=slice(0, 2))
        tb2_c = tb2._extract(row_key=slice(0, 2))

        self.assertEqual(tb1_c.shape, tb2_c.shape)
        self.assertTrue((tb1_c.values == tb2_c.values).all())

        tb1_d = tb1._extract(row_key=slice(0, 2), column_key=3)
        tb2_d = tb2._extract(row_key=slice(0, 2), column_key=3)

        self.assertEqual(tb1_d.shape, tb2_d.shape)
        self.assertTrue((tb1_d.values == tb2_d.values).all())

        tb1_e = tb1._extract(row_key=slice(0, 2), column_key=slice(2,4))
        tb2_e = tb2._extract(row_key=slice(0, 2), column_key=slice(2,4))

        self.assertEqual(tb1_e.shape, tb2_e.shape)
        self.assertTrue((tb1_e.values == tb2_e.values).all())

        self.assertTrue(tb1._extract(row_key=2, column_key=2) ==
                tb2._extract(row_key=2, column_key=2) ==
                3345)

    def test_type_blocks_extract_c(self):
        # test negative slices

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))


        # test negative row slices
        self.assertTypeBlocksArrayEqual(
                tb1._extract(-1),
                [0, 0, 1, True, False, True, 'oe', 'od'],
                match_dtype=object
                )
        self.assertTypeBlocksArrayEqual(
                tb1._extract(slice(-2, None)),
                [[4, 5, 6, True, False, True, 'c', 'd'],
                [0, 0, 1, True, False, True, 'oe', 'od']],
                match_dtype=object
                )
        self.assertTypeBlocksArrayEqual(
                tb1._extract(slice(-3, -1)),
                [[1, 2, 3, False, False, True, 'a', 'b'],
                [4, 5, 6, True, False, True, 'c', 'd']],
                match_dtype=object
                )
        self.assertTypeBlocksArrayEqual(
                tb1._extract(slice(None), -2),
                ['a', 'c', 'oe'],
                match_dtype=object
                )
        self.assertTypeBlocksArrayEqual(
                tb1._extract(slice(None), slice(-6, -1)),
                [[3, False, False, True, 'a'],
                [6, True, False, True, 'c'],
                [1, True, False, True, 'oe']],
                match_dtype=object
                )

        self.assertTypeBlocksArrayEqual(
                tb1._extract(slice(None), slice(-1, -4, -1)),
                [['b', 'a', True],
                ['d', 'c', True],
                ['od', 'oe', True]],
                match_dtype=object)


    def test_type_blocks_extract_array_a(self):
        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a4 = tb1._extract_array(row_key=1)
        self.assertEqual(a4.tolist(),
                [4, 5, 6, True, False, True, 'c', 'd'])

        a5 = tb1._extract_array(column_key=5)
        self.assertEqual(a5.tolist(),
                [True, True, True])


    def test_immutable_filter(self):
        a1 = np.array([3, 4, 5])
        a2 = TypeBlocks.immutable_filter(a1)
        with self.assertRaises(ValueError):
            a2[0] = 34
        a3 = a2[:2]
        with self.assertRaises(ValueError):
            a3[0] = 34


    def test_type_blocks_static_frame(self):
        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertTrue(tb1.dtypes[0] == np.int64)


    def test_type_blocks_attributes(self):

        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertEqual(tb1.size, 9)
        self.assertEqual(tb2.size, 24)




    def test_type_blocks_block_pointers(self):

        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertTrue((tb1[0:2].mloc == tb1.mloc[:2]).all())
        self.assertTrue((tb1.mloc[:2] == tb1.iloc[0:2, 0:2].mloc).all())


    def test_type_blocks_append(self):
        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        tb1 = TypeBlocks.from_blocks((a1, a2))
        self.assertTrue(tb1.shape, (3, 2))

        tb1.append(np.array((3,5,4)))
        self.assertTrue(tb1.shape, (3, 3))

        tb1.append(np.array([(3,5),(4,6),(5,10)]))
        self.assertTrue(tb1.shape, (3, 5))

        self.assertEqual(tb1.iloc[0].values.tolist(), [1, False, 3, 3, 5])
        self.assertEqual(tb1.iloc[1].values.tolist(), [2, True, 5, 4, 6])
        self.assertEqual(tb1.iloc[:, 3].values.tolist(), [3, 4, 5])




    def test_type_blocks_unary_operator_a(self):

        a1 = np.array([1,-2,-3])
        a2 = np.array([False, True, False])
        tb1 = TypeBlocks.from_blocks((a1, a2))

        tb2 = ~tb1 # tilde
        self.assertEqual(
            (~tb1.values).tolist(),
            [[-2, -1], [1, -2], [2, -1]])

    def test_type_blocks_unary_operator_b(self):

        a1 = np.array([[1,2,3], [-4,5,6], [0,0,-1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertTypeBlocksArrayEqual(
                -tb2[0:3],
                [[-1, -2, -3],
                 [ 4, -5, -6],
                 [ 0,  0,  1]],
                )

        self.assertTypeBlocksArrayEqual(
                ~tb2[3:5],
                [[ True,  True],
                [False,  True],
                [False,  True]],
                )



    def test_type_blocks_block_compatible_a(self):

        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))

        # get slices with unified types
        tb3 = tb1[[0, 1]]
        tb4 = tb2[[2, 3]]


        self.assertTrue(tb3.block_compatible(tb4))
        self.assertTrue(tb4.block_compatible(tb3))

        self.assertFalse(tb1.block_compatible(tb2))
        self.assertFalse(tb2.block_compatible(tb1))


    #@unittest.skip('to fix')
    def test_type_blocks_block_compatible_b(self):

        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))

        tb2a = tb2[[2,3,7]]
        self.assertTrue(tb1.block_compatible(tb2a))



    def test_type_blocks_consolidate_a(self):

        a1 = np.array([1,2,3])
        a2 = np.array([10,50,30])
        a3 = np.array([1345,2234,3345])

        a4 = np.array([False, True, False])
        a5 = np.array([False, False, False])

        tb1 = TypeBlocks.from_blocks((a1, a2, a3, a4, a5))
        self.assertEqual(tb1.shape, (3, 5))

        tb2 = tb1.consolidate()

        self.assertTrue([b.dtype for b in tb2._blocks], [np.int, np.bool])
        self.assertEqual(tb2.shape, (3, 5))

        # we have perfect correspondence between the two
        self.assertTrue((tb1.values == tb2.values).all())


    def test_type_blocks_consolidate_b(self):
        # if we hava part of TB consolidated, we do not reallocate


        a1 = np.array([
            [1,2,3],
            [10,50,30],
            [1345,2234,3345]])

        a2 = np.array([False, True, False])
        a3 = np.array([False, False, False])

        tb1 = TypeBlocks.from_blocks((a1, a2, a3))
        self.assertEqual(tb1.shape, (3, 5))
        self.assertEqual(len(tb1.mloc), 3)

        tb2 = tb1.consolidate()
        self.assertEqual(tb1.shape, (3, 5))
        self.assertEqual(len(tb2.mloc), 2)
        # the first block is the same instance
        self.assertEqual(tb1.mloc[0], tb2.mloc[0])




    def test_type_blocks_binary_operator_a(self):

        a1 = np.array([
            [1, 2, 3, -5],
            [10, 50, 30, -7],
            [1345, 2234, 3345, -200]])
        tb1 = TypeBlocks.from_blocks(a1)

        a1 = np.array([1,10,1345])
        a2 = np.array([2, 50, 2234])
        a3 = np.array([3, 30, 3345])
        a4 = np.array([-5, -7, -200])
        tb2 = TypeBlocks.from_blocks((a1, a2, a3, a4))

        self.assertTrue(((tb1 + tb2).values == (tb1 + tb1).values).all())

        post1 = tb1 + tb2



    def test_type_blocks_binary_operator_b(self):

        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']], dtype=object)
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))


        self.assertTypeBlocksArrayEqual(
                tb2 * 3,
                [[3, 6, 9, 0, 0, 3, 'aaa', 'bbb'],
                [12, 15, 18, 3, 0, 3, 'ccc', 'ddd'],
                [0, 0, 3, 3, 0, 3, 'oeoeoe', 'ododod']],
                match_dtype=object
                )

        self.assertTypeBlocksArrayEqual(
                tb1[:2] + 10,
                [[11, 10],
                [12, 11],
                [13, 10]],
                )

        self.assertTypeBlocksArrayEqual(
                tb1[:2] + 10,
                [[11, 10],
                [12, 11],
                [13, 10]],
                )


    def test_type_blocks_binary_operator_c(self):
        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']], dtype=object)
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))

        # these result in the same thing
        self.assertTypeBlocksArrayEqual(
                tb1[:2] * tb2[[2,3]],
                [[3, False],
                [12, True],
                [3, False]]
                )

        self.assertTypeBlocksArrayEqual(
                tb1[0:2] * tb1[0:2],
                [[1, False],
                [4, True],
                [9, False]]
                )

        self.assertTypeBlocksArrayEqual(
                tb2[:3] % 2,
                [[1, 0, 1],
                [0, 1, 0],
                [0, 0, 1]]
            )


    def test_type_blocks_binary_operator_d(self):
        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[1.5,2.6], [4.2,5.5], [0.2,0.1]])
        tb1 = TypeBlocks.from_blocks((a1, a2))

        post = tb1 * (1, 0, 2, 0, 1)
        self.assertTypeBlocksArrayEqual(post,
                [[  1. ,   0. ,   6. ,   0. ,   2.6],
                [  4. ,   0. ,  12. ,   0. ,   5.5],
                [  0. ,   0. ,   2. ,   0. ,   0.1]])



    def test_type_blocks_extend_a(self):
        a1 = np.array([1,2,3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']], dtype=object)
        tb2 = TypeBlocks.from_blocks((a1, a2, a3))

        # mutates in place
        tb1.extend(tb2)
        self.assertEqual(tb1.shape, (3, 11))

        self.assertTypeBlocksArrayEqual(
                tb1.iloc[2],
                [3, False, 'd', 0, 0, 1, True, False, True, 'oe', 'od'],
                match_dtype=object,
                )



    def test_type_blocks_mask_blocks_a(self):
        # test negative slices

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        mask = TypeBlocks.from_blocks(tb1._mask_blocks(column_key=[2,3,5,6]))

        self.assertTypeBlocksArrayEqual(mask,
            [[False, False, True, True, False, True, True, False], [False, False, True, True, False, True, True, False], [False, False, True, True, False, True, True, False]]
            )



    def test_type_blocks_assign_blocks_a(self):
        # test negative slices

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        tb2 = TypeBlocks.from_blocks(tb1._assign_blocks_from_keys(column_key=[2,3,5], value=300))

        self.assertTypeBlocksArrayEqual(tb2,
            [[1, 2, 300, 300, False, 300, 'a', 'b'],
            [4, 5, 300, 300, False, 300, 'c', 'd'],
            [0, 0, 300, 300, False, 300, 'oe', 'od']], match_dtype=object)

        # blocks not mutated will be the same
        self.assertEqual(tb1.mloc[2], tb2.mloc[2])

    def test_type_blocks_group_a(self):

        a1 = np.array([
                [1,2,3,4],
                [4,2,6,3],
                [0,0,1,2],
                [0,0,1,1]
                ])
        a2 = np.array([[False, False, True],
                [False, False, True],
                [True, False, True],
                [True, False, True]])

        tb1 = TypeBlocks.from_blocks((a1, a2))

        # return rows, by columns key 1
        groups = list(tb1.group(axis=0, key=1))

        self.assertEqual(len(groups), 2)


        group, selection, subtb = groups[0]
        self.assertEqual(group, 0)
        self.assertEqual(subtb.values.tolist(),
                [[0, 0, 1, 2, True, False, True], [0, 0, 1, 1, True, False, True]])


        group, selection, subtb = groups[1]
        self.assertEqual(group, 2)
        self.assertEqual(subtb.values.tolist(),
                [[1, 2, 3, 4, False, False, True], [4, 2, 6, 3, False, False, True]])


    def test_type_blocks_group_b(self):

        a1 = np.array([
                [1,2,3,4],
                [4,2,6,3],
                [0,0,1,2],
                [0,0,1,1]
                ])
        a2 = np.array([[False, False, True],
                [False, False, True],
                [True, False, True],
                [True, False, True]])

        tb1 = TypeBlocks.from_blocks((a1, a2))

        # return rows, by columns key [4, 5]
        groups = list(tb1.group(axis=0, key=[4, 5]))

        self.assertEqual(len(groups), 2)


        group, selection, subtb = groups[0]
        self.assertEqual(group.tolist(), [False, False])
        self.assertEqual(subtb.values.tolist(),
                [[1, 2, 3, 4, False, False, True], [4, 2, 6, 3, False, False, True]]
                )

        group, selection, subtb = groups[1]
        self.assertEqual(group.tolist(), [True, False])
        self.assertEqual(subtb.values.tolist(),
                [[0, 0, 1, 2, True, False, True], [0, 0, 1, 1, True, False, True]])
        # TODO: add more tests here


    def test_type_blocks_transpose_a(self):

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        tb2 = tb1.transpose()

        self.assertEqual(tb2.values.tolist(),
                [[1, 4, 0], [2, 5, 0], [3, 6, 1], [False, True, True], [False, False, False], [True, True, True], ['a', 'c', 'oe'], ['b', 'd', 'od']])

        self.assertEqual(tb1.transpose().transpose().values.tolist(),
                tb1.values.tolist())



    def test_type_blocks_display_a(self):

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb = TypeBlocks.from_blocks((a1, a2, a3))

        disp = tb.display()
        self.assertEqual(len(disp), 5)
        # self.assertEqual(list(disp),
        #     [['<TypeBlocks> ', '                  ', '          '],
        #     ['1 2 3        ', 'False False  True ', "'a' 'b'   "],
        #     ['4 5 6        ', ' True False  True ', "'c' 'd'   "],
        #     ['0 0 1        ', ' True False  True ', "'oe' 'od' "],
        #     ['int64        ', 'bool              ', '<U2       ']])


    def test_type_blocks_axis_values_a(self):
        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertEqual(
            list(a.tolist() for a in tb.axis_values(1)),
            [[1, 2, 3, False, False, True, 'a', 'b'], [4, 5, 6, True, False, True, 'c', 'd'], [0, 0, 1, True, False, True, 'oe', 'od']]
            )

        self.assertEqual(list(a.tolist() for a in tb.axis_values(0)),
            [[1, 4, 0], [2, 5, 0], [3, 6, 1], [False, True, True], [False, False, False], [True, True, True], ['a', 'c', 'oe'], ['b', 'd', 'od']]
            )

        # we are iterating over slices so we get views of columns without copying
        self.assertEqual(tb.mloc[0], mloc(next(tb.axis_values(0))))


    def test_type_blocks_axis_values_b(self):
        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertEqual(
                [a.tolist() for a in tb.axis_values(axis=0, reverse=True)],
                [['b', 'd', 'od'], ['a', 'c', 'oe'], [True, True, True], [False, False, False], [False, True, True], [3, 6, 1], [2, 5, 0], [1, 4, 0]])
        self.assertEqual([a.tolist() for a in tb.axis_values(axis=0, reverse=False)],
                [[1, 4, 0], [2, 5, 0], [3, 6, 1], [False, True, True], [False, False, False], [True, True, True], ['a', 'c', 'oe'], ['b', 'd', 'od']])


    def test_type_blocks_extract_iloc_mask_a(self):

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb = TypeBlocks.from_blocks((a1, a2, a3))

        self.assertEqual(tb.extract_iloc_mask((slice(None), [4, 5])).values.tolist(),
                [[False, False, False, False, True, True, False, False], [False, False, False, False, True, True, False, False], [False, False, False, False, True, True, False, False]])

        self.assertEqual(tb.extract_iloc_mask(([0,2], slice(None))).values.tolist(),
                [[True, True, True, True, True, True, True, True], [False, False, False, False, False, False, False, False], [True, True, True, True, True, True, True, True]]
                )

        self.assertEqual(tb.extract_iloc_mask(([0,2], [3,7])).values.tolist(),
                [[False, False, False, True, False, False, False, True], [False, False, False, False, False, False, False, False], [False, False, False, True, False, False, False, True]]
                )

        self.assertEqual(tb.extract_iloc_mask((slice(1, None), slice(4, None))).values.tolist(),
                [[False, False, False, False, False, False, False, False], [False, False, False, False, True, True, True, True], [False, False, False, False, True, True, True, True]])


    def test_type_blocks_extract_iloc_assign_a(self):

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        tb = TypeBlocks.from_blocks((a1, a2, a3))


        self.assertEqual(tb.extract_iloc_assign(1, 600).values.tolist(),
                [[1, 2, 3, False, False, True, 'a', 'b'], [600, 600, 600, 600, 600, 600, 600, 600], [0, 0, 1, True, False, True, 'oe', 'od']])

        self.assertEqual(tb.extract_iloc_assign((1, 5), 20).values.tolist(),
                [[1, 2, 3, False, False, True, 'a', 'b'], [4, 5, 6, True, False, 20, 'c', 'd'], [0, 0, 1, True, False, True, 'oe', 'od']])

        self.assertEqual(tb.extract_iloc_assign((slice(2), slice(5)), 'X').values.tolist(),
                [['X', 'X', 'X', 'X', 'X', True, 'a', 'b'], ['X', 'X', 'X', 'X', 'X', True, 'c', 'd'], [0, 0, 1, True, False, True, 'oe', 'od']]
                )

        self.assertEqual(tb.extract_iloc_assign(([0,1], [1,4,7]), -5).values.tolist(),
                [[1, -5, 3, False, -5, True, 'a', -5], [4, -5, 6, True, -5, True, 'c', -5], [0, 0, 1, True, False, True, 'oe', 'od']])


        self.assertEqual(
                tb.extract_iloc_assign((1, slice(4)), (-1, -2, -3, -4)).values.tolist(),
                [[1, 2, 3, False, False, True, 'a', 'b'], [-1, -2, -3, -4, False, True, 'c', 'd'], [0, 0, 1, True, False, True, 'oe', 'od']])

        self.assertEqual(
                tb.extract_iloc_assign((2, slice(3,7)), (-1, -2, -3, -4)).values.tolist(),
                [[1, 2, 3, False, False, True, 'a', 'b'], [4, 5, 6, True, False, True, 'c', 'd'], [0, 0, 1, -1, -2, -3, -4, 'od']])
        self.assertEqual(
                tb.extract_iloc_assign((0, slice(4,8)), (-1, -2, -3, -4)).values.tolist(),
                [[1, 2, 3, False, -1, -2, -3, -4], [4, 5, 6, True, False, True, 'c', 'd'], [0, 0, 1, True, False, True, 'oe', 'od']])


    def test_type_blocks_elements_items_a(self):

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        a4 = np.array([None, None, None])
        tb = TypeBlocks.from_blocks((a1, a2, a4, a3))

        post = [x for x in tb.element_items()]

        self.assertEqual(post,
                [((0, 0), 1), ((0, 1), 2), ((0, 2), 3), ((0, 3), False), ((0, 4), False), ((0, 5), True), ((0, 6), None), ((0, 7), 'a'), ((0, 8), 'b'), ((1, 0), 4), ((1, 1), 5), ((1, 2), 6), ((1, 3), True), ((1, 4), False), ((1, 5), True), ((1, 6), None), ((1, 7), 'c'), ((1, 8), 'd'), ((2, 0), 0), ((2, 1), 0), ((2, 2), 1), ((2, 3), True), ((2, 4), False), ((2, 5), True), ((2, 6), None), ((2, 7), 'oe'), ((2, 8), 'od')]
                )

        tb2 = TypeBlocks.from_element_items(post, tb.shape, tb._row_dtype)
        self.assertTrue((tb.values == tb2.values).all())


    def test_type_blocks_reblock_signature_a(self):

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        a4 = np.array([None, None, None])
        tb = TypeBlocks.from_blocks((a1, a2, a4, a3))

        dtype = np.dtype
        self.assertEqual(
                list(tb._reblock_signature()),
                [(dtype('int64'), 3), (dtype('bool'), 3), (dtype('O'), 1), (dtype('<U2'), 2)])

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        a4 = np.array([None, None, None])
        tb = TypeBlocks.from_blocks((a1, a2, a3, a4))

        self.assertEqual(
                list(tb._reblock_signature()),
                [(dtype('int64'), 3), (dtype('bool'), 3), (dtype('<U2'), 2), (dtype('O'), 1)])



#     @unittest.skip('implement operators for same sized but differently typed blocks')
    def test_type_blocks_binary_operator_e(self):

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        a4 = np.array([None, None, None])
        tb = TypeBlocks.from_blocks((a1, a2, a4, a3))

        post = [x for x in tb.element_items()]

        tb2 = TypeBlocks.from_element_items(post, tb.shape, tb._row_dtype)
        self.assertTrue((tb.values == tb2.values).all())

        post = tb == tb2
        self.assertEqual(post.values.tolist(),
                [[True, True, True, True, True, True, True, True, True], [True, True, True, True, True, True, True, True, True], [True, True, True, True, True, True, True, True, True]])


    def test_type_blocks_copy_a(self):

        a1 = np.array([[1,2,3], [4,5,6], [0,0,1]])
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        a4 = np.array([None, None, None])
        tb1 = TypeBlocks.from_blocks((a1, a2, a4, a3))

        tb2 = tb1.copy()
        tb1.append(np.array((1,2,3)))

        self.assertEqual(tb2.shape, (3, 9))
        self.assertEqual(tb1.shape, (3, 10))

        self.assertEqual(tb1.iloc[2].values.tolist(),
                [0, 0, 1, True, False, True, None, 'oe', 'od', 3])

        self.assertEqual(tb2.iloc[2].values.tolist(),
                [0, 0, 1, True, False, True, None, 'oe', 'od'])




    def test_type_blocks_isna_a(self):

        a1 = np.array([[1,2,3], [4, np.nan, 6], [0,0,1]], dtype=object)
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        a4 = np.array([None, None, None])
        tb1 = TypeBlocks.from_blocks((a1, a2, a4, a3))

        self.assertEqual(tb1.isna().values.tolist(),
                [[False, False, False, False, False, False, True, False, False], [False, True, False, False, False, False, True, False, False], [False, False, False, False, False, False, True, False, False]])

        self.assertEqual(tb1.notna().values.tolist(),
                [[True, True, True, True, True, True, False, True, True], [True, False, True, True, True, True, False, True, True], [True, True, True, True, True, True, False, True, True]])


    def test_type_blocks_dropna_to_slices(self):

        a1 = np.array([
                [1,np.nan,3, 4],
                [4, np.nan, 6, 2],
                [np.nan, np.nan, np.nan, np.nan]
                ], dtype=object)
        a2 = np.array([
                [1,np.nan,3, 4],
                [4, np.nan, 6, 2],
                [np.nan, np.nan, np.nan, np.nan]
                ], dtype=object)

        tb1 = TypeBlocks.from_blocks((a1, a2))

        row_key, column_key = tb1.dropna_to_keep_locations(axis=1)

        self.assertEqual(column_key.tolist(),
                [True, False, True, True, True, False, True, True])
        self.assertEqual(row_key, None)

        row_key, column_key = tb1.dropna_to_keep_locations(axis=0)
        self.assertEqual(row_key.tolist(),
                [True, True, False])

        self.assertEqual(column_key, None)


    def test_type_blocks_fillna_a(self):

        a1 = np.array([
                [1,np.nan,3, 4],
                [4, np.nan, 6, 2],
                [np.nan, np.nan, np.nan, np.nan]
                ], dtype=float)
        a2 = np.array([
                [1,np.nan,3, 4],
                [4, np.nan, 6, 2],
                [np.nan, np.nan, np.nan, np.nan]
                ], dtype=object)

        tb1 = TypeBlocks.from_blocks((a1, a2))
        tb2 = tb1.fillna(0)
        self.assertEqual([b.dtype for b in tb2._blocks],
                [np.dtype('float64'), np.dtype('O')])
        self.assertEqual(tb2.isna().values.any(), False)

        tb3 = tb1.fillna(None)
        self.assertEqual([b.dtype for b in tb3._blocks],
                [np.dtype('O'), np.dtype('O')])
        # we ahve Nones, which are na
        self.assertEqual(tb3.isna().values.any(), True)


    def test_type_blocks_from_none_a(self):

        a1 = np.array([[1,2,3], [4, np.nan, 6], [0,0,1]], dtype=object)
        a2 = np.array([[False, False, True], [True, False, True], [True, False, True]])
        a3 = np.array([['a', 'b'], ['c', 'd'], ['oe', 'od']])
        a4 = np.array([None, None, None])

        tb1 = TypeBlocks.from_none()
        tb1.append(a1)
        self.assertEqual(tb1.shape, (3, 3))
        tb1.append(a4)
        self.assertEqual(tb1.shape, (3, 4))

        tb1 = TypeBlocks.from_none()
        tb1.append(a4)
        self.assertEqual(tb1.shape, (3, 1))
        tb1.append(a1)
        self.assertEqual(tb1.shape, (3, 4))


    #---------------------------------------------------------------------------
    # index tests

    def test_index_loc_to_iloc_a(self):

        idx = Index(('a', 'b', 'c', 'd'))

        self.assertEqual(
                idx.loc_to_iloc(np.array([True, False, True, False])).tolist(),
                [0, 2])

        self.assertEqual(idx.loc_to_iloc(slice('c',)), slice(None, 3, None))
        self.assertEqual(idx.loc_to_iloc(slice('b','d')), slice(1, 4, None))
        self.assertEqual(idx.loc_to_iloc('d'), 3)


    def test_index_mloc_a(self):
        idx = Index(('a', 'b', 'c', 'd'))
        self.assertTrue(idx.mloc == idx[:2].mloc)


    def test_index_unique(self):

        with self.assertRaises(KeyError):
            idx = Index(('a', 'b', 'c', 'a'))
        with self.assertRaises(KeyError):
            idx = IndexGO(('a', 'b', 'c', 'a'))

        with self.assertRaises(KeyError):
            idx = Index(['a', 'a'])
        with self.assertRaises(KeyError):
            idx = IndexGO(['a', 'a'])

        with self.assertRaises(KeyError):
            idx = Index(np.array([True, False, True], dtype=bool))
        with self.assertRaises(KeyError):
            idx = IndexGO(np.array([True, False, True], dtype=bool))

        # acceptable but not advisiable
        idx = Index([0, '0'])


    def test_index_creation_a(self):
        idx = Index(('a', 'b', 'c', 'd'))

        #idx2 = idx['b':'d']

        self.assertEqual(idx.values.tolist(), ['a', 'b', 'c', 'd'])

        self.assertEqual(idx[2:].values.tolist(), ['c', 'd'])

        self.assertEqual(idx.loc['b':].values.tolist(), ['b', 'c', 'd'])

        self.assertEqual(idx.loc['b':'d'].values.tolist(), ['b', 'c', 'd'])

        self.assertEqual(idx.loc_to_iloc(['b', 'b', 'c']), [1, 1, 2])

        self.assertEqual(idx.loc['c'].values.tolist(), ['c'])



        idxgo = IndexGO(('a', 'b', 'c', 'd'))
        self.assertEqual(idxgo.values.tolist(), ['a', 'b', 'c', 'd'])

        idxgo.append('e')
        self.assertEqual(idxgo.values.tolist(), ['a', 'b', 'c', 'd', 'e'])

        idxgo.extend(('f', 'g'))
        self.assertEqual(idxgo.values.tolist(), ['a', 'b', 'c', 'd', 'e', 'f', 'g'])



    def test_index_creation_b(self):
        idx = Index((x for x in ('a', 'b', 'c', 'd') if x in {'b', 'd'}))
        self.assertEqual(idx.loc_to_iloc('b'), 0)
        self.assertEqual(idx.loc_to_iloc('d'), 1)


    def test_index_unary_operators_a(self):
        idx = Index((20, 30, 40, 50))

        invert_idx = -idx
        self.assertEqual(invert_idx.tolist(),
                [-20, -30, -40, -50],)

        # this is strange but consistent with NP
        not_idx = ~idx
        self.assertEqual(not_idx.tolist(),
                [-21, -31, -41, -51],)

    def test_index_binary_operators_a(self):
        idx = Index((20, 30, 40, 50))

        self.assertEqual((idx + 2).tolist(),
                [22, 32, 42, 52])
        self.assertEqual((2 + idx).tolist(),
                [22, 32, 42, 52])
        self.assertEqual((idx * 2).tolist(),
                [40, 60, 80, 100])
        self.assertEqual((2 * idx).tolist(),
                [40, 60, 80, 100])
        self.assertEqual((idx - 2).tolist(),
                [18, 28, 38, 48])
        self.assertEqual(
                (2 - idx).tolist(),
                [-18, -28, -38, -48])


    def test_index_binary_operators_b(self):
        '''Both opperands are Index instances
        '''
        idx1 = Index((20, 30, 40, 50))
        idx2 = Index((20, 3, 4, 5))

        self.assertEqual((idx1 == idx2).tolist(), [True, False, False, False])



    def test_index_ufunc_axis_a(self):

        idx = Index((30, 40, 50))

        self.assertEqual(idx.min(), 30)
        self.assertEqual(idx.max(), 50)
        self.assertEqual(idx.sum(), 120)

    def test_index_isin_a(self):

        idx = Index((30, 40, 50))

        self.assertEqual(idx.isin([40, 50]).tolist(), [False, True, True])
        self.assertEqual(idx.isin({40, 50}).tolist(), [False, True, True])

        self.assertEqual(idx.isin(frozenset((40, 50))).tolist(), [False, True, True])

        self.assertEqual(idx.isin({40: 'a', 50: 'b'}).tolist(), [False, True, True])

        self.assertEqual(idx.isin(range(35, 45)).tolist(), [False, True, False])

        self.assertEqual(idx.isin((x * 10 for x in (3, 4, 5, 6, 6))).tolist(), [True, True, True])



    def test_index_contains_a(self):

        index = Index(('a', 'b', 'c'))
        self.assertTrue('a' in index)
        self.assertTrue('d' not in index)


    def test_index_grow_only_a(self):

        index = IndexGO(('a', 'b', 'c'))
        index.append('d')
        self.assertEqual(index.loc_to_iloc('d'), 3)

        index.extend(('e', 'f'))
        self.assertEqual(index.loc_to_iloc('e'), 4)
        self.assertEqual(index.loc_to_iloc('f'), 5)

        # creating an index form an Index go takes the np arrays, but not the mutable bits
        index2 = Index(index)
        index.append('h')

        self.assertEqual(len(index2), 6)
        self.assertEqual(len(index), 7)


    def test_index_sort(self):

        index = Index(('a', 'c', 'd', 'e', 'b'))
        self.assertEqual(
                [index.sort().loc_to_iloc(x) for x in sorted(index.values)],
                [0, 1, 2, 3, 4])
        self.assertEqual(
                [index.sort(ascending=False).loc_to_iloc(x) for x in sorted(index.values)],
                [4, 3, 2, 1, 0])


    def test_index_relable(self):

        index = Index(('a', 'c', 'd', 'e', 'b'))

        self.assertEqual(
                index.relabel(lambda x: x.upper()).values.tolist(),
                ['A', 'C', 'D', 'E', 'B'])

        # letter to number
        s1 = Series(range(5), index=index.values)

        self.assertEqual(
                index.relabel(s1).values.tolist(),
                [0, 1, 2, 3, 4]
                )

        self.assertEqual(index.relabel({'e': 'E'}).values.tolist(),
                ['a', 'c', 'd', 'E', 'b'])


#-------------------------------------------------------------------------------

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


    #---------------------------------------------------------------------------
    # frame tests



    def test_frame_init_a(self):

        f = Frame(dict(a=(1,2), b=(3,4)), index=('x', 'y'))
        self.assertEqual(f.to_pairs(0),
                (('a', (('x', 1), ('y', 2))), ('b', (('x', 3), ('y', 4))))
                )

        f = Frame(dict(b=(3,4), a=(1,2)), index=('x', 'y'))
        self.assertEqual(f.to_pairs(0),
                (('a', (('x', 1), ('y', 2))), ('b', (('x', 3), ('y', 4))))
                )

        f = Frame(OrderedDict([('b', (3,4)), ('a', (1,2))]), index=('x', 'y'))
        self.assertEqual(f.to_pairs(0),
                (('b', (('x', 3), ('y', 4))), ('a', (('x', 1), ('y', 2)))))


    def test_frame_from_pairs_a(self):

        frame = Frame.from_items(sorted(dict(a=[3,4,5], b=[6,3,2]).items()))
        self.assertEqual(
            list((k, list(v.items())) for k, v in frame.items()),
            [('a', [(0, 3), (1, 4), (2, 5)]), ('b', [(0, 6), (1, 3), (2, 2)])])

        frame = Frame.from_items(OrderedDict((('b', [6,3,2]), ('a', [3,4,5]))).items())
        self.assertEqual(list((k, list(v.items())) for k, v in frame.items()),
            [('b', [(0, 6), (1, 3), (2, 2)]), ('a', [(0, 3), (1, 4), (2, 5)])])


    def test_frame_getitem_a(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        # # show that block hav ebeen consolidated
        # self.assertEqual(len(f1._blocks._blocks), 3)

        # s1 = f1['s']
        # self.assertTrue((s1.index == f1.index).all())

        # # we have not copied the index array
        # self.assertEqual(mloc(f1.index.values), mloc(s1.index.values))

        f2 = f1['r':]
        self.assertEqual(f2.columns.values.tolist(), ['r', 's', 't'])
        self.assertTrue((f2.index == f1.index).all())
        self.assertEqual(mloc(f2.index.values), mloc(f1.index.values))




    def test_frame_length_a(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        self.assertEqual(len(f1), 2)



    def test_frame_iloc_a(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        self.assertEqual((f1.iloc[0].values == f1.loc['x'].values).all(), True)
        self.assertEqual((f1.iloc[1].values == f1.loc['y'].values).all(), True)


    def test_frame_iloc_b(self):
        # this is example dervied from this question:
        # https://stackoverflow.com/questions/22927181/selecting-specific-rows-and-columns-from-numpy-array

        a = np.arange(20).reshape((5,4))
        f1 = FrameGO(a)
        a[1,1] = 3000 # ensure we made a copy
        self.assertEqual(f1.loc[[0,1,3], [0,2]].values.tolist(),
                [[0, 2], [4, 6], [12, 14]])
        self.assertEqual(f1.iloc[[0,1,3], [0,2]].values.tolist(),
                [[0, 2], [4, 6], [12, 14]])

        self.assertTrue(f1._index._loc_is_iloc)
        self.assertTrue(f1._columns._loc_is_iloc)

        f1[4] = list(range(5))
        self.assertTrue(f1._columns._loc_is_iloc)

        f1[20] = list(range(5))
        self.assertFalse(f1._columns._loc_is_iloc)

        self.assertEqual(f1.values.tolist(),
                [[0, 1, 2, 3, 0, 0],
                [4, 5, 6, 7, 1, 1],
                [8, 9, 10, 11, 2, 2],
                [12, 13, 14, 15, 3, 3],
                [16, 17, 18, 19, 4, 4]])


    def test_frame_iter_a(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        self.assertEqual((f1.keys() == f1.columns).all(), True)
        self.assertEqual([x for x in f1.columns], ['p', 'q', 'r', 's', 't'])
        self.assertEqual([x for x in f1], ['p', 'q', 'r', 's', 't'])




    def test_frame_iter_array_a(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        self.assertEqual(
                next(iter(f1.iter_array(axis=0))).tolist(),
                [1, 30])

        self.assertEqual(
                next(iter(f1.iter_array(axis=1))).tolist(),
                [1, 2, 'a', False, True])


    def test_frame_iter_array_b(self):

        # thest of multi threaded apply

        f1 = Frame.from_items(
                zip(range(100), (np.random.rand(1000) for _ in range(100)))
                )
        # iter columns
        post = f1.iter_array(0).apply(np.sum, max_workers=4, use_threads=True)
        self.assertEqual(post.shape, (100,))




    def test_frame_setitem_a(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = FrameGO.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        f1['a'] = (False, True)
        self.assertEqual(f1['a'].values.tolist(), [False, True])

        # test index alginment
        f1['b'] = Series((3,2,5), index=('y', 'x', 'g'))
        self.assertEqual(f1['b'].values.tolist(), [2, 3])

        f1['c'] = Series((300,200,500), index=('y', 'j', 'k'))
        self.assertAlmostEqualItems(f1['c'].items(), [('x', nan), ('y', 300)])


    def test_frame_setitem_b(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = FrameGO.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        f1['u'] = 0

        self.assertEqual(f1.loc['x'].values.tolist(),
                [1, 2, 'a', False, True, 0])

        with self.assertRaises(Exception):
            f1['w'] = [[1,2], [4,5]]



    def test_frame_extend_columns_a(self):
        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = FrameGO.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        columns = OrderedDict(
            (('c', np.array([0, -1])), ('d', np.array([3, 5]))))

        f1.extend_columns(columns.keys(), columns.values())

        self.assertEqual(f1.columns.values.tolist(),
                ['p', 'q', 'r', 's', 't', 'c', 'd'])

        self.assertTypeBlocksArrayEqual(f1._blocks,
                [[1, 2, 'a', False, True, 0, 3],
                [30, 50, 'b', True, False, -1, 5]],
                match_dtype=object)

    def test_frame_extend_blocks_a(self):
        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))
        f1 = FrameGO.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        blocks = ([[50, 40], [30, 20]], [[50, 40], [30, 20]])
        columns = ('a', 'b', 'c', 'd')
        f1.extend_blocks(columns, blocks)

        self.assertEqual(f1.columns.values.tolist(),
                ['p', 'q', 'r', 's', 't', 'a', 'b', 'c', 'd'])

        self.assertEqual(f1.values.tolist(),
                [[1, 2, 'a', False, True, 50, 40, 50, 40],
                [30, 50, 'b', True, False, 30, 20, 30, 20]]
                )


    def test_frame_extract_a(self):
        # reindex both axis
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))


        f2 = f1._extract(row_key=np.array((False, True, True, False), dtype=bool))

        self.assertEqual(f2.to_pairs(0),
                (('p', (('x', 30), ('y', 2))), ('q', (('x', 34), ('y', 95))), ('r', (('x', 'b'), ('y', 'c'))), ('s', (('x', True), ('y', False))), ('t', (('x', False), ('y', False)))))


        f3 = f1._extract(row_key=np.array((True, False, False, True), dtype=bool))

        self.assertEqual(f3.to_pairs(0),
                (('p', (('w', 2), ('z', 30))), ('q', (('w', 2), ('z', 73))), ('r', (('w', 'a'), ('z', 'd'))), ('s', (('w', False), ('z', True))), ('t', (('w', False), ('z', True)))))


        # attempting to select any single row results in a problem, as the first block given to the TypeBlocks constructor is a 1d array that looks it is a (2,1) instead of a (1, 2)
        f4 = f1._extract(row_key=np.array((False, False, True, False), dtype=bool))

        self.assertEqual(
                f4.to_pairs(0),
                (('p', (('y', 2),)), ('q', (('y', 95),)), ('r', (('y', 'c'),)), ('s', (('y', False),)), ('t', (('y', False),)))
                )




    def test_frame_loc_a(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        # cases of single series extraction
        s1 = f1.loc['x']
        self.assertEqual(list(s1.items()),
                [('p', 1), ('q', 2), ('r', 'a'), ('s', False), ('t', True)])

        s2 = f1.loc[:, 'p']
        self.assertEqual(list(s2.items()),
                [('x', 1), ('y', 30)])

        self.assertEqual(
                f1.loc[['y', 'x']].index.values.tolist(),
                ['y', 'x'])

        self.assertEqual(f1['r':].columns.values.tolist(),
                ['r', 's', 't'])


    def test_frame_items_a(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        self.assertEqual(
                list((k, list(v.items())) for k, v in f1.items()),
                [('p', [('x', 1), ('y', 30)]), ('q', [('x', 2), ('y', 50)]), ('r', [('x', 'a'), ('y', 'b')]), ('s', [('x', False), ('y', True)]), ('t', [('x', True), ('y', False)])]
                )



    def test_frame_loc_b(self):
        # dimensionality of returned item based on selectors
        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        # return a series if one axis is multi
        post = f1.loc['x', 't':]
        self.assertEqual(post.__class__, Series)
        self.assertEqual(post.index.values.tolist(), ['t'])

        post = f1.loc['y':, 't']
        self.assertEqual(post.__class__, Series)
        self.assertEqual(post.index.values.tolist(), ['y'])

        # if both are multi than we get a Frame
        post = f1.loc['y':, 't':]
        self.assertEqual(post.__class__, Frame)
        self.assertEqual(post.index.values.tolist(), ['y'])
        self.assertEqual(post.columns.values.tolist(), ['t'])

        # return a series
        post = f1.loc['x', 's':]
        self.assertEqual(post.__class__, Series)
        self.assertEqual(post.index.values.tolist(),['s', 't'])

        post = f1.loc[:, 's']
        self.assertEqual(post.__class__, Series)
        self.assertEqual(post.index.values.tolist(), ['x', 'y'])

        self.assertEqual(f1.loc['x', 's'], False)
        self.assertEqual(f1.loc['y', 'p'], 30)


    def test_frame_attrs_a(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        self.assertEqual(str(f1.dtypes.values.tolist()),
                "[dtype('int64'), dtype('int64'), dtype('<U1'), dtype('bool'), dtype('bool')]")

        self.assertEqual(f1.size, 10)
        self.assertEqual(f1.ndim, 2)
        self.assertEqual(f1.shape, (2, 5))



    def test_frame_assign_iloc_a(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))


        self.assertEqual(f1.assign.iloc[1,1](3000).iloc[1,1], 3000)


    def test_frame_assign_loc_a(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        self.assertEqual(f1.assign.loc['x', 's'](3000).values.tolist(),
                [[1, 2, 'a', 3000, True], [30, 50, 'b', True, False]])

        # can assign to a columne
        self.assertEqual(
                f1.assign['s']('x').values.tolist(),
                [[1, 2, 'a', 'x', True], [30, 50, 'b', 'x', False]])


    def test_frame_assign_loc_b(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        # unindexed tuple/list assingment
        self.assertEqual(
                f1.assign['s']([50, 40]).values.tolist(),
                [[1, 2, 'a', 50, True], [30, 50, 'b', 40, False]]
                )

        self.assertEqual(
                f1.assign.loc['y'](list(range(5))).values.tolist(),
                [[1, 2, 'a', False, True], [0, 1, 2, 3, 4]])




    def test_frame_assign_loc_c(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        # assinging a series to a part of wone row
        post = f1.assign.loc['x', 'r':](Series((-1, -2, -3), index=('t', 'r', 's')))

        self.assertEqual(post.values.tolist(),
                [[1, 2, -2, -3, -1], [30, 50, 'b', True, False]])

        post = f1.assign.loc[['x', 'y'], 'r'](Series((-1, -2), index=('y', 'x')))

        self.assertEqual(post.values.tolist(),
                [[1, 2, -2, False, True], [30, 50, -1, True, False]])

        # ordere here does not matter
        post = f1.assign.loc[['y', 'x'], 'r'](Series((-1, -2), index=('y', 'x')))

        self.assertEqual(post.values.tolist(),
                [[1, 2, -2, False, True], [30, 50, -1, True, False]])


    def test_frame_assign_loc_d(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        value1 = Frame.from_records(((20, 21, 22),(23, 24, 25)),
                index=('x', 'y'),
                columns=('s', 't', 'w'))

        f2 = f1.assign.loc[['x', 'y'], ['s', 't']](value1)
        self.assertEqual(f2.to_pairs(0),
                (('p', (('x', 1), ('y', 30))), ('q', (('x', 2), ('y', 50))), ('r', (('x', 'a'), ('y', 'b'))), ('s', (('x', 20), ('y', 23))), ('t', (('x', 21), ('y', 24)))))



    def test_frame_assign_coercion_a(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))
        f2 = f1.assign.loc['x', 'r'](None)
        self.assertEqual(f2.to_pairs(0),
                (('p', (('x', 1), ('y', 30))), ('q', (('x', 2), ('y', 50))), ('r', (('x', None), ('y', 'b'))), ('s', (('x', False), ('y', True))), ('t', (('x', True), ('y', False)))))


    def test_frame_mask_loc_a(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        self.assertEqual(
                f1.mask.loc['x', 'r':].values.tolist(),
                [[False, False, True, True, True], [False, False, False, False, False]])


        self.assertEqual(f1.mask['s'].values.tolist(),
                [[False, False, False, True, False], [False, False, False, True, False]])


    def test_frame_masked_array_loc_a(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        # mask our the non-integers
        self.assertEqual(
                f1.masked_array.loc[:, 'r':].sum(), 83)


    def test_reindex_other_like_iloc_a(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        value1 = Series((100, 200, 300), index=('s', 'u', 't'))
        iloc_key1 = (0, slice(2, None))
        v1 = f1._reindex_other_like_iloc(value1, iloc_key1)

        self.assertAlmostEqualItems(v1.items(),
                [('r', nan), ('s', 100), ('t', 300)])


        value2 = Series((100, 200), index=('y', 'x'))
        iloc_key2 = (slice(0, None), 2)
        v2 = f1._reindex_other_like_iloc(value2, iloc_key2)

        self.assertAlmostEqualItems(v2.items(),
                [('x', 200), ('y', 100)])


    def test_reindex_other_like_iloc_b(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        value1 = Frame.from_records(((20, 21, 22),(23, 24, 25)),
                index=('x', 'y'),
                columns=('s', 't', 'w'))

        iloc_key1 = (slice(0, None), slice(3, None))
        v1 = f1._reindex_other_like_iloc(value1, iloc_key1)

        self.assertEqual(v1.to_pairs(0),
                (('s', (('x', 20), ('y', 23))), ('t', (('x', 21), ('y', 24)))))


    def test_frame_reindex_a(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 34, 'b', True, False),
                (54, 95, 'c', False, False),
                (65, 73, 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        # subset index reindex
        self.assertEqual(
                f1.reindex(('z', 'w')).to_pairs(axis=0),
                (('p', (('z', 65), ('w', 1))), ('q', (('z', 73), ('w', 2))), ('r', (('z', 'd'), ('w', 'a'))), ('s', (('z', True), ('w', False))), ('t', (('z', True), ('w', True)))))

        # index only with nan filling
        self.assertEqual(
                f1.reindex(('z', 'b', 'w'), fill_value=None).to_pairs(0),
                (('p', (('z', 65), ('b', None), ('w', 1))), ('q', (('z', 73), ('b', None), ('w', 2))), ('r', (('z', 'd'), ('b', None), ('w', 'a'))), ('s', (('z', True), ('b', None), ('w', False))), ('t', (('z', True), ('b', None), ('w', True)))))


    def test_frame_axis_flat_a(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 34, 'b', True, False),
                (54, 95, 'c', False, False),
                (65, 73, 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        self.assertEqual(f1.to_pairs(axis=1),
                (('w', (('p', 1), ('q', 2), ('r', 'a'), ('s', False), ('t', True))), ('x', (('p', 30), ('q', 34), ('r', 'b'), ('s', True), ('t', False))), ('y', (('p', 54), ('q', 95), ('r', 'c'), ('s', False), ('t', False))), ('z', (('p', 65), ('q', 73), ('r', 'd'), ('s', True), ('t', True)))))


        self.assertEqual(f1.to_pairs(axis=0),
                (('p', (('w', 1), ('x', 30), ('y', 54), ('z', 65))), ('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73))), ('r', (('w', 'a'), ('x', 'b'), ('y', 'c'), ('z', 'd'))), ('s', (('w', False), ('x', True), ('y', False), ('z', True))), ('t', (('w', True), ('x', False), ('y', False), ('z', True)))))


    def test_frame_reindex_b(self):

        records = (
                (1, 2, 'a', False, True),
                (30, 34, 'b', True, False),
                (54, 95, 'c', False, False),
                (65, 73, 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        self.assertEqual(
                f1.reindex(columns=('q', 'p', 'w'), fill_value=None).to_pairs(0),
                (('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73))), ('p', (('w', 1), ('x', 30), ('y', 54), ('z', 65))), ('w', (('w', None), ('x', None), ('y', None), ('z', None)))))

        self.assertEqual(
                f1.reindex(columns=('q', 'p', 's')).to_pairs(0),
                (('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73))), ('p', (('w', 1), ('x', 30), ('y', 54), ('z', 65))), ('s', (('w', False), ('x', True), ('y', False), ('z', True)))))

        f2 = f1[['p', 'q']]

        self.assertEqual(
                f2.reindex(columns=('q', 'p')).to_pairs(0),
                (('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73))), ('p', (('w', 1), ('x', 30), ('y', 54), ('z', 65)))))

        self.assertEqual(
                f2.reindex(columns=('a', 'b'), fill_value=None).to_pairs(0),
                (('a', (('w', None), ('x', None), ('y', None), ('z', None))), ('b', (('w', None), ('x', None), ('y', None), ('z', None)))))


    def test_frame_reindex_c(self):
        # reindex both axis
        records = (
                (1, 2, 'a', False, True),
                (30, 34, 'b', True, False),
                (54, 95, 'c', False, False),
                (65, 73, 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))


        self.assertEqual(
                f1.reindex(index=('y', 'x'), columns=('s', 'q')).to_pairs(0),
                (('s', (('y', False), ('x', True))), ('q', (('y', 95), ('x', 34)))))

        self.assertEqual(
                f1.reindex(index=('x', 'y'), columns=('s', 'q', 'u'),
                        fill_value=None).to_pairs(0),
                (('s', (('x', True), ('y', False))), ('q', (('x', 34), ('y', 95))), ('u', (('x', None), ('y', None)))))

        self.assertEqual(
                f1.reindex(index=('a', 'b'), columns=('c', 'd'),
                        fill_value=None).to_pairs(0),
                (('c', (('a', None), ('b', None))), ('d', (('a', None), ('b', None)))))


        f2 = f1[['p', 'q']]

        self.assertEqual(
                f2.reindex(index=('x',), columns=('q',)).to_pairs(0),
                (('q', (('x', 34),)),))

        self.assertEqual(
                f2.reindex(index=('y', 'x', 'q'), columns=('q',),
                        fill_value=None).to_pairs(0),
                (('q', (('y', 95), ('x', 34), ('q', None))),))



    def test_frame_axis_interface_a(self):
        # reindex both axis
        records = (
                (1, 2, 'a', False, True),
                (30, 34, 'b', True, False),
                (54, 95, 'c', False, False),
                (65, 73, 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        self.assertEqual(f1.to_pairs(1),
                (('w', (('p', 1), ('q', 2), ('r', 'a'), ('s', False), ('t', True))), ('x', (('p', 30), ('q', 34), ('r', 'b'), ('s', True), ('t', False))), ('y', (('p', 54), ('q', 95), ('r', 'c'), ('s', False), ('t', False))), ('z', (('p', 65), ('q', 73), ('r', 'd'), ('s', True), ('t', True)))))

        for x in f1.iter_tuple(0):
            self.assertTrue(len(x), 4)

        for x in f1.iter_tuple(1):
            self.assertTrue(len(x), 5)


        f2 = f1[['p', 'q']]

        s1 = f2.iter_array(0).apply(np.sum)
        self.assertEqual(list(s1.items()), [('p', 150), ('q', 204)])

        s2 = f2.iter_array(1).apply(np.sum)
        self.assertEqual(list(s2.items()),
                [('w', 3), ('x', 64), ('y', 149), ('z', 138)])

        def sum_if(idx, vals):
            if idx in ('x', 'z'):
                return np.sum(vals)

        s3 = f2.iter_array_items(1).apply(sum_if)
        self.assertEqual(list(s3.items()),
                [('w', None), ('x', 64), ('y', None), ('z', 138)])



    def test_frame_group_a(self):
        # reindex both axis
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        post = tuple(f1._axis_group_iloc_items(4, axis=0)) # row iter, group by column 4

        group1, group_frame_1 = post[0]
        group2, group_frame_2 = post[1]

        self.assertEqual(group1, False)
        self.assertEqual(group2, True)

        self.assertEqual(group_frame_1.to_pairs(0),
                (('p', (('w', 2), ('x', 30), ('y', 2))), ('q', (('w', 2), ('x', 34), ('y', 95))), ('r', (('w', 'a'), ('x', 'b'), ('y', 'c'))), ('s', (('w', False), ('x', True), ('y', False))), ('t', (('w', False), ('x', False), ('y', False)))))

        self.assertEqual(group_frame_2.to_pairs(0),
                (('p', (('z', 30),)), ('q', (('z', 73),)), ('r', (('z', 'd'),)), ('s', (('z', True),)), ('t', (('z', True),))))


    def test_frame_group_b(self):
        # reindex both axis
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        # column iter, group by row 0
        post = list(f1._axis_group_iloc_items(0, axis=1))

        self.assertEqual(post[0][0], 2)
        self.assertEqual(post[0][1].to_pairs(0),
                (('p', (('w', 2), ('x', 30), ('y', 2), ('z', 30))), ('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73)))))

        self.assertEqual(post[1][0], False)
        self.assertEqual(post[1][1].to_pairs(0),
                (('s', (('w', False), ('x', True), ('y', False), ('z', True))), ('t', (('w', False), ('x', False), ('y', False), ('z', True)))))

        self.assertEqual(post[2][0], 'a')

        self.assertEqual(post[2][1].to_pairs(0),
                (('r', (('w', 'a'), ('x', 'b'), ('y', 'c'), ('z', 'd'))),))



    def test_frame_axis_interface_b(self):
        # reindex both axis
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        post = list(f1.iter_group_items('s', axis=0))

        self.assertEqual(post[0][1].to_pairs(0),
                (('p', (('w', 2), ('y', 2))), ('q', (('w', 2), ('y', 95))), ('r', (('w', 'a'), ('y', 'c'))), ('s', (('w', False), ('y', False))), ('t', (('w', False), ('y', False)))))

        self.assertEqual(post[1][1].to_pairs(0),
                (('p', (('x', 30), ('z', 30))), ('q', (('x', 34), ('z', 73))), ('r', (('x', 'b'), ('z', 'd'))), ('s', (('x', True), ('z', True))), ('t', (('x', False), ('z', True)))))


        s1 = f1.iter_group('p', axis=0).apply(lambda f: f['q'].values.sum())
        self.assertEqual(list(s1.items()), [(2, 97), (30, 107)])


    def test_frame_contains_a(self):

        f1 = Frame.from_items(zip(('a', 'b'), ([20, 30, 40], [80, 10, 30])),
                index=('x', 'y', 'z'))

        self.assertTrue('a' in f1)
        self.assertTrue('b' in f1)
        self.assertFalse('x' in f1)
        self.assertFalse('y' in f1)



    def test_frame_sum_a(self):
        # reindex both axis
        records = (
                (2, 2, 3, 4.23, 50.234),
                (30, 34, 60, 80.6, 90.123),
                (2, 95, 1, 1.96, 1.54),
                (30, 73, 50, 40.23, 30.234),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        post = f1.sum(axis=0)
        self.assertAlmostEqualItems(list(post.items()),
                [('p', 64.0), ('q', 204.0), ('r', 114.0), ('s', 127.01999999999998), ('t', 172.131)])
        self.assertEqual(post.dtype, np.float64)

        post = f1.sum(axis=1) # sum columns, return row index
        self.assertEqual(list(f1.sum(axis=1).items()),
                [('w', 61.463999999999999), ('x', 294.72300000000001), ('y', 101.5), ('z', 223.464)])
        self.assertEqual(post.dtype, np.float64)


    def test_frame_sum_b(self):

        records = (
                (2, 2, 3),
                (30, 34, 60),
                (2, 95, 1),
                (30, 73, 50),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r'),
                index=('w', 'x', 'y', 'z'))

        post = f1.sum(axis=0)

        self.assertEqual(list(post.items()),
                [('p', 64), ('q', 204), ('r', 114)])

        self.assertEqual(list(f1.sum(axis=1).items()),
                [('w', 7), ('x', 124), ('y', 98), ('z', 153)])


    def test_frame_min_a(self):
        # reindex both axis
        records = (
                (2, 2, 3, 4.23, 50.234),
                (30, 34, 60, 80.6, 90.123),
                (2, 95, 1, 1.96, 1.54),
                (30, 73, 50, 40.23, 30.234),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        self.assertAlmostEqualItems(tuple(f1.min().items()),
                (('p', 2.0), ('q', 2.0), ('r', 1.0), ('s', 1.96), ('t', 1.54)))

        self.assertAlmostEqualItems(tuple(f1.min(axis=1).items()),
                (('w', 2.0), ('x', 30.0), ('y', 1.0), ('z', 30.0)))

    def test_frame_row_dtype_a(self):
        # reindex both axis
        records = (
                (2, 2, 3, 4.23, 50.234),
                (30, 34, 60, 80.6, 90.123),
                (2, 95, 1, 1.96, 1.54),
                (30, 73, 50, 40.23, 30.234),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        self.assertEqual(f1['t'].dtype, np.float64)
        self.assertEqual(f1['p'].dtype, np.int64)

        self.assertEqual(f1.loc['w'].dtype, np.float64)
        self.assertEqual(f1.loc['z'].dtype, np.float64)

        self.assertEqual(f1[['r', 's']].values.dtype, np.float64)

    def test_frame_unary_operator_a(self):

        records = (
                (2, 2, 3, False, True),
                (30, 34, 60, True, False),
                (2, 95, 1, True, True),
                (30, 73, 50, False, False),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        # raises exception with NP14
        # self.assertEqual((-f1).to_pairs(0),
        #         (('p', (('w', -2), ('x', -30), ('y', -2), ('z', -30))), ('q', (('w', -2), ('x', -34), ('y', -95), ('z', -73))), ('r', (('w', -3), ('x', -60), ('y', -1), ('z', -50))), ('s', (('w', True), ('x', False), ('y', False), ('z', True))), ('t', (('w', False), ('x', True), ('y', False), ('z', True)))))

        self.assertEqual((~f1).to_pairs(0),
                (('p', (('w', -3), ('x', -31), ('y', -3), ('z', -31))), ('q', (('w', -3), ('x', -35), ('y', -96), ('z', -74))), ('r', (('w', -4), ('x', -61), ('y', -2), ('z', -51))), ('s', (('w', True), ('x', False), ('y', False), ('z', True))), ('t', (('w', False), ('x', True), ('y', False), ('z', True)))))


    def test_frame_binary_operator_a(self):
        # constants

        records = (
                (2, 2, 3.5),
                (30, 34, 60.2),
                (2, 95, 1.2),
                (30, 73, 50.2),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r'),
                index=('w', 'x', 'y', 'z'))

        self.assertEqual((f1 * 30).to_pairs(0),
                (('p', (('w', 60), ('x', 900), ('y', 60), ('z', 900))), ('q', (('w', 60), ('x', 1020), ('y', 2850), ('z', 2190))), ('r', (('w', 105.0), ('x', 1806.0), ('y', 36.0), ('z', 1506.0))))
                )



    def test_frame_binary_operator_b(self):

        records = (
                (2, 2, 3.5),
                (30, 34, 60.2),
                (2, 95, 1.2),
                (30, 73, 50.2),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r'),
                index=('w', 'x', 'y', 'z'))

        f2 = f1.loc[['y', 'z'], ['r']]
        f3 = f1 * f2

        self.assertAlmostEqualItems(list(f3['p'].items()),
                [('w', nan), ('x', nan), ('y', nan), ('z', nan)])
        self.assertAlmostEqualItems(list(f3['r'].items()),
                [('w', nan), ('x', nan), ('y', 1.4399999999999999), ('z', 2520.0400000000004)])

    def test_frame_binary_operator_c(self):

        records = (
                (2, 2, 3.5),
                (30, 34, 60.2),
                (2, 95, 1.2),
                (30, 73, 50.2),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r'),
                index=('w', 'x', 'y', 'z'))

        s1 = Series([0, 1, 2], index=('r', 'q', 'p'))

        self.assertEqual((f1 * s1).to_pairs(0),
                (('p', (('w', 4), ('x', 60), ('y', 4), ('z', 60))), ('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73))), ('r', (('w', 0.0), ('x', 0.0), ('y', 0.0), ('z', 0.0)))))

        self.assertEqual((f1 * [0, 1, 0]).to_pairs(0),
                (('p', (('w', 0), ('x', 0), ('y', 0), ('z', 0))), ('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73))), ('r', (('w', 0.0), ('x', 0.0), ('y', 0.0), ('z', 0.0)))))


    def test_frame_isin_a(self):
        # reindex both axis
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        post = f1.isin({'a', 73, 30})
        self.assertEqual(post.to_pairs(0),
                (('p', (('w', False), ('x', True), ('y', False), ('z', True))), ('q', (('w', False), ('x', False), ('y', False), ('z', True))), ('r', (('w', True), ('x', False), ('y', False), ('z', False))), ('s', (('w', False), ('x', False), ('y', False), ('z', False))), ('t', (('w', False), ('x', False), ('y', False), ('z', False)))))


        post = f1.isin(['a', 73, 30])
        self.assertEqual(post.to_pairs(0),
                (('p', (('w', False), ('x', True), ('y', False), ('z', True))), ('q', (('w', False), ('x', False), ('y', False), ('z', True))), ('r', (('w', True), ('x', False), ('y', False), ('z', False))), ('s', (('w', False), ('x', False), ('y', False), ('z', False))), ('t', (('w', False), ('x', False), ('y', False), ('z', False)))))


    def test_frame_transpose_a(self):
        # reindex both axis
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        f2 = f1.transpose()

        self.assertEqual(f2.to_pairs(0),
                (('w', (('p', 2), ('q', 2), ('r', 'a'), ('s', False), ('t', False))), ('x', (('p', 30), ('q', 34), ('r', 'b'), ('s', True), ('t', False))), ('y', (('p', 2), ('q', 95), ('r', 'c'), ('s', False), ('t', False))), ('z', (('p', 30), ('q', 73), ('r', 'd'), ('s', True), ('t', True)))))



    def test_frame_from_element_iloc_items_a(self):
        items = (((0,1), 'g'), ((1,0), 'q'))

        f1 = Frame.from_element_iloc_items(items,
                index=('a', 'b'),
                columns=('x', 'y'),
                dtype=object
                )

        self.assertEqual(f1.to_pairs(0),
                (('x', (('a', None), ('b', 'q'))), ('y', (('a', 'g'), ('b', None)))))


        items = (((0,1), .5), ((1,0), 1.5))

        f2 = Frame.from_element_iloc_items(items,
                index=('a', 'b'),
                columns=('x', 'y'),
                dtype=float
                )

        self.assertAlmostEqualItems(tuple(f2['x'].items()),
                (('a', nan), ('b', 1.5)))

        self.assertAlmostEqualItems(tuple(f2['y'].items()),
                (('a', 0.5), ('b', nan)))


    def test_frame_from_element_loc_items_a(self):
        items = ((('b', 'x'), 'g'), (('a','y'), 'q'))

        f1 = Frame.from_element_loc_items(items,
                index=('a', 'b'),
                columns=('x', 'y'),
                dtype=object
                )

        self.assertEqual(f1.to_pairs(0),
                (('x', (('a', None), ('b', 'g'))), ('y', (('a', 'q'), ('b', None)))))



    def test_frame_iter_element_a(self):
        # reindex both axis
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        self.assertEqual(
                [x for x in f1.iter_element()],
                [2, 2, 'a', False, False, 30, 34, 'b', True, False, 2, 95, 'c', False, False, 30, 73, 'd', True, True])

        self.assertEqual([x for x in f1.iter_element_items()],
                [(('w', 'p'), 2), (('w', 'q'), 2), (('w', 'r'), 'a'), (('w', 's'), False), (('w', 't'), False), (('x', 'p'), 30), (('x', 'q'), 34), (('x', 'r'), 'b'), (('x', 's'), True), (('x', 't'), False), (('y', 'p'), 2), (('y', 'q'), 95), (('y', 'r'), 'c'), (('y', 's'), False), (('y', 't'), False), (('z', 'p'), 30), (('z', 'q'), 73), (('z', 'r'), 'd'), (('z', 's'), True), (('z', 't'), True)])


        post = f1.iter_element().apply(lambda x: '_' + str(x) + '_')

        self.assertEqual(post.to_pairs(0),
                (('p', (('w', '_2_'), ('x', '_30_'), ('y', '_2_'), ('z', '_30_'))), ('q', (('w', '_2_'), ('x', '_34_'), ('y', '_95_'), ('z', '_73_'))), ('r', (('w', '_a_'), ('x', '_b_'), ('y', '_c_'), ('z', '_d_'))), ('s', (('w', '_False_'), ('x', '_True_'), ('y', '_False_'), ('z', '_True_'))), ('t', (('w', '_False_'), ('x', '_False_'), ('y', '_False_'), ('z', '_True_')))))




    def test_frame_iter_element_b(self):
        # reindex both axis
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        # support working with mappings
        post = f1.iter_element().apply({2: 200, False: 200})

        self.assertEqual(post.to_pairs(0),
                (('p', (('w', 200), ('x', 30), ('y', 200), ('z', 30))), ('q', (('w', 200), ('x', 34), ('y', 95), ('z', 73))), ('r', (('w', 'a'), ('x', 'b'), ('y', 'c'), ('z', 'd'))), ('s', (('w', 200), ('x', True), ('y', 200), ('z', True))), ('t', (('w', 200), ('x', 200), ('y', 200), ('z', True))))
                )


    def test_frame_sort_index_a(self):
        # reindex both axis
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('z', 'x', 'w', 'y'))

        self.assertEqual(f1.sort_index().to_pairs(0),
                (('p', (('w', 2), ('x', 30), ('y', 30), ('z', 2))), ('q', (('w', 95), ('x', 34), ('y', 73), ('z', 2))), ('r', (('w', 'c'), ('x', 'b'), ('y', 'd'), ('z', 'a'))), ('s', (('w', False), ('x', True), ('y', True), ('z', False))), ('t', (('w', False), ('x', False), ('y', True), ('z', False)))))


        self.assertEqual(f1.sort_index(ascending=False).to_pairs(0),
                (('p', (('z', 2), ('y', 30), ('x', 30), ('w', 2))), ('q', (('z', 2), ('y', 73), ('x', 34), ('w', 95))), ('r', (('z', 'a'), ('y', 'd'), ('x', 'b'), ('w', 'c'))), ('s', (('z', False), ('y', True), ('x', True), ('w', False))), ('t', (('z', False), ('y', True), ('x', False), ('w', False)))))


    def test_frame_sort_columns_a(self):
        # reindex both axis
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('t', 's', 'r', 'q', 'p'),
                index=('z', 'x', 'w', 'y'))

        self.assertEqual(
                f1.sort_columns().to_pairs(0),
                (('p', (('z', False), ('x', False), ('w', False), ('y', True))), ('q', (('z', False), ('x', True), ('w', False), ('y', True))), ('r', (('z', 'a'), ('x', 'b'), ('w', 'c'), ('y', 'd'))), ('s', (('z', 2), ('x', 34), ('w', 95), ('y', 73))), ('t', (('z', 2), ('x', 30), ('w', 2), ('y', 30)))))



    def test_frame_sort_values_a(self):
        # reindex both axis
        records = (
                (2, 2, 'c', False, False),
                (30, 34, 'd', True, False),
                (2, 95, 'a', False, False),
                (30, 73, 'b', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'r', 'q', 't', 's'),
                index=('z', 'x', 'w', 'y'))

        post = f1.sort_values('q')


        self.assertEqual(post.to_pairs(0),
                (('p', (('w', 2), ('y', 30), ('z', 2), ('x', 30))), ('r', (('w', 95), ('y', 73), ('z', 2), ('x', 34))), ('q', (('w', 'a'), ('y', 'b'), ('z', 'c'), ('x', 'd'))), ('t', (('w', False), ('y', True), ('z', False), ('x', True))), ('s', (('w', False), ('y', True), ('z', False), ('x', False)))))


        self.assertEqual(f1.sort_values('p').to_pairs(0),
                (('p', (('z', 2), ('w', 2), ('x', 30), ('y', 30))), ('r', (('z', 2), ('w', 95), ('x', 34), ('y', 73))), ('q', (('z', 'c'), ('w', 'a'), ('x', 'd'), ('y', 'b'))), ('t', (('z', False), ('w', False), ('x', True), ('y', True))), ('s', (('z', False), ('w', False), ('x', False), ('y', True))))
                )


    def test_frame_sort_values_b(self):
        # reindex both axis
        records = (
                (2, 2, 'c', False, False),
                (30, 34, 'd', True, False),
                (2, 95, 'a', True, False),
                (30, 73, 'b', False, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'r', 'q', 't', 's'),
                index=('z', 'x', 'w', 'y'))

        post = f1.sort_values(('p', 't'))

        self.assertEqual(post.to_pairs(0),
                (('p', (('z', 2), ('w', 2), ('y', 30), ('x', 30))), ('r', (('z', 2), ('w', 95), ('y', 73), ('x', 34))), ('q', (('z', 'c'), ('w', 'a'), ('y', 'b'), ('x', 'd'))), ('t', (('z', False), ('w', True), ('y', False), ('x', True))), ('s', (('z', False), ('w', False), ('y', True), ('x', False)))))



    def test_frame_sort_values_c(self):

        records = (
                (2, 2, 3.5),
                (30, 34, 60.2),
                (2, 95, 1.2),
                (30, 73, 50.2),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r'),
                index=('w', 'x', 'y', 'z'))

        self.assertEqual(f1.sort_values('y', axis=0).to_pairs(0),
                (('r', (('w', 3.5), ('x', 60.2), ('y', 1.2), ('z', 50.2))), ('p', (('w', 2), ('x', 30), ('y', 2), ('z', 30))), ('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73)))))



    def test_frame_relabel_a(self):
        # reindex both axis
        records = (
                (2, 2, 'c', False, False),
                (30, 34, 'd', True, False),
                (2, 95, 'a', False, False),
                (30, 73, 'b', True, True),
                )

        f1 = FrameGO.from_records(records,
                columns=('p', 'r', 'q', 't', 's'),
                index=('z', 'x', 'w', 'y'))

        f2 = f1.relabel(columns={'q': 'QQQ'})

        self.assertEqual(f2.to_pairs(0),
                (('p', (('z', 2), ('x', 30), ('w', 2), ('y', 30))), ('r', (('z', 2), ('x', 34), ('w', 95), ('y', 73))), ('QQQ', (('z', 'c'), ('x', 'd'), ('w', 'a'), ('y', 'b'))), ('t', (('z', False), ('x', True), ('w', False), ('y', True))), ('s', (('z', False), ('x', False), ('w', False), ('y', True))))
                )

        f3 = f1.relabel(index={'y': 'YYY'})

        self.assertEqual(f3.to_pairs(0),
                (('p', (('z', 2), ('x', 30), ('w', 2), ('YYY', 30))), ('r', (('z', 2), ('x', 34), ('w', 95), ('YYY', 73))), ('q', (('z', 'c'), ('x', 'd'), ('w', 'a'), ('YYY', 'b'))), ('t', (('z', False), ('x', True), ('w', False), ('YYY', True))), ('s', (('z', False), ('x', False), ('w', False), ('YYY', True)))))

        self.assertTrue((f1.mloc == f2.mloc).all())
        self.assertTrue((f2.mloc == f3.mloc).all())


    def test_frame_get_a(self):
        # reindex both axis
        records = (
                (2, 2, 'c', False, False),
                (30, 34, 'd', True, False),
                (2, 95, 'a', False, False),
                (30, 73, 'b', True, True),
                )

        f1 = FrameGO.from_records(records,
                columns=('p', 'r', 'q', 't', 's'),
                index=('z', 'x', 'w', 'y'))

        self.assertEqual(f1.get('r').values.tolist(),
                [2, 34, 95, 73])

        self.assertEqual(f1.get('a'), None)
        self.assertEqual(f1.get('w'), None)
        self.assertEqual(f1.get('a', -1), -1)

    def test_frame_isna_a(self):
        f1 = FrameGO([
                [np.nan, 2, np.nan, 0],
                [3, 4, np.nan, 1],
                [np.nan, np.nan, np.nan, 5]],
                columns=list('ABCD'))

        self.assertEqual(f1.isna().to_pairs(0),
                (('A', ((0, True), (1, False), (2, True))), ('B', ((0, False), (1, False), (2, True))), ('C', ((0, True), (1, True), (2, True))), ('D', ((0, False), (1, False), (2, False)))))

        self.assertEqual(f1.notna().to_pairs(0),
                (('A', ((0, False), (1, True), (2, False))), ('B', ((0, True), (1, True), (2, False))), ('C', ((0, False), (1, False), (2, False))), ('D', ((0, True), (1, True), (2, True)))))

    def test_frame_dropna_a(self):
        f1 = FrameGO([
                [np.nan, 2, np.nan, 0],
                [3, 4, np.nan, 1],
                [np.nan, np.nan, np.nan, np.nan]],
                columns=list('ABCD'))

        self.assertAlmostEqualFramePairs(
                f1.dropna(axis=0, condition=np.all).to_pairs(0),
                (('A', ((0, nan), (1, 3.0))), ('B', ((0, 2.0), (1, 4.0))), ('C', ((0, nan), (1, nan))), ('D', ((0, 0.0), (1, 1.0)))))

        self.assertAlmostEqualFramePairs(
                f1.dropna(axis=1, condition=np.all).to_pairs(0),
                (('A', ((0, nan), (1, 3.0), (2, nan))), ('B', ((0, 2.0), (1, 4.0), (2, nan))), ('D', ((0, 0.0), (1, 1.0), (2, nan)))))


        f2 = f1.dropna(axis=0, condition=np.any)
        # dropping to zero results in an empty DF in the same manner as Pandas; not sure if this is correct or ideal
        self.assertEqual(f2.shape, (0, 4))

        f3 = f1.dropna(axis=1, condition=np.any)
        self.assertEqual(f3.shape, (0, 0))

    def test_frame_dropna_b(self):
        f1 = FrameGO([
                [np.nan, 2, 3, 0],
                [3, 4, np.nan, 1],
                [0 ,1, 2, 3]],
                columns=list('ABCD'))

        self.assertEqual(f1.dropna(axis=0, condition=np.any).to_pairs(0),
                (('A', ((2, 0.0),)), ('B', ((2, 1.0),)), ('C', ((2, 2.0),)), ('D', ((2, 3.0),))))
        self.assertEqual(f1.dropna(axis=1, condition=np.any).to_pairs(0),
                (('B', ((0, 2.0), (1, 4.0), (2, 1.0))), ('D', ((0, 0.0), (1, 1.0), (2, 3.0)))))



    def test_frame_fillna_a(self):
        dtype = np.dtype

        f1 = FrameGO([
                [np.nan, 2, 3, 0],
                [3, 4, np.nan, 1],
                [0 ,1, 2, 3]],
                columns=list('ABCD'))

        f2 = f1.fillna(0)
        self.assertEqual(f2.to_pairs(0),
                (('A', ((0, 0.0), (1, 3.0), (2, 0.0))), ('B', ((0, 2.0), (1, 4.0), (2, 1.0))), ('C', ((0, 3.0), (1, 0.0), (2, 2.0))), ('D', ((0, 0.0), (1, 1.0), (2, 3.0)))))

        post = f2.dtypes
        self.assertEqual(post.to_pairs(),
                (('A', dtype('float64')), ('B', dtype('float64')), ('C', dtype('float64')), ('D', dtype('float64'))))

        f3 = f1.fillna(None)
        self.assertEqual(f3.to_pairs(0),
                (('A', ((0, None), (1, 3.0), (2, 0.0))), ('B', ((0, 2.0), (1, 4.0), (2, 1.0))), ('C', ((0, 3.0), (1, None), (2, 2.0))), ('D', ((0, 0.0), (1, 1.0), (2, 3.0)))))

        post = f3.dtypes
        self.assertEqual(post.to_pairs(),
                (('A', dtype('O')), ('B', dtype('O')), ('C', dtype('O')), ('D', dtype('O'))))


    def test_frame_empty_a(self):

        f1 = FrameGO(index=('a', 'b', 'c'))
        f1['w'] = Series.from_items(zip('cebga', (10, 20, 30, 40, 50)))
        f1['x'] = Series.from_items(zip('abc', range(3, 6)))
        f1['y'] = Series.from_items(zip('abcd', range(2, 6)))
        f1['z'] = Series.from_items(zip('qabc', range(7, 11)))

        self.assertEqual(f1.to_pairs(0),
                (('w', (('a', 50), ('b', 30), ('c', 10))), ('x', (('a', 3), ('b', 4), ('c', 5))), ('y', (('a', 2), ('b', 3), ('c', 4))), ('z', (('a', 8), ('b', 9), ('c', 10)))))


    def test_frame_from_csv_a(self):
        # header, mixed types, no index

        s1 = StringIO('count,score,color\n1,1.3,red\n3,5.2,green\n100,3.4,blue\n4,9.0,black')

        f1 = Frame.from_csv(s1)
        post = f1.iloc[:, :2].sum(axis=0)
        self.assertEqual(post.to_pairs(),
                (('count', 108.0), ('score', 18.9)))
        self.assertEqual(f1.shape, (4, 3))
        self.assertEqual(f1.dtypes.iter_element().apply(str).to_pairs(),
                (('count', 'int64'), ('score', 'float64'), ('color', '<U5')))

        s2 = StringIO('color,count,score\nred,1,1.3\ngreen,3,5.2\nblue,100,3.4\nblack,4,9.0')

        f2 = Frame.from_csv(s2)
        self.assertEqual(f2['count':].sum().to_pairs(),
                (('count', 108.0), ('score', 18.9)))
        self.assertEqual(f2.shape, (4, 3))
        self.assertEqual(f2.dtypes.iter_element().apply(str).to_pairs(),
                (('color', '<U5'), ('count', 'int64'), ('score', 'float64')))


        # add junk at beginning and end
        s3 = StringIO('junk\ncolor,count,score\nred,1,1.3\ngreen,3,5.2\nblue,100,3.4\nblack,4,9.0\njunk')

        f3 = Frame.from_csv(s3, skip_header=1, skip_footer=1)
        self.assertEqual(f3.shape, (4, 3))
        self.assertEqual(f3.dtypes.iter_element().apply(str).to_pairs(),
                (('color', '<U5'), ('count', 'int64'), ('score', 'float64')))



    def test_frame_from_csv_b(self):
        filelike = StringIO('''count,number,weight,scalar,color,active
0,4,234.5,5.3,'red',False
30,50,9.234,5.434,'blue',True''')
        f1 = Frame.from_csv(filelike)

        self.assertEqual(f1.columns.values.tolist(),
                ['count', 'number', 'weight', 'scalar', 'color', 'active'])


    def test_frame_from_csv_c(self):

        s1 = StringIO('color,count,score\nred,1,1.3\ngreen,3,5.2\nblue,100,3.4\nblack,4,9.0')
        f1 = Frame.from_csv(s1, index_column='color')
        self.assertEqual(f1.to_pairs(0),
                (('count', (('red', 1), ('green', 3), ('blue', 100), ('black', 4))), ('score', (('red', 1.3), ('green', 5.2), ('blue', 3.4), ('black', 9.0)))))


    def test_frame_to_csv_a(self):
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        file = StringIO()
        f1.to_csv(file)
        file.seek(0)
        self.assertEqual(file.read(),
'index,p,q,r,s,t\nw,2,2,a,False,False\nx,30,34,b,True,False\ny,2,95,c,False,False\nz,30,73,d,True,True')

        file = StringIO()
        f1.to_csv(file, include_index=False)
        file.seek(0)
        self.assertEqual(file.read(),
'p,q,r,s,t\n2,2,a,False,False\n30,34,b,True,False\n2,95,c,False,False\n30,73,d,True,True')

        file = StringIO()
        f1.to_csv(file, include_index=False, include_columns=False)
        file.seek(0)
        self.assertEqual(file.read(),
'2,2,a,False,False\n30,34,b,True,False\n2,95,c,False,False\n30,73,d,True,True')


    def test_frame_to_tsv_a(self):
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        file = StringIO()
        f1.to_tsv(file)
        file.seek(0)
        self.assertEqual(file.read(),
'index\tp\tq\tr\ts\tt\nw\t2\t2\ta\tFalse\tFalse\nx\t30\t34\tb\tTrue\tFalse\ny\t2\t95\tc\tFalse\tFalse\nz\t30\t73\td\tTrue\tTrue')




    def test_frame_and_a(self):

        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))
        f2 = FrameGO([
                [np.nan, 2, 3, 0],
                [3, 4, np.nan, 1],
                [0 ,1, 2, 3]],
                columns=list('ABCD'))

        self.assertEqual(f1.all(axis=0).to_pairs(),
                (('p', True), ('q', True), ('r', True), ('s', False), ('t', False)))

        self.assertEqual(f1.any(axis=0).to_pairs(),
                (('p', True), ('q', True), ('r', True), ('s', True), ('t', True)))

        self.assertEqual(f1.all(axis=1).to_pairs(),
                (('w', False), ('x', False), ('y', False), ('z', True)))

        self.assertEqual(f1.any(axis=1).to_pairs(),
                (('w', True), ('x', True), ('y', True), ('z', True)))



    def test_frame_unique_a(self):

        records = (
                (2, 2, 3.5),
                (30, 34, 60.2),
                (2, 95, 1.2),
                (30, 73, 50.2),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r'),
                index=('w', 'x', 'y', 'z'))

        self.assertEqual(f1.unique().tolist(),
                [1.2, 2.0, 3.5, 30.0, 34.0, 50.2, 60.2, 73.0, 95.0])

        records = (
                (2, 2, 2),
                (30, 34, 34),
                (2, 2, 2),
                (30, 73, 73),
                )
        f2 = Frame.from_records(records,
                columns=('p', 'q', 'r'),
                index=('w', 'x', 'y', 'z'))

        self.assertEqual(f2.unique().tolist(), [2, 30, 34, 73])

        self.assertEqual(f2.unique(axis=0).tolist(),
                [[2, 2, 2], [30, 34, 34], [30, 73, 73]])
        self.assertEqual(f2.unique(axis=1).tolist(),
                [[2, 2], [30, 34], [2, 2], [30, 73]])


    def test_frame_duplicated_a(self):

        a1 = np.array([[50, 50, 32, 17, 17], [2,2,1,3,3]])
        f1 = Frame(a1, index=('a', 'b'), columns=('p', 'q', 'r', 's','t'))

        self.assertEqual(f1.duplicated(axis=1).to_pairs(),
                (('p', True), ('q', True), ('r', False), ('s', True), ('t', True)))

        self.assertEqual(f1.duplicated(axis=0).to_pairs(),
                (('a', False), ('b', False)))


    def test_frame_duplicated_b(self):

        a1 = np.array([[50, 50, 32, 17, 17], [2,2,1,3,3]])
        f1 = Frame(a1, index=('a', 'b'), columns=('p', 'q', 'r', 's','t'))

        self.assertEqual(f1.drop_duplicated(axis=1, exclude_first=True).to_pairs(1),
                (('a', (('p', 50), ('r', 32), ('s', 17))), ('b', (('p', 2), ('r', 1), ('s', 3)))))

    def test_frame_from_concat_a(self):
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x', 'a'))

        records = (
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )

        f2 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x', 'a'))

        records = (
                (2, 2, 'a', False, False),
                (30, 73, 'd', True, True),
                )

        f3 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x', 'a'))

        f = Frame.from_concat((f1, f2, f3), axis=1, columns=range(15))

        # no blocks are copied or reallcoated
        self.assertEqual(f.mloc.tolist(),
                f1.mloc.tolist() + f2.mloc.tolist() + f3.mloc.tolist()
                )
        # order of index is retained
        self.assertEqual(f.to_pairs(1),
                (('x', ((0, 2), (1, 2), (2, 'a'), (3, False), (4, False), (5, 2), (6, 95), (7, 'c'), (8, False), (9, False), (10, 2), (11, 2), (12, 'a'), (13, False), (14, False))), ('a', ((0, 30), (1, 34), (2, 'b'), (3, True), (4, False), (5, 30), (6, 73), (7, 'd'), (8, True), (9, True), (10, 30), (11, 73), (12, 'd'), (13, True), (14, True)))))


    def test_frame_from_concat_b(self):
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x', 'a'))

        records = (
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )

        f2 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x', 'b'))

        records = (
                (2, 2, 'a', False, False),
                (30, 73, 'd', True, True),
                )

        f3 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x', 'c'))

        f = Frame.from_concat((f1, f2, f3), axis=1, columns=range(15))

        self.assertEqual(f.index.values.tolist(),
                ['a', 'b', 'c', 'x'])

        self.assertAlmostEqualFramePairs(f.to_pairs(1),
                (('a', ((0, 30), (1, 34), (2, 'b'), (3, True), (4, False), (5, nan), (6, nan), (7, nan), (8, nan), (9, nan), (10, nan), (11, nan), (12, nan), (13, nan), (14, nan))), ('b', ((0, nan), (1, nan), (2, nan), (3, nan), (4, nan), (5, 30), (6, 73), (7, 'd'), (8, True), (9, True), (10, nan), (11, nan), (12, nan), (13, nan), (14, nan))), ('c', ((0, nan), (1, nan), (2, nan), (3, nan), (4, nan), (5, nan), (6, nan), (7, nan), (8, nan), (9, nan), (10, 30), (11, 73), (12, 'd'), (13, True), (14, True))), ('x', ((0, 2), (1, 2), (2, 'a'), (3, False), (4, False), (5, 2), (6, 95), (7, 'c'), (8, False), (9, False), (10, 2), (11, 2), (12, 'a'), (13, False), (14, False))))
                )


        f = Frame.from_concat((f1, f2, f3), union=False, axis=1, columns=range(15))

        self.assertEqual(f.index.values.tolist(),
                ['x'])
        self.assertEqual(f.to_pairs(0),
                ((0, (('x', 2),)), (1, (('x', 2),)), (2, (('x', 'a'),)), (3, (('x', False),)), (4, (('x', False),)), (5, (('x', 2),)), (6, (('x', 95),)), (7, (('x', 'c'),)), (8, (('x', False),)), (9, (('x', False),)), (10, (('x', 2),)), (11, (('x', 2),)), (12, (('x', 'a'),)), (13, (('x', False),)), (14, (('x', False),))))


    def test_frame_from_concat_c(self):
        records = (
                (2, 2, False),
                (30, 34, False),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 't'),
                index=('x', 'a'))

        records = (
                ('c', False),
                ('d', True),
                )
        f2 = Frame.from_records(records,
                columns=('r', 's',),
                index=('x', 'a'))

        # get combined columns as they are unique
        f = Frame.from_concat((f1, f2), axis=1)
        self.assertEqual(f.to_pairs(0),
                (('p', (('x', 2), ('a', 30))), ('q', (('x', 2), ('a', 34))), ('t', (('x', False), ('a', False))), ('r', (('x', 'c'), ('a', 'd'))), ('s', (('x', False), ('a', True))))
                )


    def test_frame_from_concat_d(self):
        records = (
                (2, 2, False),
                (30, 34, False),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r'),
                index=('a', 'b'))

        records = (
                (2, 2, False),
                (30, 34, False),
                )

        f2 = Frame.from_records(records,
                columns=('p', 'q', 'r'),
                index=('c', 'd'))

        f = Frame.from_concat((f1, f2), axis=0)

        # block copmatible will result in attempt to keep vertical types
        self.assertEqual(
                [str(x) for x in f.dtypes.values.tolist()],
                ['int64', 'int64', 'bool'])

        self.assertEqual(f.to_pairs(0),
                (('p', (('a', 2), ('b', 30), ('c', 2), ('d', 30))), ('q', (('a', 2), ('b', 34), ('c', 2), ('d', 34))), ('r', (('a', False), ('b', False), ('c', False), ('d', False)))))


    def test_frame_from_concat_e(self):

        f1 = Frame.from_items(zip(
                ('a', 'b', 'c'),
                ((1, 2), (1, 2), (False, True))
                ))

        f = Frame.from_concat((f1, f1, f1), index=range(6))
        self.assertEqual(
                f.to_pairs(0),
                (('a', ((0, 1), (1, 2), (2, 1), (3, 2), (4, 1), (5, 2))), ('b', ((0, 1), (1, 2), (2, 1), (3, 2), (4, 1), (5, 2))), ('c', ((0, False), (1, True), (2, False), (3, True), (4, False), (5, True)))))
        self.assertEqual(
                [str(x) for x in f.dtypes.values.tolist()],
                ['int64', 'int64', 'bool'])

        f = Frame.from_concat((f1, f1, f1), axis=1, columns=range(9))

        self.assertEqual(f.to_pairs(0),
                ((0, ((0, 1), (1, 2))), (1, ((0, 1), (1, 2))), (2, ((0, False), (1, True))), (3, ((0, 1), (1, 2))), (4, ((0, 1), (1, 2))), (5, ((0, False), (1, True))), (6, ((0, 1), (1, 2))), (7, ((0, 1), (1, 2))), (8, ((0, False), (1, True)))))

        self.assertEqual([str(x) for x in f.dtypes.values.tolist()],
                ['int64', 'int64', 'bool', 'int64', 'int64', 'bool', 'int64', 'int64', 'bool'])

    def test_frame_from_concat_f(self):
        # force a reblock before concatenating

        a1 = np.array([1,2,3])
        a2 = np.array([10,50,30])
        a3 = np.array([1345,2234,3345])
        a4 = np.array([False, True, False])
        a5 = np.array([False, False, False])
        a6 = np.array(['g', 'd', 'e'])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3, a4, a5, a6))

        f1 = Frame(TypeBlocks.from_blocks((a1, a2, a3, a4, a5, a6)),
                columns = ('a', 'b', 'c', 'd', 'e', 'f'),
                own_data=True)
        self.assertEqual(len(f1._blocks._blocks), 6)

        f2 = Frame(f1.iloc[1:]._blocks.consolidate(),
                columns = ('a', 'b', 'c', 'd', 'e', 'f'),
                own_data=True)
        self.assertEqual(len(f2._blocks._blocks), 3)

        f = Frame.from_concat((f1 ,f2), index=range(5))

        self.assertEqual(
                [str(x) for x in f.dtypes.values.tolist()],
                ['int64', 'int64', 'int64', 'bool', 'bool', '<U1'])

        self.assertEqual(
                [str(x.dtype) for x in f._blocks._blocks],
                ['int64', 'bool', '<U1'])


    def test_frame_from_concat_g(self):
        records = (
                (2, 2, False),
                (30, 34, False),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 't'),
                index=('x', 'a'))

        records = (
                ('c', False),
                ('d', True),
                )
        f2 = Frame.from_records(records,
                columns=('r', 's',),
                index=('x', 'a'))

        # get combined columns as they are unique
        f = Frame.from_concat((f1, f2), axis=1)
        self.assertEqual(f.to_pairs(0),
                (('p', (('x', 2), ('a', 30))), ('q', (('x', 2), ('a', 34))), ('t', (('x', False), ('a', False))), ('r', (('x', 'c'), ('a', 'd'))), ('s', (('x', False), ('a', True))))
                )



    def test_frame_set_index_a(self):
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x', 'y'))

        self.assertEqual(f1.set_index('r').to_pairs(0),
                (('p', (('a', 2), ('b', 30))), ('q', (('a', 2), ('b', 34))), ('r', (('a', 'a'), ('b', 'b'))), ('s', (('a', False), ('b', True))), ('t', (('a', False), ('b', False)))))

        self.assertEqual(f1.set_index('r', drop=True).to_pairs(0),
                (('p', (('a', 2), ('b', 30))), ('q', (('a', 2), ('b', 34))), ('s', (('a', False), ('b', True))), ('t', (('a', False), ('b', False)
                ))))

        f2 = f1.set_index('r', drop=True)

        # in extracting the index, we leave unconnected blocks unchanged.
        self.assertTrue(f1.mloc[[0, 2]].tolist() == f2.mloc.tolist())


    def test_frame_head_tail_a(self):

        # thest of multi threaded apply

        f1 = Frame.from_items(
                zip(range(10), (np.random.rand(1000) for _ in range(10)))
                )
        self.assertEqual(f1.head(3).index.values.tolist(),
                [0, 1, 2])
        self.assertEqual(f1.tail(3).index.values.tolist(),
                [997, 998, 999])



if __name__ == '__main__':
    unittest.main()
#     t = TestUnit()
#     t.test_display_display_columns_b()
