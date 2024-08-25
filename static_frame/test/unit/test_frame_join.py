from __future__ import annotations

import numpy as np

import static_frame as sf
from static_frame import Frame
from static_frame import FrameGO
from static_frame import IndexDate
from static_frame import IndexHierarchy
from static_frame.core.fill_value_auto import FillValueAuto
from static_frame.core.join import join
from static_frame.test.test_case import TestCase

dt64 = np.datetime64

class TestUnit(TestCase):

    #---------------------------------------------------------------------------
    def test_frame_join_a1(self) -> None:

        # joining index to index

        f1 = Frame.from_dict(
                dict(a=(10,10,np.nan,20,20), b=('x','x','y','y','z')),
                index=(0, 1, 2, 'foo', 'x'))
        f2 = Frame.from_dict(
                dict(c=('foo', 'bar'), d=(10, 20)),
                index=('x', 'y'))

        f3 = f1.join_inner(f2, left_depth_level=0, right_depth_level=0, include_index=True)

        self.assertEqual(f3.to_pairs(0),
                (('a', (('x', 20.0),)), ('b', (('x', 'z'),)), ('c', (('x', 'foo'),)), ('d', (('x', 10),)))
                )

        f4 = f1.join_outer(f2,
                left_depth_level=0,
                right_depth_level=0,
                include_index=True,
                ).fillna(None)


        self.assertEqual(f4.to_pairs(0),
                (('a', (((0, None), 10.0), ((1, None), 10.0), ((2, None), None), (('foo', None), 20.0), (('x', 'x'), 20.0), ((None, 'y'), None))), ('b', (((0, None), 'x'), ((1, None), 'x'), ((2, None), 'y'), (('foo', None), 'y'), (('x', 'x'), 'z'), ((None, 'y'), None))), ('c', (((0, None), None), ((1, None), None), ((2, None), None), (('foo', None), None), (('x', 'x'), 'foo'), ((None, 'y'), 'bar'))), ('d', (((0, None), None), ((1, None), None), ((2, None), None), (('foo', None), None), (('x', 'x'), 10.0), ((None, 'y'), 20.0))))
                )

        f5 = f1.join_left(f2,
                left_depth_level=0,
                right_depth_level=0,
                include_index=True,
                ).fillna(None)

        self.assertEqual(f5.to_pairs(0),
                (('a', ((0, 10.0), (1, 10.0), (2, None), ('foo', 20.0), ('x', 20.0))), ('b', ((0, 'x'), (1, 'x'), (2, 'y'), ('foo', 'y'), ('x', 'z'))), ('c', ((0, None), (1, None), (2, None), ('foo', None), ('x', 'foo'))), ('d', ((0, None), (1, None), (2, None), ('foo', None), ('x', 10.0))))
                )

        f6 = f1.join_right(f2,
                left_depth_level=0,
                right_depth_level=0,
                include_index=True,
                ).fillna(None)
        self.assertEqual(f6.to_pairs(0),
                (('a', (('x', 20.0), ('y', None))), ('b', (('x', 'z'), ('y', None))), ('c', (('x', 'foo'), ('y', 'bar'))), ('d', (('x', 10), ('y', 20))))
                )

    def test_frame_join_a2(self) -> None:

        # joining index to index

        f1 = Frame.from_dict(
                dict(a=(10,10,np.nan,20,20), b=('x','x','y','y','z')),
                index=(0, 1, 2, 'foo', 'x'))
        f2 = Frame.from_dict(
                dict(c=('foo', 'bar'), d=(10, 20)),
                index=('x', 'y'))

        f3 = f1.join_inner(f2, left_depth_level=0, right_depth_level=0, include_index=False)
        self.assertEqual(f3.to_pairs(0),
                (('a', ((0, 20.0),)), ('b', ((0, 'z'),)), ('c', ((0, 'foo'),)), ('d', ((0, 10),)))
                )

    def test_frame_join_b1(self) -> None:

        # joining on column to column

        f1 = Frame.from_dict(
            {
            'LastName': ('Raf', 'Jon', 'Hei', 'Rob', 'Smi', 'Wil'),
            'DepartmentID': (31, 33, 33, 34, 34, None),
            },
            index=tuple('abcdef'),
            )

        f2 = Frame.from_dict(
            {
            'DepartmentID': (31, 33, 34, 35),
            'DepartmentName': ('Sales', 'Engineering', 'Clerical', 'Marketing'),
            },
            index=range(10, 14),
            )

        f3 = f1.join_outer(f2,
                left_columns='DepartmentID',
                left_template='Employee.{}',
                right_columns='DepartmentID',
                right_template='Department.{}',
                fill_value=None,
                include_index=True,
                )
        # <Frame>
        # <Index>     Employee.LastName Employee.Departme... Department.Depart... Department.Depart... <<U25>
        # <Index>
        # ('a', 10)   Raf               31                   31                   Sales
        # ('b', 11)   Jon               33                   33                   Engineering
        # ('c', 11)   Hei               33                   33                   Engineering
        # ('d', 12)   Rob               34                   34                   Clerical
        # ('e', 12)   Smi               34                   34                   Clerical
        # ('f', None) Wil               None                 None                 None
        # (None, 13)  None              None                 35                   Marketing
        # <object>    <object>          <object>             <object>             <object>

        self.assertEqual(f3.shape, (7, 4))
        self.assertEqual(f3.fillna(None).to_pairs(0),
                (('Employee.LastName', ((('a', 10), 'Raf'), (('b', 11), 'Jon'), (('c', 11), 'Hei'), (('d', 12), 'Rob'), (('e', 12), 'Smi'), (('f', None), 'Wil'), ((None, 13), None))), ('Employee.DepartmentID', ((('a', 10), 31), (('b', 11), 33), (('c', 11), 33), (('d', 12), 34), (('e', 12), 34), (('f', None), None), ((None, 13), None))), ('Department.DepartmentID', ((('a', 10), 31.0), (('b', 11), 33.0), (('c', 11), 33.0), (('d', 12), 34.0), (('e', 12), 34.0), (('f', None), None), ((None, 13), 35.0))), ('Department.DepartmentName', ((('a', 10), 'Sales'), (('b', 11), 'Engineering'), (('c', 11), 'Engineering'), (('d', 12), 'Clerical'), (('e', 12), 'Clerical'), (('f', None), None), ((None, 13), 'Marketing'))))
                )

    def test_frame_join_b2(self) -> None:
        # joining on column to column
        f1 = Frame.from_dict(
            {
            'LastName': ('Raf', 'Jon', 'Hei', 'Rob', 'Smi', 'Wil'),
            'DepartmentID': (31, 33, 33, 34, 34, None),
            },
            index=tuple('abcdef'),
            )
        f2 = Frame.from_dict(
            {
            'DepartmentID': (31, 33, 34, 35),
            'DepartmentName': ('Sales', 'Engineering', 'Clerical', 'Marketing'),
            },
            index=range(10, 14),
            )
        f4 = f1.join_inner(f2,
                left_columns='DepartmentID',
                left_template='Employee.{}',
                right_columns='DepartmentID',
                right_template='Department.{}',
                include_index=True,
                )
        self.assertEqual(f4.shape, (5, 4))
        # <Frame>
        # <Index>   Employee.LastName Employee.Departme... Department.Depart... Department.Depart... <<U25>
        # <Index>
        # ('a', 10) Raf               31                   31                   Sales
        # ('b', 11) Jon               33                   33                   Engineering
        # ('c', 11) Hei               33                   33                   Engineering
        # ('d', 12) Rob               34                   34                   Clerical
        # ('e', 12) Smi               34                   34                   Clerical
        # <object>  <<U3>             <object>             <int64>              <<U11>

        self.assertEqual(f4.fillna(None).to_pairs(0),
                (('Employee.LastName', ((('a', 10), 'Raf'), (('b', 11), 'Jon'), (('c', 11), 'Hei'), (('d', 12), 'Rob'), (('e', 12), 'Smi'))), ('Employee.DepartmentID', ((('a', 10), 31), (('b', 11), 33), (('c', 11), 33), (('d', 12), 34), (('e', 12), 34))), ('Department.DepartmentID', ((('a', 10), 31), (('b', 11), 33), (('c', 11), 33), (('d', 12), 34), (('e', 12), 34))), ('Department.DepartmentName', ((('a', 10), 'Sales'), (('b', 11), 'Engineering'), (('c', 11), 'Engineering'), (('d', 12), 'Clerical'), (('e', 12), 'Clerical'))))
                )

    def test_frame_join_b3(self) -> None:

        # joining on column to column

        f1 = Frame.from_dict(
            {
            'LastName': ('Raf', 'Jon', 'Hei', 'Rob', 'Smi', 'Wil'),
            'DepartmentID': (31, 33, 33, 34, 34, None),
            },
            index=tuple('abcdef'),
            )

        f2 = Frame.from_dict(
            {
            'DepartmentID': (31, 33, 34, 35),
            'DepartmentName': ('Sales', 'Engineering', 'Clerical', 'Marketing'),
            },
            index=range(10, 14),
            )
        f5 = f1.join_left(f2,
                left_columns='DepartmentID',
                left_template='Employee.{}',
                right_columns='DepartmentID',
                right_template='Department.{}',
                include_index=True,
                )
        self.assertEqual(f5.shape, (6, 4))
        self.assertEqual(f5.fillna(None).to_pairs(0),
                (('Employee.LastName', ((('a', 10), 'Raf'), (('b', 11), 'Jon'), (('c', 11), 'Hei'), (('d', 12), 'Rob'), (('e', 12), 'Smi'), (('f', None), 'Wil'))), ('Employee.DepartmentID', ((('a', 10), 31), (('b', 11), 33), (('c', 11), 33), (('d', 12), 34), (('e', 12), 34), (('f', None), None))), ('Department.DepartmentID', ((('a', 10), 31.0), (('b', 11), 33.0), (('c', 11), 33.0), (('d', 12), 34.0), (('e', 12), 34.0), (('f', None), None))), ('Department.DepartmentName', ((('a', 10), 'Sales'), (('b', 11), 'Engineering'), (('c', 11), 'Engineering'), (('d', 12), 'Clerical'), (('e', 12), 'Clerical'), (('f', None), None))))
                )

    def test_frame_join_b4(self) -> None:

        # joining on column to column

        f1 = Frame.from_dict(
            {
            'LastName': ('Raf', 'Jon', 'Hei', 'Rob', 'Smi', 'Wil'),
            'DepartmentID': (31, 33, 33, 34, 34, None),
            },
            index=tuple('abcdef'),
            )

        f2 = Frame.from_dict(
            {
            'DepartmentID': (31, 33, 34, 35),
            'DepartmentName': ('Sales', 'Engineering', 'Clerical', 'Marketing'),
            },
            index=range(10, 14),
            )
        # df1.merge(df2, how='right', left_on='DepartmentID', right_on='DepartmentID')

        f6 = f1.join_right(f2,
                left_columns='DepartmentID',
                left_template='Employee.{}',
                right_columns='DepartmentID',
                right_template='Department.{}',
                include_index=True,
                )

        self.assertEqual(f6.shape, (6, 4))
        self.assertEqual(f6.fillna(None).to_pairs(0),
                (('Employee.LastName', ((('a', 10), 'Raf'), (('b', 11), 'Jon'), (('c', 11), 'Hei'), (('d', 12), 'Rob'), (('e', 12), 'Smi'), ((None, 13), None))), ('Employee.DepartmentID', ((('a', 10), 31), (('b', 11), 33), (('c', 11), 33), (('d', 12), 34), (('e', 12), 34), ((None, 13), None))), ('Department.DepartmentID', ((('a', 10), 31), (('b', 11), 33), (('c', 11), 33), (('d', 12), 34), (('e', 12), 34), ((None, 13), 35))), ('Department.DepartmentName', ((('a', 10), 'Sales'), (('b', 11), 'Engineering'), (('c', 11), 'Engineering'), (('d', 12), 'Clerical'), (('e', 12), 'Clerical'), ((None, 13), 'Marketing'))))
                )


    def test_frame_join_c(self) -> None:
        f1 = sf.Frame.from_dict(dict(a=(10,10,20,20,20), b=('x','x','y','y','z')))
        f2 = sf.Frame.from_dict(dict(c=('foo', 'bar'), d=(10, 20)), index=('x', 'y'))

        with self.assertRaises(RuntimeError):
            _ = f1.join_left(f2, left_columns=['a', 'b'], right_depth_level=0)


        f3 = f1.join_left(f2, left_columns='b', right_depth_level=0, include_index=True)
        self.assertEqual(f3.fillna(None).to_pairs(0),
                (('a', (((0, 'x'), 10), ((1, 'x'), 10), ((2, 'y'), 20), ((3, 'y'), 20), ((4, None), 20))), ('b', (((0, 'x'), 'x'), ((1, 'x'), 'x'), ((2, 'y'), 'y'), ((3, 'y'), 'y'), ((4, None), 'z'))), ('c', (((0, 'x'), 'foo'), ((1, 'x'), 'foo'), ((2, 'y'), 'bar'), ((3, 'y'), 'bar'), ((4, None), None))), ('d', (((0, 'x'), 10.0), ((1, 'x'), 10.0), ((2, 'y'), 20.0), ((3, 'y'), 20.0), ((4, None), None))))
                )

        f4 = f1.join_inner(f2, left_columns='b', right_depth_level=0, include_index=True)
        self.assertEqual(f4.to_pairs(0),
                (('a', (((0, 'x'), 10), ((1, 'x'), 10), ((2, 'y'), 20), ((3, 'y'), 20))), ('b', (((0, 'x'), 'x'), ((1, 'x'), 'x'), ((2, 'y'), 'y'), ((3, 'y'), 'y'))), ('c', (((0, 'x'), 'foo'), ((1, 'x'), 'foo'), ((2, 'y'), 'bar'), ((3, 'y'), 'bar'))), ('d', (((0, 'x'), 10), ((1, 'x'), 10), ((2, 'y'), 20), ((3, 'y'), 20))))
                )

        # right is same as inner
        f5 = f1.join_right(f2, left_columns='b', right_depth_level=0, include_index=True)
        self.assertTrue(f5.equals(f4, compare_dtype=True))

        # left is same as outer
        f6 = f1.join_outer(f2, left_columns='b', right_depth_level=0, include_index=True)
        self.assertTrue(f6.equals(f3, compare_dtype=True))

    def test_frame_join_d1(self) -> None:
        index1 = IndexDate.from_date_range('2020-05-04', '2020-05-08')
        index2 = IndexHierarchy.from_product(('A', 'B'), index1)
        f1 = Frame.from_dict(dict(a=tuple(range(10)), b=tuple('pqrstuvwxy')), index=index2, dtypes=(np.int64, np.str_))
        f2 = Frame.from_dict(dict(c=tuple(range(10, 15)), d=tuple('fffgg')), index=index1, dtypes=(np.int64, np.str_))

        f3 = f1.join_left(f2, left_depth_level=1, right_depth_level=0, include_index=True)

        # <Frame>
        # <Index>                              a       b     c       d     <<U1>
        # <Index>
        # (('A', dt64('2020-05-... 0       p     10      f
        # (('A', dt64('2020-05-... 1       q     11      f
        # (('A', dt64('2020-05-... 2       r     12      f
        # (('A', dt64('2020-05-... 3       s     13      g
        # (('A', dt64('2020-05-... 4       t     14      g
        # (('B', dt64('2020-05-... 5       u     10      f
        # (('B', dt64('2020-05-... 6       v     11      f
        # (('B', dt64('2020-05-... 7       w     12      f
        # (('B', dt64('2020-05-... 8       x     13      g
        # (('B', dt64('2020-05-... 9       y     14      g
        # <object>                             <int64> <<U1> <int64> <<U1>
        self.assertEqual(f3.dtypes.values.tolist(),
                [np.dtype('int64'), np.dtype('<U1'), np.dtype('int64'), np.dtype('<U1')]
                )
        self.assertEqual(f3.shape, (10, 4))

        self.assertEqual(
                f3.to_pairs(0),
                (('a', (((('A', dt64('2020-05-04')), dt64('2020-05-04')), 0), ((('A', dt64('2020-05-05')), dt64('2020-05-05')), 1), ((('A', dt64('2020-05-06')), dt64('2020-05-06')), 2), ((('A', dt64('2020-05-07')), dt64('2020-05-07')), 3), ((('A', dt64('2020-05-08')), dt64('2020-05-08')), 4), ((('B', dt64('2020-05-04')), dt64('2020-05-04')), 5), ((('B', dt64('2020-05-05')), dt64('2020-05-05')), 6), ((('B', dt64('2020-05-06')), dt64('2020-05-06')), 7), ((('B', dt64('2020-05-07')), dt64('2020-05-07')), 8), ((('B', dt64('2020-05-08')), dt64('2020-05-08')), 9))), ('b', (((('A', dt64('2020-05-04')), dt64('2020-05-04')), 'p'), ((('A', dt64('2020-05-05')), dt64('2020-05-05')), 'q'), ((('A', dt64('2020-05-06')), dt64('2020-05-06')), 'r'), ((('A', dt64('2020-05-07')), dt64('2020-05-07')), 's'), ((('A', dt64('2020-05-08')), dt64('2020-05-08')), 't'), ((('B', dt64('2020-05-04')), dt64('2020-05-04')), 'u'), ((('B', dt64('2020-05-05')), dt64('2020-05-05')), 'v'), ((('B', dt64('2020-05-06')), dt64('2020-05-06')), 'w'), ((('B', dt64('2020-05-07')), dt64('2020-05-07')), 'x'), ((('B', dt64('2020-05-08')), dt64('2020-05-08')), 'y'))), ('c', (((('A', dt64('2020-05-04')), dt64('2020-05-04')), 10), ((('A', dt64('2020-05-05')), dt64('2020-05-05')), 11), ((('A', dt64('2020-05-06')), dt64('2020-05-06')), 12), ((('A', dt64('2020-05-07')), dt64('2020-05-07')), 13), ((('A', dt64('2020-05-08')), dt64('2020-05-08')), 14), ((('B', dt64('2020-05-04')), dt64('2020-05-04')), 10), ((('B', dt64('2020-05-05')), dt64('2020-05-05')), 11), ((('B', dt64('2020-05-06')), dt64('2020-05-06')), 12), ((('B', dt64('2020-05-07')), dt64('2020-05-07')), 13), ((('B', dt64('2020-05-08')), dt64('2020-05-08')), 14))), ('d', (((('A', dt64('2020-05-04')), dt64('2020-05-04')), 'f'), ((('A', dt64('2020-05-05')), dt64('2020-05-05')), 'f'), ((('A', dt64('2020-05-06')), dt64('2020-05-06')), 'f'), ((('A', dt64('2020-05-07')), dt64('2020-05-07')), 'g'), ((('A', dt64('2020-05-08')), dt64('2020-05-08')), 'g'), ((('B', dt64('2020-05-04')), dt64('2020-05-04')), 'f'), ((('B', dt64('2020-05-05')), dt64('2020-05-05')), 'f'), ((('B', dt64('2020-05-06')), dt64('2020-05-06')), 'f'), ((('B', dt64('2020-05-07')), dt64('2020-05-07')), 'g'), ((('B', dt64('2020-05-08')), dt64('2020-05-08')), 'g'))))
                )

        # inner join is equivalent to left, right, outer
        self.assertTrue(f1.join_inner(f2, left_depth_level=1, right_depth_level=0, include_index=True).equals(f3))
        self.assertTrue(f1.join_outer(f2, left_depth_level=1, right_depth_level=0, include_index=True).equals(f3))

    def test_frame_join_d2(self) -> None:
        index1 = IndexDate.from_date_range('2020-05-04', '2020-05-08')
        index2 = IndexHierarchy.from_product(('A', 'B'), index1)
        f1 = Frame.from_dict(dict(a=tuple(range(10)), b=tuple('pqrstuvwxy')), index=index2)
        f2 = Frame.from_dict(dict(c=tuple(range(10, 15)), d=tuple('fffgg')), index=index1)

        f3 = f1.join_right(f2, left_depth_level=1, right_depth_level=0, include_index=True)
        self.assertEqual([dt.kind for dt in f3.dtypes.values], ['i', 'U', 'i', 'U'])

        self.assertEqual(f3.to_pairs(),
                (('a', (((('A', dt64('2020-05-04')), dt64('2020-05-04')), 0), ((('B', dt64('2020-05-04')), dt64('2020-05-04')), 5), ((('A', dt64('2020-05-05')), dt64('2020-05-05')), 1), ((('B', dt64('2020-05-05')), dt64('2020-05-05')), 6), ((('A', dt64('2020-05-06')), dt64('2020-05-06')), 2), ((('B', dt64('2020-05-06')), dt64('2020-05-06')), 7), ((('A', dt64('2020-05-07')), dt64('2020-05-07')), 3), ((('B', dt64('2020-05-07')), dt64('2020-05-07')), 8), ((('A', dt64('2020-05-08')), dt64('2020-05-08')), 4), ((('B', dt64('2020-05-08')), dt64('2020-05-08')), 9))), ('b', (((('A', dt64('2020-05-04')), dt64('2020-05-04')), 'p'), ((('B', dt64('2020-05-04')), dt64('2020-05-04')), 'u'), ((('A', dt64('2020-05-05')), dt64('2020-05-05')), 'q'), ((('B', dt64('2020-05-05')), dt64('2020-05-05')), 'v'), ((('A', dt64('2020-05-06')), dt64('2020-05-06')), 'r'), ((('B', dt64('2020-05-06')), dt64('2020-05-06')), 'w'), ((('A', dt64('2020-05-07')), dt64('2020-05-07')), 's'), ((('B', dt64('2020-05-07')), dt64('2020-05-07')), 'x'), ((('A', dt64('2020-05-08')), dt64('2020-05-08')), 't'), ((('B', dt64('2020-05-08')), dt64('2020-05-08')), 'y'))), ('c', (((('A', dt64('2020-05-04')), dt64('2020-05-04')), 10), ((('B', dt64('2020-05-04')), dt64('2020-05-04')), 10), ((('A', dt64('2020-05-05')), dt64('2020-05-05')), 11), ((('B', dt64('2020-05-05')), dt64('2020-05-05')), 11), ((('A', dt64('2020-05-06')), dt64('2020-05-06')), 12), ((('B', dt64('2020-05-06')), dt64('2020-05-06')), 12), ((('A', dt64('2020-05-07')), dt64('2020-05-07')), 13), ((('B', dt64('2020-05-07')), dt64('2020-05-07')), 13), ((('A', dt64('2020-05-08')), dt64('2020-05-08')), 14), ((('B', dt64('2020-05-08')), dt64('2020-05-08')), 14))), ('d', (((('A', dt64('2020-05-04')), dt64('2020-05-04')), 'f'), ((('B', dt64('2020-05-04')), dt64('2020-05-04')), 'f'), ((('A', dt64('2020-05-05')), dt64('2020-05-05')), 'f'), ((('B', dt64('2020-05-05')), dt64('2020-05-05')), 'f'), ((('A', dt64('2020-05-06')), dt64('2020-05-06')), 'f'), ((('B', dt64('2020-05-06')), dt64('2020-05-06')), 'f'), ((('A', dt64('2020-05-07')), dt64('2020-05-07')), 'g'), ((('B', dt64('2020-05-07')), dt64('2020-05-07')), 'g'), ((('A', dt64('2020-05-08')), dt64('2020-05-08')), 'g'), ((('B', dt64('2020-05-08')), dt64('2020-05-08')), 'g'))))
                )

    def test_frame_join_e1(self) -> None:
        # matching on hierarchical indices
        index1 = IndexHierarchy.from_product(('A', 'B'), (1, 2, 3, 4, 5))
        index2 = IndexHierarchy.from_labels((('B', 3), ('B', 5), ('A', 2)))
        f1 = Frame.from_dict(dict(a=tuple(range(10)), b=tuple('pqrstuvwxy')),
                index=index1)
        f2 = Frame.from_dict(dict(c=tuple(range(10, 13)), d=tuple('fgh')),
                index=index2)

        f3 = f1.join_left(f2,
                left_depth_level=[0, 1],
                right_depth_level=[0, 1],
                fill_value=None,
                include_index=True,
                )

        self.assertEqual(f3.to_pairs(0),
                (('a', ((('A', 1), 0), (('A', 2), 1), (('A', 3), 2), (('A', 4), 3), (('A', 5), 4), (('B', 1), 5), (('B', 2), 6), (('B', 3), 7), (('B', 4), 8), (('B', 5), 9))), ('b', ((('A', 1), 'p'), (('A', 2), 'q'), (('A', 3), 'r'), (('A', 4), 's'), (('A', 5), 't'), (('B', 1), 'u'), (('B', 2), 'v'), (('B', 3), 'w'), (('B', 4), 'x'), (('B', 5), 'y'))), ('c', ((('A', 1), None), (('A', 2), 12), (('A', 3), None), (('A', 4), None), (('A', 5), None), (('B', 1), None), (('B', 2), None), (('B', 3), 10), (('B', 4), None), (('B', 5), 11))), ('d', ((('A', 1), None), (('A', 2), 'h'), (('A', 3), None), (('A', 4), None), (('A', 5), None), (('B', 1), None), (('B', 2), None), (('B', 3), 'f'), (('B', 4), None), (('B', 5), 'g'))))
                )

    def test_frame_join_e2(self) -> None:
        # matching on hierarchical indices
        index1 = IndexHierarchy.from_product(('A', 'B'), (1, 2, 3, 4, 5))
        index2 = IndexHierarchy.from_labels((('B', 3), ('B', 5), ('A', 2)))
        f1 = Frame.from_dict(dict(a=tuple(range(10)), b=tuple('pqrstuvwxy')),
                index=index1)
        f2 = Frame.from_dict(dict(c=tuple(range(10, 13)), d=tuple('fgh')),
                index=index2)

        f4 = f1.join_left(f2,
                left_depth_level=[0, 1],
                right_depth_level=[0, 1],
                fill_value=None,
                include_index=True,
                )

        self.assertEqual(f4.to_pairs(0),
                (('a', ((('A', 1), 0), (('A', 2), 1), (('A', 3), 2), (('A', 4), 3), (('A', 5), 4), (('B', 1), 5), (('B', 2), 6), (('B', 3), 7), (('B', 4), 8), (('B', 5), 9))), ('b', ((('A', 1), 'p'), (('A', 2), 'q'), (('A', 3), 'r'), (('A', 4), 's'), (('A', 5), 't'), (('B', 1), 'u'), (('B', 2), 'v'), (('B', 3), 'w'), (('B', 4), 'x'), (('B', 5), 'y'))), ('c', ((('A', 1), None), (('A', 2), 12), (('A', 3), None), (('A', 4), None), (('A', 5), None), (('B', 1), None), (('B', 2), None), (('B', 3), 10), (('B', 4), None), (('B', 5), 11))), ('d', ((('A', 1), None), (('A', 2), 'h'), (('A', 3), None), (('A', 4), None), (('A', 5), None), (('B', 1), None), (('B', 2), None), (('B', 3), 'f'), (('B', 4), None), (('B', 5), 'g'))))
                )

    def test_frame_join_f1(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(10,10,np.nan,20,20), b=('x','x','y','y','z')),
                index=tuple('abcde'))

        f2 = Frame.from_dict(
                dict(c=('y', 'y', 'w'), d=(1000, 3000, 2000)),
                index=('q', 'p', 'r'))

        f3 = f1.join_left(f2, left_columns='b', right_columns='c', include_index=True, fill_value=None)
        self.assertEqual(f3.shape, (7, 4))

        self.assertEqual(f3.fillna(None).to_pairs(0),
                (('a', ((('a', None), 10.0), (('b', None), 10.0), (('c', 'q'), None), (('c', 'p'), None), (('d', 'q'), 20.0), (('d', 'p'), 20.0), (('e', None), 20.0))), ('b', ((('a', None), 'x'), (('b', None), 'x'), (('c', 'q'), 'y'), (('c', 'p'), 'y'), (('d', 'q'), 'y'), (('d', 'p'), 'y'), (('e', None), 'z'))), ('c', ((('a', None), None), (('b', None), None), (('c', 'q'), 'y'), (('c', 'p'), 'y'), (('d', 'q'), 'y'), (('d', 'p'), 'y'), (('e', None), None))), ('d', ((('a', None), None), (('b', None), None), (('c', 'q'), 1000), (('c', 'p'), 3000), (('d', 'q'), 1000), (('d', 'p'), 3000), (('e', None), None))))
                )

    def test_frame_join_f2(self) -> None:
        f1 = Frame.from_dict(
                dict(a=(10,10,-1,20,20), b=('x','x','y','y','z')),
                index=tuple('abcde'))

        f2 = Frame.from_dict(
                dict(c=('y', 'y', 'w'), d=(1000, 3000, 2000)),
                index=('q', 'p', 'r'))

        f3 = f1.join_right(f2, left_columns='b', right_columns='c', fill_value=None, include_index=True)

        self.assertEqual(f3.fillna(None).to_pairs(0),
                (('a', ((('c', 'q'), -1), (('d', 'q'), 20), (('c', 'p'), -1), (('d', 'p'), 20), ((None, 'r'), None))), ('b', ((('c', 'q'), 'y'), (('d', 'q'), 'y'), (('c', 'p'), 'y'), (('d', 'p'), 'y'), ((None, 'r'), None))), ('c', ((('c', 'q'), 'y'), (('d', 'q'), 'y'), (('c', 'p'), 'y'), (('d', 'p'), 'y'), ((None, 'r'), 'w'))), ('d', ((('c', 'q'), 1000), (('d', 'q'), 1000), (('c', 'p'), 3000), (('d', 'p'), 3000), ((None, 'r'), 2000))))
                )

    def test_frame_join_f3(self) -> None:
        # column on column
        f1 = Frame.from_dict(
                dict(a=(10,10,np.nan,20,20), b=('x','x','y','y','z')),
                index=tuple('abcde'))

        f2 = Frame.from_dict(
                dict(c=('y', 'y', 'w'), d=(1000, 3000, 2000)),
                index=('q', 'p', 'r'))

        f5 = f1.join_inner(f2, left_columns='b', right_columns='c', include_index=True)
        self.assertEqual(f5.fillna(None).to_pairs(0),
                (('a', ((('c', 'q'), None), (('c', 'p'), None), (('d', 'q'), 20.0), (('d', 'p'), 20.0))), ('b', ((('c', 'q'), 'y'), (('c', 'p'), 'y'), (('d', 'q'), 'y'), (('d', 'p'), 'y'))), ('c', ((('c', 'q'), 'y'), (('c', 'p'), 'y'), (('d', 'q'), 'y'), (('d', 'p'), 'y'))), ('d', ((('c', 'q'), 1000), (('c', 'p'), 3000), (('d', 'q'), 1000), (('d', 'p'), 3000))))
                )

    def test_frame_join_f4(self) -> None:
        # column on column
        f1 = Frame.from_dict(
                dict(a=(10,10,-1,20,20), b=('x','x','y','y','z')),
                index=tuple('abcde'))

        f2 = Frame.from_dict(
                dict(c=('y', 'y', 'w'), d=(1000, 3000, 2000)),
                index=('q', 'p', 'r'))

        f3 = f1.join_outer(f2, left_columns='b', right_columns='c', fill_value=None, include_index=True)
        self.assertEqual(f3.shape, (8, 4))

        self.assertEqual(f3.to_pairs(0),
                (('a', ((('a', None), 10), (('b', None), 10), (('c', 'q'), -1), (('c', 'p'), -1), (('d', 'q'), 20), (('d', 'p'), 20), (('e', None), 20), ((None, 'r'), None))), ('b', ((('a', None), 'x'), (('b', None), 'x'), (('c', 'q'), 'y'), (('c', 'p'), 'y'), (('d', 'q'), 'y'), (('d', 'p'), 'y'), (('e', None), 'z'), ((None, 'r'), None))), ('c', ((('a', None), None), (('b', None), None), (('c', 'q'), 'y'), (('c', 'p'), 'y'), (('d', 'q'), 'y'), (('d', 'p'), 'y'), (('e', None), None), ((None, 'r'), 'w'))), ('d', ((('a', None), None), (('b', None), None), (('c', 'q'), 1000), (('c', 'p'), 3000), (('d', 'q'), 1000), (('d', 'p'), 3000), (('e', None), None), ((None, 'r'), 2000))))
                )

    def test_frame_join_g(self) -> None:

        f1 = Frame.from_records(
                ((1,'apple'),
                (2,'banana'),
                (3,'kiwi fruit'),
                (4,'strawberries'),
                (5,'flour'),
                (6,'fruit juice'),
                (7,'butter'),
                (8,'sugar')),
                columns=('ingredient_id', 'ingredient_name'),
                index=tuple('abcdefgh'))

        f2 = Frame.from_records(
                ((1,'Apple Crumble'),
                (2,'Fruit Salad',),
                (3,'Weekday Risotto',),
                (4,'Beans Chili',),
                (5,'Chicken Casserole',)),
                columns=('recipe_id', 'recipe_name'),
                index=tuple('stuvw')
                )

        f3 = Frame.from_records(
                ((1,1),(1,5),(1,7),(1,8),(2,6),(2,2),(2,1),(2,3),(2,4)),
                index=tuple('ijklmnopq'),
                columns=('recipe_id', 'ingredient_id')
                )

        f4 = f2.join_inner(f3,
                left_columns='recipe_id',
                right_columns='recipe_id',
                right_template='new_{}',
                include_index=True,
                )
        self.assertEqual(f4.to_pairs(0),
                (('recipe_id', ((('s', 'i'), 1), (('s', 'j'), 1), (('s', 'k'), 1), (('s', 'l'), 1), (('t', 'm'), 2), (('t', 'n'), 2), (('t', 'o'), 2), (('t', 'p'), 2), (('t', 'q'), 2))), ('recipe_name', ((('s', 'i'), 'Apple Crumble'), (('s', 'j'), 'Apple Crumble'), (('s', 'k'), 'Apple Crumble'), (('s', 'l'), 'Apple Crumble'), (('t', 'm'), 'Fruit Salad'), (('t', 'n'), 'Fruit Salad'), (('t', 'o'), 'Fruit Salad'), (('t', 'p'), 'Fruit Salad'), (('t', 'q'), 'Fruit Salad'))), ('new_recipe_id', ((('s', 'i'), 1), (('s', 'j'), 1), (('s', 'k'), 1), (('s', 'l'), 1), (('t', 'm'), 2), (('t', 'n'), 2), (('t', 'o'), 2), (('t', 'p'), 2), (('t', 'q'), 2))), ('new_ingredient_id', ((('s', 'i'), 1), (('s', 'j'), 5), (('s', 'k'), 7), (('s', 'l'), 8), (('t', 'm'), 6), (('t', 'n'), 2), (('t', 'o'), 1), (('t', 'p'), 3), (('t', 'q'), 4))))
                )

        f7 = f2.join_outer(f3,
                left_columns='recipe_id',
                right_columns='recipe_id',
                right_template='new_{}',
                include_index=True,
                )

        self.assertEqual(f7.fillna(None).to_pairs(0),
                (('recipe_id', ((('s', 'i'), 1), (('s', 'j'), 1), (('s', 'k'), 1), (('s', 'l'), 1), (('t', 'm'), 2), (('t', 'n'), 2), (('t', 'o'), 2), (('t', 'p'), 2), (('t', 'q'), 2), (('u', None), 3), (('v', None), 4), (('w', None), 5))), ('recipe_name', ((('s', 'i'), 'Apple Crumble'), (('s', 'j'), 'Apple Crumble'), (('s', 'k'), 'Apple Crumble'), (('s', 'l'), 'Apple Crumble'), (('t', 'm'), 'Fruit Salad'), (('t', 'n'), 'Fruit Salad'), (('t', 'o'), 'Fruit Salad'), (('t', 'p'), 'Fruit Salad'), (('t', 'q'), 'Fruit Salad'), (('u', None), 'Weekday Risotto'), (('v', None), 'Beans Chili'), (('w', None), 'Chicken Casserole'))), ('new_recipe_id', ((('s', 'i'), 1.0), (('s', 'j'), 1.0), (('s', 'k'), 1.0), (('s', 'l'), 1.0), (('t', 'm'), 2.0), (('t', 'n'), 2.0), (('t', 'o'), 2.0), (('t', 'p'), 2.0), (('t', 'q'), 2.0), (('u', None), None), (('v', None), None), (('w', None), None))), ('new_ingredient_id', ((('s', 'i'), 1.0), (('s', 'j'), 5.0), (('s', 'k'), 7.0), (('s', 'l'), 8.0), (('t', 'm'), 6.0), (('t', 'n'), 2.0), (('t', 'o'), 1.0), (('t', 'p'), 3.0), (('t', 'q'), 4.0), (('u', None), None), (('v', None), None), (('w', None), None))))
                )


        f5 = f2.join_right(f3,
                left_columns='recipe_id',
                right_columns='recipe_id',
                right_template='new_{}',
                include_index=True,
                )

        self.assertEqual(f5.to_pairs(0),
                (('recipe_id', ((('s', 'i'), 1), (('s', 'j'), 1), (('s', 'k'), 1), (('s', 'l'), 1), (('t', 'm'), 2), (('t', 'n'), 2), (('t', 'o'), 2), (('t', 'p'), 2), (('t', 'q'), 2))), ('recipe_name', ((('s', 'i'), 'Apple Crumble'), (('s', 'j'), 'Apple Crumble'), (('s', 'k'), 'Apple Crumble'), (('s', 'l'), 'Apple Crumble'), (('t', 'm'), 'Fruit Salad'), (('t', 'n'), 'Fruit Salad'), (('t', 'o'), 'Fruit Salad'), (('t', 'p'), 'Fruit Salad'), (('t', 'q'), 'Fruit Salad'))), ('new_recipe_id', ((('s', 'i'), 1), (('s', 'j'), 1), (('s', 'k'), 1), (('s', 'l'), 1), (('t', 'm'), 2), (('t', 'n'), 2), (('t', 'o'), 2), (('t', 'p'), 2), (('t', 'q'), 2))), ('new_ingredient_id', ((('s', 'i'), 1), (('s', 'j'), 5), (('s', 'k'), 7), (('s', 'l'), 8), (('t', 'm'), 6), (('t', 'n'), 2), (('t', 'o'), 1), (('t', 'p'), 3), (('t', 'q'), 4))))
                )


        f6 = f2.join_left(f3,
                left_columns='recipe_id',
                right_columns='recipe_id',
                right_template='new_{}',
                include_index=True,
                )

        self.assertEqual(f6.fillna(None).to_pairs(0),
                (('recipe_id', ((('s', 'i'), 1), (('s', 'j'), 1), (('s', 'k'), 1), (('s', 'l'), 1), (('t', 'm'), 2), (('t', 'n'), 2), (('t', 'o'), 2), (('t', 'p'), 2), (('t', 'q'), 2), (('u', None), 3), (('v', None), 4), (('w', None), 5))), ('recipe_name', ((('s', 'i'), 'Apple Crumble'), (('s', 'j'), 'Apple Crumble'), (('s', 'k'), 'Apple Crumble'), (('s', 'l'), 'Apple Crumble'), (('t', 'm'), 'Fruit Salad'), (('t', 'n'), 'Fruit Salad'), (('t', 'o'), 'Fruit Salad'), (('t', 'p'), 'Fruit Salad'), (('t', 'q'), 'Fruit Salad'), (('u', None), 'Weekday Risotto'), (('v', None), 'Beans Chili'), (('w', None), 'Chicken Casserole'))), ('new_recipe_id', ((('s', 'i'), 1.0), (('s', 'j'), 1.0), (('s', 'k'), 1.0), (('s', 'l'), 1.0), (('t', 'm'), 2.0), (('t', 'n'), 2.0), (('t', 'o'), 2.0), (('t', 'p'), 2.0), (('t', 'q'), 2.0), (('u', None), None), (('v', None), None), (('w', None), None))), ('new_ingredient_id', ((('s', 'i'), 1.0), (('s', 'j'), 5.0), (('s', 'k'), 7.0), (('s', 'l'), 8.0), (('t', 'm'), 6.0), (('t', 'n'), 2.0), (('t', 'o'), 1.0), (('t', 'p'), 3.0), (('t', 'q'), 4.0), (('u', None), None), (('v', None), None), (('w', None), None))))
                )

    def test_frame_join_h1(self) -> None:

        f1 = sf.Frame.from_dict(dict(a=(10,10,20,20,20), b=('x','x','y','y','z')))
        f2 = sf.Frame.from_dict(dict(c=('foo', 'bar'), d=(10, 20)), index=('x', 'y'))
        # df1 = f1.to_pandas()
        # df2 = f2.to_pandas()
        #df1.merge(df2, left_on='b', right_index=True)

        f3 = f2.join_inner(f1, left_depth_level=0, right_depth_level=0)
        self.assertEqual(f3.to_pairs(0),
                (('c', ()), ('d', ()), ('a', ()), ('b', ()))
                )

    def test_frame_join_h2(self) -> None:

        f1 = sf.Frame.from_dict(dict(a=(10,10,20,20,20), b=('x','x','y','y','z')))
        f2 = sf.Frame.from_dict(dict(c=('foo', 'bar'), d=(10, 20)), index=('x', 'y'))

        f4 = f2.join_right(f1,
                left_depth_level=0,
                right_depth_level=0,
                fill_value=None,
                include_index=True,
                )
        self.assertEqual(f4.to_pairs(0),
                (('c', ((0, None), (1, None), (2, None), (3, None), (4, None))), ('d', ((0, None), (1, None), (2, None), (3, None), (4, None))), ('a', ((0, 10), (1, 10), (2, 20), (3, 20), (4, 20))), ('b', ((0, 'x'), (1, 'x'), (2, 'y'), (3, 'y'), (4, 'z'))))
                )

    def test_frame_join_h3(self) -> None:

        f1 = sf.Frame.from_dict(dict(a=(10,10,20,20,20), b=('x','x','y','y','z')))
        f2 = sf.Frame.from_dict(dict(c=('foo', 'bar'), d=(10, 20)), index=('x', 'y'))

        f5 = f2.join_left(f1,
                left_depth_level=0,
                right_depth_level=0,
                fill_value=None,
                include_index=True,
                )
        self.assertEqual(f5.to_pairs(0),
                (('c', (('x', 'foo'), ('y', 'bar'))), ('d', (('x', 10), ('y', 20))), ('a', (('x', None), ('y', None))), ('b', (('x', None), ('y', None))))
                )

    def test_frame_join_h4(self) -> None:

        f1 = sf.Frame.from_dict(dict(a=(10,10,20,20,20), b=('x','x','y','y','z')))
        f2 = sf.Frame.from_dict(dict(c=('foo', 'bar'), d=(10, 20)), index=('x', 'y'))

        f3 = f2.join_outer(f1,
                left_depth_level=0,
                right_depth_level=0,
                fill_value=None,
                include_index=True,
                )
        self.assertEqual(f3.to_pairs(0),
                (('c', ((('x', None), 'foo'), (('y', None), 'bar'), ((None, 0), None), ((None, 1), None), ((None, 2), None), ((None, 3), None), ((None, 4), None))), ('d', ((('x', None), 10), (('y', None), 20), ((None, 0), None), ((None, 1), None), ((None, 2), None), ((None, 3), None), ((None, 4), None))), ('a', ((('x', None), None), (('y', None), None), ((None, 0), 10), ((None, 1), 10), ((None, 2), 20), ((None, 3), 20), ((None, 4), 20))), ('b', ((('x', None), None), (('y', None), None), ((None, 0), 'x'), ((None, 1), 'x'), ((None, 2), 'y'), ((None, 3), 'y'), ((None, 4), 'z'))))
                )

    def test_frame_join_h5(self) -> None:

        f1 = sf.Frame.from_dict(dict(a=(10,10,20,20,20), b=('x','x','y','y','z')))
        f2 = sf.Frame.from_dict(dict(c=('foo', 'bar'), d=(10, 20)), index=('x', 'y'))
        f3 = f2.join_outer(f1, left_depth_level=0, right_columns='b', fill_value=FillValueAuto)
        self.assertEqual(f3.to_pairs(),
                (('c', ((0, 'foo'), (1, 'foo'), (2, 'bar'), (3, 'bar'), (4, ''))), ('d', ((0, 10), (1, 10), (2, 20), (3, 20), (4, 0))), ('a', ((0, 10), (1, 10), (2, 20), (3, 20), (4, 20))), ('b', ((0, 'x'), (1, 'x'), (2, 'y'), (3, 'y'), (4, 'z'))))
                )

    def test_frame_join_i(self) -> None:

        f1 = Frame.from_dict(
                dict(a=(10,10,20,20), b=('x','x','y','z')),
                index=('a', 'b', 'c', 'd'))
        f2 = Frame.from_dict(
                dict(c=('foo', 'bar'), d=(10, 20)),
                index=('c', 'd'))

        f3 = f1.join_left(f2, left_depth_level=0,
                right_depth_level=0,
                fill_value=None,
                include_index=True,
                )

        self.assertEqual(f3.to_pairs(0),
                (('a', (('a', 10), ('b', 10), ('c', 20), ('d', 20))), ('b', (('a', 'x'), ('b', 'x'), ('c', 'y'), ('d', 'z'))), ('c', (('a', None), ('b', None), ('c', 'foo'), ('d', 'bar'))), ('d', (('a', None), ('b', None), ('c', 10), ('d', 20))))
                )

        f4 = f1.join_inner(f2, left_depth_level=0,
                right_depth_level=0,
                fill_value=None,
                include_index=True,
                )
        self.assertEqual( f4.to_pairs(0),
                (('a', (('c', 20), ('d', 20))), ('b', (('c', 'y'), ('d', 'z'))), ('c', (('c', 'foo'), ('d', 'bar'))), ('d', (('c', 10), ('d', 20))))
                )

    def test_frame_join_j(self) -> None:

        f1 = sf.Frame.from_dict(dict(a=(10,10,20,20,20), b=('x','x','y','y','z')))
        f2 = sf.Frame.from_dict(dict(c=('foo', 'bar'), d=(10, 20)), index=('x', 'y'))

        f3 = f2.join_left(f1, left_depth_level=0, right_columns='b', include_index=True)

        self.assertEqual(f3.to_pairs(0),
                (('c', ((('x', 0), 'foo'), (('x', 1), 'foo'), (('y', 2), 'bar'), (('y', 3), 'bar'))), ('d', ((('x', 0), 10), (('x', 1), 10), (('y', 2), 20), (('y', 3), 20))), ('a', ((('x', 0), 10), (('x', 1), 10), (('y', 2), 20), (('y', 3), 20))), ('b', ((('x', 0), 'x'), (('x', 1), 'x'), (('y', 2), 'y'), (('y', 3), 'y'))))
                )

    def test_frame_join_k(self) -> None:
        f1 = sf.Frame.from_dict(dict(a=(10,10,20,20,20), b=('x','x','y','y','z')))
        f2 = sf.Frame.from_dict(dict(c=('foo', 'bar'), d=(10, 20)), index=('x', 'y'))

        with self.assertRaises(RuntimeError):
            join(f1, f2, join_type=None)
        with self.assertRaises(RuntimeError):
            join(f1, f2, join_type=None, left_depth_level=0)

        with self.assertRaises(NotImplementedError):
            join(f1, f2, join_type=None, left_depth_level=0, right_depth_level=0)

    def test_frame_join_l(self) -> None:
        f1 = sf.Frame.from_dict(dict(a=(10, 20), b=('y','z')))
        f2 = sf.Frame.from_dict(dict(c=('foo', 'bar'), d=(10, 20)), index=('x', 'y'))

        f2 = f1.join_inner(f2,
                left_columns='a',
                right_columns='d',
                include_index=True,
                )
        self.assertEqual(f2.to_pairs(),
                (('a', ((0, 10), (1, 20))), ('b', ((0, 'y'), (1, 'z'))), ('c', ((0, 'foo'), (1, 'bar'))), ('d', ((0, 10), (1, 20))))
                )


    def test_frame_join_m1(self) -> None:

        f1 = sf.Frame.from_dict_records([
                {"a1": "111", "b": "R"},
                {"a1": "555", "b": "S"},
                {"a1": "000", "b": "B"},
                {"a1": "333", "b": "C"},
                {"a1": "444", "b": "D"},
                ])

        f2 = sf.Frame.from_dict_records([
                {"a2": "111", "c": 1111},
                {"a2": "555", "c": 1111},
                {"a2": "333", "c": 3333},
                {"a2": "444", "c": 4444},
                ])

        f3 = f1.join_left(f2, left_columns='a1', right_columns='a2', fill_value='')

        # <Frame>
        # <Index> a1    b     a2    c        <<U2>
        # <Index>
        # 0       111   R     111   1111
        # 1       555   S     555   1111
        # 2       000   B
        # 3       333   C     333   3333
        # 4       444   D     444   4444
        # <int64> <<U3> <<U1> <<U3> <object>

        self.assertEqual(f3.to_pairs(),
                (('a1', ((0, '111'), (1, '555'), (2, '000'), (3, '333'), (4, '444'))), ('b', ((0, 'R'), (1, 'S'), (2, 'B'), (3, 'C'), (4, 'D'))), ('a2', ((0, '111'), (1, '555'), (2, ''), (3, '333'), (4, '444'))), ('c', ((0, 1111), (1, 1111), (2, ''), (3, 3333), (4, 4444))))
                )


    def test_frame_join_m2(self) -> None:

        f1 = sf.Frame.from_dict_records([
                {"a1": "111", "b": "R"},
                {"a1": "555", "b": "S"},
                {"a1": "000", "b": "B"},
                {"a1": "333", "b": "C"},
                {"a1": "444", "b": "D"},
                ])

        f2 = sf.Frame.from_dict_records([
                {"a2": "111", "c": 1111},
                {"a2": "555", "c": 1111},
                {"a2": "333", "c": 3333},
                {"a2": "444", "c": 4444},
                ])

        f3 = f2.join_left(f1, left_columns='a2', right_columns='a1', fill_value='')
        self.assertEqual(f3.shape, (4, 4))
        self.assertEqual(f3.to_pairs(),
                (('a2', ((0, '111'), (1, '555'), (2, '333'), (3, '444'))), ('c', ((0, 1111), (1, 1111), (2, 3333), (3, 4444))), ('a1', ((0, '111'), (1, '555'), (2, '333'), (3, '444'))), ('b', ((0, 'R'), (1, 'S'), (2, 'C'), (3, 'D'))))
                )

    def test_frame_join_m3(self) -> None:

        f1 = sf.Frame.from_dict_records([
                {"a1": "111", "b": "R"},
                {"a1": "555", "b": "S"},
                {"a1": "000", "b": "B"},
                {"a1": "333", "b": "C"},
                {"a1": "444", "b": "D"},
                ])

        f2 = sf.Frame.from_dict_records([
                {"a2": "111", "c": 1111},
                {"a2": "555", "c": 1111},
                {"a2": "333", "c": 3333},
                {"a2": "444", "c": 4444},
                ])

        f3 = f1.join_right(f2, left_columns='a1', right_columns='a2', fill_value='')

        # <Frame>
        # <Index> a1    b     a2    c       <<U2>
        # <Index>
        # 0       111   R     111   1111
        # 1       555   S     555   1111
        # 2       333   C     333   3333
        # 3       444   D     444   4444
        # <int64> <<U3> <<U1> <<U3> <int64>

        self.assertEqual(f3.shape, (4, 4))
        self.assertEqual(f3.to_pairs(),
                (('a1', ((0, '111'), (1, '555'), (2, '333'), (3, '444'))), ('b', ((0, 'R'), (1, 'S'), (2, 'C'), (3, 'D'))), ('a2', ((0, '111'), (1, '555'), (2, '333'), (3, '444'))), ('c', ((0, 1111), (1, 1111), (2, 3333), (3, 4444))))
                )


    def test_frame_join_n1(self) -> None:
        f1 = sf.Frame.from_dict_records([
                {"a1": "111", "b": "R"},
                {"a1": "555", "b": "S"},
                {"a1": "000", "b": "B"}, # exclusive
                {"a1": "333", "b": "C"},
                {"a1": "444", "b": "D"},
                {"a1": "555", "b": "X"},
                {"a1": "555", "b": "Y"},
                ])

        f2 = sf.Frame.from_dict_records([
                {"a2": "444", "c": 4444},
                {"a2": "555", "c": 5555},
                {"a2": "111", "c": 1111},
                {"a2": "333", "c": 3333},
                {"a2": "555", "c": 2222},
                {"a2": "888", "c": 8888}, # exclusive
                ])

        f3 = f1.join_left(f2, left_columns='a1', right_columns='a2', fill_value='')
        self.assertEqual([dt.kind for dt in f3.dtypes.values], ['U', 'U', 'U', 'O'])

        # <Frame>
        # <Index> a1    b     a2    c        <<U2>
        # <Index>
        # 0       111   R     111   1111
        # 1       555   S     555   5555
        # 2       555   S     555   2222
        # 3       000   B
        # 4       333   C     333   3333
        # 5       444   D     444   4444
        # 6       555   X     555   5555
        # 7       555   X     555   2222
        # 8       555   Y     555   5555
        # 9       555   Y     555   2222
        # <int64> <<U3> <<U1> <<U3> <object>

        self.assertEqual(f3.to_pairs(),
                (('a1', ((0, '111'), (1, '555'), (2, '555'), (3, '000'), (4, '333'), (5, '444'), (6, '555'), (7, '555'), (8, '555'), (9, '555'))), ('b', ((0, 'R'), (1, 'S'), (2, 'S'), (3, 'B'), (4, 'C'), (5, 'D'), (6, 'X'), (7, 'X'), (8, 'Y'), (9, 'Y'))), ('a2', ((0, '111'), (1, '555'), (2, '555'), (3, ''), (4, '333'), (5, '444'), (6, '555'), (7, '555'), (8, '555'), (9, '555'))), ('c', ((0, 1111), (1, 5555), (2, 2222), (3, ''), (4, 3333), (5, 4444), (6, 5555), (7, 2222), (8, 5555), (9, 2222)))))


    def test_frame_join_n2(self) -> None:
        f1 = sf.Frame.from_dict_records([
                {"a1": "111", "b": "R"},
                {"a1": "555", "b": "S"},
                {"a1": "000", "b": "B"}, # exclusive
                {"a1": "333", "b": "C"},
                {"a1": "444", "b": "D"},
                {"a1": "666", "b": "X"}, # exclusive
                ])

        f2 = sf.Frame.from_dict_records([
                {"a2": "444", "c": 4444},
                {"a2": "555", "c": 5555},
                {"a2": "111", "c": 1111},
                {"a2": "333", "c": 3333},
                {"a2": "888", "c": 8888}, # exclusive
                ])

        f3 = f1.join_left(f2, left_columns='a1', right_columns='a2', fill_value='')
        self.assertEqual([dt.kind for dt in f3.dtypes.values], ['U', 'U', 'U', 'O'])

        # <Frame>
        # <Index> a1    b     a2    c        <<U2>
        # <Index>
        # 0       111   R     111   1111
        # 1       555   S     555   5555
        # 2       000   B
        # 3       333   C     333   3333
        # 4       444   D     444   4444
        # 5       666   X
        # <int64> <<U3> <<U1> <<U3> <object>

        self.assertEqual(f3.to_pairs(),
                (('a1', ((0, '111'), (1, '555'), (2, '000'), (3, '333'), (4, '444'), (5, '666'))), ('b', ((0, 'R'), (1, 'S'), (2, 'B'), (3, 'C'), (4, 'D'), (5, 'X'))), ('a2', ((0, '111'), (1, '555'), (2, ''), (3, '333'), (4, '444'), (5, ''))), ('c', ((0, 1111), (1, 5555), (2, ''), (3, 3333), (4, 4444), (5, ''))))
                )


    def test_frame_join_n3(self) -> None:
        f1 = sf.Frame.from_dict_records([
                {"a1": "111", "b": "R"},
                {"a1": "555", "b": "S"},
                {"a1": "000", "b": "B"}, # exclusive
                {"a1": "333", "b": "C"},
                {"a1": "444", "b": "D"},
                {"a1": "666", "b": "X"}, # exclusive
                ])

        f2 = sf.Frame.from_dict_records([
                {"a2": "444", "c": 4444},
                {"a2": "555", "c": 5555},
                {"a2": "111", "c": 1111},
                {"a2": "333", "c": 3333},
                {"a2": "888", "c": 8888}, # exclusive
                ])

        f3 = f1.join_inner(f2, left_columns='a1', right_columns='a2', fill_value='')
        self.assertEqual([dt.kind for dt in f3.dtypes.values], ['U', 'U', 'U', 'i'])

        # <Frame>
        # <Index> a1    b     a2    c       <<U2>
        # <Index>
        # 0       111   R     111   1111
        # 1       555   S     555   5555
        # 2       333   C     333   3333
        # 3       444   D     444   4444
        # <int64> <<U3> <<U1> <<U3> <int64>

        self.assertEqual(f3.to_pairs(),
                (('a1', ((0, '111'), (1, '555'), (2, '333'), (3, '444'))),
                 ('b', ((0, 'R'), (1, 'S'), (2, 'C'), (3, 'D'))),
                 ('a2', ((0, '111'), (1, '555'), (2, '333'), (3, '444'))),
                 ('c', ((0, 1111), (1, 5555), (2, 3333), (3, 4444))))
                 )


    def test_frame_join_n4(self) -> None:
        f1 = sf.Frame.from_dict_records([
                {"a1": "111", "b": "R"},
                {"a1": "555", "b": "S"},
                {"a1": "000", "b": "B"}, # exclusive
                {"a1": "333", "b": "C"},
                {"a1": "444", "b": "D"},
                {"a1": "666", "b": "X"}, # exclusive
                ])

        f2 = sf.Frame.from_dict_records([
                {"a2": "444", "c": 4444},
                {"a2": "555", "c": 5555},
                {"a2": "111", "c": 1111},
                {"a2": "333", "c": 3333},
                {"a2": "888", "c": 8888}, # exclusive
                ])

        f3 = f1.join_outer(f2, left_columns='a1', right_columns='a2', fill_value='')
        self.assertEqual([dt.kind for dt in f3.dtypes.values], ['U', 'U', 'U', 'O'])
        # <Frame>
        # <Index> a1    b     a2    c        <<U2>
        # <Index>
        # 0       111   R     111   1111
        # 1       555   S     555   5555
        # 2       000   B
        # 3       333   C     333   3333
        # 4       444   D     444   4444
        # 5       666   X
        # 6                   888   8888
        # <int64> <<U3> <<U1> <<U3> <object>

        self.assertEqual(f3.to_pairs(),
                (('a1', ((0, '111'), (1, '555'), (2, '000'), (3, '333'), (4, '444'), (5, '666'), (6, ''))), ('b', ((0, 'R'), (1, 'S'), (2, 'B'), (3, 'C'), (4, 'D'), (5, 'X'), (6, ''))), ('a2', ((0, '111'), (1, '555'), (2, ''), (3, '333'), (4, '444'), (5, ''), (6, '888'))), ('c', ((0, 1111), (1, 5555), (2, ''), (3, 3333), (4, 4444), (5, ''), (6, 8888))))
                )

    def test_frame_join_n5(self) -> None:
        f1 = sf.Frame.from_dict_records([
                {"a1": "111", "b": "R"},
                {"a1": "555", "b": "S"},
                {"a1": "000", "b": "B"}, # exclusive
                {"a1": "333", "b": "C"},
                {"a1": "333", "b": "Q"},
                {"a1": "666", "b": "X"}, # exclusive
                ])

        f2 = sf.Frame.from_dict_records([
                {"a2": "444", "c": 4444},
                {"a2": "333", "c": 7777},
                {"a2": "555", "c": 5555},
                {"a2": "111", "c": 1111},
                {"a2": "111", "c": 2222},
                {"a2": "333", "c": 3333},
                {"a2": "888", "c": 8888}, # exclusive
                ])

        f3 = f1.join_outer(f2, left_columns='a1', right_columns='a2', fill_value='')

        # <Frame>
        # <Index> a1    b     a2    c        <<U2>
        # <Index>
        # 0       111   R     111   1111
        # 1       111   R     111   2222
        # 2       555   S     555   5555
        # 3       000   B
        # 4       333   C     333   7777
        # 5       333   C     333   3333
        # 6       333   Q     333   7777
        # 7       333   Q     333   3333
        # 8       666   X
        # 9                   444   4444
        # 10                  888   8888
        # <int64> <<U3> <<U1> <<U3> <object>

        self.assertEqual([dt.kind for dt in f3.dtypes.values], ['U', 'U', 'U', 'O'])
        self.assertEqual( f3.to_pairs(),
                (('a1', ((0, '111'), (1, '111'), (2, '555'), (3, '000'), (4, '333'), (5, '333'), (6, '333'), (7, '333'), (8, '666'), (9, ''), (10, ''))), ('b', ((0, 'R'), (1, 'R'), (2, 'S'), (3, 'B'), (4, 'C'), (5, 'C'), (6, 'Q'), (7, 'Q'), (8, 'X'), (9, ''), (10, ''))), ('a2', ((0, '111'), (1, '111'), (2, '555'), (3, ''), (4, '333'), (5, '333'), (6, '333'), (7, '333'), (8, ''), (9, '444'), (10, '888'))), ('c', ((0, 1111), (1, 2222), (2, 5555), (3, ''), (4, 7777), (5, 3333), (6, 7777), (7, 3333), (8, ''), (9, 4444), (10, 8888))))
                )

    def test_frame_join_o1(self) -> None:
        f1 = sf.Frame.from_dict_records([
                {"a1": "111", "a2": 1, "b": "R"},
                {"a1": "555", "a2": 5, "b": "S"},
                {"a1": "000", "a2": 0, "b": "B"}, # exclusive
                {"a1": "333", "a2": 3, "b": "C"},
                {"a1": "333", "a2": 3, "b": "Q"},
                {"a1": "666", "a2": 6, "b": "X"}, # exclusive
                ])

        f2 = sf.Frame.from_dict_records([
                {"a3": "444", "a4": 4, "c": 4444},
                {"a3": "333", "a4": 3, "c": 7777},
                {"a3": "555", "a4": 5, "c": 5555},
                {"a3": "111", "a4": 1, "c": 1111},
                {"a3": "111", "a4": 11, "c": 2222}, # exclusive
                {"a3": "333", "a4": 3, "c": 3333},
                {"a3": "888", "a4": 8, "c": 8888}, # exclusive
                ])

        f3 = f1.join_left(f2, left_columns=['a1', 'a2'], right_columns=['a3', 'a4'], fill_value='')
        self.assertEqual(f3.shape, (8, 6))
        self.assertEqual(f3.to_pairs(),
                (('a1', ((0, '111'), (1, '555'), (2, '000'), (3, '333'), (4, '333'), (5, '333'), (6, '333'), (7, '666'))), ('a2', ((0, 1), (1, 5), (2, 0), (3, 3), (4, 3), (5, 3), (6, 3), (7, 6))), ('b', ((0, 'R'), (1, 'S'), (2, 'B'), (3, 'C'), (4, 'C'), (5, 'Q'), (6, 'Q'), (7, 'X'))), ('a3', ((0, '111'), (1, '555'), (2, ''), (3, '333'), (4, '333'), (5, '333'), (6, '333'), (7, ''))), ('a4', ((0, 1), (1, 5), (2, ''), (3, 3), (4, 3), (5, 3), (6, 3), (7, ''))), ('c', ((0, 1111), (1, 5555), (2, ''), (3, 7777), (4, 3333), (5, 7777), (6, 3333), (7, ''))))
                )

    def test_frame_join_o2(self) -> None:
        f1 = sf.Frame.from_dict_records([
                {"a1": "111", "a2": 1, "b": "R"},
                {"a1": "555", "a2": 5, "b": "S"},
                {"a1": "000", "a2": 0, "b": "B"}, # exclusive
                {"a1": "333", "a2": 3, "b": "C"},
                {"a1": "333", "a2": 3, "b": "Q"},
                {"a1": "666", "a2": 6, "b": "X"}, # exclusive
                ])

        f2 = sf.Frame.from_dict_records([
                {"a3": "444", "a4": 4, "c": 4444},
                {"a3": "333", "a4": 3, "c": 7777},
                {"a3": "555", "a4": 5, "c": 5555},
                {"a3": "111", "a4": 1, "c": 1111},
                {"a3": "111", "a4": 11, "c": 2222}, # exclusive
                {"a3": "333", "a4": 3, "c": 3333},
                {"a3": "888", "a4": 8, "c": 8888}, # exclusive
                ])

        f3 = f1.join_outer(f2, left_columns=['a1', 'a2'], right_columns=['a3', 'a4'], fill_value='')
        self.assertEqual(f3.shape, (11, 6))
        self.assertEqual(f3.to_pairs(),
                (('a1', ((0, '111'), (1, '555'), (2, '000'), (3, '333'), (4, '333'), (5, '333'), (6, '333'), (7, '666'), (8, ''), (9, ''), (10, ''))), ('a2', ((0, 1), (1, 5), (2, 0), (3, 3), (4, 3), (5, 3), (6, 3), (7, 6), (8, ''), (9, ''), (10, ''))), ('b', ((0, 'R'), (1, 'S'), (2, 'B'), (3, 'C'), (4, 'C'), (5, 'Q'), (6, 'Q'), (7, 'X'), (8, ''), (9, ''), (10, ''))), ('a3', ((0, '111'), (1, '555'), (2, ''), (3, '333'), (4, '333'), (5, '333'), (6, '333'), (7, ''), (8, '444'), (9, '111'), (10, '888'))), ('a4', ((0, 1), (1, 5), (2, ''), (3, 3), (4, 3), (5, 3), (6, 3), (7, ''), (8, 4), (9, 11), (10, 8))), ('c', ((0, 1111), (1, 5555), (2, ''), (3, 7777), (4, 3333), (5, 7777), (6, 3333), (7, ''), (8, 4444), (9, 2222), (10, 8888))))
                )

    def test_frame_join_p(self):

        f1 = sf.Frame.from_dict_records([
                {"a1": "111", "b": "R"},
                {"a1": "555", "b": "S"},
                {"a1": "333", "b": "C"},
                ])

        f2 = sf.Frame.from_dict_records([
                {"a2": "111", "c": 1111},
                {"a2": "555", "c": 1111},
                {"a2": "333", "c": 3333},
                ])

        with self.assertRaises(RuntimeError):
            _ = f1.join_right(f2, right_columns='a2', fill_value='')

        with self.assertRaises(RuntimeError):
            _ = f1.join_right(f2, left_columns='a1', fill_value='')


    def test_frame_join_q1(self) -> None:
        f1 = sf.Frame.from_dict_records([
                {"a1": "111", "b": False},
                {"a1": "555", "b": True},
                {"a1": "000", "b": False}, # exclusive
                {"a1": "333", "b": True},
                {"a1": "333", "b": True},
                ])

        f2 = sf.Frame.from_dict_records([
                {"a2": "444", "c": 4444},
                {"a2": "333", "c": 7777},
                {"a2": "555", "c": 5555},
                {"a2": "111", "c": 1111},
                {"a2": "111", "c": 2222},
                {"a2": "333", "c": 3333},
                {"a2": "888", "c": 8888}, # exclusive
                ])

        f3 = f1.join_outer(f2,
                left_columns='a1',
                right_columns='a2',
                fill_value={'c': -1, 'a2': 'y', 'a1': 'x', 'b': False}
                )
        self.assertEqual(f3.to_pairs(),
                (('a1', ((0, '111'), (1, '111'), (2, '555'), (3, '000'), (4, '333'), (5, '333'), (6, '333'), (7, '333'), (8, 'x'), (9, 'x'))), ('b', ((0, False), (1, False), (2, True), (3, False), (4, True), (5, True), (6, True), (7, True), (8, False), (9, False))), ('a2', ((0, '111'), (1, '111'), (2, '555'), (3, 'y'), (4, '333'), (5, '333'), (6, '333'), (7, '333'), (8, '444'), (9, '888'))), ('c', ((0, 1111), (1, 2222), (2, 5555), (3, -1), (4, 7777), (5, 3333), (6, 7777), (7, 3333), (8, 4444), (9, 8888))))
                )


    #---------------------------------------------------------------------------
    def test_frame_merge_a(self) -> None:
        f1 = sf.Frame.from_dict_records([
                {"a1": "111", "a2": 1, "b": "R"},
                {"a1": "555", "a2": 5, "b": "S"},
                {"a1": "000", "a2": 0, "b": "B"}, # exclusive
                {"a1": "333", "a2": 3, "b": "C"},
                {"a1": "333", "a2": 3, "b": "Q"},
                {"a1": "666", "a2": 6, "b": "X"}, # exclusive
                ])

        f2 = sf.Frame.from_dict_records([
                {"a3": "444", "a4": 4, "c": 4444},
                {"a3": "333", "a4": 3, "c": 7777},
                {"a3": "555", "a4": 5, "c": 5555},
                {"a3": "111", "a4": 1, "c": 1111},
                {"a3": "111", "a4": 11, "c": 2222}, # exclusive
                {"a3": "333", "a4": 3, "c": 3333},
                {"a3": "888", "a4": 8, "c": 8888}, # exclusive
                ])

        f3 = f1.merge_left(f2, left_columns=['a1', 'a2'], right_columns=['a3', 'a4'], fill_value='')
        self.assertEqual(f3.columns.values.tolist(), ['a1', 'a2', 'b', 'c'])

        f4 = f1.merge_right(f2, left_columns=['a1', 'a2'], right_columns=['a3', 'a4'], fill_value='')
        self.assertEqual(f4.columns.values.tolist(), ['a3', 'a4', 'b', 'c'])

        f5 = f1.merge_right(f2, left_columns=['a1', 'a2'], right_columns=['a3', 'a4'], fill_value='', merge_labels=['x', 'y'])
        self.assertEqual(f5.columns.values.tolist(), ['x', 'y', 'b', 'c'])

        with self.assertRaises(RuntimeError):
            _ = f1.merge_right(f2, left_columns=['a1', 'a2'], right_columns=['a3', 'a4'], fill_value='', merge_labels=['y'])

        with self.assertRaises(RuntimeError):
            _ = f1.merge_right(f2, left_columns=['a1', 'a2'], right_columns=['a3', 'a4'], fill_value='', merge_labels='xy')


    def test_frame_merge_b(self) -> None:
        f1 = sf.FrameGO.from_dict_records([
                {"a1": "111", "a2": 1, "b": "R"},
                {"a1": "555", "a2": 5, "b": "S"},
                {"a1": "000", "a2": 0, "b": "B"}, # exclusive
                {"a1": "333", "a2": 3, "b": "C"},
                {"a1": "333", "a2": 3, "b": "Q"},
                {"a1": "666", "a2": 6, "b": "X"}, # exclusive
                ])

        f2 = sf.FrameGO.from_dict_records([
                {"a3": "444", "a4": 4, "c": 4444},
                {"a3": "333", "a4": 3, "c": 7777},
                {"a3": "555", "a4": 5, "c": 5555},
                {"a3": "111", "a4": 1, "c": 1111},
                {"a3": "111", "a4": 11, "c": 2222}, # exclusive
                {"a3": "333", "a4": 3, "c": 3333},
                {"a3": "888", "a4": 8, "c": 8888}, # exclusive
                ])

        f2 = f1.merge_outer(f2, left_columns='a1', right_columns='a3', fill_value=FillValueAuto, merge_labels='ax')
        self.assertEqual(f2.columns.values.tolist(), ['ax', 'a2', 'b', 'a4', 'c'])
        self.assertEqual(f2.__class__, FrameGO)
        self.assertEqual(f2.to_pairs(),
                (('ax', ((0, '111'), (1, '111'), (2, '555'), (3, '000'), (4, '333'), (5, '333'), (6, '333'), (7, '333'), (8, '666'), (9, '444'), (10, '888'))), ('a2', ((0, 1), (1, 1), (2, 5), (3, 0), (4, 3), (5, 3), (6, 3), (7, 3), (8, 6), (9, 0), (10, 0))), ('b', ((0, 'R'), (1, 'R'), (2, 'S'), (3, 'B'), (4, 'C'), (5, 'C'), (6, 'Q'), (7, 'Q'), (8, 'X'), (9, ''), (10, ''))), ('a4', ((0, 1), (1, 11), (2, 5), (3, 0), (4, 3), (5, 3), (6, 3), (7, 3), (8, 0), (9, 4), (10, 8))), ('c', ((0, 1111), (1, 2222), (2, 5555), (3, 0), (4, 7777), (5, 3333), (6, 7777), (7, 3333), (8, 0), (9, 4444), (10, 8888)))))

    def test_frame_merge_c(self) -> None:
        f1 = sf.FrameGO.from_dict_records([
                {"a1": "111", "a2": 1, "b": "R"},
                {"a1": "555", "a2": 5, "b": "S"},
                {"a1": "000", "a2": 0, "b": "B"}, # exclusive
                {"a1": "333", "a2": 3, "b": "C"},
                {"a1": "333", "a2": 3, "b": "Q"},
                {"a1": "666", "a2": 6, "b": "X"}, # exclusive
                ])

        f2 = sf.FrameGO.from_dict_records([
                {"a3": "444", "a4": 4, "c": 4444},
                {"a3": "333", "a4": 3, "c": 7777},
                {"a3": "555", "a4": 5, "c": 5555},
                {"a3": "111", "a4": 1, "c": 1111},
                {"a3": "333", "a4": 3, "c": 3333},
                {"a3": "888", "a4": 8, "c": 8888}, # exclusive
                ])

        f2 = f1.merge_inner(f2, left_columns='a1', right_columns='a3')
        self.assertEqual(f2.columns.values.tolist(), ['a1', 'a2', 'b', 'a4', 'c'])
        self.assertEqual(f2.to_pairs(),
                (('a1', ((0, '111'), (1, '555'), (2, '333'), (3, '333'), (4, '333'), (5, '333'))), ('a2', ((0, 1), (1, 5), (2, 3), (3, 3), (4, 3), (5, 3))), ('b', ((0, 'R'), (1, 'S'), (2, 'C'), (3, 'C'), (4, 'Q'), (5, 'Q'))), ('a4', ((0, 1), (1, 5), (2, 3), (3, 3), (4, 3), (5, 3))), ('c', ((0, 1111), (1, 5555), (2, 7777), (3, 3333), (4, 7777), (5, 3333))))
                )
