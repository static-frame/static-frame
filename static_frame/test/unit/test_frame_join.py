from __future__ import annotations

import numpy as np

import static_frame as sf
from static_frame import Frame
from static_frame import IndexDate
from static_frame import IndexHierarchy
from static_frame.core.exception import InvalidFillValue
from static_frame.core.fill_value_auto import FillValueAuto
from static_frame.core.join import join
from static_frame.test.test_case import TestCase
from static_frame.test.test_case import skip_win


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

        # NOTE: this indexes ordering after union is not stable, so do an explict selection before testing
        locs4 = [0, 1, 2, 'foo', 'x', 'y']
        f4 = f4.reindex(locs4)

        self.assertEqual(f4.to_pairs(0),
                (('a', ((0, 10.0), (1, 10.0), (2, None), ('foo', 20.0), ('x', 20.0), ('y', None))), ('b', ((0, 'x'), (1, 'x'), (2, 'y'), ('foo', 'y'), ('x', 'z'), ('y', None))), ('c', ((0, None), (1, None), (2, None), ('foo', None), ('x', 'foo'), ('y', 'bar'))), ('d', ((0, None), (1, None), (2, None), ('foo', None), ('x', 10.0), ('y', 20.0))))
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

    def test_frame_join_b(self) -> None:

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
                include_index=True,
                )
        self.assertEqual(f3.shape, (7, 4))
        self.assertEqual(f3.fillna(None).to_pairs(0),
                (('Employee.LastName', ((('a', 10), 'Raf'), (('b', 11), 'Jon'), (('c', 11), 'Hei'), (('d', 12), 'Rob'), (('e', 12), 'Smi'), (('f', None), 'Wil'), ((None, 13), None))), ('Employee.DepartmentID', ((('a', 10), 31), (('b', 11), 33), (('c', 11), 33), (('d', 12), 34), (('e', 12), 34), (('f', None), None), ((None, 13), None))), ('Department.DepartmentID', ((('a', 10), 31.0), (('b', 11), 33.0), (('c', 11), 33.0), (('d', 12), 34.0), (('e', 12), 34.0), (('f', None), None), ((None, 13), 35.0))), ('Department.DepartmentName', ((('a', 10), 'Sales'), (('b', 11), 'Engineering'), (('c', 11), 'Engineering'), (('d', 12), 'Clerical'), (('e', 12), 'Clerical'), (('f', None), None), ((None, 13), 'Marketing'))))

                )

        f4 = f1.join_inner(f2,
                left_columns='DepartmentID',
                left_template='Employee.{}',
                right_columns='DepartmentID',
                right_template='Department.{}',
                include_index=True,
                )
        self.assertEqual(f4.shape, (5, 4))

        self.assertEqual(f4.fillna(None).to_pairs(0),
                (('Employee.LastName', ((('a', 10), 'Raf'), (('b', 11), 'Jon'), (('c', 11), 'Hei'), (('d', 12), 'Rob'), (('e', 12), 'Smi'))), ('Employee.DepartmentID', ((('a', 10), 31), (('b', 11), 33), (('c', 11), 33), (('d', 12), 34), (('e', 12), 34))), ('Department.DepartmentID', ((('a', 10), 31), (('b', 11), 33), (('c', 11), 33), (('d', 12), 34), (('e', 12), 34))), ('Department.DepartmentName', ((('a', 10), 'Sales'), (('b', 11), 'Engineering'), (('c', 11), 'Engineering'), (('d', 12), 'Clerical'), (('e', 12), 'Clerical'))))

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

    @skip_win
    def test_frame_join_d(self) -> None:
        index1 = IndexDate.from_date_range('2020-05-04', '2020-05-08')
        index2 = IndexHierarchy.from_product(('A', 'B'), index1)

        f1 = Frame.from_dict(dict(a=tuple(range(10)), b=tuple('pqrstuvwxy')), index=index2)
        f2 = Frame.from_dict(dict(c=tuple(range(10, 15)), d=tuple('fffgg')), index=index1)

        f3 = f1.join_left(f2, left_depth_level=1, right_depth_level=0, include_index=True)

        self.assertEqual(f3.dtypes.values.tolist(),
                [np.dtype('int64'), np.dtype('<U1'), np.dtype('int64'), np.dtype('<U1')]
                )
        self.assertEqual(f3.shape, (10, 4))

        self.assertEqual(
                f3.to_pairs(0),
                (('a', (((('A', np.datetime64('2020-05-04')), np.datetime64('2020-05-04')), 0), ((('A', np.datetime64('2020-05-05')), np.datetime64('2020-05-05')), 1), ((('A', np.datetime64('2020-05-06')), np.datetime64('2020-05-06')), 2), ((('A', np.datetime64('2020-05-07')), np.datetime64('2020-05-07')), 3), ((('A', np.datetime64('2020-05-08')), np.datetime64('2020-05-08')), 4), ((('B', np.datetime64('2020-05-04')), np.datetime64('2020-05-04')), 5), ((('B', np.datetime64('2020-05-05')), np.datetime64('2020-05-05')), 6), ((('B', np.datetime64('2020-05-06')), np.datetime64('2020-05-06')), 7), ((('B', np.datetime64('2020-05-07')), np.datetime64('2020-05-07')), 8), ((('B', np.datetime64('2020-05-08')), np.datetime64('2020-05-08')), 9))), ('b', (((('A', np.datetime64('2020-05-04')), np.datetime64('2020-05-04')), 'p'), ((('A', np.datetime64('2020-05-05')), np.datetime64('2020-05-05')), 'q'), ((('A', np.datetime64('2020-05-06')), np.datetime64('2020-05-06')), 'r'), ((('A', np.datetime64('2020-05-07')), np.datetime64('2020-05-07')), 's'), ((('A', np.datetime64('2020-05-08')), np.datetime64('2020-05-08')), 't'), ((('B', np.datetime64('2020-05-04')), np.datetime64('2020-05-04')), 'u'), ((('B', np.datetime64('2020-05-05')), np.datetime64('2020-05-05')), 'v'), ((('B', np.datetime64('2020-05-06')), np.datetime64('2020-05-06')), 'w'), ((('B', np.datetime64('2020-05-07')), np.datetime64('2020-05-07')), 'x'), ((('B', np.datetime64('2020-05-08')), np.datetime64('2020-05-08')), 'y'))), ('c', (((('A', np.datetime64('2020-05-04')), np.datetime64('2020-05-04')), 10), ((('A', np.datetime64('2020-05-05')), np.datetime64('2020-05-05')), 11), ((('A', np.datetime64('2020-05-06')), np.datetime64('2020-05-06')), 12), ((('A', np.datetime64('2020-05-07')), np.datetime64('2020-05-07')), 13), ((('A', np.datetime64('2020-05-08')), np.datetime64('2020-05-08')), 14), ((('B', np.datetime64('2020-05-04')), np.datetime64('2020-05-04')), 10), ((('B', np.datetime64('2020-05-05')), np.datetime64('2020-05-05')), 11), ((('B', np.datetime64('2020-05-06')), np.datetime64('2020-05-06')), 12), ((('B', np.datetime64('2020-05-07')), np.datetime64('2020-05-07')), 13), ((('B', np.datetime64('2020-05-08')), np.datetime64('2020-05-08')), 14))), ('d', (((('A', np.datetime64('2020-05-04')), np.datetime64('2020-05-04')), 'f'), ((('A', np.datetime64('2020-05-05')), np.datetime64('2020-05-05')), 'f'), ((('A', np.datetime64('2020-05-06')), np.datetime64('2020-05-06')), 'f'), ((('A', np.datetime64('2020-05-07')), np.datetime64('2020-05-07')), 'g'), ((('A', np.datetime64('2020-05-08')), np.datetime64('2020-05-08')), 'g'), ((('B', np.datetime64('2020-05-04')), np.datetime64('2020-05-04')), 'f'), ((('B', np.datetime64('2020-05-05')), np.datetime64('2020-05-05')), 'f'), ((('B', np.datetime64('2020-05-06')), np.datetime64('2020-05-06')), 'f'), ((('B', np.datetime64('2020-05-07')), np.datetime64('2020-05-07')), 'g'), ((('B', np.datetime64('2020-05-08')), np.datetime64('2020-05-08')), 'g'))))
                )

        # inner join is equivalent to left, right, outer
        self.assertTrue(f1.join_inner(f2, left_depth_level=1, right_depth_level=0, include_index=True).equals(f3))
        self.assertTrue(f1.join_right(f2, left_depth_level=1, right_depth_level=0, include_index=True).equals(f3))
        self.assertTrue(f1.join_outer(f2, left_depth_level=1, right_depth_level=0, include_index=True).equals(f3))

    def test_frame_join_e(self) -> None:

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

        f4 = f1.join_left(f2,
                left_depth_level=[0, 1],
                right_depth_level=[0, 1],
                fill_value=None,
                include_index=True,
                )

        self.assertEqual(f4.to_pairs(0),
                (('a', ((('A', 1), 0), (('A', 2), 1), (('A', 3), 2), (('A', 4), 3), (('A', 5), 4), (('B', 1), 5), (('B', 2), 6), (('B', 3), 7), (('B', 4), 8), (('B', 5), 9))), ('b', ((('A', 1), 'p'), (('A', 2), 'q'), (('A', 3), 'r'), (('A', 4), 's'), (('A', 5), 't'), (('B', 1), 'u'), (('B', 2), 'v'), (('B', 3), 'w'), (('B', 4), 'x'), (('B', 5), 'y'))), ('c', ((('A', 1), None), (('A', 2), 12), (('A', 3), None), (('A', 4), None), (('A', 5), None), (('B', 1), None), (('B', 2), None), (('B', 3), 10), (('B', 4), None), (('B', 5), 11))), ('d', ((('A', 1), None), (('A', 2), 'h'), (('A', 3), None), (('A', 4), None), (('A', 5), None), (('B', 1), None), (('B', 2), None), (('B', 3), 'f'), (('B', 4), None), (('B', 5), 'g'))))
                )

    def test_frame_join_f(self) -> None:
        # column on column

        f1 = Frame.from_dict(
                dict(a=(10,10,np.nan,20,20), b=('x','x','y','y','z')),
                index=tuple('abcde'))

        f2 = Frame.from_dict(
                dict(c=('y', 'y', 'w'), d=(1000, 3000, 2000)),
                index=('q', 'p', 'r'))

        # case of when a non-index value is joined on, where the right as repeated values; Pandas df1.merge(df2, how='left', left_on='b', right_on='c') will add rows for all unique combinations and drop the resulting index.

        f3 = f1.join_left(f2, left_columns='b', right_columns='c', include_index=True)
        self.assertEqual(f3.fillna(None).to_pairs(0),
                (('a', ((('c', 'q'), None), (('c', 'p'), None), (('d', 'q'), 20.0), (('d', 'p'), 20.0), (('a', None), 10.0), (('b', None), 10.0), (('e', None), 20.0))), ('b', ((('c', 'q'), 'y'), (('c', 'p'), 'y'), (('d', 'q'), 'y'), (('d', 'p'), 'y'), (('a', None), 'x'), (('b', None), 'x'), (('e', None), 'z'))), ('c', ((('c', 'q'), 'y'), (('c', 'p'), 'y'), (('d', 'q'), 'y'), (('d', 'p'), 'y'), (('a', None), None), (('b', None), None), (('e', None), None))), ('d', ((('c', 'q'), 1000.0), (('c', 'p'), 3000.0), (('d', 'q'), 1000.0), (('d', 'p'), 3000.0), (('a', None), None), (('b', None), None), (('e', None), None))))
                )

        f4 = f1.join_right(f2, left_columns='b', right_columns='c', fill_value=None, include_index=True)
        self.assertEqual(f4.fillna(None).to_pairs(0),
                (('a', ((('c', 'q'), None), (('c', 'p'), None), (('d', 'q'), 20.0), (('d', 'p'), 20.0), ((None, 'r'), None))), ('b', ((('c', 'q'), 'y'), (('c', 'p'), 'y'), (('d', 'q'), 'y'), (('d', 'p'), 'y'), ((None, 'r'), None))), ('c', ((('c', 'q'), 'y'), (('c', 'p'), 'y'), (('d', 'q'), 'y'), (('d', 'p'), 'y'), ((None, 'r'), 'w'))), ('d', ((('c', 'q'), 1000), (('c', 'p'), 3000), (('d', 'q'), 1000), (('d', 'p'), 3000), ((None, 'r'), 2000))))
                )

        f5 = f1.join_inner(f2, left_columns='b', right_columns='c', include_index=True)
        self.assertEqual(f5.fillna(None).to_pairs(0),
                (('a', ((('c', 'q'), None), (('c', 'p'), None), (('d', 'q'), 20.0), (('d', 'p'), 20.0))), ('b', ((('c', 'q'), 'y'), (('c', 'p'), 'y'), (('d', 'q'), 'y'), (('d', 'p'), 'y'))), ('c', ((('c', 'q'), 'y'), (('c', 'p'), 'y'), (('d', 'q'), 'y'), (('d', 'p'), 'y'))), ('d', ((('c', 'q'), 1000), (('c', 'p'), 3000), (('d', 'q'), 1000), (('d', 'p'), 3000))))
                )

        f6 = f1.join_outer(f2, left_columns='b', right_columns='c', fill_value=None, include_index=True)
        self.assertEqual(f6.fillna(None).to_pairs(0),
                (('a', ((('c', 'q'), None), (('c', 'p'), None), (('d', 'q'), 20.0), (('d', 'p'), 20.0), (('a', None), 10.0), (('b', None), 10.0), (('e', None), 20.0), ((None, 'r'), None))), ('b', ((('c', 'q'), 'y'), (('c', 'p'), 'y'), (('d', 'q'), 'y'), (('d', 'p'), 'y'), (('a', None), 'x'), (('b', None), 'x'), (('e', None), 'z'), ((None, 'r'), None))), ('c', ((('c', 'q'), 'y'), (('c', 'p'), 'y'), (('d', 'q'), 'y'), (('d', 'p'), 'y'), (('a', None), None), (('b', None), None), (('e', None), None), ((None, 'r'), 'w'))), ('d', ((('c', 'q'), 1000), (('c', 'p'), 3000), (('d', 'q'), 1000), (('d', 'p'), 3000), (('a', None), None), (('b', None), None), (('e', None), None), ((None, 'r'), 2000))))
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

        f4 = f2.join_right(f1,
                left_depth_level=0,
                right_depth_level=0,
                fill_value=None,
                include_index=True,
                )
        self.assertEqual(f4.to_pairs(0),
                (('c', ((0, None), (1, None), (2, None), (3, None), (4, None))), ('d', ((0, None), (1, None), (2, None), (3, None), (4, None))), ('a', ((0, 10), (1, 10), (2, 20), (3, 20), (4, 20))), ('b', ((0, 'x'), (1, 'x'), (2, 'y'), (3, 'y'), (4, 'z'))))
                )

        f5 = f2.join_left(f1,
                left_depth_level=0,
                right_depth_level=0,
                fill_value=None,
                include_index=True,
                )
        self.assertEqual(f5.to_pairs(0),
                (('c', (('x', 'foo'), ('y', 'bar'))), ('d', (('x', 10), ('y', 20))), ('a', (('x', None), ('y', None))), ('b', (('x', None), ('y', None))))
                )

        f6 = f2.join_outer(f1,
                left_depth_level=0,
                right_depth_level=0,
                fill_value=None,
                include_index=True,
                )
        f6 = f6.loc[[0, 1, 2, 3, 4, 'y', 'x']] # get stable ordering
        self.assertEqual(f6.to_pairs(0),
                (('c', ((0, None), (1, None), (2, None), (3, None), (4, None), ('y', 'bar'), ('x', 'foo'))), ('d', ((0, None), (1, None), (2, None), (3, None), (4, None), ('y', 20), ('x', 10))), ('a', ((0, 10), (1, 10), (2, 20), (3, 20), (4, 20), ('y', None), ('x', None))), ('b', ((0, 'x'), (1, 'x'), (2, 'y'), (3, 'y'), (4, 'z'), ('y', None), ('x', None))))
                )

    def test_frame_join_h2(self) -> None:

        f1 = sf.Frame.from_dict(dict(a=(10,10,20,20,20), b=('x','x','y','y','z')))
        f2 = sf.Frame.from_dict(dict(c=('foo', 'bar'), d=(10, 20)), index=('x', 'y'))
        with self.assertRaises(InvalidFillValue):
            _ = f2.join_inner(f1, left_depth_level=0, right_depth_level=0, fill_value=FillValueAuto)


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



    # def test_frame_join_sort_a(self) -> None:
    #     from static_frame.core.join import join_sort

    #     sff_left = ff.parse('s(20,4)|v(int)|i(I,str)|c(I,str)').assign[sf.ILoc[0]].apply(lambda s: s % 3)

    #     sff_right = ff.parse('s(8,3)|v(int,bool,bool)|i(I,str)').assign[sf.ILoc[0]].apply(lambda s: s % 3)

    #     post = join_sort(sff_left, sff_right, left_columns='zZbu', right_columns=0)
    #     ref = sff_left.join_left(sff_right, left_columns='zZbu', right_columns=0)

    #     # import ipdb; ipdb.set_trace()


