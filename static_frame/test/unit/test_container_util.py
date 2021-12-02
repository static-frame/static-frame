import unittest
import datetime

import numpy as np

from static_frame.core.container_util import bloc_key_normalize
from static_frame.core.container_util import get_col_dtype_factory
from static_frame.core.container_util import index_from_optional_constructor
from static_frame.core.container_util import index_many_concat
from static_frame.core.container_util import index_many_set
from static_frame.core.container_util import is_static
from static_frame.core.container_util import key_to_ascending_key
from static_frame.core.container_util import matmul
from static_frame.core.container_util import pandas_to_numpy
from static_frame.core.container_util import pandas_version_under_1
from static_frame.core.container_util import apex_to_name
from static_frame.core.container_util import apply_binary_operator_blocks_columnar
from static_frame.core.container_util import container_to_exporter_attr
from static_frame.core.container_util import get_block_match


from static_frame.core.frame import FrameHE

from static_frame.test.test_case import TestCase
from static_frame.core.exception import AxisInvalid


from static_frame import Frame
from static_frame import Index
from static_frame import IndexDate
from static_frame import IndexDateGO
from static_frame import IndexGO
from static_frame import IndexHierarchy
from static_frame import IndexHierarchyGO
from static_frame import IndexSecond
from static_frame import Series

class TestUnit(TestCase):

    def test_is_static_a(self) -> None:
        self.assertTrue(is_static(Index))
        self.assertFalse(is_static(IndexGO))

        self.assertTrue(is_static(IndexHierarchy))
        self.assertFalse(is_static(IndexHierarchyGO))

    def test_is_static_b(self) -> None:

        self.assertTrue(is_static(Index.from_labels))
        self.assertTrue(is_static(IndexHierarchy.from_labels))
        self.assertTrue(is_static(IndexHierarchy.from_product))
        self.assertTrue(is_static(IndexHierarchy.from_labels_delimited))
        self.assertTrue(is_static(IndexHierarchy.from_tree))
        self.assertTrue(is_static(IndexHierarchy.from_index_items))

        self.assertFalse(is_static(IndexGO.from_labels))
        self.assertFalse(is_static(IndexHierarchyGO.from_labels))
        self.assertFalse(is_static(IndexHierarchyGO.from_product))
        self.assertFalse(is_static(IndexHierarchyGO.from_labels_delimited))
        self.assertFalse(is_static(IndexHierarchyGO.from_tree))
        self.assertFalse(is_static(IndexHierarchyGO.from_index_items))


    def test_index_from_optional_constructor_a(self) -> None:
        idx1 = index_from_optional_constructor([1, 3, 4],
                default_constructor=Index)
        self.assertEqual(idx1.__class__, Index)

        # given a mutable index and an immutable default, get immutable version
        idx2 = index_from_optional_constructor(IndexGO((1, 3, 4)),
                default_constructor=Index)
        self.assertEqual(idx2.__class__, Index)

        # given a mutable index and an immutable default, get immutable version
        idx3 = index_from_optional_constructor(IndexGO((1, 3, 4)),
                default_constructor=IndexGO)
        self.assertEqual(idx3.__class__, IndexGO)

        # given a mutable index and an immutable default, get immutable version
        idx4 = index_from_optional_constructor(
                IndexSecond((1, 3, 4)),
                default_constructor=Index)
        self.assertEqual(idx4.__class__, IndexSecond)


    def test_index_from_optional_constructor_b(self) -> None:
        idx0 = IndexHierarchy.from_labels(
                [('a', 0), ('a', 1), ('b', 0), ('b', 1)])
        idx1 = index_from_optional_constructor(
                idx0,
                default_constructor=IndexHierarchy.from_labels)

        # Since the default constructo is static, we should be able to reuse the index
        self.assertEqual(id(idx0), id(idx1))


    def test_index_from_optional_constructor_c(self) -> None:
        idx0 = IndexHierarchyGO.from_labels(
                [('a', 0), ('a', 1), ('b', 0), ('b', 1)])
        idx1 = index_from_optional_constructor(
                idx0,
                default_constructor=IndexHierarchy.from_labels)

        # Since the default constructo is static, we should be able to reuse the index
        self.assertNotEqual(id(idx0), id(idx1))
        self.assertTrue(idx1.STATIC)


    def test_index_from_optional_constructor_d(self) -> None:
        idx0 = IndexHierarchy.from_labels(
                [('a', 0), ('a', 1), ('b', 0), ('b', 1)])
        idx1 = index_from_optional_constructor(
                idx0,
                default_constructor=IndexHierarchyGO.from_labels)

        # Since the default constructo is static, we should be able to reuse the index
        self.assertNotEqual(id(idx0), id(idx1))
        self.assertFalse(idx1.STATIC)



    def test_matmul_a(self) -> None:
        # lhs: frame, rhs: array

        f1 = Frame.from_items((('a', (1, 2, 3)), ('b', (3, 4, 5))),
                index=('x', 'y', 'z'))

        self.assertEqual(
                matmul(f1, [4, 3]).to_pairs(),
                (('x', 13), ('y', 20), ('z', 27))
                )

        self.assertEqual(
                matmul(f1, np.array([4, 3])).to_pairs(),
                (('x', 13), ('y', 20), ('z', 27))
                )


        self.assertEqual(
                matmul(f1, [3, 4]).to_pairs(),
                (('x', 15), ('y', 22), ('z', 29))
                )

        self.assertEqual(
                matmul(f1, np.array([3, 4])).to_pairs(),
                (('x', 15), ('y', 22), ('z', 29))
                )


    def test_matmul_b(self) -> None:
        # lhs: frame, rhs: array

        f1 = Frame.from_items((('a', (1, 2, 3)), ('b', (3, 4, 5))),
                index=('x', 'y', 'z'))

        # get an auto incremented integer columns
        self.assertEqual(
            matmul(f1, np.arange(10).reshape(2, 5)).to_pairs(0),
            ((0, (('x', 15), ('y', 20), ('z', 25))), (1, (('x', 19), ('y', 26), ('z', 33))), (2, (('x', 23), ('y', 32), ('z', 41))), (3, (('x', 27), ('y', 38), ('z', 49))), (4, (('x', 31), ('y', 44), ('z', 57))))
            )

    def test_matmul_c(self) -> None:
        # lhs: frame, rhs: Series, 1D array

        f1 = Frame.from_items((('a', (1, 2, 3)), ('b', (3, 4, 5))),
                index=('x', 'y', 'z'))
        s1 = Series((10, 11), index=('a', 'b'))

        self.assertEqual(matmul(f1, s1).to_pairs(),
                (('x', 43), ('y', 64), ('z', 85)))

        self.assertEqual(matmul(f1, s1.values).to_pairs(),
                (('x', 43), ('y', 64), ('z', 85)))

        with self.assertRaises(RuntimeError):
            matmul(f1, np.arange(20).reshape(5, 4))


    def test_matmul_d(self) -> None:
        # lhs: series, rhs: frame

        f1 = Frame.from_items((('a', (1, 2, 3)), ('b', (3, 4, 5))),
                index=('x', 'y', 'z'))

        s1 = Series((3, 4, 2), index=('x', 'y', 'z'))

        self.assertEqual(
            matmul(s1, f1).to_pairs(),
            (('a', 17), ('b', 35))
            )

        # produces a Series indexed 0, 1
        self.assertEqual(matmul(s1, f1.values).to_pairs(),
            ((0, 17), (1, 35)))

    def test_matmul_e(self) -> None:
        # lhs: series, rhs: series

        s1 = Series((3, 4, 2), index=('x', 'y', 'z'))

        s2 = Series((10, 11, 12), index=('x', 'y', 'z'))

        self.assertEqual(matmul(s1, s2), 98)
        self.assertEqual(matmul(s1, s2.values), 98)


    def test_matmul_f(self) -> None:
        # lhs: array 1D, rhs: array 2D, Frame

        f1 = Frame.from_items((('a', (1, 2, 3)), ('b', (3, 4, 5))),
                index=('x', 'y', 'z'))

        self.assertEqual(matmul([3, 4, 5], f1.values).tolist(),
                [26, 50])

        self.assertEqual(matmul([3, 4, 5], f1).to_pairs(),
                (('a', 26), ('b', 50))
                )


    def test_matmul_g(self) -> None:
        # lhs: array 1D, rhs: array 1D, Series

        s1 = Series((3, 4, 2), index=('x', 'y', 'z'))
        self.assertEqual(matmul([10, 11, 12], s1.values), 98)
        self.assertEqual(matmul([10, 11, 12], s1), 98)

        with self.assertRaises(RuntimeError):
            self.assertEqual(matmul(s1, [10, 11]), 98)


    def test_matmul_h(self) -> None:
        # lhs: array 2D, rhs: array 2D, Frame

        f1 = Frame.from_dict(dict(a=(1, 2, 3, 4), b=(5, 6, 7, 8)), index=tuple('wxyz'))
        f2 = Frame.from_dict(dict(p=(1, 2), q=(3, 4), r=(5, 6)), index=tuple('ab'))


        self.assertEqual(matmul(f1.values, f2).to_pairs(0),
                (('p', ((0, 11), (1, 14), (2, 17), (3, 20))), ('q', ((0, 23), (1, 30), (2, 37), (3, 44))), ('r', ((0, 35), (1, 46), (2, 57), (3, 68))))
                )

        self.assertEqual(matmul(f1, f2.values).to_pairs(0),
                ((0, (('w', 11), ('x', 14), ('y', 17), ('z', 20))), (1, (('w', 23), ('x', 30), ('y', 37), ('z', 44))), (2, (('w', 35), ('x', 46), ('y', 57), ('z', 68))))
                )

        with self.assertRaises(RuntimeError):
            matmul(f1, np.arange(25).reshape(5, 5))


    def test_matmul_i(self) -> None:
        import itertools as it

        f1 = Frame.from_dict(dict(a=(1, 2), b=(5, 6)), index=tuple('yz'))

        f_container = lambda x: x
        f_values = lambda x: x.values

        for pair in ((f1, f1.T), (f1, f1.loc['y']), (f1['a'], f1), (f1.loc['y'], f1.loc['z'])):
            for x, y in it.combinations((f_container, f_values, f_container, f_values), 2):
                post = matmul(x(pair[0]), y(pair[1])) # type: ignore
                if isinstance(post, (Series, Frame)):
                    self.assertTrue(post.values.tolist(), (pair[0].values @ pair[1].values).tolist())
                elif isinstance(post, np.ndarray):
                    self.assertTrue(post.tolist(), (pair[0].values @ pair[1].values).tolist())


    def test_matmul_j(self) -> None:

        f1 = Frame.from_dict(dict(a=(1, 2, 3), b=(5, 6, 7)),
                index=tuple('xyz'),
                name='foo')
        a1 = np.array([[5], [0]])

        with self.assertRaises(RuntimeError):
            _ = matmul(a1, f1)


    #---------------------------------------------------------------------------
    def test_key_to_ascending_key_a(self) -> None:
        self.assertEqual(key_to_ascending_key([9, 5, 1], 3), [1, 5, 9])
        self.assertEqual(key_to_ascending_key(np.array([9, 5, 1]), 3).tolist(), [1, 5, 9]) # type: ignore

        self.assertEqual(key_to_ascending_key(slice(3, 0, -1), 3), slice(1, 4, 1))
        self.assertEqual(key_to_ascending_key(100, 3), 100)

        self.assertEqual(key_to_ascending_key([], 3), [])

        self.assertEqual(key_to_ascending_key( # type: ignore
                Series(('a', 'b', 'c'), index=(9, 5, 1)), 3).values.tolist(),
                ['c', 'b', 'a'])

        f1 = Frame.from_dict(dict(b=(1, 2), a=(5, 6)), index=tuple('yz'))
        f2 = key_to_ascending_key(f1, f1.shape[1])
        self.assertEqual(f2.columns.values.tolist(), ['a', 'b']) # type: ignore


    def test_key_to_ascending_key_b(self) -> None:

        with self.assertRaises(RuntimeError):
            key_to_ascending_key(dict(a=3), size=3)

    def test_pandas_to_numpy_a(self) -> None:
        import pandas as pd
        pdvu1 = pandas_version_under_1()

        if not pdvu1:

            s1 = pd.Series([3, 4, np.nan]).convert_dtypes()

            a1 = pandas_to_numpy(s1, own_data=False)
            self.assertEqual(a1.dtype, np.dtype('O'))
            self.assertAlmostEqualValues(a1.tolist(), [3, 4, np.nan])

            a2 = pandas_to_numpy(s1[:2], own_data=False)
            self.assertEqual(a2.dtype, np.dtype('int64'))

            s2 = pd.Series([False, True, np.nan]).convert_dtypes()

            a3 = pandas_to_numpy(s2, own_data=False)
            self.assertEqual(a3.dtype, np.dtype('O'))
            self.assertAlmostEqualValues(a3.tolist(), [False, True, np.nan])

            a4 = pandas_to_numpy(s2[:2], own_data=False)
            self.assertEqual(a4.dtype, np.dtype('bool'))



    def test_bloc_key_normalize_a(self) -> None:
        f1 = Frame.from_dict(dict(b=(1, 2), a=(5, 6)), index=tuple('yz'))

        with self.assertRaises(RuntimeError):
            bloc_key_normalize(np.arange(4).reshape(2, 2), f1)

        post1 = bloc_key_normalize(f1['a':] >= 5, f1) #type: ignore
        self.assertEqual(post1.tolist(), [[False, True], [False, True]])

        post2 = bloc_key_normalize(f1 < 5, f1)
        self.assertEqual(post2.tolist(), [[True, False], [True, False]])


    def test_index_many_concat_a(self) -> None:

        idx0 = Index(('1997-01-01', '1997-01-02'), name='foo')
        idx1 = IndexDate(('2020-01-01', '2020-01-02'), name='foo')
        idx2 = IndexDate(('2020-02-01', '2020-02-02'))


        post1 = index_many_concat((idx0,  idx1), Index)
        assert isinstance(post1, Index)

        self.assertEqual(post1.values.tolist(),
                ['1997-01-01',
                '1997-01-02',
                datetime.date(2020, 1, 1),
                datetime.date(2020, 1, 2)])
        self.assertEqual(post1.name, 'foo')
        self.assertEqual(post1.__class__, Index)

        post2 = index_many_concat((idx1,  idx2), Index)
        assert isinstance(post2, Index)

        self.assertEqual(post2.__class__, IndexDate)
        self.assertEqual(post2.values.tolist(),
                [datetime.date(2020, 1, 1),
                datetime.date(2020, 1, 2),
                datetime.date(2020, 2, 1),
                datetime.date(2020, 2, 2)])

    def test_index_many_concat_b(self) -> None:

        idx0 = Index(('1997-01-01', '1997-01-02'), name='foo')
        idx1 = IndexDate(('2020-01-01', '2020-01-02'), name='foo')
        idx2 = IndexDate(('2020-02-01', '2020-02-02'))

        post1 = index_many_concat((idx0,  idx1), IndexGO)
        self.assertEqual(post1.__class__, IndexGO)

        post2 = index_many_concat((idx1,  idx2), IndexGO)
        self.assertEqual(post2.__class__, IndexDateGO)

    def test_index_many_concat_c(self) -> None:
        from datetime import date
        i1 = IndexHierarchy.from_labels([[1, date(2019, 1, 1)], [2, date(2019, 1, 2)]], index_constructors=[Index, IndexDate])

        i2 = IndexHierarchy.from_labels([[2, date(2019, 1, 3)], [3, date(2019, 1, 4)]], index_constructors=[Index, IndexDate])

        i3 = IndexHierarchy.from_labels([[4, date(2019, 1, 5)], [5, date(2019, 1, 6)]], index_constructors=[Index, IndexDate])

        i4 = IndexHierarchy.from_labels([[4, date(2019, 1, 5)], [5, date(2019, 1, 6)]])


        i5 = index_many_concat((i1, i2, i3), cls_default=Index)
        assert isinstance(i5, IndexHierarchy)

        self.assertEqual(i5.index_types.to_pairs(),
                ((0, Index), (1, IndexDate))
                )
        self.assertEqual(i5.values.tolist(),
                [[1, date(2019, 1, 1)], [2, date(2019, 1, 2)], [2, date(2019, 1, 3)], [3, date(2019, 1, 4)], [4, date(2019, 1, 5)], [5, date(2019, 1, 6)]])

        # with unaligned index types we fall back in Index
        i6 = index_many_concat((i1, i2, i4), cls_default=Index)
        assert isinstance(i6, IndexHierarchy)

        self.assertEqual(i6.index_types.to_pairs(),
                ((0, Index), (1, Index))
                )

    def test_index_many_concat_d(self) -> None:
        from datetime import date
        i1 = IndexHierarchy.from_labels([[1, date(2019, 1, 1)], [2, date(2019, 1, 2)]], index_constructors=[Index, IndexDate])

        i2 = IndexHierarchy.from_labels([[2, date(2019, 1, 3)], [3, date(2019, 1, 4)]], index_constructors=[Index, IndexDate])

        post1 = index_many_concat((i1, i2), cls_default=IndexGO)
        self.assertEqual(post1.__class__, IndexHierarchyGO)
        assert isinstance(post1, IndexHierarchy)
        self.assertEqual(post1.values.tolist(),
                [[1, date(2019, 1, 1)], [2, date(2019, 1, 2)], [2, date(2019, 1, 3)], [3, date(2019, 1, 4)]]
                )


    def test_index_many_concat_e(self) -> None:

        idx1 = IndexDateGO(('2020-01-01', '2020-01-02'))
        idx2 = IndexDateGO(('2020-02-01', '2020-02-02'))

        post1 = index_many_concat((idx1, idx2), cls_default=Index)

        self.assertEqual(post1.__class__, IndexDate)
        self.assertEqual(post1.values.tolist(), #type: ignore
                [datetime.date(2020, 1, 1), datetime.date(2020, 1, 2), datetime.date(2020, 2, 1), datetime.date(2020, 2, 2)]
                )

    #---------------------------------------------------------------------------
    def test_index_many_set_a(self) -> None:

        idx0 = Index(('1997-01-01', '1997-01-02'), name='foo')
        idx1 = IndexDate(('2020-01-01', '2020-01-02'), name='foo')
        idx2 = IndexDate(('2020-01-02', '2020-01-03'))


        post1 = index_many_set((idx0,  idx1), Index, union=True)
        assert isinstance(post1, Index)

        self.assertEqual(post1.name, 'foo')
        self.assertEqual(post1.__class__, Index)

        # self.assertEqual(set(post1.values),
        #         {'1997-01-02',
        #         '1997-01-01',
        #         np.datetime64('2020-01-01'),
        #         np.datetime64('2020-01-02')})

        # the result of this operation is an unstable ordering
        values = set(post1.values)
        self.assertTrue('1997-01-01' in values)
        self.assertTrue('1997-01-02' in values)
        self.assertTrue(datetime.date(2020, 1, 1) in values)
        self.assertTrue(datetime.date(2020, 1, 2) in values)

        post2 = index_many_set((idx1,  idx2), Index, union=True)
        assert isinstance(post2, Index)

        self.assertEqual(post2.name, None)
        self.assertEqual(post2.__class__, IndexDate)
        self.assertEqual(post2.values.tolist(),
                [datetime.date(2020, 1, 1),
                datetime.date(2020, 1, 2),
                datetime.date(2020, 1, 3)])

        post3 = index_many_set((idx1,  idx2), Index, union=False)
        assert isinstance(post3, Index)

        self.assertEqual(post3.name, None)
        self.assertEqual(post3.__class__, IndexDate)
        self.assertEqual(post3.values.tolist(),
                [datetime.date(2020, 1, 2)])

    def test_index_many_set_b(self) -> None:

        idx0 = Index(('1997-01-01', '1997-01-02'), name='foo')
        idx1 = IndexDate(('2020-01-01', '2020-01-02'), name='foo')
        idx2 = IndexDate(('2020-02-01', '2020-02-02'))

        post1 = index_many_set((idx0,  idx1), IndexGO, union=True)
        self.assertEqual(post1.__class__, IndexGO)

        post2 = index_many_set((idx1,  idx2), IndexGO, union=False)
        self.assertEqual(post2.__class__, IndexDateGO)


    def test_index_many_set_c(self) -> None:
        idx1 = IndexDate(('2020-02-01', '2020-02-02'))

        post1 = index_many_set((idx1,), Index, union=True)
        self.assertEqual(post1.__class__, IndexDate)
        self.assertTrue(idx1.equals(post1))

        # empty iterable returns an empty index
        post2 = index_many_set((), Index, union=True)
        self.assertEqual(len(post2), 0) #type: ignore


    def test_index_many_set_d(self) -> None:
        idx1 = Index(range(3), loc_is_iloc=True)
        idx2 = Index(range(3), loc_is_iloc=True)
        idx3 = index_many_set((idx1, idx2), Index, union=True)
        self.assertTrue(idx3._map is None) #type: ignore
        self.assertEqual(idx3.values.tolist(), [0, 1, 2]) #type: ignore

    def test_index_many_set_e(self) -> None:
        idx1 = Index(range(2), loc_is_iloc=True)
        idx2 = Index(range(4), loc_is_iloc=True)
        idx3 = index_many_set((idx1, idx2), Index, union=True)
        self.assertTrue(idx3._map is None) #type: ignore
        self.assertEqual(idx3.values.tolist(), [0, 1, 2, 3]) #type: ignore

    def test_index_many_set_f(self) -> None:
        idx1 = Index(range(2), loc_is_iloc=True)
        idx2 = Index(range(4), loc_is_iloc=True)
        idx3 = index_many_set((idx1, idx2), Index, union=False)
        self.assertTrue(idx3._map is None) #type: ignore
        self.assertEqual(idx3.values.tolist(), [0, 1]) #type: ignore

    def test_index_many_set_g(self) -> None:
        idx1 = Index(range(2), loc_is_iloc=True)
        idx2 = Index([3, 2, 1, 0])
        idx3 = index_many_set((idx1, idx2), Index, union=False)
        self.assertTrue(idx3._map is not None) #type: ignore
        self.assertEqual(idx3.values.tolist(), [0, 1]) #type: ignore



    #---------------------------------------------------------------------------
    def test_get_col_dtype_factory_a(self) -> None:

        func1 = get_col_dtype_factory((np.dtype(float), np.dtype(object)), None)
        self.assertEqual(func1(0), np.dtype(float))
        self.assertEqual(func1(1), np.dtype(object))


        func2 = get_col_dtype_factory((np.dtype(float), np.dtype(object)), ['foo', 'bar'])

        self.assertEqual(func2(0), np.dtype(float))
        self.assertEqual(func2(1), np.dtype(object))


        func3 = get_col_dtype_factory(dict(bar=np.dtype(bool)), ['foo', 'bar'])
        self.assertEqual(func3(0), None)
        self.assertEqual(func3(1), np.dtype(bool))

        with self.assertRaises(RuntimeError):
            _ = get_col_dtype_factory(dict(bar=np.dtype(bool)), None)


    #---------------------------------------------------------------------------
    def test_apex_to_name_a(self) -> None:
        self.assertEqual(
                apex_to_name([['foo']],
                        depth_level=-1,
                        axis=0,
                        axis_depth=1),
                'foo',
                )
        self.assertEqual(
                apex_to_name([['foo', 'bar']],
                        depth_level=-1,
                        axis=0,
                        axis_depth=2),
                ('foo', 'bar'),
                )
        self.assertEqual(
                apex_to_name([['', ''], ['foo', 'bar']],
                        depth_level=-1,
                        axis=0,
                        axis_depth=2),
                ('foo', 'bar'),
                )
        self.assertEqual(
                apex_to_name([['', ''], ['foo', 'bar']],
                        depth_level=0,
                        axis=0,
                        axis_depth=2),
                ('', ''),
                )
        self.assertEqual(
                apex_to_name([['a', 'b'], ['c', 'd']],
                        depth_level=[0, 1],
                        axis=0,
                        axis_depth=2),
                (('a', 'c'), ('b', 'd')),
                )
        self.assertEqual(
                apex_to_name([['a', 'b'], ['c', 'd']],
                        depth_level=[1, 0],
                        axis=0,
                        axis_depth=2),
                (('c', 'a'), ('d', 'b')),
                )

    def test_apex_to_name_b(self) -> None:
        self.assertEqual(
                apex_to_name([['foo']],
                        depth_level=-1,
                        axis=1,
                        axis_depth=1),
                'foo',
                )
        self.assertEqual(
                apex_to_name([['foo'], ['bar']],
                        depth_level=-1,
                        axis=1,
                        axis_depth=2),
                ('foo', 'bar'),
                )

        self.assertEqual(
                apex_to_name([['', 'foo'], ['', 'bar']],
                        depth_level=-1,
                        axis=1,
                        axis_depth=2),
                ('foo', 'bar'),
                )

        self.assertEqual(
                apex_to_name([['', 'foo'], ['', 'bar']],
                        depth_level=0,
                        axis=1,
                        axis_depth=2),
                ('', ''),
                )

        self.assertEqual(
                apex_to_name([['a', 'b'], ['c', 'd']],
                        depth_level=[0, 1],
                        axis=1,
                        axis_depth=2),
                (('a', 'b'), ('c', 'd')),
                )

        self.assertEqual(
                apex_to_name([['a', 'b'], ['c', 'd']],
                        depth_level=[1, 0],
                        axis=1,
                        axis_depth=2),
                (('b', 'a'), ('d', 'c')),
                )

    def test_apex_to_name_c(self) -> None:
        with self.assertRaises(AxisInvalid):
            _ = apex_to_name([['foo']], depth_level=-1, axis=3, axis_depth=1)


    #---------------------------------------------------------------------------
    def test_apply_binary_operator_blocks_by_column_a(self) -> None:
        blocks = (np.arange(3), np.arange(6).reshape(3, 2), np.arange(3) * 10)
        other = np.array([1, 0, 1])
        post = tuple(apply_binary_operator_blocks_columnar(
                    values=blocks, other=other, operator=lambda x, y: x * y)
                    )
        self.assertEqual(len(post), 4)
        self.assertTrue(all(p.ndim == 1 for p in post))
        self.assertEqual(np.stack(post, axis=1).tolist(),
                [[0, 0, 1, 0], [0, 0, 0, 0], [2, 4, 5, 20]])


    #---------------------------------------------------------------------------
    def test_container_to_exporter_attr(self) -> None:
        self.assertEqual(container_to_exporter_attr(Frame), 'to_frame')
        self.assertEqual(container_to_exporter_attr(FrameHE), 'to_frame_he')

        with self.assertRaises(NotImplementedError):
            container_to_exporter_attr(Series)

    #---------------------------------------------------------------------------
    def test_block_match_a(self) -> None:

        # opperate on the back of the list
        stack = [np.arange(4).reshape(2, 2), np.arange(0, 2), np.arange(6).reshape(2, 3)]
        post1 = list(get_block_match(1, stack))

        # takes front of each block in reverse order
        self.assertEqual(post1[0].tolist(), [0, 3])

        self.assertEqual([a.shape for a in post1],
                [(2,)])
        self.assertEqual([a.shape for a in stack],
                [(2, 2), (2,), (2, 2)])

    def test_block_match_b(self) -> None:

        # opperate on the back of the list
        stack = [np.arange(4).reshape(2, 2), np.arange(0, 2), np.arange(6).reshape(2, 3)]
        post1 = list(get_block_match(5, stack))

        # takes front of each block in reverse order
        self.assertEqual(post1[0].tolist(), [[0, 1, 2], [3, 4, 5]])

        self.assertEqual([a.shape for a in post1],
                [(2, 3), (2,), (2, 1)])
        self.assertEqual([a.shape for a in stack],
                [(2, 1)])

if __name__ == '__main__':
    unittest.main()
