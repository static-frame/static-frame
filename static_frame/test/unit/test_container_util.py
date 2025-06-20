from __future__ import annotations

import datetime

import frame_fixtures as ff
import numpy as np
import typing_extensions as tp

from static_frame import (
    Frame,
    Index,
    IndexDate,
    IndexDateGO,
    IndexGO,
    IndexHierarchy,
    IndexHierarchyGO,
    IndexSecond,
    Series,
)
from static_frame.core.container_util import (
    ContainerMap,
    apex_to_name,
    apply_binary_operator_blocks_columnar,
    arrays_from_index_frame,
    bloc_key_normalize,
    container_to_exporter_attr,
    get_block_match,
    get_col_dtype_factory,
    get_col_fill_value_factory,
    get_col_format_factory,
    group_from_container,
    index_from_optional_constructor,
    index_many_concat,
    index_many_to_one,
    is_static,
    key_to_ascending_key,
    matmul,
    pandas_to_numpy,
)

# from static_frame.core.container_util import pandas_version_under_1
from static_frame.core.exception import AxisInvalid
from static_frame.core.fill_value_auto import FillValueAuto
from static_frame.core.frame import FrameHE
from static_frame.core.util import ManyToOneType
from static_frame.test.test_case import TestCase


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
        idx1 = index_from_optional_constructor([1, 3, 4], default_constructor=Index)
        self.assertEqual(idx1.__class__, Index)

        # given a mutable index and an immutable default, get immutable version
        idx2 = index_from_optional_constructor(
            IndexGO((1, 3, 4)), default_constructor=Index
        )
        self.assertEqual(idx2.__class__, Index)

        # given a mutable index and an immutable default, get immutable version
        idx3 = index_from_optional_constructor(
            IndexGO((1, 3, 4)), default_constructor=IndexGO
        )
        self.assertEqual(idx3.__class__, IndexGO)

        # given a mutable index and an immutable default, get immutable version
        idx4 = index_from_optional_constructor(
            IndexSecond((1, 3, 4)), default_constructor=Index
        )
        self.assertEqual(idx4.__class__, IndexSecond)

    def test_index_from_optional_constructor_b(self) -> None:
        idx0 = IndexHierarchy.from_labels([('a', 0), ('a', 1), ('b', 0), ('b', 1)])
        idx1 = index_from_optional_constructor(
            idx0, default_constructor=IndexHierarchy.from_labels
        )

        # Since the default constructo is static, we should be able to reuse the index
        self.assertEqual(id(idx0), id(idx1))

    def test_index_from_optional_constructor_c(self) -> None:
        idx0 = IndexHierarchyGO.from_labels([('a', 0), ('a', 1), ('b', 0), ('b', 1)])
        idx1 = index_from_optional_constructor(
            idx0, default_constructor=IndexHierarchy.from_labels
        )

        # Since the default constructo is static, we should be able to reuse the index
        self.assertNotEqual(id(idx0), id(idx1))
        self.assertTrue(idx1.STATIC)

    def test_index_from_optional_constructor_d(self) -> None:
        idx0 = IndexHierarchy.from_labels([('a', 0), ('a', 1), ('b', 0), ('b', 1)])
        idx1 = index_from_optional_constructor(
            idx0, default_constructor=IndexHierarchyGO.from_labels
        )

        # Since the default constructo is static, we should be able to reuse the index
        self.assertNotEqual(id(idx0), id(idx1))
        self.assertFalse(idx1.STATIC)

    def test_matmul_a(self) -> None:
        # lhs: frame, rhs: array

        f1 = Frame.from_items((('a', (1, 2, 3)), ('b', (3, 4, 5))), index=('x', 'y', 'z'))

        self.assertEqual(matmul(f1, [4, 3]).to_pairs(), (('x', 13), ('y', 20), ('z', 27)))

        self.assertEqual(
            matmul(f1, np.array([4, 3])).to_pairs(), (('x', 13), ('y', 20), ('z', 27))
        )

        self.assertEqual(matmul(f1, [3, 4]).to_pairs(), (('x', 15), ('y', 22), ('z', 29)))

        self.assertEqual(
            matmul(f1, np.array([3, 4])).to_pairs(), (('x', 15), ('y', 22), ('z', 29))
        )

    def test_matmul_b(self) -> None:
        # lhs: frame, rhs: array

        f1 = Frame.from_items((('a', (1, 2, 3)), ('b', (3, 4, 5))), index=('x', 'y', 'z'))

        # get an auto incremented integer columns
        self.assertEqual(
            matmul(f1, np.arange(10).reshape(2, 5)).to_pairs(),
            (
                (0, (('x', 15), ('y', 20), ('z', 25))),
                (1, (('x', 19), ('y', 26), ('z', 33))),
                (2, (('x', 23), ('y', 32), ('z', 41))),
                (3, (('x', 27), ('y', 38), ('z', 49))),
                (4, (('x', 31), ('y', 44), ('z', 57))),
            ),
        )

    def test_matmul_c(self) -> None:
        # lhs: frame, rhs: Series, 1D array

        f1 = Frame.from_items((('a', (1, 2, 3)), ('b', (3, 4, 5))), index=('x', 'y', 'z'))
        s1 = Series((10, 11), index=('a', 'b'))

        self.assertEqual(matmul(f1, s1).to_pairs(), (('x', 43), ('y', 64), ('z', 85)))

        self.assertEqual(
            matmul(f1, s1.values).to_pairs(), (('x', 43), ('y', 64), ('z', 85))
        )

        with self.assertRaises(RuntimeError):
            matmul(f1, np.arange(20).reshape(5, 4))

    def test_matmul_d(self) -> None:
        # lhs: series, rhs: frame

        f1 = Frame.from_items((('a', (1, 2, 3)), ('b', (3, 4, 5))), index=('x', 'y', 'z'))

        s1 = Series((3, 4, 2), index=('x', 'y', 'z'))

        self.assertEqual(matmul(s1, f1).to_pairs(), (('a', 17), ('b', 35)))

        # produces a Series indexed 0, 1
        self.assertEqual(
            matmul(s1, f1.values).to_pairs(),  # type: ignore
            ((0, 17), (1, 35)),
        )

    def test_matmul_e(self) -> None:
        # lhs: series, rhs: series

        s1 = Series((3, 4, 2), index=('x', 'y', 'z'))

        s2 = Series((10, 11, 12), index=('x', 'y', 'z'))

        self.assertEqual(matmul(s1, s2), 98)
        self.assertEqual(matmul(s1, s2.values), 98)

    def test_matmul_f(self) -> None:
        # lhs: array 1D, rhs: array 2D, Frame

        f1 = Frame.from_items((('a', (1, 2, 3)), ('b', (3, 4, 5))), index=('x', 'y', 'z'))

        self.assertEqual(
            matmul([3, 4, 5], f1.values).tolist(),  # type: ignore
            [26, 50],
        )

        self.assertEqual(matmul([3, 4, 5], f1).to_pairs(), (('a', 26), ('b', 50)))

    def test_matmul_g(self) -> None:
        # lhs: array 1D, rhs: array 1D, Series

        s1 = Series((3, 4, 2), index=('x', 'y', 'z'))
        self.assertEqual(matmul([10, 11, 12], s1.values), 98)  # type: ignore
        self.assertEqual(matmul([10, 11, 12], s1), 98)

        with self.assertRaises(RuntimeError):
            self.assertEqual(matmul(s1, [10, 11]), 98)

    def test_matmul_h(self) -> None:
        # lhs: array 2D, rhs: array 2D, Frame

        f1 = Frame.from_dict(dict(a=(1, 2, 3, 4), b=(5, 6, 7, 8)), index=tuple('wxyz'))
        f2 = Frame.from_dict(dict(p=(1, 2), q=(3, 4), r=(5, 6)), index=tuple('ab'))

        self.assertEqual(
            matmul(f1.values, f2).to_pairs(),
            (
                ('p', ((0, 11), (1, 14), (2, 17), (3, 20))),
                ('q', ((0, 23), (1, 30), (2, 37), (3, 44))),
                ('r', ((0, 35), (1, 46), (2, 57), (3, 68))),
            ),
        )

        self.assertEqual(
            matmul(f1, f2.values).to_pairs(),
            (
                (0, (('w', 11), ('x', 14), ('y', 17), ('z', 20))),
                (1, (('w', 23), ('x', 30), ('y', 37), ('z', 44))),
                (2, (('w', 35), ('x', 46), ('y', 57), ('z', 68))),
            ),
        )

        with self.assertRaises(RuntimeError):
            matmul(f1, np.arange(25).reshape(5, 5))

    def test_matmul_i(self) -> None:
        import itertools as it

        f1 = Frame.from_dict(dict(a=(1, 2), b=(5, 6)), index=tuple('yz'))

        f_container = lambda x: x
        f_values = lambda x: x.values

        for pair in (
            (f1, f1.T),
            (f1, f1.loc['y']),
            (f1['a'], f1),
            (f1.loc['y'], f1.loc['z']),
        ):
            for x, y in it.combinations(
                (f_container, f_values, f_container, f_values), 2
            ):
                post = matmul(x(pair[0]), y(pair[1]))  # type: ignore
                if isinstance(post, (Series, Frame)):  # type: ignore
                    self.assertTrue(
                        post.values.tolist(), (pair[0].values @ pair[1].values).tolist()
                    )  # type: ignore
                elif isinstance(post, np.ndarray):
                    self.assertTrue(
                        post.tolist(), (pair[0].values @ pair[1].values).tolist()
                    )

    def test_matmul_j(self) -> None:
        f1 = Frame.from_dict(
            dict(a=(1, 2, 3), b=(5, 6, 7)), index=tuple('xyz'), name='foo'
        )
        a1 = np.array([[5], [0]])

        with self.assertRaises(RuntimeError):
            _ = matmul(a1, f1)

    # ---------------------------------------------------------------------------

    def test_key_to_ascending_key_a(self) -> None:
        self.assertEqual(key_to_ascending_key([9, 5, 1], 3), [1, 5, 9])
        self.assertEqual(key_to_ascending_key(np.array([9, 5, 1]), 3).tolist(), [1, 5, 9])  # type: ignore

        self.assertEqual(key_to_ascending_key(slice(3, 0, -1), 3), slice(1, 3, None))

        self.assertEqual(key_to_ascending_key(100, 3), 100)

        self.assertEqual(key_to_ascending_key([], 3), [])

        self.assertEqual(
            key_to_ascending_key(  # type: ignore
                Series(('a', 'b', 'c'), index=(9, 5, 1)), 3
            ).values.tolist(),
            ['c', 'b', 'a'],
        )

        f1 = Frame.from_dict(dict(b=(1, 2), a=(5, 6)), index=tuple('yz'))
        f2 = key_to_ascending_key(f1, f1.shape[1])
        self.assertEqual(f2.columns.values.tolist(), ['a', 'b'])  # type: ignore

    def test_key_to_ascending_key_b(self) -> None:
        with self.assertRaises(RuntimeError):
            key_to_ascending_key(dict(a=3), size=3)

    # ---------------------------------------------------------------------------

    def test_pandas_to_numpy_a(self) -> None:
        import pandas as pd

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

        post1 = bloc_key_normalize(f1['a':] >= 5, f1)  # type: ignore
        self.assertEqual(post1.tolist(), [[False, True], [False, True]])

        post2 = bloc_key_normalize(f1 < 5, f1)
        self.assertEqual(post2.tolist(), [[True, False], [True, False]])

    def test_index_many_concat_a1(self) -> None:
        idx0 = Index(('1997-01-01', '1997-01-02'), name='foo')
        idx1 = IndexDate(('2020-01-01', '2020-01-02'), name='foo')
        idx2 = IndexDate(('2020-02-01', '2020-02-02'))

        post1 = index_many_concat((idx0, idx1), Index)
        assert isinstance(post1, Index)

        post1b = index_many_concat((idx0, idx1), Index, IndexDate)
        assert isinstance(post1b, IndexDate)

        post1c = index_many_concat((idx0, idx1, idx2), Index, IndexDate)
        assert isinstance(post1c, IndexDate)

        self.assertEqual(
            post1.values.tolist(),
            [
                '1997-01-01',
                '1997-01-02',
                datetime.date(2020, 1, 1),
                datetime.date(2020, 1, 2),
            ],
        )
        self.assertEqual(post1.name, 'foo')
        self.assertEqual(post1.__class__, Index)

    def test_index_many_concat_a2(self) -> None:
        # idx0 = Index(('1997-01-01', '1997-01-02'), name='foo')
        idx1 = IndexDate(('2020-01-01', '2020-01-02'), name='foo')
        idx2 = IndexDate(('2020-02-01', '2020-02-02'))

        post2 = index_many_concat((idx1, idx2), Index)
        assert isinstance(post2, Index)
        self.assertEqual(post2.__class__, IndexDate)
        self.assertEqual(
            post2.values.tolist(),
            [
                datetime.date(2020, 1, 1),
                datetime.date(2020, 1, 2),
                datetime.date(2020, 2, 1),
                datetime.date(2020, 2, 2),
            ],
        )

    def test_index_many_concat_b(self) -> None:
        idx0 = Index(('1997-01-01', '1997-01-02'), name='foo')
        idx1 = IndexDate(('2020-01-01', '2020-01-02'), name='foo')
        idx2 = IndexDate(('2020-02-01', '2020-02-02'))

        post1 = index_many_concat((idx0, idx1), IndexGO)
        self.assertEqual(post1.__class__, IndexGO)

        post2 = index_many_concat((idx1, idx2), IndexGO)
        self.assertEqual(post2.__class__, IndexDateGO)

    def test_index_many_concat_c(self) -> None:
        from datetime import date

        i1 = IndexHierarchy.from_labels(
            [[1, date(2019, 1, 1)], [2, date(2019, 1, 2)]],
            index_constructors=[Index, IndexDate],
        )

        i2 = IndexHierarchy.from_labels(
            [[2, date(2019, 1, 3)], [3, date(2019, 1, 4)]],
            index_constructors=[Index, IndexDate],
        )

        i3 = IndexHierarchy.from_labels(
            [[4, date(2019, 1, 5)], [5, date(2019, 1, 6)]],
            index_constructors=[Index, IndexDate],
        )

        i4 = IndexHierarchy.from_labels([[4, date(2019, 1, 5)], [5, date(2019, 1, 6)]])

        i5 = index_many_concat((i1, i2, i3), cls_default=Index)
        assert isinstance(i5, IndexHierarchy)

        self.assertEqual(i5.index_types.to_pairs(), ((0, Index), (1, IndexDate)))
        self.assertEqual(
            i5.values.tolist(),
            [
                [1, date(2019, 1, 1)],
                [2, date(2019, 1, 2)],
                [2, date(2019, 1, 3)],
                [3, date(2019, 1, 4)],
                [4, date(2019, 1, 5)],
                [5, date(2019, 1, 6)],
            ],
        )

        # with unaligned index types we fall back in Index
        i6 = index_many_concat((i1, i2, i4), cls_default=Index)
        assert isinstance(i6, IndexHierarchy)

        self.assertEqual(i6.index_types.to_pairs(), ((0, Index), (1, Index)))

    def test_index_many_concat_d(self) -> None:
        from datetime import date

        i1 = IndexHierarchy.from_labels(
            [[1, date(2019, 1, 1)], [2, date(2019, 1, 2)]],
            index_constructors=[Index, IndexDate],
        )

        i2 = IndexHierarchy.from_labels(
            [[2, date(2019, 1, 3)], [3, date(2019, 1, 4)]],
            index_constructors=[Index, IndexDate],
        )

        post1 = index_many_concat((i1, i2), cls_default=IndexGO)
        self.assertEqual(post1.__class__, IndexHierarchyGO)
        assert isinstance(post1, IndexHierarchy)
        self.assertEqual(
            post1.values.tolist(),
            [
                [1, date(2019, 1, 1)],
                [2, date(2019, 1, 2)],
                [2, date(2019, 1, 3)],
                [3, date(2019, 1, 4)],
            ],
        )

    def test_index_many_concat_e(self) -> None:
        idx1 = IndexDateGO(('2020-01-01', '2020-01-02'))
        idx2 = IndexDateGO(('2020-02-01', '2020-02-02'))

        post1 = index_many_concat((idx1, idx2), cls_default=Index)

        self.assertEqual(post1.__class__, IndexDate)
        self.assertEqual(
            post1.values.tolist(),  # type: ignore
            [
                datetime.date(2020, 1, 1),
                datetime.date(2020, 1, 2),
                datetime.date(2020, 2, 1),
                datetime.date(2020, 2, 2),
            ],
        )

    def test_index_many_concat_f(self) -> None:
        idx1 = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        idx2 = IndexHierarchy.from_product(('c', 'd'), (1, 2))

        post = index_many_concat((idx1, idx2), cls_default=Index)
        post = tp.cast(IndexHierarchy, post)

        self.assertEqual([d.kind for d in post.dtypes.values], ['U', 'i'])
        self.assertEqual(
            post.to_frame().to_pairs(),
            (
                (
                    0,
                    (
                        (0, 'a'),
                        (1, 'a'),
                        (2, 'b'),
                        (3, 'b'),
                        (4, 'c'),
                        (5, 'c'),
                        (6, 'd'),
                        (7, 'd'),
                    ),
                ),
                (1, ((0, 1), (1, 2), (2, 1), (3, 2), (4, 1), (5, 2), (6, 1), (7, 2))),
            ),
        )

    def test_index_many_concat_g(self) -> None:
        idx1 = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        idx2 = Index(('a', 'b'))

        # both raise for un-aligned depths, regardless of order
        with self.assertRaises(RuntimeError):
            post = index_many_concat((idx1, idx2), cls_default=Index)

        with self.assertRaises(RuntimeError):
            post = index_many_concat((idx2, idx1), cls_default=Index)

    # ---------------------------------------------------------------------------

    def test_index_many_set_a(self) -> None:
        idx0 = Index(('1997-01-01', '1997-01-02'), name='foo')
        idx1 = IndexDate(('2020-01-01', '2020-01-02'), name='foo')
        idx2 = IndexDate(('2020-01-02', '2020-01-03'))

        post1 = index_many_to_one(
            (idx0, idx1), Index, many_to_one_type=ManyToOneType.UNION
        )
        assert isinstance(post1, Index)

        self.assertEqual(post1.name, 'foo')
        self.assertEqual(post1.__class__, Index)

        # the result of this operation is an unstable ordering
        values = set(post1.values)
        self.assertTrue('1997-01-01' in values)
        self.assertTrue('1997-01-02' in values)
        self.assertTrue(datetime.date(2020, 1, 1) in values)
        self.assertTrue(datetime.date(2020, 1, 2) in values)

        post2 = index_many_to_one(
            (idx1, idx2), Index, many_to_one_type=ManyToOneType.UNION
        )
        assert isinstance(post2, Index)

        self.assertEqual(post2.name, None)
        self.assertEqual(post2.__class__, IndexDate)
        self.assertEqual(
            post2.values.tolist(),
            [
                datetime.date(2020, 1, 1),
                datetime.date(2020, 1, 2),
                datetime.date(2020, 1, 3),
            ],
        )

        post3 = index_many_to_one(
            (idx1, idx2), Index, many_to_one_type=ManyToOneType.INTERSECT
        )
        assert isinstance(post3, Index)

        self.assertEqual(post3.name, None)
        self.assertEqual(post3.__class__, IndexDate)
        self.assertEqual(post3.values.tolist(), [datetime.date(2020, 1, 2)])

    def test_index_many_set_b(self) -> None:
        idx0 = Index(('1997-01-01', '1997-01-02'), name='foo')
        idx1 = IndexDate(('2020-01-01', '2020-01-02'), name='foo')
        idx2 = IndexDate(('2020-02-01', '2020-02-02'))

        post1 = index_many_to_one(
            (idx0, idx1), IndexGO, many_to_one_type=ManyToOneType.UNION
        )
        self.assertEqual(post1.__class__, IndexGO)

        post2 = index_many_to_one(
            (idx1, idx2), IndexGO, many_to_one_type=ManyToOneType.INTERSECT
        )
        self.assertEqual(post2.__class__, IndexDateGO)

    def test_index_many_set_c(self) -> None:
        idx1 = IndexDate(('2020-02-01', '2020-02-02'))

        post1 = index_many_to_one((idx1,), Index, many_to_one_type=ManyToOneType.UNION)
        self.assertEqual(post1.__class__, IndexDate)
        self.assertTrue(idx1.equals(post1))

        # empty iterable returns an empty index
        post2 = index_many_to_one((), Index, many_to_one_type=ManyToOneType.UNION)
        self.assertEqual(len(post2), 0)

    def test_index_many_set_d(self) -> None:
        idx1 = Index(range(3), loc_is_iloc=True)
        idx2 = Index(range(3), loc_is_iloc=True)
        idx3 = index_many_to_one(
            (idx1, idx2), Index, many_to_one_type=ManyToOneType.UNION
        )
        self.assertTrue(idx3._map is None)  # type: ignore
        self.assertEqual(idx3.values.tolist(), [0, 1, 2])

    def test_index_many_set_e(self) -> None:
        idx1 = Index(range(2), loc_is_iloc=True)
        idx2 = Index(range(4), loc_is_iloc=True)
        idx3 = index_many_to_one(
            (idx1, idx2), Index, many_to_one_type=ManyToOneType.UNION
        )
        self.assertTrue(idx3._map is None)  # type: ignore
        self.assertEqual(idx3.values.tolist(), [0, 1, 2, 3])

    def test_index_many_set_f(self) -> None:
        idx1 = Index(range(2), loc_is_iloc=True)
        idx2 = Index(range(4), loc_is_iloc=True)
        idx3 = index_many_to_one(
            (idx1, idx2), Index, many_to_one_type=ManyToOneType.INTERSECT
        )
        self.assertTrue(idx3._map is None)  # type: ignore
        self.assertEqual(idx3.values.tolist(), [0, 1])

    def test_index_many_set_g(self) -> None:
        idx1 = Index(range(2), loc_is_iloc=True)
        idx2 = Index([3, 2, 1, 0])
        idx3 = index_many_to_one(
            (idx1, idx2), Index, many_to_one_type=ManyToOneType.INTERSECT
        )
        self.assertTrue(idx3._map is not None)  # type: ignore
        self.assertEqual(idx3.values.tolist(), [0, 1])

    def test_index_many_set_h(self) -> None:
        post1 = index_many_to_one(
            (),
            Index,
            many_to_one_type=ManyToOneType.UNION,
            explicit_constructor=IndexDate,
        )
        self.assertIs(post1.__class__, IndexDate)

    def test_index_many_set_i(self) -> None:
        idx1 = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        idx2 = IndexHierarchy.from_product(('a', 'b'), (1, 2))

        post = index_many_to_one(
            (idx1, idx2), Index, many_to_one_type=ManyToOneType.UNION
        )
        post = tp.cast(IndexHierarchy, post)

        self.assertEqual([d.kind for d in post.dtypes.values], ['U', 'i'])
        self.assertEqual(post.values.tolist(), [['a', 1], ['a', 2], ['b', 1], ['b', 2]])

    def test_index_many_set_j(self) -> None:
        idx1 = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        idx2 = IndexHierarchy.from_product(('a', 'c'), (1, 2))

        post = index_many_to_one(
            (idx1, idx2), Index, many_to_one_type=ManyToOneType.UNION
        )
        post = tp.cast(IndexHierarchy, post)

        self.assertEqual([d.kind for d in post.dtypes.values], ['U', 'i'])
        self.assertEqual(
            post.values.tolist(),
            [['a', 1], ['a', 2], ['b', 1], ['b', 2], ['c', 1], ['c', 2]],
        )

    def test_index_many_set_k(self) -> None:
        idx1 = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        idx2 = IndexHierarchy.from_product(('a', 'c'), (1, 2))

        post = index_many_to_one(
            (idx1, idx2), Index, many_to_one_type=ManyToOneType.INTERSECT
        )
        post = tp.cast(IndexHierarchy, post)

        self.assertEqual([d.kind for d in post.dtypes.values], ['U', 'i'])
        self.assertEqual(
            post.values.tolist(),
            [['a', 1], ['a', 2]],
        )

    def test_index_many_set_l(self) -> None:
        idx0 = Index(('1997-01-01', '1997-01-02'), name='foo')
        idx1 = IndexDate(('2020-01-01', '2020-01-02'), name='foo')
        # idx2 = IndexDate(('2020-01-02', '2020-01-03'))

        post1 = index_many_to_one(
            (idx0, idx1), Index, many_to_one_type=ManyToOneType.UNION
        )
        assert isinstance(post1, Index)

        self.assertEqual(post1.name, 'foo')
        self.assertEqual(post1.__class__, Index)

    def test_index_many_set_m(self) -> None:
        idx0 = Index((2, 5, 6, 8), name='foo')
        idx1 = Index((2, 8), name='foo')

        idx2 = index_many_to_one(
            (idx0, idx1), Index, many_to_one_type=ManyToOneType.DIFFERENCE
        )

        self.assertEqual(list(idx2), [5, 6])
        self.assertEqual(idx2.name, 'foo')

    def test_index_many_set_n(self) -> None:
        idx0 = IndexDate(('1997-01-01', '1997-01-02', '2020-01-02'), name='foo')
        idx1 = IndexDate(('1997-01-01', '2020-01-02'), name='foo')

        post1 = index_many_to_one(
            (idx0, idx1), Index, many_to_one_type=ManyToOneType.DIFFERENCE
        )

        self.assertEqual(post1.name, 'foo')
        self.assertEqual(post1.__class__, IndexDate)
        self.assertEqual(list(post1), [np.datetime64('1997-01-02')])

    # ---------------------------------------------------------------------------

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

    def test_get_col_dtype_factory_b(self) -> None:
        func = get_col_dtype_factory({1: np.dtype(bool)}, None)
        self.assertEqual(func(0), None)
        self.assertEqual(func(1), np.dtype(bool))

    # ---------------------------------------------------------------------------

    def test_get_col_fill_value_a(self) -> None:
        func1 = get_col_fill_value_factory({'a': -1, 'b': 2}, columns=('b', 'a'))
        self.assertEqual(func1(0, np.dtype(float)), 2)
        self.assertEqual(func1(1, np.dtype(float)), -1)

    def test_get_col_fill_value_b(self) -> None:
        func1 = get_col_fill_value_factory(('x', 1), columns=('b', 'a'))
        self.assertEqual(func1(0, np.dtype(float)), ('x', 1))
        self.assertEqual(func1(1, np.dtype(float)), ('x', 1))

    def test_get_col_fill_value_c(self) -> None:
        func1 = get_col_fill_value_factory(('x', 1), columns=('b', 'a'))
        self.assertEqual(func1(0, np.dtype(float)), ('x', 1))
        self.assertEqual(func1(1, np.dtype(float)), ('x', 1))

    def test_get_col_fill_value_d(self) -> None:
        func1 = get_col_fill_value_factory('x', columns=('b', 'a'))
        self.assertEqual(func1(0, np.dtype(float)), 'x')
        self.assertEqual(func1(1, np.dtype(float)), 'x')

    def test_get_col_fill_value_e(self) -> None:
        func1 = get_col_fill_value_factory(
            (c for c in 'xy'),
            columns=('b', 'a'),
        )
        self.assertEqual(func1(0, np.dtype(float)), 'x')
        self.assertEqual(func1(1, np.dtype(float)), 'y')

    def test_get_col_fill_value_f(self) -> None:
        func1 = get_col_fill_value_factory(FillValueAuto, columns=None)
        self.assertEqual(func1(0, np.dtype(object)), None)
        self.assertEqual(func1(1, np.dtype(str)), '')

    def test_get_col_fill_value_g(self) -> None:
        func1 = get_col_fill_value_factory(FillValueAuto(O='', U='na'), columns=None)
        self.assertEqual(func1(0, np.dtype(object)), '')
        self.assertEqual(func1(1, np.dtype(str)), 'na')

    # ---------------------------------------------------------------------------

    def test_get_col_format_a(self) -> None:
        func = get_col_format_factory('{}', ('a', 'b', 'c'))
        self.assertEqual(func(0), '{}')
        self.assertEqual(func(1), '{}')
        self.assertEqual(func(2), '{}')

    def test_get_col_format_b(self) -> None:
        func = get_col_format_factory(('a', 'b', 'c'), ('a', 'b', 'c'))
        self.assertEqual(func(0), 'a')
        self.assertEqual(func(1), 'b')
        self.assertEqual(func(2), 'c')

    def test_get_col_format_c(self) -> None:
        func = get_col_format_factory(('a', 'b', 'c'), range(3))
        self.assertEqual(func(0), 'a')
        self.assertEqual(func(1), 'b')
        self.assertEqual(func(2), 'c')

    def test_get_col_format_d1(self) -> None:
        func = get_col_format_factory({'b': 'x{}', 'c': 'y{}'}, ('a', 'b', 'c'))
        self.assertEqual(func(0), '{}')
        self.assertEqual(func(1), 'x{}')
        self.assertEqual(func(2), 'y{}')

    def test_get_col_format_e(self) -> None:
        func = get_col_format_factory((f'{{:{i}}}' for i in range(3)), ('a', 'b', 'c'))
        self.assertEqual(func(0), '{:0}')
        self.assertEqual(func(1), '{:1}')
        self.assertEqual(func(2), '{:2}')

    # ---------------------------------------------------------------------------

    def test_apex_to_name_a(self) -> None:
        self.assertEqual(
            apex_to_name([['foo']], depth_level=-1, axis=0, axis_depth=1),
            'foo',
        )
        self.assertEqual(
            apex_to_name([['foo', 'bar']], depth_level=-1, axis=0, axis_depth=2),
            ('foo', 'bar'),
        )
        self.assertEqual(
            apex_to_name(
                [['', ''], ['foo', 'bar']], depth_level=-1, axis=0, axis_depth=2
            ),
            ('foo', 'bar'),
        )
        self.assertEqual(
            apex_to_name([['', ''], ['foo', 'bar']], depth_level=0, axis=0, axis_depth=2),
            ('', ''),
        )
        self.assertEqual(
            apex_to_name(
                [['a', 'b'], ['c', 'd']], depth_level=[0, 1], axis=0, axis_depth=2
            ),
            (('a', 'c'), ('b', 'd')),
        )
        self.assertEqual(
            apex_to_name(
                [['a', 'b'], ['c', 'd']], depth_level=[1, 0], axis=0, axis_depth=2
            ),
            (('c', 'a'), ('d', 'b')),
        )

    def test_apex_to_name_b(self) -> None:
        self.assertEqual(
            apex_to_name([['foo']], depth_level=-1, axis=1, axis_depth=1),
            'foo',
        )
        self.assertEqual(
            apex_to_name([['foo'], ['bar']], depth_level=-1, axis=1, axis_depth=2),
            ('foo', 'bar'),
        )

        self.assertEqual(
            apex_to_name(
                [['', 'foo'], ['', 'bar']], depth_level=-1, axis=1, axis_depth=2
            ),
            ('foo', 'bar'),
        )

        self.assertEqual(
            apex_to_name([['', 'foo'], ['', 'bar']], depth_level=0, axis=1, axis_depth=2),
            ('', ''),
        )

        self.assertEqual(
            apex_to_name(
                [['a', 'b'], ['c', 'd']], depth_level=[0, 1], axis=1, axis_depth=2
            ),
            (('a', 'b'), ('c', 'd')),
        )

        self.assertEqual(
            apex_to_name(
                [['a', 'b'], ['c', 'd']], depth_level=[1, 0], axis=1, axis_depth=2
            ),
            (('b', 'a'), ('d', 'c')),
        )

    def test_apex_to_name_c(self) -> None:
        with self.assertRaises(AxisInvalid):
            _ = apex_to_name([['foo']], depth_level=-1, axis=3, axis_depth=1)

    # ---------------------------------------------------------------------------

    def test_apply_binary_operator_blocks_by_column_a(self) -> None:
        blocks = (np.arange(3), np.arange(6).reshape(3, 2), np.arange(3) * 10)
        other = np.array([1, 0, 1])
        post = tuple(
            apply_binary_operator_blocks_columnar(
                values=blocks, other=other, operator=lambda x, y: x * y
            )
        )
        self.assertEqual(len(post), 4)
        self.assertTrue(all(p.ndim == 1 for p in post))
        self.assertEqual(
            np.stack(post, axis=1).tolist(), [[0, 0, 1, 0], [0, 0, 0, 0], [2, 4, 5, 20]]
        )

    # ---------------------------------------------------------------------------

    def test_container_to_exporter_attr(self) -> None:
        self.assertEqual(container_to_exporter_attr(Frame), 'to_frame')
        self.assertEqual(container_to_exporter_attr(FrameHE), 'to_frame_he')

        with self.assertRaises(NotImplementedError):
            container_to_exporter_attr(Series)

    # ---------------------------------------------------------------------------

    def test_block_match_a(self) -> None:
        # opperate on the back of the list
        stack = [
            np.arange(4).reshape(2, 2),
            np.arange(0, 2),
            np.arange(6).reshape(2, 3),
        ]
        post1 = list(get_block_match(1, stack))

        # takes front of each block in reverse order
        self.assertEqual(post1[0].tolist(), [0, 3])

        self.assertEqual([a.shape for a in post1], [(2,)])
        self.assertEqual([a.shape for a in stack], [(2, 2), (2,), (2, 2)])

    def test_block_match_b(self) -> None:
        # opperate on the back of the list
        stack = [
            np.arange(4).reshape(2, 2),
            np.arange(0, 2),
            np.arange(6).reshape(2, 3),
        ]
        post1 = list(get_block_match(5, stack))

        # takes front of each block in reverse order
        self.assertEqual(post1[0].tolist(), [[0, 1, 2], [3, 4, 5]])

        self.assertEqual([a.shape for a in post1], [(2, 3), (2,), (2, 1)])
        self.assertEqual([a.shape for a in stack], [(2, 1)])

    # ---------------------------------------------------------------------------
    def test_group_from_container_a(self) -> None:
        idx = Index(('a', 'b', 'c'))
        with self.assertRaises(ValueError):
            group_from_container(idx, np.arange(12).reshape(3, 2, 2), None, 0)

    def test_group_from_container_b(self) -> None:
        idx = Index(('a', 'b', 'c'))
        with self.assertRaises(ValueError):
            group_from_container(idx, 'a', None, 0)

    def test_group_from_container_c(self) -> None:
        idx = Index(('a', 'b', 'c'))
        with self.assertRaises(RuntimeError):
            group_from_container(idx, (2, 2), None, 0)

    def test_group_from_container_d(self) -> None:
        idx = Index(('a', 'b', 'c'))
        gs = np.arange(6).reshape(3, 2)
        group_from_container(idx, gs, None, 0)
        # if wrong axis does not align
        with self.assertRaises(RuntimeError):
            group_from_container(idx, gs, None, 1)

    def test_group_from_container_e(self) -> None:
        idx = Index(('a', 'b', 'c'))
        s = Frame.from_element(0, index=('a', 'c'), columns=('x',))
        post = group_from_container(idx, s, None, 0)
        self.assertEqual(post.tolist(), [[0], [None], [0]])

    def test_get_containers(self) -> None:
        keys_gc = set(cls.__name__ for cls in TestCase.get_containers())
        keys_cm = set(ContainerMap.keys())

        # these two different utilities have slightly different constituents as they are used for different purposes
        self.assertEqual(
            keys_gc - keys_cm, {'ContainerOperand', 'ContainerOperandSequence'}
        )
        self.assertEqual(
            keys_cm - keys_gc,
            {
                'ILoc',
                'TypeClinic',
                'CallGuard',
                'MemoryDisplay',
                'ClinicResult',
                'HLoc',
                'FillValueAuto',
                'Require',
            },
        )

        self.assertEqual(
            keys_cm & keys_gc,
            {
                'FrameHE',
                'IndexSecondGO',
                'IndexSecond',
                'IndexDateGO',
                'Bus',
                'IndexMinute',
                'Index',
                'Frame',
                'IndexDate',
                'IndexYearMonth',
                'IndexYearGO',
                'IndexMicrosecondGO',
                'Yarn',
                'IndexNanosecond',
                'IndexYearMonthGO',
                'IndexNanosecondGO',
                'IndexHourGO',
                'Batch',
                'Quilt',
                'IndexMinuteGO',
                'FrameGO',
                'IndexHour',
                'Series',
                'IndexGO',
                'IndexHierarchy',
                'IndexMillisecondGO',
                'TypeBlocks',
                'IndexYear',
                'SeriesHE',
                'IndexMicrosecond',
                'IndexMillisecond',
                'IndexHierarchyGO',
            },
        )

    def test_get_container_map_a(self) -> None:
        if hasattr(ContainerMap, '_map'):
            delattr(ContainerMap, '_map')
        keys = set(ContainerMap.keys())

        delattr(ContainerMap, '_map')
        for k in keys:
            ContainerMap.get(k)

    def test_arrays_from_index_frame_a(self) -> None:
        f = ff.parse('s(4,4)|v(int,str,bool,str)|i((I,I,I),(str,int,str))')
        post, labels = arrays_from_index_frame(f, 0, 2)
        self.assertEqual(
            [a.tolist() for a in post],
            [['zZbu', 'zZbu', 'zZbu', 'zZbu'], [True, False, False, True]],
        )
        self.assertEqual(labels, [0, 2])

    def test_arrays_from_index_frame_b(self) -> None:
        f = ff.parse('s(4,4)|v(int,str,bool,str)|i((I,I,I),(str,int,str))')
        post, labels = arrays_from_index_frame(f, 0, [0, 3])
        self.assertEqual(
            [a.tolist() for a in post],
            [
                ['zZbu', 'zZbu', 'zZbu', 'zZbu'],
                [-88017, 92867, 84967, 13448],
                ['z2Oo', 'z5l6', 'zCE3', 'zr4u'],
            ],
        )
        self.assertEqual(labels, [0, 0, 3])

    def test_arrays_from_index_frame_c(self) -> None:
        f = ff.parse('s(4,4)|v(int,str,bool,str)|i((I,I,I),(str,int,str))')
        post, labels = arrays_from_index_frame(f, [0, 2], [0, 3])
        self.assertEqual(
            [a.tolist() for a in post],
            [
                ['zZbu', 'zZbu', 'zZbu', 'zZbu'],
                ['zDVQ', 'z5hI', 'zyT8', 'zS6w'],
                [-88017, 92867, 84967, 13448],
                ['z2Oo', 'z5l6', 'zCE3', 'zr4u'],
            ],
        )
        self.assertEqual(labels, [0, 2, 0, 3])

    def test_arrays_from_index_frame_d(self) -> None:
        f = ff.parse('s(4,4)|v(int,str,bool,str)|i((I,I,I),(str,int,str))')
        post, labels = arrays_from_index_frame(f, slice(1, None), None)
        self.assertEqual(
            [a.tolist() for a in post],
            [[105269, 105269, 119909, 119909], ['zDVQ', 'z5hI', 'zyT8', 'zS6w']],
        )
        self.assertEqual(labels, [1, 2])


if __name__ == '__main__':
    import unittest

    unittest.main()
