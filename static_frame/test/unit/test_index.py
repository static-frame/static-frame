
import unittest
import numpy as np  # type: ignore
import pickle
import datetime
import typing as tp
from io import StringIO

from static_frame import Index
from static_frame import IndexGO
from static_frame import IndexDate
from static_frame import IndexHierarchy
from static_frame import Series
# from static_frame import Frame
from static_frame import IndexYear

# from static_frame import HLoc
from static_frame import ILoc


from static_frame.test.test_case import TestCase
from static_frame.core.index import _requires_reindex
from static_frame.core.exception import ErrorInitIndex

class TestUnit(TestCase):

    def test_index_init_a(self) -> None:
        idx1 = Index(('a', 'b', 'c', 'd'), name='foo')
        idx2 = Index(idx1)

        self.assertEqual(idx1.name, 'foo')
        self.assertEqual(idx2.name, 'foo')


    def test_index_init_b(self) -> None:

        idx1 = IndexHierarchy.from_product(['A', 'B'], [1, 2])

        idx2 = Index(idx1)

        self.assertEqual(idx2.values.tolist(),
            [('A', 1), ('A', 2), ('B', 1), ('B', 2)])


    def test_index_init_c(self) -> None:

        s1 = Series(('a', 'b', 'c'))
        idx2 = Index(s1)
        self.assertEqual(idx2.values.tolist(),
                ['a', 'b', 'c']
                )

    def test_index_init_d(self) -> None:
        idx = Index((0, '1', 2))
        self.assertEqual(idx.values.tolist(),
                [0, '1', 2]
                )

    def test_index_init_e(self) -> None:
        labels = [0.0, 36028797018963969]
        idx = Index(labels)
        # cannot extract the value once converted to float
        self.assertEqual(idx.loc[idx.values[1]], 36028797018963969)


    def test_index_loc_to_iloc_a(self) -> None:

        idx = Index(('a', 'b', 'c', 'd'))

        self.assertEqual(
                tp.cast(np.ndarray, idx.loc_to_iloc(np.array([True, False, True, False]))).tolist(),
                [0, 2])

        self.assertEqual(idx.loc_to_iloc(slice('c',)), slice(None, 3, None))  # type: ignore
        self.assertEqual(idx.loc_to_iloc(slice('b','d')), slice(1, 4, None))  # type: ignore
        self.assertEqual(idx.loc_to_iloc('d'), 3)


    def test_index_mloc_a(self) -> None:
        idx = Index(('a', 'b', 'c', 'd'))
        self.assertTrue(idx.mloc == idx[:2].mloc)

    def test_index_mloc_b(self) -> None:
        idx = IndexGO(('a', 'b', 'c', 'd'))
        idx.append('e')
        self.assertTrue(idx.mloc == idx[:2].mloc)


    def test_index_dtype_a(self) -> None:
        idx = IndexGO(('a', 'b', 'c', 'd'))
        self.assertEqual(str(idx.dtype), '<U1')
        idx.append('eee')
        self.assertEqual(str(idx.dtype), '<U3')


    def test_index_shape_a(self) -> None:
        idx = IndexGO(('a', 'b', 'c', 'd'))
        self.assertEqual(idx.shape, (4,))
        idx.append('e')
        self.assertEqual(idx.shape, (5,))


    def test_index_ndim_a(self) -> None:
        idx = IndexGO(('a', 'b', 'c', 'd'))
        self.assertEqual(idx.ndim, 1)
        idx.append('e')
        self.assertEqual(idx.ndim, 1)


    def test_index_size_a(self) -> None:
        idx = IndexGO(('a', 'b', 'c', 'd'))
        self.assertEqual(idx.size, 4)
        idx.append('e')
        self.assertEqual(idx.size, 5)

    def test_index_nbytes_a(self) -> None:
        idx = IndexGO(('a', 'b', 'c', 'd'))
        self.assertEqual(idx.nbytes, 16)
        idx.append('e')
        self.assertEqual(idx.nbytes, 20)


    def test_index_rename_a(self) -> None:
        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='foo')
        idx1.append('e')
        idx2 = idx1.rename('bar')
        self.assertEqual(idx2.name, 'bar')

    def test_index_positions_a(self) -> None:
        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='foo')
        self.assertEqual(idx1.positions.tolist(), list(range(4)))

        idx1.append('e')
        self.assertEqual(idx1.positions.tolist(), list(range(5)))


    def test_index_unique(self) -> None:

        with self.assertRaises(ErrorInitIndex):
            idx = Index(('a', 'b', 'c', 'a'))
        with self.assertRaises(ErrorInitIndex):
            idx = IndexGO(('a', 'b', 'c', 'a'))

        with self.assertRaises(ErrorInitIndex):
            idx = Index(['a', 'a'])
        with self.assertRaises(ErrorInitIndex):
            idx = IndexGO(['a', 'a'])

        with self.assertRaises(ErrorInitIndex):
            idx = Index(np.array([True, False, True], dtype=bool))
        with self.assertRaises(ErrorInitIndex):
            idx = IndexGO(np.array([True, False, True], dtype=bool))

        # acceptable but not advisiable
        idx = Index([0, '0'])


    def test_index_creation_a(self) -> None:
        idx = Index(('a', 'b', 'c', 'd'))

        #idx2 = idx['b':'d']

        self.assertEqual(idx.values.tolist(), ['a', 'b', 'c', 'd'])

        self.assertEqual(idx[2:].values.tolist(), ['c', 'd'])

        self.assertEqual(idx.loc['b':].values.tolist(), ['b', 'c', 'd'])  # type: ignore  # https://github.com/python/typeshed/pull/3024

        self.assertEqual(idx.loc['b':'d'].values.tolist(), ['b', 'c', 'd'])  # type: ignore  # https://github.com/python/typeshed/pull/3024

        self.assertEqual(idx.loc_to_iloc(['b', 'b', 'c']), [1, 1, 2])

        self.assertEqual(idx.loc['c'], 'c')

        idxgo = IndexGO(('a', 'b', 'c', 'd'))
        self.assertEqual(idxgo.values.tolist(), ['a', 'b', 'c', 'd'])

        idxgo.append('e')
        self.assertEqual(idxgo.values.tolist(), ['a', 'b', 'c', 'd', 'e'])

        idxgo.extend(('f', 'g'))
        self.assertEqual(idxgo.values.tolist(), ['a', 'b', 'c', 'd', 'e', 'f', 'g'])


    def test_index_creation_b(self) -> None:
        idx = Index((x for x in ('a', 'b', 'c', 'd') if x in {'b', 'd'}))
        self.assertEqual(idx.loc_to_iloc('b'), 0)
        self.assertEqual(idx.loc_to_iloc('d'), 1)


    def test_index_unary_operators_a(self) -> None:
        idx = Index((20, 30, 40, 50))

        invert_idx = -idx
        self.assertEqual(invert_idx.tolist(),
                [-20, -30, -40, -50],)

        # this is strange but consistent with NP
        not_idx = ~idx
        self.assertEqual(not_idx.tolist(),
                [-21, -31, -41, -51],)

    def test_index_binary_operators_a(self) -> None:
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


    def test_index_binary_operators_b(self) -> None:
        '''Both operands are Index instances
        '''
        idx1 = Index((20, 30, 40, 50))
        idx2 = Index((20, 3, 4, 5))

        self.assertEqual((idx1 == idx2).tolist(), [True, False, False, False])

    def test_index_binary_operators_c(self) -> None:
        idx1 = Index((20, 30, 40, 50))
        idx2 = Index((20, 3, 4, 5))
        self.assertEqual(idx1 @ idx2, idx1.values @ idx2.values)


    def test_index_ufunc_axis_a(self) -> None:

        idx = Index((30, 40, 50))

        self.assertEqual(idx.min(), 30)
        self.assertEqual(idx.max(), 50)
        self.assertEqual(idx.sum(), 120)

    def test_index_isin_a(self) -> None:

        idx = Index((30, 40, 50))

        self.assertEqual(idx.isin([40, 50]).tolist(), [False, True, True])
        self.assertEqual(idx.isin({40, 50}).tolist(), [False, True, True])

        self.assertEqual(idx.isin(frozenset((40, 50))).tolist(), [False, True, True])

        self.assertEqual(idx.isin({40: 'a', 50: 'b'}).tolist(), [False, True, True])

        self.assertEqual(idx.isin(range(35, 45)).tolist(), [False, True, False])

        self.assertEqual(idx.isin((x * 10 for x in (3, 4, 5, 6, 6))).tolist(), [True, True, True])



    def test_index_contains_a(self) -> None:

        index = Index(('a', 'b', 'c'))
        self.assertTrue('a' in index)
        self.assertTrue('d' not in index)


    def test_index_grow_only_a(self) -> None:

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

        index3 = index[2:]
        index3.append('i')

        self.assertEqual(index3.values.tolist(), ['c', 'd', 'e', 'f', 'h', 'i'])
        self.assertEqual(index.values.tolist(), ['a', 'b', 'c', 'd', 'e', 'f', 'h'])



    def test_index_sort_a(self) -> None:

        index = Index(('a', 'c', 'd', 'e', 'b'))
        self.assertEqual(
                [index.sort().loc_to_iloc(x) for x in sorted(index.values)],
                [0, 1, 2, 3, 4])
        self.assertEqual(
                [index.sort(ascending=False).loc_to_iloc(x) for x in sorted(index.values)],
                [4, 3, 2, 1, 0])


    def test_index_reversed(self) -> None:

        labels = tuple('acdeb')
        index = Index(labels=labels)
        index_reversed_generator = reversed(index)
        self.assertEqual(
            tuple(index_reversed_generator),
            tuple(reversed(labels))
        )

    def test_index_relable(self) -> None:

        index = Index(('a', 'c', 'd', 'e', 'b'))

        self.assertEqual(
                index.relabel(lambda x: x.upper()).values.tolist(),
                ['A', 'C', 'D', 'E', 'B'])

        self.assertEqual(
                index.relabel(lambda x: 'pre_' + x.upper()).values.tolist(),
                ['pre_A', 'pre_C', 'pre_D', 'pre_E', 'pre_B'])


        # letter to number
        s1 = Series(range(5), index=index.values)

        self.assertEqual(
                index.relabel(s1).values.tolist(),
                [0, 1, 2, 3, 4]
                )

        self.assertEqual(index.relabel({'e': 'E'}).values.tolist(),
                ['a', 'c', 'd', 'E', 'b'])




    def test_index_tuples_a(self) -> None:

        index = Index([('a','b'), ('b','c'), ('c','d')])
        s1 = Series(range(3), index=index)

        self.assertEqual(s1[('b', 'c'):].values.tolist(), [1, 2])  # type: ignore  # https://github.com/python/typeshed/pull/3024

        self.assertEqual(s1[[('b', 'c'), ('a', 'b')]].values.tolist(), [1, 0])

        self.assertEqual(s1[('b', 'c')], 1)
        self.assertEqual(s1[('c', 'd')], 2)

        s2 = Series(range(10), index=((1, x) for x in range(10)))
        self.assertEqual(s2[(1, 5):].values.tolist(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
                [5, 6, 7, 8, 9])

        self.assertEqual(s2[[(1, 7), (1, 5), (1, 0)]].values.tolist(),
                [7, 5, 0])



    def test_requires_reindex(self) -> None:
        a = Index([1, 2, 3])
        b = Index([1, 2, 3])
        c = Index([1, 3, 2])
        d = Index([1, 2, 3, 4])
        e = Index(['a', 2, 3])

        self.assertFalse(_requires_reindex(a, b))
        self.assertTrue(_requires_reindex(a, c))
        self.assertTrue(_requires_reindex(a, c))
        self.assertTrue(_requires_reindex(a, d))
        self.assertTrue(_requires_reindex(a, e))

    def test_index_pickle_a(self) -> None:
        a = Index([('a','b'), ('b','c'), ('c','d')])
        b = Index([1, 2, 3, 4])
        c = IndexYear.from_date_range('2014-12-15', '2018-03-15')

        for index in (a, b, c):
            pbytes = pickle.dumps(index)
            index_new = pickle.loads(pbytes)
            for v in index: # iter labels
                # import ipdb; ipdb.set_trace()
                # this compares Index objects
                self.assertFalse(index_new._labels.flags.writeable)
                self.assertEqual(index_new.loc[v], index.loc[v])

    def test_index_drop_a(self) -> None:

        index = Index(list('abcdefg'))

        self.assertEqual(index._drop_loc('d').values.tolist(),
                ['a', 'b', 'c', 'e', 'f', 'g'])

        self.assertEqual(index._drop_loc(['a', 'g']).values.tolist(),
                ['b', 'c', 'd', 'e', 'f'])

        self.assertEqual(index._drop_loc(slice('b', None)).values.tolist(),  # type: ignore
                ['a'])


    def test_index_iloc_loc_to_iloc(self) -> None:

        idx = Index(('a', 'b', 'c', 'd'))

        self.assertEqual(idx.loc_to_iloc(ILoc[1]), 1)
        self.assertEqual(idx.loc_to_iloc(ILoc[[0, 2]]), [0, 2])


    def test_index_loc_to_iloc_boolen_a(self) -> None:

        idx = Index(('a', 'b', 'c', 'd'))

        # unlike Pandas, both of these presently fail
        with self.assertRaises(KeyError):
            idx.loc_to_iloc([False, True])

        with self.assertRaises(KeyError):
            idx.loc_to_iloc([False, True, False, True])

        # but a Boolean array works
        post = idx.loc_to_iloc(np.array([False, True, False, True]))
        assert isinstance(post, np.ndarray)
        self.assertEqual(post.tolist(), [1, 3])


    def test_index_loc_to_iloc_boolen_b(self) -> None:

        idx = Index(('a', 'b', 'c', 'd'))

        # returns nothing as index does not match anything
        post = idx.loc_to_iloc(Series([False, True, False, True]))
        self.assertTrue(len(tp.cast(tp.Sized, post)) == 0)

        post = idx.loc_to_iloc(Series([False, True, False, True],
                index=('b', 'c', 'd', 'a')))
        assert isinstance(post, np.ndarray)
        self.assertEqual(post.tolist(), [0, 2])

        post = idx.loc_to_iloc(Series([False, True, False, True],
                index=list('abcd')))
        assert isinstance(post, np.ndarray)
        self.assertEqual(post.tolist(), [1,3])


    def test_index_drop_b(self) -> None:

        idx = Index(('a', 'b', 'c', 'd'))

        self.assertEqual(idx.drop.iloc[2].values.tolist(), ['a', 'b', 'd'])
        self.assertEqual(idx.drop.iloc[2:].values.tolist(), ['a', 'b'])
        self.assertEqual(
                idx.drop.iloc[np.array([True, False, False, True])].values.tolist(),
                ['b', 'c'])

    def test_index_drop_c(self) -> None:

        idx = Index(('a', 'b', 'c', 'd'))

        self.assertEqual(idx.drop.loc['c'].values.tolist(), ['a', 'b', 'd'])
        self.assertEqual(idx.drop.loc['b':'c'].values.tolist(), ['a', 'd'])  # type: ignore  # https://github.com/python/typeshed/pull/3024

        self.assertEqual(
                idx.drop.loc[np.array([True, False, False, True])].values.tolist(),
                ['b', 'c']
                )

    def test_index_roll_a(self) -> None:

        idx = Index(('a', 'b', 'c', 'd'))

        self.assertEqual(idx.roll(-2).values.tolist(),
                ['c', 'd', 'a', 'b'])

        self.assertEqual(idx.roll(1).values.tolist(),
                ['d', 'a', 'b', 'c'])


    def test_index_attributes_a(self) -> None:
        idx = Index(('a', 'b', 'c', 'd'))

        self.assertEqual(idx.shape, (4,))
        self.assertEqual(idx.dtype.kind, 'U')
        self.assertEqual(idx.ndim, 1)
        self.assertEqual(idx.nbytes, 16)


    def test_index_name_a(self) -> None:

        idx1 = Index(('a', 'b', 'c', 'd'), name='foo')
        self.assertEqual(idx1.name, 'foo')
        self.assertEqual(idx1.names, ('foo',))

        idx2 = idx1.rename('bar')
        self.assertEqual(idx2.name, 'bar')
        self.assertEqual(idx2.names, ('bar',))

    def test_name_b(self) -> None:

        with self.assertRaises(TypeError):
            Index(('a', 'b', 'c', 'd'), name=['x', 'y'])

        with self.assertRaises(TypeError):
            Index(('a', 'b', 'c', 'd'), name={'x', 'y'})


    def test_index_name_c(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='foo')
        self.assertEqual(idx1.name, 'foo')

        idx2 = idx1.rename('bar')
        self.assertEqual(idx2.name, 'bar')

        idx1.append('e')
        idx2.append('x')

        self.assertEqual(idx1.values.tolist(),
                ['a', 'b', 'c', 'd', 'e'])

        self.assertEqual(idx2.values.tolist(),
                ['a', 'b', 'c', 'd', 'x'])


    def test_index_to_pandas_a(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='foo')
        pdidx = idx1.to_pandas()
        self.assertEqual(pdidx.name, idx1.name)
        self.assertTrue((pdidx.values == idx1.values).all())


    def test_index_to_pandas_b(self) -> None:
        import pandas  # type: ignore
        idx1 = IndexDate(('2018-01-01', '2018-06-01'), name='foo')
        pdidx = idx1.to_pandas()
        self.assertEqual(pdidx.name, idx1.name)
        self.assertTrue((pdidx.values == idx1.values).all())
        self.assertTrue(pdidx[1].__class__ == pandas.Timestamp)


    def test_index_from_pandas_a(self) -> None:
        import pandas

        pdidx = pandas.Index(list('abcd'))
        idx = Index.from_pandas(pdidx)
        self.assertEqual(idx.values.tolist(), ['a', 'b', 'c', 'd'])


    def test_index_from_pandas_b(self) -> None:
        import pandas

        pdidx = pandas.DatetimeIndex(('2018-01-01', '2018-06-01'), name='foo')
        idx = IndexDate.from_pandas(pdidx)
        self.assertEqual(idx.values.tolist(),
                [datetime.date(2018, 1, 1), datetime.date(2018, 6, 1)])


    def test_index_iter_label_a(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='foo')
        self.assertEqual(list(idx1.iter_label(0)), ['a', 'b', 'c', 'd'])

        post = idx1.iter_label(0).apply(lambda x: x.upper())
        self.assertEqual(post.to_pairs(),
                ((0, 'A'), (1, 'B'), (2, 'C'), (3, 'D')))


    def test_index_intersection_a(self) -> None:

        idx1 = Index(('a', 'b', 'c', 'd', 'e'))

        a1 = np.array(['c', 'dd', 'b', 'a'])

        idx2 = idx1.intersection(a1)

        self.assertEqual(idx2.values.tolist(),
                ['a', 'b', 'c'])


    def test_index_intersection_b(self) -> None:

        idx1 = Index(('c', 'b', 'a'))
        idx2 = Index(('c', 'b', 'a'))

        idx3 = idx1.intersection(idx2)
        self.assertEqual(idx3.values.tolist(),
                ['c', 'b', 'a']
                )


    # TODO; add test for hierarchical index


    def test_index_union_a(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd', 'e'))
        idx1.append('f')
        a1 = np.array(['c', 'dd', 'b', 'a'])

        idx2 = idx1.union(a1)

        self.assertEqual(idx2.values.tolist(),
                ['a', 'b', 'c', 'd', 'dd', 'e', 'f'])


    def test_index_union_b(self) -> None:

        idx1 = Index(('c', 'b', 'a'))
        idx2 = Index(('c', 'b', 'a'))

        idx3 = idx1.union(idx2)
        self.assertEqual(idx3.values.tolist(),
                ['c', 'b', 'a']
                )



    def test_index_to_html_a(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c'))
        self.assertEqual(idx1.to_html(),
                '<table border="1"><thead></thead><tbody><tr><td>a</td></tr><tr><td>b</td></tr><tr><td>c</td></tr></tbody></table>')

    def test_index_to_html_datatables_a(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c'))

        sio = StringIO()

        post = idx1.to_html_datatables(sio, show=False)

        self.assertEqual(post, None)
        self.assertTrue(len(sio.read()) > 1200)


    def test_index_empty_a(self) -> None:
        idx1 = Index(())
        idx2 = Index(iter(()))
        self.assertEqual(idx1.dtype, idx2.dtype)

    def test_index_cumprod_a(self) -> None:
        idx1 = IndexGO(range(1, 11, 2))

        # sum applies to the labels
        self.assertEqual(idx1.sum(), 25)

        self.assertEqual(
                idx1.cumprod().tolist(),
                [1, 3, 15, 105, 945]
                )


    def test_index_label_widths_at_depth(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd', 'e'))
        self.assertEqual(tuple(idx1.label_widths_at_depth(0)),
            (('a', 1), ('b', 1), ('c', 1), ('d', 1), ('e', 1))
            )


    def test_index_bool_a(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd', 'e'))
        self.assertTrue(bool(idx1))

        idx2 = IndexGO(())
        self.assertFalse(bool(idx2))


if __name__ == '__main__':
    unittest.main()
