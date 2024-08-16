from __future__ import annotations

import copy
import datetime
import pickle
import unittest
from hashlib import sha256
from io import StringIO

import numpy as np
import typing_extensions as tp
from arraykit import mloc

from static_frame import DisplayConfig
from static_frame import Frame
from static_frame import ILoc
from static_frame import Index
from static_frame import IndexAutoFactory
from static_frame import IndexDate
from static_frame import IndexDateGO
from static_frame import IndexGO
from static_frame import IndexHierarchy
from static_frame import IndexYear
from static_frame import Series
from static_frame.core.exception import ErrorInitIndex
from static_frame.core.exception import ErrorInitIndexNonUnique
from static_frame.core.exception import LocInvalid
from static_frame.core.index import _index_initializer_needs_init
from static_frame.core.util import NULL_SLICE
from static_frame.core.util import PositionsAllocator
from static_frame.core.util import arrays_equal
from static_frame.test.test_case import TestCase

if tp.TYPE_CHECKING:
    TNDArrayAny = np.ndarray[tp.Any, tp.Any] #pragma: no cover
    TDtypeAny = np.dtype[tp.Any] #pragma: no cover

class TestUnit(TestCase):

    def test_iloc_repr(self) -> None:
        self.assertEqual(repr(ILoc[1]), '<ILoc[1]>')
        self.assertEqual(repr(ILoc[1,]), '<ILoc[1]>')
        self.assertEqual(repr(ILoc[1, 2]), '<ILoc[1,2]>')
        self.assertEqual(repr(ILoc[:, 1]), '<ILoc[:,1]>')
        self.assertEqual(repr(ILoc[:, 1, 2]), '<ILoc[:,1,2]>')
        self.assertEqual(repr(ILoc[1:2, :]), '<ILoc[1:2,:]>')
        self.assertEqual(repr(ILoc[1:2, :2]), '<ILoc[1:2,:2]>')
        self.assertEqual(repr(ILoc[1:2, :2, :3]), '<ILoc[1:2,:2,:3]>')
        self.assertEqual(repr(ILoc[::1]), '<ILoc[:]>')
        self.assertEqual(repr(ILoc[1:, 1:2, 1:2:3, :2, :2:3, ::3]), '<ILoc[1:,1:2,1:2:3,:2,:2:3,::3]>')
        # self.assertEqual(repr(ILoc[()]), '<ILoc[()]>')
        self.assertEqual(repr(ILoc[(1,),]), '<ILoc[(1,)]>')
        self.assertEqual(repr(ILoc[(),]), '<ILoc[()]>')
        self.assertEqual(repr(ILoc[:]), '<ILoc[:]>')
        self.assertEqual(repr(ILoc[:, :]), '<ILoc[:,:]>')
        self.assertEqual(repr(ILoc[:, :, 4]), '<ILoc[:,:,4]>')

    def test_positions_allocator_a(self) -> None:

        a1 = PositionsAllocator.get(3)
        a2 = PositionsAllocator.get(4)
        a3 = PositionsAllocator.get(5)

        # we get different object IDs, but point to the same data
        self.assertTrue(mloc(a1) == mloc(a2))
        self.assertTrue(mloc(a3) == mloc(a2))

    def test_index_slotted_a(self) -> None:
        idx1 = Index(('a', 'b', 'c', 'd'), name='foo')

        with self.assertRaises(AttributeError):
            idx1.g = 30 #type: ignore #pylint: disable=E0237
        with self.assertRaises(AttributeError):
            idx1.__dict__ #pylint: disable=W0104

    #---------------------------------------------------------------------------

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

    def test_index_init_f(self) -> None:

        labels = np.arange(3)
        mapping = {x:x for x in range(3)}

        with self.assertRaises(RuntimeError):
            _ = Index._extract_labels(
                    mapping=mapping, #type: ignore
                    labels=labels,
                    dtype=float
                    )

    def test_index_init_g(self) -> None:
        index = Index(Frame(np.arange(6).reshape((2, 3))))
        self.assertEqual(
                index.values.tolist(),
                [(0, 1, 2), (3, 4, 5)]
                )

    def test_index_init_h(self) -> None:
        index = Index(range(10, 20, 2))
        self.assertEqual(index.values.tolist(), list(range(10, 20, 2)))

    def test_index_init_i(self) -> None:
        i1 = Index([10, 20, 30], name='foo')
        i2 = Index(i1)
        self.assertEqual(i2.name, 'foo')

    def test_index_init_j(self) -> None:
        from itertools import chain

        with self.assertRaises(ErrorInitIndexNonUnique):
            idx1 = Index(list(chain(range(100), range(50, 200))))

        with self.assertRaises(ErrorInitIndexNonUnique):
            idx1 = Index(list(chain(range(100), range(50, 200))))

    def test_index_init_k(self) -> None:
        with self.assertRaises(ErrorInitIndexNonUnique):
            _ = Index((x for x in (3, 5, 3)))

    def test_index_init_l(self) -> None:
        with self.assertRaises(ErrorInitIndex):
            _ = Index('bar')

    def test_index_init_m(self) -> None:
        with self.assertRaises(ErrorInitIndex):
            _ = Index(np.array(('2021-02', '2022-04'), dtype=np.datetime64))

    def test_index_init_n(self) -> None:
        a1 = np.array((np.nan, np.nan, 3), dtype=np.float64)
        a2 = a1.astype(np.float16)
        a2.flags.writeable = False
        idx = Index(a2)
        self.assertEqual(idx.loc_to_iloc(3), 2)

    def test_index_init_o(self) -> None:
        a1 = np.array((10, 40, 20))
        idx = Index(a1) # permit a mutable array
        self.assertEqual(idx.values.tolist(), [10, 40, 20])


    #---------------------------------------------------------------------------

    def test_index_loc_to_iloc_a(self) -> None:

        idx = Index(('a', 'b', 'c', 'd'))

        self.assertEqual(
                idx._loc_to_iloc(np.array([True, False, True, False])).tolist(), # type: ignore
                [0, 2])

        self.assertEqual(idx._loc_to_iloc(slice('c',)), slice(None, 3, None))
        self.assertEqual(idx._loc_to_iloc(slice('b','d')), slice(1, 4, None))
        self.assertEqual(idx._loc_to_iloc('d'), 3)

    def test_index_loc_to_iloc_b(self) -> None:
        idx = Index(('a', 'b', 'c', 'd'))
        post = idx._loc_to_iloc(Series(['b', 'c']))
        self.assertEqual(post.tolist(), [1, 2]) #type: ignore

    def test_index_loc_to_iloc_c(self) -> None:
        idx = Index(('a', 'b', 'c', 'd'))
        with self.assertRaises(KeyError):
            _ = idx._loc_to_iloc(['c', 'd', 'e'])

        post = idx._loc_to_iloc(['c', 'd', 'e'], partial_selection=True)
        self.assertEqual(post, [2, 3])

    def test_index_loc_to_iloc_d(self) -> None:
        # testing the public interface
        idx1 = Index(('a', 'b', 'c', 'd'))

        self.assertEqual(idx1.loc_to_iloc('b'), 1)
        with self.assertRaises(KeyError):
            _ = idx1.loc_to_iloc('g')

        self.assertEqual(idx1.loc_to_iloc(slice('b', 'd')), slice(1, 4, None))
        with self.assertRaises(LocInvalid):
            _ = idx1.loc_to_iloc(slice('x', 'y'))

        self.assertEqual(idx1.loc_to_iloc(['d', 'a']).tolist(), [3, 0]) #type: ignore
        with self.assertRaises(KeyError):
            _ = idx1.loc_to_iloc(['d', 'x'])

        self.assertEqual(idx1.loc_to_iloc(np.array([False, True, True, False])).tolist(), [1, 2]) #type: ignore [union-attr]
        with self.assertRaises(IndexError):
            _ = idx1.loc_to_iloc(np.array([False, True, False]))

    def test_index_loc_to_iloc_e(self) -> None:

        idx2 = Index(range(4), loc_is_iloc=True)

        self.assertEqual(idx2.loc_to_iloc(1), 1)
        with self.assertRaises(KeyError):
            _ = idx2.loc_to_iloc(5)

        self.assertEqual(idx2.loc_to_iloc(slice(1, 3)), slice(1, 4))

        with self.assertRaises(LocInvalid):
            _ = idx2.loc_to_iloc(slice('x', 'y'))

        with self.assertRaises(LocInvalid):
            # loc slices are always interpreted as inclusive, so going beyond the inclusive boundary is an error
            _ = idx2.loc_to_iloc(slice(0, 4))

        self.assertEqual(idx2.loc_to_iloc([3, 0]).tolist(), [3, 0]) #type: ignore
        with self.assertRaises(KeyError):
            _ = idx2.loc_to_iloc([3, 20])

        self.assertEqual(idx2.loc_to_iloc(np.array([False, True, True, False])).tolist(), [1, 2]) #type: ignore [union-attr]
        with self.assertRaises(IndexError):
            _ = idx2.loc_to_iloc(np.array([False, True, False]))

    def test_index_loc_to_iloc_f(self) -> None:
        dt = datetime.date
        dt64 = np.datetime64

        idx1 = Index((
                dt(2020,12,31),
                dt(2021,1,15),
                dt(2021,1,31),
                ))

        self.assertEqual(
                idx1.loc_to_iloc(dt64('2021-01-15')),
                1
                )
        # NOTE: this fails as we only see a list of dt64s and cannot match them in the AutoMap dictionary unless we were to directly examine and conert each element
        with self.assertRaises(KeyError):
            _ = idx1.loc_to_iloc([dt64(d) for d in reversed(idx1)]) # type: ignore

        post = idx1.loc_to_iloc(np.array([dt64(d) for d in reversed(idx1)])) # type: ignore
        self.assertEqual(post.tolist(), [2, 1, 0]) #type: ignore

    def test_index_loc_to_iloc_g(self) -> None:
        dt = datetime.date
        dt64 = np.datetime64

        idx1 = IndexYear(('2021', '2018', '2001'))

        with self.assertRaises(KeyError):
            idx1.loc_to_iloc(dt64('2001-01-01'))

        with self.assertRaises(KeyError):
            idx1.loc_to_iloc(np.array((dt64('2001-01-01'), dt64('2018-01-01'))))

    def test_index_loc_to_iloc_h(self) -> None:
        dt = datetime.date
        dt64 = np.datetime64

        idx1 = IndexDate(('2021-01-01', '2021-01-02', '1543-08-31', '1988-05-01', '1988-05-02'))

        self.assertEqual(idx1.loc_to_iloc(dt64('2021-01')).tolist(), [0, 1]) #type: ignore

        self.assertEqual(
                idx1.loc_to_iloc(np.array((dt64('2021'), dt64('1988')))).tolist(), #type: ignore
                [0, 1, 3, 4]
                )

    def test_index_loc_to_iloc_i(self) -> None:
        idx1 = Index(range(4), loc_is_iloc=True)
        self.assertTrue(idx1._map is None)

        idx2 = idx1[:]
        self.assertTrue(idx2._map is None)

    def test_index_loc_to_iloc_j(self) -> None:
        idx1 = IndexAutoFactory.from_optional_constructor(10,
                default_constructor=Index)
        post = idx1.loc_to_iloc(NULL_SLICE)
        self.assertEqual(post, NULL_SLICE)

    def test_index_loc_to_iloc_k(self) -> None:
        idx1 = Index(range(4), loc_is_iloc=True)
        self.assertTrue(idx1._map is None)
        # for now, lists of Bools only work on indicies without maps
        post = idx1.loc[[True, False, True, False]]
        self.assertEqual(post.values.tolist(), [0, 2]) #type: ignore

    def test_index_loc_to_iloc_l(self) -> None:
        idx1 = Index(range(4), loc_is_iloc=True)
        self.assertTrue(idx1._map is None)

        post1 = idx1[Series((3, 1), index=('a', 'b'))]
        post2 = idx1.loc_to_iloc(Series((3, 1), index=('a', 'b')))
        self.assertEqual(post1.tolist(), post2.tolist()) #type: ignore

    def test_index_loc_to_iloc_m(self) -> None:
        idx1 = IndexGO(range(4), loc_is_iloc=True)
        idx1.append(4)
        self.assertTrue(idx1._map is None)
        post1 = idx1.loc_to_iloc([3, 0])
        self.assertEqual(post1.tolist(), [3, 0]) #type: ignore

    #---------------------------------------------------------------------------

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

    #---------------------------------------------------------------------------

    def test_index_rename_a(self) -> None:
        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='foo')
        idx1.append('e')
        idx2 = idx1.rename('bar')
        self.assertEqual(idx2.name, 'bar')

    def test_index_rename_b(self) -> None:
        a = Index([1], name='foo')
        self.assertEqual(a.name, 'foo')
        b = a.rename(None)
        self.assertEqual(b.name, None)

    #---------------------------------------------------------------------------

    def test_index_positions_a(self) -> None:
        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='foo')
        self.assertEqual(idx1.positions.tolist(), list(range(4)))

        idx1.append('e')
        self.assertEqual(idx1.positions.tolist(), list(range(5)))

    def test_index_unique_a(self) -> None:

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

    def test_index_unique_b(self) -> None:
        idx = Index(('a', 'b', 'c', 'd'))
        self.assertEqual(idx.unique().tolist(), idx.values.tolist())

    def test_index_creation_a(self) -> None:
        idx = Index(('a', 'b', 'c', 'd'))

        self.assertEqual(idx.values.tolist(), ['a', 'b', 'c', 'd'])

        self.assertEqual(idx[2:].values.tolist(), ['c', 'd'])

        self.assertEqual(idx.loc['b':].values.tolist(), ['b', 'c', 'd'])  # type: ignore  # https://github.com/python/typeshed/pull/3024

        self.assertEqual(idx.loc['b':'d'].values.tolist(), ['b', 'c', 'd'])  # type: ignore  # https://github.com/python/typeshed/pull/3024

        self.assertEqual(idx._loc_to_iloc(['b', 'b', 'c']).tolist(), [1, 1, 2]) #type: ignore

        self.assertEqual(idx.loc['c'], 'c')

        idxgo = IndexGO(('a', 'b', 'c', 'd'))
        self.assertEqual(idxgo.values.tolist(), ['a', 'b', 'c', 'd'])

        idxgo.append('e')
        self.assertEqual(idxgo.values.tolist(), ['a', 'b', 'c', 'd', 'e'])

        idxgo.extend(('f', 'g'))
        self.assertEqual(idxgo.values.tolist(), ['a', 'b', 'c', 'd', 'e', 'f', 'g'])

    def test_index_creation_b(self) -> None:
        idx = Index((x for x in ('a', 'b', 'c', 'd') if x in {'b', 'd'}))
        self.assertEqual(idx._loc_to_iloc('b'), 0)
        self.assertEqual(idx._loc_to_iloc('d'), 1)

    #---------------------------------------------------------------------------

    def test_index_unary_operators_a(self) -> None:
        idx = Index((20, 30, 40, 50))

        invert_idx = -idx
        self.assertEqual(invert_idx.tolist(),
                [-20, -30, -40, -50],)

        # this is strange but consistent with NP
        not_idx = ~idx
        self.assertEqual(not_idx.tolist(),
                [-21, -31, -41, -51],)

    def test_index_unary_operators_b(self) -> None:
        idx = IndexGO((20, 30, 40))
        idx.append(50)
        a1 = -idx
        self.assertEqual(a1.tolist(), [-20, -30, -40, -50])

    def test_index_unary_operators_c(self) -> None:
        idx1 = Index((-10, -30))
        a1 = +idx1
        self.assertEqual(a1.tolist(), [-10, -30])


    def test_index_unary_operators_d(self) -> None:
        idx1 = Index((-10, -30))
        a1 = abs(idx1)
        self.assertEqual(a1.tolist(), [10, 30])



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

    def test_index_binary_operators_d(self) -> None:
        idx = IndexGO((20, 30, 40))
        idx.extend((50, 60))
        a1 = idx * 2
        self.assertEqual(a1.tolist(), [40, 60, 80, 100, 120])

    def test_index_binary_operators_e(self) -> None:
        idx1 = Index((20, 30, 40, 50))
        idx2 = Index((20, 3, 4, 5))
        self.assertEqual(idx1.values @ idx2, idx1.values @ idx2.values)
        self.assertEqual(idx1.values.tolist() @ idx2, idx1.values @ idx2.values)

    def test_index_binary_operators_f(self) -> None:
        idx1 = Index(('a', 'b', 'c'))

        self.assertEqual((idx1 + '_').tolist(), ['a_', 'b_', 'c_'])
        self.assertEqual(('_' + idx1).tolist(), ['_a', '_b', '_c'])
        self.assertEqual((idx1 * 3).tolist(), ['aaa', 'bbb', 'ccc'])

    def test_index_binary_operators_g(self) -> None:
        idx1 = Index((1, 2, 3))
        s1 = Series(('a', 'b', 'c'))
        with self.assertRaises(ValueError):
            _ = idx1 * s1

    #---------------------------------------------------------------------------

    def test_index_ufunc_axis_a(self) -> None:

        idx = Index((30, 40, 50))
        self.assertEqual(idx.min(), 30)
        self.assertEqual(idx.max(), 50)
        self.assertEqual(idx.sum(), 120)

    def test_index_ufunc_axis_b(self) -> None:

        idx = IndexGO((30, 40, 20))
        idx.append(10)
        self.assertEqual(idx.sum(), 100)

    def test_index_ufunc_axis_c(self) -> None:

        idx = Index((30, 40, np.nan))
        self.assertEqual(idx.sum(), 70)

    def test_index_ufunc_axis_d(self) -> None:

        idx = Index((np.nan,))
        self.assertEqual(idx.sum(), 0.0)
        self.assertEqual(idx.sum(allna=-1), -1)


    def test_index_isin_a(self) -> None:

        idx = Index((30, 40, 50))

        self.assertEqual(idx.isin([40, 50]).tolist(), [False, True, True])
        self.assertEqual(idx.isin({40, 50}).tolist(), [False, True, True])

        self.assertEqual(idx.isin(frozenset((40, 50))).tolist(), [False, True, True])

        self.assertEqual(idx.isin({40: 'a', 50: 'b'}).tolist(), [False, True, True])

        self.assertEqual(idx.isin(range(35, 45)).tolist(), [False, True, False])

        self.assertEqual(idx.isin((x * 10 for x in (3, 4, 5, 6, 6))).tolist(), [True, True, True])

    def test_index_isin_b(self) -> None:
        idx = Index(('a', 'b', 'c'))
        self.assertEqual(
                idx.isin(('b','c')).tolist(),
                [False, True, True]
                )

        self.assertEqual(
                idx.isin(('b', 'c', 'b', 'c')).tolist(),
                [False, True, True]
                )

    #---------------------------------------------------------------------------

    def test_index_copy_a(self) -> None:
        idx1 = IndexGO(('a', 'b', 'c'))
        idx1.append('d')
        idx2 = idx1.copy()
        idx2.append('e')
        self.assertEqual(idx2.values.tolist(), ['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(idx1.values.tolist(), ['a', 'b', 'c', 'd'])

    #---------------------------------------------------------------------------

    def test_index_contains_a(self) -> None:

        index = Index(('a', 'b', 'c'))
        self.assertTrue('a' in index)
        self.assertTrue('d' not in index)

    #---------------------------------------------------------------------------

    def test_index_go_a(self) -> None:

        index = IndexGO(('a', 'b', 'c'))
        index.append('d')
        self.assertEqual(index._loc_to_iloc('d'), 3)

        index.extend(('e', 'f'))
        self.assertEqual(index._loc_to_iloc('e'), 4)
        self.assertEqual(index._loc_to_iloc('f'), 5)

        # creating an index form an Index go takes the np arrays, but not the mutable bits
        index2 = Index(index)
        index.append('h')

        self.assertEqual(len(index2), 6)
        self.assertEqual(len(index), 7)

        index3 = index[2:]
        index3.append('i')

        self.assertEqual(index3.values.tolist(), ['c', 'd', 'e', 'f', 'h', 'i'])
        self.assertEqual(index.values.tolist(), ['a', 'b', 'c', 'd', 'e', 'f', 'h'])

    def test_index_go_b(self) -> None:

        index = IndexGO(('a', 'b', 'c'))
        index.append('d')
        self.assertEqual(len(index.__slots__), 3)
        self.assertFalse(index.STATIC)
        self.assertEqual(index._IMMUTABLE_CONSTRUCTOR, Index)
        self.assertEqual(Index._MUTABLE_CONSTRUCTOR, IndexGO)

    def test_index_go_c(self) -> None:

        index = IndexGO(('a', (2, 5), 'c'))
        with self.assertRaises(KeyError):
            index.append((2, 5))

    def test_index_go_d(self) -> None:

        index = IndexGO((), loc_is_iloc=True)
        index.append(0)
        self.assertTrue(index._map is None)

        index.append(1)
        self.assertTrue(1 in index)
        self.assertFalse('a' in index)
        self.assertTrue(index._map is None)

        index.append('a')
        self.assertFalse(index._map is None)
        self.assertTrue('a' in index)
        self.assertTrue(1 in index)

    def test_index_go_e(self) -> None:

        index = IndexGO((), loc_is_iloc=True)
        index.append(0)
        self.assertTrue(index._map is None)

        index.append(1)
        self.assertTrue(1 in index)
        self.assertFalse('a' in index)
        self.assertTrue(index._map is None)

        index.append(-1)
        self.assertFalse(index._map is None)
        self.assertTrue(-1 in index)
        self.assertTrue(1 in index)

    def test_index_go_f(self) -> None:

        idx1 = IndexAutoFactory.from_optional_constructor(3,
                default_constructor=IndexGO)
        idx1.append(3) # type: ignore
        post = idx1._loc_to_iloc(np.array([True, False, True, False]))
        self.assertEqual(post.tolist(), [True, False, True, False]) #type: ignore

    #---------------------------------------------------------------------------

    def test_index_sort_a(self) -> None:

        index = Index(('a', 'c', 'd', 'e', 'b'))
        self.assertEqual(
                [index.sort()._loc_to_iloc(x) for x in sorted(index.values)],
                [0, 1, 2, 3, 4])
        self.assertEqual(
                [index.sort(ascending=False)._loc_to_iloc(x) for x in sorted(index.values)],
                [4, 3, 2, 1, 0])

    def test_index_sort_b(self) -> None:

        index = Index(('ax', 'cb', 'dg', 'eb', 'bq'))

        self.assertEqual(index.sort(
                key=lambda i: i.iter_label().apply(lambda x: x[1])
                ).values.tolist(),
                ['cb', 'eb', 'dg', 'bq', 'ax']
                )

    def test_index_sort_c(self) -> None:
        index = Index(('a', 'c', 'd', 'e', 'b'))
        with self.assertRaises(RuntimeError):
            index.sort(ascending=(True, False))

    #---------------------------------------------------------------------------

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

    def test_index_pickle_a(self) -> None:
        a = Index([('a','b'), ('b','c'), ('c','d')])
        b = Index([1, 2, 3, 4])
        c = IndexYear.from_date_range('2014-12-15', '2018-03-15')

        for index in (a, b, c):
            pbytes = pickle.dumps(index)
            index_new = pickle.loads(pbytes)
            for v in index: # iter labels
                # this compares Index objects
                self.assertFalse(index_new._labels.flags.writeable)
                self.assertEqual(index_new.loc[v], index.loc[v])

    def test_index_drop_a1(self) -> None:

        index = Index(list('abcdefg'))

        self.assertEqual(index._drop_loc('d').values.tolist(),
                ['a', 'b', 'c', 'e', 'f', 'g'])

        self.assertEqual(index._drop_loc(slice('b', None)).values.tolist(),
                ['a'])

    def test_index_drop_a2(self) -> None:

        index = Index(list('abcdefg'))

        self.assertEqual(index._drop_loc(['a', 'g']).values.tolist(),
                ['b', 'c', 'd', 'e', 'f'])

    #---------------------------------------------------------------------------

    def test_index_iloc_loc_to_iloc_a(self) -> None:

        idx = Index(('a', 'b', 'c', 'd'))

        self.assertEqual(idx._loc_to_iloc(ILoc[1]), 1)
        self.assertEqual(idx._loc_to_iloc(ILoc[[0, 2]]), [0, 2])

    def test_index_extract_iloc_a(self) -> None:

        idx = IndexGO(('a', 'b', 'c', 'd'))
        idx.append('e')
        post = idx._extract_iloc(slice(None))
        self.assertEqual(post.values.tolist(),
                ['a', 'b', 'c', 'd', 'e'])

    def test_index_drop_iloc_a(self) -> None:

        idx = IndexGO(('a', 'b', 'c', 'd'))
        idx.append('e')
        post = idx._drop_iloc([1, 2])
        self.assertEqual(post.values.tolist(),
                ['a', 'd', 'e'])

    def test_index_drop_iloc_b(self) -> None:

        idx = IndexGO(('a', 'b', 'c', 'd'))
        idx.append('e')
        post = idx._drop_iloc(None)
        self.assertEqual(post.values.tolist(),
                ['a', 'b', 'c', 'd', 'e'])

    #---------------------------------------------------------------------------

    def test_index_loc_to_iloc_boolen_a(self) -> None:

        idx = Index(('a', 'b', 'c', 'd'))

        # unlike Pandas, both of these presently fail
        with self.assertRaises(KeyError):
            idx._loc_to_iloc([False, True])

        with self.assertRaises(KeyError):
            idx._loc_to_iloc([False, True, False, True])

        # but a Boolean array works
        post = idx._loc_to_iloc(np.array([False, True, False, True]))
        assert isinstance(post, np.ndarray)
        self.assertEqual(post.tolist(), [1, 3])

    def test_index_loc_to_iloc_boolen_b(self) -> None:

        idx = Index(('a', 'b', 'c', 'd'))

        # returns nothing as index does not match anything
        post = idx._loc_to_iloc(Series([False, True, False, True]))
        self.assertTrue(len(tp.cast(tp.Sized, post)) == 0)

        post = idx._loc_to_iloc(Series([False, True, False, True],
                index=('b', 'c', 'd', 'a')))
        assert isinstance(post, np.ndarray)
        self.assertEqual(post.tolist(), [0, 2])

        post = idx._loc_to_iloc(Series([False, True, False, True],
                index=list('abcd')))
        assert isinstance(post, np.ndarray)
        self.assertEqual(post.tolist(), [1,3])

    def test_index_drop_b(self) -> None:

        idx = Index(('a', 'b', 'c', 'd'))

        self.assertEqual(idx.drop.iloc[2].values.tolist(), ['a', 'b', 'd']) #type: ignore
        self.assertEqual(idx.drop.iloc[2:].values.tolist(), ['a', 'b']) #type: ignore
        self.assertEqual(
                idx.drop.iloc[np.array([True, False, False, True])].values.tolist(), #type: ignore
                ['b', 'c'])

    def test_index_drop_c(self) -> None:

        idx = Index(('a', 'b', 'c', 'd'))

        self.assertEqual(idx.drop.loc['c'].values.tolist(), ['a', 'b', 'd']) #type: ignore
        self.assertEqual(idx.drop.loc['b':'c'].values.tolist(), ['a', 'd'])  # type: ignore  # https://github.com/python/typeshed/pull/3024

        self.assertEqual(
                idx.drop.loc[np.array([True, False, False, True])].values.tolist(), #type: ignore
                ['b', 'c']
                )

    def test_index_roll_a(self) -> None:

        idx = Index(('a', 'b', 'c', 'd'))

        self.assertEqual(idx.roll(-2).values.tolist(),
                ['c', 'd', 'a', 'b'])

        self.assertEqual(idx.roll(1).values.tolist(),
                ['d', 'a', 'b', 'c'])

    #---------------------------------------------------------------------------
    def test_index_fillna_a(self) -> None:

        idx1 = Index(('a', 'b', 'c', None))
        idx2 = idx1.fillna('d')
        self.assertEqual(idx2.values.tolist(),
                ['a', 'b', 'c', 'd'])

        idx3 = Index((10, 20, np.nan))
        idx4 = idx3.fillna(30.1)
        self.assertEqual(idx4.values.tolist(),
                [10, 20, 30.1])

    def test_index_fillna_b(self) -> None:

        idx1 = Index(('a', 'b', 'c'))
        idx2 = idx1.fillna('d')
        self.assertEqual(id(idx1), id(idx2))

        idx3 = IndexGO(('a', 'b', 'c'))
        idx4 = idx3.fillna('d')
        self.assertNotEqual(id(idx3), id(idx4))

    def test_index_fillna_c(self) -> None:

        idx1 = Index((3.0, 2.0, np.nan))
        idx2 = idx1.fillna('foo')
        self.assertEqual(idx2.values.tolist(),
                [3.0, 2.0, 'foo'],
                )

    #---------------------------------------------------------------------------
    def test_index_fillfalsy_a(self) -> None:

        idx1 = Index(('a', 'b', 'c', ''))
        idx2 = idx1.fillfalsy('d')
        self.assertEqual(idx2.values.tolist(),
                ['a', 'b', 'c', 'd'])

    #---------------------------------------------------------------------------

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

    #---------------------------------------------------------------------------

    def test_index_to_series_a(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='foo')
        s1 = idx1.to_series()
        self.assertFalse(s1.values.flags.writeable)
        self.assertEqual(s1.to_pairs(),
                ((0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'))
                )

    #---------------------------------------------------------------------------

    def test_index_to_pandas_a(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='foo')
        pdidx = idx1.to_pandas()
        self.assertEqual(pdidx.name, idx1.name)
        self.assertTrue((pdidx.values == idx1.values).all())

    def test_index_to_pandas_b(self) -> None:
        import pandas
        idx1 = IndexDate(('2018-01-01', '2018-06-01'), name='foo')
        pdidx = idx1.to_pandas()
        self.assertEqual(pdidx.name, idx1.name)
        self.assertTrue((pdidx.values == idx1.values).all())
        self.assertTrue(pdidx[1].__class__ == pandas.Timestamp)

    #---------------------------------------------------------------------------

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

    def test_index_from_pandas_c(self) -> None:
        import pandas

        pdidx = pandas.DatetimeIndex(('2018-01-01', '2018-06-01'), name='foo')
        idx = IndexDateGO.from_pandas(pdidx)
        self.assertFalse(idx.STATIC)
        self.assertEqual(idx.values.tolist(),
                [datetime.date(2018, 1, 1), datetime.date(2018, 6, 1)]
                )

    def test_index_from_pandas_d(self) -> None:
        import pandas
        pdidx = pandas.DatetimeIndex(('2018-01-01', '2018-06-01'), name='foo')
        idx = Index.from_pandas(pdidx)
        self.assertEqual(
                idx.values.tolist(),
                [1514764800000000000, 1527811200000000000]
                )

    def test_index_from_pandas_e(self) -> None:
        import pandas
        idx = pandas.DatetimeIndex([datetime.date(2014, 12, i) for i in range(1, 10)])
        index1 = Index.from_pandas(idx)
        self.assertTrue(index1.STATIC)
        self.assertEqual(index1.values.tolist(),
                [1417392000000000000, 1417478400000000000, 1417564800000000000, 1417651200000000000, 1417737600000000000, 1417824000000000000, 1417910400000000000, 1417996800000000000, 1418083200000000000]
                )
        index2 = IndexGO.from_pandas(idx)
        self.assertFalse(index2.STATIC)
        self.assertEqual(index2.values.tolist(),
                [1417392000000000000, 1417478400000000000, 1417564800000000000, 1417651200000000000, 1417737600000000000, 1417824000000000000, 1417910400000000000, 1417996800000000000, 1418083200000000000]
                )

    def test_index_from_pandas_f(self) -> None:
        with self.assertRaises(ErrorInitIndex):
            idx = Index.from_pandas(Index(('a', 'b')))

    #---------------------------------------------------------------------------

    def test_index_iter_a(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c'), name='foo')
        idx1.append('d')
        self.assertEqual(list(idx1), ['a', 'b', 'c', 'd'])

    #---------------------------------------------------------------------------

    def test_index_reversed_a(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c'), name='foo')
        idx1.append('d')
        self.assertEqual(list(reversed(idx1)), ['d', 'c', 'b', 'a'])

    def test_index_reversed_b(self) -> None:

        labels = tuple('acdeb')
        index = Index(labels=labels)
        index_reversed_generator = reversed(index)
        self.assertEqual(
            tuple(index_reversed_generator),
            tuple(reversed(labels)))

    #---------------------------------------------------------------------------

    def test_index_iter_label_a(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='foo')
        self.assertEqual(list(idx1.iter_label(0)), ['a', 'b', 'c', 'd'])

        post = idx1.iter_label(0).apply(lambda x: x.upper())
        self.assertEqual(post.tolist(), ['A', 'B', 'C', 'D'])

    def test_index_iter_label_b(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='foo')
        post = tuple(idx1.iter_label().apply_iter_items(
            lambda x: x.upper()
            ))
        self.assertEqual(post,
            ((0, 'A'), (1, 'B'), (2, 'C'), (3, 'D')),
            )

    def test_index_iter_label_c(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='foo')
        with self.assertRaises(RuntimeError):
            _ = tuple(idx1.iter_label().apply(
                    lambda x: x.upper(),
                    index_constructor=IndexDate,
                    ))

    #---------------------------------------------------------------------------

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

    def test_index_intersection_c(self) -> None:
        idx1 = Index((10, 20))
        with self.assertRaises(RuntimeError):
            # raises as it identifies labelled data
            _ = idx1.intersection(Series([20, 30]))

        idx2 = idx1.intersection(Series([20, 30]).values)
        self.assertEqual(idx2.values.tolist(), [20])

        idx3 = idx1.intersection([20, 30])
        self.assertEqual(idx3.values.tolist(), [20])

    def test_index_intersection_d(self) -> None:
        idx1 = Index((10, 20))
        idx2 = idx1.intersection('b')
        self.assertEqual(len(idx2), 0)

    def test_index_intersection_e(self) -> None:

        idx1 = Index((10, 'foo', None, 4.1))
        idx2 = idx1.union(idx1)
        self.assertEqual(id(idx1), id(idx2))
        self.assertTrue(idx1.equals(idx1))

        idx3 = idx1.intersection(idx1)
        self.assertEqual(id(idx1), id(idx3))
        self.assertTrue(idx1.equals(idx3))

        idx4 = idx1.difference(idx1)
        self.assertEqual(len(idx4), 0)
        self.assertEqual(idx4.dtype, np.dtype(object))

    def test_index_intersection_f(self) -> None:

        idx1 = IndexDate.from_date_range('2020-01-02', '2020-01-08')
        idx2 = IndexDate.from_date_range('2020-01-02', '2020-01-08')
        idx3 = IndexDate.from_date_range('2020-01-03', '2020-01-09')

        self.assertEqual(idx1.union(idx2).values.tolist(),
                [datetime.date(2020, 1, 2), datetime.date(2020, 1, 3), datetime.date(2020, 1, 4), datetime.date(2020, 1, 5), datetime.date(2020, 1, 6), datetime.date(2020, 1, 7), datetime.date(2020, 1, 8)]
                )
        self.assertEqual(id(idx1), id(idx1.union(idx2)))

        self.assertEqual(idx1.intersection(idx2).values.tolist(),
                [datetime.date(2020, 1, 2), datetime.date(2020, 1, 3), datetime.date(2020, 1, 4), datetime.date(2020, 1, 5), datetime.date(2020, 1, 6), datetime.date(2020, 1, 7), datetime.date(2020, 1, 8)]
                )

        self.assertEqual(id(idx1), id(idx1.intersection(idx2)))

        self.assertEqual(idx1.difference(idx2).values.tolist(), [])

        self.assertEqual(idx1.intersection(idx3).values.tolist(),
                [datetime.date(2020, 1, 3), datetime.date(2020, 1, 4), datetime.date(2020, 1, 5), datetime.date(2020, 1, 6), datetime.date(2020, 1, 7), datetime.date(2020, 1, 8)]
                )

    def test_index_intersection_g(self) -> None:

        idx1 = Index(('c', 'b', 'a'))
        idx2 = Index(('b', 'a', 'r'))
        idx3 = Index(('w', 'b', 'x'))

        idx4 = idx1.intersection(idx2, idx3)
        self.assertEqual(idx4.values.tolist(), ['b'])

    #---------------------------------------------------------------------------

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

    def test_index_union_c(self) -> None:

        idx1 = Index(('c', 'b', 'a'))
        idx2 = Index(('r', 'b', 'x'))
        idx3 = Index(('t', 's'))
        idx4 = Index(('t', 'q'))

        idx5 = idx1.union(idx2, idx3, idx4)
        self.assertEqual(idx5.values.tolist(),
                ['a', 'b', 'c', 'q', 'r', 's', 't', 'x']
                )

    @unittest.skip('pending resolution of AutoMap issue')
    def test_index_union_d(self) -> None:
        # with self.assertRaises(RuntimeError):
        idx = Index.from_labels(np.array((np.nan, np.nan)))
        self.assertEqual(len(idx), 1)

    #---------------------------------------------------------------------------

    def test_index_difference_a(self) -> None:
        idx1 = Index(('c', 'b', 'a'))
        idx2 = Index(('c', 'b', 'a'))

        idx3 = idx1.difference(idx2)
        self.assertEqual(idx3.values.tolist(), [])

    def test_index_difference_b(self) -> None:
        idx1 = Index(())
        idx2 = Index(('c', 'b', 'a'))

        idx3 = idx1.difference(idx2)
        self.assertEqual(idx3.values.tolist(), [])

        idx4 = Index(('c', 'b', 'a'))
        idx5 = Index(())

        idx6 = idx4.difference(idx5)
        self.assertEqual(idx6.values.tolist(),
                ['c', 'b', 'a']
                )

    def test_index_difference_c(self) -> None:
        obj = object()
        idx1 = Index((1, None, '3', np.nan, 4.4, obj))
        idx2 = Index((2, 3, '4', 'five', 6.6, object()))

        idx3 = idx1.difference(idx2)
        self.assertEqual(set(idx3.values.tolist()),
                set([np.nan, 1, 4.4, obj, '3', None])
                ) # Note: order is lost...

    def test_index_difference_d(self) -> None:
        obj = object()
        idx1 = Index((1, None, '3', np.nan, 4.4, obj))
        idx2 = Index((2, 1, '3', 'five', object()))

        idx3 = idx1.difference(idx2)
        self.assertEqual(set(idx3.values.tolist()),
                set([np.nan, None, 4.4, obj])
                ) # Note: order is lost...

    def test_index_to_html_a(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c'))

        self.assertEqual(idx1.to_html(),
                '<table style="border-collapse:collapse;border-width:1px;border-color:#898b8e;border-style:solid"><tbody><tr><td style="background-color:#ffffff;font-weight:normal;padding:2px;font-size:14px;border-width:1px;border-color:#898b8e;border-style:solid;color:#2b2a2a">a</td></tr><tr><td style="background-color:#f2f2f2;font-weight:normal;padding:2px;font-size:14px;border-width:1px;border-color:#898b8e;border-style:solid;color:#2b2a2a">b</td></tr><tr><td style="background-color:#ffffff;font-weight:normal;padding:2px;font-size:14px;border-width:1px;border-color:#898b8e;border-style:solid;color:#2b2a2a">c</td></tr></tbody></table>'
                )
        self.assertEqual(idx1.to_html(style_config=None),
                '<table><tbody><tr><td>a</td></tr><tr><td>b</td></tr><tr><td>c</td></tr></tbody></table>')

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

    #---------------------------------------------------------------------------

    def test_index_label_widths_at_depth_a(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd', 'e'))
        self.assertEqual(tuple(idx1.label_widths_at_depth(0)),
            (('a', 1), ('b', 1), ('c', 1), ('d', 1), ('e', 1))
            )

    def test_index_label_widths_at_depth_b(self) -> None:
        idx1 = IndexGO(('a', 'b', 'c', 'd', 'e'))
        with self.assertRaises(RuntimeError):
            next(idx1.label_widths_at_depth(1))

    #---------------------------------------------------------------------------

    def test_index_bool_a(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd', 'e'))
        with self.assertRaises(ValueError):
            bool(idx1)

    def test_index_bool_b(self) -> None:

        idx1 = IndexGO(())
        with self.assertRaises(ValueError):
            bool(idx1)

    def test_index_astype_a(self) -> None:

        idx1 = Index((3, 10, 50))
        self.assertEqual(idx1.astype(float).values.dtype, np.dtype(float))

    #---------------------------------------------------------------------------

    def test_index_initializer_needs_init(self) -> None:
        self.assertEqual(_index_initializer_needs_init(None), False)
        self.assertEqual(_index_initializer_needs_init(Index((1, 2))), False)

        self.assertEqual(_index_initializer_needs_init(np.arange(0)), False)
        self.assertEqual(_index_initializer_needs_init(np.arange(3)), True)

        self.assertEqual(_index_initializer_needs_init(()), False)
        self.assertEqual(_index_initializer_needs_init((3, 5)), True)

    #---------------------------------------------------------------------------

    def test_index_values_at_depth_a(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c'))
        self.assertEqual(idx1.values_at_depth(0).tolist(),
                ['a', 'b', 'c'])
        with self.assertRaises(RuntimeError):
            idx1.values_at_depth(1)

    def test_index_values_at_depth_b(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd', 'e'))

        with self.assertRaises(RuntimeError):
            idx1.values_at_depth([3, 4])

        post = idx1.values_at_depth([0])

        self.assertEqual(post.tolist(),
                ['a', 'b', 'c', 'd', 'e'])

    #---------------------------------------------------------------------------

    def test_index_head_a(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd', 'e'))
        self.assertEqual(idx1.head(2).values.tolist(), ['a' ,'b'])

    def test_index_tail_a(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd', 'e'))
        self.assertEqual(idx1.tail(2).values.tolist(), ['d' ,'e'])

    def test_index_via_str_a(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd', 'e'))
        a1 = idx1.via_str.upper()

    def test_index_via_str_b(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd', 'e'))

        idx1.append('f')
        idx1.append('g')

        a1 = idx1.via_str.upper()
        self.assertEqual(a1.tolist(),
                ['A', 'B', 'C', 'D', 'E', 'F', 'G']
                )

    def test_index_via_dt_a(self) -> None:

        idx1 = IndexDate(('2020-01-01', '2021-02-05', '2019-03-17'))

        self.assertEqual(idx1.via_dt.day.tolist(),
                [1, 5, 17]
                )

        self.assertEqual(idx1.via_dt.weekday().tolist(),
                [2, 4, 6]
                )

    def test_index_via_dt_b(self) -> None:

        idx1 = IndexDateGO(('2020-01-01', '2021-02-05', '2019-03-17'))
        idx1.append('2020-04-01')
        idx1.append('2020-05-01')

        self.assertEqual(idx1.via_dt.weekday().tolist(),
                [2, 4, 6, 2, 4]
                )


    def test_index_via_values_a(self) -> None:

        idx1 = IndexGO((10, 20, 30))
        idx1.append(40)
        idx2 = idx1.via_values.apply(lambda x: (x * .5).astype(int))
        self.assertEqual(idx2.__class__, IndexGO)
        self.assertEqual(idx2.values.tolist(), [5, 10, 15, 20])

    #---------------------------------------------------------------------------

    def test_index_equals_a(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd', 'e'))
        idx2 = Index(('a', 'b', 'c', 'd', 'e'))
        idx3 = IndexGO(('a', 'b', 'c', 'd', 'e'), name='foo')
        idx4 = IndexGO(('a', 'b', 'c', 'd', 'e'))
        idx5 = IndexGO(('a', 'b', 'c', 'd'))

        self.assertEqual(idx1.equals(3), False)
        self.assertEqual(idx1.equals(False), False)
        self.assertEqual(idx1.equals([3, 4, 5]), False)

        self.assertEqual(idx1.equals(idx2, compare_class=True), False)
        self.assertEqual(idx1.equals(idx2, compare_class=False), True)

        self.assertEqual(idx1.equals(idx3, compare_name=True), False)
        self.assertEqual(idx1.equals(idx3, compare_name=False), True)
        self.assertEqual(idx1.equals(idx5), False)

        self.assertEqual(idx1.equals(idx1), True)
        self.assertEqual(idx1.equals(idx4), True)

    def test_index_equals_b(self) -> None:

        idx1 = Index((5, 3, 20), dtype=np.int64)
        idx2 = Index((5, 3, 20), dtype=np.int32)

        self.assertFalse(idx1.equals(idx2, compare_dtype=True))
        self.assertTrue(idx1.equals(idx2, compare_dtype=False))

    def test_index_equals_c(self) -> None:

        idx1 = IndexDate.from_year_range('2010', '2011')
        idx2 = Index(idx1.values.astype(object))

        self.assertFalse(idx1.equals(idx2, compare_class=True))
        self.assertTrue(idx1.equals(idx2, compare_class=False),)

    def test_index_equals_d(self) -> None:

        idx1 = Index((5, 3, np.nan))
        idx2 = Index((5, 3, np.nan))
        idx3 = Index((5, 3, None))

        self.assertTrue(idx1.equals(idx2))
        self.assertFalse(idx1.equals(idx2, skipna=False))

        # nan and None are not treated equivalent, even with skipna true
        self.assertFalse(idx1.equals(idx3, compare_dtype=False, skipna=True))

    def test_index_equals_e(self) -> None:
        a = Index([1, 2, 3])
        b = Index([1, 2, 3])
        c = Index([1, 3, 2])
        d = Index([1, 2, 3, 4])
        e = Index(['a', 2, 3])

        self.assertFalse(not a.equals(b))
        self.assertTrue(not a.equals(c))
        self.assertTrue(not a.equals(c))
        self.assertTrue(not a.equals(d))
        self.assertTrue(not a.equals(e))

    def test_index_equals_f(self) -> None:
        a = IndexGO([1, 2, 3])
        b = IndexGO([1, 2, 3])
        b.append(4)
        self.assertFalse(a.equals(b))

    def test_index_equals_g(self) -> None:
        dt64 = np.datetime64

        a = IndexDate([dt64('2021-01-01'), dt64('1954-01-01')])
        b = IndexYear([dt64('2021'), dt64('1954')])

        self.assertFalse(arrays_equal(a.values, b.values, skipna=True))

    def test_index_equals_h(self) -> None:
        a = IndexGO([1, 2, 3])
        a.append(4)
        b = IndexGO([1, 2, 3])
        self.assertFalse(a.equals(b))

    #---------------------------------------------------------------------------

    def test_index_sample_a(self) -> None:
        a = IndexGO([1, 2, 3])
        a.append(None)
        b = a.sample(2, seed=3)

        self.assertEqual(b.values.tolist(), [2, None])

    #---------------------------------------------------------------------------

    def test_index_deepcopy_a(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd'))
        idx1.append('e')

        idx2 = copy.deepcopy(idx1)
        idx1.append('f')

        self.assertEqual(idx2.values.tolist(), ['a', 'b', 'c', 'd', 'e'])
        self.assertTrue(id(idx1._labels) != id(idx2._labels))

    def test_index_deepcopy_b(self) -> None:
        idx1 = IndexGO(range(5), loc_is_iloc=True)

        idx2 = copy.deepcopy(idx1)

        self.assertEqual(idx2.values.tolist(), [0, 1, 2, 3, 4])
        self.assertTrue(id(idx1._labels) != id(idx2._labels))
        self.assertTrue(idx2._map is None)

    def test_index_deepcopy_c(self) -> None:
        idx1 = IndexGO(range(3))
        idx1.append(100)
        idx2 = copy.deepcopy(idx1)
        idx1.append(200)
        self.assertEqual(idx2.values.tolist(), [0, 1, 2, 100])

    #---------------------------------------------------------------------------

    def test_index_iloc_searchsorted_a(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd'))
        self.assertEqual(idx1.iloc_searchsorted('c'), 2)
        self.assertEqual(idx1.iloc_searchsorted('c', side_left=False), 3)
        self.assertEqual(idx1.iloc_searchsorted(('a', 'c'), side_left=False).tolist(), [1, 3])

    def test_index_loc_searchsorted_b(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd', 'e'))
        self.assertEqual(idx1.loc_searchsorted('c'), 'c')
        self.assertEqual(idx1.loc_searchsorted('c', side_left=False), 'd')
        self.assertEqual(idx1.loc_searchsorted(('a', 'c'), side_left=False).tolist(), # type: ignore
                ['b', 'd'])

        self.assertEqual(idx1.loc_searchsorted(
                ('a', 'e'), side_left=False, fill_value=None).tolist(), # type: ignore
                ['b', None])

    def test_index_iloc_searchsorted_c(self) -> None:
        idx1 = Index(())
        self.assertEqual(idx1.loc_searchsorted(0, fill_value=None), None)

    #---------------------------------------------------------------------------

    def test_index_via_re_a(self) -> None:

        idx1 = IndexGO(('aabbcc', 'bbcccc', 'cc', 'ddddbb'))
        idx1.append('q')
        a1 = idx1.via_re('bb').search()

        self.assertEqual(a1.tolist(),
                [True, True, False, True, False])

    #---------------------------------------------------------------------------

    def test_index_index_types_a(self) -> None:

        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='foo')
        self.assertEqual(idx1.index_types.to_pairs(), (('foo', IndexGO),))

    #---------------------------------------------------------------------------

    def test_index_level_add_a(self) -> None:
        idx1 = Index(('a', 'b', 'c'), name='foo')
        idx2 = idx1.level_add('2012', index_constructor=IndexYear)
        self.assertEqual(idx2.name, 'foo')
        self.assertEqual(idx1.index_types.values.tolist(), [Index])
        self.assertEqual(idx2.index_types.values.tolist(), [IndexYear, Index])

        self.assertTrue(
            (idx2.values_at_depth(0) == np.array(['2012', '2012', '2012'], dtype='datetime64[Y]')).all()
            )

    #---------------------------------------------------------------------------

    def test_index_iloc_map_a(self) -> None:

        ih1 = Index(tuple('QafDeHbdc'))
        ih2 = Index(tuple('ABcdHebDEsaQf'))

        post = ih1._index_iloc_map(ih2)
        expected= ih1.iter_label().apply(ih2._loc_to_iloc)
        self.assertEqual(post.tolist(), expected.tolist())
        self.assertEqual([11, 10, 12, 7, 5, 4, 6, 3, 2], post.tolist())

    def test_index_iloc_map_b(self) -> None:

        ih1 = Index(tuple('abcdefg'))
        ih2 = Index(tuple('bcdefg'))

        with self.assertRaises(KeyError):
            ih1._index_iloc_map(ih2)

        with self.assertRaises(KeyError):
            ih1.iter_label().apply(ih2._loc_to_iloc)

    #---------------------------------------------------------------------------
    def test_index_extract_iloc_by_int(self) -> None:
        idx = IndexGO(('a', 'b'))
        idx.append('c')
        post = idx._extract_iloc_by_int(2)
        self.assertEqual(post, 'c')

    #---------------------------------------------------------------------------
    def test_index_dropna_a(self) -> None:
        idx1 = Index((3.5, np.nan, 1.5))
        idx2 = idx1.dropna()
        self.assertEqual(idx2.values.tolist(), [3.5, 1.5])

    def test_index_dropna_b(self) -> None:
        idx1 = Index((3.5, None, 1.5))
        idx2 = idx1.dropna()
        self.assertEqual(idx2.values.tolist(), [3.5, 1.5])

    def test_index_dropna_c(self) -> None:
        idx1 = IndexDate(('2020-03', None, '1981-05-30'))
        idx2 = idx1.dropna()
        self.assertEqual(idx2.values.tolist(), [datetime.date(2020, 3, 1), datetime.date(1981, 5, 30)])

    def test_index_dropna_d(self) -> None:
        idx1 = Index((None, np.nan))
        idx2 = idx1.dropna()
        self.assertEqual(idx2.values.tolist(), [])

    def test_index_dropna_e(self) -> None:
        idx1 = Index((3, 4))
        idx2 = idx1.dropna()
        self.assertEqual(idx2.values.tolist(), [3, 4])
        self.assertIs(idx2, idx1)

    def test_index_dropna_f(self) -> None:
        idx1 = IndexGO((3, 4))
        idx2 = idx1.dropna()
        self.assertEqual(idx2.values.tolist(), [3, 4])
        self.assertIsNot(idx2, idx1)

    def test_index_dropna_g(self) -> None:
        idx1 = IndexDate(('2020-03', '1981-05-30'))
        idx2 = idx1.dropna()
        self.assertEqual(idx2.values.tolist(), [datetime.date(2020, 3, 1), datetime.date(1981, 5, 30)])
        self.assertIs(idx1, idx2)

    def test_index_dropna_h(self) -> None:
        idx1 = IndexDateGO(('2020-03', '1981-05-30'))
        idx2 = idx1.dropna()
        self.assertEqual(idx2.values.tolist(), [datetime.date(2020, 3, 1), datetime.date(1981, 5, 30)])
        self.assertIsNot(idx1, idx2)

    #---------------------------------------------------------------------------
    def test_index_dropfalsy_a(self) -> None:
        idx1 = Index((3.5, np.nan, 1.5))
        idx2 = idx1.dropfalsy()
        self.assertEqual(idx2.values.tolist(), [3.5, 1.5])

    def test_index_dropfalsy_b(self) -> None:
        idx1 = Index((0, '', None, 2))
        idx2 = idx1.dropfalsy()
        self.assertEqual(idx2.values.tolist(), [2])

    #---------------------------------------------------------------------------
    def test_index_display_a(self) -> None:
        idx = IndexGO(('a', 'b', 'c', 'd'))
        idx.append('e')
        post = idx.display(DisplayConfig(type_show=False, type_color=False))
        self.assertEqual(str(post), 'a\nb\nc\nd\ne')

    #---------------------------------------------------------------------------
    def test_index_hash_bytes_a(self) -> None:
        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='')
        bytes1 = idx1._to_signature_bytes()
        self.assertEqual(sha256(bytes1).hexdigest(),
            'c767ec91c4609de269307eb178d169503f5ae91f2e690cfc11a83c78b6687b1e')

        idx2 = Index(('a', 'b', 'c', 'd'), name='')
        bytes2 = idx2._to_signature_bytes()
        self.assertEqual(sha256(bytes2).hexdigest(),
            '108f99787a5b8c8acc45ebfcc934ad1a1eaedda394679c192d3b4be385590d93')

    def test_index_hash_bytes_b(self) -> None:
        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='')
        bytes1 = idx1._to_signature_bytes(include_class=False)

        idx2 = Index(('a', 'b', 'c', 'd'), name='')
        bytes2 = idx2._to_signature_bytes(include_class=False)

        self.assertEqual(
                sha256(bytes1).hexdigest(),
                sha256(bytes2).hexdigest(),
                )

    def test_index_hash_bytes_c(self) -> None:
        idx1 = IndexGO(('a', 'b', 'c', 'd'), name=None)
        with self.assertRaises(TypeError):
            _ = idx1._to_signature_bytes()

        bytes1 = idx1._to_signature_bytes(include_name=False)
        self.assertEqual(sha256(bytes1).hexdigest(),
            'c767ec91c4609de269307eb178d169503f5ae91f2e690cfc11a83c78b6687b1e')

    def test_index_hash_bytes_d(self) -> None:
        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='')
        bytes1 = idx1._to_signature_bytes(include_class=False)

        idx2 = Index(('a', 'b', 'c', 'd', 'e'), name='')
        bytes2 = idx2._to_signature_bytes(include_class=False)

        self.assertNotEqual(
                sha256(bytes1).hexdigest(),
                sha256(bytes2).hexdigest(),
                )

    def test_index_via_hashlib_a(self) -> None:
        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='')
        self.assertEqual(idx1.via_hashlib().sha256().hexdigest(),
            'c767ec91c4609de269307eb178d169503f5ae91f2e690cfc11a83c78b6687b1e')

    def test_index_via_hashlib_b(self) -> None:
        idx1 = IndexGO(('a', False, None))
        with self.assertRaises(TypeError):
            _ = idx1.via_hashlib().sha256().hexdigest()

    def test_index_get_argsort_cache_a(self) -> None:
        idx1 = Index(('a', 'b', 'c', 'd'), name='')
        idx2 = IndexGO(('a', 'b', 'c', 'd'), name='')

        unique1, indexers1 = idx1._get_argsort_cache()
        unique2, indexers2 = idx2._get_argsort_cache()

        # Force re-cache
        idx2.append("e")
        unique3, indexers3 = idx2._get_argsort_cache()

        assert (unique1 == unique2).all()
        assert (indexers1 == indexers2).all()

        assert unique1.size != unique3.size
        assert indexers1.size != indexers3.size

        assert unique1.size == indexers1.size
        assert unique3.size == indexers3.size

    def test_index_get_argsort_cache_b(self) -> None:
        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='')

        unique1, indexers1 = idx1._get_argsort_cache()
        assert (unique1 == idx1.values).all()

        idx2 = idx1.__deepcopy__({})

        # Force re-cache
        idx1.append("e")

        idx3 = idx1.__deepcopy__({})

        unique2, indexers2 = idx1._get_argsort_cache()
        unique3, indexers3 = idx2._get_argsort_cache()
        unique4, indexers4 = idx3._get_argsort_cache()

        assert (unique2 == idx1.values).all()
        assert (unique1 == idx2.values).all()
        assert (unique2 == idx3.values).all()

        assert (unique1 == unique3).all()
        assert (unique2 == unique4).all()

        assert (indexers1 == indexers3).all()
        assert (indexers2 == indexers4).all()

        assert len(unique1) == len(unique3) == len(indexers1) == len(indexers3) == 4
        assert len(unique2) == len(unique4) == len(indexers2) == len(indexers4) == 5

    def test_index_get_argsort_cache_c(self) -> None:
        idx1 = Index(('a', 'b', 'c', 'd'), name='')
        assert idx1._argsort_cache is None

        idx2 = idx1.__deepcopy__({})
        assert idx2._argsort_cache is None

        unique1, indexers1 = idx1._get_argsort_cache()
        assert idx1._argsort_cache is not None
        idx3 = idx1.__deepcopy__({}) # type: ignore
        assert idx3._argsort_cache is not None

        unique2, indexers2 = idx2._get_argsort_cache()
        unique3, indexers3 = idx3._get_argsort_cache()

        assert len(unique1) == len(unique2) == len(indexers1) == len(indexers2) == len(unique3) == len(indexers3) == 4

        assert unique1 is not unique2
        assert unique1 is not unique3
        assert unique2 is not unique3
        assert indexers1 is not indexers2
        assert indexers1 is not indexers3
        assert indexers2 is not indexers3
        assert (unique1 == unique2).all()
        assert (unique2 == unique3).all()
        assert (indexers1 == indexers2).all()
        assert (indexers2 == indexers3).all()

    #---------------------------------------------------------------------------
    def test_index_isna_a(self) -> None:
        idx1 = Index(('a', 'b', 'c', 'd'), name='')
        self.assertEqual(idx1.isna().tolist(), [False, False, False, False])

        idx1 = Index(('a', None, 'c', 'd'))
        self.assertEqual(idx1.isna().tolist(), [False, True, False, False])

        idx1 = Index(('a', np.nan, 'c', 'd'))
        self.assertEqual(idx1.isna().tolist(), [False, True, False, False])

        idx1 = Index((3, np.nan, 1, 2))
        self.assertEqual(idx1.isna().tolist(), [False, True, False, False])

    def test_index_notna_a(self) -> None:
        idx1 = Index(('a', 'b', 'c', 'd'), name='')
        self.assertEqual(idx1.notna().tolist(), [True, True, True, True])

        idx1 = Index(('a', None, 'c', 'd'))
        self.assertEqual(idx1.notna().tolist(), [True, False, True, True])

        idx1 = Index(('a', np.nan, 'c', 'd'))
        self.assertEqual(idx1.notna().tolist(), [True, False, True, True])

        idx1 = Index((3, np.nan, 1, 2))
        self.assertEqual(idx1.notna().tolist(), [True, False, True, True])

    def test_index_isfalsy_a(self) -> None:
        idx1 = Index(('a', 'b', 'c', 'd'), name='')
        self.assertEqual(idx1.isfalsy().tolist(), [False, False, False, False])

        idx1 = Index(('a', None, 'c', ''))
        self.assertEqual(idx1.isfalsy().tolist(), [False, True, False, True])

        idx1 = Index(('a', np.nan, None, ''))
        self.assertEqual(idx1.isfalsy().tolist(), [False, True, True, True])

        idx1 = Index((3, np.nan, 1, 0))
        self.assertEqual(idx1.isfalsy().tolist(), [False, True, False, True])

    def test_index_notfalsy_a(self) -> None:
        idx1 = Index(('a', 'b', 'c', 'd'), name='')
        self.assertEqual(idx1.notfalsy().tolist(), [True, True, True, True])

        idx1 = Index(('a', None, 'c', ''))
        self.assertEqual(idx1.notfalsy().tolist(), [True, False, True, False])

        idx1 = Index(('a', np.nan, None, ''))
        self.assertEqual(idx1.notfalsy().tolist(), [True, False, False, False])

        idx1 = Index((3, np.nan, 1, 0))
        self.assertEqual(idx1.notfalsy().tolist(), [True, False, True, False])


if __name__ == '__main__':
    unittest.main()
