import unittest
from collections import OrderedDict
import itertools as it
from collections import namedtuple
from io import StringIO
import string
import pickle
import sqlite3
import datetime

import numpy as np
import typing as tp

import static_frame as sf

# assuming located in the same directory
from static_frame import Index
# from static_frame import IndexGO
from static_frame import IndexHierarchy
from static_frame import IndexHierarchyGO
from static_frame import IndexYearMonth
from static_frame import Series
from static_frame import Frame
from static_frame import FrameGO
from static_frame import TypeBlocks
# from static_frame import Display
from static_frame import mloc
from static_frame import ILoc
from static_frame import HLoc
from static_frame import DisplayConfig
from static_frame import IndexAutoFactory

from static_frame.core.store_xlsx import StoreXLSX
from static_frame.core.store import StoreConfig

from static_frame.test.test_case import TestCase
from static_frame.test.test_case import skip_win
from static_frame.test.test_case import temp_file
from static_frame.core.exception import ErrorInitFrame

nan = np.nan


class TestUnit(TestCase):

    def test_frame_slotted_a(self) -> None:

        f1 = Frame(1, index=(1,2), columns=(3,4,5))

        with self.assertRaises(AttributeError):
            f1.g = 30 # type: ignore #pylint: disable=E0237
        with self.assertRaises(AttributeError):
            f1.__dict__ #pylint: disable=W0104


    def test_frame_init_a(self) -> None:

        f = Frame.from_dict(OrderedDict([('a', (1,2)), ('b', (3,4))]), index=('x', 'y'))
        self.assertEqual(f.to_pairs(0),
                (('a', (('x', 1), ('y', 2))), ('b', (('x', 3), ('y', 4))))
                )

        f = Frame.from_dict(OrderedDict([('b', (3,4)), ('a', (1,2))]), index=('x', 'y'))
        self.assertEqual(f.to_pairs(0),
                (('b', (('x', 3), ('y', 4))), ('a', (('x', 1), ('y', 2)))))


    def test_frame_init_b(self) -> None:
        # test unusual instantiation cases

        # create a frame with a single value
        f1 = Frame(1, index=(1,2), columns=(3,4,5))
        self.assertEqual(f1.to_pairs(0),
                ((3, ((1, 1), (2, 1))), (4, ((1, 1), (2, 1))), (5, ((1, 1), (2, 1))))
                )

        # with columns not defined, we create a DF with just an index
        f2 = FrameGO(index=(1,2))
        f2['a'] = (-1, -1)
        self.assertEqual(f2.to_pairs(0),
                (('a', ((1, -1), (2, -1))),)
                )

        # with columns and index defined, we fill the value even if None
        f3 = Frame(None, index=(1,2), columns=(3,4,5))
        self.assertEqual(f3.to_pairs(0),
                ((3, ((1, None), (2, None))), (4, ((1, None), (2, None))), (5, ((1, None), (2, None)))))

        # auto populated index/columns based on shape
        f4 = Frame([[1,2], [3,4], [5,6]])
        self.assertEqual(f4.to_pairs(0),
                ((0, ((0, 1), (1, 3), (2, 5))), (1, ((0, 2), (1, 4), (2, 6))))
                )
        self.assertTrue(f4._index._loc_is_iloc)
        self.assertTrue(f4._columns._loc_is_iloc)


    def test_frame_init_c(self) -> None:
        f = sf.FrameGO.from_dict(dict(color=('black',)))
        s = f['color']
        self.assertEqual(s.to_pairs(),
                ((0, 'black'),))

    def test_frame_init_d(self) -> None:
        a1 = np.array([[1, 2, 3], [4, 5, 6]])

        f = sf.Frame(a1, own_data=True)
        self.assertEqual(mloc(a1), f.mloc[0])

    def test_frame_init_e(self) -> None:
        a1 = np.array([1, 2, 3])
        a2 = np.array([4, 5, 6])

        f = sf.Frame.from_dict(dict(a=a1, b=a2))

    def test_frame_init_f(self) -> None:
        a1 = np.array([1, 2, 3])
        a2 = np.array([4, 5, 6])

        f = sf.Frame.from_dict(dict(a=a1, b=a2))

        self.assertEqual(f.to_pairs(0),
            (('a', ((0, 1), (1, 2), (2, 3))), ('b', ((0, 4), (1, 5), (2, 6))))
            )

    def test_frame_init_g(self) -> None:

        f1 = sf.Frame(index=tuple('abc'))
        self.assertEqual(f1.shape, (3, 0))

        f2 = sf.Frame(columns=tuple('abc'))
        self.assertEqual(f2.shape, (0, 3))

        f3 = sf.Frame()
        self.assertEqual(f3.shape, (0, 0))

    def test_frame_init_h(self) -> None:

        f1 = sf.Frame(index=tuple('abc'), columns=())
        self.assertEqual(f1.shape, (3, 0))

        f2 = sf.Frame(columns=tuple('abc'), index=())
        self.assertEqual(f2.shape, (0, 3))

        f3 = sf.Frame(columns=(), index=())
        self.assertEqual(f3.shape, (0, 0))


    def test_frame_init_i(self) -> None:

        f1 = sf.FrameGO(index=tuple('abc'))
        f1['x'] = (3, 4, 5)
        f1['y'] = Series.from_dict(dict(b=10, c=11, a=12))

        self.assertEqual(f1.to_pairs(0),
            (('x', (('a', 3), ('b', 4), ('c', 5))), ('y', (('a', 12), ('b', 10), ('c', 11)))))

    def test_frame_init_j(self) -> None:
        f1 = sf.Frame('q', index=tuple('ab'), columns=tuple('xy'))
        self.assertEqual(f1.to_pairs(0),
            (('x', (('a', 'q'), ('b', 'q'))), ('y', (('a', 'q'), ('b', 'q'))))
            )

    def test_frame_init_k(self) -> None:
        # check that we got autoincrement indices if no col/index provided
        f1 = Frame([[0, 1], [2, 3]])
        self.assertEqual(f1.to_pairs(0), ((0, ((0, 0), (1, 2))), (1, ((0, 1), (1, 3)))))

    def test_frame_init_m(self) -> None:
        # cannot create a single element filled Frame specifying a shape (with index and columns) but not specifying a data value
        with self.assertRaises(RuntimeError):
            f1 = Frame(index=(3,4,5), columns=list('abv'))

    def test_frame_init_n(self) -> None:
        # cannot supply a single value to unfillabe sized Frame

        with self.assertRaises(RuntimeError):
            f1 = Frame(None, index=(3,4,5), columns=())

    def test_frame_init_o(self) -> None:
        f1 = Frame()
        self.assertEqual(f1.shape, (0, 0))


    def test_frame_init_p(self) -> None:

        # raise when a data values ir provided but an axis is size zero

        with self.assertRaises(RuntimeError):
            f1 = sf.Frame('x', index=(1,2,3), columns=iter(()))

        with self.assertRaises(RuntimeError):
            f1 = sf.Frame(None, index=(1,2,3), columns=iter(()))


    def test_frame_init_q(self) -> None:

        f1 = sf.Frame(index=(1,2,3), columns=iter(()))
        self.assertEqual(f1.shape, (3, 0))
        self.assertEqual(f1.to_pairs(0), ())


    def test_frame_init_r(self) -> None:

        f1 = sf.Frame(index=(), columns=iter(range(3)))

        self.assertEqual(f1.shape, (0, 3))
        self.assertEqual(f1.to_pairs(0),
                ((0, ()), (1, ()), (2, ())))

        with self.assertRaises(RuntimeError):
            # cannot create an unfillable array with a data value
            f1 = sf.Frame('x', index=(), columns=iter(range(3)))

    def test_frame_init_s(self) -> None:
        # check that we got autoincrement indices if no col/index provided
        f1 = Frame([[0, 1], [2, 3]],
                index=IndexAutoFactory,
                columns=IndexAutoFactory)

        self.assertEqual(f1.to_pairs(0),
                ((0, ((0, 0), (1, 2))), (1, ((0, 1), (1, 3))))
                )

        f2 = Frame([[0, 1], [2, 3]],
                index=IndexAutoFactory,
                columns=list('ab')
                )
        self.assertEqual(
                f2.to_pairs(0),
                (('a', ((0, 0), (1, 2))), ('b', ((0, 1), (1, 3))))
                )

    def test_frame_init_t(self) -> None:

        # 3d array raises exception
        a1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        with self.assertRaises(RuntimeError):
            f1 = Frame(a1)


    def test_frame_init_u(self) -> None:

        # 3d array raises exception
        a1 = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]

        with self.assertRaises(RuntimeError):
            f1 = Frame(a1)


    def test_frame_init_index_constructor_a(self) -> None:

        f1 = sf.Frame('q',
                index=[('a', 'b'), (1, 2)],
                columns=tuple('xy'),
                index_constructor=IndexHierarchy.from_labels
                )
        self.assertTrue(isinstance(f1.index, IndexHierarchy))
        self.assertEqual(f1.to_pairs(0),
                (('x', ((('a', 'b'), 'q'), ((1, 2), 'q'))), ('y', ((('a', 'b'), 'q'), ((1, 2), 'q'))))
                )

        with self.assertRaises(RuntimeError):
            f1 = sf.Frame('q',
                    index=[('a', 'b'), (1, 2)],
                    columns=tuple('xy'),
                    index_constructor=IndexHierarchyGO.from_labels
                    )


    def test_frame_init_columns_constructor_a(self) -> None:

        # using from_priduct is awkard, as it does not take a single iterable of products, but multiple args; we can get around this with a simple lambda
        f1 = sf.Frame('q',
                index=tuple('xy'),
                columns=[('a', 'b'), (1, 2)],
                columns_constructor=lambda args: IndexHierarchy.from_product(*args)
                )
        self.assertTrue(isinstance(f1.columns, IndexHierarchy))
        self.assertEqual(f1.to_pairs(0),
                ((('a', 1), (('x', 'q'), ('y', 'q'))), (('a', 2), (('x', 'q'), ('y', 'q'))), (('b', 1), (('x', 'q'), ('y', 'q'))), (('b', 2), (('x', 'q'), ('y', 'q'))))
                )

        with self.assertRaises(RuntimeError):
            f1 = sf.Frame('q',
                index=tuple('xy'),
                columns=[('a', 'b'), (1, 2)],
                columns_constructor=lambda args: IndexHierarchyGO.from_product(*args)
                )


    def test_frame_init_iter(self) -> None:

        f1 = Frame(None, index=iter(range(3)), columns=("A",))
        self.assertEqual(
            f1.to_pairs(0),
            (('A', ((0, None), (1, None), (2, None))),)
        )

        f2 = Frame(None, index=("A",), columns=iter(range(3)))
        self.assertEqual(
            f2.to_pairs(0),
            ((0, (('A', None),)), (1, (('A', None),)), (2, (('A', None),)))
        )

    def test_frame_values_a(self) -> None:
        f = sf.Frame([[3]])
        self.assertEqual(f.values.tolist(), [[3]])


    def test_frame_values_b(self) -> None:
        f = sf.Frame(np.array([[3, 2, 1]]))
        self.assertEqual(f.values.tolist(), [[3, 2, 1]])

    def test_frame_values_c(self) -> None:
        f = sf.Frame(np.array([[3], [2], [1]]))
        self.assertEqual(f.values.tolist(), [[3], [2], [1]])


    def test_frame_from_pairs_a(self) -> None:

        frame = Frame.from_items(sorted(dict(a=[3,4,5], b=[6,3,2]).items()))
        self.assertEqual(
            list((k, list(v.items())) for k, v in frame.items()),
            [('a', [(0, 3), (1, 4), (2, 5)]), ('b', [(0, 6), (1, 3), (2, 2)])])

        frame = Frame.from_items(OrderedDict((('b', [6,3,2]), ('a', [3,4,5]))).items())
        self.assertEqual(list((k, list(v.items())) for k, v in frame.items()),
            [('b', [(0, 6), (1, 3), (2, 2)]), ('a', [(0, 3), (1, 4), (2, 5)])])


    def test_frame_from_pandas_a(self) -> None:
        import pandas as pd

        df = pd.DataFrame(dict(a=(1,2), b=(3,4)))
        df.name = 'foo'

        f = Frame.from_pandas(df)
        self.assertEqual(f.to_pairs(0),
                (('a', ((0, 1), (1, 2))), ('b', ((0, 3), (1, 4))))
                )


    def test_frame_from_pandas_b(self) -> None:
        import pandas as pd

        df = pd.DataFrame(dict(a=(1,2), b=(False, True)), index=('x', 'y'))

        f = Frame.from_pandas(df)

        self.assertEqual(f.to_pairs(0),
                (('a', (('x', 1), ('y', 2))), ('b', (('x', False), ('y', True))))
                )

        with self.assertRaises(Exception):
            f['c'] = 0 #pylint: disable=E1137


    def test_frame_from_pandas_c(self) -> None:
        import pandas as pd

        df = pd.DataFrame(dict(a=(1,2), b=(False, True)), index=('x', 'y'))

        f = FrameGO.from_pandas(df)
        f['c'] = -1

        self.assertEqual(f.to_pairs(0),
                (('a', (('x', 1), ('y', 2))), ('b', (('x', False), ('y', True))), ('c', (('x', -1), ('y', -1)))))

    def test_frame_from_pandas_d(self) -> None:
        import pandas as pd

        df = pd.DataFrame(dict(a=(1,2), b=(3,4)))
        df.name = 'foo'

        f = Frame.from_pandas(df, own_data=True)
        self.assertEqual(f.to_pairs(0),
                (('a', ((0, 1), (1, 2))), ('b', ((0, 3), (1, 4))))
                )

    #---------------------------------------------------------------------------

    def test_frame_to_pandas_a(self) -> None:
        records = (
                (1, 2, 'a', False),
                (30, 34, 'b', True),
                (54, 95, 'c', False),
                (65, 73, 'd', True),
                )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        index = IndexHierarchy.from_product((100, 200), (True, False))
        f1 = Frame.from_records(records,
                columns=columns,
                index=index)

        df = f1.to_pandas()

        self.assertEqual(df.index.values.tolist(),
            [(100, True), (100, False), (200, True), (200, False)]
            )

        self.assertEqual(df.columns.values.tolist(),
            [('a', 1), ('a', 2), ('b', 1), ('b', 2)]
            )

        self.assertEqual(df.values.tolist(),
            [[1, 2, 'a', False], [30, 34, 'b', True], [54, 95, 'c', False], [65, 73, 'd', True]])


    def test_frame_to_pandas_b(self) -> None:
        f1 = sf.Frame.from_dict_records(
                [dict(a=1,b=1), dict(a=2,b=3), dict(a=1,b=1), dict(a=2,b=3)], index=sf.IndexHierarchy.from_labels(
                [(1,'dd',0),(1,'b',0),(2,'cc',0),(2,'ee',0)]))
        df = f1.loc[sf.HLoc[(1,'dd')]].to_pandas()

        self.assertEqual(df.index.values.tolist(),
                [(1, 'dd', 0)])
        self.assertEqual(df.values.tolist(),
                [[1, 1]]
                )


    def test_frame_to_pandas_c(self) -> None:
        f = sf.FrameGO(['a' for x in range(5)], columns=['a'])
        f['b'] = [1.0 for i in range(5)]
        df = f.to_pandas()
        self.assertEqual(df.dtypes.tolist(), [np.dtype(object), np.dtype(np.float64)])


    @skip_win  # type: ignore
    def test_frame_to_pandas_d(self) -> None:
        records = (
                (1, 2, 'a', False),
                (30, 34, 'b', True),
                (54, 95, 'c', False),
                (65, 73, 'd', True),
                )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        index = IndexHierarchy.from_product((100, 200), (True, False))
        f1 = Frame.from_records(records,
                columns=columns,
                index=index)

        df = f1.to_pandas()

        self.assertEqual( df.dtypes.tolist(),
                [np.dtype('int64'), np.dtype('int64'), np.dtype('O'), np.dtype('bool')]
                )


    def test_frame_to_pandas_e(self) -> None:
        f = Frame.from_records(
            [['a', 1, 10], ['a', 2, 200], ['b', 1, -3], ['b', 2, 7]],
            columns=('x', 'y', 'z'))
        df = f.set_index_hierarchy(['x', 'y']).to_pandas()
        self.assertEqual(list(df.index.names), ['x', 'y'])


    #---------------------------------------------------------------------------


    def test_frame_to_arrow_a(self) -> None:
        records = (
                (1, 2, 'a', False),
                (30, 34, 'b', True),
                (54, 95, 'c', False),
                (65, 73, 'd', True),
                )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        index = IndexHierarchy.from_product((100, 200), (True, False))
        f1 = Frame.from_records(records,
                columns=columns,
                index=index)

        at = f1.to_arrow()
        self.assertEqual(at.shape, (4, 6))
        self.assertEqual(at.column_names,
                ['__index0__', '__index1__', "['a' 1]", "['a' 2]", "['b' 1]", "['b' 2]"])
        self.assertEqual(at.to_pydict(),
                {'__index0__': [100, 100, 200, 200], '__index1__': [True, False, True, False], "['a' 1]": [1, 30, 54, 65], "['a' 2]": [2, 34, 95, 73], "['b' 1]": ['a', 'b', 'c', 'd'], "['b' 2]": [False, True, False, True]}
                )


    def test_frame_from_arrow_a(self) -> None:
        records = (
                (1, 2, 'a', False),
                (30, 34, 'b', True),
                (54, 95, 'c', False),
                (65, 73, 'd', True),
                )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        index = IndexHierarchy.from_product((100, 200), (True, False))
        f1 = Frame.from_records(records,
                columns=columns,
                index=index)
        at = f1.to_arrow()

        f2 = Frame.from_arrow(at,
                index_depth=f1.index.depth,
                columns_depth=f1.columns.depth
                )
        # String arrays will come in as objects
        self.assertEqualFrames(f1, f2, check_dtypes=False)

    def test_frame_to_parquet_a(self) -> None:
        records = (
                (1, 2, 'a', False),
                (30, 34, 'b', True),
                (54, 95, 'c', False),
                (65, 73, 'd', True),
                )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        index = IndexHierarchy.from_product((100, 200), (True, False))
        f1 = Frame.from_records(records,
                columns=columns,
                index=index)

        with temp_file('.parquet') as fp:
            f1.to_parquet(fp)



    def test_frame_from_parquet_a(self) -> None:
        records = (
                (1, 2, 'a', False),
                (30, 34, 'b', True),
                (54, 95, 'c', False),
                (65, 73, 'd', True),
                )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        index = IndexHierarchy.from_product((100, 200), (True, False))
        f1 = Frame.from_records(records,
                columns=columns,
                index=index)

        with temp_file('.parquet') as fp:
            f1.to_parquet(fp)
            f2 = Frame.from_parquet(fp,
                    index_depth=f1.index.depth,
                    columns_depth=f1.columns.depth)

        self.assertEqualFrames(f1, f2, check_dtypes=False)


    #---------------------------------------------------------------------------

    def test_frame_to_xarray_a(self) -> None:
        records = (
                (1, 2, 'a', False),
                (30, 34, 'b', True),
                (54, 95, 'c', False),
                (65, 73, 'd', True),
                )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2), name=('a', 'b'))
        index = IndexHierarchy.from_labels(((200, False, 'a'), (200, True, 'b'), (100, False, 'a'), (300, True, 'b')))

        f1 = Frame.from_records(records,
                columns=columns,
                index=index)
        ds1 = f1.to_xarray()
        self.assertEqual(tuple(ds1.data_vars.keys()),
                (('a', 1), ('a', 2), ('b', 1), ('b', 2))
                )
        self.assertEqual(tuple(ds1.coords.keys()),
                ('__index0__', '__index1__', '__index2__')
                )
        self.assertEqual(ds1[('b', 1)].values.ndim, 3)

        f2 = Frame.from_records(records)
        ds2 = f2.to_xarray()
        self.assertEqual(tuple(ds2.data_vars.keys()), (0, 1, 2, 3))
        self.assertEqual(tuple(ds2.coords.keys()), ('__index0__',))
        self.assertEqual(ds2[3].values.tolist(),
                [False, True, False, True])


    def test_frame_getitem_a(self) -> None:

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        f2 = f1['r':]  # type: ignore  # https://github.com/python/typeshed/pull/3024
        self.assertEqual(f2.columns.values.tolist(), ['r', 's', 't'])
        self.assertTrue((f2.index == f1.index).all())
        self.assertEqual(mloc(f2.index.values), mloc(f1.index.values))

    def test_frame_getitem_b(self) -> None:

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        # using an Index object for selection
        self.assertEqual(
                f1[f1.columns.loc['r':]].to_pairs(0),  # type: ignore  # https://github.com/python/typeshed/pull/3024
                (('r', (('x', 'a'), ('y', 'b'))), ('s', (('x', False), ('y', True))), ('t', (('x', True), ('y', False))))
                )


    def test_frame_length_a(self) -> None:

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        self.assertEqual(len(f1), 2)

    #---------------------------------------------------------------------------

    def test_frame_iloc_a(self) -> None:

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        self.assertEqual((f1.iloc[0].values == f1.loc['x'].values).all(), True)
        self.assertEqual((f1.iloc[1].values == f1.loc['y'].values).all(), True)


    def test_frame_iloc_b(self) -> None:
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


    #---------------------------------------------------------------------------

    def test_frame_iter_a(self) -> None:

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        self.assertEqual((f1.keys() == f1.columns).all(), True)
        self.assertEqual([x for x in f1.columns], ['p', 'q', 'r', 's', 't'])
        self.assertEqual([x for x in f1], ['p', 'q', 'r', 's', 't'])




    def test_frame_iter_array_a(self) -> None:

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


    def test_frame_iter_array_b(self) -> None:

        arrays = list(np.random.rand(1000) for _ in range(100))
        f1 = Frame.from_items(
                zip(range(100), arrays)
                )

        # iter columns
        post = f1.iter_array(0).apply_pool(np.sum, max_workers=4, use_threads=True)
        self.assertEqual(post.shape, (100,))
        self.assertAlmostEqual(f1.sum().sum(), post.sum())

        post = f1.iter_array(0).apply_pool(np.sum, max_workers=4, use_threads=False)
        self.assertEqual(post.shape, (100,))
        self.assertAlmostEqual(f1.sum().sum(), post.sum())

    def test_frame_iter_array_c(self) -> None:
        arrays = []
        for _ in range(8):
            arrays.append(list(range(8)))
        f1 = Frame.from_items(
                zip(range(8), arrays)
                )

        func = {x: chr(x+65) for x in range(8)}
        # iter columns
        post = f1.iter_element().apply_pool(func, max_workers=4, use_threads=True)

        self.assertEqual(post.to_pairs(0),
                ((0, ((0, 'A'), (1, 'B'), (2, 'C'), (3, 'D'), (4, 'E'), (5, 'F'), (6, 'G'), (7, 'H'))), (1, ((0, 'A'), (1, 'B'), (2, 'C'), (3, 'D'), (4, 'E'), (5, 'F'), (6, 'G'), (7, 'H'))), (2, ((0, 'A'), (1, 'B'), (2, 'C'), (3, 'D'), (4, 'E'), (5, 'F'), (6, 'G'), (7, 'H'))), (3, ((0, 'A'), (1, 'B'), (2, 'C'), (3, 'D'), (4, 'E'), (5, 'F'), (6, 'G'), (7, 'H'))), (4, ((0, 'A'), (1, 'B'), (2, 'C'), (3, 'D'), (4, 'E'), (5, 'F'), (6, 'G'), (7, 'H'))), (5, ((0, 'A'), (1, 'B'), (2, 'C'), (3, 'D'), (4, 'E'), (5, 'F'), (6, 'G'), (7, 'H'))), (6, ((0, 'A'), (1, 'B'), (2, 'C'), (3, 'D'), (4, 'E'), (5, 'F'), (6, 'G'), (7, 'H'))), (7, ((0, 'A'), (1, 'B'), (2, 'C'), (3, 'D'), (4, 'E'), (5, 'F'), (6, 'G'), (7, 'H'))))
                )

    def test_frame_iter_array_d(self) -> None:
        arrays = []
        for _ in range(8):
            arrays.append(list(range(8)))
        f1 = Frame.from_items(
                zip(range(8), arrays)
                )

        # when called with a pool, values are gien the func as a single argument, which for an element iteration is a tuple of coord, value
        func = lambda arg: arg[0][1]
        # iter columns
        post = f1.iter_element_items().apply_pool(func, max_workers=4, use_threads=True)

        self.assertEqual(post.to_pairs(0),
                ((0, ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0))), (1, ((0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1))), (2, ((0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2))), (3, ((0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3))), (4, ((0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4))), (5, ((0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5))), (6, ((0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (7, 6))), (7, ((0, 7), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7))))
                )


    def test_frame_iter_array_e(self) -> None:

        f = sf.Frame.from_dict(
                dict(diameter=(12756, 6792, 142984),
                mass=(5.97, 0.642, 1898)),
                index=('Earth', 'Mars', 'Jupiter'),
                dtypes=dict(diameter=np.int64))

        post = f.iter_array(axis=0).apply(np.sum)
        self.assertTrue(post.dtype == float)



    def test_frame_iter_tuple_a(self) -> None:
        post = tuple(sf.Frame(range(5)).iter_tuple(axis=0))
        self.assertEqual(post, ((0, 1, 2, 3, 4),))




    def test_frame_iter_tuple_b(self) -> None:
        post = tuple(sf.Frame(range(3), index=tuple('abc')).iter_tuple(axis=0))
        self.assertEqual(post, ((0, 1, 2),))

        self.assertEqual(tuple(post[0]._asdict().items()),
                (('a', 0), ('b', 1), ('c', 2))
                )


    #---------------------------------------------------------------------------

    def test_frame_setitem_a(self) -> None:

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


    def test_frame_setitem_b(self) -> None:

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = FrameGO.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        f1['u'] = 0

        self.assertEqual(f1.loc['x'].values.tolist(),
                [1, 2, 'a', False, True, 0])

        # with self.assertRaises(Exception):
        f1['w'] = [[1,2], [4,5]]
        self.assertEqual(f1['w'].to_pairs(),
                (('x', [1, 2]), ('y', [4, 5])))


    def test_frame_setitem_c(self) -> None:


        f1 = FrameGO(index=sf.Index(tuple('abcde')))
        f1['a'] = 30
        self.assertEqual(f1.to_pairs(0),
                (('a', (('a', 30), ('b', 30), ('c', 30), ('d', 30), ('e', 30))),))


    def test_frame_setitem_d(self) -> None:

        # 3d array raises exception
        f = sf.FrameGO(index=range(3))
        f['a'] = 5
        self.assertEqual(f.sum(), 15)



    def test_frame_setitem_e(self) -> None:

        # 3d array raises exception
        f = sf.FrameGO(index=range(3))
        f['a'] = 'foo'
        self.assertEqual(f.to_pairs(0),
                (('a', ((0, 'foo'), (1, 'foo'), (2, 'foo'))),)
                )





    def test_frame_extend_items_a(self) -> None:
        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = FrameGO.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        columns = OrderedDict(
            (('c', np.array([0, -1])), ('d', np.array([3, 5]))))

        f1.extend_items(columns.items())

        self.assertEqual(f1.columns.values.tolist(),
                ['p', 'q', 'r', 's', 't', 'c', 'd'])

        self.assertTypeBlocksArrayEqual(f1._blocks,
                [[1, 2, 'a', False, True, 0, 3],
                [30, 50, 'b', True, False, -1, 5]],
                match_dtype=object)

    def test_frame_extend_a(self) -> None:
        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))
        f1 = FrameGO.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        blocks = (np.array([[50, 40], [30, 20]]),
                np.array([[50, 40], [30, 20]]))
        columns = ('a', 'b', 'c', 'd')
        f2 = Frame(TypeBlocks.from_blocks(blocks), columns=columns, index=('y', 'z'))

        f1.extend(f2)
        f3 = f1.fillna(None)

        self.assertEqual(f1.columns.values.tolist(),
                ['p', 'q', 'r', 's', 't', 'a', 'b', 'c', 'd'])

        self.assertEqual(f3.to_pairs(0),
                (('p', (('x', 1), ('y', 30))), ('q', (('x', 2), ('y', 50))), ('r', (('x', 'a'), ('y', 'b'))), ('s', (('x', False), ('y', True))), ('t', (('x', True), ('y', False))), ('a', (('x', None), ('y', 50))), ('b', (('x', None), ('y', 40))), ('c', (('x', None), ('y', 50))), ('d', (('x', None), ('y', 40))))
                )

    def test_frame_extend_b(self) -> None:
        records = (
                ('a', False, True),
                ('b', True, False))
        f1 = FrameGO.from_records(records,
                columns=('p', 'q', 'r'),
                index=('x','y'))

        s1 = Series((200, -3), index=('y', 'x'))

        # this will work with a None name

        f1.extend(s1)

        self.assertEqual(f1.columns.values.tolist(), ['p', 'q', 'r', None])
        self.assertEqual(f1[None].values.tolist(), [-3, 200])


    def test_frame_extend_c(self) -> None:
        records = (
                ('a', False, True),
                ('b', True, False))
        f1 = FrameGO.from_records(records,
                columns=('p', 'q', 'r'),
                index=('x','y'))

        s1 = Series((200, -3), index=('y', 'x'), name='s')

        f1.extend(s1)

        self.assertEqual(f1.columns.values.tolist(), ['p', 'q', 'r', 's'])
        self.assertEqual(f1['s'].values.tolist(), [-3, 200])

    def test_frame_extend_d(self) -> None:
        records = (
                ('a', False, True),
                ('b', True, False))
        f1 = FrameGO.from_records(records,
                columns=('p', 'q', 'r'),
                index=('x','y'))

        s1 = Series((200, -3), index=('q', 'x'), name='s')

        f1.extend(s1, fill_value=0)

        self.assertEqual(f1.columns.values.tolist(), ['p', 'q', 'r', 's'])
        self.assertEqual(f1['s'].values.tolist(), [-3, 0])


    def test_frame_extend_empty_a(self) -> None:
        # full Frame, empty extensions with no index
        records = (
                (1, 2, 'a', False, True),
                (30, 34, 'b', True, False),
                (54, 95, 'c', False, False),
                )
        f1 = FrameGO.from_records(records,
                columns=('a', 'b', 'c', 'd', 'e'),
                index=('x', 'y', 'z'))

        f2 = FrameGO() # no index or columns

        f1.extend(f2)
        self.assertEqual(f1.shape, (3, 5)) # extension happens, but no change in shape



    def test_frame_extend_empty_b(self) -> None:
        # full Frame, empty extension with index
        records = (
                (1, 2, 'a', False, True),
                (30, 34, 'b', True, False),
                (54, 95, 'c', False, False),
                )
        f1 = FrameGO.from_records(records,
                columns=('a', 'b', 'c', 'd', 'e'),
                index=('x', 'y', 'z'))

        f2 = FrameGO(index=('x', 'y', 'z'))
        f1.extend(f2)
        self.assertEqual(f1.shape, (3, 5)) # extension happens, but no change in shape


    def test_frame_extend_empty_c(self) -> None:
        # empty with index, full frame extension

        records = (
                (1, 2, 'a', False, True),
                (30, 34, 'b', True, False),
                (54, 95, 'c', False, False),
                )
        f1 = FrameGO.from_records(records,
                columns=('a', 'b', 'c', 'd', 'e'),
                index=('x', 'y', 'z'))

        f2 = FrameGO(index=('x', 'y', 'z'))
        f2.extend(f1)
        self.assertEqual(f2.shape, (3, 5)) # extension happens, but no change in shape

        self.assertEqual(f2.to_pairs(0),
                (('a', (('x', 1), ('y', 30), ('z', 54))), ('b', (('x', 2), ('y', 34), ('z', 95))), ('c', (('x', 'a'), ('y', 'b'), ('z', 'c'))), ('d', (('x', False), ('y', True), ('z', False))), ('e', (('x', True), ('y', False), ('z', False))))
                )

    def test_frame_extend_empty_d(self) -> None:
        # full Frame, empty extension with different index
        records = (
                (1, 2, 'a', False, True),
                (30, 34, 'b', True, False),
                (54, 95, 'c', False, False),
                )
        f1 = FrameGO.from_records(records,
                columns=('a', 'b', 'c', 'd', 'e'),
                index=('x', 'y', 'z'))

        f2 = FrameGO(index=('w', 'x', 'y', 'z'))

        # import ipdb; ipdb.set_trace()
        f1.extend(f2)
        self.assertEqual(f1.shape, (3, 5)) # extension happens, but no change in shape



    def test_frame_extend_empty_e(self) -> None:
        # empty Frame with no index extended by full frame
        records = (
                (1, 2, 'a', False, True),
                (30, 34, 'b', True, False),
                (54, 95, 'c', False, False),
                )
        f1 = FrameGO.from_records(records,
                columns=('a', 'b', 'c', 'd', 'e'),
                index=('x', 'y', 'z'))

        f2 = FrameGO() # no index or columns

        f2.extend(f1)
        # as we align on the caller's index, if that index is empty, there is nothing to take from the passed Frame; however, since we observe columns, we add those (empty columns). this falls out of lower-level implementations: could be done differently if desirable.
        self.assertEqual(f2.shape, (0, 5))





    def test_frame_extract_a(self) -> None:
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


    def test_frame_extract_b(self) -> None:
        # examining cases where shape goes to zero in one dimension

        f1 = Frame(None, index=tuple('ab'), columns=('c',))
        f2 = f1[[]]
        self.assertEqual(len(f2.columns), 0)
        self.assertEqual(len(f2.index), 2)
        self.assertEqual(f2.shape, (2, 0))


    def test_frame_extract_c(self) -> None:
        # examining cases where shape goes to zero in one dimension
        f1 = Frame(None, columns=tuple('ab'), index=('c',))
        f2 = f1.loc[[]]
        self.assertEqual(f2.shape, (0, 2))
        self.assertEqual(len(f2.columns), 2)
        self.assertEqual(len(f2.index), 0)


    #---------------------------------------------------------------------------

    def test_frame_loc_a(self) -> None:

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

        self.assertEqual(f1['r':].columns.values.tolist(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
                ['r', 's', 't'])


    def test_frame_loc_b(self) -> None:
        # dimensionality of returned item based on selectors
        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        # return a series if one axis is multi
        post = f1.loc['x', 't':]  # type: ignore  # https://github.com/python/typeshed/pull/3024
        self.assertEqual(post.__class__, Series)
        self.assertEqual(post.index.values.tolist(), ['t'])

        post = f1.loc['y':, 't']  # type: ignore  # https://github.com/python/typeshed/pull/3024
        self.assertEqual(post.__class__, Series)
        self.assertEqual(post.index.values.tolist(), ['y'])

        # if both are multi than we get a Frame
        post = f1.loc['y':, 't':]  # type: ignore  # https://github.com/python/typeshed/pull/3024
        self.assertEqual(post.__class__, Frame)
        self.assertEqual(post.index.values.tolist(), ['y'])
        self.assertEqual(post.columns.values.tolist(), ['t'])

        # return a series
        post = f1.loc['x', 's':]  # type: ignore  # https://github.com/python/typeshed/pull/3024
        self.assertEqual(post.__class__, Series)
        self.assertEqual(post.index.values.tolist(),['s', 't'])

        post = f1.loc[:, 's']
        self.assertEqual(post.__class__, Series)
        self.assertEqual(post.index.values.tolist(), ['x', 'y'])

        self.assertEqual(f1.loc['x', 's'], False)
        self.assertEqual(f1.loc['y', 'p'], 30)


    def test_frame_loc_c(self) -> None:
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        post = f1.loc['x':]  # type: ignore  # https://github.com/python/typeshed/pull/3024
        self.assertEqual(post.index.values.tolist(),
                ['x', 'y', 'z'])


    def test_frame_loc_d(self) -> None:

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'),
                name='foo')

        f2 = f1['r':]  # type: ignore  # https://github.com/python/typeshed/pull/3024
        f3 = f1.loc[['y'], ['r']]
        self.assertEqual(f1.name, 'foo')
        self.assertEqual(f2.name, 'foo')
        self.assertEqual(f3.name, 'foo')

        s1 = f2.loc[:, 's']
        self.assertEqual(s1.name, 's')

        s2 = f1.loc['x', :'r']  # type: ignore  # https://github.com/python/typeshed/pull/3024
        self.assertEqual(s2.name, 'x')


    def test_frame_loc_e(self) -> None:
        fp = self.get_test_input('jph_photos.txt')
        # using a raw string to avoid unicode decoding issues on windows
        f = sf.Frame.from_tsv(fp, dtypes=dict(albumId=np.int64, id=np.int64), encoding='utf-8')
        post = f.loc[f['albumId'] >= 98]
        self.assertEqual(post.shape, (150, 5))


    def test_frame_loc_f(self) -> None:
        f = Frame(range(3), index=sf.Index(tuple('abc'), name='index'))
        self.assertEqual(f.loc['b':].index.name, 'index') # type: ignore


    #---------------------------------------------------------------------------

    def test_frame_items_a(self) -> None:

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




    @skip_win  # type: ignore
    def test_frame_attrs_a(self) -> None:

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

    #---------------------------------------------------------------------------
    def test_frame_assign_getitem_a(self) -> None:

        f1 = FrameGO(index=(0,1,2))
        for idx, col in enumerate(string.ascii_uppercase + string.ascii_lowercase):
            f1[col] = idx

        fields = ['m','V','P','c','Y','r','q','R','j','X','a','E','K','p','u','G','D','w','d','e','H','i','h','N','O','k','l','F','g','o','M','T','n','L','Q','W','t','v','s','Z','J','I','b']

        # check that normal selection works
        f1_sub = f1[fields]
        self.assertEqual(f1_sub.columns.values.tolist(), fields)

        f2 = f1.assign[fields](f1[fields] * 0.5)

        self.assertTrue((f2.columns == f1.columns).all())
        self.assertTrue((f2.index == f1.index).all())

        # as expected, values is coercing all to floats
        self.assertEqual(f2.values.tolist(),
                [[0.0, 1.0, 2.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 18.0, 9.5, 20.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 31.0, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 49.0, 50.0, 51.0], [0.0, 1.0, 2.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 18.0, 9.5, 20.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 31.0, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 49.0, 50.0, 51.0], [0.0, 1.0, 2.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 18.0, 9.5, 20.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 31.0, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 49.0, 50.0, 51.0]]
                )

    def test_frame_assign_getitem_b(self) -> None:

        f1 = FrameGO(index=(0,1,2))
        for idx, col in enumerate('abcdef'):
            f1[col] = idx

        f2 = f1.assign['b'](Series(('b','c','d'), index=(2, 0, 1)))
        self.assertEqual(f2['b'].to_pairs(),
                ((0, 'c'), (1, 'd'), (2, 'b')))


    def test_frame_assign_getitem_c(self) -> None:
        f1 = sf.Frame(False, index=range(2), columns=tuple('ab'))
        f2 = f1.assign['a']([1.1, 2.1])
        self.assertEqual(f2._blocks.shapes.tolist(), [(2,), (2,1)])
        self.assertEqual(f2.dtypes.values.tolist(), [np.dtype('float64'), np.dtype('bool')])
        self.assertEqual(f2.to_pairs(0),
                (('a', ((0, 1.1), (1, 2.1))), ('b', ((0, False), (1, False))))
                )

    def test_frame_assign_getitem_d(self) -> None:
        f1 = sf.Frame(False, index=range(2), columns=tuple('abcd'))
        f2 = f1.assign['b']([1.1, 2.1])
        self.assertEqual(f2._blocks.shapes.tolist(), [(2, 1), (2,), (2, 2)])
        self.assertEqual(f2.dtypes.values.tolist(),
                [np.dtype('bool'), np.dtype('float64'), np.dtype('bool'), np.dtype('bool')]
                )
        self.assertEqual( f2.to_pairs(0),
                (('a', ((0, False), (1, False))), ('b', ((0, 1.1), (1, 2.1))), ('c', ((0, False), (1, False))), ('d', ((0, False), (1, False))))
                )

    def test_frame_assign_getitem_e(self) -> None:
        f1 = sf.Frame(False, index=range(2), columns=tuple('abcd'))
        f2 = f1.assign['c']([1.1, 2.1])
        self.assertEqual(f2._blocks.shapes.tolist(), [(2, 2), (2,), (2, 1)])
        self.assertEqual(f2.dtypes.values.tolist(),
                [np.dtype('bool'), np.dtype('bool'), np.dtype('float64'), np.dtype('bool')]
                )
        self.assertEqual(f2.to_pairs(0),
                (('a', ((0, False), (1, False))), ('b', ((0, False), (1, False))), ('c', ((0, 1.1), (1, 2.1))), ('d', ((0, False), (1, False))))
                )

    def test_frame_assign_getitem_f(self) -> None:
        f1 = sf.Frame(False, index=range(2), columns=tuple('abcd'))
        f2 = f1.assign['d']([1.1, 2.1])
        self.assertEqual(f2._blocks.shapes.tolist(), [(2, 3), (2,),])
        self.assertEqual(f2.dtypes.values.tolist(),
                [np.dtype('bool'), np.dtype('bool'), np.dtype('bool'), np.dtype('float64')]
                )
        self.assertEqual(f2.to_pairs(0),
                (('a', ((0, False), (1, False))), ('b', ((0, False), (1, False))), ('c', ((0, False), (1, False))), ('d', ((0, 1.1), (1, 2.1))))
                )

    def test_frame_assign_getitem_g(self) -> None:
        f1 = sf.Frame(False, index=range(2), columns=tuple('abcde'))
        f2 = f1.assign['b':'d']('x') # type: ignore
        self.assertEqual(f2._blocks.shapes.tolist(), [(2, 1), (2, 3), (2, 1)])
        self.assertEqual(f2.dtypes.values.tolist(),
                [np.dtype('bool'), np.dtype('<U1'), np.dtype('<U1'), np.dtype('<U1'), np.dtype('bool')]
                )
        self.assertEqual(f2.to_pairs(0),
                (('a', ((0, False), (1, False))), ('b', ((0, 'x'), (1, 'x'))), ('c', ((0, 'x'), (1, 'x'))), ('d', ((0, 'x'), (1, 'x'))), ('e', ((0, False), (1, False)))))



    def test_frame_assign_iloc_a(self) -> None:

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))


        self.assertEqual(f1.assign.iloc[1,1](3000).iloc[1,1], 3000)


    def test_frame_assign_loc_a(self) -> None:

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


    def test_frame_assign_loc_b(self) -> None:

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




    def test_frame_assign_loc_c(self) -> None:

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        # assinging a series to a part of wone row
        post = f1.assign.loc['x', 'r':](Series((-1, -2, -3), index=('t', 'r', 's')))  # type: ignore  # https://github.com/python/typeshed/pull/3024

        self.assertEqual(post.values.tolist(),
                [[1, 2, -2, -3, -1], [30, 50, 'b', True, False]])

        post = f1.assign.loc[['x', 'y'], 'r'](Series((-1, -2), index=('y', 'x')))

        self.assertEqual(post.values.tolist(),
                [[1, 2, -2, False, True], [30, 50, -1, True, False]])

        # ordere here does not matter
        post = f1.assign.loc[['y', 'x'], 'r'](Series((-1, -2), index=('y', 'x')))

        self.assertEqual(post.values.tolist(),
                [[1, 2, -2, False, True], [30, 50, -1, True, False]])


    def test_frame_assign_loc_d(self) -> None:

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'),
                consolidate_blocks=True)

        value1 = Frame.from_records(((20, 21, 22),(23, 24, 25)),
                index=('x', 'y'),
                columns=('s', 't', 'w'),
                consolidate_blocks=True)

        f2 = f1.assign.loc[['x', 'y'], ['s', 't']](value1)
        self.assertEqual(f2.to_pairs(0),
                (('p', (('x', 1), ('y', 30))), ('q', (('x', 2), ('y', 50))), ('r', (('x', 'a'), ('y', 'b'))), ('s', (('x', 20), ('y', 23))), ('t', (('x', 21), ('y', 24)))))


    def test_frame_assign_loc_e(self) -> None:

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False),
                (30, 50, 'c', False, False),
                (3, 5, 'c', False, True),
                (30, 500, 'd', True, True),
                (30, 2, 'e', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=list('abcdef')
                )

        f3 = f1.assign.iloc[5](f1.iloc[0])
        self.assertEqual(f3.to_pairs(0),
                (('p', (('a', 1), ('b', 30), ('c', 30), ('d', 3), ('e', 30), ('f', 1))), ('q', (('a', 2), ('b', 50), ('c', 50), ('d', 5), ('e', 500), ('f', 2))), ('r', (('a', 'a'), ('b', 'b'), ('c', 'c'), ('d', 'c'), ('e', 'd'), ('f', 'a'))), ('s', (('a', False), ('b', True), ('c', False), ('d', False), ('e', True), ('f', False))), ('t', (('a', True), ('b', False), ('c', False), ('d', True), ('e', True), ('f', True))))
                )

        f4 = f1.assign['q'](f1['q'] * 2)
        self.assertEqual(f4.to_pairs(0),
                (('p', (('a', 1), ('b', 30), ('c', 30), ('d', 3), ('e', 30), ('f', 30))), ('q', (('a', 4), ('b', 100), ('c', 100), ('d', 10), ('e', 1000), ('f', 4))), ('r', (('a', 'a'), ('b', 'b'), ('c', 'c'), ('d', 'c'), ('e', 'd'), ('f', 'e'))), ('s', (('a', False), ('b', True), ('c', False), ('d', False), ('e', True), ('f', True))), ('t', (('a', True), ('b', False), ('c', False), ('d', True), ('e', True), ('f', True))))
                )

        f5 = f1.assign[['p', 'q']](f1[['p', 'q']] * 2)
        self.assertEqual(f5.to_pairs(0),
                (('p', (('a', 2), ('b', 60), ('c', 60), ('d', 6), ('e', 60), ('f', 60))), ('q', (('a', 4), ('b', 100), ('c', 100), ('d', 10), ('e', 1000), ('f', 4))), ('r', (('a', 'a'), ('b', 'b'), ('c', 'c'), ('d', 'c'), ('e', 'd'), ('f', 'e'))), ('s', (('a', False), ('b', True), ('c', False), ('d', False), ('e', True), ('f', True))), ('t', (('a', True), ('b', False), ('c', False), ('d', True), ('e', True), ('f', True))))
                )


    def test_frame_assign_loc_f(self) -> None:
        f1 = sf.Frame(False, index=range(2), columns=tuple('abcde'))
        f2 = f1.assign.loc[1, 'b':'d']('x') # type: ignore
        self.assertEqual(f2.dtypes.values.tolist(),
                [np.dtype('bool'), np.dtype('O'), np.dtype('O'), np.dtype('O'), np.dtype('bool')])
        self.assertEqual(f2.to_pairs(0),
                (('a', ((0, False), (1, False))), ('b', ((0, False), (1, 'x'))), ('c', ((0, False), (1, 'x'))), ('d', ((0, False), (1, 'x'))), ('e', ((0, False), (1, False)))))


    def test_frame_assign_coercion_a(self) -> None:

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))
        f2 = f1.assign.loc['x', 'r'](None)
        self.assertEqual(f2.to_pairs(0),
                (('p', (('x', 1), ('y', 30))), ('q', (('x', 2), ('y', 50))), ('r', (('x', None), ('y', 'b'))), ('s', (('x', False), ('y', True))), ('t', (('x', True), ('y', False)))))



    def test_frame_assign_bloc_a(self) -> None:

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        sel = np.array([[False, True, False, True, False],
                [True, False, True, False, True]])
        f2 = f1.assign.bloc(sel)(-10)
        self.assertEqual(f2.to_pairs(0),
                (('p', (('x', 1), ('y', -10))), ('q', (('x', -10), ('y', 50))), ('r', (('x', 'a'), ('y', -10))), ('s', (('x', -10), ('y', True))), ('t', (('x', True), ('y', -10))))
                )

        f3 = f1.assign.bloc(sel)(None)
        self.assertEqual(f3.to_pairs(0),
                (('p', (('x', 1), ('y', None))), ('q', (('x', None), ('y', 50))), ('r', (('x', 'a'), ('y', None))), ('s', (('x', None), ('y', True))), ('t', (('x', True), ('y', None))))
                )

    def test_frame_assign_bloc_b(self) -> None:

        records = (
                (1, 2, 20, 40),
                (30, 50, -4, 5))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's',),
                index=('x','y'))

        sel = np.array([[False, True, False, True],
                [True, False, True, False]])

        # assignment from a frame of the same size
        f2 =f1.assign.bloc(sel)(f1 * 100)

        self.assertEqual(f2.to_pairs(0),
                (('p', (('x', 1), ('y', 3000))), ('q', (('x', 200), ('y', 50))), ('r', (('x', 20), ('y', -400))), ('s', (('x', 4000), ('y', 5))))
                )


    def test_frame_assign_bloc_c(self) -> None:

        records = (
                (1, 2, 20, 40),
                (30, 50, -4, 5))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's',),
                index=('x','y'))

        sel = f1 >= 20

        f2 = f1.assign.bloc(sel)(f1 * 100)

        match = (('p', (('x', 1), ('y', 3000))), ('q', (('x', 2), ('y', 5000))), ('r', (('x', 2000), ('y', -4))), ('s', (('x', 4000), ('y', 5))))

        self.assertEqual(f2.to_pairs(0), match)

        # reording the value will have no affect
        f3 = f1.reindex(columns=('r','q','s','p'))
        f4 = f1.assign.bloc(sel)(f3 * 100)

        self.assertEqual(f4.to_pairs(0), match)


    def test_frame_assign_bloc_d(self) -> None:

        records = (
                (1, 2, 20, 40),
                (30, 50, 4, 5))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's',),
                index=('x','y'))

        sel = f1 < 100 # get all true

        # get a value that will require reindexing
        f2 = Frame(-100,
                columns=('q', 'r',),
                index=('y',))

        self.assertEqual(f1.assign.bloc(sel)(f2).to_pairs(0),
                (('p', (('x', 1), ('y', 30))), ('q', (('x', 2), ('y', -100))), ('r', (('x', 20), ('y', -100))), ('s', (('x', 40), ('y', 5))))
            )

    #---------------------------------------------------------------------------
    def test_frame_mask_loc_a(self) -> None:

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        self.assertEqual(
                f1.mask.loc['x', 'r':].values.tolist(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
                [[False, False, True, True, True], [False, False, False, False, False]])


        self.assertEqual(f1.mask['s'].values.tolist(),
                [[False, False, False, True, False], [False, False, False, True, False]])


    def test_frame_masked_array_loc_a(self) -> None:

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        # mask our the non-integers
        self.assertEqual(
                f1.masked_array.loc[:, 'r':].sum(), 83)  # type: ignore  # https://github.com/python/typeshed/pull/3024


    def test_reindex_other_like_iloc_a(self) -> None:

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


    def test_reindex_other_like_iloc_b(self) -> None:

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


    def test_frame_reindex_a(self) -> None:

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



    def test_frame_axis_flat_a(self) -> None:

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


    def test_frame_reindex_b(self) -> None:

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


    def test_frame_reindex_c(self) -> None:
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


    def test_frame_reindex_d(self) -> None:

        records = (
                (1, 2, 'a', False, True),
                (30, 34, 'b', True, False),
                (54, 95, 'c', False, False),
                )

        columns = IndexHierarchy.from_labels((('a', 1), ('a', 2), ('b', 1), ('b', 2), ('b', 3)))
        f1 = Frame.from_records(records,
                columns=columns,
                index=('x', 'y', 'z'))

        # NOTE: must use HLoc on getting a single columns as otherwise looks like a multiple axis selection
        self.assertEqual(f1[sf.HLoc['a', 1]].to_pairs(),
                (('x', 1), ('y', 30), ('z', 54))
                )

        self.assertEqual(f1[sf.HLoc['b', 1]:].to_pairs(0),  # type: ignore  # https://github.com/python/typeshed/pull/3024
                ((('b', 1), (('x', 'a'), ('y', 'b'), ('z', 'c'))), (('b', 2), (('x', False), ('y', True), ('z', False))), (('b', 3), (('x', True), ('y', False), ('z', False)))))

        # reindexing with no column matches results in NaN for all values
        self.assertTrue(
                f1.iloc[:, 1:].reindex(columns=IndexHierarchy.from_product(('b', 'a'), (10, 20))).isna().all().all())

        columns = IndexHierarchy.from_product(('b', 'a'), (3, 2))
        f2 = f1.reindex(columns=columns, fill_value=None)
        self.assertEqual(f2.to_pairs(0),
                ((('b', 3), (('x', True), ('y', False), ('z', False))), (('b', 2), (('x', False), ('y', True), ('z', False))), (('a', 3), (('x', None), ('y', None), ('z', None))), (('a', 2), (('x', 2), ('y', 34), ('z', 95)))))


    def test_frame_reindex_e(self) -> None:

        records = (
                (1, 2, 'a', False),
                (30, 34, 'b', True),
                (54, 95, 'c', False),
                (65, 73, 'd', True),
                )

        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        index = IndexHierarchy.from_product((100, 200), (True, False))

        f1 = Frame.from_records(records,
                columns=columns,
                index=index)

        self.assertEqual(f1.loc[(200, True):, ('b',1):].to_pairs(0),  # type: ignore  # https://github.com/python/typeshed/pull/3024
                ((('b', 1), (((200, True), 'c'), ((200, False), 'd'))), (('b', 2), (((200, True), False), ((200, False), True)))))

        # reindex from IndexHierarchy to Index with tuples
        f2 = f1.reindex(
                index=IndexHierarchy.from_product((200, 300), (False, True)),
                columns=[('b',1),('a',1)],
                fill_value=None)
        self.assertEqual(f2.to_pairs(0),
                ((('b', 1), (((200, False), 'd'), ((200, True), 'c'), ((300, False), None), ((300, True), None))), (('a', 1), (((200, False), 65), ((200, True), 54), ((300, False), None), ((300, True), None)))))



    def test_frame_reindex_f(self) -> None:

        records = (
                (1, 2, 'a', False),
                (30, 34, 'b', True),
                )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        f1 = Frame.from_records(records, columns=columns, name='foo')
        f2 = f1.reindex(index=(0,1,2), fill_value=0)

        self.assertEqual(f1.name, 'foo')
        self.assertEqual(f2.name, 'foo')


    def test_frame_reindex_g(self) -> None:

        records = (
                (1, 2, 'a', False),
                (30, 34, 'b', True),
                )
        columns1 = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        f1 = Frame.from_records(records, columns=columns1, name='foo')

        index = Index((0, 1, 2))
        f2 = f1.reindex(index=index, fill_value=0, own_index=True)
        self.assertEqual(id(f2.index), id(index))

        columns2 = IndexHierarchy.from_labels((('a', 2), ('b', 1)))
        f2 = f1.reindex(columns=columns2, own_columns=True)
        self.assertEqual(id(f2.columns), id(columns2))

        self.assertEqual(f2.to_pairs(0),
                ((('a', 2), ((0, 2), (1, 34))), (('b', 1), ((0, 'a'), (1, 'b'))))
                )

    def test_frame_axis_interface_a(self) -> None:
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

        def sum_if(idx: tp.Hashable, vals: tp.Iterable[int]) -> tp.Optional[int]:
            if idx in ('x', 'z'):
                return tp.cast(int, np.sum(vals))
            return None

        s3 = f2.iter_array_items(1).apply(sum_if)
        self.assertEqual(list(s3.items()),
                [('w', None), ('x', 64), ('y', None), ('z', 138)])



    def test_frame_group_a(self) -> None:
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


    def test_frame_group_b(self) -> None:
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



    def test_frame_axis_interface_b(self) -> None:
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


    def test_frame_contains_a(self) -> None:

        f1 = Frame.from_items(zip(('a', 'b'), ([20, 30, 40], [80, 10, 30])),
                index=('x', 'y', 'z'))

        self.assertTrue('a' in f1)
        self.assertTrue('b' in f1)
        self.assertFalse('x' in f1)
        self.assertFalse('y' in f1)



    def test_frame_sum_a(self) -> None:
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


    def test_frame_sum_b(self) -> None:

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


    def test_frame_sum_c(self) -> None:

        index = list(''.join(x) for x in it.combinations(string.ascii_lowercase, 2))

        f1 = FrameGO(index=index)
        for col in range(100):
            s = Series(col * .1, index=index[col: col+20])
            f1[col] = s
        assert f1.sum().sum() == 9900.0


    def test_frame_sum_d(self) -> None:

        a1 = np.array([
                (2, 2, 3, 4.23, np.nan),
                (30, 34, None, 80.6, 90.123),
                ], dtype=object)
        f1 = Frame(a1,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x'),
                )

        self.assertEqual(f1.sum(axis=0).values.tolist(),
                [32, 36, 3, 84.83, 90.123])

        self.assertEqual(f1.sum(axis=1).values.tolist(),
                [11.23, 234.723])

        with self.assertRaises(TypeError):
            f1.sum(skipna=False)


    def test_frame_mean_a(self) -> None:

        a1 = np.array([
            [1, 2, 3, -5],
            [10, 50, 30, -7],
            [1345, 2234, 3345, -200]])
        a2 = np.array([2, 50, 2234])
        a3 = np.array([3, 30, 3345])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(tb1,
                columns=tuple('pqrstu'),
                index=('w', 'x', 'y'))

        self.assertEqual(
                f1.mean(axis=0).values.tolist(),
                f1.values.mean(axis=0).tolist())

        self.assertEqual(
                f1.mean(axis=1).values.tolist(),
                f1.values.mean(axis=1).tolist())

    def test_frame_median_a(self) -> None:

        a1 = np.array([
            [1, 2, 3, -5],
            [10, 50, 30, -7],
            [1345, 2234, 3345, -200]])
        a2 = np.array([2, 50, 2234])
        a3 = np.array([3, 30, 3345])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(tb1,
                columns=tuple('pqrstu'),
                index=('w', 'x', 'y'))

        self.assertEqual(
                f1.median(axis=0).values.tolist(),
                np.median(f1.values, axis=0).tolist())

        self.assertEqual(
                f1.median(axis=1).values.tolist(),
                np.median(f1.values, axis=1).tolist())


    def test_frame_std_a(self) -> None:

        a1 = np.array([
            [1, 2, 3, -5],
            [10, 50, 30, -7],
            [1345, 2234, 3345, -200]])
        a2 = np.array([2, 50, 2234])
        a3 = np.array([3, 30, 3345])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(tb1,
                columns=tuple('pqrstu'),
                index=('w', 'x', 'y'))

        self.assertEqual(
                f1.std(axis=0).values.tolist(),
                np.std(f1.values, axis=0).tolist())

        self.assertEqual(
                f1.std(axis=1).values.tolist(),
                np.std(f1.values, axis=1).tolist())



    def test_frame_var_a(self) -> None:

        a1 = np.array([
            [1, 2, 3, -5],
            [10, 50, 30, -7],
            [1345, 2234, 3345, -200]])
        a2 = np.array([2, 50, 2234])
        a3 = np.array([3, 30, 3345])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(tb1,
                columns=tuple('pqrstu'),
                index=('w', 'x', 'y'))

        self.assertEqual(
                f1.var(axis=0).values.tolist(),
                np.var(f1.values, axis=0).tolist())

        self.assertEqual(
                f1.var(axis=1).values.tolist(),
                np.var(f1.values, axis=1).tolist())



    def test_frame_prod_a(self) -> None:

        records = (
                (2, 2, 3),
                (30, 34, 60),
                (2, 95, 1),
                (30, 73, 50),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r'),
                index=('w', 'x', 'y', 'z'))

        self.assertEqual(
            f1.prod(axis=0).to_pairs(),
            (('p', 3600), ('q', 471580), ('r', 9000))
            )

        self.assertEqual(f1.prod(axis=1).to_pairs(),
            (('w', 12), ('x', 61200), ('y', 190), ('z', 109500))
            )


    def test_frame_prod_b(self) -> None:

        a1 = np.array([
            [1, 2, 3, -5],
            [10, 50, 30, -7],
            [1345, 2234, 3345, -200]])
        a2 = np.array([2, 50, 2234])
        a3 = np.array([3, 30, 3345.2])
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(tb1,
                columns=tuple('pqrstu'),
                index=('w', 'x', 'y'))

        self.assertEqual(
                f1.prod(axis=0).values.tolist(),
                np.prod(f1.values, axis=0).tolist())

        self.assertEqual(
                f1.prod(axis=1).values.tolist(),
                np.prod(f1.values, axis=1).tolist())


    def test_frame_cumsum_a(self) -> None:

        records = (
                (2, 2, 3),
                (30, 34, 60),
                (2, 95, 1),
                (30, 73, 50),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r'),
                index=('w', 'x', 'y', 'z'))

        f2 = f1.cumsum()

        self.assertEqual(
                f2.to_pairs(0),
                (('p', (('w', 2), ('x', 32), ('y', 34), ('z', 64))), ('q', (('w', 2), ('x', 36), ('y', 131), ('z', 204))), ('r', (('w', 3), ('x', 63), ('y', 64), ('z', 114))))
                )
        self.assertEqual(f1.cumsum(1).to_pairs(0),
                (('p', (('w', 2), ('x', 30), ('y', 2), ('z', 30))), ('q', (('w', 4), ('x', 64), ('y', 97), ('z', 103))), ('r', (('w', 7), ('x', 124), ('y', 98), ('z', 153))))
                )



    def test_frame_cumprod_a(self) -> None:

        records = (
                (2, 2, 3),
                (30, 34, 60),
                (2, 95, 1),
                (30, 73, 50),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r'),
                index=('w', 'x', 'y', 'z'))

        self.assertEqual(f1.cumprod(0).to_pairs(0),
                (('p', (('w', 2), ('x', 60), ('y', 120), ('z', 3600))), ('q', (('w', 2), ('x', 68), ('y', 6460), ('z', 471580))), ('r', (('w', 3), ('x', 180), ('y', 180), ('z', 9000))))
                )

        self.assertEqual(f1.cumprod(1).to_pairs(0),
                (('p', (('w', 2), ('x', 30), ('y', 2), ('z', 30))), ('q', (('w', 4), ('x', 1020), ('y', 190), ('z', 2190))), ('r', (('w', 12), ('x', 61200), ('y', 190), ('z', 109500))))
                )

    def test_frame_min_a(self) -> None:
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

    @skip_win  # type: ignore
    def test_frame_row_dtype_a(self) -> None:
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

    def test_frame_unary_operator_a(self) -> None:

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


    def test_frame_binary_operator_a(self) -> None:
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



    def test_frame_binary_operator_b(self) -> None:

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

    def test_frame_binary_operator_c(self) -> None:

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


    def test_frame_binary_operator_d(self) -> None:

        records = (
                (2, True, ''),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r'))

        self.assertEqual((f1['q'] == True).to_pairs(),
                ((0, True),))

        # this handles the case where, because we are comparing to an empty string, NumPy returns a single Boolean. This is manually handled in Series._ufunc_binary_operator
        self.assertEqual((f1['r'] == True).to_pairs(),
                ((0, False),))


    def test_frame_binary_operator_e(self) -> None:
        # keep column order when columns are the same
        f = sf.Frame(1, columns=['dog', 3, 'bat'], index=[1, 2])
        post = f / f.sum()
        self.assertEqual(post.columns.values.tolist(), f.columns.values.tolist())


    def test_frame_binary_operator_f(self) -> None:

        # matrix multiplication, with realigment of same sized axis

        a = Frame.from_dict(dict(a=(1, 2, 3, 4), b=(5, 6, 7, 8)), index=tuple('wxyz'))
        b = Frame.from_dict(dict(p=(1, 2), q=(3, 4), r=(5, 6)), index=tuple('ab'))

        post1 = a @ b

        self.assertEqual(
                post1.to_pairs(0),
                (('p', (('w', 11), ('x', 14), ('y', 17), ('z', 20))), ('q', (('w', 23), ('x', 30), ('y', 37), ('z', 44))), ('r', (('w', 35), ('x', 46), ('y', 57), ('z', 68))))
                )

        # with reorded index on b, we get the same result, as opposite axis align
        post2 = a @ b.reindex(index=('b', 'a'))

        self.assertEqual(
                post2.to_pairs(0),
                (('p', (('w', 11), ('x', 14), ('y', 17), ('z', 20))), ('q', (('w', 23), ('x', 30), ('y', 37), ('z', 44))), ('r', (('w', 35), ('x', 46), ('y', 57), ('z', 68))))
                )

        post3 = a @ Series([1, 2], index=tuple('ab'))
        self.assertEqual(post3.to_pairs(),
                (('w', 11), ('x', 14), ('y', 17), ('z', 20)))

        # index is aligned
        post4 = a @ Series([2, 1], index=tuple('ba'))
        self.assertEqual(post3.to_pairs(),
                (('w', 11), ('x', 14), ('y', 17), ('z', 20)))


    def test_frame_binary_operator_g(self) -> None:

        # matrix multiplication, with realigment of different sized axis

        a = FrameGO.from_dict(dict(a=(1, 2, 3, 4), b=(5, 6, 7, 8)), index=tuple('wxyz'))
        a['c'] = 30

        b = Frame.from_dict(dict(p=(1, 2), q=(3, 4), r=(5, 6)), index=tuple('ab'))

        with self.assertRaises(RuntimeError):
            post1 = a @ b
            # all values would go to NaN

        a = FrameGO.from_dict(dict(a=(1, 2, 3, 4), b=(5, 6, 7, 8)), index=tuple('wxyz'))

        b = Frame.from_dict(dict(p=(1, 2, 3), q=(3, 4, 5), r=(5, 6, 7)), index=tuple('abc'))

        with self.assertRaises(RuntimeError):
            post2 = a @ b


    def test_frame_binary_operator_h(self) -> None:

        a = Frame.from_dict(dict(a=(1, 2, 3, 4), b=(5, 6, 7, 8)), index=tuple('wxyz'))
        b = Frame.from_dict(dict(p=(1, 2), q=(3, 4), r=(5, 6)), index=tuple('ab'))


        self.assertEqual(
                (a @ b.values).to_pairs(0),
                ((0, (('w', 11), ('x', 14), ('y', 17), ('z', 20))), (1, (('w', 23), ('x', 30), ('y', 37), ('z', 44))), (2, (('w', 35), ('x', 46), ('y', 57), ('z', 68))))
                )
        # NOTE: the following yields a ValueError from the interpreter
        # post2 = a.values @ b


    def test_frame_binary_operator_i(self) -> None:

        a = sf.Frame((1, 2, 3))
        post = a == a.to_frame_go()

        self.assertEqual(post.__class__, FrameGO)
        self.assertEqual(post.to_pairs(0),
            ((0, ((0, True), (1, True), (2, True))),))


    def test_frame_isin_a(self) -> None:
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


    def test_frame_transpose_a(self) -> None:
        # reindex both axis
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'),
                name='foo')

        f2 = f1.transpose()

        self.assertEqual(f2.to_pairs(0),
                (('w', (('p', 2), ('q', 2), ('r', 'a'), ('s', False), ('t', False))), ('x', (('p', 30), ('q', 34), ('r', 'b'), ('s', True), ('t', False))), ('y', (('p', 2), ('q', 95), ('r', 'c'), ('s', False), ('t', False))), ('z', (('p', 30), ('q', 73), ('r', 'd'), ('s', True), ('t', True)))))

        self.assertEqual(f2.name, f1.name)


    def test_frame_from_element_iloc_items_a(self) -> None:
        items = (((0,1), 'g'), ((1,0), 'q'))

        f1 = Frame.from_element_iloc_items(items,
                index=('a', 'b'),
                columns=('x', 'y'),
                dtype=object,
                name='foo'
                )

        self.assertEqual(f1.to_pairs(0),
                (('x', (('a', None), ('b', 'q'))), ('y', (('a', 'g'), ('b', None)))))


        self.assertEqual(f1.name, 'foo')


    def test_frame_from_element_iloc_items_b(self) -> None:

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


    def test_frame_from_element_loc_items_a(self) -> None:
        items = ((('b', 'x'), 'g'), (('a','y'), 'q'))

        f1 = Frame.from_element_loc_items(items,
                index=('a', 'b'),
                columns=('x', 'y'),
                dtype=object,
                name='foo'
                )

        self.assertEqual(f1.to_pairs(0),
                (('x', (('a', None), ('b', 'g'))), ('y', (('a', 'q'), ('b', None)))))
        self.assertEqual(f1.name, 'foo')

    def test_frame_from_items_a(self) -> None:

        f1 = Frame.from_items(
                zip(range(10), (np.random.rand(1000) for _ in range(10))),
                name='foo'
                )
        self.assertEqual(f1.name, 'foo')

    def test_frame_from_items_b(self) -> None:

        s1 = Series((1, 2, 3), index=('a', 'b', 'c'))
        s2 = Series((4, 5, 6), index=('a', 'b', 'c'))

        with self.assertRaises(RuntimeError):
            # must have an index to consume Series
            Frame.from_items(zip(list('xy'), (s1, s2)))

    def test_frame_from_items_c(self) -> None:

        s1 = Series((1, 2, 3), index=('a', 'b', 'c'))
        s2 = Series((4, 5, 6), index=('a', 'b', 'c'))

        f1 = Frame.from_items(zip(list('xy'), (s1, s2)), index=s1.index)

        self.assertEqual(f1.to_pairs(0),
                (('x', (('a', 1), ('b', 2), ('c', 3))), ('y', (('a', 4), ('b', 5), ('c', 6)))))

    def test_frame_from_items_d(self) -> None:

        s1 = Series((1, 2, 3), index=('a', 'b', 'c'))
        s2 = Series((4, 5, 6), index=('a', 'b', 'c'))

        f1 = Frame.from_items(zip(list('xy'), (s1, s2)), index=('c', 'a'))

        self.assertEqual(f1.to_pairs(0),
            (('x', (('c', 3), ('a', 1))), ('y', (('c', 6), ('a', 4)))))


    def test_frame_from_items_e(self) -> None:

        s1 = Series((1, 2, 3), index=('a', 'b', 'c'))
        s2 = Series((4, 5, 6), index=('a', 'b', 'c'))
        s3 = Series((7, 8, 9), index=('a', 'b', 'c'))

        f1 = Frame.from_items(zip(list('xy'), (s1, s2, s3)), index=('c', 'a'),
                consolidate_blocks=True)

        self.assertEqual(len(f1._blocks._blocks), 1)



    def test_frame_from_structured_array_a(self) -> None:
        a = np.array([('Venus', 4.87, 464), ('Neptune', 102, -200)],
                dtype=[('name', object), ('mass', 'f4'), ('temperature', 'i4')])

        f = sf.Frame.from_structured_array(a,
                index_depth=1,
                index_column_first='name',
                name='foo')

        self.assertEqual(f.shape, (2, 2))
        self.assertEqual(f.name, 'foo')
        self.assertEqual(f['temperature'].sum(), 264)


    def test_frame_from_structured_array_b(self) -> None:
        a = np.array([('Venus', 4.87, 464), ('Neptune', 102, -200)],
                dtype=[('name', object), ('mass', 'f4'), ('temperature', 'i4')])

        f = sf.Frame.from_structured_array(a,
                index_column_first=2,
                index_depth=1,
                name='foo')
        self.assertEqual(f['name'].to_pairs(),
                ((464, 'Venus'), (-200, 'Neptune')))



    def test_frame_iter_element_a(self) -> None:
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




    def test_frame_iter_element_b(self) -> None:
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

    def test_frame_iter_element_c(self) -> None:

        a2 = np.array([
                [None, None],
                [None, 1],
                [None, 5]
                ], dtype=object)
        a1 = np.array([True, False, True])
        a3 = np.array([['a'], ['b'], ['c']])

        tb1 = TypeBlocks.from_blocks((a3, a1, a2))

        f1 = Frame(tb1,
                index=self.get_letters(None, tb1.shape[0]),
                columns=IndexHierarchy.from_product(('i', 'ii'), ('a', 'b'))
                )
        values = list(f1.iter_element())
        self.assertEqual(values,
                ['a', True, None, None, 'b', False, None, 1, 'c', True, None, 5]
                )

        f2 = f1.iter_element().apply(lambda x: str(x).lower().replace('e', ''))

        self.assertEqual(f1.columns.__class__, f2.columns.__class__,)

        self.assertEqual(f2.to_pairs(0),
                ((('i', 'a'), (('a', 'a'), ('b', 'b'), ('c', 'c'))), (('i', 'b'), (('a', 'tru'), ('b', 'fals'), ('c', 'tru'))), (('ii', 'a'), (('a', 'non'), ('b', 'non'), ('c', 'non'))), (('ii', 'b'), (('a', 'non'), ('b', '1'), ('c', '5'))))
                )


    def test_frame_iter_group_a(self) -> None:
        columns = tuple('pqrst')
        index = tuple('zxwy')
        records = (('A', 1, 'a', False, False),
                   ('A', 2, 'b', True, False),
                   ('B', 1, 'c', False, False),
                   ('B', 2, 'd', True, True))

        f = Frame.from_records(
                records, columns=columns, index=index,name='foo')
        f = f.set_index_hierarchy(('p', 'q'), drop=True)
        post = f.iter_group('s').apply(lambda x: x.shape)
        self.assertEqual(post.to_pairs(),
                ((False, (2, 3)), (True, (2, 3)))
                )


    def test_frame_iter_group_b(self) -> None:
        columns = tuple('pqrst')
        index = tuple('zxwy')
        records = (('A', 1, 'a', False, False),
                   ('A', 2, 'b', True, False),
                   ('B', 1, 'c', False, False),
                   ('B', 2, 'd', True, True))

        f = Frame.from_records(
                records, columns=columns, index=index, name='foo')
        post = f.iter_group(['p', 'q']).apply(len)
        self.assertEqual(post.to_pairs(),
                ((('A', 1), 1), (('A', 2), 1), (('B', 1), 1), (('B', 2), 1))
                )


    def test_frame_iter_group_items_a(self) -> None:

        # testing a hierarchical index and columns, selecting column with a tuple

        records = (
                ('a', 999999, 0.1),
                ('a', 201810, 0.1),
                ('b', 999999, 0.4),
                ('b', 201810, 0.4))
        f1 = Frame.from_records(records, columns=list('abc'))

        f1 = f1.set_index_hierarchy(['a', 'b'], drop=False)
        f1 = f1.relabel_add_level(columns='i')

        groups = list(f1.iter_group_items(('i', 'a'), axis=0))
        self.assertEqual(groups[0][0], 'a')
        self.assertEqual(groups[0][1].to_pairs(0),
                ((('i', 'a'), ((('a', 999999), 'a'), (('a', 201810), 'a'))), (('i', 'b'), ((('a', 999999), 999999), (('a', 201810), 201810))), (('i', 'c'), ((('a', 999999), 0.1), (('a', 201810), 0.1)))))

        self.assertEqual(groups[1][0], 'b')
        self.assertEqual(groups[1][1].to_pairs(0),
                ((('i', 'a'), ((('b', 999999), 'b'), (('b', 201810), 'b'))), (('i', 'b'), ((('b', 999999), 999999), (('b', 201810), 201810))), (('i', 'c'), ((('b', 999999), 0.4), (('b', 201810), 0.4)))))


    def test_frame_iter_group_items_b(self) -> None:
        columns = tuple('pqrst')
        index = tuple('zxwy')
        records = (('A', 1, 'a', False, False),
                   ('A', 2, 'b', True, False),
                   ('B', 1, 'c', False, False),
                   ('B', 2, 'd', True, True))

        f = Frame.from_records(
                records, columns=columns, index=index,name='foo')
        f = f.set_index_hierarchy(('p', 'q'), drop=True)
        post = f.iter_group_items('s').apply(
                lambda k, x: f'{k}: {len(x)}')
        self.assertEqual(post.to_pairs(),
                ((False, 'False: 2'), (True, 'True: 2'))
                )


    def test_frame_iter_group_items_c(self) -> None:
        # Test optimized sorting approach. Data must have a non-object dtype and key must be single
        data = np.array([[0, 1, 1, 3],
                         [3, 3, 2, 3],
                         [5, 5, 1, 3],
                         [7, 2, 2, 4]])

        frame = sf.Frame(data, columns=tuple('abcd'), index=tuple('wxyz'))

        # Column
        groups = list(frame.iter_group_items('c', axis=0))
        expected_pairs = [
                (('a', (('w', 0), ('y', 5))),
                 ('b', (('w', 1), ('y', 5))),
                 ('c', (('w', 1), ('y', 1))),
                 ('d', (('w', 3), ('y', 3)))),
                (('a', (('x', 3), ('z', 7))),
                 ('b', (('x', 3), ('z', 2))),
                 ('c', (('x', 2), ('z', 2))),
                 ('d', (('x', 3), ('z', 4))))]

        self.assertEqual([1, 2], [group[0] for group in groups])
        self.assertEqual(expected_pairs, [group[1].to_pairs(axis=0) for group in groups])


        # Index
        groups = list(frame.iter_group_items('w', axis=1))
        expected_pairs = [ # type: ignore
                (('a', (('w', 0), ('x', 3), ('y', 5), ('z', 7))),),
                (('b', (('w', 1), ('x', 3), ('y', 5), ('z', 2))),
                 ('c', (('w', 1), ('x', 2), ('y', 1), ('z', 2)))),
                (('d', (('w', 3), ('x', 3), ('y', 3), ('z', 4))),)]

        self.assertEqual([0, 1, 3], [group[0] for group in groups])
        self.assertEqual(expected_pairs, [group[1].to_pairs(axis=0) for group in groups])


    def test_frame_iter_group_items_d(self) -> None:
        # Test iterating with multiple key selection
        data = np.array([[0, 1, 1, 3],
                         [3, 3, 2, 3],
                         [5, 5, 1, 3],
                         [7, 2, 2, 4]])

        frame = sf.Frame(data, columns=tuple('abcd'), index=tuple('wxyz'))

        # Column
        groups = list(frame.iter_group_items(['c', 'd'], axis=0))
        expected_pairs = [
                (('a', (('w', 0), ('y', 5))),
                 ('b', (('w', 1), ('y', 5))),
                 ('c', (('w', 1), ('y', 1))),
                 ('d', (('w', 3), ('y', 3)))),
                (('a', (('x', 3),)),
                 ('b', (('x', 3),)),
                 ('c', (('x', 2),)),
                 ('d', (('x', 3),))),
                (('a', (('z', 7),)),
                 ('b', (('z', 2),)),
                 ('c', (('z', 2),)),
                 ('d', (('z', 4),)))]

        self.assertEqual([(1, 3), (2, 3), (2, 4)], [group[0] for group in groups])
        self.assertEqual(expected_pairs, [group[1].to_pairs(axis=0) for group in groups])


        # Index
        groups = list(frame.iter_group_items(['x', 'y'], axis=1))
        expected_pairs = [ # type: ignore
                (('c', (('w', 1), ('x', 2), ('y', 1), ('z', 2))),),
                (('d', (('w', 3), ('x', 3), ('y', 3), ('z', 4))),),
                (('a', (('w', 0), ('x', 3), ('y', 5), ('z', 7))),
                 ('b', (('w', 1), ('x', 3), ('y', 5), ('z', 2)))),
        ]

        self.assertEqual([(2, 1), (3, 3), (3, 5)], [group[0] for group in groups])
        self.assertEqual(expected_pairs, [group[1].to_pairs(axis=0) for group in groups])


    def test_frame_iter_group_index_a(self) -> None:

        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x', 'y', 'z'))

        post = tuple(f1.iter_group_index(0, axis=0))

        self.assertEqual(len(post), 3)
        self.assertEqual(
                f1.iter_group_index(0, axis=0).apply(lambda x: x[['p', 'q']].values.sum()).to_pairs(),
                (('x', 4), ('y', 64), ('z', 97))
                )

    def test_frame_iter_group_index_b(self) -> None:

        records = (
                (2, 2, 'a', 'q', False, False),
                (30, 34, 'b', 'c', True, False),
                (2, 95, 'c', 'd', False, False),
                )
        f1 = Frame.from_records(records,
                columns=IndexHierarchy.from_product((1, 2, 3), ('a', 'b')),
                index=('x', 'y', 'z'))

        # with axis 1, we are grouping based on columns while maintain the index
        post_tuple = tuple(f1.iter_group_index(1, axis=1))

        self.assertEqual(len(post_tuple), 2)

        post = f1[HLoc[f1.columns[0]]]
        self.assertEqual(post.__class__, Series)
        self.assertEqual(post.to_pairs(),
            (('x', 2), ('y', 30), ('z', 2))
            )

        post = f1.loc[:, HLoc[f1.columns[0]]]
        self.assertEqual(post.__class__, Series)
        self.assertEqual(post.to_pairs(),
            (('x', 2), ('y', 30), ('z', 2))
            )

        self.assertEqual(
                f1.iter_group_index(1, axis=1).apply(lambda x: x.iloc[:, 0].sum()).to_pairs(),
                (('a', 34), ('b', 131))
                )


    def test_frame_iter_group_index_c(self) -> None:
        columns = tuple('pqrst')
        index = tuple('zxwy')
        records = (('A', 1, 'a', False, False),
                   ('A', 2, 'b', True, False),
                   ('B', 1, 'c', False, False),
                   ('B', 2, 'd', True, True))

        f = Frame.from_records(
                records, columns=columns, index=index,name='foo')
        f = f.set_index_hierarchy(('p', 'q'), drop=True)

        post = f.iter_group_index_items(0).apply(lambda k, x: f'{k}:{x.size}')

        self.assertEqual(post.to_pairs(),
                (('A', 'A:6'), ('B', 'B:6'))
        )

    #---------------------------------------------------------------------------

    def test_frame_reversed(self) -> None:
        columns = tuple('pqrst')
        index = tuple('zxwy')
        records = ((2, 2, 'a', False, False),
                   (30, 34, 'b', True, False),
                   (2, 95, 'c', False, False),
                   (30, 73, 'd', True, True))

        f = Frame.from_records(
                records, columns=columns, index=index,name='foo')

        self.assertTrue(tuple(reversed(f)) == tuple(reversed(columns)))


    def test_frame_sort_index_a(self) -> None:
        # reindex both axis
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('z', 'x', 'w', 'y'),
                name='foo')

        f2 = f1.sort_index()
        self.assertEqual(f2.to_pairs(0),
                (('p', (('w', 2), ('x', 30), ('y', 30), ('z', 2))), ('q', (('w', 95), ('x', 34), ('y', 73), ('z', 2))), ('r', (('w', 'c'), ('x', 'b'), ('y', 'd'), ('z', 'a'))), ('s', (('w', False), ('x', True), ('y', True), ('z', False))), ('t', (('w', False), ('x', False), ('y', True), ('z', False)))))
        self.assertEqual(f1.name, f2.name)

        self.assertEqual(f1.sort_index(ascending=False).to_pairs(0),
                (('p', (('z', 2), ('y', 30), ('x', 30), ('w', 2))), ('q', (('z', 2), ('y', 73), ('x', 34), ('w', 95))), ('r', (('z', 'a'), ('y', 'd'), ('x', 'b'), ('w', 'c'))), ('s', (('z', False), ('y', True), ('x', True), ('w', False))), ('t', (('z', False), ('y', True), ('x', False), ('w', False)))))



    def test_frame_sort_index_b(self) -> None:
        # reindex both axis
        records = (
                ('a', False, False),
                ('b', True, False),
                ('c', False, False),
                ('d', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r'),
                index=IndexHierarchy.from_product((1, 2), (10, 20)),
                )

        post = f1.sort_index(ascending=False)

        self.assertEqual(post.to_pairs(0),
                (('p', (((2, 20), 'd'), ((2, 10), 'c'), ((1, 20), 'b'), ((1, 10), 'a'))), ('q', (((2, 20), True), ((2, 10), False), ((1, 20), True), ((1, 10), False))), ('r', (((2, 20), True), ((2, 10), False), ((1, 20), False), ((1, 10), False))))
                )


    def test_frame_sort_columns_a(self) -> None:
        # reindex both axis
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('t', 's', 'r', 'q', 'p'),
                index=('z', 'x', 'w', 'y'),
                name='foo')

        f2 = f1.sort_columns()
        self.assertEqual(
                f2.to_pairs(0),
                (('p', (('z', False), ('x', False), ('w', False), ('y', True))), ('q', (('z', False), ('x', True), ('w', False), ('y', True))), ('r', (('z', 'a'), ('x', 'b'), ('w', 'c'), ('y', 'd'))), ('s', (('z', 2), ('x', 34), ('w', 95), ('y', 73))), ('t', (('z', 2), ('x', 30), ('w', 2), ('y', 30)))))

        self.assertEqual(f2.name, f1.name)


    def test_frame_sort_columns_b(self) -> None:
        # reindex both axis
        records = (
                (2, 2, False, False),
                (30, 34, True, False),
                (2, 95, False, False),
                (30, 73, True, True),
                )

        f1 = Frame.from_records(records,
                columns=IndexHierarchy.from_product((1, 2), (10, 20)),
                index=('z', 'x', 'w', 'y'),
                )

        f2 = f1.sort_columns(ascending=False)

        self.assertEqual(
            f2.to_pairs(0),
            (((2, 20), (('z', False), ('x', False), ('w', False), ('y', True))), ((2, 10), (('z', False), ('x', True), ('w', False), ('y', True))), ((1, 20), (('z', 2), ('x', 34), ('w', 95), ('y', 73))), ((1, 10), (('z', 2), ('x', 30), ('w', 2), ('y', 30))))
            )


    def test_frame_sort_columns_c(self) -> None:
        # reindex both axis
        records = (
                (2, 2, False, False),
                (30, 34, True, False),
                (2, 95, False, False),
                (30, 73, True, True),
                )

        f1 = Frame.from_records(records,
                columns=IndexYearMonth.from_year_month_range('2018-01', '2018-04'),
                index=('z', 'x', 'w', 'y'),
                )
        f2 = f1.sort_columns(ascending=False)

        self.assertEqual(f2.to_pairs(0),
            ((np.datetime64('2018-04'), (('z', False), ('x', False), ('w', False), ('y', True))), (np.datetime64('2018-03'), (('z', False), ('x', True), ('w', False), ('y', True))), (np.datetime64('2018-02'), (('z', 2), ('x', 34), ('w', 95), ('y', 73))), (np.datetime64('2018-01'), (('z', 2), ('x', 30), ('w', 2), ('y', 30))))
        )

        self.assertEqual(f2.columns.__class__, IndexYearMonth)

    def test_frame_sort_values_a(self) -> None:
        # reindex both axis
        records = (
                (2, 2, 'c', False, False),
                (30, 34, 'd', True, False),
                (2, 95, 'a', False, False),
                (30, 73, 'b', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'r', 'q', 't', 's'),
                index=('z', 'x', 'w', 'y'),
                name='foo')

        post = f1.sort_values('q')
        self.assertEqual(post.name, f1.name)

        self.assertEqual(post.to_pairs(0),
                (('p', (('w', 2), ('y', 30), ('z', 2), ('x', 30))), ('r', (('w', 95), ('y', 73), ('z', 2), ('x', 34))), ('q', (('w', 'a'), ('y', 'b'), ('z', 'c'), ('x', 'd'))), ('t', (('w', False), ('y', True), ('z', False), ('x', True))), ('s', (('w', False), ('y', True), ('z', False), ('x', False)))))


        self.assertEqual(f1.sort_values('p').to_pairs(0),
                (('p', (('z', 2), ('w', 2), ('x', 30), ('y', 30))), ('r', (('z', 2), ('w', 95), ('x', 34), ('y', 73))), ('q', (('z', 'c'), ('w', 'a'), ('x', 'd'), ('y', 'b'))), ('t', (('z', False), ('w', False), ('x', True), ('y', True))), ('s', (('z', False), ('w', False), ('x', False), ('y', True))))
                )


    def test_frame_sort_values_b(self) -> None:
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



    def test_frame_sort_values_c(self) -> None:

        records = (
                (2, 2, 3.5),
                (30, 34, 60.2),
                (2, 95, 1.2),
                (30, 73, 50.2),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r'),
                index=('w', 'x', 'y', 'z'),
                name='foo')

        f2 = f1.sort_values('y', axis=0)
        self.assertEqual(f2.to_pairs(0),
                (('r', (('w', 3.5), ('x', 60.2), ('y', 1.2), ('z', 50.2))), ('p', (('w', 2), ('x', 30), ('y', 2), ('z', 30))), ('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73)))))

        self.assertEqual(f2.name, 'foo')



    def test_frame_sort_values_d(self) -> None:

        a1 = np.arange(8).reshape(4, 2) / 10
        match = (('a', ((0.6, 0.6), (0.4, 0.4), (0.2, 0.2), (0.0, 0.0))), ('b', ((0.6, 0.7), (0.4, 0.5), (0.2, 0.3), (0.0, 0.1))))

        f1 = Frame(a1, columns=('a', 'b'))
        f1 = f1.set_index('a')
        f1 = f1.sort_values('b', ascending=False)
        self.assertEqual(f1.to_pairs(0), match)

        f2 = FrameGO(a1, columns=('a', 'b'))
        f2 = f2.set_index('a') # type: ignore
        f2 = f2.sort_values('b', ascending=False) # type: ignore
        self.assertEqual(f2.to_pairs(0), match)


    def test_frame_sort_values_e(self) -> None:
        # Ensure index sorting works on internally homogenous frames
        data = np.array([[3, 7, 3],
                         [8, 1, 4],
                         [2, 9, 6]])
        f1 = sf.Frame(data, columns=tuple('abc'), index=tuple('xyz'))
        assert len(f1._blocks._blocks) == 1, 'f1 must be internally homogenous.'

        f1_sorted = f1.sort_values('x', axis=0)

        expected1 = (('x', (('a', 3), ('c', 3), ('b', 7))),
                     ('y', (('a', 8), ('c', 4), ('b', 1))),
                     ('z', (('a', 2), ('c', 6), ('b', 9))))
        self.assertEqual(expected1, f1_sorted.to_pairs(axis=1))


        # Ensure index sorting works on internally heterogeneous frames
        records = ((4, 2, 3), (2, 3.1, False), (6, False, 3.4))
        f2 = sf.Frame.from_records(records,
                columns=tuple('abc'),
                index=tuple('xyz'),
                dtypes=(object, object, object))

        assert len(f2._blocks._blocks) > 1, 'f2 must be internally heterogeneous.'
        f2_sorted = f2.sort_values('x', axis=0)

        expected2 = (('x', (('b', 2), ('c', 3), ('a', 4))),
                     ('y', (('b', 3.1), ('c', False), ('a', 2))),
                     ('z', (('b', False), ('c', 3.4), ('a', 6))))
        self.assertEqual(expected2, f2_sorted.to_pairs(axis=1))


    def test_frame_relabel_a(self) -> None:
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


    def test_frame_relabel_b(self) -> None:
        # reindex both axis
        records = (
                (2, 2, 'c', False),
                (30, 34, 'd', True),
                )

        f1 = FrameGO.from_records(records,
                columns=('p', 'r', 'q', 't'),
                index=('x', 'y')
                )

        f2 = f1.relabel(columns=IndexAutoFactory)
        self.assertEqual(f2.columns.values.tolist(), [0, 1, 2, 3])


    def test_frame_relabel_c(self) -> None:

        records = (
                (1, 2, 'a', False),
                (30, 34, 'b', True),
                )
        f1 = Frame.from_records(records,
                columns=tuple('pqrs'),
                index=tuple('ab'),
                name='foo')

        f2 = f1.relabel(index=IndexAutoFactory)
        self.assertEqual(f2.to_pairs(0),
            (('p', ((0, 1), (1, 30))), ('q', ((0, 2), (1, 34))), ('r', ((0, 'a'), (1, 'b'))), ('s', ((0, False), (1, True))))
            )

        f3 = f1.relabel(columns=IndexAutoFactory)
        self.assertEqual(
            f3.relabel(columns=IndexAutoFactory).to_pairs(0),
            ((0, (('a', 1), ('b', 30))), (1, (('a', 2), ('b', 34))), (2, (('a', 'a'), ('b', 'b'))), (3, (('a', False), ('b', True))))
            )
        self.assertTrue(f3.columns.STATIC)


    def test_frame_relabel_d(self) -> None:

        records = (
                (1, 2, 'a', False),
                (30, 34, 'b', True),
                )
        f1 = FrameGO.from_records(records,
                columns=tuple('pqrs'),
                index=tuple('ab'),
                name='foo')

        f2 = f1.relabel(columns=IndexAutoFactory)
        self.assertEqual(
            f2.relabel(columns=IndexAutoFactory).to_pairs(0),
            ((0, (('a', 1), ('b', 30))), (1, (('a', 2), ('b', 34))), (2, (('a', 'a'), ('b', 'b'))), (3, (('a', False), ('b', True))))
            )
        self.assertFalse(f2.columns.STATIC)
        f2[4] = None
        self.assertTrue(f2.columns._loc_is_iloc)
        f2[6] = None
        self.assertFalse(f2.columns._loc_is_iloc)

        self.assertEqual(f2.to_pairs(0),
                ((0, (('a', 1), ('b', 30))), (1, (('a', 2), ('b', 34))), (2, (('a', 'a'), ('b', 'b'))), (3, (('a', False), ('b', True))), (4, (('a', None), ('b', None))), (6, (('a', None), ('b', None))))
                )


    def test_frame_rehierarch_a(self) -> None:
        records = (
                (1, 2, 'a', False),
                (30, 34, 'b', True),
                (54, 95, 'c', False),
                (65, 73, 'd', True),
                )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        index = IndexHierarchy.from_product((100, 200), (True, False))
        f1 = Frame.from_records(records,
                columns=columns,
                index=index)

        f2 = f1.rehierarch(index=(1,0), columns=(1,0))
        self.assertEqual(f2.to_pairs(0),
                (((1, 'a'), (((True, 100), 1), ((True, 200), 54), ((False, 100), 30), ((False, 200), 65))), ((1, 'b'), (((True, 100), 'a'), ((True, 200), 'c'), ((False, 100), 'b'), ((False, 200), 'd'))), ((2, 'a'), (((True, 100), 2), ((True, 200), 95), ((False, 100), 34), ((False, 200), 73))), ((2, 'b'), (((True, 100), False), ((True, 200), False), ((False, 100), True), ((False, 200), True))))
                )

    def test_frame_rehierarch_b(self) -> None:

        records = (
                (1, 2, 'a', False),
                (30, 34, 'b', True),
                )
        f1 = FrameGO.from_records(records,
                columns=tuple('pqrs'),
                index=tuple('ab'),
                name='foo')

        # no hierarchy fails
        with self.assertRaises(RuntimeError):
            f1.rehierarch(index=(0, 1))

        with self.assertRaises(RuntimeError):
            f1.rehierarch(columns=(0, 1))



    #---------------------------------------------------------------------------
    def test_frame_get_a(self) -> None:
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

    def test_frame_isna_a(self) -> None:
        f1 = FrameGO([
                [np.nan, 2, np.nan, 0],
                [3, 4, np.nan, 1],
                [np.nan, np.nan, np.nan, 5]],
                columns=list('ABCD'))

        self.assertEqual(f1.isna().to_pairs(0),
                (('A', ((0, True), (1, False), (2, True))), ('B', ((0, False), (1, False), (2, True))), ('C', ((0, True), (1, True), (2, True))), ('D', ((0, False), (1, False), (2, False)))))

        self.assertEqual(f1.notna().to_pairs(0),
                (('A', ((0, False), (1, True), (2, False))), ('B', ((0, True), (1, True), (2, False))), ('C', ((0, False), (1, False), (2, False))), ('D', ((0, True), (1, True), (2, True)))))

    def test_frame_dropna_a(self) -> None:
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
        self.assertEqual(f3.shape, (3, 0))

    def test_frame_dropna_b(self) -> None:
        f1 = FrameGO([
                [np.nan, 2, 3, 0],
                [3, 4, np.nan, 1],
                [0, 1, 2, 3]],
                columns=list('ABCD'))

        self.assertEqual(f1.dropna(axis=0, condition=np.any).to_pairs(0),
                (('A', ((2, 0.0),)), ('B', ((2, 1.0),)), ('C', ((2, 2.0),)), ('D', ((2, 3.0),))))
        self.assertEqual(f1.dropna(axis=1, condition=np.any).to_pairs(0),
                (('B', ((0, 2.0), (1, 4.0), (2, 1.0))), ('D', ((0, 0.0), (1, 1.0), (2, 3.0)))))

    def test_frame_dropna_c(self) -> None:
        f1 = Frame([
                [np.nan, np.nan],
                [np.nan, np.nan],],
                columns=list('AB'))
        f2 = f1.dropna()
        self.assertEqual(f2.shape, (0, 2))



    def test_frame_fillna_a(self) -> None:
        dtype = np.dtype

        f1 = FrameGO([
                [np.nan, 2, 3, 0],
                [3, 4, np.nan, 1],
                [0, 1, 2, 3]],
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



    def test_frame_fillna_leading_a(self) -> None:
        a2 = np.array([
                [None, None, None, None],
                [None, 1, None, 6],
                [None, 5, None, None]
                ], dtype=object)
        a1 = np.array([None, None, None], dtype=object)
        a3 = np.array([
                [None, 4],
                [None, 1],
                [None, 5]
                ], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(tb1,
                index=self.get_letters(None, tb1.shape[0]),
                columns=self.get_letters(-tb1.shape[1], None)
                )

        self.assertEqual(f1.fillna_leading(0, axis=0).to_pairs(0),
                (('t', (('a', 0), ('b', 0), ('c', 0))), ('u', (('a', 0), ('b', 0), ('c', 0))), ('v', (('a', 0), ('b', 1), ('c', 5))), ('w', (('a', 0), ('b', 0), ('c', 0))), ('x', (('a', 0), ('b', 6), ('c', None))), ('y', (('a', 0), ('b', 0), ('c', 0))), ('z', (('a', 4), ('b', 1), ('c', 5)))))

        self.assertEqual(f1.fillna_leading(0, axis=1).to_pairs(0),
                (('t', (('a', 0), ('b', 0), ('c', 0))), ('u', (('a', 0), ('b', 0), ('c', 0))), ('v', (('a', 0), ('b', 1), ('c', 5))), ('w', (('a', 0), ('b', None), ('c', None))), ('x', (('a', 0), ('b', 6), ('c', None))), ('y', (('a', 0), ('b', None), ('c', None))), ('z', (('a', 4), ('b', 1), ('c', 5)))))





    def test_frame_fillna_forward_a(self) -> None:
        a2 = np.array([
                [8, None, None, None],
                [None, 1, None, 6],
                [0, 5, None, None]
                ], dtype=object)
        a1 = np.array([None, 3, None], dtype=object)
        a3 = np.array([
                [None, 4],
                [None, 1],
                [None, 5]
                ], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(tb1,
                index=self.get_letters(None, tb1.shape[0]),
                columns=self.get_letters(-tb1.shape[1], None)
                )

        self.assertEqual(
                f1.fillna_forward().to_pairs(0),
                (('t', (('a', None), ('b', 3), ('c', 3))), ('u', (('a', 8), ('b', 8), ('c', 0))), ('v', (('a', None), ('b', 1), ('c', 5))), ('w', (('a', None), ('b', None), ('c', None))), ('x', (('a', None), ('b', 6), ('c', 6))), ('y', (('a', None), ('b', None), ('c', None))), ('z', (('a', 4), ('b', 1), ('c', 5))))
                )

        self.assertEqual(
                f1.fillna_backward().to_pairs(0),
                (('t', (('a', 3), ('b', 3), ('c', None))), ('u', (('a', 8), ('b', 0), ('c', 0))), ('v', (('a', 1), ('b', 1), ('c', 5))), ('w', (('a', None), ('b', None), ('c', None))), ('x', (('a', 6), ('b', 6), ('c', None))), ('y', (('a', None), ('b', None), ('c', None))), ('z', (('a', 4), ('b', 1), ('c', 5))))
                )



    def test_frame_fillna_forward_b(self) -> None:
        a2 = np.array([
                [8, None, None, None],
                [None, 1, None, 6],
                [0, 5, None, None]
                ], dtype=object)
        a1 = np.array([None, 3, None], dtype=object)
        a3 = np.array([
                [None, 4],
                [None, 1],
                [None, 5]
                ], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(tb1,
                index=self.get_letters(None, tb1.shape[0]),
                columns=self.get_letters(-tb1.shape[1], None)
                )
        # axis 1 tests
        self.assertEqual(
                f1.fillna_forward(axis=1).to_pairs(0),
                (('t', (('a', None), ('b', 3), ('c', None))), ('u', (('a', 8), ('b', 3), ('c', 0))), ('v', (('a', 8), ('b', 1), ('c', 5))), ('w', (('a', 8), ('b', 1), ('c', 5))), ('x', (('a', 8), ('b', 6), ('c', 5))), ('y', (('a', 8), ('b', 6), ('c', 5))), ('z', (('a', 4), ('b', 1), ('c', 5))))
                )

        self.assertEqual(
                f1.fillna_backward(axis=1).to_pairs(0),
                (('t', (('a', 8), ('b', 3), ('c', 0))), ('u', (('a', 8), ('b', 1), ('c', 0))), ('v', (('a', 4), ('b', 1), ('c', 5))), ('w', (('a', 4), ('b', 6), ('c', 5))), ('x', (('a', 4), ('b', 6), ('c', 5))), ('y', (('a', 4), ('b', 1), ('c', 5))), ('z', (('a', 4), ('b', 1), ('c', 5))))
                )

    def test_frame_fillna_forward_c(self) -> None:
        a2 = np.array([
                [8, None, None, None],
                [None, 1, None, 6],
                [0, 5, None, None]
                ], dtype=object)
        a1 = np.array([None, 3, None], dtype=object)
        a3 = np.array([
                [None, 4],
                [None, None],
                [None, 5]
                ], dtype=object)
        tb1 = TypeBlocks.from_blocks((a1, a2, a3))

        f1 = Frame(tb1,
                index=self.get_letters(None, tb1.shape[0]),
                columns=self.get_letters(-tb1.shape[1], None)
                )
        post = f1.fillna_forward(axis=1)

        self.assertEqual(f1.fillna_forward(axis=1, limit=1).to_pairs(0),
                (('t', (('a', None), ('b', 3), ('c', None))), ('u', (('a', 8), ('b', 3), ('c', 0))), ('v', (('a', 8), ('b', 1), ('c', 5))), ('w', (('a', None), ('b', 1), ('c', 5))), ('x', (('a', None), ('b', 6), ('c', None))), ('y', (('a', None), ('b', 6), ('c', None))), ('z', (('a', 4), ('b', None), ('c', 5))))
                )

        self.assertEqual(f1.fillna_forward(axis=1, limit=2).to_pairs(0),
                (('t', (('a', None), ('b', 3), ('c', None))), ('u', (('a', 8), ('b', 3), ('c', 0))), ('v', (('a', 8), ('b', 1), ('c', 5))), ('w', (('a', 8), ('b', 1), ('c', 5))), ('x', (('a', None), ('b', 6), ('c', 5))), ('y', (('a', None), ('b', 6), ('c', None))), ('z', (('a', 4), ('b', 6), ('c', 5))))
                )

        self.assertEqual(f1.fillna_backward(axis=1, limit=2).to_pairs(0),
                (('t', (('a', 8), ('b', 3), ('c', 0))), ('u', (('a', 8), ('b', 1), ('c', 0))), ('v', (('a', None), ('b', 1), ('c', 5))), ('w', (('a', None), ('b', 6), ('c', None))), ('x', (('a', 4), ('b', 6), ('c', 5))), ('y', (('a', 4), ('b', None), ('c', 5))), ('z', (('a', 4), ('b', None), ('c', 5))))
                )


    def test_frame_empty_a(self) -> None:

        f1 = FrameGO(index=('a', 'b', 'c'))
        f1['w'] = Series.from_items(zip('cebga', (10, 20, 30, 40, 50)))
        f1['x'] = Series.from_items(zip('abc', range(3, 6)))
        f1['y'] = Series.from_items(zip('abcd', range(2, 6)))
        f1['z'] = Series.from_items(zip('qabc', range(7, 11)))

        self.assertEqual(f1.to_pairs(0),
                (('w', (('a', 50), ('b', 30), ('c', 10))), ('x', (('a', 3), ('b', 4), ('c', 5))), ('y', (('a', 2), ('b', 3), ('c', 4))), ('z', (('a', 8), ('b', 9), ('c', 10)))))


    #---------------------------------------------------------------------------
    @skip_win  # type: ignore
    def test_frame_from_csv_a(self) -> None:
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
        self.assertEqual(f2['count':].sum().to_pairs(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
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



    def test_frame_from_csv_b(self) -> None:
        filelike = StringIO('''count,number,weight,scalar,color,active
0,4,234.5,5.3,'red',False
30,50,9.234,5.434,'blue',True''')
        f1 = Frame.from_csv(filelike)

        self.assertEqual(f1.columns.values.tolist(),
                ['count', 'number', 'weight', 'scalar', 'color', 'active'])


    def test_frame_from_csv_c(self) -> None:
        s1 = StringIO('color,count,score\nred,1,1.3\ngreen,3,5.2\nblue,100,3.4\nblack,4,9.0')
        f1 = Frame.from_csv(s1, index_depth=1)
        self.assertEqual(f1.to_pairs(0),
                (('count', (('red', 1), ('green', 3), ('blue', 100), ('black', 4))), ('score', (('red', 1.3), ('green', 5.2), ('blue', 3.4), ('black', 9.0)))))


    def test_frame_from_csv_d(self) -> None:
        s1 = StringIO('color,count,score\n')
        f1 = Frame.from_csv(s1, columns_depth=1)
        self.assertEqual(f1.to_pairs(0),
            (('color', ()), ('count', ()), ('score', ()))
            )

    def test_frame_from_csv_e(self) -> None:
        s1 = StringIO('group,count,score,color\nA,1,1.3,red\nA,3,5.2,green\nB,100,3.4,blue\nB,4,9.0,black')

        f1 = sf.Frame.from_csv(
                s1,
                index_depth=2,
                columns_depth=1)
        self.assertEqual(f1.index.__class__, IndexHierarchy)
        self.assertEqual(f1.to_pairs(0),
                (('score', ((('A', 1), 1.3), (('A', 3), 5.2), (('B', 100), 3.4), (('B', 4), 9.0))), ('color', ((('A', 1), 'red'), (('A', 3), 'green'), (('B', 100), 'blue'), (('B', 4), 'black')))))


    def test_frame_from_csv_f(self) -> None:
        s1 = StringIO('group,count,score,color\nA,nan,1.3,red\nB,NaN,5.2,green\nC,NULL,3.4,blue\nD,,9.0,black')

        f1 = sf.Frame.from_csv(
                s1,
                index_depth=1,
                columns_depth=1)

        self.assertAlmostEqualFramePairs(f1.to_pairs(0),
                (('count', (('A', np.nan), ('B', np.nan), ('C', np.nan), ('D', np.nan))), ('score', (('A', 1.3), ('B', 5.2), ('C', 3.4), ('D', 9.0))), ('color', (('A', 'red'), ('B', 'green'), ('C', 'blue'), ('D', 'black'))))
                )


    def test_frame_from_csv_g(self) -> None:
        filelike = StringIO('''0,4,234.5,5.3,'red',False
30,50,9.234,5.434,'blue',True''')
        f1 = Frame.from_csv(filelike, columns_depth=0)
        self.assertEqual(f1.to_pairs(0),
            ((0, ((0, 0), (1, 30))), (1, ((0, 4), (1, 50))), (2, ((0, 234.5), (1, 9.234))), (3, ((0, 5.3), (1, 5.434))), (4, ((0, "'red'"), (1, "'blue'"))), (5, ((0, False), (1, True))))
            )

    def test_frame_from_csv_h(self) -> None:
        s1 = StringIO('group,count,score,color\nA,nan,1.3,red\nB,NaN,5.2,green\nC,NULL,3.4,blue\nD,,9.0,black')

        f1 = sf.Frame.from_csv(
                s1,
                index_depth=1,
                columns_depth=1,
                dtypes=dict(score=np.float16))

        self.assertEqual(f1.dtypes.to_pairs(),
                (('count', np.dtype('O')),
                ('score', np.dtype('float16')),
                ('color', np.dtype('<U5'))))

    @skip_win  # type: ignore
    def test_frame_from_csv_i(self) -> None:
        s1 = StringIO('1,2,3\n4,5,6')

        f1 = sf.Frame.from_csv(
                s1,
                index_depth=0,
                columns_depth=0,
                dtypes=[np.int64, str, np.int64]
                )

        self.assertEqual(f1.dtypes.values.tolist(),
                [np.dtype('int64'), np.dtype('<U21'), np.dtype('int64')]
                )

    def test_frame_from_csv_j(self) -> None:
        s1 = StringIO('1,2,3\n4,5,6')

        f2 = sf.Frame.from_csv(
                s1,
                index_depth=2,
                columns_depth=0,
                dtypes=[np.int64, str, np.int64]
                )

        self.assertEqual(f2.to_pairs(0,),
                ((0, (((1, '2'), 3), ((4, '5'), 6))),)
                )

    def test_frame_from_csv_k(self) -> None:
        s1 = StringIO('1\t2\t3\t4\n')
        f1 = Frame.from_csv(s1, index_depth=0, columns_depth=0)
        self.assertEqual(f1.to_pairs(0),
                ((0, ((0, 1),)), (1, ((0, 2),)), (2, ((0, 3),)), (3, ((0, 4),)))
                )


    @skip_win  # type: ignore
    def test_structured_array_to_blocks_and_index_a(self) -> None:

        a1 = np.array(np.arange(12).reshape((3, 4)))
        post, _, _ = Frame._structured_array_to_d_ia_cl(
                a1,
                dtypes=[np.int64, str, np.int64, str]
                )

        self.assertEqual(post.dtypes.tolist(),
                [np.dtype('int64'), np.dtype('<U21'), np.dtype('int64'), np.dtype('<U21')]
                )


    #---------------------------------------------------------------------------

    def test_frame_from_tsv_a(self) -> None:

        with temp_file('.txt', path=True) as fp:

            with open(fp, 'w') as file:
                file.write('\n'.join(('index\tA\tB', 'a\tTrue\t20.2', 'b\tFalse\t85.3')))
                file.close()

            f = Frame.from_tsv(fp, index_depth=1, dtypes={'a': bool})
            self.assertEqual(
                    f.to_pairs(0),
                    (('A', (('a', True), ('b', False))), ('B', (('a', 20.2), ('b', 85.3))))
                    )


    def test_frame_from_tsv_b(self) -> None:
        # a generator of delimited strings also works

        def lines() -> tp.Iterator[str]:
            yield 'a\tb\tc\td'
            for i in range(4):
                yield f'{i}\t{i + 1}\t{i + 2}\t{i + 3}'

        f = Frame.from_tsv(lines())
        self.assertEqual(f.to_pairs(0),
                (('a', ((0, 0), (1, 1), (2, 2), (3, 3))), ('b', ((0, 1), (1, 2), (2, 3), (3, 4))), ('c', ((0, 2), (1, 3), (2, 4), (3, 5))), ('d', ((0, 3), (1, 4), (2, 5), (3, 6))))
                )

    def test_frame_from_tsv_c(self) -> None:
        input_stream = StringIO('''
        196412	0.0
        196501	0.0
        196502	0.0
        196503	0.0
        196504	0.0
        196505	0.0''')


        f1 = sf.Frame.from_tsv(
                input_stream,
                index_depth=1,
                columns_depth=0)

        self.assertEqual(f1.to_pairs(0),
                ((0, ((196412, 0.0), (196501, 0.0), (196502, 0.0), (196503, 0.0), (196504, 0.0), (196505, 0.0))),))

        input_stream = StringIO('''
        196412	0.0	0.1
        196501	0.0	0.1
        196502	0.0	0.1
        196503	0.0	0.1
        196504	0.0	0.1
        196505	0.0	0.1''')


        f2 = sf.Frame.from_tsv(
                input_stream,
                index_depth=1,
                columns_depth=0)

        self.assertEqual(f2.to_pairs(0),
                ((0, ((196412, 0.0), (196501, 0.0), (196502, 0.0), (196503, 0.0), (196504, 0.0), (196505, 0.0))), (1, ((196412, 0.1), (196501, 0.1), (196502, 0.1), (196503, 0.1), (196504, 0.1), (196505, 0.1)))))



    def test_frame_from_tsv_d(self) -> None:

        f1 = sf.Frame([1], columns=['a'])

        with temp_file('.txt', path=True) as fp:
            f1.to_tsv(fp)
            f2 = sf.Frame.from_tsv(fp, index_depth=1)
            self.assertEqual(f2.to_pairs(0),
                    (('a', ((0, 1),)),))


    def test_frame_from_tsv_e(self) -> None:

        f1 = sf.Frame([1], columns=['with space'])

        with temp_file('.txt', path=True) as fp:
            f1.to_tsv(fp)
            f2 = sf.Frame.from_tsv(fp, index_depth=1)
            self.assertEqual(
                    f2.columns.values.tolist(),
                    ['with space']
                    )

    def test_frame_from_tsv_f(self) -> None:

        f1 = sf.Frame([1], columns=[':with:colon:'])

        with temp_file('.txt', path=True) as fp:
            f1.to_tsv(fp)
            f2 = sf.Frame.from_tsv(fp, index_depth=1)
            self.assertEqual(f2.to_pairs(0),
                    ((':with:colon:', ((0, 1),)),)
                    )

    #---------------------------------------------------------------------------
    def test_frame_to_csv_a(self) -> None:
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
'__index0__,p,q,r,s,t\nw,2,2,a,False,False\nx,30,34,b,True,False\ny,2,95,c,False,False\nz,30,73,d,True,True')

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


    def test_frame_to_csv_b(self) -> None:

        f = sf.Frame([1, 2, 3],
                columns=['a'],
                index=sf.Index(range(3), name='Important Name'))
        file = StringIO()
        f.to_csv(file)
        file.seek(0)
        self.assertEqual(file.read(), 'Important Name,a\n0,1\n1,2\n2,3')


    def test_frame_to_csv_c(self) -> None:
        records = (
                (2, np.nan, 'a', False, None),
                (30, np.nan, 'b', True, None),
                (2, np.inf, 'c', False, None),
                (30, -np.inf, 'd', True, None),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        with temp_file('.csv') as fp:
            f1.to_csv(fp)

            with open(fp) as f:
                lines = f.readlines()
                # nan has been converted to string
                self.assertEqual(lines[1], 'w,2,,a,False,None\n')
                self.assertEqual(lines[4], 'z,30,-inf,d,True,None')


    def test_frame_to_csv_d(self) -> None:
        f1 = Frame.from_records(
                ((10, 20, 50, 60), (50.0, 60.4, -50, -60)),
                index=('p', 'q'),
                columns=IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
                name='f3')

        with temp_file('.csv') as fp:
            f1.to_csv(fp)

            with open(fp) as f:
                lines = f.readlines()

            self.assertEqual(lines,
                    ['__index0__,I,I,II,II\n', ',a,b,a,b\n', 'p,10,20,50,60\n', 'q,50,60,-50,-60']
                    )

            f2 = Frame.from_csv(fp, columns_depth=2, index_depth=1)
            self.assertEqual(f2.to_pairs(0),
                    ((('I', 'a'), (('p', 10), ('q', 50))), (('I', 'b'), (('p', 20), ('q', 60))), (('II', 'a'), (('p', 50), ('q', -50))), (('II', 'b'), (('p', 60), ('q', -60))))
                    )


    def test_frame_to_csv_e(self) -> None:
        f1 = Frame.from_records(
                ((10, 20, 50, 60), (50.0, 60.4, -50, -60)),
                index=('p', 'q'),
                columns=IndexHierarchy.from_product((10, 20),('I', 'II'),),
                name='f3')

        with temp_file('.csv') as fp:
            f1.to_csv(fp)

            with open(fp) as f:
                lines = f.readlines()

            self.assertEqual(lines,
                    ['__index0__,10,10,20,20\n', ',I,II,I,II\n', 'p,10,20,50,60\n', 'q,50,60,-50,-60']
                    )
            f2 = Frame.from_csv(fp, columns_depth=2, index_depth=1)
            self.assertEqual(
                    f2.to_pairs(0),
                    (((10, 'I'), (('p', 10), ('q', 50))), ((10, 'II'), (('p', 20), ('q', 60))), ((20, 'I'), (('p', 50), ('q', -50))), ((20, 'II'), (('p', 60), ('q', -60))))
                    )


    #---------------------------------------------------------------------------
    def test_frame_to_tsv_a(self) -> None:
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
'__index0__\tp\tq\tr\ts\tt\nw\t2\t2\ta\tFalse\tFalse\nx\t30\t34\tb\tTrue\tFalse\ny\t2\t95\tc\tFalse\tFalse\nz\t30\t73\td\tTrue\tTrue')


    def test_frame_to_tsv_b(self) -> None:
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=IndexHierarchy.from_product(('A', 'B'), (1, 2))
                )

        with temp_file('.txt', path=True) as fp:
            f1.to_tsv(fp, include_index=True)
            f2 = Frame.from_tsv(fp, index_depth=2)
            self.assertEqualFrames(f1, f2)

    def test_frame_to_tsv_c(self) -> None:
        f1 = sf.Frame(
                np.arange(16).reshape((4,4)),
                index=sf.IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
                columns=sf.IndexHierarchy.from_product(('III', 'IV'), (10, 20))
                )

        with temp_file('.txt', path=True) as fp:
            f1.to_tsv(fp, include_index=True)
            f2 = Frame.from_tsv(fp, index_depth=2, columns_depth=2)
            self.assertEqualFrames(f1, f2)


    #---------------------------------------------------------------------------
    def test_frame_to_html_a(self) -> None:
        records = (
                (2, 'a', False),
                (3, 'b', False),
                )
        f1 = Frame.from_records(records,
                columns=('r', 's', 't'),
                index=('w', 'x'))
        post = f1.to_html()
        self.assertEqual(post, '<table border="1"><thead><tr><th></th><th>r</th><th>s</th><th>t</th></tr></thead><tbody><tr><th>w</th><td>2</td><td>a</td><td>False</td></tr><tr><th>x</th><td>3</td><td>b</td><td>False</td></tr></tbody></table>'
        )


    def test_frame_to_html_datatables_a(self) -> None:
        records = (
                (2, 'a', False),
                (3, 'b', False),
                )
        f1 = Frame.from_records(records,
                columns=('r', 's', 't'),
                index=('w', 'x'))

        sio = StringIO()

        post = f1.to_html_datatables(sio, show=False)

        self.assertEqual(post, None)
        self.assertTrue(len(sio.read()) > 1300)


    #---------------------------------------------------------------------------

    def test_frame_to_rst_a(self) -> None:
        records = (
                (2, 'a', False),
                (3, 'b', False),
                )
        f1 = Frame.from_records(records,
                columns=('r', 's', 't'),
                index=('w', 'x'))
        post = f1.to_rst()
        msg = '''
                +--+--+--+-----+
                |  |r |s |t    |
                +==+==+==+=====+
                |w |2 |a |False|
                +--+--+--+-----+
                |x |3 |b |False|
                +--+--+--+-----+
                '''
        self.assertEqualLines(post, msg)


    def test_frame_to_markdown_a(self) -> None:
        records = (
                (2, 'a', False),
                (3, 'b', False),
                )
        f1 = Frame.from_records(records,
                columns=('r', 's', 't'),
                index=('w', 'x'))
        post = f1.to_markdown()

        msg = '''
                |  |r |s |t    |
                |--|--|--|-----|
                |w |2 |a |False|
                |x |3 |b |False|
                '''
        self.assertEqualLines(post, msg)


    def test_frame_to_latex_a(self) -> None:
        records = (
                (2, 'a', False),
                (3, 'b', False),
                )
        f1 = Frame.from_records(records,
                columns=('r', 's', 't'),
                index=('w', 'x'))
        post = f1.to_latex()
        msg = r'''
                \begin{table}[ht]
                \centering
                \begin{tabular}{c c c c}
                \hline\hline
                   & r  & s  & t     \\
                \hline
                w  & 2  & a  & False \\
                x  & 3  & b  & False \\
                \hline\end{tabular}
                \end{table}
                '''
        self.assertEqualLines(post, msg)



    #---------------------------------------------------------------------------

    def test_frame_to_xlsx_a(self) -> None:
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        config = StoreConfig(index_depth=1)

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp)
            st = StoreXLSX(fp)
            f2 = st.read(label=None, config=config)
            self.assertEqualFrames(f1, f2)



    def test_frame_from_xlsx_a(self) -> None:
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp)
            f2 = Frame.from_xlsx(fp, index_depth=f1.index.depth)
            self.assertEqualFrames(f1, f2)


    def test_frame_from_xlsx_b(self) -> None:

        f1 = Frame.from_records((
                (10, 20, 50, False, 10, 20, 50, False),
                (50.0, 60.4, -50, True, 50.0, 60.4, -50, True),
                (234, 44452, 0, False, 234, 44452, 0, False),
                (4, -4, 2000, True, 4, -4, 2000, True),
                (10, 20, 50, False, 10, 20, 50, False),
                (50.0, 60.4, -50, True, 50.0, 60.4, -50, True),
                (234, 44452, 0, False, 234, 44452, 0, False),
                (4, -4, 2000, True, 4, -4, 2000, True),
                ),
                index=IndexHierarchy.from_product(('top', 'bottom'), ('far', 'near'), ('left', 'right')),
                columns=IndexHierarchy.from_product(('I', 'II'), ('a', 'b'), (1, 2))
                )

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp)
            f2 = Frame.from_xlsx(fp,
                    index_depth=f1.index.depth,
                    columns_depth=f1.columns.depth)
            self.assertEqualFrames(f1, f2)

    @unittest.skip('need to progrmatically generate bad_sheet.xlsx')
    def test_frame_from_xlsx_c(self) -> None:
        # https://github.com/InvestmentSystems/static-frame/issues/146
        fp = '/tmp/bad_sheet.xlsx'
        f2 = Frame.from_xlsx(fp)
        self.assertEqual(f2.shape, (5, 6))

    def test_frame_from_xlsx_d(self) -> None:
        # isolate case of all None data that has a valid index

        f1 = Frame(None, index=('a', 'b', 'c'), columns=('x', 'y', 'z'))

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp)
            f2 = Frame.from_xlsx(fp,
                    index_depth=f1.index.depth,
                    columns_depth=f1.columns.depth)
            self.assertEqualFrames(f1, f2)

    def test_frame_from_xlsx_e(self) -> None:
        # isolate case of all None data that has a valid IndexHierarchy

        f1 = Frame(None,
                index=IndexHierarchy.from_product((0, 1), ('a', 'b')),
                columns=('x', 'y', 'z')
                )

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp)
            f2 = Frame.from_xlsx(fp,
                    index_depth=f1.index.depth,
                    columns_depth=f1.columns.depth)
            self.assertEqualFrames(f1, f2)

    def test_frame_from_xlsx_f(self) -> None:
        # isolate case of all None data and only columns
        f1 = Frame(None, index=('a', 'b', 'c'), columns=('x', 'y', 'z'))

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp, include_index=False)
            f2 = Frame.from_xlsx(fp,
                    index_depth=0,
                    columns_depth=f1.columns.depth)
            # with out the index, we only have columns, and drop all-empty rows
            self.assertEqual(f2.shape, (0, 3))

    def test_frame_from_xlsx_g(self) -> None:
        # isolate case of all None data, no index, no columns
        f1 = Frame(None, index=('a', 'b', 'c'), columns=('x', 'y', 'z'))

        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp, include_index=False, include_columns=False)
            with self.assertRaises(ErrorInitFrame):
                f2 = Frame.from_xlsx(fp,
                        index_depth=0,
                        columns_depth=0)

    #---------------------------------------------------------------------------
    def test_frame_from_sqlite_a(self) -> None:
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        with temp_file('.sqlite') as fp:
            f1.to_sqlite(fp)
            f2 = Frame.from_sqlite(fp, index_depth=f1.index.depth)
            self.assertEqualFrames(f1, f2)


    def test_frame_from_sqlite_b(self) -> None:

        f1 = Frame.from_records((
                (10, 20, 50, False, 10, 20, 50, False),
                (50.0, 60.4, -50, True, 50.0, 60.4, -50, True),
                (234, 44452, 0, False, 234, 44452, 0, False),
                (4, -4, 2000, True, 4, -4, 2000, True),
                (10, 20, 50, False, 10, 20, 50, False),
                (50.0, 60.4, -50, True, 50.0, 60.4, -50, True),
                (234, 44452, 0, False, 234, 44452, 0, False),
                (4, -4, 2000, True, 4, -4, 2000, True),
                ),
                index=IndexHierarchy.from_product(('top', 'bottom'), ('far', 'near'), ('left', 'right')),
                columns=IndexHierarchy.from_product(('I', 'II'), ('a', 'b'), (1, 2))
                )

        with temp_file('.sqlite') as fp:
            f1.to_sqlite(fp)
            f2 = Frame.from_sqlite(fp,
                    index_depth=f1.index.depth,
                    columns_depth=f1.columns.depth)
            self.assertEqualFrames(f1, f2)


    def test_frame_from_hdf5_a(self) -> None:
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'),
                name='f1'
                )

        with temp_file('.h5') as fp:
            f1.to_hdf5(fp)
            f2 = Frame.from_hdf5(fp, label=f1.name, index_depth=f1.index.depth)
            self.assertEqualFrames(f1, f2)

    #---------------------------------------------------------------------------

    def test_frame_and_a(self) -> None:

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
                [0, 1, 2, 3]],
                columns=list('ABCD'))

        self.assertEqual(f1.all(axis=0).to_pairs(),
                (('p', True), ('q', True), ('r', True), ('s', False), ('t', False)))

        self.assertEqual(f1.any(axis=0).to_pairs(),
                (('p', True), ('q', True), ('r', True), ('s', True), ('t', True)))

        self.assertEqual(f1.all(axis=1).to_pairs(),
                (('w', False), ('x', False), ('y', False), ('z', True)))

        self.assertEqual(f1.any(axis=1).to_pairs(),
                (('w', True), ('x', True), ('y', True), ('z', True)))



    def test_frame_unique_a(self) -> None:

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

    def test_frame_unique_b(self) -> None:

        records = (
                (None, 2, None),
                ('30', 34, '30'),
                (None, 2, None),
                ('30', 34, '30'),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r'),
                index=('w', 'x', 'y', 'z'))

        self.assertEqual(len(f1.unique()), 4)

        self.assertEqual(len(f1.unique(axis=0)), 2)

        self.assertEqual(len(f1.unique(axis=1)), 2)


    def test_frame_duplicated_a(self) -> None:

        a1 = np.array([[50, 50, 32, 17, 17], [2,2,1,3,3]])
        f1 = Frame(a1, index=('a', 'b'), columns=('p', 'q', 'r', 's','t'))

        self.assertEqual(f1.duplicated(axis=1).to_pairs(),
                (('p', True), ('q', True), ('r', False), ('s', True), ('t', True)))

        self.assertEqual(f1.duplicated(axis=0).to_pairs(),
                (('a', False), ('b', False)))


    def test_frame_duplicated_b(self) -> None:

        a1 = np.array([[50, 50, 32, 17, 17], [2,2,1,3,3]])
        f1 = Frame(a1, index=('a', 'b'), columns=('p', 'q', 'r', 's','t'))

        self.assertEqual(f1.drop_duplicated(axis=1, exclude_first=True).to_pairs(1),
                (('a', (('p', 50), ('r', 32), ('s', 17))), ('b', (('p', 2), ('r', 1), ('s', 3)))))

    def test_frame_from_concat_a(self) -> None:
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


    def test_frame_from_concat_b(self) -> None:
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


    def test_frame_from_concat_c(self) -> None:
        records1 = (
                (2, 2, False),
                (30, 34, False),
                )

        f1 = Frame.from_records(records1,
                columns=('p', 'q', 't'),
                index=('x', 'a'))

        records2 = (
                ('c', False),
                ('d', True),
                )
        f2 = Frame.from_records(records2,
                columns=('r', 's',),
                index=('x', 'a'))

        # get combined columns as they are unique
        f = Frame.from_concat((f1, f2), axis=1)
        self.assertEqual(f.to_pairs(0),
                (('p', (('x', 2), ('a', 30))), ('q', (('x', 2), ('a', 34))), ('t', (('x', False), ('a', False))), ('r', (('x', 'c'), ('a', 'd'))), ('s', (('x', False), ('a', True))))
                )


    @skip_win  # type: ignore
    def test_frame_from_concat_d(self) -> None:
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


    @skip_win  # type: ignore
    def test_frame_from_concat_e(self) -> None:

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

    def test_frame_from_concat_f(self) -> None:
        # force a reblock before concatenating

        a1 = np.array([1, 2, 3], dtype=np.int64)
        a2 = np.array([10,50,30], dtype=np.int64)
        a3 = np.array([1345,2234,3345], dtype=np.int64)
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


    def test_frame_from_concat_g(self) -> None:
        records1 = (
                (2, 2, False),
                (30, 34, False),
                )

        f1 = Frame.from_records(records1,
                columns=('p', 'q', 't'),
                index=('x', 'a'))

        records2 = (
                ('c', False),
                ('d', True),
                )
        f2 = Frame.from_records(records2,
                columns=('r', 's',),
                index=('x', 'a'))

        # get combined columns as they are unique
        f = Frame.from_concat((f1, f2), axis=1)
        self.assertEqual(f.to_pairs(0),
                (('p', (('x', 2), ('a', 30))), ('q', (('x', 2), ('a', 34))), ('t', (('x', False), ('a', False))), ('r', (('x', 'c'), ('a', 'd'))), ('s', (('x', False), ('a', True))))
                )


    def test_frame_from_concat_h(self) -> None:

        index = list(''.join(x) for x in it.combinations(string.ascii_lowercase, 3))
        columns = list(''.join(x) for x in it.combinations(string.ascii_uppercase, 2))
        data = np.random.rand(len(index), len(columns))
        f1 = Frame(data, index=index, columns=columns)

        f2 = f1[[c for c in f1.columns if tp.cast(str, c).startswith('D')]]
        f3 = f1[[c for c in f1.columns if tp.cast(str, c).startswith('G')]]
        post = sf.Frame.from_concat((f2, f3), axis=1)

        # this form of concatenation has no copy
        assert post.mloc.tolist() == [f2.mloc[0], f3.mloc[0]]
        self.assertEqual(post.shape, (2600, 41))


    def test_frame_from_concat_i(self) -> None:

        sf1 = sf.Frame.from_dict(dict(a=[1,2,3],b=[1,2,3]),index=[100,200,300]).relabel_add_level(columns='A')
        sf2 = sf.Frame.from_dict(dict(a=[1,2,3],b=[1,2,3]),index=[100,200,300]).relabel_add_level(columns='B')

        f = sf.Frame.from_concat((sf1, sf2), axis=1)
        self.assertEqual(f.to_pairs(0),
                ((('A', 'a'), ((100, 1), (200, 2), (300, 3))), (('A', 'b'), ((100, 1), (200, 2), (300, 3))), (('B', 'a'), ((100, 1), (200, 2), (300, 3))), (('B', 'b'), ((100, 1), (200, 2), (300, 3)))))


    def test_frame_from_concat_j(self) -> None:

        sf1 = sf.Frame.from_dict(dict(a=[1,2,3],b=[1,2,3]),index=[100,200,300]).relabel_add_level(index='A')
        sf2 = sf.Frame.from_dict(dict(a=[1,2,3],b=[1,2,3]),index=[100,200,300]).relabel_add_level(index='B')

        f = sf.Frame.from_concat((sf1, sf2), axis=0)

        self.assertEqual(f.to_pairs(0),
                (('a', ((('A', 100), 1), (('A', 200), 2), (('A', 300), 3), (('B', 100), 1), (('B', 200), 2), (('B', 300), 3))), ('b', ((('A', 100), 1), (('A', 200), 2), (('A', 300), 3), (('B', 100), 1), (('B', 200), 2), (('B', 300), 3))))
                )


    def test_frame_from_concat_k(self) -> None:
        records1 = (
                (2, 2, False),
                (30, 34, False),
                )
        f1 = Frame.from_records(records1,
                columns=('p', 'q', 't'),
                index=('x', 'a'))

        records2 = (
                ('c', False),
                ('d', True),
                )
        f2 = Frame.from_records(records2,
                columns=('r', 's',),
                index=('x', 'a'))

        # get combined columns as they are unique
        f = Frame.from_concat((f1, f2), axis=1, name='foo')
        self.assertEqual(f.name, 'foo')


    def test_frame_from_concat_m(self) -> None:
        records1 = (
                (2, 2, False),
                (30, 34, False),
                )
        f1 = Frame.from_records(records1,
                columns=('p', 'q', 't'),
                index=('x', 'a'))

        records2 = (
                ('c', False),
                ('d', True),
                )
        f2 = Frame.from_records(records2,
                columns=(3, 4,),
                index=('x', 'a'))

        f = Frame.from_concat((f1, f2), axis=1, name='foo')

        self.assertEqual(f.columns.values.tolist(),
                ['p', 'q', 't', 3, 4])
        self.assertEqual(f.to_pairs(0),
                (('p', (('x', 2), ('a', 30))), ('q', (('x', 2), ('a', 34))), ('t', (('x', False), ('a', False))), (3, (('x', 'c'), ('a', 'd'))), (4, (('x', False), ('a', True))))
                )

    def test_frame_from_concat_n(self) -> None:
        records1 = (
                (2, False),
                (30, False),
                )
        f1 = Frame.from_records(records1,
                columns=('p', 'q'),
                index=('x', 'a'))

        records2 = (
                ('c', False),
                ('d', True),
                )
        f2 = Frame.from_records(records2,
                columns=('p', 'q'),
                index=(3, 10))

        f = Frame.from_concat((f1, f2), axis=0, name='foo')

        self.assertEqual(f.index.values.tolist(),
                ['x', 'a', 3, 10])
        self.assertEqual(f.to_pairs(0),
                (('p', (('x', 2), ('a', 30), (3, 'c'), (10, 'd'))), ('q', (('x', False), ('a', False), (3, False), (10, True))))
                )


    def test_frame_from_concat_o(self) -> None:
        records1 = (
                (2, False),
                (34, False),
                )

        f1 = Frame.from_records(records1,
                columns=('p', 'q',),
                index=('x', 'z'))

        records2 = (
                ('c', False),
                ('d', True),
                )
        f2 = Frame.from_records(records2,
                columns=('r', 's',),
                index=('x', 'z'))


        s1 = Series((0, 100), index=('x', 'z'), name='t')

        f = Frame.from_concat((f1, f2, s1), axis=1)

        self.assertEqual(f.to_pairs(0),
                (('p', (('x', 2), ('z', 34))), ('q', (('x', False), ('z', False))), ('r', (('x', 'c'), ('z', 'd'))), ('s', (('x', False), ('z', True))), ('t', (('x', 0), ('z', 100))))
                )



    def test_frame_from_concat_p(self) -> None:
        records = (
                (2, False),
                (34, False),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q',),
                index=('a', 'b'))

        s1 = Series((0, True), index=('p', 'q'), name='c', dtype=object)
        s2 = Series((-2, False), index=('p', 'q'), name='d', dtype=object)

        f = Frame.from_concat((s2, f1, s1), axis=0)

        self.assertEqual(f.to_pairs(0),
                (('p', (('d', -2), ('a', 2), ('b', 34), ('c', 0))), ('q', (('d', False), ('a', False), ('b', False), ('c', True)))))



    def test_frame_from_concat_q(self) -> None:
        s1 = Series((2, 3, 0,), index=list('abc'), name='x').relabel_add_level('i')
        s2 = Series(('10', '20', '100'), index=list('abc'), name='y').relabel_add_level('i')

        # stack horizontally
        f = Frame.from_concat((s1, s2), axis=1)

        self.assertEqual(f.to_pairs(0),
                (('x', ((('i', 'a'), 2), (('i', 'b'), 3), (('i', 'c'), 0))), ('y', ((('i', 'a'), '10'), (('i', 'b'), '20'), (('i', 'c'), '100'))))
            )

        # stack vertically
        f = Frame.from_concat((s1, s2), axis=0)
        self.assertEqual(f.to_pairs(0),
                ((('i', 'a'), (('x', 2), ('y', '10'))), (('i', 'b'), (('x', 3), ('y', '20'))), (('i', 'c'), (('x', 0), ('y', '100'))))
            )


    def test_frame_from_concat_r(self) -> None:
        f1 = sf.Frame.from_dict_records(
                [dict(a=1,b=1),dict(a=2,b=3),dict(a=1,b=1),dict(a=2,b=3)],
                index=sf.IndexHierarchy.from_labels([(1,'dd'),(1,'bb'),(2,'cc'),(2,'dd')]))

        f2 = sf.Frame.from_dict_records(
                [dict(a=1,b=1),dict(a=2,b=3),dict(a=1,b=1),dict(a=2,b=3)],
                index=sf.IndexHierarchy.from_labels([(3,'ddd'),(3,'bbb'),(4,'ccc'),(4,'ddd')])) * 100

        self.assertEqual(Frame.from_concat((f1, f2), axis=0).to_pairs(0),
                (('a', (((1, 'dd'), 1), ((1, 'bb'), 2), ((2, 'cc'), 1), ((2, 'dd'), 2), ((3, 'ddd'), 100), ((3, 'bbb'), 200), ((4, 'ccc'), 100), ((4, 'ddd'), 200))), ('b', (((1, 'dd'), 1), ((1, 'bb'), 3), ((2, 'cc'), 1), ((2, 'dd'), 3), ((3, 'ddd'), 100), ((3, 'bbb'), 300), ((4, 'ccc'), 100), ((4, 'ddd'), 300))))
                )

    def test_frame_from_concat_s(self) -> None:
        records1 = (
                (2, False),
                (34, False),
                )

        f1 = Frame.from_records(records1,
                columns=('p', 'q',),
                index=('x', 'z'))

        records2 = (
                ('c', False),
                ('d', True),
                )
        f2 = Frame.from_records(records2,
                columns=('r', 's',),
                index=('x', 'z'))

        with self.assertRaises(NotImplementedError):
            f = Frame.from_concat((f1, f2), axis=None)


    def test_frame_from_concat_t(self) -> None:
        frame1 = sf.Frame.from_dict_records(
                [dict(a=1,b=1), dict(a=2,b=3), dict(a=1,b=1), dict(a=2,b=3)], index=sf.IndexHierarchy.from_labels([(1,'dd',0), (1,'bb',0), (2,'cc',0), (2,'ee',0)]))
        frame2 = sf.Frame.from_dict_records(
                [dict(a=100,b=200), dict(a=20,b=30), dict(a=101,b=101), dict(a=201,b=301)], index=sf.IndexHierarchy.from_labels([(1,'ddd',0), (1,'bbb',0), (2,'ccc',0), (2,'eee',0)]))

        # produce invalid index labels into an IndexHierarchy constructor
        with self.assertRaises(RuntimeError):
            sf.Frame.from_concat((frame1, frame2))


    def test_frame_from_concat_u(self) -> None:
        # this fails; figure out why
        a = sf.Series(('a', 'b', 'c'), index=range(3, 6))
        f = sf.Frame.from_concat((
                a,
                sf.Series(a.index.values, index=a.index)),
                axis=0,
                columns=(3, 4, 5), index=(1,2))

        self.assertEqual(f.to_pairs(0),
                ((3, ((1, 'a'), (2, 3))), (4, ((1, 'b'), (2, 4))), (5, ((1, 'c'), (2, 5))))
                )


    def test_frame_from_concat_v(self) -> None:
        records1 = (
                (2, False),
                (34, False),
                )

        f1 = Frame.from_records(records1,
                columns=('p', 'q'),
                index=('x', 'y'))

        records2 = (
                ('c', False),
                ('d', True),
                )
        f2 = Frame.from_records(records2,
                columns=('p', 'q',),
                index=('x', 'y'))

        # get combined columns as they are unique
        post1 = Frame.from_concat((f1, f2), axis=1, columns=IndexAutoFactory)
        self.assertEqual(post1.to_pairs(0),
                ((0, (('x', 2), ('y', 34))), (1, (('x', False), ('y', False))), (2, (('x', 'c'), ('y', 'd'))), (3, (('x', False), ('y', True))))
                )

        with self.assertRaises(ErrorInitFrame):
            Frame.from_concat((f1, f2), axis=1, columns=IndexAutoFactory, index=IndexAutoFactory)

        post2 = Frame.from_concat((f1, f2), axis=0, index=IndexAutoFactory)
        self.assertEqual(post2.to_pairs(0),
                (('p', ((0, 2), (1, 34), (2, 'c'), (3, 'd'))), ('q', ((0, False), (1, False), (2, False), (3, True))))
                )

        with self.assertRaises(ErrorInitFrame):
            Frame.from_concat((f1, f2), axis=0, index=IndexAutoFactory, columns=IndexAutoFactory)




    @skip_win  # type: ignore
    def test_frame_from_concat_w(self) -> None:

        a = sf.Frame.from_dict({0:(1,2), 1:(2,3), 2:(True, True)})
        b = sf.Frame.from_dict({0:(1,2), 1:(np.nan, np.nan), 2:(False, False)})

        # reblock first two columns into integers
        c = a.astype[[0,1]](int)
        self.assertEqual(c._blocks.shapes.tolist(),
                [(2, 2), (2,)])

        # unaligned blocks compared column to column
        post1 = sf.Frame.from_concat([c, b], index=sf.IndexAutoFactory)

        self.assertEqual(post1.dtypes.values.tolist(),
                [np.dtype('int64'), np.dtype('float64'), np.dtype('bool')]
                )

        post2 = sf.Frame.from_concat([a, b], index=sf.IndexAutoFactory)

        self.assertEqual(post2.dtypes.values.tolist(),
                [np.dtype('int64'), np.dtype('float64'), np.dtype('bool')]
                )


    def test_frame_from_concat_x(self) -> None:
        f1 = Frame.from_concat([])
        self.assertEqual((0,0), f1.shape)

        f2 = Frame.from_concat([], columns='a')
        self.assertEqual((0,1), f2.shape)
        self.assertEqual((1,),  f2.columns.shape)

        f3 = Frame.from_concat([], index=[])
        self.assertEqual((0,0), f3.shape)
        self.assertEqual((0,),  f3.index.shape)

        f4 = Frame.from_concat([], name='f4')
        self.assertEqual((0,0), f4.shape)
        self.assertEqual('f4',  f4.name)

        f5 = Frame.from_concat([], columns='a', index=[], name='f5')
        self.assertEqual((0,1), f5.shape)
        self.assertEqual((1,),  f5.columns.shape)
        self.assertEqual((0,),  f5.index.shape)
        self.assertEqual('f5',  f5.name)


    #---------------------------------------------------------------------------


    def test_frame_from_concat_items_a(self) -> None:
        records1 = (
                (2, 2, False),
                (30, 34, False),
                )

        f1 = Frame.from_records(records1,
                columns=('p', 'q', 't'),
                index=('x', 'a'))

        records2 = (
                ('c', False),
                ('d', True),
                )
        f2 = Frame.from_records(records2,
                columns=('r', 's',),
                index=('x', 'a'))

        f3 = Frame.from_concat_items(dict(A=f1, B=f2).items(), axis=1)

        self.assertEqual(f3.to_pairs(0),
                ((('A', 'p'), (('x', 2), ('a', 30))), (('A', 'q'), (('x', 2), ('a', 34))), (('A', 't'), (('x', False), ('a', False))), (('B', 'r'), (('x', 'c'), ('a', 'd'))), (('B', 's'), (('x', False), ('a', True)))))

        f4 = FrameGO.from_concat_items(dict(A=f1, B=f2).items(), axis=1)
        self.assertEqual(f4.__class__, FrameGO)
        self.assertEqual(f4.columns.__class__, IndexHierarchyGO)
        self.assertEqual(f4.index.__class__, Index)

        self.assertEqual(f4.to_pairs(0),
                ((('A', 'p'), (('x', 2), ('a', 30))), (('A', 'q'), (('x', 2), ('a', 34))), (('A', 't'), (('x', False), ('a', False))), (('B', 'r'), (('x', 'c'), ('a', 'd'))), (('B', 's'), (('x', False), ('a', True)))))

    def test_frame_from_concat_items_b(self) -> None:

        f1 = Frame.from_records(((2, False), (34, False)),
                columns=('p', 'q',),
                index=('d', 'c'))

        s1 = Series((0, True), index=('p', 'q'), name='c', dtype=object)
        s2 = Series((-2, False), index=('p', 'q'), name='d', dtype=object)

        f2 = Frame.from_concat_items(dict(A=s2, B=f1, C=s1).items(), axis=0)

        self.assertEqual(f2.to_pairs(0),
                (('p', ((('A', 'd'), -2), (('B', 'd'), 2), (('B', 'c'), 34), (('C', 'c'), 0))), ('q', ((('A', 'd'), False), (('B', 'd'), False), (('B', 'c'), False), (('C', 'c'), True))))
                )
        self.assertEqual(f2.index.__class__, IndexHierarchy)

    def test_frame_from_concat_items_c(self) -> None:
        f1 = Frame.from_concat_items([])
        self.assertEqual((0,0), f1.shape)

        f2 = Frame.from_concat_items([], name='f2')
        self.assertEqual((0,0), f2.shape)
        self.assertEqual('f2',  f2.name)

        # Demonstrate the other arguments are inconsequential
        f3 = Frame.from_concat_items([],
                axis=1,
                union=False,
                name='f3',
                fill_value=True,
                consolidate_blocks=False)
        self.assertEqual((0,0), f3.shape)
        self.assertEqual('f3',  f3.name)


    def test_frame_set_index_a(self) -> None:
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x', 'y'),
                consolidate_blocks=True)

        self.assertEqual(f1.set_index('r').to_pairs(0),
                (('p', (('a', 2), ('b', 30))), ('q', (('a', 2), ('b', 34))), ('r', (('a', 'a'), ('b', 'b'))), ('s', (('a', False), ('b', True))), ('t', (('a', False), ('b', False)))))

        self.assertEqual(f1.set_index('r', drop=True).to_pairs(0),
                (('p', (('a', 2), ('b', 30))), ('q', (('a', 2), ('b', 34))), ('s', (('a', False), ('b', True))), ('t', (('a', False), ('b', False)
                ))))

        f2 = f1.set_index('r', drop=True)

        self.assertEqual(f2.to_pairs(0),
                (('p', (('a', 2), ('b', 30))), ('q', (('a', 2), ('b', 34))), ('s', (('a', False), ('b', True))), ('t', (('a', False), ('b', False))))
                )

        self.assertTrue(f1.mloc[[0, 2]].tolist() == f2.mloc.tolist())


    def test_frame_set_index_b(self) -> None:
        records = (
                (2, 2, 'a', False, True),
                (30, 34, 'b', True, False),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x', 'y'),
                consolidate_blocks=True)

        for col in f1.columns:
            f2 = f1.set_index(col)
            self.assertEqual(f2.index.name, col)


    def test_frame_set_index_c(self) -> None:
        records = (
                (2, 2, 'a', False, True),
                (30, 34, 'b', True, False),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                )

    def test_frame_set_index_d(self) -> None:

        for arrays in self.get_arrays_a():
            tb1 = TypeBlocks.from_blocks(arrays)

            f1 = FrameGO(tb1)
            f1[tb1.shape[1]] = range(tb1.shape[0])

            for i in range(f1.shape[1]):
                f2 = f1.set_index(i, drop=True)
                self.assertTrue(f2.shape == (3, f1.shape[1] - 1))


    def test_frame_head_tail_a(self) -> None:

        # thest of multi threaded apply

        f1 = Frame.from_items(
                zip(range(10), (np.random.rand(1000) for _ in range(10)))
                )
        self.assertEqual(f1.head(3).index.values.tolist(),
                [0, 1, 2])
        self.assertEqual(f1.tail(3).index.values.tolist(),
                [997, 998, 999])

    #---------------------------------------------------------------------------
    def test_frame_from_records_date_a(self) -> None:

        d = np.datetime64

        records = (
                (d('2018-01-02'), d('2018-01-02'), 'a', False, False),
                (d('2017-01-02'), d('2017-01-02'), 'b', True, False),
                (d('2016-01-02'), d('2016-01-02'), 'c', False, False),
                (d('2015-01-02'), d('2015-01-02'), 'd', True, True),
                )

        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=None)

        dtype = np.dtype

        self.assertEqual(list(f1._blocks._reblock_signature()),
                [(dtype('<M8[D]'), 2), (dtype('<U1'), 1), (dtype('bool'), 2)])


    def test_frame_from_records_a(self) -> None:

        NT = namedtuple('Sample', ('a', 'b', 'c'))
        records = [NT(x, x, x) for x in range(4)]
        f1 = Frame.from_records(records)
        self.assertEqual(f1.columns.values.tolist(), ['a', 'b', 'c'])
        self.assertEqual(f1.sum().to_pairs(),
                (('a', 6), ('b', 6), ('c', 6)))

    def test_frame_from_records_b(self) -> None:

        f1 = sf.Frame.from_records([[1, 2], [2, 3]], columns=['a', 'b'])
        self.assertEqual(f1.to_pairs(0),
                (('a', ((0, 1), (1, 2))), ('b', ((0, 2), (1, 3)))))

        with self.assertRaises(Exception):
            f2 = sf.Frame.from_records([[1, 2], [2, 3]], columns=['a'])


    def test_frame_from_records_c(self) -> None:

        s1 = Series([3, 4, 5], index=('x', 'y', 'z'))
        s2 = Series(list('xyz'), index=('x', 'y', 'z'))

        with self.assertRaises(Exception):
            # cannot use Series in from_records
            f1 = sf.Frame.from_records([s1, s2], columns=['a', 'b', 'c'])


    def test_frame_from_records_d(self) -> None:

        a1 = np.array([[1,2,3], [4,5,6]])

        f1 = sf.Frame.from_records(a1, index=('x', 'y'), columns=['a', 'b', 'c'])

        self.assertEqual(f1.to_pairs(0),
                (('a', (('x', 1), ('y', 4))), ('b', (('x', 2), ('y', 5))), ('c', (('x', 3), ('y', 6)))))


    def test_frame_from_records_e(self) -> None:

        records = [[1,'2',3], [4,'5',6]]
        dtypes = (np.int64, str, str)
        f1 = sf.Frame.from_records(records,
                index=('x', 'y'),
                columns=['a', 'b', 'c'],
                dtypes=dtypes)
        self.assertEqual(f1.dtypes.iter_element().apply(str).to_pairs(),
                (('a', 'int64'), ('b', '<U1'), ('c', '<U1'))
                )

    def test_frame_from_records_f(self) -> None:

        records = [[1,'2',3], [4,'5',6]]
        dtypes = {'b': np.int64}
        f1 = sf.Frame.from_records(records,
                index=('x', 'y'),
                columns=['a', 'b', 'c'],
                dtypes=dtypes)

        self.assertEqual(str(f1.dtypes['b']), 'int64')


    def test_frame_from_records_g(self) -> None:

        NT = namedtuple('NT', ('a', 'b', 'c'))

        records = [NT(1,'2',3), NT(4,'5',6)]
        dtypes = {'b': np.int64}
        f1 = sf.Frame.from_records(records, dtypes=dtypes)

        self.assertEqual(str(f1.dtypes['b']), 'int64')


    def test_frame_from_records_h(self) -> None:

        with self.assertRaises(ErrorInitFrame):
            Frame.from_records(())

        with self.assertRaises(ErrorInitFrame):
            Frame.from_records(((0, 1, 2) for x in range(3) if x < 0))

    def test_frame_from_records_i(self) -> None:

        f1 = sf.Frame.from_records([
            (88,),
            (27, ),
            (27,),
            (None,)])

        self.assertEqual(f1.to_pairs(0),
                ((0, ((0, 88), (1, 27), (2, 27), (3, None))),))


    def test_frame_from_records_j(self) -> None:

        records = [
            dict(a=1, b=2),
            dict(b=10, c=4),
            dict(c=20, d=-1)
        ]
        with self.assertRaises(ErrorInitFrame):
            # cannot supply columns when records are dictionaries
            f1 = Frame.from_records(records, columns=('b', 'c', 'd'))



    def test_frame_from_records_k(self) -> None:
        def gen() -> tp.Iterator[int]:
            empty: tp.Iterable[int] = ()
            for k in empty:
                yield k

        f1 = Frame.from_records(gen(), columns=('a', 'b', 'c'))
        self.assertEqual(f1.to_pairs(0),
                (('a', ()), ('b', ()), ('c', ())))



    #---------------------------------------------------------------------------

    def test_frame_from_dict_records_a(self) -> None:

        records = [{'a':x, 'b':x, 'c':x} for x in range(4)]
        f1 = Frame.from_dict_records(records)
        self.assertEqual(f1.columns.values.tolist(), ['a', 'b', 'c'])
        self.assertEqual(f1.sum().to_pairs(),
                (('a', 6), ('b', 6), ('c', 6)))

    def test_frame_from_dict_records_b(self) -> None:
        # handle case of dict views
        a = {1: {'a': 1, 'b': 2,}, 2: {'a': 4, 'b': 3,}}

        post = Frame.from_dict_records(a.values(), index=list(a.keys()))

        self.assertEqual(post.to_pairs(0),
                (('a', ((1, 1), (2, 4))), ('b', ((1, 2), (2, 3)))))


    def test_frame_from_dict_records_c(self) -> None:

        records = [dict(a=1, b='2', c=3), dict(a=4, b='5', c=6)]
        dtypes = {'b': np.int64}
        f1 = sf.Frame.from_dict_records(records, dtypes=dtypes)

        self.assertEqual(str(f1.dtypes['b']), 'int64')

    def test_frame_from_dict_records_d(self) -> None:

        records = [
            dict(a=1, b=2),
            dict(b=10, c=4),
            dict(c=20, d=-1)
        ]
        f1 = Frame.from_dict_records(records, fill_value=0)
        self.assertEqual(f1.to_pairs(0),
                (('a', ((0, 1), (1, 0), (2, 0))), ('b', ((0, 2), (1, 10), (2, 0))), ('c', ((0, 0), (1, 4), (2, 20))), ('d', ((0, 0), (1, 0), (2, -1))))
                )


    #---------------------------------------------------------------------------
    def test_frame_from_json_a(self) -> None:

        msg = """[
        {
        "userId": 1,
        "id": 1,
        "title": "delectus aut autem",
        "completed": false
        },
        {
        "userId": 1,
        "id": 2,
        "title": "quis ut nam facilis et officia qui",
        "completed": false
        },
        {
        "userId": 1,
        "id": 3,
        "title": "fugiat veniam minus",
        "completed": false
        },
        {
        "userId": 1,
        "id": 4,
        "title": "et porro tempora",
        "completed": true
        }]"""

        f1 = Frame.from_json(msg, name=msg)
        self.assertEqual(sorted(f1.columns.values.tolist()),
                sorted(['completed', 'id', 'title', 'userId']))
        self.assertEqual(f1['id'].sum(), 10)

        self.assertEqual(f1.name, msg)



    def test_frame_reindex_flat_a(self) -> None:

        records = (
                (1, 2, 'a', False, True),
                (30, 34, 'b', True, False),
                (54, 95, 'c', False, False),
                )

        columns = IndexHierarchy.from_labels(
                (('a', 1), ('a', 2), ('b', 1), ('b', 2), ('b', 3)))
        f1 = Frame.from_records(records,
                columns=columns,
                index=('x', 'y', 'z'))

        f2 = f1.relabel_flat(columns=True)

        self.assertEqual(f2.to_pairs(0),
                ((('a', 1), (('x', 1), ('y', 30), ('z', 54))), (('a', 2), (('x', 2), ('y', 34), ('z', 95))), (('b', 1), (('x', 'a'), ('y', 'b'), ('z', 'c'))), (('b', 2), (('x', False), ('y', True), ('z', False))), (('b', 3), (('x', True), ('y', False), ('z', False)))))


    def test_frame_add_level_a(self) -> None:

        records = (
                (1, 2, 'a', False, True),
                (30, 34, 'b', True, False),
                (54, 95, 'c', False, False),
                )
        f1 = Frame.from_records(records,
                columns=('a', 'b', 'c', 'd', 'e'),
                index=('x', 'y', 'z'))

        f2 = f1.relabel_add_level(index='I', columns='II')

        self.assertEqual(f2.to_pairs(0),
                ((('II', 'a'), ((('I', 'x'), 1), (('I', 'y'), 30), (('I', 'z'), 54))), (('II', 'b'), ((('I', 'x'), 2), (('I', 'y'), 34), (('I', 'z'), 95))), (('II', 'c'), ((('I', 'x'), 'a'), (('I', 'y'), 'b'), (('I', 'z'), 'c'))), (('II', 'd'), ((('I', 'x'), False), (('I', 'y'), True), (('I', 'z'), False))), (('II', 'e'), ((('I', 'x'), True), (('I', 'y'), False), (('I', 'z'), False))))
                )


    def test_frame_from_from_pandas_a(self) -> None:
        import pandas as pd

        pdf = pd.DataFrame(
                dict(a=(False, True, False),
                b=(False, False,False),
                c=(1,2,3),
                d=(4,5,6),
                e=(None, None, None)))

        sff = Frame.from_pandas(pdf)
        self.assertTrue((pdf.dtypes.values == sff.dtypes.values).all())


    def test_frame_to_frame_go_a(self) -> None:
        records = (
                (1, 2, 'a', False, True),
                (30, 34, 'b', True, False),
                (54, 95, 'c', False, False),
                )
        f1 = Frame.from_records(records,
                columns=('a', 'b', 'c', 'd', 'e'),
                index=('x', 'y', 'z'))

        f2 = f1.to_frame_go()
        f2['f'] = None
        self.assertEqual(f1.columns.values.tolist(),
                ['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(f2.columns.values.tolist(),
                ['a', 'b', 'c', 'd', 'e', 'f'])

        # underlying map objects must be different
        self.assertTrue(id(f1.columns._map) != id(f2.columns._map))


    def test_frame_to_frame_go_b(self) -> None:
        records = (
                (1, 2, 'a', False, True),
                (54, 95, 'c', False, False),
                )
        f1 = Frame.from_records(records,
                columns=('a', 'b', 'c', 'd', 'e'),
                index=('x', 'y'),
                name='foo')

        f2 = f1.to_frame_go()
        f3 = f2.to_frame()

        self.assertTrue(f1.name, 'foo')
        self.assertTrue(f2.name, 'foo')
        self.assertTrue(f3.name, 'foo')



    def test_frame_to_frame_go_c(self) -> None:
        records = (
                (1, 'a', False, True),
                (1, 'b', False, False),
                (2, 'a', False, True),
                (2, 'b', False, False),
                )
        f1 = Frame.from_records(records,
                columns=IndexHierarchy.from_product((1, 2), ('a', 'b')),
                index=('w', 'x', 'y', 'z'),
                name='foo')

        f2 = f1.to_frame_go()

        self.assertTrue(isinstance(f2.columns, IndexHierarchyGO))

        f2[(3, 'a')] = 10

        self.assertEqual(
                f2.to_pairs(0),
                (((1, 'a'), (('w', 1), ('x', 1), ('y', 2), ('z', 2))), ((1, 'b'), (('w', 'a'), ('x', 'b'), ('y', 'a'), ('z', 'b'))), ((2, 'a'), (('w', False), ('x', False), ('y', False), ('z', False))), ((2, 'b'), (('w', True), ('x', False), ('y', True), ('z', False))), ((3, 'a'), (('w', 10), ('x', 10), ('y', 10), ('z', 10))))
        )




    def test_frame_astype_a(self) -> None:
        records = (
                (1, 2, 'a', False, True),
                (30, 34, 'b', True, False),
                (54, 95, 'c', False, False),
                )
        f1 = Frame.from_records(records,
                columns=('a', 'b', 'c', 'd', 'e'),
                index=('x', 'y', 'z'))

        f2 = f1.astype['d':](int)  # type: ignore  # https://github.com/python/typeshed/pull/3024
        self.assertEqual(f2.to_pairs(0),
                (('a', (('x', 1), ('y', 30), ('z', 54))), ('b', (('x', 2), ('y', 34), ('z', 95))), ('c', (('x', 'a'), ('y', 'b'), ('z', 'c'))), ('d', (('x', 0), ('y', 1), ('z', 0))), ('e', (('x', 1), ('y', 0), ('z', 0))))
                )

        f3 = f1.astype[['a', 'b']](bool)
        self.assertEqual(f3.to_pairs(0),
                (('a', (('x', True), ('y', True), ('z', True))), ('b', (('x', True), ('y', True), ('z', True))), ('c', (('x', 'a'), ('y', 'b'), ('z', 'c'))), ('d', (('x', False), ('y', True), ('z', False))), ('e', (('x', True), ('y', False), ('z', False))))
                )


    def test_frame_pickle_a(self) -> None:

        records = (
                (1, 2, 'a', False, True),
                (30, 50, 'b', True, False))

        f1 = FrameGO.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x','y'))

        pbytes = pickle.dumps(f1)
        f2 = pickle.loads(pbytes)

        self.assertEqual([b.flags.writeable for b in f2._blocks._blocks],
                [False, False, False, False, False])

    def test_frame_set_index_hierarchy_a(self) -> None:

        records = (
                (1, 2, 'a', False, True),
                (30, 2, 'b', True, False),
                (30, 50, 'a', True, False),
                (30, 50, 'b', True, False),
                )

        f1 = FrameGO.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        f2 = f1.set_index_hierarchy(['q', 'r'])
        self.assertEqual(f2.index.name, ('q', 'r'))

        self.assertEqual(f2.to_pairs(0),
                (('p', (((2, 'a'), 1), ((2, 'b'), 30), ((50, 'a'), 30), ((50, 'b'), 30))), ('q', (((2, 'a'), 2), ((2, 'b'), 2), ((50, 'a'), 50), ((50, 'b'), 50))), ('r', (((2, 'a'), 'a'), ((2, 'b'), 'b'), ((50, 'a'), 'a'), ((50, 'b'), 'b'))), ('s', (((2, 'a'), False), ((2, 'b'), True), ((50, 'a'), True), ((50, 'b'), True))), ('t', (((2, 'a'), True), ((2, 'b'), False), ((50, 'a'), False), ((50, 'b'), False)))))

        f3 = f1.set_index_hierarchy(['q', 'r'], drop=True)
        self.assertEqual(f3.index.name, ('q', 'r'))

        self.assertEqual(f3.to_pairs(0),
                (('p', (((2, 'a'), 1), ((2, 'b'), 30), ((50, 'a'), 30), ((50, 'b'), 30))), ('s', (((2, 'a'), False), ((2, 'b'), True), ((50, 'a'), True), ((50, 'b'), True))), ('t', (((2, 'a'), True), ((2, 'b'), False), ((50, 'a'), False), ((50, 'b'), False))))
                )

        f4 = f1.set_index_hierarchy(slice('q', 'r'), drop=True)  # type: ignore
        self.assertEqual(f4.index.name, ('q', 'r'))

        self.assertEqual(f4.to_pairs(0),
                (('p', (((2, 'a'), 1), ((2, 'b'), 30), ((50, 'a'), 30), ((50, 'b'), 30))), ('s', (((2, 'a'), False), ((2, 'b'), True), ((50, 'a'), True), ((50, 'b'), True))), ('t', (((2, 'a'), True), ((2, 'b'), False), ((50, 'a'), False), ((50, 'b'), False))))
                )


    def test_frame_set_index_hierarchy_b(self) -> None:

        labels = (
                (1, 1, 'a'),
                (1, 2, 'b'),
                (1, 3, 'c'),
                (2, 1, 'd'),
                (2, 2, 'e'),
                (2, 3, 'f'),
                (3, 1, 'g'),
                (3, 2, 'h'),
                (3, 3, 'i'),
                )

        f = Frame(labels)
        # import ipdb; ipdb.set_trace()
        f = f.astype[[0, 1]](int)


        fh = f.set_index_hierarchy([0, 1], drop=True)

        self.assertEqual(fh.columns.values.tolist(),
                [2]
                )

        self.assertEqual( fh.to_pairs(0),
                ((2, (((1, 1), 'a'), ((1, 2), 'b'), ((1, 3), 'c'), ((2, 1), 'd'), ((2, 2), 'e'), ((2, 3), 'f'), ((3, 1), 'g'), ((3, 2), 'h'), ((3, 3), 'i'))),))

        fh = f.set_index_hierarchy([0, 1], drop=False)
        self.assertEqual(
                fh.loc[HLoc[:, 3]].to_pairs(0),
                ((0, (((1, 3), 1), ((2, 3), 2), ((3, 3), 3))), (1, (((1, 3), 3), ((2, 3), 3), ((3, 3), 3))), (2, (((1, 3), 'c'), ((2, 3), 'f'), ((3, 3), 'i'))))
                )


    def test_frame_set_index_hierarchy_d(self) -> None:
        f1 = sf.Frame.from_records([('one', 1, 'hello')],
                columns=['name', 'val', 'msg'])

        f2 = f1.set_index_hierarchy(['name', 'val'], drop=True)

        self.assertEqual(f2.to_pairs(0),
                (('msg', ((('one', 1), 'hello'),)),))



    def test_frame_set_index_hierarchy_e(self) -> None:

        records = (
                (1, '2018-12', 10),
                (1, '2019-01', 20),
                (1, '2019-02', 30),
                (2, '2018-12', 40),
                (2, '2019-01', 50),
                (2, '2019-02', 60),
                )
        f = Frame.from_records(records)
        fh = f.set_index_hierarchy([0, 1],
                drop=True,
                index_constructors=(Index, IndexYearMonth))

        self.assertEqual(fh.loc[HLoc[:, '2018']].to_pairs(0),
                ((2, (((1, datetime.date(2018, 12, 1)), 10), ((2, datetime.date(2018, 12, 1)), 40))),))



    def test_frame_set_index_hierarchy_f(self) -> None:

        records = (
                (1, 'a', 10),
                (2, 'c', 60),
                (1, 'c', 30),
                (2, 'a', 40),
                (2, 'b', 50),
                (1, 'b', 20),
                )
        f = Frame.from_records(records)
        fh = f.set_index_hierarchy([0, 1],
                drop=True,
                reorder_for_hierarchy=True,
                )

        self.assertEqual(fh.to_pairs(0),
                ((2, (((1, 'a'), 10), ((1, 'c'), 30), ((1, 'b'), 20), ((2, 'a'), 40), ((2, 'c'), 60), ((2, 'b'), 50))),)
                )


    #---------------------------------------------------------------------------


    def test_frame_iloc_in_loc_a(self) -> None:
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        self.assertEqual(f1.loc[ILoc[-2:], ['q', 't']].to_pairs(0),
                (('q', (('y', 95), ('z', 73))), ('t', (('y', False), ('z', True)))))

        self.assertEqual(f1.loc[ILoc[[0, -1]], 's':].to_pairs(0),  # type: ignore  # https://github.com/python/typeshed/pull/3024
                (('s', (('w', False), ('z', True))), ('t', (('w', False), ('z', True)))))

        self.assertEqual(f1.loc[['w', 'x'], ILoc[[0, -1]]].to_pairs(0),
                (('p', (('w', 2), ('x', 30))), ('t', (('w', False), ('x', False))))
                )



    def test_frame_drop_a(self) -> None:
        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        self.assertEqual(f1.drop['r':].to_pairs(0),  # type: ignore  # https://github.com/python/typeshed/pull/3024
                (('p', (('w', 2), ('x', 30), ('y', 2), ('z', 30))), ('q', (('w', 2), ('x', 34), ('y', 95), ('z', 73)))))

        self.assertEqual(f1.drop.loc[['x', 'z'], 's':].to_pairs(0),  # type: ignore  # https://github.com/python/typeshed/pull/3024
                (('p', (('w', 2), ('y', 2))), ('q', (('w', 2), ('y', 95))), ('r', (('w', 'a'), ('y', 'c')))))

        self.assertEqual(f1.drop.loc['x':, 'q':].to_pairs(0),  # type: ignore  # https://github.com/python/typeshed/pull/3024
                (('p', (('w', 2),)),))


    def test_frame_roll_a(self) -> None:

        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        self.assertEqual(f1.roll(1).to_pairs(0),
                (('p', (('w', 30), ('x', 2), ('y', 30), ('z', 2))), ('q', (('w', 73), ('x', 2), ('y', 34), ('z', 95))), ('r', (('w', 'd'), ('x', 'a'), ('y', 'b'), ('z', 'c'))), ('s', (('w', True), ('x', False), ('y', True), ('z', False))), ('t', (('w', True), ('x', False), ('y', False), ('z', False))))
                )

        self.assertEqual(f1.roll(-2, include_index=True).to_pairs(0),
                (('p', (('y', 2), ('z', 30), ('w', 2), ('x', 30))), ('q', (('y', 95), ('z', 73), ('w', 2), ('x', 34))), ('r', (('y', 'c'), ('z', 'd'), ('w', 'a'), ('x', 'b'))), ('s', (('y', False), ('z', True), ('w', False), ('x', True))), ('t', (('y', False), ('z', True), ('w', False), ('x', False))))
                )

        self.assertEqual(f1.roll(-3, 3).to_pairs(0),
                (('p', (('w', 'd'), ('x', 'a'), ('y', 'b'), ('z', 'c'))), ('q', (('w', True), ('x', False), ('y', True), ('z', False))), ('r', (('w', True), ('x', False), ('y', False), ('z', False))), ('s', (('w', 30), ('x', 2), ('y', 30), ('z', 2))), ('t', (('w', 73), ('x', 2), ('y', 34), ('z', 95))))
                )

        self.assertEqual(
                f1.roll(-3, 3, include_index=True, include_columns=True).to_pairs(0),
                (('r', (('z', 'd'), ('w', 'a'), ('x', 'b'), ('y', 'c'))), ('s', (('z', True), ('w', False), ('x', True), ('y', False))), ('t', (('z', True), ('w', False), ('x', False), ('y', False))), ('p', (('z', 30), ('w', 2), ('x', 30), ('y', 2))), ('q', (('z', 73), ('w', 2), ('x', 34), ('y', 95)))))


    def test_frame_shift_a(self) -> None:


        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                (30, 73, 'd', True, True),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('w', 'x', 'y', 'z'))

        dtype = np.dtype

        # nan as default forces floats and objects
        self.assertEqual(f1.shift(2).dtypes.values.tolist(),
                [dtype('float64'), dtype('float64'), dtype('O'), dtype('O'), dtype('O')])

        self.assertEqual(f1.shift(1, fill_value=-1).to_pairs(0),
                (('p', (('w', -1), ('x', 2), ('y', 30), ('z', 2))), ('q', (('w', -1), ('x', 2), ('y', 34), ('z', 95))), ('r', (('w', -1), ('x', 'a'), ('y', 'b'), ('z', 'c'))), ('s', (('w', -1), ('x', False), ('y', True), ('z', False))), ('t', (('w', -1), ('x', False), ('y', False), ('z', False))))
                )

        self.assertEqual(f1.shift(1, 1, fill_value=-1).to_pairs(0),
                (('p', (('w', -1), ('x', -1), ('y', -1), ('z', -1))), ('q', (('w', -1), ('x', 2), ('y', 30), ('z', 2))), ('r', (('w', -1), ('x', 2), ('y', 34), ('z', 95))), ('s', (('w', -1), ('x', 'a'), ('y', 'b'), ('z', 'c'))), ('t', (('w', -1), ('x', False), ('y', True), ('z', False))))
                )

        self.assertEqual(f1.shift(0, 5, fill_value=-1).to_pairs(0),
                (('p', (('w', -1), ('x', -1), ('y', -1), ('z', -1))), ('q', (('w', -1), ('x', -1), ('y', -1), ('z', -1))), ('r', (('w', -1), ('x', -1), ('y', -1), ('z', -1))), ('s', (('w', -1), ('x', -1), ('y', -1), ('z', -1))), ('t', (('w', -1), ('x', -1), ('y', -1), ('z', -1))))
                )


    def test_frame_name_a(self) -> None:

        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                )
        f1 = Frame.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x', 'y', 'z'),
                name='test')

        self.assertEqual(f1.name, 'test')

        f2 = f1.rename('alt')

        self.assertEqual(f1.name, 'test')
        self.assertEqual(f2.name, 'alt')



    def test_frame_name_b(self) -> None:

        with self.assertRaises(TypeError):
            f = Frame.from_dict(dict(a=(1,2), b=(3,4)), name=['test'])

        with self.assertRaises(TypeError):
            f = Frame.from_dict(dict(a=(1,2), b=(3,4)), name={'a': 30})

        with self.assertRaises(TypeError):
            f = Frame.from_dict(dict(a=(1,2), b=(3,4)), name=('a', [1, 2]))



    def test_frame_name_c(self) -> None:

        records = (
                (2, 2, 'a', False, False),
                (30, 34, 'b', True, False),
                (2, 95, 'c', False, False),
                )
        f1 = FrameGO.from_records(records,
                columns=('p', 'q', 'r', 's', 't'),
                index=('x', 'y', 'z'),
                name='test')

        self.assertEqual(f1.name, 'test')

        f2 = f1.rename('alt')

        self.assertEqual(f1.name, 'test')
        self.assertEqual(f2.name, 'alt')

        f2['u'] = -1

        self.assertEqual(f1.columns.values.tolist(), ['p', 'q', 'r', 's', 't'])
        self.assertEqual(f2.columns.values.tolist(), ['p', 'q', 'r', 's', 't', 'u'])



    @skip_win  # type: ignore
    def test_frame_display_a(self) -> None:

        f1 = Frame(((1,2),(True,False)), name='foo',
                index=Index(('x', 'y'), name='bar'),
                columns=Index(('a', 'b'), name='rig')
                )

        match = tuple(f1.display(DisplayConfig(type_color=False)))

        self.assertEqual(
            match,
            (['<Frame: foo>'], ['<Index: rig>', 'a', 'b', '<<U1>'], ['<Index: bar>', '', ''], ['x', '1', '2'], ['y', '1', '0'], ['<<U1>', '<int64>', '<int64>'])
            )



    def test_frame_reindex_drop_level_a(self) -> None:

        f1 = Frame.from_dict_records(
                (dict(a=x, b=x) for x in range(4)),
                index=sf.IndexHierarchy.from_labels([(1,1),(1,2),(2,3),(2,4)]))

        with self.assertRaises(Exception):
            # this results in an index of size 2 being created, as we dro the leves with a postive depth; next support negative depth?
            f2 = f1.relabel_drop_level(index=-1)




    def test_frame_clip_a(self) -> None:

        records = (
                (2, 2),
                (30, 34),
                (2, 95),
                )
        f1 = Frame.from_records(records,
                columns=('a', 'b'),
                index=('x', 'y', 'z')
                )

        self.assertEqual(f1.clip(upper=0).to_pairs(0),
                (('a', (('x', 0), ('y', 0), ('z', 0))), ('b', (('x', 0), ('y', 0), ('z', 0)))))

        self.assertEqual(f1.clip(lower=90).to_pairs(0),
                (('a', (('x', 90), ('y', 90), ('z', 90))), ('b', (('x', 90), ('y', 90), ('z', 95)))))


    def test_frame_clip_b(self) -> None:

        records = (
                (2, 2),
                (30, 34),
                (2, 95),
                )
        f1 = Frame.from_records(records,
                columns=('a', 'b'),
                index=('x', 'y', 'z')
                )

        s1 = Series((1, 20), index=('a', 'b'))

        self.assertEqual(f1.clip(upper=s1, axis=1).to_pairs(0),
            (('a', (('x', 1), ('y', 1), ('z', 1))), ('b', (('x', 2), ('y', 20), ('z', 20)))))

        s2 = Series((3, 33, 80), index=('x', 'y', 'z'))

        self.assertEqual(f1.clip(s2, axis=0).to_pairs(0),
            (('a', (('x', 3), ('y', 33), ('z', 80))), ('b', (('x', 3), ('y', 34), ('z', 95)))))


    def test_frame_clip_c(self) -> None:

        records = (
                (2, 2),
                (30, 34),
                (2, 95),
                )
        f1 = Frame.from_records(records,
                columns=('a', 'b'),
                index=('x', 'y', 'z')
                )

        f2 = sf.Frame([[5, 4], [0, 10]], index=list('yz'), columns=list('ab'))

        self.assertEqual(f1.clip(upper=f2).to_pairs(0),
                (('a', (('x', 2.0), ('y', 5.0), ('z', 0.0))), ('b', (('x', 2.0), ('y', 4.0), ('z', 10.0)))))


    @unittest.skip('precedence of min/max changed in numpy 1.17.4')
    def test_frame_clip_d(self) -> None:

        records = (
                (2, 2),
                (30, 34),
                (2, 95),
                )
        f1 = Frame.from_records(records,
                columns=('a', 'b'),
                index=('x', 'y', 'z')
                )

        f2 = sf.Frame([[5, 4], [0, 10]], index=list('yz'), columns=list('ab'))

        self.assertEqual(f1.clip(lower=3, upper=f2).to_pairs(0),
            (('a', (('x', 3.0), ('y', 5.0), ('z', 3.0))), ('b', (('x', 3.0), ('y', 4.0), ('z', 10.0))))
            )


    def test_frame_clip_e(self) -> None:

        records = (
                (2, 2),
                (30, 34),
                (22, 95),
                )
        f1 = Frame.from_records(records,
                columns=('a', 'b'),
                index=('x', 'y', 'z')
                )
        f2 = f1.clip(20, 31)
        self.assertEqual(f2.to_pairs(0),
                (('a', (('x', 20), ('y', 30), ('z', 22))), ('b', (('x', 20), ('y', 31), ('z', 31)))))


    def test_frame_clip_f(self) -> None:

        records = (
                (2, 2),
                (30, 34),
                (22, 95),
                )
        f1 = Frame.from_records(records,
                columns=('a', 'b'),
                index=('x', 'y', 'z')
                )
        f2 = f1.clip()

        self.assertEqual(f2.to_pairs(0),
                (('a', (('x', 2), ('y', 30), ('z', 22))), ('b', (('x', 2), ('y', 34), ('z', 95))))
                )


    #---------------------------------------------------------------------------

    def test_frame_from_dict_a(self) -> None:

        with self.assertRaises(RuntimeError):
            # mismatched length
            sf.Frame.from_dict(dict(a=(1,2,3,4,5), b=tuple('abcdef')))


    def test_frame_from_sql_a(self) -> None:

        conn: sqlite3.Connection = self.get_test_db_a()

        f1 = sf.Frame.from_sql('select * from events', connection=conn)

        # this might be different on windows
        self.assertEqual([x.kind for x in f1.dtypes.values],
                ['U', 'U', 'f', 'i'])

        self.assertEqual(f1.to_pairs(0),
                (('date', ((0, '2006-01-01'), (1, '2006-01-02'), (2, '2006-01-01'), (3, '2006-01-02'))), ('identifier', ((0, 'a1'), (1, 'a1'), (2, 'b2'), (3, 'b2'))), ('value', ((0, 12.5), (1, 12.5), (2, 12.5), (3, 12.5))), ('count', ((0, 8), (1, 8), (2, 8), (3, 8))))
                )


    def test_frame_from_records_items_a(self) -> None:

        def gen() -> tp.Iterator[tp.Tuple[tp.Hashable, tp.Dict[tp.Hashable, tp.Any]]]:
            for i in range(3):
                yield f'000{i}', {'squared': i**2, 'cubed': i**3}

        f = Frame.from_dict_records_items(gen())

        self.assertEqual(
                f.to_pairs(0),
                (('squared', (('0000', 0), ('0001', 1), ('0002', 4))), ('cubed', (('0000', 0), ('0001', 1), ('0002', 8))))
        )



    def test_frame_loc_min_a(self) -> None:

        records = (
                (2, 2),
                (30, 34),
                (2, -95),
                )
        f1 = Frame.from_records(records,
                columns=('a', 'b'),
                index=('x', 'y', 'z')
                )

        self.assertEqual(f1.loc_min().to_pairs(),
                (('a', 'x'), ('b', 'z')))

        self.assertEqual(f1.loc_min(axis=1).to_pairs(),
                (('x', 'a'), ('y', 'a'), ('z', 'b')))


    def test_frame_loc_min_b(self) -> None:

        records = (
                (2, 2),
                (np.nan, 34),
                (2, -95),
                )
        f1 = Frame.from_records(records,
                columns=('a', 'b'),
                index=('x', 'y', 'z')
                )

        with self.assertRaises(RuntimeError):
            f1.loc_min(skipna=False)

    def test_frame_iloc_min_a(self) -> None:

        records = (
                (2, 2),
                (30, 34),
                (2, -95),
                )
        f1 = Frame.from_records(records,
                columns=('a', 'b'),
                index=('x', 'y', 'z')
                )

        self.assertEqual(f1.iloc_min().to_pairs(),
                (('a', 0), ('b', 2)))

        self.assertEqual(f1.iloc_min(axis=1).to_pairs(),
                (('x', 0), ('y', 0), ('z', 1)))


    def test_frame_iloc_min_b(self) -> None:

        records = (
                (2, 2),
                (30, np.nan),
                (2, -95),
                )
        f1 = Frame.from_records(records,
                columns=('a', 'b'),
                index=('x', 'y', 'z')
                )

        self.assertAlmostEqualItems(
                f1.iloc_min(skipna=False).to_pairs(),
                (('a', 0), ('b', np.nan)))

        self.assertAlmostEqualItems(
                f1.iloc_min(axis=1, skipna=False).to_pairs(),
                (('x', 0), ('y', np.nan), ('z', 1)))


    def test_frame_loc_max_a(self) -> None:

        records = (
                (2000, 2),
                (30, 34),
                (2, -95),
                )
        f1 = Frame.from_records(records,
                columns=('a', 'b'),
                index=('x', 'y', 'z')
                )

        self.assertEqual(f1.loc_max().to_pairs(),
                (('a', 'x'), ('b', 'y')))

        self.assertEqual(f1.loc_max(axis=1).to_pairs(),
                (('x', 'a'), ('y', 'b'), ('z', 'a')))

    def test_frame_loc_max_b(self) -> None:

        records = (
                (2, 2),
                (np.nan, 34),
                (2, -95),
                )
        f1 = Frame.from_records(records,
                columns=('a', 'b'),
                index=('x', 'y', 'z')
                )

        with self.assertRaises(RuntimeError):
            f1.loc_max(skipna=False)

    def test_frame_iloc_max_a(self) -> None:

        records = (
                (2000, 2),
                (30, 34),
                (2, -95),
                )
        f1 = Frame.from_records(records,
                columns=('a', 'b'),
                index=('x', 'y', 'z')
                )

        self.assertEqual(f1.iloc_max().to_pairs(),
                (('a', 0), ('b', 1)))

        self.assertEqual(f1.iloc_max(axis=1).to_pairs(),
                (('x', 0), ('y', 1), ('z', 0)))


    def test_frame_bloc_a(self) -> None:

        f1= Frame.from_dict(
                dict(a=(3,2,1), b=(4,5,6)),
                index=('x', 'y', 'z'),
                name='f2')
        f2 = Frame.from_dict(
                dict(x=(1,2,-5,200), y=(3,4,-5,-3000)),
                index=IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
                name='f1')
        f3 = Frame.from_records(
                ((10, 20, 50, 60), (50.0, 60.4, -50, -60)),
                index=('p', 'q'),
                columns=IndexHierarchy.from_product(('I', 'II'), ('a', 'b')),
                name='f3')

        s1 = f1.bloc((f1 <= 2) | (f1 > 4))
        self.assertEqual(s1.to_pairs(),
                ((('y', 'a'), 2), (('y', 'b'), 5), (('z', 'a'), 1), (('z', 'b'), 6))
                )

        s2 = f2.bloc((f2 < 0))
        self.assertEqual(s2.to_pairs(),
                (((('II', 'a'), 'x'), -5), ((('II', 'a'), 'y'), -5), ((('II', 'b'), 'y'), -3000))
                )

        s3 = f3.bloc(f3 < 11)
        self.assertEqual(s3.to_pairs(),
                ((('p', ('I', 'a')), 10), (('q', ('II', 'a')), -50), (('q', ('II', 'b')), -60))
                )


    def test_frame_bloc_b(self) -> None:

        f = sf.Frame([[True, False], [False, True]], index=('a', 'b'), columns=['d', 'c'])
        self.assertEqual(
                f.assign.bloc(f)('T').assign.bloc(~f)('').to_pairs(0),
                (('d', (('a', 'T'), ('b', ''))), ('c', (('a', ''), ('b', 'T'))))
                )

    #---------------------------------------------------------------------------
    def test_frame_unset_index_a(self) -> None:
        records = (
                (2, 2),
                (30, 3),
                (2, -95),
                )
        f1 = Frame.from_records(records,
                columns=('a', 'b'),
                index=('x', 'y', 'z')
                )
        self.assertEqual(f1.unset_index().to_pairs(0),
                (('__index0__', ((0, 'x'), (1, 'y'), (2, 'z'))), ('a', ((0, 2), (1, 30), (2, 2))), ('b', ((0, 2.0), (1, 3), (2, -95.0))))
                )


    def test_frame_unset_index_b(self) -> None:
        records = (
                (1, 2, 'a', False),
                (30, 34, 'b', True),
                (54, 95, 'c', False),
                (65, 73, 'd', True),
                )
        columns = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        index = IndexHierarchy.from_product((100, 200), (True, False))
        f1 = Frame.from_records(records,
                columns=columns,
                index=index)

        with self.assertRaises(ErrorInitFrame):
            f1.unset_index()


    def test_frame_unset_index_c(self) -> None:
        records = (
                (1, 2, 'a', False),
                (30, 34, 'b', True),
                (54, 95, 'c', False),
                (65, 73, 'd', True),
                )
        index = IndexHierarchy.from_product((100, 200), (True, False), name=('a', 'b'))
        f1 = Frame.from_records(records,
                index=index)
        self.assertEqual(f1.unset_index().to_pairs(0),
                (('a', ((0, 100), (1, 100), (2, 200), (3, 200))), ('b', ((0, True), (1, False), (2, True), (3, False))), (0, ((0, 1), (1, 30), (2, 54), (3, 65))), (1, ((0, 2), (1, 34), (2, 95), (3, 73))), (2, ((0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'))), (3, ((0, False), (1, True), (2, False), (3, True))))
                )


    def test_frame_unset_index_d(self) -> None:
        records = (
                (2, 2),
                (30, 3),
                (2, -95),
                )
        f1 = Frame.from_records(records,
                columns=('a', 'b'),
                index=('x', 'y', 'z')
                )
        f2 = f1.unset_index(names=('index',))

        self.assertEqual(f2.to_pairs(0),
                (('index', ((0, 'x'), (1, 'y'), (2, 'z'))), ('a', ((0, 2), (1, 30), (2, 2))), ('b', ((0, 2), (1, 3), (2, -95))))
                )


    def test_frame_unset_index_e(self) -> None:
        # using ILoc after unset led to an error because of no handling when loc is iloc
        f1 = sf.Frame.from_records(['a', 'b'], index=tuple(('c', 'd')))
        self.assertEqual(f1.loc['d', 0], 'b')
        self.assertEqual(f1.loc[sf.ILoc[0], 0], 'a')

        f2 = f1.unset_index()
        self.assertEqual(
                f2.to_pairs(0),
                (('__index0__', ((0, 'c'), (1, 'd'))), (0, ((0, 'a'), (1, 'b'))))
        )

        self.assertEqual(f2.loc[0, 0], 'a')
        self.assertEqual(f2.loc[sf.ILoc[0], 0], 'a')




    #---------------------------------------------------------------------------
    def test_frame_pivot_a(self) -> None:

        index = IndexHierarchy.from_product(
                ('far', 'near'), ('up', 'down'), ('left', 'right'),
                name=('z', 'y', 'x')
                )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()
        post = f2.pivot(
                index_fields=('x', 'y'), # values in this field become the index
                columns_fields=('z',), # values in this field become columns
                data_fields=('a', 'b')
                )

        self.assertEqual(post.to_pairs(0),
                ((('far', 'a'), ((('left', 'down'), 2), (('left', 'up'), 0), (('right', 'down'), 3), (('right', 'up'), 1))), (('far', 'b'), ((('left', 'down'), 21), (('left', 'up'), 19), (('right', 'down'), 22), (('right', 'up'), 20))), (('near', 'a'), ((('left', 'down'), 6), (('left', 'up'), 4), (('right', 'down'), 7), (('right', 'up'), 5))), (('near', 'b'), ((('left', 'down'), 22), (('left', 'up'), 20), (('right', 'down'), 23), (('right', 'up'), 21))))
                )


    def test_frame_pivot_b(self) -> None:

        index = IndexHierarchy.from_product(
                ('far', 'near'), ('up', 'down'), ('left', 'right'),
                name=('z', 'y', 'x')
                )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()
        post = f2.pivot(
                index_fields=('x', 'y'), # values in this field become the index
                columns_fields=('z',), # values in this field become columns
                data_fields=('a',)
                )

        self.assertEqual(post.to_pairs(0),
                (('far', ((('left', 'down'), 2), (('left', 'up'), 0), (('right', 'down'), 3), (('right', 'up'), 1))), ('near', ((('left', 'down'), 6), (('left', 'up'), 4), (('right', 'down'), 7), (('right', 'up'), 5))))
                )


    def test_frame_pivot_c(self) -> None:

        index = IndexHierarchy.from_product(
                ('far', 'near'), ('up', 'down'), ('left', 'right'),
                name=('z', 'y', 'x')
                )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()
        post = f2.pivot(
                index_fields=('x', 'y'), # values in this field become the index
                data_fields=('b',)
                )

        self.assertEqual(post.to_pairs(0),
                (('b', ((('left', 'down'), 43), (('left', 'up'), 39), (('right', 'down'), 45), (('right', 'up'), 41))),)
                )


    def test_frame_pivot_d(self) -> None:

        index = IndexHierarchy.from_product(
                ('far', 'near'), ('up', 'down'), ('left', 'right'),
                name=('z', 'y', 'x')
                )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()

        # single argument specs
        self.assertEqual(f2.pivot('z', 'x', 'a').to_pairs(0),
                (('left', (('far', 2), ('near', 10))), ('right', (('far', 4), ('near', 12))))
                )

        self.assertEqual(f2.pivot('z', 'x', 'b').to_pairs(0),
                (('left', (('far', 40), ('near', 42))), ('right', (('far', 42), ('near', 44))))
                )

        self.assertEqual(f2.pivot('x', 'y', 'a').to_pairs(0),
                (('down', (('left', 8), ('right', 10))), ('up', (('left', 4), ('right', 6))))
                )

        self.assertEqual(f2.pivot('x', 'y', 'b').to_pairs(0),
                (('down', (('left', 43), ('right', 45))), ('up', (('left', 39), ('right', 41))))
                )

    def test_frame_pivot_e(self) -> None:

        index = IndexHierarchy.from_product(
                ('far', 'near'), ('up', 'down'), ('left', 'right'),
                name=('z', 'y', 'x')
                )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()

        # no columns provided

        self.assertEqual(
                f2.pivot('z', data_fields='b').to_pairs(0),
                (('b', (('far', 82), ('near', 86))),)
                )

        self.assertEqual(
                f2.pivot('z', data_fields=('a', 'b')).to_pairs(0),
                (('a', (('far', 6), ('near', 22))), ('b', (('far', 82), ('near', 86))))
                )


    def test_frame_pivot_f(self) -> None:

        index = IndexHierarchy.from_product(
                ('far', 'near'), ('up', 'down'), ('left', 'right'),
                name=('z', 'y', 'x')
                )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()

        # shows unique values of 'b' as columns, then shows values for z, a
        post = f2.pivot(('x', 'y'), ('b',), fill_value='')

        self.assertEqual(post.to_pairs(0),
                (((19, 'z'), ((('left', 'down'), ''), (('left', 'up'), 'far'), (('right', 'down'), ''), (('right', 'up'), ''))), ((19, 'a'), ((('left', 'down'), ''), (('left', 'up'), 0), (('right', 'down'), ''), (('right', 'up'), ''))), ((20, 'z'), ((('left', 'down'), ''), (('left', 'up'), 'near'), (('right', 'down'), ''), (('right', 'up'), 'far'))), ((20, 'a'), ((('left', 'down'), ''), (('left', 'up'), 4), (('right', 'down'), ''), (('right', 'up'), 1))), ((21, 'z'), ((('left', 'down'), 'far'), (('left', 'up'), ''), (('right', 'down'), ''), (('right', 'up'), 'near'))), ((21, 'a'), ((('left', 'down'), 2), (('left', 'up'), ''), (('right', 'down'), ''), (('right', 'up'), 5))), ((22, 'z'), ((('left', 'down'), 'near'), (('left', 'up'), ''), (('right', 'down'), 'far'), (('right', 'up'), ''))), ((22, 'a'), ((('left', 'down'), 6), (('left', 'up'), ''), (('right', 'down'), 3), (('right', 'up'), ''))), ((23, 'z'), ((('left', 'down'), ''), (('left', 'up'), ''), (('right', 'down'), 'near'), (('right', 'up'), ''))), ((23, 'a'), ((('left', 'down'), ''), (('left', 'up'), ''), (('right', 'down'), 7), (('right', 'up'), ''))))
                )

    def test_frame_pivot_g(self) -> None:

        index = IndexHierarchy.from_product(
                ('far', 'near'), ('up', 'down'), ('left', 'right'),
                name=('z', 'y', 'x')
                )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()

        # multiple columns; all reamining fields go to data_fields
        post1 = f2.pivot('z', ('y', 'x'))

        self.assertEqual(post1.to_pairs(0),
                ((('down', 'left', 'a'), (('far', 2), ('near', 6))), (('down', 'left', 'b'), (('far', 21), ('near', 22))), (('down', 'right', 'a'), (('far', 3), ('near', 7))), (('down', 'right', 'b'), (('far', 22), ('near', 23))), (('up', 'left', 'a'), (('far', 0), ('near', 4))), (('up', 'left', 'b'), (('far', 19), ('near', 20))), (('up', 'right', 'a'), (('far', 1), ('near', 5))), (('up', 'right', 'b'), (('far', 20), ('near', 21))))
                )


    def test_frame_pivot_h(self) -> None:

        index = IndexHierarchy.from_product(
                ('far', 'near'), ('up', 'down'), ('left', 'right'),
                name=('z', 'y', 'x')
                )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()


        # specifying a data_fields value
        post1 = f2.pivot('z', ('y', 'x'), 'a')

        self.assertEqual(post1.to_pairs(0),
                ((('down', 'left'), (('far', 2), ('near', 6))), (('down', 'right'), (('far', 3), ('near', 7))), (('up', 'left'), (('far', 0), ('near', 4))), (('up', 'right'), (('far', 1), ('near', 5)))))



    def test_frame_pivot_i(self) -> None:

        index = IndexHierarchy.from_product(
                ('far', 'near'), ('up', 'down'), ('left', 'right'),
                name=('z', 'y', 'x')
                )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()

        post1 = f2.pivot('z', 'y', 'a', func={'min': np.min, 'max': np.max})

        self.assertEqual(post1.to_pairs(0),
                ((('down', 'min'), (('far', 2), ('near', 6))), (('down', 'max'), (('far', 3), ('near', 7))), (('up', 'min'), (('far', 0), ('near', 4))), (('up', 'max'), (('far', 1), ('near', 5))))
                )


    def test_frame_pivot_j(self) -> None:

        index = IndexHierarchy.from_product(
                ('far', 'near'), ('up', 'down'), ('left', 'right'),
                name=('z', 'y', 'x')
                )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()

        post1 = f2.pivot('z', ('y', 'x'), 'b', func={'min': np.min, 'max': np.max})
        self.assertEqual(
                post1.to_pairs(0),
                ((('down', 'left', 'min'), (('far', 21), ('near', 22))), (('down', 'left', 'max'), (('far', 21), ('near', 22))), (('down', 'right', 'min'), (('far', 22), ('near', 23))), (('down', 'right', 'max'), (('far', 22), ('near', 23))), (('up', 'left', 'min'), (('far', 19), ('near', 20))), (('up', 'left', 'max'), (('far', 19), ('near', 20))), (('up', 'right', 'min'), (('far', 20), ('near', 21))), (('up', 'right', 'max'), (('far', 20), ('near', 21))))
                )

        # default populates data values for a, b
        post2 = f2.pivot('z', ('y', 'x'), func={'min': np.min, 'max': np.max})
        self.assertEqual(
                post2.to_pairs(0),
                ((('down', 'left', 'a', 'min'), (('far', 2), ('near', 6))), (('down', 'left', 'a', 'max'), (('far', 2), ('near', 6))), (('down', 'left', 'b', 'min'), (('far', 21), ('near', 22))), (('down', 'left', 'b', 'max'), (('far', 21), ('near', 22))), (('down', 'right', 'a', 'min'), (('far', 3), ('near', 7))), (('down', 'right', 'a', 'max'), (('far', 3), ('near', 7))), (('down', 'right', 'b', 'min'), (('far', 22), ('near', 23))), (('down', 'right', 'b', 'max'), (('far', 22), ('near', 23))), (('up', 'left', 'a', 'min'), (('far', 0), ('near', 4))), (('up', 'left', 'a', 'max'), (('far', 0), ('near', 4))), (('up', 'left', 'b', 'min'), (('far', 19), ('near', 20))), (('up', 'left', 'b', 'max'), (('far', 19), ('near', 20))), (('up', 'right', 'a', 'min'), (('far', 1), ('near', 5))), (('up', 'right', 'a', 'max'), (('far', 1), ('near', 5))), (('up', 'right', 'b', 'min'), (('far', 20), ('near', 21))), (('up', 'right', 'b', 'max'), (('far', 20), ('near', 21))))
                )


    def test_frame_pivot_k(self) -> None:

        index = IndexHierarchy.from_product(
                ('far', 20), (None, 'down'), (False, 'right'),
                name=('z', 'y', 'x')
                )
        f1 = FrameGO(index=index)
        f1['a'] = range(len(f1))
        f1['b'] = (len(str(f1.index.values[i])) for i in range(len(f1)))

        f2 = f1.unset_index()
        post1 = f2.pivot('z', 'y', 'a')

        self.assertEqual(post1.to_pairs(0),
                ((None, (('far', 1), (20, 9))), ('down', (('far', 5), (20, 13))))
                )


    #---------------------------------------------------------------------------

    def test_frame_axis_window_items_a(self) -> None:

        base = np.array([1, 2, 3, 4])
        records = (base * n for n in range(1, 21))

        f1 = Frame.from_records(records,
                columns=list('ABCD'),
                index=self.get_letters(20))

        post0 = tuple(f1._axis_window_items(axis=0))
        self.assertEqual(len(post0), 19)
        self.assertEqual(post0[0][0], 'b')
        self.assertEqual(post0[0][1].__class__, Frame)
        self.assertEqual(post0[0][1].shape, (2, 4))

        self.assertEqual(post0[-1][0], 't')
        self.assertEqual(post0[-1][1].__class__, Frame)
        self.assertEqual(post0[-1][1].shape, (2, 4))

        post1 = tuple(f1._axis_window_items(axis=1))
        self.assertEqual(len(post1), 3)

        self.assertEqual(post1[0][0], 'B')
        self.assertEqual(post1[0][1].__class__, Frame)
        self.assertEqual(post1[0][1].shape, (20, 2))

        self.assertEqual(post1[-1][0], 'D')
        self.assertEqual(post1[-1][1].__class__, Frame)
        self.assertEqual(post1[-1][1].shape, (20, 2))



    def test_frame_axis_window_items_b(self) -> None:

        base = np.array([1, 2, 3, 4])
        records = (base * n for n in range(1, 21))

        f1 = Frame.from_records(records,
                columns=list('ABCD'),
                index=self.get_letters(20))

        post0 = tuple(f1._axis_window_items(axis=0, window_array=True))
        self.assertEqual(len(post0), 19)
        self.assertEqual(post0[0][0], 'b')
        self.assertEqual(post0[0][1].__class__, np.ndarray)
        self.assertEqual(post0[0][1].shape, (2, 4))

        self.assertEqual(post0[-1][0], 't')
        self.assertEqual(post0[-1][1].__class__, np.ndarray)
        self.assertEqual(post0[-1][1].shape, (2, 4))

        post1 = tuple(f1._axis_window_items(axis=1, window_array=True))
        self.assertEqual(len(post1), 3)

        self.assertEqual(post1[0][0], 'B')
        self.assertEqual(post1[0][1].__class__, np.ndarray)
        self.assertEqual(post1[0][1].shape, (20, 2))

        self.assertEqual(post1[-1][0], 'D')
        self.assertEqual(post1[-1][1].__class__, np.ndarray)
        self.assertEqual(post1[-1][1].shape, (20, 2))



    def test_frame_iter_window_a(self) -> None:

        base = np.array([1, 2, 3, 4])
        records = (base * n for n in range(1, 21))

        f1 = Frame.from_records(records,
                columns=list('ABCD'),
                index=self.get_letters(20))

        self.assertEqual(
                f1.iter_window(size=3).apply(lambda f: f['B'].sum()).to_pairs(),
                (('c', 12), ('d', 18), ('e', 24), ('f', 30), ('g', 36), ('h', 42), ('i', 48), ('j', 54), ('k', 60), ('l', 66), ('m', 72), ('n', 78), ('o', 84), ('p', 90), ('q', 96), ('r', 102), ('s', 108), ('t', 114))
        )



    def test_frame_bool_a(self) -> None:
        records = (
                (2, 2),
                (30, 3),
                (2, -95),
                )
        f1 = Frame.from_records(records,
                columns=('a', 'b'),
                index=('x', 'y', 'z')
                )
        self.assertTrue(bool(f1))
        self.assertTrue(bool(f1.T))



    def test_frame_bool_b(self) -> None:
        f1 = Frame(columns=('a', 'b'))

        self.assertFalse(bool(f1))
        self.assertFalse(bool(f1.T))



if __name__ == '__main__':
    unittest.main()
